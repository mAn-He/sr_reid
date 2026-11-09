"""Stage 2 CLIP-ReID 학습기 (공식 레시피 전체, vit_clipreid.yml). env: srreid-clipreid (torch 1.13).

공식 코드(third_party/CLIP-ReID)는 수정하지 않는다. 공식 train_clipreid.py의 순서(set_seed → 로더 → make_model → make_loss
→ optimizer_1stage·scheduler → do_train_stage1 → optimizer_2stage·WarmupMultiStepLR → do_train_stage2)를 따르고,
공식 do_train_stage1 / do_train_stage2 함수를 그대로 호출한다. 바꾸는 것은 런타임 대체뿐이다.
  - 데이터: 공식 Market1501 클래스(*.jpg, test 폴더 필수) 대신 memmap 캐시(stage05/cache.py). 행 순서·재번호·camid는 공식
    _process_dir와 같게 만든다(sorted 이름, enumerate(set(pid)), camid-1, trackid 0). stage 2 로더는 ClipSet + worker_norm
    (공식 변환과 같은 난수 순서·값, STAGE05 D.5), stage 1 로더는 val 변환(Resize no-op, ToTensor, Normalize)과 같은 값. pin_memory=True.
  - config 덮어쓰기: INPUT.SIZE_TRAIN/TEST = [128k, 64k](입력 크기 정책), SOLVER.SEED = 42, OUTPUT_DIR = --out,
    STAGE2.EVAL_PERIOD·STAGE1/2.CHECKPOINT_PERIOD = 10**9(평가·공식 저장 끔. 저장은 이 스크립트가 한다), NUM_WORKERS(--workers).
  - 평가 끔: 공식 주기 평가는 에폭 60(마지막) 학습이 끝난 뒤 no_grad로만 돈다. 끄면 val_loader(=None)를 만들지도 순회하지도 않는다.
  - 처리 모듈의 `amp`를 GradScaler 생성만 가로채는 대리 객체로 바꾼다(재개 때 scaler 상태 주입, 저장 때 참조). autocast는 원본 그대로.
  - 스케줄러를 대리 객체로 감싸 에폭 시작(scheduler.step 호출) 때 이전 에폭 기록·재개용 저장을 한다. lr 계산은 원본 그대로.
  - 손실 함수를 감싸 에폭 평균 손실·정확도를 GPU에서 누적한다(detach, 난수 없음).
재개: <out>/resume.pth(모델, 옵티마이저, center 옵티마이저, 스케줄러, scaler, 파이썬·numpy·torch·CUDA 난수 상태, 경과 기록)를
  stage 1은 S1 에폭마다(기본 20)와 stage 1 끝, stage 2는 S2 에폭마다(기본 5) 원자적으로 쓴다. 재개하면 남은 에폭 수로 cfg를
  복제해 공식 함수를 다시 부르고, 스케줄러는 offset(stage 1: step(epoch+offset), stage 2: last_epoch 복원)으로 이어간다.
  비트 일치는 보장하지 않는다(cudnn.benchmark 알고리즘 선택, stage 1 특징·stage 2 텍스트 특징 재계산). 재개 사실은 train_meta.json에 남는다.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CR = ROOT / "third_party" / "CLIP-ReID"
CLIP_WEIGHT = ROOT / "weights" / "stage0" / "ViT-B-16.pt"
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, str(CR))
from stage05 import cache as C  # noqa: E402
from stage05.backends import ClipSet  # noqa: E402

NEVER = 10 ** 9
N_TRAIN, N_PID = 12936, 751
SPLIT = "bounding_box_train"
CONDS = ("bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman")
PATTERN = re.compile(r"([-\d]+)_c(\d)")
LOG = logging.getLogger("transreid")


def now():
    return time.strftime("%F %T")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------- config
def build_cfg(k, out, workers=None, epochs=None):
    """공식 yml + 결정된 덮어쓰기. DATASETS 키는 값이 전부 주석이라 yacs가 거부하므로 지우고 병합한다(Stage 0과 같음)."""
    from config import cfg as base
    cfg = base.clone()
    cfg.defrost()
    with open(CR / "configs" / "person" / "vit_clipreid.yml") as f:
        recipe = type(cfg).load_cfg(f)
    del recipe["DATASETS"]
    cfg.merge_from_other_cfg(recipe)
    opts = ["INPUT.SIZE_TRAIN", [128 * k, 64 * k], "INPUT.SIZE_TEST", [128 * k, 64 * k], "DATASETS.NAMES", "market1501",
            "SOLVER.SEED", 42, "OUTPUT_DIR", str(out),
            "SOLVER.STAGE2.EVAL_PERIOD", NEVER, "SOLVER.STAGE1.CHECKPOINT_PERIOD", NEVER, "SOLVER.STAGE2.CHECKPOINT_PERIOD", NEVER]
    if workers is not None:
        opts += ["DATALOADER.NUM_WORKERS", int(workers)]
    if epochs is not None:  # 숨은 디버그 옵션 전용
        opts += ["SOLVER.STAGE1.MAX_EPOCHS", epochs[0], "SOLVER.STAGE2.MAX_EPOCHS", epochs[1]]
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg, opts


def with_epochs(cfg, key, n):
    c = cfg.clone()
    c.defrost()
    getattr(c.SOLVER, key).MAX_EPOCHS = n
    c.freeze()
    return c


# ---------------------------------------------------------------- data
def market_rows(names):
    """공식 Market1501._process_dir(relabel=True)와 같은 규칙. 입력 names는 memmap 행 순서."""
    ref = sorted(p.name[:-4] for p in (ROOT / "data" / "market" / SPLIT).iterdir() if p.suffix == ".jpg")
    if list(names) != ref:  # 공식은 sorted(glob(*.jpg)) 순서. 캐시 행 순서가 같아야 sampler 인덱스가 공식과 같다
        raise RuntimeError(f"캐시 이름 순서가 Market train 정렬 순서와 다름 (n={len(names)}, ref={len(ref)})")
    pid_container = set()
    for n in names:
        pid, _ = map(int, PATTERN.search(n).groups())
        if pid == -1:
            raise RuntimeError("train에 pid -1이 있으면 memmap 행과 목록 인덱스가 어긋난다")
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}
    rows = []
    for n in names:
        pid, camid = map(int, PATTERN.search(n).groups())
        rows.append((n, pid2label[pid], camid - 1, 0))
    return rows


def prepare_cache(cond, k):
    base = C.cache_path(cond, k, SPLIT)
    if not base.with_name(base.name + ".u8").exists():
        src = C.source_dir(cond, k, SPLIT)
        n = sum(1 for p in src.iterdir() if p.suffix in (".webp", ".png", ".jpg"))
        if n != N_TRAIN:  # 생성 중인 SR 폴더로 불완전한 캐시를 만들지 않는다
            raise RuntimeError(f"{src}: {n}장, {N_TRAIN}장이어야 함 (생성 미완료?)")
    info = C.build(cond, k, SPLIT)
    meta = json.loads(base.with_name(base.name + ".json").read_text())
    h, w = 128 * k, 64 * k
    if tuple(meta["shape"]) != (N_TRAIN, h, w, 3):
        raise RuntimeError(f"캐시 shape {meta['shape']} != {(N_TRAIN, h, w, 3)} (Resize가 no-op이어야 함)")
    rows = market_rows(meta["names"])
    return rows, {"cache": info["path"], "cache_build_s": round(info["seconds"], 1), "cache_reused": info["reused"],
                  "source": meta["source"]}


class S2Set(ClipSet):
    """ClipSet(worker_norm)의 출력을 공식 ImageDataset 형식 (img, pid, camid, trackid, 이름)으로 바꾼다."""

    def __getitem__(self, i):
        t, pid, e = super().__getitem__(i)
        assert e is None
        r = self.rows[i]
        return t, pid, r[2], r[3], r[0]


class ValSet(Dataset):
    """공식 val_transforms(Resize(SIZE_TEST) → ToTensor → Normalize)와 같은 값. 캐시가 SIZE_TEST 크기라 Resize는 no-op."""

    def __init__(self, cond, k, split, rows, mean, std):
        self.cond, self.k, self.split, self.rows = cond, k, split, rows
        self.mean, self.std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
        self.mm = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = C.load(self.cond, self.k, self.split)[0]
        t = torch.from_numpy(np.ascontiguousarray(self.mm[i].transpose(2, 0, 1))).float().div(255).sub_(self.mean).div_(self.std)
        r = self.rows[i]
        return t, r[1], r[2], r[3], r[0]


class TimedLoader:
    """DataLoader 대리. 배치 요청 시각(=직전 반복의 synchronize 이후)을 기록하고, 디버그·시험에서는 반복 수를 자른다."""

    def __init__(self, loader, max_iters=None, keep_times=False):
        self.loader, self.max_iters, self.keep_times = loader, max_iters, keep_times
        self.batch_size = loader.batch_size
        self.epochs = []

    def __len__(self):
        n = len(self.loader)
        return min(n, self.max_iters) if self.max_iters else n

    def __iter__(self):
        rec = {"t0": time.perf_counter(), "n_iter": 0, "n_img": 0, "times": []}
        self.epochs.append(rec)
        it = iter(self.loader)
        while self.max_iters is None or rec["n_iter"] < self.max_iters:
            t = time.perf_counter()
            if self.keep_times:
                rec["times"].append(t)
            try:
                b = next(it)
            except StopIteration:
                break
            rec["n_iter"] += 1
            rec["n_img"] += b[0].shape[0]
            yield b
        rec["t_end"] = time.perf_counter()
        del it


# ---------------------------------------------------------------- hooks
class AmpShim:
    """처리 모듈의 `from torch.cuda import amp`를 대신한다. autocast는 원본, GradScaler는 기본 인자로 만들고 참조·상태 주입만 한다."""

    def __init__(self, tr):
        self.autocast = torch.cuda.amp.autocast
        self.tr = tr

    def GradScaler(self, *a, **kw):
        s = torch.cuda.amp.GradScaler(*a, **kw)
        if self.tr.pending_scaler is not None:
            s.load_state_dict(self.tr.pending_scaler)
            self.tr.pending_scaler = None
        self.tr.scaler = s
        return s


class SchedProxy:
    """공식 스케줄러 대리. step() 직전에 에폭 시작 콜백을 부르고, stage 1(step(epoch))은 epoch+offset으로 넘긴다."""

    def __init__(self, sched, offset, on_start):
        self.__dict__.update(_s=sched, _off=offset, _cb=on_start, _n=0)

    def step(self, epoch=None):
        self._n += 1
        self._cb(self._n)
        return self._s.step() if epoch is None else self._s.step(epoch + self._off)

    def _get_lr(self, epoch):
        return self._s._get_lr(epoch + self._off)

    def get_lr(self):
        return self._s.get_lr()

    def __getattr__(self, n):
        return getattr(self._s, n)


class Acc:
    """GPU 누적기(동기화는 에폭 끝에서만)."""

    def __init__(self, keep=False):
        self.keep = keep
        self.reset()

    def reset(self):
        self.loss = torch.zeros((), device="cuda", dtype=torch.float64)
        self.acc = torch.zeros((), device="cuda", dtype=torch.float64)
        self.n = 0
        self.hist = []

    def add(self, loss, acc=None):
        l = loss.detach().double()
        self.loss += l
        if acc is not None:
            self.acc += acc.detach().double()
        self.n += 1
        if self.keep:
            self.hist.append(l)


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "np": np.random.get_state(), "py": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["np"])
    random.setstate(s["py"])


def gpu_mem():
    return {"peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}


def env_meta(cfg, a):
    import torch.backends.cudnn as cudnn
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn_version": cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host": socket.gethostname(), "SRREID_CONC_NOTE": os.environ.get("SRREID_CONC_NOTE"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "cudnn": {"enabled": cudnn.enabled, "benchmark": cudnn.benchmark, "deterministic": cudnn.deterministic,
                      "allow_tf32": cudnn.allow_tf32},
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "workers": cfg.DATALOADER.NUM_WORKERS, "pin_memory": True, "ckpt": a.ckpt}


# ---------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, a, trial=False):
        self.a, self.out = a, Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        if torch.cuda.device_count() != 1:  # 공식 코드는 GPU가 2장 이상 보이면 DataParallel로 감싼다. 단일 GPU 레시피만 허용
            raise SystemExit(f"GPU가 정확히 1장 보여야 함 (현재 {torch.cuda.device_count()}). CUDA_VISIBLE_DEVICES를 지정")
        self.cfg, self.opts = build_cfg(a.k, self.out, a.workers, a.debug_epochs)
        cfg = self.cfg
        fh = logging.FileHandler(self.out / ("trial_official_log.txt" if trial else "official_log.txt"), mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        LOG.setLevel(logging.DEBUG)
        LOG.addHandler(fh)
        LOG.propagate = False  # 공식 로그(모델 출력 포함)는 파일에만 쓴다

        import train_clipreid as TC  # 진입 스크립트의 set_seed를 그대로 쓴다(main은 실행되지 않음)
        import processor.processor_clipreid_stage1 as P1
        import processor.processor_clipreid_stage2 as P2
        from datasets.make_dataloader_clipreid import train_collate_fn
        from datasets.sampler import RandomIdentitySampler
        from loss.make_loss import make_loss
        from model.clip import clip
        from model.make_model_clipreid import make_model
        from solver.make_optimizer_prompt import make_optimizer_1stage
        from solver.scheduler_factory import create_scheduler
        from timm.data.random_erasing import RandomErasing
        self.P1, self.P2 = P1, P2
        clip._download = lambda url, root=None: str(CLIP_WEIGHT)  # 저장소 수정 없이 기록된 가중치(Stage 0과 같음)
        self.shim = AmpShim(self)
        P1.amp = self.shim
        P2.amp = self.shim
        self.s1_acc, self.s2_acc = Acc(), Acc(keep=trial)
        acc1 = self.s1_acc

        class SupConRec(P1.SupConLoss):  # stage 1 손실 기록용(값은 그대로 돌려준다)
            def forward(self, *x, **kw):
                loss = super().forward(*x, **kw)
                acc1.add(loss)
                return loss
        P1.SupConLoss = SupConRec

        self.rows, self.data_meta = prepare_cache(a.cond, a.k)
        t_setup = time.perf_counter()
        TC.set_seed(cfg.SOLVER.SEED)  # 공식: torch/cuda/np/random 시드 + cudnn.deterministic=True, cudnn.benchmark=True
        self.cudnn_after_seed = env_meta(cfg, a)["cudnn"]

        # 로더 (공식 make_dataloader의 생성 순서: stage 2 → val(생략) → stage 1. 생성 자체는 난수를 쓰지 않는다)
        nw = cfg.DATALOADER.NUM_WORKERS
        norm = (torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1), torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1))
        re_obj = RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu")
        s2set = S2Set(a.cond, a.k, self.rows, cfg.INPUT.PADDING, cfg.INPUT.PROB, re_obj, norm)
        self.sampler = RandomIdentitySampler(self.rows, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        dbg_iters = a.debug_iters or None
        self.loader2 = TimedLoader(DataLoader(s2set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, sampler=self.sampler,
                                              num_workers=nw, collate_fn=train_collate_fn, pin_memory=True),
                                   max_iters=a.max_iters if trial else dbg_iters, keep_times=trial)
        # 디버그: stage 1 특징을 앞 dbg*64+8장만(공식 루프는 장수가 64의 배수면 마지막 빈 배치에서 실패한다. 12936 % 64 = 8)
        s1rows = self.rows[:dbg_iters * cfg.SOLVER.STAGE1.IMS_PER_BATCH + 8] if dbg_iters else self.rows
        s1set = ValSet(a.cond, a.k, SPLIT, s1rows, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        self.loader1 = TimedLoader(DataLoader(s1set, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=nw,
                                              collate_fn=train_collate_fn, pin_memory=True))
        pids = {r[1] for r in self.rows}
        cams = {r[2] for r in self.rows}
        views = {r[3] for r in self.rows}
        assert len(pids) == N_PID and len(self.rows) == N_TRAIN
        self.num_classes, self.camera_num, self.view_num = len(pids), len(cams), len(views)

        self.model = make_model(cfg, num_class=self.num_classes, camera_num=self.camera_num, view_num=self.view_num)
        self.loss_func, self.center = make_loss(cfg, num_classes=self.num_classes)
        self.opt1 = make_optimizer_1stage(cfg, self.model)
        self.sched1 = create_scheduler(self.opt1, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min=cfg.SOLVER.STAGE1.LR_MIN,
                                       warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
                                       noise_range=None)
        if a.ckpt:  # 활성값 재계산(ClipBackend와 같음). 이미지 인코더 resblocks만. 모델에 dropout이 없어 난수 영향 없음
            from torch.utils.checkpoint import checkpoint
            for block in self.model.image_encoder.transformer.resblocks:
                block.forward = (lambda o: lambda x: checkpoint(o, x, use_reentrant=False, preserve_rng_state=True))(block.forward)
        self.setup_s = time.perf_counter() - t_setup
        self.scaler, self.pending_scaler, self.pending_rng = None, None, None

    # -------- stage 2 pieces
    def make_stage2(self):
        from solver.lr_scheduler import WarmupMultiStepLR
        from solver.make_optimizer_prompt import make_optimizer_2stage
        cfg = self.cfg
        self.opt2, self.optc = make_optimizer_2stage(cfg, self.model, self.center)
        self.sched2 = WarmupMultiStepLR(self.opt2, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                        cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    def loss2(self):
        acc2, fn = self.s2_acc, self.loss_func

        def loss_fn(score, feat, target, target_cam, i2tscore=None):
            loss = fn(score, feat, target, target_cam, i2tscore)
            acc2.add(loss, (i2tscore.max(1)[1] == target).float().mean())
            return loss
        return loss_fn


# ---------------------------------------------------------------- full run
class FullRun:
    def __init__(self, a):
        self.a, self.out = a, Path(a.out)
        self.t = Trainer(a)
        self.resume = self.out / "resume.pth"
        self.jsonl = self.out / "train_log.jsonl"
        self.state = {"wall_s": {"setup": 0.0, "stage1": 0.0, "stage2": 0.0}, "stage1": {}, "stage2": {},
                      "resume_events": [], "mem": {"stage1": {}, "stage2": {}}, "s2_img": 0, "s2_loop_s": 0.0, "s2_iters": 0,
                      "s1_iters": 0}
        self.save_every = {1: a.save_every[0], 2: a.save_every[1]}

    def log_epoch(self, rec):
        rec["time"] = now()
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{rec['time']}] stage{rec['stage']} ep{rec['epoch']} loss={rec['loss']:.4f} lr={rec['lr']:.3e} "
              f"{rec['seconds']:.1f}s" + (f" {rec['img_s']:.1f} img/s" if "img_s" in rec else ""), flush=True)

    def save(self, stage, epoch, stage1_done):
        t = self.t
        ck = {"stage": stage, "epoch": epoch, "stage1_done": stage1_done, "model": t.model.state_dict(),
              "scaler": t.scaler.state_dict() if (t.scaler is not None and not (stage == 1 and stage1_done)) else None,
              "rng": rng_state(), "state": self.state, "time": now()}
        if stage == 1 and not stage1_done:
            ck["opt1"] = t.opt1.state_dict()
        if stage == 2:
            ck.update(opt2=t.opt2.state_dict(), optc=t.optc.state_dict(), sched2=t.sched2.state_dict())
        tmp = Path(str(self.resume) + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, self.resume)
        print(f"[{now()}] resume 저장: stage{stage} epoch {epoch} stage1_done={stage1_done}", flush=True)

    def epoch_cb(self, stage, offset, n_epochs):
        t, st = self.t, self.state
        marks = {"t": None}

        def finish(g):
            dt = time.perf_counter() - marks["t"]
            st["wall_s"][f"stage{stage}"] += dt
            acc = t.s1_acc if stage == 1 else t.s2_acc
            iters = acc.n // 2 if stage == 1 else acc.n  # stage 1은 반복마다 SupCon을 두 번(i2t, t2i) 부른다
            rec = {"stage": stage, "epoch": g, "loss": (acc.loss / max(iters, 1)).item(), "iters": iters,
                   "lr": (t.opt1 if stage == 1 else t.opt2).param_groups[0]["lr"], "seconds": round(dt, 2),
                   "scale": t.scaler.get_scale() if t.scaler is not None else None}
            if stage == 1:
                st["s1_iters"] += iters
            else:
                rec["acc"] = (acc.acc / max(iters, 1)).item()
                L = t.loader2.epochs[-1]
                loop = L["t_end"] - L["t0"]
                rec.update(n_img=L["n_img"], loop_s=round(loop, 2), img_s=L["n_img"] / loop)
                st["s2_img"] += L["n_img"]
                st["s2_loop_s"] += loop
                st["s2_iters"] += L["n_iter"]
            mem = gpu_mem()
            m = st["mem"][f"stage{stage}"]
            for kk, v in mem.items():
                m[kk] = max(m.get(kk, 0), v)
            rec.update(mem)
            self.log_epoch(rec)
            acc.reset()

        def on_start(local):
            g = offset + local
            if local == 1:
                if t.pending_rng is not None:
                    set_rng_state(t.pending_rng)
                    t.pending_rng = None
                if stage == 1:
                    st["stage1"]["extract_s"] = st["stage1"].get("extract_s", 0) + time.perf_counter() - marks["t0"]
                    st["wall_s"]["stage1"] += time.perf_counter() - marks["t0"]
                else:
                    st["stage2"]["text_feat_s"] = time.perf_counter() - marks["t0"]
                    st["wall_s"]["stage2"] += time.perf_counter() - marks["t0"]
            else:
                finish(g - 1)
                if (g - 1) % self.save_every[stage] == 0:
                    self.save(stage, g - 1, stage == 2)
            marks["t"] = time.perf_counter()
        return on_start, finish, marks

    def run(self):
        a, t, cfg = self.a, self.t, self.t.cfg
        E1, E2 = cfg.SOLVER.STAGE1.MAX_EPOCHS, cfg.SOLVER.STAGE2.MAX_EPOCHS
        ck = None
        if self.resume.exists():
            ck = torch.load(self.resume, map_location="cpu")
            self.state = ck["state"]
            ev = {"time": now(), "stage": ck["stage"], "epoch": ck["epoch"], "stage1_done": ck["stage1_done"], "saved_at": ck["time"]}
            self.state["resume_events"].append(ev)
            print(f"[{now()}] 재개: {ev}", flush=True)
            t.model.load_state_dict(ck["model"])
        self.state["wall_s"]["setup"] += t.setup_s
        t.model.to(0)

        # ---- stage 1
        if ck is None or (ck["stage"] == 1 and not ck["stage1_done"]):
            off = 0
            if ck is not None:
                off = ck["epoch"]
                t.opt1.load_state_dict(ck["opt1"])
                t.pending_scaler, t.pending_rng = ck["scaler"], ck["rng"]
            torch.cuda.reset_peak_memory_stats()
            on_start, finish, marks = self.epoch_cb(1, off, E1)
            marks["t0"] = time.perf_counter()
            sched = SchedProxy(t.sched1, off, on_start)
            t.P1.do_train_stage1(with_epochs(cfg, "STAGE1", E1 - off), t.model, t.loader1, t.opt1, sched, 0)
            finish(E1)
            t.scaler = None
            self.save(1, E1, True)
            del sched
        else:
            set_rng_state(ck["rng"])  # stage 1 종료 시점 상태
        torch.cuda.empty_cache()

        # ---- stage 2
        t.make_stage2()
        off = 0
        if ck is not None and ck["stage"] == 2:
            off = ck["epoch"]
            t.opt2.load_state_dict(ck["opt2"])
            t.optc.load_state_dict(ck["optc"])
            t.sched2.load_state_dict(ck["sched2"])
            t.pending_scaler, t.pending_rng = ck["scaler"], ck["rng"]
        torch.cuda.reset_peak_memory_stats()
        on_start, finish, marks = self.epoch_cb(2, off, E2)
        marks["t0"] = time.perf_counter()
        sched = SchedProxy(t.sched2, 0, on_start)
        if off < E2:
            t.P2.do_train_stage2(with_epochs(cfg, "STAGE2", E2 - off), t.model, t.center, t.loader2, None, t.opt2, t.optc, sched,
                                 t.loss2(), 0, 0)
            finish(E2)
        self.finish_all(E1, E2)

    def finish_all(self, E1, E2):
        a, t, st, cfg = self.a, self.t, self.state, self.t.cfg
        mp = self.out / "model.pth"
        tmp = self.out / "model.pth.tmp"
        torch.save(t.model.state_dict(), tmp)  # 공식 저장 형식(model.state_dict())과 같음
        os.replace(tmp, mp)
        (self.out / "config.yaml").write_text(cfg.dump())
        meta = {"cond": a.cond, "k": a.k, "input_hw": list(cfg.INPUT.SIZE_TRAIN), "seed": cfg.SOLVER.SEED,
                "official_seed_default": 1234, "config_overrides": [str(x) for x in t.opts],
                "epochs": {"stage1": E1, "stage2": E2}, "debug": {"epochs": a.debug_epochs, "iters": a.debug_iters},
                "iters": {"stage1_total": st["s1_iters"], "stage1_per_epoch": N_TRAIN // cfg.SOLVER.STAGE1.IMS_PER_BATCH + 1,
                          "stage2_total": st["s2_iters"]},
                "batch": {"stage1": cfg.SOLVER.STAGE1.IMS_PER_BATCH, "stage2": cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                          "num_instance": cfg.DATALOADER.NUM_INSTANCE, "sampler": "RandomIdentitySampler"},
                "amp": "official: autocast(fp16) + GradScaler in both stages (always on, no config key)",
                "cudnn_after_set_seed": t.cudnn_after_seed, "env": env_meta(cfg, a), "data": t.data_meta,
                "wall_s": {kk: round(v, 1) for kk, v in st["wall_s"].items()},
                "stage1_extract_s": round(st["stage1"].get("extract_s", 0), 1),
                "stage2_text_feat_s": round(st["stage2"].get("text_feat_s", 0), 2),
                "stage2_img_s": st["s2_img"] / st["s2_loop_s"] if st["s2_loop_s"] else None,
                "mem": st["mem"], "resume_events": st["resume_events"],
                "resume_bit_exact": "not guaranteed (cudnn.benchmark; stage-1 features / stage-2 text features recomputed)",
                "eval": "disabled: SOLVER.STAGE2.EVAL_PERIOD=1e9, val_loader=None (official eval runs only after the last epoch)",
                "model": {"path": str(mp), "sha256": sha256(mp), "bytes": mp.stat().st_size}, "finished": now()}
        atomic_json(self.out / "train_meta.json", meta)
        tmp = self.out / "DONE.tmp"
        tmp.write_text(now() + "\n")
        os.replace(tmp, self.out / "DONE")
        self.resume.unlink(missing_ok=True)
        print(f"[{now()}] DONE {self.out}", flush=True)


# ---------------------------------------------------------------- trial
def run_trial(a):
    """stage 2 처리량 시험. stage 1은 건너뛰고 학습 안 된 프롬프트의 텍스트 특징을 쓴다(시험 전용).
    --trial-s1 1이면 먼저 stage 1 비용을 잰다: 전체 train 특징 추출 1회(공식 do_train_stage1:43-53과 같은 루프) +
    공식 스텝(:64-90)을 옮긴 프롬프트 스텝 몇 회. 이 부분만 공식 함수가 아니라 줄 단위로 옮긴 코드다(반복 수를 줄일 수 없어서)."""
    t = Trainer(a, trial=True)
    cfg, out = t.cfg, Path(a.out)
    t.model.to(0)
    res = {"cond": a.cond, "k": a.k, "input_hw": list(cfg.INPUT.SIZE_TRAIN), "max_iters": a.max_iters,
           "env": env_meta(cfg, a), "cudnn_after_set_seed": t.cudnn_after_seed, "data": t.data_meta, "started": now()}
    if a.trial_s1:
        from loss.supcontrast import SupConLoss
        from solver.make_optimizer_prompt import make_optimizer_1stage
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        labels, feats = [], []
        with torch.no_grad():
            for img, vid, _, _ in t.loader1:
                img, target = img.to("cuda"), vid.to("cuda")
                with torch.cuda.amp.autocast(enabled=True):
                    f = t.model(img, target, get_image=True)
                    for i, fe in zip(target, f):
                        labels.append(i)
                        feats.append(fe.cpu())
            labels = torch.stack(labels, 0).cuda()
            feats = torch.stack(feats, 0).cuda()
        torch.cuda.synchronize()
        ext = time.perf_counter() - t0
        n_img = labels.shape[0]
        opt1 = make_optimizer_1stage(cfg, t.model)
        t.sched1.step(1)
        scaler, xent = torch.cuda.amp.GradScaler(), SupConLoss("cuda")
        batch, M, W = cfg.SOLVER.STAGE1.IMS_PER_BATCH, a.trial_s1_iters, 5
        perm = torch.randperm(n_img).to("cuda")
        t.model.train()
        times = []
        for i in range(M + W):
            if i == W:
                torch.cuda.synchronize()
                ts = time.perf_counter()
            opt1.zero_grad()
            b = perm[(i * batch) % (n_img - batch):(i * batch) % (n_img - batch) + batch]
            target, imf = labels[b], feats[b]
            with torch.cuda.amp.autocast(enabled=True):
                text = t.model(label=target, get_text=True)
            loss = xent(imf, text, target, target) + xent(text, imf, target, target)
            scaler.scale(loss).backward()
            scaler.step(opt1)
            scaler.update()
            loss.item()
            torch.cuda.synchronize()
        step_s = (time.perf_counter() - ts) / M
        per_epoch = N_TRAIN // batch + 1
        res["stage1"] = {"extract_images": n_img, "extract_s": round(ext, 2), "extract_img_s": n_img / ext,
                         "extract_s_full_est": round(ext * N_TRAIN / n_img, 2), "step_s": step_s, "iters_per_epoch": per_epoch,
                         "epoch_s_est": step_s * per_epoch, "epochs": cfg.SOLVER.STAGE1.MAX_EPOCHS,
                         "total_s_est": ext * N_TRAIN / n_img + step_s * per_epoch * cfg.SOLVER.STAGE1.MAX_EPOCHS, **gpu_mem()}
        del labels, feats, opt1, scaler
        torch.cuda.empty_cache()
        print(f"[{now()}] stage1 추정 {res['stage1']}", flush=True)
    t.make_stage2()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    t.P2.do_train_stage2(with_epochs(cfg, "STAGE2", 1), t.model, t.center, t.loader2, None, t.opt2, t.optc,
                         SchedProxy(t.sched2, 0, lambda n: None), t.loss2(), 0, 0)
    wall = time.perf_counter() - t0
    L = t.loader2.epochs[-1]
    ts, n = L["times"] + [L["t_end"]], L["n_iter"]
    skip = math.ceil(0.2 * n)
    bs = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    hist = [x.item() for x in t.s2_acc.hist]
    real_iters = len(list(iter(t.sampler))) // bs  # 샘플러를 한 번 돌려 실제 에폭 반복 수를 센다(시험 끝에서만)
    img_s = (n - skip) * bs / (ts[n] - ts[skip])
    res["stage2"] = {"iters": n, "skip_first": skip, "img_s": img_s, "it_s": img_s / bs, "wall_s_incl_text_feat": round(wall, 2),
                     "loss_first": hist[0], "loss_last": hist[-1], **gpu_mem(), "iters_per_epoch_len": len(t.loader2.loader),
                     "iters_per_epoch_sampled": real_iters, "epochs": cfg.SOLVER.STAGE2.MAX_EPOCHS,
                     "total_s_est": real_iters * bs * cfg.SOLVER.STAGE2.MAX_EPOCHS / img_s}
    res["finished"] = now()
    atomic_json(out / "trial.json", res)
    print(json.dumps(res["stage2"], indent=1), flush=True)


# ---------------------------------------------------------------- extract
def run_extract(a):
    """공식 do_inference의 특징 부분: model.eval(), autocast 없음, feat = model(img, cam_label, view_label).
    vit_clipreid.yml은 SIE_CAMERA·SIE_VIEW가 False(기본값)라 cam/view는 None. TEST.IMS_PER_BATCH, shuffle=False, val_transforms.
    test 경로는 guard가 막는다(ALLOW_TEST=1일 때만)."""
    from datasets.bases import ImageDataset
    from datasets.make_dataloader_clipreid import make_dataloader, val_collate_fn  # noqa: F401
    from model.clip import clip
    from model.make_model_clipreid import make_model
    import torchvision.transforms as T
    if torch.cuda.device_count() != 1:
        raise SystemExit("GPU가 정확히 1장 보여야 함")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, _ = build_cfg(a.k, out, a.workers)
    clip._download = lambda url, root=None: str(CLIP_WEIGHT)
    if a.images:
        paths = []
        for p in a.images:
            paths += [l.strip() for l in open(p) if l.strip()] if p.endswith(".txt") else [p]
        items = []
        for p in paths:
            pid, cam = map(int, PATTERN.search(os.path.basename(p)).groups())
            items.append((p, pid, cam - 1, 0))
        val_tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
        ds = ImageDataset(items, val_tf)
    else:
        cond, split = a.memmap
        base = C.cache_path(cond, a.k, split)
        names = json.loads(base.with_name(base.name + ".json").read_text())["names"]
        rows = []
        for n in names:
            pid, cam = map(int, PATTERN.search(n).groups())
            rows.append((n, pid, cam - 1, 0))
        if a.limit:
            rows = rows[:a.limit]
        ds = ValSet(cond, a.k, split, rows, cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    sd = torch.load(a.model_pth, map_location="cpu")
    model = make_model(cfg, num_class=sd["classifier.weight"].shape[0], camera_num=6, view_num=1)
    del sd
    model.load_param(a.model_pth)
    model.to("cuda").eval()
    loader = DataLoader(ds, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                        collate_fn=val_collate_fn)
    feats, pids, camids, names = [], [], [], []
    t0 = time.perf_counter()
    for img, pid, camid, camids_t, target_view, imgpath in loader:
        with torch.no_grad():
            img = img.to("cuda")
            cam = camids_t.to("cuda") if cfg.MODEL.SIE_CAMERA else None
            view = target_view.to("cuda") if cfg.MODEL.SIE_VIEW else None
            feat = model(img, cam_label=cam, view_label=view)
        feats.append(feat.float().cpu())
        pids += list(pid)
        camids += list(camid)
        names += list(imgpath)
    f = torch.cat(feats).numpy()
    np.save(out / "feats.npy", f)
    atomic_json(out / "feats_meta.json", {"n": len(names), "dim": int(f.shape[1]), "dtype": str(f.dtype), "names": names,
                                          "pids": [int(x) for x in pids], "camids": [int(x) for x in camids],
                                          "neck_feat": cfg.TEST.NECK_FEAT, "feat_norm_in_eval": cfg.TEST.FEAT_NORM,
                                          "note": "features are NOT L2-normalized here; official R1_mAP_eval normalizes",
                                          "sie_camera": cfg.MODEL.SIE_CAMERA, "sie_view": cfg.MODEL.SIE_VIEW,
                                          "batch": cfg.TEST.IMS_PER_BATCH, "seconds": round(time.perf_counter() - t0, 2),
                                          "model_pth": str(a.model_pth), "model_sha256": sha256(a.model_pth)})
    print(f"[{now()}] feats {f.shape} → {out / 'feats.npy'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cond", choices=CONDS, default="bicubic")
    ap.add_argument("--k", type=int, choices=(3, 4), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iters", type=int, default=0, help="시험 모드: stage 2 반복 N회 처리량만 재고 trial.json")
    ap.add_argument("--trial-s1", type=int, default=1, help="시험 모드에서 stage 1 비용 추정(특징 추출 1회 + 스텝 몇 회)")
    ap.add_argument("--trial-s1-iters", type=int, default=30)
    ap.add_argument("--ckpt", type=int, choices=(0, 1), default=0, help="이미지 인코더 resblocks 활성값 재계산")
    ap.add_argument("--workers", type=int, default=None, help="기본: 레시피 DATALOADER.NUM_WORKERS(8)")
    ap.add_argument("--save-every", type=int, nargs=2, default=[20, 5], metavar=("S1", "S2"), help="재개용 저장 주기(에폭)")
    ap.add_argument("--extract", action="store_true", help="특징 추출 모드(--model-pth, --images 또는 --memmap)")
    ap.add_argument("--model-pth")
    ap.add_argument("--images", nargs="+", help="이미지 경로들 또는 경로 목록 .txt")
    ap.add_argument("--memmap", nargs=2, metavar=("COND", "SPLIT"))
    ap.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--debug-epochs", type=int, nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--debug-iters", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.extract:
        return run_extract(a)
    out = Path(a.out)
    if a.max_iters:
        return run_trial(a)
    if (out / "DONE").exists():
        print(f"이미 완료: {out / 'DONE'}", flush=True)
        return
    FullRun(a).run()


if __name__ == "__main__":
    main()
