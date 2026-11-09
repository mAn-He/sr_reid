"""Stage 2 KPR·KPR+SOLIDER 학습기(2026-11-09). srreid-kpr env(torch 1.13). 외부 저장소는 수정하지 않고 런타임 대체만 쓴다.

학습 루프는 공식 torchreid engine.run 그대로다(epoch, fixbase, 옵티마이저, 스케줄러, AMP, 손실, 샘플러 모두 공식 yaml + default_config).
config: configs/kpr/{imagenet,solider}/kpr_market_train.yaml. opts로 바꾸는 값은 아래뿐이다.
  data.height/width = 128k×64k(SR 출력 크기 그대로), train.batch_size 64(공식 기본과 같음), train.seed 42(CLAUDE.md, 공식 기본 1),
  data.workers(기본 4 = 공식 default_config), model.backbone_pretrained_path = weights/stage0/, test.visrank False,
  train.eval_freq -1·test.evaluate False·model.save_model_flag False(공식 기본과 같음, 명시만).
cudnn은 공식 builder가 정하는 대로 둔다(build_torchreid_model_engine: cudnn.benchmark=True, deterministic 기본 False). 실제 값은 train_meta.json에 기록.
Stage 0.5 확정 설정:
  - memmap + mask_f32: stage05/backends.py:install_kpr_cache(cond, k). read_image는 memmap에서 np.array 복사본을 돌려준다(lru_cache 없음)
    → KPR copy_read 조건(캐시 배열 제자리 수정 방지)을 만족한다. 시작 때 확인한다.
  - 마스크: 원본 키포인트 기반 float32 36×17×9(data/kpr_cache/masks), 모든 condition 공통. 키포인트: data/kpr_cache/x{k}/keypoints.npz.
  - BIPO는 학습 변환에만 있다(공식 transforms). 핀 메모리 켬(공식 datamanager pin_memory=use_gpu).
  - BIPO occluder: 공식 load_human_occluders는 cv2.imread(img_path)로 직접 읽는다. img_path는 data/kpr/x{k}의 bicubic PNG이므로
    그대로 두면 SR condition에서도 occluder 픽셀이 bicubic이 된다. 그 모듈의 cv2.imread만 같은 condition memmap 조회(BGR)로 대체한다.
    마스크(np.load, 원본 마스크 링크)와 키포인트(JSON)는 공식 경로 그대로다.
학습 중 평가: train.eval_freq=-1이라 Writer.intermediate_evaluate가 항상 False다. 학습 뒤 공식 "Final test"는 engine.test를
  기록용 대체 함수로 바꿔 실행하지 않는다(학습 루프가 모두 끝난 뒤라 가중치·RNG와 무관). query/gallery는 train 앞 8장의 더미이며 순회하지 않는다.
체크포인트·재개: 공식 저장(save_model_flag=False)은 아무것도 저장하지 않고, 공식 resume은 EngineState를 resume 전에 만들어 시작 epoch가
  반영되지 않으며 GradScaler·RNG를 저장하지 않는다. 그래서 epoch 끝 리스너에서 <out>/resume.pth.tar(model, optimizer, scheduler, scaler,
  epoch, torch/cuda/numpy/random RNG 상태)를 --save-every epoch마다 원자적으로 저장하고, 다시 실행하면 거기서 이어간다.
  재개는 RNG·데이터 순서 기준으로 정확하다(재개 직후 epoch의 첫 배치 sha1을 train_log.jsonl에 기록해 대조 가능).
  수치는 비트 단위로 같지 않다: cudnn.benchmark=True와 비결정 커널 때문에 중단 없는 두 실행끼리도 비트 단위로 같지 않다.

사용:
  학습  train_kpr.py --model kpr|kpr_solider --cond bicubic --k 3 --out runs/stage2/kpr/bicubic_x3 [--ckpt 0] [--workers 4]
        → <out>/model.pth.tar, train_meta.json, train_log.jsonl, DONE (DONE이 있으면 바로 종료)
  시험  ... --max-iters 200 → <out>/trial.json (fixbase_epoch>0이면 fixbase 구간 N/2 + 전체 학습 구간 N/2)
  추출  train_kpr.py --extract --model kpr --model-pth <model.pth.tar> --cond C --k K --split S [--limit N] --out <dir>
        → <dir>/<split>/{feat,vis,pid,cam}.npy, names.txt, extract.json (공식 test 임베딩 bn_foreg+parts, 가시성, autocast 없음)
test split은 가드로 막혀 있다(ALLOW_TEST=1일 때만). 추출의 test 연결은 메인 세션이 한다.
"""
import argparse
import fcntl
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

KPR_REPO = ROOT / "third_party" / "keypoint_promptable_reidentification"
sys.path.insert(0, str(KPR_REPO))

from common.util import sha256_file  # noqa: E402
from stage05 import cache as C  # noqa: E402
from stage05.backends import install_kpr_cache  # noqa: E402

SEED = 42
BATCH = 64
SPLIT = "bounding_box_train"
CONDS = ("bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman")
PATTERN = re.compile(r"([-\d]+)_c(\d)")
W_STAGE0 = ROOT / "weights" / "stage0"
EVAL_POOL = 8


def size_of(k):
    return 128 * k, 64 * k


def stem(name):
    return os.path.basename(name).split(".")[0]


def gib(b):
    return round(b / 2**30, 3)


def atomic_write(path, text):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save(obj, path):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def train_paths(k):
    """공식 데이터셋 파일 목록(TrainOnly와 같은 규칙). 경로는 키로만 쓰고 픽셀은 memmap에서 파일명으로 찾는다."""
    return sorted(glob.glob(str(ROOT / "data" / "kpr" / f"x{k}" / SPLIT / "*.png")))


def prepare_cache(cond, k):
    """condition memmap을 준비하고 TrainOnly 목록과 파일명 집합이 정확히 같은지 확인한다. 원본이 불완전하면 만들지 않고 멈춘다."""
    want = sorted(stem(p) for p in train_paths(k))
    assert len(want) == 12936 and len(set(want)) == len(want), len(want)
    src = C.source_dir(cond, k, SPLIT)
    files = [p for p in src.iterdir() if p.suffix in (".webp", ".png", ".jpg")]
    have = sorted(stem(p.name) for p in files)
    if have != want:
        raise SystemExit(f"원본 불완전: {src} {len(have)}장 (필요 {len(want)}장, 누락 {len(set(want) - set(have))}, 초과 {len(set(have) - set(want))})")
    newest = max(p.stat().st_mtime for p in files)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    base = C.cache_path(cond, k, SPLIT)
    with open(base.with_name(base.name + ".lock"), "w") as lk:  # 같은 캐시를 동시에 만드는 다른 학습기와 직렬화
        fcntl.flock(lk, fcntl.LOCK_EX)
        info = C.build(cond, k, SPLIT)
        meta = json.loads(base.with_name(base.name + ".json").read_text())
        names = sorted(stem(n) for n in meta["names"])
        stale = names != want or meta.get("source") != str(src) or Path(info["path"]).stat().st_mtime < newest
        if stale:
            print(f"memmap 재생성(stale: names {names == want}, source {meta.get('source')}, mtime)", flush=True)
            info = C.build(cond, k, SPLIT, force=True)
            meta = json.loads(base.with_name(base.name + ".json").read_text())
            names = sorted(stem(n) for n in meta["names"])
        fcntl.flock(lk, fcntl.LOCK_UN)
    assert names == want, "memmap 파일명 집합이 TrainOnly 목록과 다름"
    h, w = size_of(k)
    assert tuple(meta["shape"]) == (len(want), h, w, 3), meta["shape"]
    return {"source": str(src), "n": len(want), "shape": meta["shape"], "cache": info["path"], "cache_reused": info["reused"],
            "names_sha256": hashlib.sha256("\n".join(want).encode()).hexdigest()}


class Cv2Proxy:
    """batch_wise_inter_person_occlusion 모듈의 cv2 대체. imread만 condition memmap(RGB→BGR 복사본)으로, 나머지는 cv2 그대로."""

    def __init__(self, cond, k):
        import cv2
        self._cv2, self._cond, self._k, self._mm, self.reads = cv2, cond, k, None, 0

    def imread(self, path, *a):
        if self._mm is None:
            self._mm = C.load(self._cond, self._k, SPLIT)
        mm, idx = self._mm
        self.reads += 1
        return np.ascontiguousarray(np.array(mm[idx[stem(path)]])[..., ::-1])

    def __getattr__(self, name):
        return getattr(self._cv2, name)


def install_data(cond, k):
    install_kpr_cache(k, "f32", cond=cond)
    from torchreid.data.data_augmentation import batch_wise_inter_person_occlusion as bw
    from torchreid.data.datasets import dataset as dmod
    proxy = Cv2Proxy(cond, k)
    bw.cv2 = proxy
    img = dmod.read_image(train_paths(k)[0])  # copy_read 확인: memmap 뷰가 아니라 자기 메모리를 가진 복사본이어야 한다
    assert type(img) is np.ndarray and img.flags.owndata and img.flags.writeable and not hasattr(dmod.read_image, "cache_info")
    return proxy


def register_trainonly(k, eval_pool=EVAL_POOL):
    """stage0/backends.py:KprBackend의 TrainOnly와 같다(bounding_box_train만, 더미 query/gallery = train 앞 eval_pool장)."""
    import copy
    import torchreid
    from torchreid.data.datasets.image.market1501 import Market1501
    from torchreid.data.datasets.dataset import ImageDataset
    name = f"srreid_s2_x{k}"
    root = ROOT / "data" / "kpr" / f"x{k}"

    class TrainOnly(Market1501):
        dataset_dir = str(root)

        def __init__(self, root="", masks_dir=None, **kw):
            self.kp_dir = kw["config"].model.kpr.keypoints.kp_dir
            self.masks_dir = masks_dir
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[masks_dir]
            self.dataset_dir = TrainOnly.dataset_dir
            train = self.process_dir(os.path.join(self.dataset_dir, SPLIT), relabel=True)
            ImageDataset.__init__(self, train, copy.deepcopy(train[:eval_pool]), copy.deepcopy(train[:eval_pool]), **kw)

        def process_dir(self, dir_path, relabel=False):
            paths = sorted(glob.glob(os.path.join(dir_path, "*.png")))
            pids = sorted({int(PATTERN.search(os.path.basename(p)).group(1)) for p in paths})
            remap = {p: i for i, p in enumerate(pids)}
            out = []
            for p in paths:
                pid, cam = map(int, PATTERN.search(os.path.basename(p)).groups())
                out.append({"img_path": p, "pid": remap[pid] if relabel else pid, "camid": cam - 1, "kp_path": self.infer_kp_path(p),
                            "masks_path": self.infer_masks_path(p, self.masks_dir, self.masks_suffix)})
            return out

    torchreid.data.register_image_dataset(name, TrainOnly)
    return name, TrainOnly


def build(model, k, workers, save_root, job, yaml_kind="train", extra=()):
    from torchreid.scripts.builder import build_config, build_torchreid_model_engine
    name, _ = register_trainonly(k)
    h, w = size_of(k)
    pre = "solider" if model == "kpr_solider" else "imagenet"
    yaml_path = KPR_REPO / "configs" / "kpr" / pre / f"kpr_market_{yaml_kind}.yaml"
    opts = ["data.workers", workers, "data.height", h, "data.width", w, "train.batch_size", BATCH, "train.seed", SEED,
            "model.backbone_pretrained_path", str(W_STAGE0) + "/", "test.visrank", False,
            "train.eval_freq", -1, "test.evaluate", False, "model.save_model_flag", False, *extra]
    args = SimpleNamespace(root=str(ROOT / "data" / "kpr"), save_dir=str(save_root), inference_enabled=False, sources=[name], targets=[name],
                           transforms=None, job_id=job, opts=opts)
    cfg = build_config(args=args, config_path=str(yaml_path))
    engine, net = build_torchreid_model_engine(cfg)
    return cfg, engine, net, str(yaml_path)


def set_ckpt(net, on):
    n = 0
    for m in net.modules():
        if hasattr(m, "with_cp"):
            m.with_cp = bool(on)
            n += 1
    if on and n == 0:
        raise RuntimeError("with_cp 모듈을 찾지 못함")
    return n


def env_info(a):
    cd = torch.backends.cudnn
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn_version": cd.version(), "gpu": torch.cuda.get_device_name(0),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "torch_num_threads": torch.get_num_threads(),
            "cudnn": {"enabled": cd.enabled, "benchmark": cd.benchmark, "deterministic": cd.deterministic, "allow_tf32": cd.allow_tf32},
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "conc_note": os.environ.get("SRREID_CONC_NOTE"),
            "host": os.uname().nodename, "pid": os.getpid()}


def recipe(cfg, engine, yaml_path):
    t = cfg.train
    return {"yaml": os.path.relpath(yaml_path, ROOT), "epochs": int(t.max_epoch), "fixbase_epoch": int(t.fixbase_epoch), "open_layers": list(t.open_layers),
            "batch": int(t.batch_size), "num_instances": int(cfg.sampler.num_instances), "sampler": cfg.sampler.train_sampler,
            "optim": t.optim, "lr": float(t.lr), "weight_decay": float(t.weight_decay), "scheduler": t.lr_scheduler,
            "amp": bool(t.mixed_precision), "scaler": engine.scaler is not None, "seed": int(t.seed), "transforms": list(cfg.data.transforms),
            "bipo_p": float(cfg.data.bipo.p), "norm_mean": list(cfg.data.norm_mean), "norm_std": list(cfg.data.norm_std),
            "input_hw": [int(cfg.data.height), int(cfg.data.width)], "backbone": cfg.model.kpr.backbone, "workers": int(cfg.data.workers),
            "eval_freq": int(t.eval_freq), "test_evaluate": bool(cfg.test.evaluate), "save_model_flag": bool(cfg.model.save_model_flag),
            "iters_per_epoch": len(engine.train_loader), "pin_memory": bool(engine.train_loader.pin_memory)}


def masks_frozen(net):
    m = net.module if isinstance(net, torch.nn.DataParallel) else net
    mpe = getattr(m.backbone_appearance_feature_extractor, "masks_patch_embed", None)
    return None if mpe is None else not any(p.requires_grad for p in mpe.parameters())


class Limited:
    """--debug-iters 전용. 공식 train_loader의 앞 n 배치만 낸다."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n
        self.pin_memory = loader.pin_memory

    def __len__(self):
        return min(self.n, len(self.loader))

    def __iter__(self):
        it = iter(self.loader)
        for _ in range(len(self)):
            yield next(it)


def batch_sha1(data):
    return hashlib.sha1("\n".join(os.path.basename(p) for p in data["img_path"]).encode()).hexdigest()


def common_setup(a):
    torch.cuda.set_device(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = prepare_cache(a.cond, a.k)
    proxy = install_data(a.cond, a.k)
    job = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    cfg, engine, net, yaml_path = build(a.model, a.k, a.workers, out / "engine", job)
    ckpt_blocks = set_ckpt(net, a.ckpt) if a.ckpt else 0
    calls = []

    def no_test(epoch, *x, **kw):  # 학습 뒤 공식 Final test 대체. 학습 루프가 끝난 뒤에만 불린다(eval_freq=-1).
        calls.append({"epoch": int(epoch), "training": bool(engine.engine_state.is_training)})
        if engine.engine_state.is_training:
            raise RuntimeError("학습 중 평가가 호출됨")
        return 0.0, 0.0, 0.0
    engine.test = no_test
    assert not engine.writer.intermediate_evaluate() and cfg.train.eval_freq <= 0
    return out, cache, proxy, cfg, engine, net, yaml_path, ckpt_blocks, calls


def trial(a):
    out, cache, proxy, cfg, engine, net, yaml_path, ckpt_blocks, _ = common_setup(a)
    es, w = engine.engine_state, engine.writer
    fixbase = int(cfg.train.fixbase_epoch)
    phases = [("fixbase", 0), ("train_all", fixbase)] if fixbase > 0 else [("train_all", 0)]
    per = a.max_iters // len(phases)

    def cycle(loader):
        while True:
            for x in loader:
                yield x
    it = cycle(engine.train_loader)
    res = {}
    torch.cuda.reset_peak_memory_stats()
    for name, ep in phases:  # engine.train과 같은 순서: set_model_mode → two_stepped_transfer_learning → 배치 루프
        es.epoch = ep
        engine.set_model_mode("train")
        engine.two_stepped_transfer_learning(ep, fixbase, cfg.train.open_layers)
        torch.cuda.reset_peak_memory_stats()
        times, losses = [], []
        torch.cuda.synchronize()
        for _ in range(per):
            t0 = time.perf_counter()
            w.data_loading_timer.start()
            data = next(it)
            w.data_loading_timer.stop()
            w.batch_timer.start()
            loss, summary = engine.forward_backward(data)
            w.batch_timer.stop()
            w.losses.update(summary)
            w.loss.update(loss)
            losses.append(float(loss.item()))
            times.append(time.perf_counter() - t0)
        warm = int(np.ceil(0.2 * per))
        res[name] = {"epoch_state": ep, "masks_patch_embed_frozen": masks_frozen(net), "iters": per, "warmup_iters": warm,
                     "img_s": BATCH * (per - warm) / sum(times[warm:]), "s_per_iter": float(np.mean(times[warm:])),
                     "peak_alloc_gib": gib(torch.cuda.max_memory_allocated()), "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                     "loss_first": losses[0], "loss_last": losses[-1], "lr": engine.get_current_lr()}
        print(name, json.dumps(res[name]), flush=True)
    r = {"mode": "trial", "model": a.model, "cond": a.cond, "k": a.k, "max_iters": a.max_iters, "phases": res,
         "img_s": res[phases[-1][0]]["img_s"], "peak_alloc_gib": max(v["peak_alloc_gib"] for v in res.values()),
         "peak_reserved_gib": max(v["peak_reserved_gib"] for v in res.values()), "ckpt": int(a.ckpt), "ckpt_blocks": ckpt_blocks,
         "recipe": recipe(cfg, engine, yaml_path), "env": env_info(a), "data": cache, "occluder_reads_from_memmap": proxy.reads,
         "note": "fixbase는 epoch 0 상태, train_all은 epoch=fixbase_epoch에서 공식 two_stepped_transfer_learning을 적용한 상태. lr은 epoch 0 값 그대로(처리량 측정용)."}
    atomic_write(out / "trial.json", json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps({k: r[k] for k in ("img_s", "peak_alloc_gib", "peak_reserved_gib")}), flush=True)


def train(a):
    out = Path(a.out)
    if (out / "DONE").exists():
        print("DONE 있음, 종료", flush=True)
        return
    t_start = time.time()
    out, cache, proxy, cfg, engine, net, yaml_path, ckpt_blocks, test_calls = common_setup(a)
    es = engine.engine_state
    if a.debug_epochs:
        es.max_epoch = engine.writer.max_epoch = a.debug_epochs  # 스케줄러는 공식 max_epoch 그대로
    if a.debug_iters:
        engine.train_loader = Limited(engine.train_loader, a.debug_iters)
    max_epoch = es.max_epoch
    resume_path, log_path = out / "resume.pth.tar", out / "train_log.jsonl"
    events, prior_seconds, rng = [], 0.0, None
    if resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu")
        assert ck["model"] == a.model and ck["cond"] == a.cond and ck["k"] == a.k and ck["max_epoch"] == max_epoch, "resume 파일의 설정이 다름"
        net.load_state_dict(ck["state_dict"])
        engine.optimizer.load_state_dict(ck["optimizer"])
        engine.scheduler.load_state_dict(ck["scheduler"])
        if engine.scaler is not None:
            engine.scaler.load_state_dict(ck["scaler"])
        es.start_epoch = es.epoch = cfg.train.start_epoch = ck["epoch_done"]
        events, prior_seconds, rng = ck["events"], ck["train_seconds"], ck["rng"]
        events.append({"resume_from_epoch": ck["epoch_done"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()})
        keep = [l for l in log_path.read_text().splitlines() if l.strip() and json.loads(l).get("epoch", -1) < ck["epoch_done"]] if log_path.exists() else []
        atomic_write(log_path, "".join(x + "\n" for x in keep))
        print(f"재개: epoch {ck['epoch_done']}부터", flush=True)
        del ck
    elif log_path.exists():
        log_path.unlink()
    st = {"loss": None, "n": 0, "sha1": None, "t0": 0.0, "lr": 0.0, "seconds": prior_seconds}
    orig_fb = engine.forward_backward

    def fb(data):
        if st["n"] == 0:
            st["sha1"] = batch_sha1(data)
        loss, summary = orig_fb(data)
        d = loss.detach().float()
        st["loss"] = d if st["loss"] is None else st["loss"] + d
        st["n"] += 1
        return loss, summary
    engine.forward_backward = fb

    from torchreid.utils.engine_state import EngineStateListener

    class Hook(EngineStateListener):
        def epoch_started(self):
            st.update(loss=None, n=0, sha1=None, lr=engine.get_current_lr())
            torch.cuda.synchronize()
            st["t0"] = time.perf_counter()

        def epoch_completed(self):  # 공식 train()이 update_lr()까지 마친 뒤, epoch 증가 전에 불린다
            torch.cuda.synchronize()
            dt = time.perf_counter() - st["t0"]
            st["seconds"] += dt
            e = es.epoch
            rec = {"epoch": e, "iters": st["n"], "loss_mean": float(st["loss"].item() / max(st["n"], 1)), "lr": st["lr"],
                   "next_lr": engine.get_current_lr(), "seconds": round(dt, 2), "img_s": round(BATCH * st["n"] / dt, 2),
                   "masks_patch_embed_frozen": masks_frozen(net), "first_batch_sha1": st["sha1"], "scaler_scale": engine.scaler.get_scale() if engine.scaler else None,
                   "peak_alloc_gib": gib(torch.cuda.max_memory_allocated()), "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if (e + 1) % a.save_every == 0 and e + 1 < max_epoch:
                t0 = time.perf_counter()
                atomic_save({"model": a.model, "cond": a.cond, "k": a.k, "max_epoch": max_epoch, "epoch_done": e + 1,
                             "state_dict": net.state_dict(), "optimizer": engine.optimizer.state_dict(), "scheduler": engine.scheduler.state_dict(),
                             "scaler": engine.scaler.state_dict() if engine.scaler else None, "events": events, "train_seconds": st["seconds"],
                             "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(),
                                     "random": random.getstate()}}, resume_path)
                print(f"resume 저장 epoch {e + 1} ({time.perf_counter() - t0:.1f}s)", flush=True)
                if a.debug_kill_after == e + 1:
                    print("debug: 강제 종료", flush=True)
                    os._exit(9)
    es.add_listener(Hook(), last=True)

    if rng is not None:  # 빌드(시드 42, occluder 추출, 모델 초기화)를 끝낸 뒤 저장 시점의 RNG로 되돌린다
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["random"])
    from torchreid.scripts.default_config import engine_run_kwargs
    kw = engine_run_kwargs(cfg)
    assert kw["test_only"] is False and kw["visrank"] is False
    torch.cuda.reset_peak_memory_stats()
    engine.run(**kw)
    assert es.epoch == max_epoch - 1 and len(test_calls) == 1 and not test_calls[0]["training"], (es.epoch, test_calls)

    final = out / "model.pth.tar"
    atomic_save({"state_dict": net.state_dict(), "epoch": max_epoch, "config": cfg,
                 "srreid": {"model": a.model, "cond": a.cond, "k": a.k, "seed": SEED}}, final)
    logs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    imgs = BATCH * sum(r["iters"] for r in logs)
    meta = {"model": a.model, "cond": a.cond, "k": a.k, "seed": SEED, "debug": {"epochs": a.debug_epochs, "iters": a.debug_iters} if (a.debug_epochs or a.debug_iters) else None,
            "epochs_run": len(logs), "recipe": recipe(cfg, engine, yaml_path), "ckpt": int(a.ckpt), "ckpt_blocks": ckpt_blocks, "env": env_info(a),
            "data": cache, "occluder_reads_from_memmap_this_session": proxy.reads, "train_seconds": round(st["seconds"], 1),
            "img_s": round(imgs / st["seconds"], 2), "wall_seconds_this_session": round(time.time() - t_start, 1),
            "peak_alloc_gib_this_session": gib(torch.cuda.max_memory_allocated()), "peak_reserved_gib_this_session": gib(torch.cuda.max_memory_reserved()),
            "peak_alloc_gib_max_epoch_log": max(r["peak_alloc_gib"] for r in logs), "resume_events": events, "final_test": "skipped (engine.test 대체)",
            "model_path": str(final), "model_sha256": sha256_file(final), "save_every": a.save_every}
    atomic_write(out / "train_meta.json", json.dumps(meta, indent=1, ensure_ascii=False))
    resume_path.unlink(missing_ok=True)
    atomic_write(out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    print("완료", json.dumps({k: meta[k] for k in ("train_seconds", "img_s", "model_sha256")}), flush=True)


def extract(a):
    """stage05/kpr_run.py task=extract와 같은 경량 추출. 공식 test yaml, 학습한 가중치(strict 로드), autocast 없음."""
    import resource
    from stage05 import eval_core
    from stage05.kpr_run import install_cache
    from torchreid.data.datasets import dataset as dmod
    from torchreid.utils.tools import extract_test_embeddings
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)
    torch.cuda.set_device(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C.build(a.cond, a.k, a.split)  # test split이면 가드가 ALLOW_TEST=1을 요구한다
    mm, idx = C.load(a.cond, a.k, a.split)
    names = sorted(idx, key=lambda n: idx[n])
    names = names[:a.limit] if a.limit else names
    install_cache(SimpleNamespace(cond=a.cond, k=a.k, mask_dtype="f32"))
    dmod.load_human_occluders = lambda *p, **kw: {}  # 평가 변환에는 BIPO가 없다. train 목록의 occluder 적재만 막는다.
    import torchreid
    name, TrainOnly = register_trainonly(a.k)
    split = a.split

    class Extract(TrainOnly):
        def __init__(self, root="", masks_dir=None, **kw):
            from torchreid.data.datasets.dataset import ImageDataset
            self.kp_dir = kw["config"].model.kpr.keypoints.kp_dir
            self.masks_dir = masks_dir
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[masks_dir]
            self.dataset_dir = TrainOnly.dataset_dir
            train = self.process_dir(os.path.join(self.dataset_dir, SPLIT), relabel=True)  # 분류기 크기(751)만 맞춘다. 순회하지 않음
            q = []
            for n in names:  # 경로는 키로만 쓴다. 픽셀·마스크·키포인트는 split 디렉터리 이름 + 파일명으로 memmap에서 찾는다
                p = os.path.join(self.dataset_dir, split, n + ".png")
                pid, cam = map(int, PATTERN.search(n).groups())
                q.append({"img_path": p, "pid": pid, "camid": cam - 1, "kp_path": self.infer_kp_path(p),
                          "masks_path": self.infer_masks_path(p, self.masks_dir, self.masks_suffix)})
            ImageDataset.__init__(self, train, q, q[:1], **kw)
    ename = f"srreid_s2x_x{a.k}"
    torchreid.data.register_image_dataset(ename, Extract)
    from torchreid.scripts.builder import build_config, build_torchreid_model_engine
    h, w = size_of(a.k)
    pre = "solider" if a.model == "kpr_solider" else "imagenet"
    yaml_path = KPR_REPO / "configs" / "kpr" / pre / "kpr_market_test.yaml"
    opts = ["data.height", h, "data.width", w, "test.batch_size", a.eval_batch, "data.workers", a.workers, "test.visrank", False,
            "model.load_weights", str(Path(a.model_pth).resolve()), "model.backbone_pretrained_path", str(W_STAGE0) + "/"]
    args = SimpleNamespace(root=str(ROOT / "data" / "kpr"), save_dir=str(out / "engine"), inference_enabled=False, sources=[ename], targets=[ename],
                           transforms=None, job_id=f"x{os.getpid()}", opts=opts)
    cfg = build_config(args=args, config_path=str(yaml_path))
    engine, net = build_torchreid_model_engine(cfg)
    sd = torch.load(a.model_pth, map_location="cpu")["state_dict"]
    net.load_state_dict(sd, strict=True)  # 공식 load_pretrained_weights는 불일치 층을 조용히 버리므로 strict로 다시 확인
    engine.set_model_mode("eval")
    loader = engine.test_loader[ename]["query"]
    feats, vis, pids, cams, got = [], [], [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for data in loader:
            imgs, tm, pm, kp, pid, cam = engine.parse_data_for_eval(data)
            imgs = imgs.cuda()
            tm = tm.cuda() if tm is not None else None
            pm = pm.cuda() if pm is not None else None
            o = net(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
            f, v, _, _ = extract_test_embeddings(o, cfg.model.kpr.test_embeddings)
            feats.append(f.cpu().numpy().astype(np.float32))
            vis.append(v.cpu().numpy())
            pids.append(pid.numpy())
            cams.append(cam.numpy())
            got += [stem(p) for p in data["img_path"]]
    assert got == names
    F = np.concatenate(feats)
    eval_core.save_split(out / split, F, np.concatenate(vis), np.concatenate(pids), np.concatenate(cams), got)
    r = {"mode": "extract", "model": a.model, "model_pth": str(a.model_pth), "model_sha256": sha256_file(a.model_pth), "cond": a.cond, "k": a.k, "split": split,
         "n": len(got), "feat_shape": list(F.shape), "vis_dtype": str(vis[0].dtype), "test_embeddings": list(cfg.model.kpr.test_embeddings),
         "normalize_feature": bool(cfg.test.normalize_feature), "dist_metric": cfg.test.dist_metric, "combine": cfg.test.part_based.dist_combine_strat,
         "amp": False, "eval_batch": a.eval_batch, "seconds": round(time.perf_counter() - t0, 2), "env": env_info(a),
         "peak_alloc_gib": gib(torch.cuda.max_memory_allocated())}
    atomic_write(out / "extract.json", json.dumps(r, indent=1, ensure_ascii=False))
    os.system(f"rm -rf '{out / 'engine'}'")
    print(json.dumps({k: r[k] for k in ("n", "feat_shape", "vis_dtype", "seconds")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["kpr", "kpr_solider"])
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--k", type=int, required=True, choices=[3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iters", type=int, default=None, help="시험 모드: 실제 학습 스텝 N회, trial.json만 쓴다")
    ap.add_argument("--ckpt", type=int, default=0, choices=[0, 1])
    ap.add_argument("--workers", type=int, default=4, help="공식 default_config data.workers=4")
    ap.add_argument("--save-every", type=int, default=5, help="resume 체크포인트 간격(epoch)")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model-pth", default=None)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--debug-epochs", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--debug-iters", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--debug-kill-after", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.extract:
        assert a.model_pth, "--model-pth 필요"
        extract(a)
    elif a.max_iters:
        trial(a)
    else:
        train(a)
    sys.stdout.flush()
    os._exit(0)  # DataLoader 작업자 정리 지연으로 GPU 메모리를 붙잡는 일을 막는다


if __name__ == "__main__":
    main()
