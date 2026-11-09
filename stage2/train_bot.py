"""Stage 2 자체 구현 BoT 학습기(2026-11-09). srreid-core env.

레시피는 reid/trainer.py:train과 configs/pilot.yaml reid 절 그대로다: 120에폭, Adam lr 3.5e-4 wd 5e-4, warmup 10(0.1),
milestones 40/70 ×0.1, PK 16×4 = 배치 64, CE(ls 0.1) + batch-hard triplet(0.3), flip 0.5 / pad 10 crop / erasing 0.5,
AMP 없음, cudnn.deterministic=True·benchmark=False, 에폭마다 seed_all(seed*100003+epoch)와 같은 seed의 pk_batches,
에폭마다 DataLoader 새로 생성(worker_init로 워커 rng = torch.initial_seed()).
Stage 0.5에서 결과 불변으로 확정한 차이만 있다: memmap 캐시(stage05/cache.py) + stage05/backends.py:BotSet(worker_norm),
pin_memory=True, 입력은 128k×64k 그대로(리사이즈 없음), 선택적 gradient checkpointing(--ckpt 1, layer1~4, stage0 BotBackend와 같은 방식).
seed는 CLAUDE.md 규칙대로 42(configs/pilot.yaml의 seed 0이 아님).

사용:
  학습   train_bot.py --cond bicubic --k 3 --out runs/stage2/bot/bicubic_x3 [--ckpt 0] [--workers 4]
  시험   train_bot.py --cond bicubic --k 3 --out <dir> --max-iters 200   → <dir>/trial.json (모델 저장 안 함)
  추출   train_bot.py --extract --model-pt <model.pt> --out <feats.npy> (--paths <목록 txt> | --cache-cond C --cache-k K --cache-split S)
test split은 가드로 막혀 있다(ALLOW_TEST=1일 때만). 추출의 test 연결은 메인 세션이 한다.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.util import sha256_file  # noqa: E402
from reid.model import BoT  # noqa: E402
from reid.trainer import MEAN, STD, lr_at, pk_batches, seed_all, triplet_hard, worker_init  # noqa: E402
from stage05 import cache as C  # noqa: E402
from stage05.backends import BotSet  # noqa: E402

SEED = 42
SPLIT = "bounding_box_train"
PATTERN = re.compile(r"([-\d]+)_c(\d)")
CONDS = ("bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman")


def recipe():
    return yaml.safe_load(open(ROOT / "configs" / "pilot.yaml"))["reid"]


def items_from_cache(cond, k):
    """memmap 순서(meta names = 정렬된 파일 stem)대로 (이름, 재번호 pid, camid). pid는 정렬 순서로 0..750."""
    base = C.cache_path(cond, k, SPLIT)
    meta = json.loads(base.with_name(base.name + ".json").read_text())
    names = meta["names"]
    parsed = [tuple(map(int, PATTERN.search(n).groups())) for n in names]
    pids = sorted({p for p, _ in parsed})
    remap = {p: i for i, p in enumerate(pids)}
    return names, [remap[p] for p, _ in parsed], [c - 1 for _, c in parsed], meta


def atomic_write(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def add_ckpt(model):
    """stage0/backends.py:BotBackend와 같은 방식: backbone[4..7] = layer1~4를 checkpoint로 감싼다."""
    from torch.utils.checkpoint import checkpoint
    for i in (4, 5, 6, 7):
        block = model.backbone[i]
        orig = block.forward
        block.forward = lambda x, o=orig: checkpoint(o, x, use_reentrant=False, preserve_rng_state=True)


def gib(b):
    return round(b / 2**30, 3)


def train(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print("DONE 있음, 종료", flush=True)
        return
    r = recipe()
    workers = a.workers if a.workers is not None else r["num_workers"]
    epochs = r["epochs"] if a.debug_epochs is None else a.debug_epochs
    p, kk = r["sampler"]["p"], r["sampler"]["k"]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    C.build(a.cond, a.k, SPLIT)
    names, labels, _, meta = items_from_cache(a.cond, a.k)
    n, h, w, _ = meta["shape"]
    assert (h, w) == (128 * a.k, 64 * a.k), (h, w)
    num_classes = len(set(labels))
    labels_np = np.array(labels)

    seed_all(SEED)
    ds = BotSet(a.cond, a.k, r["augment"], labels, SEED, (MEAN, STD))
    pretrained = str(ROOT / r["pretrained"])
    model = BoT(num_classes, pretrained).cuda()
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=r["optim"]["lr"],
                           weight_decay=r["optim"]["weight_decay"])
    if a.ckpt:
        add_ckpt(model)

    start, acc = 0, {"seconds": 0.0, "images": 0, "iters": 0}
    last = out / "last.pt"
    trial = a.max_iters is not None
    if last.exists() and not trial:
        st = torch.load(last, map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start, acc = st["epoch"] + 1, st["acc"]
        print(f"재개: epoch {start}", flush=True)

    log = None if trial else open(out / "train_log.jsonl", "a")
    t_wall = time.time()
    it_times, losses, it = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, epochs):
        seed_all(SEED * 100003 + epoch)
        batches = pk_batches(labels_np, p, kk, np.random.default_rng(SEED * 100003 + epoch))
        loader = DataLoader(ds, batch_sampler=batches, num_workers=workers, worker_init_fn=worker_init, pin_memory=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(r["optim"], epoch)
        model.train()
        t0, total, imgs = time.time(), 0.0, 0
        tp = time.perf_counter()
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logit, gfeat = model(x)
            loss = F.cross_entropy(logit, y, label_smoothing=r["loss"]["ce_label_smoothing"]) \
                + triplet_hard(gfeat, y, r["loss"]["triplet_margin"])
            if not torch.isfinite(loss):
                print(f"loss가 유한하지 않음: epoch {epoch}", flush=True)
                sys.exit(2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lv = loss.item()  # 동기화 지점(레시피와 같음)
            total += lv
            imgs += x.shape[0]
            if trial:
                now = time.perf_counter()
                it_times.append(now - tp)
                tp = now
                losses.append(lv)
                it += 1
                if it >= a.max_iters:
                    break
        if trial:
            if it >= a.max_iters:
                break
            continue
        sec = time.time() - t0
        acc["seconds"] += sec
        acc["images"] += imgs
        acc["iters"] += len(batches)
        log.write(json.dumps({"epoch": epoch, "loss": total / max(len(batches), 1), "lr": opt.param_groups[0]["lr"],
                              "seconds": sec, "img_s": imgs / sec}) + "\n")
        log.flush()
        if (epoch + 1) % r["ckpt_every"] == 0 or epoch + 1 == epochs:
            tmp = str(last) + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "acc": acc}, tmp)
            os.replace(tmp, last)

    common = {"cond": a.cond, "k": a.k, "seed": SEED, "input_hw": [h, w], "n_images": n, "num_classes": num_classes,
              "batch": p * kk, "amp": False, "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark, "ckpt": int(a.ckpt), "workers": workers,
              "omp_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "conc_note": os.environ.get("SRREID_CONC_NOTE"), "data_source": meta["source"],
              "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
              "peak_reserved_gib": gib(torch.cuda.max_memory_reserved())}
    if trial:
        skip = int(len(it_times) * 0.2)
        steady = it_times[skip:]
        res = dict(common, mode="trial", iters=len(it_times), measured_iters=len(steady),
                   img_s=round(p * kk * len(steady) / sum(steady), 2) if steady else None,
                   wall_seconds=round(time.time() - t_wall, 2), loss_first=losses[0] if losses else None,
                   loss_last=losses[-1] if losses else None)
        atomic_write(out / "trial.json", json.dumps(res, indent=1))
        print("시험 완료", res, flush=True)
        return

    final = out / "model.pt"
    tmp = str(final) + ".tmp"
    torch.save({"model": model.state_dict(), "num_classes": num_classes, "cond": a.cond, "k": a.k}, tmp)
    os.replace(tmp, final)
    res = dict(common, mode="full", epochs=epochs, recipe_epochs=r["epochs"], debug_epochs=a.debug_epochs,
               iters=acc["iters"], train_seconds=round(acc["seconds"], 1), mean_img_s=round(acc["images"] / max(acc["seconds"], 1e-9), 2),
               wall_seconds_this_process=round(time.time() - t_wall, 1), resumed_from_epoch=start,
               peak_note="peak는 마지막 프로세스(재개 뒤) 기준",
               model_sha256=sha256_file(final), pretrained=r["pretrained"], pretrained_sha256=sha256_file(pretrained),
               labels_sha256=hashlib.sha256(json.dumps([names, labels]).encode()).hexdigest())
    atomic_write(out / "train_meta.json", json.dumps(res, indent=1))
    atomic_write(out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    print("완료", res, flush=True)


@torch.no_grad()
def extract(a):
    """reid/extract.py와 같다: eval, flip 없음, BN 뒤 특징 L2 정규화, 배치 128. 입력은 ×k 원래 크기 그대로(리사이즈 없음)."""
    from PIL import Image
    st = torch.load(a.model_pt, map_location="cpu")
    model = BoT(st["num_classes"])
    model.load_state_dict(st["model"])
    model = model.eval().cuda()
    batch = recipe()["eval"]["batch"]
    if a.paths:
        paths = [s for s in Path(a.paths).read_text().split("\n") if s]
        load = lambda i: np.asarray(Image.open(paths[i]).convert("RGB"))  # noqa: E731
        n = len(paths)
    else:
        C.build(a.cache_cond, a.cache_k, a.cache_split)
        mm, _ = C.load(a.cache_cond, a.cache_k, a.cache_split)
        load = lambda i: mm[i]  # noqa: E731
        n = len(mm) if a.limit is None else min(a.limit, len(mm))
    feats = []
    for i in range(0, n, batch):
        x = np.stack([(load(j).astype(np.float32) / 255 - MEAN) / STD for j in range(i, min(i + batch, n))])
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).cuda()
        feats.append(F.normalize(model(x), dim=1).cpu().numpy())
    f = np.concatenate(feats) if feats else np.zeros((0, 2048), np.float32)
    np.save(a.out, f)
    print("추출", f.shape, a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=CONDS)
    ap.add_argument("--k", type=int, choices=(3, 4))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--ckpt", type=int, default=0, choices=(0, 1))
    ap.add_argument("--workers", type=int)
    ap.add_argument("--debug-epochs", type=int, help=argparse.SUPPRESS)  # 재개 시험 전용. 기본값 없음
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model-pt")
    ap.add_argument("--paths")
    ap.add_argument("--cache-cond")
    ap.add_argument("--cache-k", type=int)
    ap.add_argument("--cache-split", default=SPLIT)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.extract:
        extract(a)
    else:
        if a.cond is None or a.k is None:
            ap.error("--cond, --k 필요")
        train(a)


if __name__ == "__main__":
    main()
