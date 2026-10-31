"""측정 하나를 별도 프로세스에서 실행한다. 모델 env의 python으로 돈다. 결과는 --out JSON. test 가드를 설치한다."""
import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage0 import backends  # noqa: E402

N_QUERY, N_GALLERY = 3368, 19732


def is_oom(e):
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def memory():
    return {"max_memory_allocated": torch.cuda.max_memory_allocated(), "max_memory_reserved": torch.cuda.max_memory_reserved(),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def task_train(a, res):
    b = backends.build(a.backend, a.scale, a.batch, a.ckpt, a.seed, stage=1 if a.task == "stage1" else 2)
    res.update(b.meta())
    print("백엔드 준비 완료", flush=True)
    sync()
    for _ in range(a.warmup):
        b.train_step()
    sync()
    res["t_start"] = time.time()
    t0 = time.perf_counter()
    losses = []
    for _ in range(a.measure):
        losses.append(b.train_step())
        sync()
    dt = time.perf_counter() - t0
    res["t_end"] = time.time()
    res.update(it_s=a.measure / dt, img_s=a.measure * a.batch / dt, loss_last=losses[-1])
    if not np.isfinite(losses).all():
        raise RuntimeError("손실이 유한하지 않음")


def task_extract(a, res):
    kw = {"eval_batch": a.batch, "eval_pool": backends.EVAL_POOL} if a.backend.startswith("kpr") else {}
    b = backends.build(a.backend, a.scale, 64 if a.backend != "bot" else 64, False, a.seed, **kw)
    res.update({k: v for k, v in b.meta().items() if k in ("torch", "visrank", "num_workers")})
    loader = b.extract_loader(a.batch)
    n = 0
    t0 = None
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i == a.warmup:
                sync()
                res["t_start"] = time.time()
                t0 = time.perf_counter()
            b.extract_batch(data)
            sync()
            if i >= a.warmup:
                n += a.batch
            if i + 1 >= a.warmup + a.measure:
                break
    res["t_end"] = time.time()
    res.update(img_s=n / (time.perf_counter() - t0), feature=b.feature_shape())


def task_equiv(a, res):
    b = backends.build(a.backend, a.scale, a.batch, False, a.seed)
    if not hasattr(b, "equivalence"):
        raise RuntimeError(f"{a.backend}: 등가 검사 미구현")
    res.update(b.equivalence(a.batch))


def random_features(a, device):
    g = torch.Generator().manual_seed(0)
    if a.parts:
        q, gal = (torch.randn(n, a.parts, a.dim, generator=g) for n in (N_QUERY, N_GALLERY))
        q, gal = torch.nn.functional.normalize(q, dim=2), torch.nn.functional.normalize(gal, dim=2)
        if "bool" in a.vis_dtype:
            qv, gv = (torch.rand(n, a.parts, generator=g) > 0.2 for n in (N_QUERY, N_GALLERY))
        else:
            qv, gv = (torch.rand(n, a.parts, generator=g) for n in (N_QUERY, N_GALLERY))
        return q, gal, qv, gv
    q, gal = (torch.nn.functional.normalize(torch.randn(n, a.dim, generator=g), dim=1).to(device) for n in (N_QUERY, N_GALLERY))
    return q, gal, None, None


def distance_fn(a):
    use_gpu = a.device == "gpu"
    if a.backend == "bot":
        return lambda q, g, qv, gv: (1 - q @ g.T).float().cpu().numpy()  # eval/metrics_reid.py evaluate_query와 같은 식
    if a.backend == "clipreid":
        sys.path.insert(0, str(backends.CLIP_REPO))
        from utils.metrics import euclidean_distance
        return lambda q, g, qv, gv: euclidean_distance(q, g)
    sys.path.insert(0, str(backends.KPR_REPO))
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features as f

    def kpr(q, g, qv, gv):
        out = f(q, g, qv, gv, "mean", 500, use_gpu, "euclidean", use_logger=False)[0]
        return out.detach().float().cpu().numpy() if torch.is_tensor(out) else np.asarray(out, np.float32)
    return kpr


def task_dist(a, res):
    device = "cuda" if a.device == "gpu" else "cpu"
    q, g, qv, gv = random_features(a, device)
    fn = distance_fn(a)
    sync()
    res["t_start"] = time.time()
    t0 = time.perf_counter()
    if a.block == 0:
        dist = fn(q, g, qv, gv)
    else:
        parts = []
        for i in range(0, N_QUERY, a.block):
            parts.append(fn(q[i:i + a.block], g, None if qv is None else qv[i:i + a.block], gv))
        dist = np.concatenate(parts, 0)
    sync()
    res["seconds"] = time.perf_counter() - t0
    res["t_end"] = time.time()
    res.update(memory())
    res["shape"] = list(dist.shape)
    ref = Path(a.ref)
    if a.block == 0:
        ref.parent.mkdir(parents=True, exist_ok=True)
        np.save(ref, dist.astype(np.float32))
        res["max_abs_error_vs_full"] = 0.0
    else:
        res["max_abs_error_vs_full"] = float(np.abs(dist.astype(np.float32) - np.load(ref)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--task", required=True, choices=["train", "stage1", "extract", "equiv", "dist"])
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--vis-dtype", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    res = {"status": "ok", "args": vars(a)}
    code = 0
    try:
        {"train": task_train, "stage1": task_train, "extract": task_extract, "equiv": task_equiv, "dist": task_dist}[a.task](a, res)
    except BaseException as e:
        if isinstance(e, guard.GuardViolation):
            code = 3
        res["status"] = "oom" if is_oom(e) else "error"
        res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        code = code or (77 if res["status"] == "oom" else 1)
        import traceback
        traceback.print_exc()
    if torch.cuda.is_available():
        res.update({k: v for k, v in memory().items() if k not in res})
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    os.replace(tmp, a.out)
    sys.stdout.flush()
    os._exit(code)  # OOM 뒤 DataLoader 작업자 정리에 걸리지 않게 바로 끝낸다.


if __name__ == "__main__":
    main()
