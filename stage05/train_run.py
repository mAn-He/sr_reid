"""학습 스텝 측정. 모델 env로 돈다. train 이미지만 읽는다(test 가드 유지).
impl=baseline: Stage 0 백엔드 그대로(파일 디코딩, 공식 변환). impl=fast: Part D 최적화.
task: run(처리량·손실 기록) / profile(단계별 시간) / sample(KPR 표본 1장 단계별 시간)."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage05 import backends as B  # noqa: E402


def sync():
    torch.cuda.synchronize()


def profile_steps(b, n):
    t = {"load": 0.0, "h2d": 0.0, "forward": 0.0, "loss": 0.0, "backward": 0.0, "opt": 0.0}
    sync()
    t_all = time.perf_counter()
    for _ in range(n):
        s = b.profile_step(t, sync)
    sync()
    total = time.perf_counter() - t_all
    return {"seconds_total": total, "iters": n, "it_s": n / total, **{f"{k}_s": v for k, v in t.items()},
            **{f"{k}_pct": 100 * v / total for k, v in t.items()}, "other_pct": 100 * (1 - sum(t.values()) / total), "last": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--impl", default="baseline")
    ap.add_argument("--task", default="run")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--flags", nargs="*", default=[])
    ap.add_argument("--deterministic", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if a.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    res = {"status": "ok", "args": vars(a), "torch": torch.__version__}
    code = 0
    try:
        b = B.build(a.backend, a.scale, a.batch, a.impl, bool(a.ckpt), a.seed, a.workers or None, a.flags)
        res["meta"] = b.meta()
        if a.task == "sample":
            res["sample"] = b.profile_samples(a.iters)
        elif a.task == "profile":
            for _ in range(a.warmup):
                b.train_step()
            res["profile"] = profile_steps(b, a.iters)
        else:
            sync()
            for _ in range(a.warmup):
                b.train_step()
            sync()
            losses, t0 = [], time.perf_counter()
            t_starts = time.time()
            for _ in range(a.iters):
                losses.append(b.train_step())
            sync()
            dt = time.perf_counter() - t0
            res.update(it_s=a.iters / dt, img_s=a.iters * a.batch / dt, t_start=t_starts, t_end=time.time(), losses=losses, seconds=dt)
            if not np.isfinite(losses).all():
                raise RuntimeError("손실이 유한하지 않음")
        res["max_allocated"] = torch.cuda.max_memory_allocated()
        res["max_reserved"] = torch.cuda.max_memory_reserved()
    except BaseException as e:
        res["status"] = "oom" if "out of memory" in str(e).lower() else "error"
        res["error"] = f"{type(e).__name__}: {str(e)[:800]}"
        code = 77 if res["status"] == "oom" else 1
        import traceback
        traceback.print_exc()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
