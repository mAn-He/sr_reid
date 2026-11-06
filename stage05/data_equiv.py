"""Part D 보강. 200 iteration 손실 비교는 GPU 비결정성에 묻히므로, 학습 없이 로더 출력 자체를 기준선과 비교한다. 모델 env, train만 읽는다.
run: 배치별 sha256과 합계를 json으로, --dump로 고른 iteration은 배열을 npz로 남긴다. compare: 두 json과 덤프를 비교해 CSV 한 행을 쓴다."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402

RES = ROOT / "results" / "stage05_data_equiv.csv"
FIELDS = ["model", "input", "batch", "workers", "comparison", "iters", "key", "n_batches_differ", "first_diff_iter", "max_abs_diff_dumped", "dumped_iters", "note"]


def batch_arrays(b, backend):
    if backend == "bot":
        x, y = b._input()
        return {"image": x, "pid": y}
    if backend == "clipreid":
        x, y = b._h2d(next(b.loader))
        return {"image": x, "pid": y}
    imgs, tm, pm, kp, pids, _, cam = b.engine.parse_data_for_train(next(b.loader))
    return {"image": imgs, "target_masks": tm, "prompt_masks": pm, "keypoints": kp, "pid": pids, "camid": cam}


def run(a):
    import torch

    if a.cpu:  # GPU를 쓰는 다른 작업과 겹칠 때 로더 출력만 CPU에서 받는다.
        torch.Tensor.cuda = lambda self, *x, **k: self
        torch.nn.Module.cuda = lambda self, *x, **k: self
        torch.cuda.is_available = lambda: False
    from stage05 import backends as B
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    b = B.build(a.backend, a.scale, a.batch, "baseline", False, a.seed, a.workers, a.flags)
    dump = {int(i) for i in a.dump.split(",") if i != ""}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.iters):
        arrs = {k: v.detach().cpu().numpy() for k, v in batch_arrays(b, a.backend).items() if v is not None}
        rows.append({k: {"sha": hashlib.sha256(v.tobytes()).hexdigest(), "dtype": str(v.dtype), "shape": list(v.shape), "sum": float(v.astype(np.float64).sum())} for k, v in arrs.items()})
        if i in dump:
            np.savez(out.with_name(f"{out.stem}_{i:04d}.npz"), **arrs)
    out.write_text(json.dumps({"args": vars(a), "rows": rows}))
    sys.stdout.flush()
    os._exit(0)


def compare(a):
    pa, pb = Path(a.a), Path(a.b)
    ra, rb = json.loads(pa.read_text())["rows"], json.loads(pb.read_text())["rows"]
    if len(ra) != len(rb):
        raise SystemExit(f"길이 다름 {len(ra)} {len(rb)}")
    out = []
    for key in ra[0]:
        diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if x[key]["sha"] != y[key]["sha"]]
        mx, dumped = None, []
        for i in diff:
            fa, fb = pa.with_name(f"{pa.stem}_{i:04d}.npz"), pb.with_name(f"{pb.stem}_{i:04d}.npz")
            if fa.exists() and fb.exists():
                d = float(np.abs(np.load(fa)[key].astype(np.float64) - np.load(fb)[key].astype(np.float64)).max())
                mx = d if mx is None else max(mx, d)
                dumped.append(i)
        out.append({"model": a.model, "input": a.input, "batch": a.batch, "workers": a.workers, "comparison": a.comparison, "iters": len(ra), "key": key, "n_batches_differ": len(diff),
                    "first_diff_iter": diff[0] if diff else "", "max_abs_diff_dumped": "" if mx is None else mx, "dumped_iters": " ".join(map(str, dumped)),
                    "note": f"dtype {ra[0][key]['dtype']}→{rb[0][key]['dtype']}" if ra[0][key]["dtype"] != rb[0][key]["dtype"] else ""})
    new = not RES.exists()
    with RES.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(out)
    for r in out:
        print(r, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", required=True)
    r.add_argument("--scale", type=int, default=3)
    r.add_argument("--batch", type=int, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--iters", type=int, default=200)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--flags", nargs="*", default=[])
    r.add_argument("--dump", default="")
    r.add_argument("--cpu", action="store_true")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    for name in ("a", "b", "model", "input", "comparison"):
        c.add_argument(f"--{name}", required=True)
    c.add_argument("--batch", type=int, required=True)
    c.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
