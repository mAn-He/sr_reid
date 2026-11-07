"""Stage 1 2단계: FFT 지표(Y 채널)와 ΔE2000. 정의는 stage1/fft_color.py 문서 문자열. srreid-quality env, CPU 다중 프로세스.
출력(재시작 시 있는 집합은 건너뜀):
  runs/stage1_metrics/fft/<method>__x<k>.csv  이미지별 지표
  runs/stage1_metrics/fft/<method>__x<k>__radial.npz  방사형 파워 스펙트럼(이미지별 행렬, 그림용), orig__radial.npz(원본, k=1)
2,250장이 다 있는 SR 집합만 처리한다."""
import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
import fft_color as fc  # noqa: E402
from metrics_embed import INP, MET, METHODS, SR, names  # noqa: E402

OD = MET / "fft"


def load(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def one(args):
    m, k, n = args
    o = load(INP / f"{n}.png")
    s = load(SR / m / f"x{k}" / f"{n}.webp")
    r = fc.fft_metrics(o, s, k)
    r["de2000"] = fc.de2000(o, s)
    f, p = fc.radial_profile(s, k)
    return n, r, p, f


def orig_radial(n):
    return fc.radial_profile(load(INP / f"{n}.png"), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    ns = names()
    OD.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool:
        if not (OD / "orig__radial.npz").exists():
            res = pool.map(orig_radial, ns, chunksize=16)
            np.savez(OD / "orig__radial.npz", f=res[0][0], P=np.stack([p for _, p in res]).astype(np.float32), names=np.array(ns))
        for m in METHODS:
            for k in (3, 4):
                s = f"{m}__x{k}"
                if (OD / f"{s}.csv").exists():
                    continue
                if not all((SR / m / f"x{k}" / f"{n}.webp").exists() for n in ns):
                    print("미완성, 건너뜀", s, flush=True)
                    continue
                res = pool.map(one, [(m, k, n) for n in ns], chunksize=8)
                keys = list(res[0][1])
                tmp = OD / f"{s}.csv.tmp"
                with open(tmp, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["name"] + keys)
                    for n, r, _, _ in res:
                        w.writerow([n] + [repr(float(r[x])) for x in keys])
                np.savez(OD / f"{s}__radial.npz", f=res[0][3], P=np.stack([p for _, _, p, _ in res]).astype(np.float32), names=np.array(ns))
                tmp.replace(OD / f"{s}.csv")
                print("fft", s, flush=True)


if __name__ == "__main__":
    main()
