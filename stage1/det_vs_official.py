"""결정적 설정(본 생성) 대 공식 기본 설정 출력 차이, 그리고 본 생성 설정의 재실행 비트 일치 확인. srreid-quality env, GPU(작음).
- det: runs/stage1_sample/<m>/x<k>/ 의 표본 앞 20장(결정적 설정 + 예열 + 파일명 seed)
- offA, offB: runs/stage1_step0/official20/{A,B}/<m>/x<k>/ (공식 기본 설정: cudnn.benchmark 공식값, SDPA 기본 백엔드, 예열 없음, 파일명 seed). 별도 프로세스 2회.
  공식 설정 자체가 실행마다 달라지므로 det−offA 차이를 offA−offB(공식 설정의 실행 간 차이)와 나란히 본다.
- det20: runs/stage1_step0/det20/ 본 생성과 같은 설정으로 20장 재생성 → 본 출력과 배열 sha1 비교.
출력: results/stage1/det_vs_official_pixels.csv, det_vs_official_metrics.csv, det20_repro.csv"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
import eval_models as em  # noqa: E402
import fft_color as fc  # noqa: E402
from metrics_embed import INP, MODELS, RES, SR, names  # noqa: E402

S0 = ROOT / "runs" / "stage1_step0"
PISA = [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]
SETS = [("hat", 3), ("hat", 4), ("mambairv2", 3), ("mambairv2", 4), ("osdhuman", 4), ("osdhuman", 3)] + [(p, k) for p in PISA for k in (4, 3)]
FFT_KEYS = ["lf_coh", "lf_relerr_db", "hf_energy", "slope_dev_hi", "grid_peak_db", "peak_db", "de2000"]


def arr(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def pil(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def main():
    ns = names()[:20]
    paths = {}
    for m, k in SETS:
        paths[(m, k, "det")] = [SR / m / f"x{k}" / f"{n}.webp" for n in ns]
        for r in ("A", "B"):
            paths[(m, k, "off" + r)] = [S0 / "official20" / r / m / f"x{k}" / f"{n}.webp" for n in ns]
    pix = []
    for m, k in SETS:
        for a, b in (("det", "offA"), ("det", "offB"), ("offA", "offB")):
            d = [np.abs(arr(p).astype(np.int16) - arr(q).astype(np.int16)) for p, q in zip(paths[(m, k, a)], paths[(m, k, b)])]
            pix.append({"method": m, "k": k, "pair": f"{a}-{b}", "n": len(d), "n_identical": sum(int(x.max() == 0) for x in d),
                        "max_abs": int(max(x.max() for x in d)), "mean_abs": float(np.mean([x.mean() for x in d])),
                        "mean_frac_px_diff": float(np.mean([(x > 0).any(-1).mean() for x in d])),
                        "p99_abs": float(np.percentile(np.concatenate([x.ravel() for x in d]), 99))})
    origs = [pil(INP / f"{n}.png") for n in ns]
    o_arr = [np.asarray(o) for o in origs]
    vals = {}
    for key, ps in paths.items():
        k = key[1]
        rows = []
        for o, p in zip(o_arr, ps):
            s = arr(p)
            r = fc.fft_metrics(o, s, k)
            r["de2000"] = fc.de2000(o, s)
            rows.append({x: r[x] for x in FFT_KEYS})
        vals[key] = {x: np.array([r[x] for r in rows]) for x in FFT_KEYS}
    for mk, build in MODELS.items():
        e = build()
        eo = e(origs)
        for key, ps in paths.items():
            vals[key][f"{mk}_cos"] = (eo * e([pil(p) for p in ps])).sum(1)
        del e
    af = em.AFINE()
    for key, ps in paths.items():
        r = af(origs, [pil(p) for p in ps])
        for x in ("afine_all_scale", "fidelity", "natural_dis"):
            vals[key][x] = r[x]
    met = []
    for m, k in SETS:
        for x in vals[(m, k, "det")]:
            d, a, b = vals[(m, k, "det")][x], vals[(m, k, "offA")][x], vals[(m, k, "offB")][x]
            met.append({"method": m, "k": k, "metric": x, "det_mean": float(d.mean()), "offA_mean": float(a.mean()), "offB_mean": float(b.mean()),
                        "mean_det_minus_offA": float((d - a).mean()), "mean_abs_det_minus_offA": float(np.abs(d - a).mean()),
                        "max_abs_det_minus_offA": float(np.abs(d - a).max()),
                        "mean_abs_offA_minus_offB": float(np.abs(a - b).mean()), "max_abs_offA_minus_offB": float(np.abs(a - b).max())})
    rep = []
    for m in ["hat", "mambairv2", "swinir", "realesrgan", "osdhuman"] + PISA:
        for k in (3, 4):
            f = S0 / "det20" / "_logs" / f"{m}__x{k}.jsonl"
            g = SR / "_logs" / f"{m}__x{k}.jsonl"
            if not f.exists():
                continue
            a = {json.loads(l)["name"]: json.loads(l)["sha1"] for l in open(f)}
            b = {}
            for l in open(g):
                d = json.loads(l)
                b.setdefault(d["name"], d["sha1"])
            common = [n for n in ns if n in a and n in b]
            rep.append({"method": m, "k": k, "n": len(common), "n_bit_identical": sum(a[n] == b[n] for n in common)})
    for name, rows in (("det_vs_official_pixels.csv", pix), ("det_vs_official_metrics.csv", met), ("det20_repro.csv", rep)):
        with open(RES / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(name, len(rows))


if __name__ == "__main__":
    main()
