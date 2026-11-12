"""Stage 2 사후 분석 a: test SR 품질 지표(Stage 1과 같은 모델·전처리·설정). srreid-quality env. ALLOW_TEST=1(사용자 잠금 해제 중)에서만.
원본 = data/locked/sr_input/<split>/<이름>.png(원본 JPEG의 무손실 디코딩, Stage 1의 data/sr_input과 같은 규칙), SR = data/<cond>/x<k>/<split>/<이름>.webp.
지표: DINOv3-L·PersonViT-B 임베딩(원본·SR, 코사인은 분석 단계에서), A-FINE(총점·충실도·자연스러움), FFT 4종 + ΔE2000(stage1/fft_color.py).
집합: {bicubic, swinir, pisa_sr_sem0.0, osdhuman} × {3, 4} × {query, bounding_box_test}(파일 전체, junk 포함). bicubic A-FINE은 기준 이미지와 같아 비교 불가(값은 저장).
출력(잠긴 영역, 재시작 시 있는 파일 건너뜀): data/locked/stage2_analysis/metrics/{emb/<model>/<split>__<set>.npy, afine/<split>__<set>.csv, fft/<split>__<set>.csv}
배치: 임베딩 256, A-FINE 128(Stage 1 A100 실행과 같음). 메모리를 위해 4,000장씩 나눠 처리(배치 경계만 달라짐)."""
import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1"))
from common import guard  # noqa: E402

guard.install()
OUT = ROOT / "data" / "locked" / "stage2_analysis" / "metrics"
CONDS = ("bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman")
SPLITS = ("query", "bounding_box_test")
CHUNK = 4000


def names(split):
    d = ROOT / "data" / "locked" / "sr_input" / split
    return sorted(p[:-4] for p in os.listdir(d) if p.endswith(".png"))


def orig_path(split, n):
    return ROOT / "data" / "locked" / "sr_input" / split / f"{n}.png"


def sr_path(c, k, split, n):
    return ROOT / "data" / c / f"x{k}" / split / f"{n}.webp"


def pil(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def load_many(paths):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(12) as ex:
        return list(ex.map(pil, paths))


def chunks(ns):
    for i in range(0, len(ns), CHUNK):
        yield ns[i:i + CHUNK]


def run_emb(model, units):
    import eval_models as em
    e = em.dinov3("large", rope_bf16=True) if model == "dinov3_large" else em.personvit("base", norm=(0.5, 0.5))
    od = OUT / "emb" / model
    od.mkdir(parents=True, exist_ok=True)
    for split, c, k in units:
        s = f"{split}__orig" if c == "orig" else f"{split}__{c}__x{k}"
        if (od / f"{s}.npy").exists():
            continue
        ns = names(split)
        vs = []
        for part in chunks(ns):
            ims = load_many([orig_path(split, n) if c == "orig" else sr_path(c, k, split, n) for n in part])
            vs.append(e(ims, bs=256))
        np.save(od / f"{s}.tmp.npy", np.concatenate(vs).astype(np.float32))
        os.replace(od / f"{s}.tmp.npy", od / f"{s}.npy")
        print(model, s, flush=True)


def run_afine(units):
    import eval_models as em
    af = em.AFINE()
    od = OUT / "afine"
    od.mkdir(parents=True, exist_ok=True)
    for split, c, k in units:
        s = f"{split}__{c}__x{k}"
        if (od / f"{s}.csv").exists():
            continue
        ns = names(split)
        res = {}
        for part in chunks(ns):
            r = af(load_many([orig_path(split, n) for n in part]), load_many([sr_path(c, k, split, n) for n in part]), bs=128)
            for key, v in r.items():
                res.setdefault(key, []).append(v)
        res = {key: np.concatenate(v) for key, v in res.items()}
        with open(od / f"{s}.csv.tmp", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name"] + list(res))
            for i, n in enumerate(ns):
                w.writerow([n] + [repr(float(res[key][i])) for key in res])
        os.replace(od / f"{s}.csv.tmp", od / f"{s}.csv")
        print("afine", s, flush=True)


def fft_one(args):
    import fft_color as fc
    split, c, k, n = args
    o = np.asarray(pil(orig_path(split, n)))
    s = np.asarray(pil(sr_path(c, k, split, n)))
    r = fc.fft_metrics(o, s, k)
    r["de2000"] = fc.de2000(o, s)
    return n, r


def run_fft(procs):
    od = OUT / "fft"
    od.mkdir(parents=True, exist_ok=True)
    with Pool(procs) as pool:
        for split in SPLITS:
            ns = names(split)
            for c in CONDS:
                for k in (3, 4):
                    s = f"{split}__{c}__x{k}"
                    if (od / f"{s}.csv").exists():
                        continue
                    res = pool.map(fft_one, [(split, c, k, n) for n in ns], chunksize=16)
                    cols = list(res[0][1])
                    with open(od / f"{s}.csv.tmp", "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["name"] + cols)
                        for n, r in res:
                            w.writerow([n] + [repr(float(r[x])) for x in cols])
                    os.replace(od / f"{s}.csv.tmp", od / f"{s}.csv")
                    print("fft", s, flush=True)


def main():
    assert os.environ.get("ALLOW_TEST") == "1", "사용자 잠금 해제 중 ALLOW_TEST=1로만"
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["emb_dinov3", "emb_personvit", "afine3", "afine4", "fft"])
    ap.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    emb_units = [(sp, "orig", 0) for sp in SPLITS] + [(sp, c, k) for sp in SPLITS for c in CONDS for k in (3, 4)]
    if a.part == "emb_dinov3":
        run_emb("dinov3_large", emb_units)
    elif a.part == "emb_personvit":
        run_emb("personvit_base", emb_units)
    elif a.part in ("afine3", "afine4"):
        k = int(a.part[-1])
        run_afine([(sp, c, k) for sp in SPLITS for c in CONDS])
    else:
        run_fft(a.procs)


if __name__ == "__main__":
    main()
