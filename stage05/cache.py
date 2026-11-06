"""memmap 캐시. <CACHE>/<condition>__x<k>__<split>.u8 (N×H×W×3 uint8) + .json(파일명, H, W). 실행 직전에 만들고 끝나면 지운다.
KPR 주석: data/kpr_cache/masks/<split>/masks_f32.npy (N×36×17×9), data/kpr_cache/x<k>/<split>/keypoints.npz는 영구 보관한다."""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CACHE = Path(guard.CACHE_ROOT)
SPLITS = ("bounding_box_train", "query", "bounding_box_test")


def cache_path(cond, k, split):
    return CACHE / f"{cond}__x{k}__{split}"


def source_dir(cond, k, split):
    """측정용 bicubic은 PNG 임시본을 쓴다. 본 데이터 계층은 data/<condition>/x<k>/<split>/*.webp."""
    new = ROOT / "data" / cond / f"x{k}" / split
    if new.is_dir():
        return new
    if cond == "bicubic":
        if split == "bounding_box_train":
            return ROOT / "data" / f"bicubic_x{k}"
        return ROOT / "data" / "locked" / "measure" / f"bicubic_x{k}" / split
    if cond == "orig":
        return ROOT / "data" / "market" / split if split == "bounding_box_train" else ROOT / "data" / "locked" / split
    raise FileNotFoundError(new)


def _decode(args):
    path, mm_path, i, shape = args
    mm = np.memmap(mm_path, np.uint8, "r+", shape=shape)
    with Image.open(path) as im:
        mm[i] = np.asarray(im.convert("RGB"))


def build(cond, k, split, workers=12, force=False):
    """같은 조건이 있으면 재사용한다. 결과: 경로, 초, 바이트."""
    base = cache_path(cond, k, split)
    data, meta = base.with_name(base.name + ".u8"), base.with_name(base.name + ".json")
    if data.exists() and meta.exists() and not force:
        return {"path": str(data), "seconds": 0.0, "bytes": data.stat().st_size, "reused": True}
    CACHE.mkdir(parents=True, exist_ok=True)
    src = source_dir(cond, k, split)
    files = sorted(p for p in src.iterdir() if p.suffix in (".webp", ".png", ".jpg"))
    if not files:
        raise FileNotFoundError(src)
    with Image.open(files[0]) as im:
        w, h = im.size
    shape = (len(files), h, w, 3)
    t0 = time.perf_counter()
    tmp = data.with_name(data.name + ".tmp")
    np.memmap(tmp, np.uint8, "w+", shape=shape).flush()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_decode, [(p, tmp, i, shape) for i, p in enumerate(files)], chunksize=64))
    os.replace(tmp, data)
    meta.write_text(json.dumps({"names": [p.stem for p in files], "shape": shape, "source": str(src)}))
    return {"path": str(data), "seconds": time.perf_counter() - t0, "bytes": data.stat().st_size, "reused": False}


def load(cond, k, split):
    base = cache_path(cond, k, split)
    meta = json.loads(base.with_name(base.name + ".json").read_text())
    mm = np.memmap(base.with_name(base.name + ".u8"), np.uint8, "r", shape=tuple(meta["shape"]))
    return mm, {n.split(".")[0]: i for i, n in enumerate(meta["names"])}  # Market 원본에 x.jpg.jpg 형식이 일부 있다


def remove(cond, k, split):
    base = cache_path(cond, k, split)
    for suf in (".u8", ".json"):
        base.with_name(base.name + suf).unlink(missing_ok=True)


def clear():
    shutil.rmtree(CACHE, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    guard.install()
    for s in a.splits:
        print(s, build(a.cond, a.k, s, force=a.force), flush=True)


if __name__ == "__main__":
    main()
