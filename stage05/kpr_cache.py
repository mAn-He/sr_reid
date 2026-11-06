"""E.4 KPR 주석 사전 계산. 키포인트는 공식 read_keypoints 출력 그대로(픽셀 중심 규약으로 스케일한 JSON에서), 마스크는 저장 해상도(36×17×9)의 uint8과 float32.
data/kpr_cache/x<k>/<split>/keypoints.npz, data/kpr_cache/masks/<split>/masks_{u8,f32}.npy. 모든 condition이 같은 세트를 쓴다. srreid-kpr env, CPU."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "third_party" / "keypoint_promptable_reidentification"))
from torchreid.utils.tools import read_keypoints  # noqa: E402

KP = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"
MASK = "pifpaf_maskrcnn_filtering"
OUT = ROOT / "data" / "kpr_cache"
MARKET = "Market-1501-v15.09.15"


def kp_dir(k, split):
    tag = "orig" if k == 1 else f"x{k}"
    if split == "bounding_box_train":
        if k == 1:
            return ROOT / "data" / "kpr" / "orig" / "external_annotation" / KP / split
        return ROOT / "data" / "kpr" / f"x{k}" / "external_annotation" / KP / split
    return ROOT / "data" / "locked" / "kpr_root" / tag / MARKET / "external_annotation" / KP / split


def mask_dir(split):
    if split == "bounding_box_train":
        return ROOT / "data" / "kpr" / "orig" / "masks" / MASK / split
    return ROOT / "data" / "locked" / "kpr_annot" / "masks" / MASK / split


def _kp(args):
    path, w, h = args
    return read_keypoints(str(path), (0, 0, w, h))


def build_keypoints(k, split):
    out = OUT / f"x{k}" / split
    if (out / "keypoints.npz").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(kp_dir(k, split).glob("*.json"))
    w, h = 64 * k, 128 * k
    with ProcessPoolExecutor(12) as ex:
        arrs = list(ex.map(_kp, [(f, w, h) for f in files], chunksize=128))
    offsets = np.cumsum([0] + [a.shape[0] for a in arrs])
    np.savez(out / "keypoints.npz", kp=np.concatenate(arrs).astype(np.float64), offsets=offsets, names=np.array([f.name.split(".")[0] for f in files]))
    print(f"키포인트 x{k} {split}: {len(files)}, 스켈레톤 {offsets[-1]}", flush=True)


def _mask(args):
    path, i, u8, f32, shape = args
    m = np.load(path)
    a = np.lib.format.open_memmap(u8, mode="r+")
    a[i] = np.round(np.clip(m, 0, 1) * 255).astype(np.uint8)
    b = np.lib.format.open_memmap(f32, mode="r+")
    b[i] = m


def build_masks(split):
    out = OUT / "masks" / split
    if (out / "names.txt").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(mask_dir(split).glob("*.npy"))
    shape = (len(files), 36, 17, 9)
    u8, f32 = str(out / "masks_u8.npy"), str(out / "masks_f32.npy")
    np.lib.format.open_memmap(u8, mode="w+", dtype=np.uint8, shape=shape).flush()
    np.lib.format.open_memmap(f32, mode="w+", dtype=np.float32, shape=shape).flush()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(12) as ex:
        list(ex.map(_mask, [(f, i, u8, f32, shape) for i, f in enumerate(files)], chunksize=128))
    (out / "names.txt").write_text("\n".join(f.stem for f in files))
    print(f"마스크 {split}: {shape}, {time.perf_counter() - t0:.1f}s, u8 {os.path.getsize(u8) / 2**20:.0f}MiB, f32 {os.path.getsize(f32) / 2**20:.0f}MiB", flush=True)


def main():
    splits = sys.argv[1:] or ["bounding_box_train"]
    for s in splits:
        build_masks(s)
        for k in (1, 3, 4):
            build_keypoints(k, s)
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
