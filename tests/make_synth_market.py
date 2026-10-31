"""스모크용 합성 데이터. Market 파일명 규칙의 128×64 JPEG를 만든다. 사람 이미지가 아니라 ID별 색 패턴이다."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", type=int, default=24)
    ap.add_argument("--per-cam", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out) / "bounding_box_train"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for pid in range(1, a.ids + 1):
        base = rng.randint(0, 255, size=(8, 4, 3), dtype=np.uint8)
        for cam in range(1, 4):
            for k in range(a.per_cam):
                noise = rng.randint(-20, 20, size=(8, 4, 3))
                arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr).resize((64, 128), Image.BICUBIC)
                img.save(out / f"{pid:04d}_c{cam}s1_{k:06d}_01.jpg", quality=95)


if __name__ == "__main__":
    main()
