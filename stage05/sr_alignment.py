"""F.5 위치 정렬 검사. 방법·배율별 100장, bicubic과 위상 상관 이동량(px). 0.5px 초과면 플래그. 이미지는 수치로만 다룬다. srreid-core env.
입력은 sr_generate --smoke 출력이다(train만). 결과: results/sr_alignment.csv."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.registration import phase_cross_correlation  # noqa: E402


def luma(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), np.float32)


def main():
    root = ROOT / "runs" / "stage05" / "sr_smoke"
    rows = []
    for md in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for k in (3, 4):
            d = md / f"x{k}" / "bounding_box_train"
            ref = root / "bicubic" / f"x{k}" / "bounding_box_train"
            if not d.exists() or not ref.exists():
                continue
            shifts = []
            for p in sorted(d.glob("*.webp")):
                q = ref / p.name
                if q.exists():
                    s, _, _ = phase_cross_correlation(luma(q), luma(p), upsample_factor=20, normalization=None)
                    shifts.append(s)
            if not shifts:
                continue
            s = np.abs(np.array(shifts))
            r = np.hypot(s[:, 0], s[:, 1])
            rows.append({"method": md.name, "k": k, "n": len(shifts), "mean_abs_dy": round(float(s[:, 0].mean()), 3), "mean_abs_dx": round(float(s[:, 1].mean()), 3), "mean_shift_px": round(float(r.mean()), 3),
                         "median_shift_px": round(float(np.median(r)), 3), "p95_shift_px": round(float(np.percentile(r, 95)), 3), "max_shift_px": round(float(r.max()), 3),
                         "frac_over_0.5px": round(float((r > 0.5).mean()), 3), "flag_mean_over_0.5px": bool(r.mean() > 0.5)})
            print(rows[-1], flush=True)
    with open(ROOT / "results" / "sr_alignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
