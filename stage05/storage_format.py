"""E.1 저장 형식 측정. 100장 기준 PNG 대비 WebP 무손실(method별) 용량, 인코딩·디코딩 시간, 픽셀 일치. 단일 프로세스로 재고, 13프로세스 실측 처리량으로 파드 CPU 한도(13.6코어) 기준 36,036장 소요를 계산한다. srreid-core env.
조건: bicubic(원본에서 직접), 그리고 있으면 스모크 SR 출력(runs/stage05/sr_smoke/<method>/x<k>/bounding_box_train)."""
import csv
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

N = 100
TOTAL = 36036
PROCS = 13  # 파드 CPU 한도 13.6코어
FIELDS = ["condition", "k", "n", "format", "method", "mean_kb", "ratio_vs_png", "encode_s_per_image", "decode_ms_per_image", "pixel_mismatch",
          "parallel_img_s_13proc", "wall_hours_36036_13proc", "cpu_hours_36036", "total_gib_36036"]


def _encode(args):
    a, fmt, method = args
    b = io.BytesIO()
    if fmt == "PNG":
        Image.fromarray(a).save(b, "PNG")
    else:
        Image.fromarray(a).save(b, "WEBP", lossless=True, quality=100, method=method)
    return b.tell()


def parallel_rate(arrs, fmt, method):
    """13프로세스 실측 처리량. 100장을 3번 돌려 준비 비용을 줄인다."""
    jobs = [(a, fmt, method) for a in arrs] * 3
    with ProcessPoolExecutor(PROCS) as ex:
        t = time.perf_counter()
        list(ex.map(_encode, jobs, chunksize=4))
        return len(jobs) / (time.perf_counter() - t)


def measure(arrs, cond, k, rows):
    png = None
    for fmt, method in (("PNG", ""), ("WEBP", 0), ("WEBP", 2), ("WEBP", 4), ("WEBP", 6)):
        size, enc, dec, bad = 0, 0.0, 0.0, 0
        for a in arrs:
            im = Image.fromarray(a)
            b = io.BytesIO()
            t = time.perf_counter()
            if fmt == "PNG":
                im.save(b, "PNG")
            else:
                im.save(b, "WEBP", lossless=True, quality=100, method=method)
            enc += time.perf_counter() - t
            size += b.tell()
            b.seek(0)
            t = time.perf_counter()
            with Image.open(b) as x:
                y = np.asarray(x.convert("RGB"))
            dec += time.perf_counter() - t
            bad += int(not np.array_equal(a, y))
        if fmt == "PNG":
            png = size
        rate = parallel_rate(arrs, fmt, method)
        rows.append({"condition": cond, "k": k, "n": len(arrs), "format": fmt, "method": method, "mean_kb": round(size / len(arrs) / 1024, 2), "ratio_vs_png": round(size / png, 3),
                     "encode_s_per_image": round(enc / len(arrs), 4), "decode_ms_per_image": round(1000 * dec / len(arrs), 3), "pixel_mismatch": bad,
                     "parallel_img_s_13proc": round(rate, 2), "wall_hours_36036_13proc": round(TOTAL / rate / 3600, 3), "cpu_hours_36036": round(TOTAL * enc / len(arrs) / 3600, 2),
                     "total_gib_36036": round(TOTAL * size / len(arrs) / 2**30, 2)})
        print(rows[-1], flush=True)


def main():
    rows = []
    src = sorted((ROOT / "data" / "market" / "bounding_box_train").glob("*.jpg"))
    picks = [src[i] for i in np.linspace(0, len(src) - 1, N).astype(int)]
    for k in (3, 4):
        arrs = []
        for p in picks:
            with Image.open(p) as im:
                arrs.append(np.asarray(im.convert("RGB").resize((64 * k, 128 * k), Image.BICUBIC)))
        measure(arrs, "bicubic", k, rows)
    smoke = ROOT / "runs" / "stage05" / "sr_smoke"
    for cond in sys.argv[1:]:
        for k in (3, 4):
            d = smoke / cond / f"x{k}" / "bounding_box_train"
            if d.is_dir():
                arrs = []
                for p in sorted(d.glob("*.webp"))[:N]:
                    with Image.open(p) as im:
                        arrs.append(np.asarray(im.convert("RGB")))
                measure(arrs, cond, k, rows)
    with open(ROOT / "results" / "storage_format.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
