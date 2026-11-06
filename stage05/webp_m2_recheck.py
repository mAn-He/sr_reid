"""WebP 무손실 method 2 단독 재측정(새 서버, 다른 작업 없이). 실제 SR 출력 100장(스모크, runs/stage05/sr_smoke/<method>/x<k>/bounding_box_train).
스모크 WebP는 무손실이라 디코딩한 배열이 SR 출력 픽셀 그대로다. 그것을 method 2로 다시 인코딩해
평균 크기, PNG 대비 비, 단일 프로세스 인코딩 시간, 13프로세스 병렬 처리량, 픽셀 일치를 잰다. srreid-core env.
출력: results/stage05_webp_m2_recheck_s2.csv"""
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

SMOKE = ROOT / "runs" / "stage05" / "sr_smoke"
OUT = ROOT / "results" / "stage05_webp_m2_recheck_s2.csv"
SERVER = "새 서버(EPYC 7282, cgroup v2, 116.4GiB)"
PROCS = 13
FIELDS = ["server", "method", "k", "n", "png_mean_kb", "webp_m2_mean_kb", "ratio_vs_png", "encode_s_per_image_1proc", "decode_ms_per_image", "pixel_mismatch",
          "parallel_img_s_13proc", "smoke_file_mean_kb"]


def enc(a, fmt):
    b = io.BytesIO()
    if fmt == "PNG":
        Image.fromarray(a).save(b, "PNG")
    else:
        Image.fromarray(a).save(b, "WEBP", lossless=True, quality=100, method=2)
    return b


def _size(a):
    return enc(a, "WEBP").tell()


def main():
    rows = []
    for mdir in sorted(p for p in SMOKE.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for k in (4, 3):
            d = mdir / f"x{k}" / "bounding_box_train"
            files = sorted(d.glob("*.webp")) if d.exists() else []
            if len(files) < 100:
                continue
            files = files[:100]
            arrs = []
            for p in files:
                with Image.open(p) as im:
                    arrs.append(np.asarray(im.convert("RGB")))
            png = sum(enc(a, "PNG").tell() for a in arrs)
            size, t_enc, t_dec, bad = 0, 0.0, 0.0, 0
            for a in arrs:
                t = time.perf_counter()
                b = enc(a, "WEBP")
                t_enc += time.perf_counter() - t
                size += b.tell()
                b.seek(0)
                t = time.perf_counter()
                with Image.open(b) as x:
                    y = np.asarray(x.convert("RGB"))
                t_dec += time.perf_counter() - t
                bad += int(not np.array_equal(a, y))
            with ProcessPoolExecutor(PROCS) as ex:
                jobs = arrs * 3
                t = time.perf_counter()
                list(ex.map(_size, jobs, chunksize=4))
                rate = len(jobs) / (time.perf_counter() - t)
            n = len(arrs)
            rows.append({"server": SERVER, "method": mdir.name, "k": k, "n": n, "png_mean_kb": round(png / n / 1024, 2), "webp_m2_mean_kb": round(size / n / 1024, 2),
                         "ratio_vs_png": round(size / png, 3), "encode_s_per_image_1proc": round(t_enc / n, 4), "decode_ms_per_image": round(1000 * t_dec / n, 3),
                         "pixel_mismatch": bad, "parallel_img_s_13proc": round(rate, 1), "smoke_file_mean_kb": round(sum(p.stat().st_size for p in files) / n / 1024, 2)})
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
