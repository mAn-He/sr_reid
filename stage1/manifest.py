"""Stage 1 1단계 manifest: runs/stage1_sample/ 의 모든 WebP(12종 × {×3, ×4} × 2,250장)에 대해 크기·SHA256·픽셀 크기를 기록하고
생성 로그(jsonl)의 webp_sha256과 대조한다. srreid-core env.
출력: results/stage1/stage1_sample_manifest.csv.gz (method, k, name, bytes, sha256, w, h, log_match), results/stage1/stage1_sample_manifest_summary.csv"""
import csv
import gzip
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "runs" / "stage1_sample"
RES = ROOT / "results" / "stage1"
SAMPLE = ROOT / "configs" / "stage1_sample.csv"
METHODS = ["bicubic", "lanczos", "realesrgan", "osdhuman", "swinir", "hat", "mambairv2"] + [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]


def info(p):
    b = p.read_bytes()
    with Image.open(p) as im:
        w, h = im.size
    return len(b), hashlib.sha256(b).hexdigest(), w, h


def main():
    ns = [Path(r["name"]).stem for r in csv.DictReader(open(SAMPLE))]
    rows, summ = [], []
    for m in METHODS:
        for k in (3, 4):
            logs = {}
            lp = SR / "_logs" / f"{m}__x{k}.jsonl"
            if lp.exists():
                for l in open(lp):
                    d = json.loads(l)
                    if "webp_sha256" in d:
                        logs[d["name"]] = d["webp_sha256"]
            ps = [SR / m / f"x{k}" / f"{n}.webp" for n in ns]
            ex = [p for p in ps if p.exists()]
            with ThreadPoolExecutor(8) as ex_:
                infos = list(ex_.map(info, ex))
            bad_size = 0
            mism = 0
            for p, (nb, sh, w, h) in zip(ex, infos):
                lm = logs.get(p.stem)
                ok = (lm == sh) if lm else None
                mism += ok is False
                bad_size += (w, h) != (64 * k, 128 * k)
                rows.append({"method": m, "k": k, "name": p.stem, "bytes": nb, "sha256": sh, "w": w, "h": h, "log_match": ok})
            extra = len(list((SR / m / f"x{k}").glob("*.webp"))) - len(ex) if (SR / m / f"x{k}").exists() else 0
            tmp = len(list((SR / m / f"x{k}").glob("*.tmp"))) if (SR / m / f"x{k}").exists() else 0
            summ.append({"method": m, "k": k, "expected": len(ns), "present": len(ex), "missing": len(ns) - len(ex), "extra_files": extra, "tmp_files": tmp,
                         "wrong_size": bad_size, "log_sha_mismatch": mism, "log_sha_missing": sum(1 for p in ex if p.stem not in logs),
                         "total_bytes": sum(i[0] for i in infos)})
            print(summ[-1], flush=True)
    with gzip.open(RES / "stage1_sample_manifest.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(RES / "stage1_sample_manifest_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    bad = [s for s in summ if s["missing"] or s["wrong_size"] or s["log_sha_mismatch"] or s["tmp_files"]]
    print("문제 있는 집합:", bad if bad else "없음")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
