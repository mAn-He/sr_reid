"""중간 지시 1. results CSV 정리: OOM·실패 칸의 측정값 0은 NaN으로, status는 ok / oom / crashed 세 값으로, 원래 값은 status_detail에 남긴다. 여러 번 돌려도 같다."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"stage05_vram.csv": ("max_allocated_gib", "max_reserved_gib", "it_s"),
         "stage05_throughput.csv": ("it_s", "img_s", "gpu_util_mean"),
         "stage05_kpr_memory.csv": ("peak_allocated_gib", ),
         "stage05_ram.csv": ()}


def norm(v):
    if v == "ok":
        return "ok"
    return "oom" if v == "oom" else "crashed"


def main():
    for name, cols in FILES.items():
        p = ROOT / "results" / name
        if not p.exists():
            print(name, "없음")
            continue
        rows = list(csv.DictReader(p.open()))
        fields = list(rows[0].keys())
        if "status_detail" not in fields:
            fields.insert(fields.index("status") + 1, "status_detail")
        for r in rows:
            detail = r.get("status_detail") or r["status"]
            r["status_detail"] = detail
            r["status"] = norm(detail)
            if r["status"] != "ok":
                for c in cols:
                    if c in r and r[c] in ("", "0", "0.0", "None", "NaN", None):
                        r[c] = "NaN"
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(p)
        print(name, len(rows), "행", {s: sum(r["status"] == s for r in rows) for s in ("ok", "oom", "crashed")})


if __name__ == "__main__":
    sys.exit(main())
