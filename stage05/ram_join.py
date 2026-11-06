"""중간 지시 7.3. 측정 단계별 cgroup 기록(logs/stage05/cgroup_mem.csv)을 done JSON 시각에 맞춰 잘라 memory.current 등가 값을 구한다.
usage는 회수 가능한 페이지 캐시를 포함하므로 total_rss+total_shmem과 usage−inactive_file(작업 집합)도 함께 낸다. 결과: results/stage05_ram_cgroup.csv"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONE = ROOT / "runs" / "stage05" / "done"
G = 2**30


def main():
    rows = list(csv.DictReader((ROOT / "logs" / "stage05" / "cgroup_mem.csv").open()))
    t = [float(r["t"]) for r in rows]
    out = []
    for f in sorted(DONE.glob("*.json")):
        d = json.loads(f.read_text())
        if "seconds" not in d:
            continue
        end, start = f.stat().st_mtime, f.stat().st_mtime - d["seconds"]
        sel = [r for r, x in zip(rows, t) if start <= x <= end]
        if len(sel) < 5 or sum(1 for r in sel) < 0.5 * d["seconds"]:
            continue
        usage = [int(r["usage"]) for r in sel]
        anon = [int(r["total_rss"]) + int(r["total_shmem"]) for r in sel]
        ws = [int(r["usage"]) - int(r["total_inactive_file"]) for r in sel]
        out.append({"step": f.stem, "seconds": round(d["seconds"]), "n_samples": len(sel), "usage_max_gib": round(max(usage) / G, 2), "anon_rss_shmem_max_gib": round(max(anon) / G, 2),
                    "working_set_max_gib": round(max(ws) / G, 2), "pss_sum_max_gib": round(d.get("max_pss_sum_gib", 0), 2), "rss_sum_max_gib": round(d.get("max_rss_sum_gib", 0), 2),
                    "status": (d.get("out") or {}).get("status", "")})
    p = ROOT / "results" / "stage05_ram_cgroup.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "행")


if __name__ == "__main__":
    main()
