"""중간 지시 7.3. 측정 단계별 cgroup 기록을 done JSON 시각에 맞춰 잘라 memory.current 등가 값을 구한다.
usage는 회수 가능한 페이지 캐시를 포함하므로 rss+shmem과 usage−inactive_file(작업 집합)도 함께 낸다. 결과: results/stage05_ram_cgroup.csv
입력: logs/stage05/cgroup_mem.csv(이전 서버, cgroup v1, total_* 열), logs/stage05/cgroup_mem_v2.csv(새 서버, stage05/cgroup_log.py 공통 열). 두 형식을 모두 읽는다."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONE = ROOT / "runs" / "stage05" / "done"
G = 2**30


SOURCES = (("cgroup_mem.csv", "이전 서버(v1)"), ("cgroup_mem_v2.csv", "새 서버"))


def load():
    """두 형식을 공통 열(t, usage, rss, shmem, inactive_file, server)로 맞춘다. '#' 주석 줄은 건너뛴다."""
    rows = []
    for name, server in SOURCES:
        p = ROOT / "logs" / "stage05" / name
        if not p.exists():
            continue
        for r in csv.DictReader(line for line in p.open(errors="replace") if not line.startswith("#") and "\0" not in line):  # 강제 종료 때 남은 NUL 줄은 버린다
            if not r.get("t") or None in r:
                continue
            g = lambda *ks: next((r[k] for k in ks if r.get(k) not in (None, "")), "0")  # noqa: E731
            rows.append({"t": r["t"], "usage": g("usage"), "rss": g("rss", "total_rss"), "shmem": g("shmem", "total_shmem"), "inactive_file": g("inactive_file", "total_inactive_file"), "server": server})
    return rows


def main():
    rows = load()
    t = [float(r["t"]) for r in rows]
    out = []
    for f in sorted(DONE.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:  # d9_data_kpr_memmap.json: 2026-11-01 01:47 쿼터 초과 사건으로 0바이트
            continue
        if "seconds" not in d:
            continue
        end, start = f.stat().st_mtime, f.stat().st_mtime - d["seconds"]
        sel = [r for r, x in zip(rows, t) if start <= x <= end]
        if len(sel) < 5 or sum(1 for r in sel) < 0.5 * d["seconds"]:
            continue
        usage = [int(r["usage"]) for r in sel]
        anon = [int(r["rss"]) + int(r["shmem"]) for r in sel]
        ws = [int(r["usage"]) - int(r["inactive_file"]) for r in sel]
        out.append({"step": f.stem, "server": sel[0]["server"], "seconds": round(d["seconds"]), "n_samples": len(sel), "usage_max_gib": round(max(usage) / G, 2), "anon_rss_shmem_max_gib": round(max(anon) / G, 2),
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
