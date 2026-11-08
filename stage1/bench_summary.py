"""A100 동시 실행 측정 요약(stage1/run_a100_check.sh bench). 출력 results/stage1/a100_concurrency_bench.csv, a100_concurrency_bitcheck.csv.
처리량 = 두 GPU 전체 처리 장수 / (공통 시작 시각 → 마지막 프로세스 끝). VRAM·GPU 사용률은 nvidia-smi 2초 간격, CPU는 top 2초 간격(측정 구간 평균).
비트 확인: 동시 실행 출력(배열 SHA1)과 A100 단독 재현성 실행(runs/stage1_step0/repro, 같은 λ, 결정적+예열, 정방향)의 표본 앞 20장 비교."""
import csv
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "runs" / "stage1_step0" / "bench"
R = ROOT / "runs" / "stage1_step0" / "repro" / "_logs"
OUT = ROOT / "results" / "stage1"
rows, bits = [], []
solo = {}
for f in R.glob("pisa_sr_sem*__x4__detw__fwd.jsonl"):
    lam = f.name.split("__")[0].replace("pisa_sr_sem", "")
    solo[lam] = {json.loads(l)["name"]: json.loads(l)["sha1"] for l in open(f)}
for d in sorted(B.glob("n*")):
    N = int(d.name[1:])
    S = [json.loads(open(p).read()) for p in sorted(d.glob("bench_*__summary.json"))]
    if not S:
        continue
    t0 = min(s["t_infer_start_epoch"] for s in S)
    t1 = max(s["t_infer_end_epoch"] for s in S)
    n = sum(s["n_done"] for s in S)
    util, mem = {0: [], 1: []}, {0: [], 1: []}
    for line in open(d / "nvsmi.csv"):
        p = [x.strip() for x in line.split(",")]
        try:
            from datetime import datetime
            ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except Exception:
            continue
        if t0 <= ts <= t1:
            g = int(p[1]); util[g].append(float(p[2].split()[0])); mem[g].append(float(p[3].split()[0]))
    cpu = []
    for line in open(d / "cpu_top.txt"):
        m = re.search(r"([\d.]+)\s*id", line)
        if m:
            cpu.append(100 - float(m.group(1)))
    cpu = cpu[1:] if len(cpu) > 1 else cpu  # 첫 top 표본은 부팅 이후 평균
    rows.append({"procs_per_gpu": N, "procs_total": len(S), "images": n, "seconds": round(t1 - t0, 1), "img_per_s_total": round(n / (t1 - t0), 3),
                 "per_proc_s_per_img_median": round(sorted(s["infer_s_per_img_median"] for s in S)[len(S) // 2], 3),
                 "gpu0_util_mean": round(sum(util[0]) / max(1, len(util[0])), 1), "gpu1_util_mean": round(sum(util[1]) / max(1, len(util[1])), 1),
                 "gpu0_mem_max_mib": max(mem[0] or [0]), "gpu1_mem_max_mib": max(mem[1] or [0]),
                 "peak_reserved_gib_per_proc_max": round(max(s["peak_reserved_gib"] for s in S), 2),
                 "host_cpu_busy_pct_mean_top": round(sum(cpu) / max(1, len(cpu)), 1), "cores_busy_est": round(sum(cpu) / max(1, len(cpu)) * 255 / 100, 1), "host_cpus": 255, "cgroup_cpu_limit": 54.4,
                 "omp_per_proc": S[0]["omp"], "note": "병렬 실행 중 측정, pisa_sr ×4 b1 결정적+예열, 50장/프로세스, WebP 저장 포함(/root 로컬)"})
    for p in sorted(d.glob("bench_*.jsonl")):
        sm = json.loads(open(str(p).replace(".jsonl", "__summary.json")).read())
        lam = sm["method"].replace("pisa_sr_sem", "")
        got = {json.loads(l)["name"]: json.loads(l)["sha1"] for l in open(p)}
        common = [k for k in solo.get(lam, {}) if k in got]
        bits.append({"procs_per_gpu": N, "proc": p.stem, "gpu": sm.get("cuda_visible_devices"), "lambda": lam, "n_compared": len(common),
                     "n_bit_identical": sum(got[k] == solo[lam][k] for k in common)})
for name, data in (("a100_concurrency_bench.csv", rows), ("a100_concurrency_bitcheck.csv", bits)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, list(data[0]))
        w.writeheader()
        w.writerows(data)
for r in rows:
    print(r)
print("비트 확인:", sum(b["n_bit_identical"] for b in bits), "/", sum(b["n_compared"] for b in bits))
