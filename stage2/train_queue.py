"""Stage 2 학습 대기열(2026-11-09). srreid-core env, 표준 라이브러리만.
32회 = {bot, clipreid, kpr, kpr_solider} × {bicubic, swinir, pisa_sr_sem0.0, osdhuman} × {3, 4}, seed 42(각 학습기 고정).
- 우선순위: KPR·KPR+SOLIDER ×4 → KPR 계열 ×3 → CLIP-ReID ×4 → BoT ×4 → CLIP-ReID ×3 → BoT ×3 (긴 작업 먼저).
- 배치 규칙: GPU마다 실행 중 작업의 VRAM 예상치 합 + 새 작업 예상치 ≤ VRAM_LIMIT, 그리고 실제 사용량(nvidia-smi) + 새 작업 예상치 ≤ VRAM_LIMIT,
  GPU당 동시 작업 ≤ MAX_PER_GPU, 전체 CPU 예상치 ≤ CPU_LIMIT. 예상치(GiB·코어)는 200 iteration 시험(results/stage2/train_trial_summary.csv)에서 정한 값(--plan JSON).
- 같은 GPU에 새 작업은 직전 시작 뒤 STAGGER초 이후(적재 중 메모리 급증 회피).
- 완료(DONE)는 건너뛰고, 실패(rc≠0)는 다시 넣지 않고 기록만 한다(사용자 보고 대상). 재시작하면 DONE 없는 작업을 처음부터(학습기 자체 재개 지원) 다시 넣는다.
- 기록: results/stage2/train_queue_log.csv(시작·끝·GPU·rc·ckpt), 로그 logs/stage2/train/<이름>.log. 시간은 병렬 실행 중 측정."""
import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MF = Path("/workspace/miniforge3/envs")
PY = {"bot": MF / "srreid-core/bin/python", "clipreid": MF / "srreid-clipreid/bin/python", "kpr": MF / "srreid-kpr/bin/python", "kpr_solider": MF / "srreid-kpr/bin/python"}
CONDS = ("bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman")
OUT = ROOT / "runs" / "stage2" / "train"
LOG = ROOT / "logs" / "stage2" / "train"
QLOG = ROOT / "results" / "stage2" / "train_queue_log.csv"


def jobs():
    order = [("kpr_solider", 4), ("kpr", 4), ("kpr_solider", 3), ("kpr", 3), ("clipreid", 4), ("bot", 4), ("clipreid", 3), ("bot", 3)]
    return [(m, c, k) for m, k in order for c in CONDS]


def name(j):
    return f"{j[0]}__{j[1]}__x{j[2]}"


def cmd(j, ckpt):
    m, c, k = j
    o = OUT / name(j)
    if m == "bot":
        a = ["stage2/train_bot.py", "--cond", c, "--k", str(k)]
    elif m == "clipreid":
        a = ["stage2/train_clipreid.py", "--cond", c, "--k", str(k)]
    else:
        a = ["stage2/train_kpr.py", "--model", m, "--cond", c, "--k", str(k)]
    return [str(PY[m])] + a + ["--out", str(o), "--ckpt", str(ckpt)]


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(l.split(",")[0]): float(l.split(",")[1]) / 1024 for l in out.strip().splitlines()}


def qlog(row):
    new = not QLOG.exists()
    QLOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QLOG, "a", newline="") as f:
        w = csv.DictWriter(f, ["time", "event", "job", "gpu", "ckpt", "rc", "seconds", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "results" / "stage2" / "train_plan.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    est, cpu, ckpt = plan["vram_gib"], plan["cpu_cores"], plan["ckpt"]  # 키 "<model>_x<k>"
    VRAM_LIMIT, MAX_PER_GPU, CPU_LIMIT, STAGGER = plan["vram_limit_gib"], plan["max_per_gpu"], plan["cpu_limit"], plan["stagger_s"]
    pending = [j for j in jobs() if not (OUT / name(j) / "DONE").exists()]
    print(f"남은 학습 {len(pending)}/32", flush=True)
    if a.dry:
        for j in pending:
            print(name(j), est[f"{j[0]}_x{j[2]}"], cpu[f"{j[0]}_x{j[2]}"], ckpt[f"{j[0]}_x{j[2]}"])
        return
    running = {}  # name -> dict(proc, gpu, key, t0)
    last_start = {0: 0.0, 1: 0.0}
    failed = []
    LOG.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for n, r in list(running.items()):
            rc = r["proc"].poll()
            if rc is None:
                continue
            dt = time.time() - r["t0"]
            try:  # 종료 뒤 남은 DataLoader 작업자가 GPU 메모리를 잡는 일이 있어(KPR) 프로세스 그룹째 정리
                os.killpg(r["proc"].pid, 9)
            except ProcessLookupError:
                pass
            ok = rc == 0 and (OUT / n / "DONE").exists()
            qlog({"time": time.strftime("%F %T"), "event": "end" if ok else "fail", "job": n, "gpu": r["gpu"], "ckpt": r["ckpt"], "rc": rc, "seconds": round(dt)})
            print(f"[{time.strftime('%F %T')}] {'완료' if ok else '실패'} {n} gpu{r['gpu']} rc={rc} {dt / 3600:.2f}h (병렬 실행 중 측정)", flush=True)
            if not ok:
                failed.append(n)
            del running[n]
        used = gpu_used()
        for g in (0, 1):
            if not pending or time.time() - last_start[g] < STAGGER:
                continue
            on_g = [r for r in running.values() if r["gpu"] == g]
            if len(on_g) >= MAX_PER_GPU:
                continue
            committed = sum(est[r["key"]] for r in on_g)
            cpu_now = sum(cpu[r["key"]] for r in running.values())
            for j in pending:
                key = f"{j[0]}_x{j[2]}"
                if committed + est[key] <= VRAM_LIMIT and used.get(g, 99) + est[key] <= VRAM_LIMIT + 2 and cpu_now + cpu[key] <= CPU_LIMIT:
                    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES=str(g),
                               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", SRREID_CACHE="/dev/shm/srreid_cache",
                               SRREID_CONC_NOTE="병렬 실행 중 측정: Stage 2 학습 대기열(GPU당 여러 학습 동시)")
                    env.pop("ALLOW_TEST", None)
                    n = name(j)
                    f = open(LOG / f"{n}.log", "a")
                    p = subprocess.Popen(cmd(j, ckpt[key]), env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
                    running[n] = {"proc": p, "gpu": g, "key": key, "t0": time.time(), "ckpt": ckpt[key]}
                    pending.remove(j)
                    last_start[g] = time.time()
                    qlog({"time": time.strftime("%F %T"), "event": "start", "job": n, "gpu": g, "ckpt": ckpt[key], "rc": "", "seconds": "",
                          "note": f"committed {committed:.0f}+{est[key]}GiB, used {used.get(g, 0):.0f}GiB, cpu {cpu_now}+{cpu[key]}"})
                    print(f"[{time.strftime('%F %T')}] 시작 {n} gpu{g} ckpt={ckpt[key]} (GPU 작업 {len(on_g) + 1}개)", flush=True)
                    break
        time.sleep(20)
    print(f"[{time.strftime('%F %T')}] 대기열 끝. 실패 {failed if failed else '없음'}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
