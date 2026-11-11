"""Stage 2 평가를 학습과 겹쳐 돌리는 반복기(2026-11-10). srreid-core env. ALLOW_TEST=1(사용자 잠금 해제 중)에서만.
- 학습이 끝난 모델의 unit(stage2/evaluate.py --list-units의 ready)을 한 번에 하나씩 처리한다(학습 속도 보호).
- GPU 선택: 실제 여유 VRAM ≥ 모델별 필요량 + 여유분인 GPU 중 여유가 큰 쪽. 없으면 기다린다.
- CPU: 최근 30초 cgroup CPU 사용량이 CPU_BUSY 코어를 넘으면 기다린다.
- 학습 대기열(logs/bg/s2train.exit)이 끝났고 ready·waiting unit이 없으면 끝낸다. 매 unit 뒤 evaluate.py가 eval_cells.csv·judgment.csv를 다시 만든다.
- 시간은 병렬 실행 중 측정(학습과 동시)."""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/workspace/miniforge3/envs/srreid-core/bin/python"
NEED = {"bot": 6, "clipreid": 8, "kpr": 16, "kpr_solider": 16}  # GiB, 추출 배치 기준 어림 + 여유
MARGIN, CPU_BUSY = 4, 1000  # 2026-11-11: CPU 조건 대신 nice 19로 돌려 학습이 쓰고 남은 CPU만 쓴다
LOG = ROOT / "logs" / "stage2" / "eval_loop.log"


def log(*a):
    s = f"[{time.strftime('%F %T')}] " + " ".join(map(str, a))
    print(s, flush=True)


def units():
    out = subprocess.run([PY, "stage2/evaluate.py", "--list-units"], capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    rows = [l.split("\t") for l in out.stdout.strip().splitlines() if "\t" in l]
    return [r[0] for r in rows if r[1] == "ready"], [r[0] for r in rows if r[1] in ("waiting_model", "running")]


def free_gib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(l.split(",")[0]): (float(l.split(",")[2]) - float(l.split(",")[1])) / 1024 for l in o.strip().splitlines()}


def recent_train_start(g, within=600):
    """학습 대기열이 이 GPU에 최근(within초) 새 학습을 넣었으면 True. 적재 중(약 3분)에는 VRAM이 아직 덜 잡혀 여유가 커 보인다."""
    p = ROOT / "results" / "stage2" / "train_queue_log.csv"
    if not p.exists():
        return False
    import csv
    for r in csv.DictReader(open(p)):
        if r["event"] == "start" and r["gpu"] == str(g):
            last = time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))
    try:
        return time.time() - last < within
    except NameError:
        return False


def cpu_cores(window=30):
    def u():
        for l in open("/sys/fs/cgroup/cpu.stat"):
            if l.startswith("usage_usec"):
                return int(l.split()[1])
    a, t = u(), time.time()
    time.sleep(window)
    return (u() - a) / 1e6 / (time.time() - t)


def main():
    assert os.environ.get("ALLOW_TEST") == "1", "사용자 잠금 해제 중 ALLOW_TEST=1로만 실행"
    done_n = 0
    oom_retry = {}
    while True:
        ready, waiting = units()
        train_over = (ROOT / "logs" / "bg" / "s2train.exit").exists()
        if not ready:
            if train_over and not waiting:
                break
            time.sleep(120)
            continue
        c = cpu_cores()
        fr = free_gib()
        u, ok = ready[0], {}
        for cand in ready:  # 지금 VRAM에 들어가는 첫 unit(작은 모델 unit이 KPR unit 뒤에서 막히지 않게)
            okc = {g: f for g, f in fr.items() if f >= NEED[cand.split("__")[0]] + MARGIN and not recent_train_start(g)}
            if okc:
                u, ok = cand, okc
                break
        if not ok or c > CPU_BUSY:
            log(f"대기 {u}: 여유 VRAM {({g: round(f) for g, f in fr.items()})}GiB, CPU {c:.1f}코어")
            time.sleep(90)
            continue
        g = max(ok, key=ok.get)
        log(f"시작 {u} gpu{g} (여유 {ok[g]:.0f}GiB, CPU {c:.1f}코어)")
        t = time.time()
        env = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", SRREID_CACHE="/dev/shm/srreid_cache",
                   SRREID_CONC_NOTE="병렬 실행 중 측정: Stage 2 학습과 동시에 평가")
        rc = subprocess.call(["nice", "-n", "19", PY, "stage2/evaluate.py", "--unit", u, "--gpu", str(g), "--cache-workers", "4"], cwd=ROOT, env=env,
                             stdout=open(ROOT / "logs" / "stage2" / f"eval_{u}.log", "w"), stderr=subprocess.STDOUT)
        log(f"종료 {u} rc={rc} {time.time() - t:.0f}s")
        if rc:
            txt = "".join(open(p, errors="ignore").read() for p in (ROOT / "data" / "locked" / "stage2_eval").glob(f"{u.split('__')[0]}__{u.split('__')[1]}/{'__'.join(u.split('__')[2:])}/logs/*.log"))
            if "OutOfMemoryError" in txt and oom_retry.get(u, 0) < 3:  # 학습과 VRAM 경합. 학습은 그대로 두고 평가만 나중에 다시
                oom_retry[u] = oom_retry.get(u, 0) + 1
                log(f"OOM {u} — {oom_retry[u]}회째, 10분 뒤 다시")
                time.sleep(600)
                continue
            log(f"실패 {u} — 반복기 중단(보고 대상)")
            raise SystemExit(1)
        done_n += 1
    subprocess.call([PY, "stage2/evaluate.py", "--aggregate"], cwd=ROOT, env=dict(os.environ, PYTHONNOUSERSITE="1"))
    log(f"평가 끝: {done_n} unit")


if __name__ == "__main__":
    main()
