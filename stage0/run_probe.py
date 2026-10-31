"""Stage 0 측정 매트릭스. 측정마다 별도 프로세스. 끝난 측정(ok, oom)은 JSON을 재사용하고 error만 다시 시도한다."""
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.util import EXIT_GUARD, gpu_memory_gb, write_csv  # noqa: E402

MF = Path("/workspace/miniforge3/envs")
ENV = {"bot": "srreid-core", "clipreid": "srreid-clipreid", "kpr": "srreid-kpr", "kpr_solider": "srreid-kpr"}
K = {"bot": 4, "clipreid": 4, "kpr": 4, "kpr_solider": 4}
RECIPE = 64
OUT = ROOT / "runs" / "stage0" / "probe"
LOG = ROOT / "logs" / "stage0" / "probe"
RESULTS = ROOT / "results"
IDLE_GB = 1.0
MAX_BATCH_SEARCH = 512
PROBE_FIELDS = ["backend", "scale", "input", "setting", "batch", "ckpt", "status", "fits", "exclusive_gpu", "max_memory_allocated_gb",
                "max_memory_reserved_gb", "nvidia_smi_peak_gb", "it_s", "img_s", "gpu_util_mean", "util_samples", "bottleneck_flag",
                "recipe_batch", "k", "amp", "optimizer", "epochs", "iters_per_epoch", "num_workers", "fixbase_epoch", "torch",
                "ckpt_blocks", "slowdown_pct", "start_used_mib", "seconds", "error", "note"]
EVAL_FIELDS = ["backend", "scale", "task", "device", "block", "batch", "status", "max_memory_allocated_gb", "max_memory_reserved_gb",
               "max_rss_gb", "seconds", "img_s", "max_abs_error_vs_full", "feature", "visrank", "exclusive_gpu", "error"]
rows, eval_rows = {}, {}


def log(msg):
    print(f"[probe {time.strftime('%F %T')}] {msg}", flush=True)


def kill_group(proc):
    """os._exit 뒤 남은 DataLoader 작업자가 GPU 메모리를 잡고 있으므로 프로세스 그룹째 정리한다."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()
    time.sleep(3)


def wait_idle():
    """다른 프로세스가 GPU를 쓰면 빌 때까지 기다린다(--require-exclusive). 다른 tmux 작업이 끝나기를 기다리는 시간 제한은 두지 않는다."""
    waited = False
    while True:
        used = gpu_memory_gb()[0]
        if used <= IDLE_GB:
            return used * 1024
        if not waited:
            log(f"GPU 사용 중 {used:.1f}GiB, 대기")
            waited = True
        time.sleep(30)


def parse_dmon(path, t0, t1):
    util, fb = [], []
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:
            continue
        try:
            ts = datetime.strptime(p[0] + p[1], "%Y%m%d%H:%M:%S").timestamp()
            sm, mem_fb = float(p[3]), float(p[9])
        except ValueError:
            continue
        fb.append(mem_fb)
        if t0 and t1 and t0 - 0.5 <= ts <= t1 + 0.5:
            util.append(sm)
    return {"nvidia_smi_peak_gb": max(fb) / 1024 if fb else None, "gpu_util_mean": sum(util) / len(util) if util else None,
            "util_samples": len(util)}


def run(key, backend, task_args, timeout=3600):
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{key}.json"
    if out.exists():
        res = json.loads(out.read_text())
        if res["status"] in ("ok", "oom"):
            log(f"재사용 {key}: {res['status']}")
            return res
        out.unlink()
    start = wait_idle()
    dmon_path = LOG / f"{key}.dmon"
    dmon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1", "-o", "DT"], stdout=open(dmon_path, "w"), stderr=subprocess.DEVNULL)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
           "TORCH_HOME": "/workspace/weights/torch", "XDG_CACHE_HOME": "/workspace/cache/xdg", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    env.pop("ALLOW_TEST", None)
    cmd = [str(MF / ENV[backend] / "bin" / "python"), str(ROOT / "stage0" / "worker.py"), "--backend", backend, "--out", str(out)] + task_args
    t0 = time.time()
    log(f"시작 {key}")
    with open(LOG / f"{key}.log", "w") as f:
        proc = subprocess.Popen(cmd, cwd=str(ROOT / "runs" / "stage0"), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = -1
        kill_group(proc)
    dmon.terminate()
    dmon.wait()
    if rc == EXIT_GUARD or (ROOT / "logs" / "GUARD_VIOLATION").exists():
        log("test 가드 위반, 전체 중단")
        sys.exit(EXIT_GUARD)
    if out.exists():
        res = json.loads(out.read_text())
    else:
        res = {"status": "error", "error": f"결과 없음 rc={rc} (시간 초과 또는 비정상 종료)"}
    res["wall_seconds"] = time.time() - t0
    res["start_used_mib"] = start
    res.update(parse_dmon(dmon_path, res.get("t_start"), res.get("t_end")))
    reserved = (res.get("max_memory_reserved") or 0) / 2**30
    peak = res.get("nvidia_smi_peak_gb") or 0
    res["exclusive_gpu"] = bool(start / 1024 <= IDLE_GB and peak - reserved <= 2.0)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"끝 {key}: {res['status']} alloc={res.get('max_memory_allocated', 0) / 2**30:.2f}GiB it/s={res.get('it_s')}")
    if res["status"] == "error":
        fail(key, res.get("error"))
    return res


def fail(key, reason):
    path = RESULTS / "stage0_failures.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "key", "reason"])
        w.writerow([time.strftime("%F %T"), key, str(reason)[:500]])


def gb(v):
    return round(v / 2**30, 3) if v is not None else None


def record(backend, scale, setting, batch, ckpt, res, note="", slowdown=None):
    util = res.get("gpu_util_mean")
    row = {"backend": backend, "scale": scale, "input": f"{128 * scale * 1}x{64 * scale}", "setting": setting, "batch": batch, "ckpt": ckpt,
           "status": res["status"], "fits": res["status"] == "ok", "exclusive_gpu": res.get("exclusive_gpu"),
           "max_memory_allocated_gb": gb(res.get("max_memory_allocated")), "max_memory_reserved_gb": gb(res.get("max_memory_reserved")),
           "nvidia_smi_peak_gb": res.get("nvidia_smi_peak_gb"), "it_s": res.get("it_s"), "img_s": res.get("img_s"), "gpu_util_mean": util,
           "util_samples": res.get("util_samples"), "bottleneck_flag": bool(res["status"] == "ok" and util is not None and util < 70),
           "seconds": res.get("wall_seconds"), "start_used_mib": res.get("start_used_mib"), "error": res.get("error", ""), "note": note,
           "slowdown_pct": slowdown}
    for k in ("recipe_batch", "k", "amp", "optimizer", "epochs", "iters_per_epoch", "num_workers", "fixbase_epoch", "torch", "ckpt_blocks"):
        row[k] = res.get(k)
    rows[(backend, scale, setting, batch, ckpt)] = row
    write_csv(RESULTS / "stage0_probe.csv", list(rows.values()), PROBE_FIELDS)
    return row


def train(backend, scale, batch, ckpt=False, warmup=10, measure=40, task="train"):
    tag = f"{backend}_x{scale}_{task}_b{batch}{'_ckpt' if ckpt else ''}_w{warmup}m{measure}"
    args = ["--task", task, "--scale", str(scale), "--batch", str(batch), "--warmup", str(warmup), "--measure", str(measure)]
    return run(tag, backend, args + (["--ckpt"] if ckpt else []))


def search_max(backend, scale, a_ok):
    """K의 배수만 후보. 짧은 시험(3+7스텝)으로 찾고 최종값은 A와 같은 10+40스텝으로 다시 잰다. 시험 결과도 CSV에 남긴다."""
    k = K[backend]

    def trial(batch):
        r = train(backend, scale, batch, warmup=3, measure=7)
        record(backend, scale, "B_trial", batch, False, r)
        if r["status"] == "error":
            raise RuntimeError(f"시험 실패: {r.get('error')}")
        return r["status"] == "ok"

    if a_ok:
        lo, hi = RECIPE, RECIPE * 2
        while hi <= MAX_BATCH_SEARCH and trial(hi):
            lo, hi = hi, hi * 2
        if hi > MAX_BATCH_SEARCH:
            hi = None
    else:
        lo, hi = None, RECIPE
    if not a_ok:
        cand = k * 2  # K장 배치는 ID가 하나뿐이라 triplet loss를 계산할 수 없다(KPR는 TypeError). 하한은 2K다.
        if not trial(cand):
            return None
        lo = cand
    while hi is not None and hi - lo > k:
        mid = (lo + hi) // 2 // k * k
        if trial(mid):
            lo = mid
        else:
            hi = mid
    best = lo
    while best >= k:
        r = train(backend, scale, best)
        note = "상한 탐색 끝값" if hi is None else ""
        record(backend, scale, "B", best, False, r, note)
        if r["status"] == "ok":
            return best
        if r["status"] == "error":
            return None
        best -= k
    return None


def probe_cell(backend, scale):
    a = train(backend, scale, RECIPE)
    record(backend, scale, "A", RECIPE, False, a)
    if backend == "clipreid":
        s1 = train(backend, scale, RECIPE, task="stage1")
        record(backend, scale, "A_stage1", RECIPE, False, s1, "프롬프트 학습. 스테이지 2와 별도로 외삽")
    a_ok = a["status"] == "ok"
    max_batch = None
    if a["status"] == "oom" or backend.startswith("kpr"):
        try:
            max_batch = search_max(backend, scale, a_ok)
        except RuntimeError as e:
            fail(f"{backend}_x{scale}_B", e)
    if a["status"] != "oom":
        return
    c = train(backend, scale, RECIPE, ckpt=True)
    record(backend, scale, "C", RECIPE, True, c, "레시피 배치 + gradient checkpointing")
    slowdown = None
    if max_batch:  # 같은 배치에서 on/off를 비교해 속도 저하율을 잰다.
        on = train(backend, scale, max_batch, ckpt=True)
        off = rows.get((backend, scale, "B", max_batch, False))
        if on["status"] == "ok" and off and off.get("img_s"):
            slowdown = 100 * (1 - on["img_s"] / off["img_s"])
        record(backend, scale, "C_at_max_batch", max_batch, True, on, "속도 저하율 비교용", slowdown)
        if c["status"] == "ok":
            record(backend, scale, "C", RECIPE, True, c, "레시피 배치 + gradient checkpointing", slowdown)
    if backend.startswith("kpr"):
        eq = run(f"{backend}_x{scale}_equiv", backend, ["--task", "equiv", "--scale", str(scale), "--batch", "8"])
        (RESULTS / f"stage0_ckpt_equiv_{backend}_x{scale}.json").write_text(json.dumps(eq, indent=1, ensure_ascii=False))
    else:
        fail(f"{backend}_x{scale}_equiv", "OOM이 났으나 이 모델의 등가 검사는 구현되지 않음")


def eval_record(key, row):
    eval_rows[key] = row
    write_csv(RESULTS / "stage0_eval_mem.csv", list(eval_rows.values()), EVAL_FIELDS)


def probe_extract(backend):
    shape = None
    for scale, batch in ((3, 128), (3, 256), (2, 128)):
        args = ["--task", "extract", "--scale", str(scale), "--batch", str(batch), "--warmup", "3", "--measure", "15"]
        r = run(f"{backend}_x{scale}_extract_b{batch}", backend, args)
        eval_record((backend, scale, "extract", batch), {
            "backend": backend, "scale": scale, "task": "extract", "device": "gpu", "block": "", "batch": batch, "status": r["status"],
            "max_memory_allocated_gb": gb(r.get("max_memory_allocated")), "max_memory_reserved_gb": gb(r.get("max_memory_reserved")),
            "max_rss_gb": (r.get("max_rss_kb") or 0) / 2**20, "img_s": r.get("img_s"), "seconds": r.get("wall_seconds"),
            "feature": json.dumps(r.get("feature")), "visrank": r.get("visrank"), "exclusive_gpu": r.get("exclusive_gpu"), "error": r.get("error", "")})
        shape = shape or r.get("feature")
    return shape


def probe_dist(backend, shape):
    if not shape:
        fail(f"{backend}_dist", "특징 형태를 얻지 못해 거리 계산을 재지 못함")
        return
    args = ["--task", "dist", "--dim", str(shape["dim"])]
    if "parts" in shape:
        args += ["--parts", str(shape["parts"]), "--vis-dtype", shape["vis_dtype"]]
    for device in ("gpu", "cpu"):
        ref = ROOT / "runs" / "stage0" / "dist" / f"{backend}_{device}_full.npy"
        for block in (0, 256, 512, 1024):
            r = run(f"{backend}_dist_{device}_b{block}", backend, args + ["--device", device, "--block", str(block), "--ref", str(ref)], timeout=7200)
            eval_record((backend, device, "dist", block), {
                "backend": backend, "scale": "", "task": "dist", "device": device, "block": block or "full", "batch": "", "status": r["status"],
                "max_memory_allocated_gb": gb(r.get("max_memory_allocated")), "max_memory_reserved_gb": gb(r.get("max_memory_reserved")),
                "max_rss_gb": (r.get("max_rss_kb") or 0) / 2**20, "seconds": r.get("seconds"), "max_abs_error_vs_full": r.get("max_abs_error_vs_full"),
                "feature": json.dumps(shape), "visrank": False if backend.startswith("kpr") else "", "exclusive_gpu": r.get("exclusive_gpu"), "error": r.get("error", "")})
        ref.unlink(missing_ok=True)


def main():
    (ROOT / "runs" / "stage0").mkdir(parents=True, exist_ok=True)
    backends_list = sys.argv[1:] or list(ENV)
    for backend in backends_list:
        for scale in (2, 3):
            try:
                probe_cell(backend, scale)
            except Exception as e:  # 한 칸의 실패가 나머지 측정을 막지 않게 한다.
                log(f"{backend} x{scale} 실패: {e}")
                fail(f"{backend}_x{scale}", e)
    for backend in backends_list:
        try:
            probe_dist(backend, probe_extract(backend))
        except Exception as e:
            log(f"{backend} 평가 측정 실패: {e}")
            fail(f"{backend}_eval", e)
    log("측정 끝")


if __name__ == "__main__":
    main()
