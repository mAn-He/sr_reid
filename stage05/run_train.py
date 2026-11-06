"""Part B3(학습 RAM), D(프로파일, 처리량, 등가성), H(메모리) 실행. 재실행하면 끝난 단계를 건너뛴다. train만 읽으므로 ALLOW_TEST 없이 돈다."""
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage05 import monitor  # noqa: E402

PY = "/workspace/miniforge3/envs/srreid-core/bin/python"
ENVPY = {"bot": PY, "clipreid": "/workspace/miniforge3/envs/srreid-clipreid/bin/python", "kpr": "/workspace/miniforge3/envs/srreid-kpr/bin/python",
         "kpr_solider": "/workspace/miniforge3/envs/srreid-kpr/bin/python"}
MODELS = ("bot", "clipreid", "kpr", "kpr_solider")
DONE = ROOT / "runs" / "stage05" / "done"
LOG = ROOT / "logs" / "stage05"
RES = ROOT / "results"


def write_done(path, obj):
    """완료 표시를 원자적으로 쓴다(임시 파일 → rename). 2026-11-01 쿼터 사건 때 0바이트 완료 표시가 남은 적이 있다."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def quarantine_bad_done():
    """읽을 수 없는 완료 표시(0바이트·깨진 JSON)를 logs/stage05/script_bug_rows/corrupt_done/로 옮겨 그 단계가 다시 돌게 한다. 지우지 않는다."""
    bad = []
    for f in sorted(DONE.glob("*.json")):
        try:
            json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            q = LOG / "script_bug_rows" / "corrupt_done"
            q.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(q / f.name))
            bad.append(f.name)
    for f in DONE.glob("*.json.tmp"):
        f.unlink()
    if bad:
        print(f"깨진 완료 표시 격리: {bad}", flush=True)


def env(**kw):
    # 스레드 제한(확정 결정, d13): 새 서버 cgroup 13.6코어에서 torch 기본 64스레드가 스로틀링된다. kw로 덮어쓸 수 있다.
    e = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", WANDB_MODE="disabled")
    e.setdefault("OMP_NUM_THREADS", "4")
    e.setdefault("MKL_NUM_THREADS", "4")
    e.update(kw)
    e.pop("ALLOW_TEST", None)
    return e


class GpuUtil:
    def __init__(self):
        self.samples, self.stop_flag = [], threading.Event()

    def run(self):
        while not self.stop_flag.wait(1):
            try:
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                self.samples.append((time.time(), float(o.strip().splitlines()[0])))
            except (subprocess.SubprocessError, OSError, ValueError):
                pass

    def mean(self, t0, t1):
        v = [u for t, u in self.samples if t0 <= t <= t1]
        return sum(v) / len(v) if v else None


def train(name, backend, scale, batch, task="run", iters=200, warmup=0, workers=0, impl="baseline", flags=(), ckpt=0, deterministic=0, limit=48.0, extra_env=None):
    done = DONE / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    out = LOG / f"{name}.out.json"
    out.unlink(missing_ok=True)
    cmd = [ENVPY[backend], "stage05/train_run.py", "--backend", backend, "--scale", scale, "--batch", batch, "--task", task, "--iters", iters, "--warmup", warmup,
           "--workers", workers, "--impl", impl, "--ckpt", ckpt, "--deterministic", deterministic, "--out", out]
    if flags:
        cmd += ["--flags"] + list(flags)
    g = GpuUtil()
    th = threading.Thread(target=g.run, daemon=True)
    th.start()
    res = monitor.run(cmd, LOG / f"{name}.log", env=env(**(extra_env or {})), limit_gib=limit, cwd=ROOT)
    g.stop_flag.set()
    th.join(timeout=3)
    res["out"] = json.loads(out.read_text()) if out.exists() else {"status": "no_output"}
    o = res["out"]
    if "t_start" in o:
        res["gpu_util_mean"] = g.mean(o["t_start"], o["t_end"])
    DONE.mkdir(parents=True, exist_ok=True)
    write_done(done, res)
    print(f"[{name}] {o.get('status')} it/s={o.get('it_s')} util={res.get('gpu_util_mean')} rss={res['max_rss_sum_gib']:.2f} pss={res['max_pss_sum_gib']:.2f} {res['seconds']:.0f}s", flush=True)
    return res


def phase_profile():
    batch = {"bot": 64, "clipreid": 64, "kpr": 32, "kpr_solider": 32}
    for m in MODELS:
        train(f"d1_profile_{m}", m, 3, batch[m], task="profile", iters=100, warmup=10)
        train(f"d1_profile_{m}_memmap", m, 3, batch[m], task="profile", iters=100, warmup=10, flags=["memmap"])
    train("d1b_sample_kpr", "kpr", 3, 32, task="sample", iters=200)
    train("d1b_sample_kpr_memmap", "kpr", 3, 32, task="sample", iters=200, flags=["memmap"])


def status(r):
    return r.get("out", {}).get("status", "no_output") if r["rc"] != -9 else "over_ram_limit"


def write_row(path, fields, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


VRAM_FIELDS = ["model", "input", "batch", "ckpt", "fits", "max_allocated_gib", "max_reserved_gib", "nvidia_peak_mib", "it_s", "status", "kind", "note"]
TP_FIELDS = ["model", "input", "batch", "ckpt", "workers", "impl", "flags", "it_s", "img_s", "gpu_util_mean", "max_rss_sum_gib", "max_pss_sum_gib", "nvidia_peak_mib", "exclusive_gpu", "seconds", "status", "note"]
RAM_FIELDS = ["model", "phase", "mode", "input", "batch", "workers", "n_query", "n_gallery", "max_rss_sum_gib", "max_pss_sum_gib", "time_v_max_rss_gib", "seconds", "rc", "killed_over_limit", "status", "note"]
EQ_FIELDS = ["model", "input", "batch", "workers", "comparison", "flags", "iters", "max_abs_loss_diff", "mean_abs_loss_diff", "noise_max", "noise_mean", "first_diff_iter", "passed", "note"]
INPUT = {3: "384x192", 4: "512x256"}
GIB = 2**30


def fits(m, k):
    """H 결과에서 학습 RAM·처리량 측정에 쓸 (배치, ckpt)를 고른다. 레시피 배치 우선."""
    for b, c in ((64, 0), (64, 1), (48, 1), (32, 1), (24, 1), (16, 1), (12, 1), (8, 1)):
        d = DONE / f"h1_{m}_x{k}_b{b}_ckpt{c}.json"
        if d.exists() and status(json.loads(d.read_text())) == "ok":
            return b, c
    raise RuntimeError(f"{m} x{k}: 들어가는 배치 없음")


H4_FIELDS = ["model", "input", "batch", "ckpt", "alloc_conf", "status", "peak_allocated_gib", "reserved_peak_gib", "nvidia_peak_mib", "peak_where", "phases", "largest_blocks_at_peak", "top_at_peak", "oom_message", "oom_stack", "inactive_at_oom_gib", "largest_free_at_oom_gib"]


def phase_h():
    """H.1: 레시피 배치 64, ckpt 끔/켬. 안 들어가면 ckpt 켠 채 배치를 낮춰 외삽 근거를 만든다."""
    def one(m, k, b, c, kind):
        name = f"h1_{m}_x{k}_b{b}_ckpt{c}"
        new = not (DONE / f"{name}.json").exists()
        r = train(name, m, k, b, iters=30, warmup=5, flags=["memmap"], ckpt=c)
        o = r["out"]
        st = status(r)
        if new:
            write_row(RES / "stage05_vram.csv", VRAM_FIELDS, {"model": m, "input": INPUT[k], "batch": b, "ckpt": c, "fits": st == "ok", "max_allocated_gib": round(o.get("max_allocated", 0) / GIB, 2),
                      "max_reserved_gib": round(o.get("max_reserved", 0) / GIB, 2), "nvidia_peak_mib": r.get("max_gpu_mib", ""), "it_s": o.get("it_s", ""), "status": st, "kind": kind})
        return st
    for m in MODELS:
        for k in (3, 4):
            s0 = one(m, k, 64, 0, "recipe")
            s1 = one(m, k, 64, 1, "recipe")
            if s0 != "ok" and s1 != "ok":
                for b in (48, 32, 24, 16, 12, 8):
                    if one(m, k, b, 1, "fallback") == "ok":
                        break


def probe(name, task, m, k, b, ckpt=1, extra=None):
    done = DONE / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    out = LOG / f"{name}.out.json"
    cmd = [ENVPY[m], "stage05/mem_probe.py", "--task", task, "--backend", m, "--scale", k, "--batch", b, "--ckpt", ckpt, "--out", out]
    r = monitor.run(cmd, LOG / f"{name}.log", env=env(**(extra or {})), limit_gib=48.0, cwd=ROOT)
    r["out"] = json.loads(out.read_text()) if out.exists() else {"status": "no_output"}
    write_done(done, r)
    print(f"[{name}] {r['out'].get('status')} rc={r['rc']} {r['seconds']:.0f}s", flush=True)
    return r


def phase_h2():
    """H.2: KPR 비선형 원인. H.3: 결정론 ckpt 등가성."""
    fields = ["model", "input", "batch", "ckpt", "status", "oom_phase", "phase", "before_gib", "peak_gib", "after_gib", "reserved_gib", "peak_allocated_gib", "top_after_forward", "top_at_oom"]
    for k, bs in ((3, (32, 48, 64)), (4, (16, 24, 32))):
        for b in bs:
            for c in (1, 0):
                r = probe(f"h2_kpr_x{k}_b{b}_ckpt{c}", "phases", "kpr", k, b, c)["out"]
                base = {"model": "kpr", "input": INPUT[k], "batch": b, "ckpt": c, "status": r.get("status"), "oom_phase": r.get("oom_phase", ""), "peak_allocated_gib": round(r.get("peak_allocated_gib", 0), 3),
                        "top_after_forward": json.dumps(r.get("snapshot_after_forward", {}).get("top", [])[:6], ensure_ascii=False), "top_at_oom": json.dumps(r.get("snapshot_at_oom", {}).get("top", [])[:6], ensure_ascii=False)}
                for ph, v in r.get("phases", {}).items():
                    write_row(RES / "stage05_kpr_memory.csv", fields, {**base, "phase": ph, **{x: round(v[x], 3) for x in ("before_gib", "peak_gib", "after_gib", "reserved_gib")}})
                if not r.get("phases"):
                    write_row(RES / "stage05_kpr_memory.csv", fields, base)
    for m, k, b in (("kpr", 3, 16), ("kpr_solider", 3, 16)):
        probe(f"h3_{m}_x{k}_b{b}_equiv", "equiv", m, k, b, 0, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})


def h4_one(m, k, b, c, extra=None, tag=""):
    name = f"h4_{m}_x{k}_b{b}_ckpt{c}{tag}"
    new = not (DONE / f"{name}.json").exists()
    r = probe(name, "phases", m, k, b, c, extra)
    o = r["out"]
    if new:
        top = o.get("snapshot_at_peak") or {}
        oom = o.get("snapshot_at_oom") or {}
        write_row(RES / "stage05_kpr_memory.csv", H4_FIELDS, {"model": m, "input": INPUT[k], "batch": b, "ckpt": c, "alloc_conf": (extra or {}).get("PYTORCH_CUDA_ALLOC_CONF", ""),
                  "status": o.get("status"), "peak_allocated_gib": round(o["peak_allocated_gib"], 3) if o.get("status") == "ok" else "", "reserved_peak_gib": round(max([v.get("reserved_gib", 0) for v in o.get("phases", {}).values()] or [0]), 3),
                  "nvidia_peak_mib": r.get("max_gpu_mib", ""), "peak_where": o.get("peak_where", ""), "phases": json.dumps({ph: round(v["peak_gib"], 3) for ph, v in o.get("phases", {}).items()}),
                  "largest_blocks_at_peak": json.dumps(top.get("largest_blocks", [])[:5], ensure_ascii=False), "top_at_peak": json.dumps(top.get("top", [])[:6], ensure_ascii=False),
                  "oom_message": o.get("oom_message", ""), "oom_stack": json.dumps(o.get("oom_stack", []), ensure_ascii=False), "inactive_at_oom_gib": round(oom.get("inactive_gib", 0), 3), "largest_free_at_oom_gib": round(oom.get("largest_free_gib", 0), 3)})
    return o.get("status"), r


def phase_h4():
    """중간 지시 3: KPR 메모리 격자. 배치를 올리다 OOM이면 멈추고, 들어간 점이 4개 미만이면 낮은 배치를 더한다. KPR+SOLIDER는 같은 식인지 4점씩 대조한다."""
    grid, low = (16, 24, 32, 40, 48), (12, 20, 8, 28, 4)

    one = h4_one
    for m, pick in (("kpr", None), ("kpr_solider", 4)):
        for k in (3, 4):
            for c in (1, 0):
                ok, first_oom = [], 10**6
                for b in grid:
                    if m == "kpr_solider" and len(ok) == pick:
                        break
                    if one(m, k, b, c)[0] != "ok":
                        first_oom = b
                        break
                    ok.append(b)
                for b in low:
                    if len(ok) >= 4:
                        break
                    if b < first_oom and b not in ok and one(m, k, b, c)[0] == "ok":
                        ok.append(b)
    one("kpr", 3, 64, 1)
    one("kpr", 3, 64, 1, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, "_split512")


def phase_h6():
    """중간 지시 3 보강. ×4 checkpointing 끔은 측정점이 3개라 14·18을 더한다(배치 4는 ID 1명이라 triplet이 None을 돌려 측정 불가). 단편화를 없앤 점은 적합에 넣지 않고 검증용으로 쓴다."""
    split = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}
    for m in ("kpr", "kpr_solider"):
        for b in (14, 18):
            h4_one(m, 4, b, 0)
    h4_one("kpr", 3, 40, 0, split, "_split512")
    h4_one("kpr", 4, 20, 0, split, "_split512")
    h4_one("kpr", 4, 32, 1, split, "_split512")
    for m in ("kpr", "kpr_solider"):
        train(f"h6_{m}_x3_b64_ckpt1_split512", m, 3, 64, iters=100, warmup=10, workers=4, ckpt=1, flags=ALL_FLAGS, extra_env=split)


def ckpt_run(name, task, m, k, b, mode, ckpt=0, extra=()):
    done = DONE / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    out = LOG / f"{name}.out.json"
    cmd = [ENVPY[m], "stage05/ckpt_equiv.py", "--task", task, "--backend", m, "--scale", k, "--batch", b, "--mode", mode, "--ckpt", ckpt, "--out", out, *extra]
    r = monitor.run(cmd, LOG / f"{name}.log", env=env(CUBLAS_WORKSPACE_CONFIG=":4096:8"), limit_gib=48.0, cwd=ROOT)
    r["out"] = json.loads(out.read_text()) if out.exists() else {"status": "crashed"}
    write_done(done, r)
    print(f"[{name}] {r['out'].get('status')} rc={r['rc']} {r['seconds']:.0f}s", flush=True)
    return r


def phase_ckpt():
    """중간 지시 4: 1스텝(stage0/warn/cpu_up), 200 iteration 곡선(warn, cpu_up의 off·off2·on·on2). KPR 배치 8은 Stage 0과 같은 조건이다."""
    steps = [("kpr", 3, 8), ("kpr", 3, 32), ("kpr_solider", 3, 8), ("kpr_solider", 3, 32), ("clipreid", 3, 64), ("clipreid", 4, 32)]
    for m, k, b in steps:
        for mode in ("stage0", "warn", "cpu_up"):
            ckpt_run(f"h5_step_{m}_x{k}_b{b}_{mode}", "step", m, k, b, mode)
    for m, k, b in (("kpr", 3, 32), ("kpr_solider", 3, 32), ("clipreid", 3, 64)):
        for mode in ("warn", "cpu_up"):
            for tag, c in (("off", 0), ("off2", 0), ("on", 1), ("on2", 1)):
                ckpt_run(f"h5_curve_{m}_x{k}_b{b}_{mode}_{tag}", "curve", m, k, b, mode, c)


def phase_ckpt4():
    """중간 지시 4 보강: ×4도 같은 판정을 받게 한다. cpu_up 모드, 곡선은 off·off2·on. h7 1스텝은 켬/끔의 allocated 피크를 함께 남겨 체크포인팅이 실제로 동작했는지 확인한다."""
    for m, k, b in (("kpr", 3, 32), ("kpr_solider", 3, 32), ("clipreid", 3, 64), ("kpr", 4, 16), ("kpr_solider", 4, 16), ("clipreid", 4, 32)):
        ckpt_run(f"h7_step_{m}_x{k}_b{b}_cpu_up", "step", m, k, b, "cpu_up")
    for m, k, b in (("clipreid", 4, 32), ("kpr", 4, 16), ("kpr_solider", 4, 16)):
        for tag, c in (("off", 0), ("off2", 0), ("on", 1)):
            ckpt_run(f"h7_curve_{m}_x{k}_b{b}_cpu_up_{tag}", "curve", m, k, b, "cpu_up", c)


def eq_cases(m):
    """D.5 비교 대상. KPR의 all은 u8 마스크 포함(채택 후보), memmap은 f32 마스크로 u8 효과를 분리한다."""
    if m.startswith("kpr"):
        return {"memmap": ["memmap", "mask_f32"], "all_f32": ALL_FLAGS + ["mask_f32"], "all_u8": ALL_FLAGS + ["mask_u8"]}
    return {"memmap": ["memmap"], "all": ALL_FLAGS}


def phase_dataeq():
    """D.5 보강: 학습 없이 로더 출력 비교. 기준선 반복이 같아야 비교가 성립한다. run 1회 = 프로세스 1개."""
    dump_iters = "0,1,2,3,12,100"
    work = Path(os.environ.get("SRREID_CACHE", "/root/srreid_cache")) / "eqdump"  # 로컬 NVMe 임시. 없으면 만든다
    for m in MODELS:
        b, _ = fits(m, 3)
        d = work / m
        d.mkdir(parents=True, exist_ok=True)

        def one(tag, flags):
            name = f"d9_data_{m}_{tag}"
            if not (DONE / f"{name}.json").exists():
                cmd = [ENVPY[m], "stage05/data_equiv.py", "run", "--backend", m, "--scale", 3, "--batch", b, "--dump", dump_iters, "--out", d / f"{tag}.json"]
                if flags:
                    cmd += ["--flags", *flags]
                r = monitor.run(cmd, LOG / f"{name}.log", env=env(), limit_gib=48.0, cwd=ROOT)
                write_done(DONE / f"{name}.json", r)
                print(f"[{name}] rc={r['rc']} rss={r['max_rss_sum_gib']:.2f} pss={r['max_pss_sum_gib']:.2f} {r['seconds']:.0f}s", flush=True)

        def cmp_(tag, label):
            subprocess.run([PY, "stage05/data_equiv.py", "compare", "--a", d / "base.json", "--b", d / f"{tag}.json", "--model", m, "--input", INPUT[3], "--comparison", label, "--batch", str(b)],
                           env=env(), cwd=ROOT, check=False)
        one("base", [])
        one("base2", [])
        cmp_("base2", "baseline vs baseline")
        for tag, flags in eq_cases(m).items():
            one(tag, flags)
            cmp_(tag, f"baseline vs {tag}")
        keep = LOG / "dataeq" / m
        keep.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.json"):
            shutil.copy(f, keep / f.name)
        shutil.rmtree(d, ignore_errors=True)


def phase_deteq():
    """D.5 보강: 결정론 모드에서 200 iteration. KPR은 upsample backward가 GPU에서 비결정이라 cpu_up. 결과는 별도 CSV에 쓰고 D.5 원 판정은 그대로 둔다."""
    path = RES / "stage05_train_equiv_det.csv"
    for m in MODELS:
        b, c = fits(m, 3)
        mode = "cpu_up" if m.startswith("kpr") else "warn"

        def go(tag, flags):
            name = f"d10_det_{m}_{tag}"
            r = ckpt_run(name, "curve", m, 3, b, mode, c, ("--occluder-samples", 0, "--workers", 4, "--flags", *(flags or ["none"])))
            return r["out"].get("losses", [])
        base, base2 = go("base", []), go("base2", [])
        noise = compare(base, base2)
        for tag, flags in {k: v for k, v in eq_cases(m).items() if k != "all_f32"}.items():
            cmp_ = compare(base, go(tag, flags))
            write_row(path, EQ_FIELDS, {"model": m, "input": INPUT[3], "batch": b, "workers": 4, "comparison": f"baseline vs {tag} ({mode})", "flags": "+".join(flags), "iters": 200,
                                        "max_abs_loss_diff": cmp_ and cmp_["max"], "mean_abs_loss_diff": cmp_ and cmp_["mean"], "noise_max": noise and noise["max"], "noise_mean": noise and noise["mean"],
                                        "first_diff_iter": cmp_ and cmp_["first"], "passed": bool(cmp_ and noise and cmp_["max"] <= max(noise["max"], 1e-9)), "note": f"기준선 반복 first_diff={noise and noise['first']}"})


def phase_wnorm():
    """D.5 보강: memmap에서 정규화·erasing을 기준선처럼 워커에 둔 구성. 입력 비교(dataeq의 기준선 json), 결정론 곡선(deteq의 기준선). KPR은 memmap이 원래 CPU 경로라 phase_kprcopy에서 따로 본다."""
    final = {"bot": ["memmap", "worker_norm", "no_timers", "prefetch4"], "clipreid": ["memmap", "worker_norm", "no_timers", "prefetch4", "pin"]}
    path = RES / "stage05_train_equiv_det.csv"
    for m in ("bot", "clipreid"):
        b, c = fits(m, 3)
        d = LOG / "dataeq" / m
        name = f"d9_data_{m}_wnorm"
        if not (DONE / f"{name}.json").exists():
            r = monitor.run([ENVPY[m], "stage05/data_equiv.py", "run", "--backend", m, "--scale", 3, "--batch", b, "--out", d / "wnorm.json", "--flags", "memmap", "worker_norm"], LOG / f"{name}.log", env=env(), limit_gib=48.0, cwd=ROOT)
            write_done(DONE / f"{name}.json", r)
            print(f"[{name}] rc={r['rc']} {r['seconds']:.0f}s", flush=True)
            subprocess.run([PY, "stage05/data_equiv.py", "compare", "--a", d / "base.json", "--b", d / "wnorm.json", "--model", m, "--input", INPUT[3], "--comparison", "baseline vs memmap+worker_norm", "--batch", str(b)], env=env(), cwd=ROOT, check=False)
        base = json.loads((DONE / f"d10_det_{m}_base.json").read_text())["out"]["losses"]
        base2 = json.loads((DONE / f"d10_det_{m}_base2.json").read_text())["out"]["losses"]
        noise = compare(base, base2)
        for tag, flags in (("wnorm", ["memmap", "worker_norm"]), ("final", final[m])):
            new = not (DONE / f"d10_det_{m}_{tag}.json").exists()
            r = ckpt_run(f"d10_det_{m}_{tag}", "curve", m, 3, b, "warn", c, ("--occluder-samples", 0, "--workers", 4, "--flags", *flags))
            cmp_ = compare(base, r["out"].get("losses", []))
            if new:
                write_row(path, EQ_FIELDS, {"model": m, "input": INPUT[3], "batch": b, "workers": 4, "comparison": f"baseline vs {tag} (warn)", "flags": "+".join(flags), "iters": 200, "max_abs_loss_diff": cmp_ and cmp_["max"], "mean_abs_loss_diff": cmp_ and cmp_["mean"],
                                            "noise_max": noise and noise["max"], "noise_mean": noise and noise["mean"], "first_diff_iter": cmp_ and cmp_["first"], "passed": bool(cmp_ and noise and cmp_["max"] <= max(noise["max"], 1e-9)), "note": f"기준선 반복 first_diff={noise and noise['first']}"})


def phase_kprcopy():
    """지시 3c. BIPO의 제자리 쓰기가 lru_cache에 남지 않게 read_image만 복사한 공식 로더(copy_read)와 memmap+float32를 결정론 모드 200 iteration으로 비교한다."""
    path = RES / "stage05_train_equiv_det.csv"
    for m in ("kpr", "kpr_solider"):
        b, c = fits(m, 3)

        def go(tag, flags):
            return ckpt_run(f"d10_det_{m}_{tag}", "curve", m, 3, b, "cpu_up", c, ("--occluder-samples", 0, "--workers", 4, "--flags", *flags))["out"].get("losses", [])
        base, mem = go("base", ["none"]), go("memmap", ["memmap", "mask_f32"])
        cr, cr2 = go("copyread", ["copy_read"]), go("copyread2", ["copy_read"])
        noise = compare(cr, cr2)
        for label, x, y in (("copy_read vs memmap+f32 (cpu_up)", cr, mem), ("baseline vs copy_read (cpu_up)", base, cr)):
            d = compare(x, y)
            write_row(path, EQ_FIELDS, {"model": m, "input": INPUT[3], "batch": b, "workers": 4, "comparison": label, "flags": "copy_read", "iters": 200, "max_abs_loss_diff": d and d["max"], "mean_abs_loss_diff": d and d["mean"],
                                        "noise_max": noise and noise["max"], "noise_mean": noise and noise["mean"], "first_diff_iter": d and d["first"], "passed": bool(d and noise and d["max"] <= max(noise["max"], 1e-9)), "note": f"copy_read 반복 first_diff={noise and noise['first']}"})


def phase_h8():
    """지시 3d 보강. ×4 checkpointing 켬의 외삽 폭이 48GB/80GB 판정을 가른다. 단편화를 없앤 설정으로 큰 배치를 더 잰다. allocated 피크는 할당기 설정과 무관하다(h6 검증)."""
    split = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}
    for m, k, c, batches in (("kpr", 4, 1, (28, 36, 40)), ("kpr_solider", 4, 1, (32, 40)), ("kpr", 3, 0, (36, 44)), ("kpr", 4, 0, (22, 24)), ("kpr_solider", 4, 0, (22,))):
        for b in batches:
            h4_one(m, k, b, c, split, "_split512")


def phase_tpfinal():
    """지시 3e. 최종 구성 처리량. 이 단계 동안 다른 작업을 돌리지 않는다. d12는 같은 배치에서 checkpointing 켬/끔 속도 비를 재서 대여 GPU 시간 추정에 쓴다."""
    final = {"bot": ["memmap", "worker_norm", "no_timers", "prefetch4"], "clipreid": ["memmap", "worker_norm", "no_timers", "prefetch4", "pin"],
             "kpr": ["memmap", "mask_f32", "no_timers", "prefetch4"], "kpr_solider": ["memmap", "mask_f32", "no_timers", "prefetch4"]}
    for m in MODELS:
        for k in (3, 4):
            b, c = fits(m, k)
            name = f"d11_{m}_x{k}_final"
            new = not (DONE / f"{name}.json").exists()
            r = train(name, m, k, b, iters=300, warmup=20, workers=4, impl="fast", flags=final[m], ckpt=c)
            if new:
                tp_row(m, k, b, c, 4, "fast", final[m], r, note="채택 후보 구성(입력 비트 동일 경로)")
    for m, k, b in (("clipreid", 4, 32), ("kpr", 3, 32), ("kpr", 4, 16), ("kpr_solider", 3, 32), ("kpr_solider", 4, 16)):
        for c in (0, 1):
            name = f"d12_{m}_x{k}_b{b}_ckpt{c}"
            new = not (DONE / f"{name}.json").exists()
            r = train(name, m, k, b, iters=150, warmup=20, workers=4, impl="fast", flags=final[m], ckpt=c)
            if new:
                tp_row(m, k, b, c, 4, "fast", final[m], r, note="checkpointing 속도 비 측정")


SERVER_S2 = "새 서버(EPYC 7282, cgroup v2, 116.4GiB)"
S2_TP = RES / "stage05_srv2_throughput.csv"
S2_RAM = RES / "stage05_srv2_ram.csv"
S2_TP_FIELDS = ["server", "step", "model", "input", "batch", "ckpt", "workers", "flags", "bipo", "pin", "iters", "it_s", "img_s", "gpu_util_mean", "max_rss_sum_gib", "max_pss_sum_gib",
                "nvidia_peak_mib", "seconds", "status", "note"]


def s2_row(step, m, k, b, c, w, flags, iters, r, note):
    o = r["out"]
    pin = "off" if "no_pin" in flags else "on"
    write_row(S2_TP, S2_TP_FIELDS, {"server": SERVER_S2, "step": step, "model": m, "input": INPUT[k], "batch": b, "ckpt": c, "workers": w, "flags": "+".join(flags),
              "bipo": "on" if m.startswith("kpr") else "해당 없음", "pin": pin, "iters": iters, "it_s": o.get("it_s", ""), "img_s": o.get("img_s", ""), "gpu_util_mean": r.get("gpu_util_mean", ""),
              "max_rss_sum_gib": round(r["max_rss_sum_gib"], 3), "max_pss_sum_gib": round(r["max_pss_sum_gib"], 3), "nvidia_peak_mib": r.get("max_gpu_mib", ""), "seconds": round(r["seconds"], 1),
              "status": status(r), "note": note})


def phase_tpfinal_s2():
    """지시 3e를 새 서버에서. 이전 서버 완료 표시(d8_*, d11_*, d12_*)는 그대로 두고 이름 끝에 _s2를 붙여 따로 기록한다.
    이 단계 동안 다른 작업을 돌리지 않는다. cgroup v2 memory.current를 1초마다 logs/stage05/cgroup_mem_v2.csv에 남기고 끝나면 ram_join으로 단계별 피크를 낸다.
    1) d11 최종 구성 처리량 + 학습 RAM(KPR은 공식 BIPO 켬, occluder 전부 로드) 300 iteration
    2) d8 핀 메모리 on/off(이전 서버 d8과 같은 플래그) 300 iteration
    3) d12 같은 배치에서 checkpointing 켬/끔 속도 비 150 iteration
    RAM 감시 한도는 limit_gib=48 그대로(57.7GiB 기준 보수 판정)."""
    final = {"bot": ["memmap", "worker_norm", "no_timers", "prefetch4"], "clipreid": ["memmap", "worker_norm", "no_timers", "prefetch4", "pin"],
             "kpr": ["memmap", "mask_f32", "no_timers", "prefetch4"], "kpr_solider": ["memmap", "mask_f32", "no_timers", "prefetch4"]}
    logger = subprocess.Popen([sys.executable, str(ROOT / "stage05" / "cgroup_log.py"), str(LOG / "cgroup_mem_v2.csv")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for m in MODELS:
            for k in (3, 4):
                b, c = fits(m, k)
                name = f"d11_{m}_x{k}_final_s2"
                new = not (DONE / f"{name}.json").exists()
                r = train(name, m, k, b, iters=300, warmup=20, workers=4, impl="fast", flags=final[m], ckpt=c)
                if new:
                    s2_row(name, m, k, b, c, 4, final[m], 300, r, "최종 구성 처리량·학습 RAM")
        for m in MODELS:
            b, c = fits(m, 3)
            for tag, flags in (("pin", ["memmap", "pin"]), ("nopin", ["memmap", "no_pin"])):
                name = f"d8_{m}_x3_{tag}_s2"
                new = not (DONE / f"{name}.json").exists()
                r = train(name, m, 3, b, iters=300, warmup=20, flags=flags, ckpt=c)
                if new:
                    s2_row(name, m, 3, b, c, r["out"].get("args", {}).get("workers", 0), flags, 300, r, "핀 메모리 비교(이전 서버 d8과 같은 플래그)")
        for m, k, b in (("clipreid", 4, 32), ("kpr", 3, 32), ("kpr", 4, 16), ("kpr_solider", 3, 32), ("kpr_solider", 4, 16)):
            for c in (0, 1):
                name = f"d12_{m}_x{k}_b{b}_ckpt{c}_s2"
                new = not (DONE / f"{name}.json").exists()
                r = train(name, m, k, b, iters=150, warmup=20, workers=4, impl="fast", flags=final[m], ckpt=c)
                if new:
                    s2_row(name, m, k, b, c, 4, final[m], 150, r, "checkpointing 속도 비")
    finally:
        logger.terminate()
        logger.wait(timeout=10)
    subprocess.run([sys.executable, str(ROOT / "stage05" / "ram_join.py")], cwd=ROOT, check=False)


def cpu_stat():
    try:
        return {l.split()[0]: int(l.split()[1]) for l in open("/sys/fs/cgroup/cpu.stat") if len(l.split()) == 2}
    except OSError:
        return {}


def phase_threads_s2():
    """3e 후속 진단(새 서버). KPR ×3 처리량이 이전 서버보다 약 30% 낮고 GPU 사용률이 58%라 원인을 본다.
    가설: torch 기본 intra-op 스레드 64개(nproc)가 cgroup CPU 할당 13.6코어를 넘어 스로틀링된다. OMP/MKL 스레드만 바꾸고 나머지는 d11과 같다.
    학습 레시피·수치 결정에는 영향이 없는 측정이며 채택 여부는 사용자가 정한다."""
    final = ["memmap", "mask_f32", "no_timers", "prefetch4"]
    for m in ("kpr", "kpr_solider"):
        b, c = fits(m, 3)
        for nt in (4, 8):
            name = f"d13_{m}_x3_final_omp{nt}_s2"
            new = not (DONE / f"{name}.json").exists()
            c0 = cpu_stat()
            r = train(name, m, 3, b, iters=300, warmup=20, workers=4, impl="fast", flags=final, ckpt=c, extra_env={"OMP_NUM_THREADS": str(nt), "MKL_NUM_THREADS": str(nt)})
            c1 = cpu_stat()
            if new:
                d = {k: c1.get(k, 0) - c0.get(k, 0) for k in ("nr_periods", "nr_throttled", "throttled_usec", "user_usec", "system_usec")}
                s2_row(name, m, 3, b, c, 4, final + [f"omp{nt}"], 300, r, "스레드 진단 cpu.stat 차이 " + " ".join(f"{k}={v}" for k, v in d.items()))


def phase_extra():
    """중간 지시 1·7: 256×128 프로파일(로딩 병목 가설 범위 확인), 핀 메모리 on/off, 실패한 표본 프로파일 재측정."""
    for m, b in (("kpr", 64), ("kpr_solider", 64)):
        for tag, flags in (("", []), ("_memmap", ["memmap"])):
            name = f"d1c_{m}_x2{tag}"
            if not (DONE / f"{name}.json").exists():
                train(name, m, 2, b, task="profile", iters=100, warmup=10, flags=flags)
    for m in MODELS:
        b, c = fits(m, 3)
        for tag, flags in (("pin", ["memmap", "pin"]), ("nopin", ["memmap", "no_pin"])):
            name = f"d8_{m}_x3_{tag}"
            new = not (DONE / f"{name}.json").exists()
            r = train(name, m, 3, b, iters=300, warmup=20, flags=flags, ckpt=c)
            if new:
                tp_row(m, 3, b, c, r["out"].get("args", {}).get("workers", 0), "fast", flags, r, note="D 핀 메모리 비교")


def ram_row(m, k, b, w, mode, r, note=""):
    o = r["out"]
    write_row(RES / "stage05_ram.csv", RAM_FIELDS, {"model": m, "phase": "train", "mode": mode, "input": INPUT[k], "batch": b, "workers": w, "max_rss_sum_gib": round(r["max_rss_sum_gib"], 3),
              "max_pss_sum_gib": round(r["max_pss_sum_gib"], 3), "time_v_max_rss_gib": round(r.get("time_v_max_rss_gib", 0), 3), "seconds": round(r["seconds"], 1), "rc": r["rc"], "killed_over_limit": r["killed_over_limit"],
              "status": status(r), "note": note or f"500 iteration, it/s={o.get('it_s')}"})


def tp_row(m, k, b, c, w, impl, flags, r, exclusive=True, note=""):
    o = r["out"]
    write_row(RES / "stage05_throughput.csv", TP_FIELDS, {"model": m, "input": INPUT[k], "batch": b, "ckpt": c, "workers": w, "impl": impl, "flags": "+".join(flags), "it_s": o.get("it_s", ""), "img_s": o.get("img_s", ""),
              "gpu_util_mean": r.get("gpu_util_mean", ""), "max_rss_sum_gib": round(r["max_rss_sum_gib"], 3), "max_pss_sum_gib": round(r["max_pss_sum_gib"], 3), "nvidia_peak_mib": r.get("max_gpu_mib", ""), "exclusive_gpu": exclusive,
              "seconds": round(r["seconds"], 1), "status": status(r), "note": note})


def phase_ram():
    """B.3 학습: 공식 경로(디코딩) 500 iteration, 워커 4, 8. KPR d) 마스크 uint8 memmap 전후."""
    for m in MODELS:
        for k in (3, 4):
            b, c = fits(m, k)
            for w in (4, 8):
                name = f"b3t_{m}_x{k}_w{w}"
                new = not (DONE / f"{name}.json").exists()
                r = train(name, m, k, b, iters=500, workers=w, ckpt=c)
                if new:
                    ram_row(m, k, b, w, "train_official_loader", r)
                    tp_row(m, k, b, c, w, "baseline", [], r, note="B3 500 iteration")
                if m.startswith("kpr"):
                    name = f"b3t_{m}_x{k}_w{w}_memmap"
                    new = not (DONE / f"{name}.json").exists()
                    r = train(name, m, k, b, iters=500, workers=w, ckpt=c, flags=["memmap"])
                    if new:
                        ram_row(m, k, b, w, "d_mask_u8_memmap", r)


def fast_flags(m):
    """프로파일 결과로 정한 최적화 조합. 등가성 phase_equiv를 통과한 것만 기본값으로 쓴다."""
    return ALL_FLAGS


ALL_FLAGS = ["memmap", "no_timers", "prefetch4"]


def losses_of(r):
    return r["out"].get("losses") or []


def compare(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return None
    d = [abs(x - y) for x, y in zip(a[:n], b[:n])]
    first = next((i for i, v in enumerate(d) if v > 0), -1)
    return {"max": max(d), "mean": sum(d) / n, "first": first, "n": n}


def phase_equiv():
    """D.5: 같은 seed, 200 iteration. 기준선 반복 간 차이를 잡음으로 삼고, 최적화 적용 차이가 그 안이면 통과."""
    for m in MODELS:
        k = 3
        b, c = fits(m, k)
        w = 4
        base = train(f"d5_{m}_base", m, k, b, iters=200, workers=w, ckpt=c)
        base2 = train(f"d5_{m}_base2", m, k, b, iters=200, workers=w, ckpt=c)
        noise = compare(losses_of(base), losses_of(base2))
        cases = {"memmap": ["memmap", "mask_f32"] if m.startswith("kpr") else ["memmap"], "all": ALL_FLAGS + (["mask_f32"] if m.startswith("kpr") else [])}
        if m.startswith("kpr"):
            cases["all_u8"] = ALL_FLAGS + ["mask_u8"]
        if m == "bot":
            cases["channels_last"] = ["channels_last"]
            cases["benchmark"] = ["benchmark"]
        if m == "clipreid":
            cases["benchmark"] = ["benchmark"]
        for tag, flags in cases.items():
            name = f"d5_{m}_{tag}"
            new = not (DONE / f"{name}.json").exists()
            r = train(name, m, k, b, iters=200, workers=w, ckpt=c, flags=flags)
            cmp_ = compare(losses_of(base), losses_of(r))
            if new:
                ok = bool(cmp_ and noise and cmp_["max"] <= max(noise["max"], 1e-9))
                write_row(RES / "stage05_train_equiv.csv", EQ_FIELDS, {"model": m, "input": INPUT[k], "batch": b, "workers": w, "comparison": f"baseline vs {tag}", "flags": "+".join(flags), "iters": cmp_["n"] if cmp_ else 0,
                          "max_abs_loss_diff": cmp_ and cmp_["max"], "mean_abs_loss_diff": cmp_ and cmp_["mean"], "noise_max": noise and noise["max"], "noise_mean": noise and noise["mean"], "first_diff_iter": cmp_ and cmp_["first"], "passed": ok,
                          "note": "" if r["out"].get("status") == "ok" else r["out"].get("error", "")[:200]})


def phase_tp():
    """D.6: 384×192 전후 비교. after만 새로 잰다(before는 phase_ram의 baseline, 같은 워커 수). 동시 실행은 phase_concurrent."""
    for m in MODELS:
        for k in (3, 4):
            b, c = fits(m, k)
            for w in (4, 8):
                name = f"d6_{m}_x{k}_w{w}_fast"
                new = not (DONE / f"{name}.json").exists()
                flags = fast_flags(m)
                r = train(name, m, k, b, iters=300, warmup=20, workers=w, ckpt=c, flags=flags)
                if new:
                    tp_row(m, k, b, c, w, "fast", flags, r, note="D.6")


def phase_concurrent():
    """D.7: BoT 2개 동시. 각자 monitor 아래. exclusive_gpu=false로 기록."""
    m, k = "bot", 3
    b, c = fits(m, k)
    flags = fast_flags(m)
    res = {}

    def go(i):
        res[i] = train(f"d7_bot_x3_concurrent{i}", m, k, b, iters=300, warmup=20, workers=4, ckpt=c, flags=flags)
    new = not (DONE / "d7_bot_x3_concurrent0.json").exists()
    ths = [threading.Thread(target=go, args=(i,)) for i in range(2)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    if new:
        for i in range(2):
            tp_row(m, k, b, c, 4, "fast", flags, res[i], exclusive=False, note=f"D.7 동시 {i}")


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)
    quarantine_bad_done()
    for p in sys.argv[1:]:
        globals()[f"phase_{p}"]()
    print("run_train 종료", flush=True)


if __name__ == "__main__":
    main()
