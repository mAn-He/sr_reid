"""SR 실행. 공통 인터페이스 run(in_dir, out_dir, cfg) -> timing.json 경로.

phase
- exclusive: 조건별 처음 N장을 하나씩 단독 실행 (--require-exclusive). 소요 외삽에는 이 값만 쓴다.
- rest: 나머지 전부. 학습과 동시 실행 허용. OOM이면 재시도 후 실패 목록에 기록.
- determinism: 같은 입력을 다시 돌려 출력 sha256 비교.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from common.util import (EXIT_GUARD, EXIT_OOM, ROOT, gpu_memory_gb, load_cfg, log_failure, read_csv, sha256_file,
                         wait_exclusive, wait_free_gb, write_csv)

MF = Path("/workspace/miniforge3/envs")
OUT_FIELDS = ["name", "input_sha256", "raw_sha256", "final_sha256", "status", "height", "width"]
PROCESS_SIZE = {"official": 512, "native": 64, "fixed8x": 512}


def conditions(cfg):
    out = []
    for method, m in cfg["sr"]["methods"].items():
        for sem in m.get("lambda_sem", [None]):
            for policy in cfg["sr"]["preup_policies"]:
                base = method if sem is None else f"{method}_sem{sem}"
                out.append({"cid": f"{base}__{policy}", "base": base, "method": method, "policy": policy,
                            "lambda_sem": sem, **m})
    return out


def cond_dir(cfg, c):
    return cfg["_run"] / "sr" / c["cid"]


def prepare_inputs(cfg, c, d):
    """official/native는 LR 그대로, fixed8x는 bicubic 8× 선확대본을 입력으로 쓴다."""
    in_dir = d / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    h, w = cfg["lr"]["lr_size"]
    lr_files = sorted((cfg["_processed"] / "val_query_lr").glob("*.png"))
    for p in lr_files:
        dst = in_dir / p.name
        if dst.exists():
            continue
        if c["policy"] == "fixed8x":
            Image.open(p).convert("RGB").resize((w * 8, h * 8), Image.BICUBIC).save(dst)
        else:
            dst.symlink_to(p)
    if not (d / "inputs.csv").exists():
        write_csv(d / "inputs.csv", [{"name": p.name, "sha256": sha256_file(in_dir / p.name)} for p in lr_files],
                  ["name", "sha256"])
    return [p.name for p in lr_files]


def external_args(cfg, c, in_dir, raw):
    W = ROOT / "weights"
    if c["kind"] == "realesrgan":
        return ["-n", "RealESRGAN_x4plus", "-i", str(in_dir), "-o", str(raw), "-s", "4", "--suffix", "",
                "--ext", "png", "--model_path", str(W / "RealESRGAN_x4plus.pth")]
    size = str(PROCESS_SIZE[c["policy"]])
    if c["kind"] == "pisa":
        return ["--pretrained_model_path", str(W / "sd21_base"), "--pretrained_path", str(W / "pisa_sr" / "pisa_sr.pkl"),
                "--process_size", size, "--upscale", "4", "--input_image", str(in_dir), "--output_dir", str(raw),
                "--lambda_pix", str(c["lambda_pix"]), "--lambda_sem", str(c["lambda_sem"]), "--seed", str(cfg["seed"])]
    return ["-i", str(in_dir), "-o", str(raw), "--osediff_path", str(W / "osdhuman" / "model.pkl"),
            "--embedding_change_path", str(W / "osdhuman" / "model_embedding_change.pkl"),
            "--pretrained_model_name_or_path", str(W / "sd21_base"), "--ram_path", str(W / "ram_swin_large_14m.pth"),
            "--ram_img_only", "--process_size", size, "--upscale", str(c["upscale"]), "--seed", str(cfg["seed"])]


def run_classic(cfg, c, in_dir, raw, names):
    """고전 보간은 이 프로세스에서 처리한다. fixed8x의 256×128 입력도 ×4 후 finalize에서 128×64로 줄인다."""
    t0 = time.perf_counter()
    used0 = gpu_memory_gb()[0]
    filt = getattr(Image, c["filter"])
    stamps = []
    for n in names:
        img = Image.open(in_dir / n).convert("RGB")
        img.resize((img.width * 4, img.height * 4), filt).save(raw / n)
        stamps.append(time.perf_counter())
    return {"rc": 0, "error": None, "n_saved": len(stamps),
            "per_image_seconds": [b - a for a, b in zip(stamps, stamps[1:])],
            "first_save_seconds": stamps[0] - t0 if stamps else None, "total_seconds": time.perf_counter() - t0,
            "max_memory_allocated": 0, "max_memory_reserved": 0, "start_used_mib": used0 * 1024,
            "max_foreign_mib": max(used0, gpu_memory_gb()[0]) * 1024}


def run_external(cfg, c, in_dir, raw, d, fraction, pause_file=None):
    work = d / "work"
    work.mkdir(exist_ok=True)
    if c["kind"] == "osd":  # ram/models/utils.py가 cwd 기준 preset/models/bert-base-uncased를 읽는다.
        link = work / "preset" / "models" / "bert-base-uncased"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(ROOT / "weights" / "bert-base-uncased")
    timing = work / f"timing_{time.time_ns()}.json"
    repo = ROOT / c["repo"]
    cmd = [str(MF / c["env"] / "bin" / "python"), str(ROOT / "sr" / "runner.py"), "--root", str(ROOT), "--repo", str(repo),
           "--script", str(repo / c["script"]), "--out-dir", str(raw), "--timing", str(timing), "--seed", str(cfg["seed"]),
           "--fraction", str(fraction), "--"] + external_args(cfg, c, in_dir, raw)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0",
           "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(repo)}
    with open(d / "external.log", "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(5)
            if pause_file and Path(pause_file).exists():  # 학습 OOM: SR을 멈추고 학습에 자리를 내준다.
                proc.terminate()
                proc.wait()
                return {"rc": EXIT_OOM, "error": "학습 우선으로 중단", "n_saved": 0, "per_image_seconds": [],
                        "max_foreign_mib": None, "max_memory_allocated": None, "max_memory_reserved": None}
    if timing.exists():
        res = json.loads(timing.read_text())
    else:
        res = {"rc": proc.returncode, "error": "timing 없음(프로세스 비정상 종료)", "n_saved": 0, "per_image_seconds": [],
               "max_foreign_mib": None, "max_memory_allocated": None, "max_memory_reserved": None}
    res["rc"] = proc.returncode
    return res


def load_done(d):
    """manifest와 sha256이 일치하는 출력만 완료로 본다. 어긋나면 지우고 다시 만든다."""
    path = d / "outputs.csv"
    rows = {r["name"]: r for r in read_csv(path)} if path.exists() else {}
    for name, r in list(rows.items()):
        f = d / "final" / name
        if r["status"] == "ok" and (not f.exists() or sha256_file(f) != r["final_sha256"]):
            del rows[name]
            f.unlink(missing_ok=True)
            (d / "raw" / name).unlink(missing_ok=True)
    return rows


def finalize(cfg, c, d, rows):
    """raw를 검사해 128×64 final로 만든다. fixed8x만 정책대로 축소하고, 다른 크기는 고치지 않고 wrong_size로 기록한다."""
    h, w = cfg["sr"]["output_size"]
    (d / "final").mkdir(exist_ok=True)
    inputs = {r["name"]: r["sha256"] for r in read_csv(d / "inputs.csv")}
    for p in sorted((d / "raw").glob("*.png")):
        if p.name in rows:
            continue
        try:
            img = Image.open(p)
            img.load()
            img = img.convert("RGB")
        except OSError as e:  # 저장 도중 끊긴 파일. 지우고 다시 만든다.
            print(f"손상된 raw 삭제 {p.name}: {e}", flush=True)
            p.unlink()
            continue
        if c["policy"] == "fixed8x":
            img = img.resize((w, h), Image.BICUBIC)
        status = "ok" if img.size == (w, h) else "wrong_size"
        final_sha = ""
        if status == "ok":
            img.save(d / "final" / p.name)
            final_sha = sha256_file(d / "final" / p.name)
        rows[p.name] = {"name": p.name, "input_sha256": inputs.get(p.name, ""), "raw_sha256": sha256_file(p),
                        "final_sha256": final_sha, "status": status, "height": img.height, "width": img.width}
    write_csv(d / "outputs.csv", list(rows.values()), OUT_FIELDS)
    return rows


def record_timing(d, phase, res, n_images, cfg):
    path = d / "timing.json"
    t = json.loads(path.read_text()) if path.exists() else {"chunks": []}
    foreign = res.get("max_foreign_mib")
    start = res.get("start_used_mib")
    gc = cfg["gpu"]
    exclusive = bool(phase == "exclusive" and res.get("per_image_seconds") and foreign is not None
                     and start is not None and start / 1024 <= gc["exclusive_max_used_gb"]
                     and foreign / 1024 <= gc["exclusive_max_foreign_gb"])
    t["chunks"].append({"phase": phase, "n_images": n_images, "exclusive_gpu": exclusive, "time": time.strftime("%F %T"),
                        **res})
    ex = [c for c in t["chunks"] if c["exclusive_gpu"]]
    sec = [s for c in ex for s in c["per_image_seconds"]]
    t["sec_per_image_exclusive"] = float(np.mean(sec)) if sec else None
    t["exclusive_gpu"] = bool(sec)
    t["max_memory_allocated_exclusive"] = max((c["max_memory_allocated"] or 0 for c in ex), default=None)
    t["max_memory_reserved_exclusive"] = max((c["max_memory_reserved"] or 0 for c in ex), default=None)
    path.write_text(json.dumps(t, indent=1))
    return path


def run(in_dir, out_dir, cfg):
    """in_dir의 todo 이미지를 처리해 out_dir에 남기고 timing.json 경로를 돌려준다. cfg는 조건과 실행 설정을 담는다."""
    c, g = cfg["condition"], cfg["global"]
    d, names, phase = Path(out_dir), cfg["names"], cfg["phase"]
    raw = d / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    chunk = d / "work" / "chunk_in"
    if chunk.exists():
        shutil.rmtree(chunk)
    chunk.mkdir(parents=True)
    for n in names:
        (chunk / n).symlink_to(Path(in_dir) / n)
    if c["kind"] == "classic":
        res = run_classic(g, c, chunk, raw, names)
    else:
        res = run_external(g, c, chunk, raw, d, cfg["fraction"], cfg.get("pause_file"))
    path = record_timing(d, phase, res, len(names), g)
    return path, res


def missing_weights(c):
    return [w for w in c.get("weights", []) if not (ROOT / w).exists()]


def check_guard(cfg, res):
    if res.get("rc") == EXIT_GUARD or (ROOT / "logs" / "GUARD_VIOLATION").exists():
        print("test 가드 위반. 전체 중단.", flush=True)
        sys.exit(EXIT_GUARD)


def phase_exclusive(cfg):
    n = cfg["sr"]["exclusive_n"]
    for c in conditions(cfg):
        d = cond_dir(cfg, c)
        miss = missing_weights(c)
        if miss:
            log_failure(cfg, "sr_exclusive", c["cid"], "", f"가중치 없음: {miss}")
            continue
        names = prepare_inputs(cfg, c, d)
        rows = load_done(d)
        todo = [x for x in names[:n] if x not in rows]
        if not todo:
            continue
        if not wait_exclusive(cfg["gpu"]["exclusive_max_used_gb"], cfg["gpu"]["wait_timeout_s"]):
            log_failure(cfg, "sr_exclusive", c["cid"], "", "--require-exclusive: GPU가 비지 않아 단독 측정 실패")
            continue
        _, res = run(d / "input", d, {"condition": c, "global": cfg, "names": todo, "phase": "exclusive",
                                      "fraction": cfg["gpu"]["fraction_exclusive"]})
        check_guard(cfg, res)
        finalize(cfg, c, d, rows)
        if res["rc"] != 0:
            log_failure(cfg, "sr_exclusive", c["cid"], "", f"rc={res['rc']} {res.get('error')}")
        print(f"[exclusive] {c['cid']}: rc={res['rc']} 저장 {res['n_saved']}장", flush=True)


def phase_rest(cfg, pause_file, train_done):
    for c in conditions(cfg):
        d = cond_dir(cfg, c)
        miss = missing_weights(c)
        if miss:
            log_failure(cfg, "sr_rest", c["cid"], "", f"가중치 없음: {miss}")
            continue
        names = prepare_inputs(cfg, c, d)
        retries = cfg["sr"]["oom_retries"]
        for attempt in range(retries + 1):
            rows = load_done(d)
            todo = [x for x in names if x not in rows]
            if not todo:
                break
            while Path(pause_file).exists():
                time.sleep(30)
            last_try = attempt == retries
            if last_try:  # 마지막 재시도는 학습이 모두 끝난 뒤 넉넉한 메모리로 한다.
                while not all(Path(p).exists() for p in train_done):
                    time.sleep(60)
            need = cfg["gpu"]["min_free_gb_sr"]
            if not wait_free_gb(need, cfg["gpu"]["wait_timeout_s"]):
                log_failure(cfg, "sr_rest", c["cid"], "", f"min_free_gb_sr {need} 미충족")
                continue
            frac = cfg["gpu"]["fraction_exclusive"] if last_try else cfg["gpu"]["fraction_sr"]
            _, res = run(d / "input", d, {"condition": c, "global": cfg, "names": todo, "phase": "rest",
                                          "fraction": frac, "pause_file": pause_file})
            check_guard(cfg, res)
            finalize(cfg, c, d, rows)
            print(f"[rest] {c['cid']}: 시도 {attempt} rc={res['rc']} 남은 {len(names) - len(rows)}장", flush=True)
            if res["rc"] == 0 and c["kind"] != "classic" and len(rows) < len(names):
                log_failure(cfg, "sr_rest", c["cid"], "", "프로세스는 정상 종료했으나 출력이 부족")
                break
            if res["rc"] not in (0, EXIT_OOM):
                log_failure(cfg, "sr_rest", c["cid"], "", f"rc={res['rc']} {res.get('error')}")
                break
            if res["rc"] == EXIT_OOM:
                time.sleep(120)
        rows = load_done(d)
        for n in names:
            if n not in rows:
                log_failure(cfg, "sr_rest", c["cid"], n, "재시도 후에도 출력 없음")


def phase_determinism(cfg):
    n = cfg["sr"]["determinism_n"]
    out = []
    for c in conditions(cfg):
        if c["kind"] == "classic" or c["policy"] != "official" or c["lambda_sem"] not in (None, 1.0):
            continue
        d = cond_dir(cfg, c)
        rows = load_done(d) if (d / "outputs.csv").exists() else {}
        names = sorted(rows)[:n]
        if not names or missing_weights(c):
            out.append({"condition": c["cid"], "n": 0, "identical": "", "note": "입력 또는 가중치 없음"})
            continue
        probe = d / "work" / "det_raw"
        if probe.exists():
            shutil.rmtree(probe)
        probe.mkdir(parents=True)
        chunk = d / "work" / "det_in"
        if chunk.exists():
            shutil.rmtree(chunk)
        chunk.mkdir()
        for x in names:
            (chunk / x).symlink_to(d / "input" / x)
        res = run_external(cfg, c, chunk, probe, d, cfg["gpu"]["fraction_exclusive"])
        check_guard(cfg, res)
        same = sum((probe / x).exists() and sha256_file(probe / x) == rows[x]["raw_sha256"] for x in names)
        out.append({"condition": c["cid"], "n": len(names), "identical": same, "note": f"rc={res['rc']}"})
        print(f"[determinism] {c['cid']}: {same}/{len(names)} 동일", flush=True)
    write_csv(cfg["_run"] / "determinism.csv", out, ["condition", "n", "identical", "note"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", required=True, choices=["exclusive", "rest", "determinism"])
    ap.add_argument("--require-exclusive", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.phase == "exclusive":
        if not a.require_exclusive:
            sys.exit("exclusive 단계는 --require-exclusive로만 실행한다")
        phase_exclusive(cfg)
    elif a.phase == "determinism":
        phase_determinism(cfg)
    else:
        done = cfg["_run"] / "done"
        phase_rest(cfg, cfg["_run"] / "pause_sr", [done / f"train_{m}.finished" for m in ("hr", "mixed", "lr_only")])


if __name__ == "__main__":
    main()
