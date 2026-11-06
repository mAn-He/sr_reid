"""Part C(등가성, 기준선)와 Part B3(평가 RAM) 실행. 재실행하면 끝난 단계를 건너뛴다. with_test.sh 아래에서 돈다. 실패도 CSV에 남기고 다음으로 간다."""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage05 import monitor  # noqa: E402

PY = "/workspace/miniforge3/envs/srreid-core/bin/python"
ENVPY = {"bot": PY, "clipreid": "/workspace/miniforge3/envs/srreid-clipreid/bin/python", "kpr": "/workspace/miniforge3/envs/srreid-kpr/bin/python",
         "kpr_solider": "/workspace/miniforge3/envs/srreid-kpr/bin/python"}
SCRIPT = {"bot": "stage05/bot_run.py", "clipreid": "stage05/clip_run.py", "kpr": "stage05/kpr_run.py", "kpr_solider": "stage05/kpr_run.py"}
FEAT = ROOT / "data" / "locked" / "features"
DONE = ROOT / "runs" / "stage05" / "done"
LOG = ROOT / "logs" / "stage05"
RAM = ROOT / "results" / "stage05_ram.csv"
FIELDS = ["model", "phase", "mode", "input", "batch", "workers", "n_query", "n_gallery", "max_rss_sum_gib", "max_pss_sum_gib", "time_v_max_rss_gib", "seconds", "rc",
          "killed_over_limit", "status", "note"]
SIZE = {1: {"bot": (256, 128), "clipreid": (256, 128), "kpr": (384, 128), "kpr_solider": (384, 128)}, 3: (384, 192), 4: (512, 256)}


def env():
    e = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", WANDB_MODE="disabled", ALLOW_TEST="1")
    return e


def append(row):
    new = not RAM.exists()
    with open(RAM, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def base_args(model, task, k, features, batch, workers, sub_q=0, sub_g=0, public=True):
    h, w = SIZE[1][model] if k == 1 else SIZE[k]
    args = ["--task", task, "--features", features, "--batch", batch, "--workers", workers, "--sub-q", sub_q, "--sub-g", sub_g]
    cond = "orig" if k == 1 else "bicubic"
    if model == "bot":
        args += ["--cond", cond, "--k", k, "--arch", "official" if public else "inhouse"]
    elif model == "clipreid":
        args += ["--cond", cond, "--k", k, "--h", h, "--w", w, "--public", int(public)]
    else:
        args += ["--model", model, "--cond", cond, "--k", k, "--h", h, "--w", w]
    return args, f"{h}x{w}"


def step(name, model, phase, mode, k, args, inp, batch, workers, nq, ng, limit=48.0, note=""):
    """monitor 아래에서 한 작업 실행 후 CSV 기록. 끝난 표식이 있으면 건너뛰고 저장된 결과를 돌려준다."""
    done = DONE / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    out = LOG / f"{name}.out.json"
    res = monitor.run([ENVPY[model], SCRIPT[model]] + args + ["--out", out], LOG / f"{name}.log", env=env(), limit_gib=limit, cwd=ROOT)
    status = "ok" if res["rc"] == 0 else ("over_ram_limit" if res["killed_over_limit"] else "failed")
    log_tail = Path(str(LOG / f"{name}.log")).read_text(errors="ignore")[-3000:].lower()
    if status == "failed" and "out of memory" in log_tail:
        status = "gpu_oom"
    append({"model": model, "phase": phase, "mode": mode, "input": inp, "batch": batch, "workers": workers, "n_query": nq, "n_gallery": ng,
            "max_rss_sum_gib": round(res["max_rss_sum_gib"], 3), "max_pss_sum_gib": round(res["max_pss_sum_gib"], 3),
            "time_v_max_rss_gib": round(res.get("time_v_max_rss_gib", 0), 3), "seconds": round(res["seconds"], 1), "rc": res["rc"],
            "killed_over_limit": res["killed_over_limit"], "status": status, "note": note})
    res["status"] = status
    if out.exists():
        res["out"] = json.loads(out.read_text())
    DONE.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[{name}] {status} rc={res['rc']} rss={res['max_rss_sum_gib']:.2f} pss={res['max_pss_sum_gib']:.2f} {res['seconds']:.0f}s", flush=True)
    return res


def extract_light(model, k, features, batch, workers, sub_q, sub_g, public, tag, phase, blocks=(256, 512, 1024), note=""):
    """배치를 낮춰 가며 추출, 그다음 블록별 스트리밍. 사용한 배치를 돌려준다."""
    b = batch
    while b >= 8:
        args, inp = base_args(model, "extract", k, features, b, workers, sub_q, sub_g, public)
        r = step(f"{tag}_extract_b{b}", model, phase, "light_extract", k, args, inp, b, workers, sub_q or 3368, sub_g or 15913, note=note)
        if r["status"] == "ok":
            break
        if r["status"] != "gpu_oom":
            return None, None
        b //= 2
    else:
        return None, None
    args = ["--task", "light", "--features", features, "--blocks"] + list(blocks)
    r2 = step(f"{tag}_light", model, phase, "light_stream", k, args, inp, b, workers, sub_q or 3368, sub_g or 15913, note=note)
    return b, r2


def part_c():
    """C.6: 공식 입력·공개 가중치. 부분집합 등가성, 이어서 전체 test 기준선."""
    n_q, n_g = 500, 5000
    batches = {"bot": 128, "clipreid": 64, "kpr": 64, "kpr_solider": 64}
    for model in ("bot", "clipreid", "kpr", "kpr_solider"):
        f = FEAT / model / "public_sub"
        b, _ = extract_light(model, 1, f, batches[model], 8, n_q, n_g, True, f"c6_{model}_sub", "eval_equiv", note="공개 가중치, 공식 입력, 부분집합")
        if model == "bot":
            args = ["--task", "official", "--features", f]
            step(f"c6_{model}_official", model, "eval_equiv", "official_subset", 1, args, "256x128", b, 8, n_q, n_g)
        elif model == "clipreid":
            args, inp = base_args(model, "official", 1, f, b, 8, n_q, n_g)
            step(f"c6_{model}_official", model, "eval_equiv", "official_subset", 1, args, inp, b, 8, n_q, n_g)
        else:
            args, inp = base_args(model, "official", 1, f, b, 8, n_q, n_g)
            step(f"c6_{model}_official", model, "eval_equiv", "official_subset", 1, args + ["--visrank", 0], inp, b, 8, n_q, n_g, note="공식 평가기, visrank 끔")
        f = FEAT / model / "public_full"
        extract_light(model, 1, f, batches[model], 8, 0, 0, True, f"baseline_{model}", "baseline", note="공개 가중치, 공식 입력, 전체 test")


def part_b_eval():
    batches = {3: {"bot": 128, "clipreid": 64, "kpr": 64, "kpr_solider": 64}, 4: {"bot": 128, "clipreid": 64, "kpr": 32, "kpr_solider": 32}}
    for k in (3, 4):
        h, w = SIZE[k]
        inp = f"{h}x{w}"
        for model in ("bot", "clipreid"):
            f = FEAT / model / f"ram_x{k}"
            b, _ = extract_light(model, k, f, batches[k][model], 8, 0, 0, False, f"b3_{model}_x{k}", "eval", note="지표 비기록(공개 가중치 아님)")
            if model == "bot":
                args = ["--task", "official", "--features", f]
                step(f"b3_{model}_x{k}_official", model, "eval", "official_full", k, args, inp, b, 8, 3368, 15913, note="공식 전체 거리 행렬+eval_func, 특징 재사용")
            else:
                args, _ = base_args(model, "official", k, f, b, 8, 0, 0, False)
                step(f"b3_{model}_x{k}_official", model, "eval", "official_full", k, args + ["--record-metrics", 0], inp, b, 8, 3368, 15913, note="공식 do_inference")
            shutil.rmtree(f, ignore_errors=True)
        for model in ("kpr", "kpr_solider"):
            b = batches[k][model]
            for sub_q, sub_g in ((100, 1000), (200, 2000)):
                f = FEAT / model / f"ram_x{k}_tmp"
                for mode, vr in (("a_official_visrank", 1), ("b_visrank_off", 0)):
                    args, _ = base_args(model, "official", k, f, b, 8, sub_q, sub_g)
                    step(f"b3_{model}_x{k}_{mode}_q{sub_q}", model, "eval", mode, k, args + ["--visrank", vr, "--record-metrics", 0], inp, b, 8, sub_q, sub_g,
                         note="공식 경로 부분집합, 전체는 역산표 기준 한도 초과")
            f = FEAT / model / f"ram_x{k}"
            extract_light(model, k, f, b, 8, 0, 0, True, f"b3_{model}_x{k}_c", "eval", note="c: 경량 평가기, 공개 가중치 입력 크기만 변경, 지표 비사용")
            shutil.rmtree(f, ignore_errors=True)


LIGHT = ["--no-pin", 1, "--no-occluders", 1]


def light_fixed(model, k, tag, features, batch, workers=8, sub_q=0, sub_g=0, phase="eval", note="", blocks=(256, 512, 1024)):
    """핀 메모리·occluder를 끈 경량 경로. 추출 후 스트리밍."""
    args, inp = base_args(model, "extract", k, features, batch, workers, sub_q, sub_g)
    r = step(f"{tag}_extract", model, phase, "c_light_extract_fixed", k, args + LIGHT, inp, batch, workers, sub_q or 3368, sub_g or 15913, limit=50.0, note=note)
    if r["status"] != "ok":
        return r, None
    return r, step(f"{tag}_light", model, phase, "c_light_stream_fixed", k, ["--task", "light", "--features", features, "--blocks"] + list(blocks), inp, batch, workers,
                   sub_q or 3368, sub_g or 15913, note=note)


def part_r():
    """원인(핀 메모리 캐시, occluder) 제거 후 재측정. 이전 행은 실패 기록으로 남긴다."""
    for model in ("kpr", "kpr_solider"):
        light_fixed(model, 1, f"r_{model}_x1", FEAT / model / "public_full_v2", 64, phase="baseline", note="핀 메모리·occluder 끔. 특징이 public_full과 같은지 확인")
        for k, b in ((3, 64), (4, 32)):
            f = FEAT / model / f"ram_x{k}"
            light_fixed(model, k, f"r_{model}_x{k}", f, b, note="핀 메모리·occluder 끔. 공개 가중치, 입력 크기만 변경, 지표 비사용")
            shutil.rmtree(f, ignore_errors=True)
    # C.6 재시도: 공식 평가기(visrank 끔)를 핀 메모리·occluder 없이, 워커 4로. 넘으면 부분집합을 줄인다.
    for model in ("kpr", "kpr_solider"):
        for nq, ng in ((500, 5000), (300, 3000)):
            f = FEAT / model / ("public_sub" if nq == 500 else f"public_sub_q{nq}")
            if nq != 500:
                light_fixed(model, 1, f"c6r_{model}_sub{nq}", f, 64, sub_q=nq, sub_g=ng, phase="eval_equiv", note="공식 평가기가 한도 안에 들도록 부분집합 축소")
            args, inp = base_args(model, "official", 1, f, 64, 4, nq, ng)
            r = step(f"c6r_{model}_official_q{nq}", model, "eval_equiv", "official_subset_nopin", 1, args + ["--visrank", 0] + LIGHT, inp, 64, 4, nq, ng, limit=50.0,
                     note="공식 evaluate 경로, visrank·핀 메모리·occluder 끔, 워커 4")
            if r["status"] == "ok":
                break
    # a와 b 비교: 양성 있는 query만 고른 부분집합(공식 visrank가 양성 없는 query에서 IndexError)
    for k, b in ((3, 64), (4, 32)):
        f = FEAT / "kpr" / f"ram_x{k}_tmp"
        args, inp = base_args("kpr", "official", k, f, b, 8, 100, 1000)
        step(f"r_kpr_x{k}_a_match_q100", "kpr", "eval", "a_official_visrank_match", k, args + ["--visrank", 1, "--record-metrics", 0, "--q-has-match", 1], inp, b, 8, 100, 1000,
             note="양성 있는 query로 한정한 부분집합")
    print("part_r 종료", flush=True)


def part_r2():
    for model in ("kpr", "kpr_solider"):
        f = FEAT / model / "ram_x4"
        light_fixed(model, 4, f"r2_{model}_x4", f, 32, note="핀 메모리·occluder 끔, nofile 상향. 공개 가중치, 입력 크기만 변경, 지표 비사용")
        shutil.rmtree(f, ignore_errors=True)
    print("part_r2 종료", flush=True)


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "locked" / "measure" / "kpr_tmp").mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["c", "b"]
    if "r" in which:
        part_r()
    if "r2" in which:
        part_r2()
    if "c" in which:
        part_c()
    if "b" in which:
        part_b_eval()
    print("run_eval 종료", flush=True)


if __name__ == "__main__":
    main()
