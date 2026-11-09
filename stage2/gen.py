"""Stage 2 SR 생성 조율기(2026-11-09). srreid-core env(표준 라이브러리 + numpy·PIL). 실제 생성은 stage1/sr_gen.py(Stage 1과 같은 경로·설정).

대상: swinir(×3 공식 ×3 가중치, ×4), pisa_sr λ0.0(×4, ×3 = ×4 후 PIL BICUBIC 축소), osdhuman(같음), bicubic(test만. train은 data/bicubic_x{3,4} 재사용).
설정: osdhuman·pisa_sr --det --warmup + 파일명 seed, swinir 공식 설정, 배치 1(배치 정책 (가)).

단계(--phase):
  gen    GPU 작업(swinir×3, swinir×4, pisa×4, osdhuman×4)을 이미지 목록 조각으로 나눠 GPU 2장 × P 프로세스가 공유 대기열에서 꺼내 실행.
         pisa·osdhuman 조각의 ×4가 끝나면 같은 조각의 ×3 축소(CPU)를 바로 띄운다. test면 bicubic ×3·×4(CPU)도 함께 돈다.
  solo   조각마다 20장(seed 42)을 GPU당 한 프로세스로 다시 생성(저장 없음) → 동시 실행 출력과 배열 SHA1 비교.
  check  (train) Stage 1 표본 2,250장을 Stage 1 출력과 비교(SR 3종 ×3·×4, bicubic은 data/bicubic_x{k} PNG 배열).
  manifest condition·split별 파일 SHA256·크기·픽셀 크기·로그 SHA256 대조.
저장: train data/<cond>/x<k>/bounding_box_train/<이름>.webp, 로그 runs/stage2_gen/_logs.
      test 실제 파일 data/locked/sr/<cond>/x<k>/<split>/, data/<cond>/x<k>/<split>는 그곳으로의 링크. 로그·manifest는 data/locked/sr/_logs, _manifest.
test 단계는 ALLOW_TEST=1(사용자 잠금 해제 중)에서만 돈다. 이미지는 열지 않고 수치로만 다룬다."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MF = Path("/workspace/miniforge3/envs")
TP = ROOT / "third_party"
ENV = {"swinir": "srreid-mambair", "pisa_sr": "srreid-pisasr", "osdhuman": "srreid-osdhuman", "bicubic": "srreid-core"}
REPO = {"swinir": "SwinIR", "pisa_sr": "PiSA-SR", "osdhuman": "OSDHuman", "bicubic": "none"}
COND = {"swinir": "swinir", "pisa_sr": "pisa_sr_sem0.0", "osdhuman": "osdhuman", "bicubic": "bicubic"}
DET = {"swinir": [], "pisa_sr": ["--det", "--warmup"], "osdhuman": ["--det", "--warmup"], "bicubic": []}
SHARDS = {("pisa_sr", 4): 12, ("osdhuman", 4): 12, ("swinir", 3): 3, ("swinir", 4): 3}  # 느린 확산 2종은 잘게
TRAIN, TESTS = "bounding_box_train", ("query", "bounding_box_test")
SERVER = "A100 서버(A100 SXM4 80GB ×2, EPYC 7713, cgroup v2)"
lock = threading.Lock()


def log(*a):
    with lock:
        print(f"[{time.strftime('%F %T')}]", *a, flush=True)


def roots(split):
    if split == TRAIN:
        return ROOT / "data", ROOT / "runs" / "stage2_gen" / "_logs"
    return ROOT / "data" / "locked" / "sr", ROOT / "data" / "locked" / "sr" / "_logs"


def names_of(split):
    d = ROOT / "data" / ("sr_input" if split == TRAIN else "locked/sr_input") / split
    return sorted(p[:-4] for p in os.listdir(d) if p.endswith(".png"))


def shards(n, s):
    b = [round(i * n / s) for i in range(s + 1)]
    return [(i, b[i], b[i + 1] - b[i]) for i in range(s)]


def run(cmd, gpu, logf, omp):
    env = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp),
               SRREID_SERVER=SERVER)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "w") as f:
        return subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)


def sr_cmd(m, k, split, out, logdir, ln, extra):
    py = MF / ENV[m] / "bin" / "python"
    base = [str(py), "stage1/sr_gen.py", "--method", m, "--k", str(k), "--split", split, "--out", str(out), "--log-dir", str(logdir), "--log-name", ln]
    if m == "pisa_sr":
        base += ["--lambda-sem", "0.0"]
    return ["env", f"PYTHONPATH={TP / REPO[m]}"] + base + extra


def done(logdir, ln):
    return (logdir / f"{ln}__summary.json").exists()


def phase_gen(split, P, notes):
    out, logdir = roots(split)
    n = len(names_of(split))
    if split != TRAIN:  # test: 실제 파일은 잠긴 영역, data/<cond>/x<k>/<split>는 링크
        for m in ("swinir", "pisa_sr", "osdhuman", "bicubic"):
            for k in (3, 4):
                real = out / COND[m] / f"x{k}" / split
                real.mkdir(parents=True, exist_ok=True)
                ln_ = ROOT / "data" / COND[m] / f"x{k}" / split
                ln_.parent.mkdir(parents=True, exist_ok=True)
                if not ln_.is_symlink():
                    ln_.symlink_to(real)
    gpu_jobs = []
    order = [("pisa_sr", 4), ("osdhuman", 4)]
    sp = {mk: shards(n, s) for mk, s in SHARDS.items()}
    for i in range(max(SHARDS[("pisa_sr", 4)], SHARDS[("osdhuman", 4)])):
        for mk in order:
            if i < len(sp[mk]):
                gpu_jobs.append((mk[0], mk[1], sp[mk][i]))
    for mk in (("swinir", 3), ("swinir", 4)):
        gpu_jobs += [(mk[0], mk[1], s) for s in sp[mk]]
    q = queue.Queue()
    for j in gpu_jobs:
        q.put(j)
    cpu = ThreadPoolExecutor(8)
    futs, fails = [], []

    def cpu_job(m, k, sh, ln, extra):
        if done(logdir, ln):
            return 0
        t = time.time()
        rc = run(sr_cmd(m, k, split, out, logdir, ln, extra), None, ROOT / "logs" / "stage2" / "gen" / f"{ln}.log", 1)
        log(f"종료 {ln} rc={rc} {time.time() - t:.0f}s (CPU)")
        if rc:
            fails.append(ln)
        return rc

    if split != TRAIN:
        for k in (3, 4):
            for (i, off, lim) in shards(n, 4):
                ln = f"bicubic__x{k}__{split}__s{i:02d}"
                futs.append(cpu.submit(cpu_job, "bicubic", k, (i, off, lim), ln, ["--offset", str(off), "--limit", str(lim)]))

    def worker(g, p):
        while True:
            try:
                m, k, (i, off, lim) = q.get_nowait()
            except queue.Empty:
                return
            ln = f"{COND[m]}__x{k}__{split}__s{i:02d}"
            if not done(logdir, ln):
                log(f"시작 {ln} gpu{g} ({lim}장)")
                t = time.time()
                rc = run(sr_cmd(m, k, split, out, logdir, ln, ["--offset", str(off), "--limit", str(lim)] + DET[m]), g,
                         ROOT / "logs" / "stage2" / "gen" / f"{ln}.log", 4)
                log(f"종료 {ln} gpu{g} rc={rc} {time.time() - t:.0f}s")
                if rc:
                    fails.append(ln)
                    continue
            else:
                log(f"[건너뜀] {ln}")
            with lock:
                notes.append({"job": ln, "gpu": g})
            if m in ("pisa_sr", "osdhuman") and k == 4:
                l3 = f"{COND[m]}__x3__{split}__s{i:02d}"
                futs.append(cpu.submit(cpu_job, m, 3, (i, off, lim), l3, ["--offset", str(off), "--limit", str(lim), "--from-x4"]))

    th = [threading.Thread(target=worker, args=(g, p)) for p in range(P) for g in (0, 1)]
    for t in th:
        t.start()
        time.sleep(2)
    for t in th:
        t.join()
    for f in futs:
        f.result()
    cpu.shutdown()
    return fails


def solo_lists(split):
    import numpy as np
    out, logdir = roots(split)
    n = len(names_of(split))
    names = names_of(split)
    d = logdir.parent / "_solo_lists" if split != TRAIN else ROOT / "runs" / "stage2_gen" / "_solo_lists"
    d.mkdir(parents=True, exist_ok=True)
    lists = {}
    for (m, k), s in SHARDS.items():
        sel = []
        for (i, off, lim) in shards(n, s):
            rng = np.random.default_rng(42 + i)
            sel += [names[off + j] for j in sorted(rng.choice(lim, 20, replace=False))]
        f = d / f"{COND[m]}__x{k}__{split}.txt"
        f.write_text("\n".join(sel) + "\n")
        lists[(m, k)] = f
    return lists


def sha_map(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        out[r["name"]] = r["sha1"]
    return out


def phase_solo(split):
    out, logdir = roots(split)
    sdir = logdir.parent / ("_solo" if split != TRAIN else "solo")
    if split == TRAIN:
        sdir = ROOT / "runs" / "stage2_gen" / "solo"
    lists = solo_lists(split)
    jobs = list(lists.items())
    lanes = {0: jobs[0::2], 1: jobs[1::2]}
    fails = []

    def lane(g):
        for (m, k), f in lanes[g]:
            ln = f"{COND[m]}__x{k}__{split}__solo"
            if done(sdir, ln):
                continue
            log(f"시작 {ln} gpu{g} (단독)")
            rc = run(sr_cmd(m, k, split, out, sdir, ln, ["--names-file", str(f), "--no-write"] + DET[m]), g, ROOT / "logs" / "stage2" / "gen" / f"{ln}.log", 4)
            log(f"종료 {ln} rc={rc}")
            if rc:
                fails.append(ln)
    th = [threading.Thread(target=lane, args=(g,)) for g in (0, 1)]
    [t.start() for t in th]
    [t.join() for t in th]
    rows = []
    for (m, k), f in lists.items():
        conc = {}
        for p in logdir.glob(f"{COND[m]}__x{k}__{split}__s*.jsonl"):
            conc.update(sha_map(p))
        solo = sha_map(sdir / f"{COND[m]}__x{k}__{split}__solo.jsonl")
        names = [l.strip() for l in open(f) if l.strip()]
        for i in range(SHARDS[(m, k)]):
            sel = names[i * 20:(i + 1) * 20]
            rows.append({"condition": COND[m], "k": k, "split": split, "shard": i, "n": len(sel), "n_bit_identical": sum(conc.get(x) == solo.get(x) and x in solo for x in sel)})
    return rows, fails


def phase_check():
    """Stage 1 표본 2,250장: Stage 2 train 출력 = Stage 1 출력(배열 SHA1). bicubic은 재사용하는 data/bicubic_x{k} PNG 배열과 Stage 1 bicubic 비교."""
    import numpy as np
    from PIL import Image
    s1 = ROOT / "runs" / "stage1_sample" / "_logs"
    _, logdir = roots(TRAIN)
    rows = []
    for m in ("swinir", "pisa_sr", "osdhuman", "bicubic"):
        for k in (3, 4):
            a = sha_map(s1 / f"{COND[m]}__x{k}.jsonl")
            if m == "bicubic":
                b = {}
                for n in a:
                    with Image.open(ROOT / "data" / f"bicubic_x{k}" / f"{n}.png") as im:
                        b[n] = hashlib.sha1(np.asarray(im.convert("RGB")).tobytes()).hexdigest()
                src = f"data/bicubic_x{k} (재사용 PNG)"
            else:
                b = {}
                for p in logdir.glob(f"{COND[m]}__x{k}__{TRAIN}__s*.jsonl"):
                    b.update(sha_map(p))
                src = f"data/{COND[m]}/x{k}/{TRAIN}"
            common = [n for n in a if n in b]
            rows.append({"condition": COND[m], "k": k, "stage2_source": src, "n_stage1": len(a), "n_compared": len(common),
                         "n_bit_identical": sum(a[n] == b[n] for n in common), "diff_names": " ".join(n for n in common if a[n] != b[n])[:2000]})
    return rows


def info(p):
    from PIL import Image
    b = p.read_bytes()
    with Image.open(p) as im:
        w, h = im.size
    return len(b), hashlib.sha256(b).hexdigest(), w, h


def phase_manifest(split):
    out, logdir = roots(split)
    names = names_of(split)
    mdir = (ROOT / "results" / "stage2" / "manifest") if split == TRAIN else (out / "_manifest")
    mdir.mkdir(parents=True, exist_ok=True)
    summ = []
    conds = [("swinir", 3), ("swinir", 4), ("pisa_sr", 3), ("pisa_sr", 4), ("osdhuman", 3), ("osdhuman", 4)] + ([] if split == TRAIN else [("bicubic", 3), ("bicubic", 4)])
    for m, k in conds:
        d = out / COND[m] / f"x{k}" / split
        logs = {}
        for p in logdir.glob(f"{COND[m]}__x{k}__{split}__s*.jsonl"):
            for l in open(p):
                r = json.loads(l)
                if "webp_sha256" in r:
                    logs[r["name"]] = r["webp_sha256"]
        ps = [d / f"{n}.webp" for n in names]
        ex = [p for p in ps if p.exists()]
        with ThreadPoolExecutor(16) as tp:
            infos = list(tp.map(info, ex))
        rows, mism, bad = [], 0, 0
        for p, (nb, sh, w, h) in zip(ex, infos):
            lm = logs.get(p.stem)
            ok = (lm == sh) if lm else None
            mism += ok is False
            bad += (w, h) != (64 * k, 128 * k)
            rows.append({"name": p.stem, "bytes": nb, "sha256": sh, "w": w, "h": h, "log_match": ok})
        with gzip.open(mdir / f"{COND[m]}__x{k}__{split}.csv.gz", "wt", newline="") as f:
            w_ = csv.DictWriter(f, ["name", "bytes", "sha256", "w", "h", "log_match"])
            w_.writeheader()
            w_.writerows(rows)
        tmp = len(list(d.glob("*.tmp")))
        summ.append({"condition": COND[m], "k": k, "split": split, "expected": len(names), "present": len(ex), "missing": len(names) - len(ex),
                     "wrong_size": bad, "log_sha_mismatch": mism, "log_sha_missing": sum(1 for p in ex if p.stem not in logs), "tmp_files": tmp,
                     "total_bytes": sum(i[0] for i in infos)})
        log(summ[-1])
    return summ


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--phase", nargs="+", default=["gen", "solo", "check", "manifest"])
    ap.add_argument("--procs-per-gpu", type=int, default=4)
    a = ap.parse_args()
    splits = [TRAIN] if a.split == "train" else list(TESTS)
    if a.split == "test" and os.environ.get("ALLOW_TEST") != "1":
        raise SystemExit("test 생성은 사용자가 잠금을 푼 뒤 ALLOW_TEST=1로만 실행한다")
    R = ROOT / "results" / "stage2"
    fails = []
    for ph in a.phase:
        if ph == "prep" and a.split == "test":  # 원본 test JPEG → 무손실 PNG(data/locked/sr_input/<split>), stage05/sr_generate.py와 같은 규칙. 있는 파일은 건너뜀
            sys.path.insert(0, str(ROOT))
            from stage05.sr_generate import prepare_input
            for s in splits:
                d = prepare_input(s)
                n = len([x for x in os.listdir(d) if x.endswith(".png")])
                log(f"입력 PNG {s}: {n}장")
                if n != {"query": 3368, "bounding_box_test": 19732}[s]:
                    raise SystemExit(f"{s} 입력 수 이상: {n}")
        elif ph == "gen":
            notes = []
            for s in splits:
                fails += phase_gen(s, a.procs_per_gpu, notes)
            if a.split == "train":
                write(R / "gen_gpu_assign_train.csv", notes)
            else:
                write(roots(TESTS[0])[1] / "gen_gpu_assign_test.csv", notes)
        elif ph == "solo":
            rows = []
            for s in splits:
                r, f = phase_solo(s)
                rows += r
                fails += f
            tgt = R / "gen_solo_bitcheck_train.csv" if a.split == "train" else roots(TESTS[0])[1] / "gen_solo_bitcheck_test.csv"
            write(tgt, rows)
            if a.split == "test":  # 결과 요약(장수·일치 수)만 잠금 밖에
                agg = {}
                for r in rows:
                    key = (r["condition"], r["k"], r["split"])
                    agg.setdefault(key, [0, 0, 0])
                    agg[key][0] += 1
                    agg[key][1] += r["n"]
                    agg[key][2] += r["n_bit_identical"]
                write(R / "gen_solo_bitcheck_test_summary.csv", [{"condition": c, "k": k, "split": s, "shards": v[0], "n": v[1], "n_bit_identical": v[2]} for (c, k, s), v in agg.items()])
            log("단독 비트 확인", sum(r["n_bit_identical"] for r in rows), "/", sum(r["n"] for r in rows))
        elif ph == "check" and a.split == "train":
            rows = phase_check()
            write(R / "gen_stage1_overlap_check.csv", rows)
            log("Stage 1 표본 비교", [(r["condition"], r["k"], r["n_bit_identical"], r["n_compared"]) for r in rows])
        elif ph == "manifest":
            summ = []
            for s in splits:
                summ += phase_manifest(s)
            write(R / f"manifest_summary_{a.split}.csv", summ)  # 장수·바이트·불일치 수만(파일별 목록은 train: results/stage2/manifest, test: 잠긴 영역)
            if any(r["missing"] or r["wrong_size"] or r["log_sha_mismatch"] or r["tmp_files"] for r in summ):
                fails.append("manifest")
    log("실패:", fails if fails else "없음")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
