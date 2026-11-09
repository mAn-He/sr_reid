"""Stage 2 평가 드라이버(2026-11-09). 실행 env: srreid-core. test 데이터를 쓰므로 ALLOW_TEST=1(사용자 승인 창)에서만 돈다.

평가 단위(unit) = (모델, k, 학습 조건, test 조건). (모델, k)마다 10개:
  C×bicubic(공유), C×s, S_s×s, S_s×bicubic (s ∈ swinir, pisa_sr_sem0.0, osdhuman).  C = bicubic 학습 모델, S_s = s 학습 모델.
unit 이름: <model>__x<k>__<train_cond>__on__<test_cond>   예) kpr__x4__swinir__on__bicubic
한 unit의 처리: query·gallery 특징 추출(stage2/eval_extract.py, 모델 env 하위 프로세스, 학습기의 --extract 재사용)
  → 지표(이 파일의 _metrics, 모델 env 하위 프로세스: 모델별 공식 거리 + stage05/eval_core 블록 스트리밍 Market CMC·AP)
  → per-query 저장 → 결과 표 다시 만들기(eval_cells.csv, 셀 4개가 모인 (모델,k,s)의 judgment.csv).
거리(각 모델의 공식 test와 같음, stage05 경량 평가기와 같은 경로):
  bot: 1 − cos (추출 때 L2 정규화된 BN 뒤 특징, eval/metrics_reid.py). clipreid: 공식 R1_mAP_eval(L2 정규화 + utils.metrics.euclidean_distance).
  kpr*: 공식 part_based_engine._evaluate(normalize_feature, compute_distance_matrix_using_bp_features, 설정은 test yaml/default_config).
저장: 잠긴 영역 data/locked/stage2_eval/<model>__x<k>/<train>__on__<test>/{per_query.csv.gz, metrics.json, feats/, logs/}
  (metrics.json이 완료 표시). 집계는 밖: results/stage2/eval_cells.csv, judgment.csv, bootstrap_meta.json.
재시작 안전: 완료 unit 건너뜀, split별 추출 완료(extract_meta.json)도 건너뜀. unit마다 파일 잠금이라 여러 프로세스가 같은 목록을 돌려도 겹치지 않는다.
memmap test 캐시(bot·kpr*용, $SRREID_CACHE/<cond>__x<k>__<split>.u8)는 잠금 아래 만들고, 이 드라이버가 만든 것만
  stage2_eval_created.txt에 적어 두었다가 그 캐시가 필요한 미완료 unit(학습 완료 모델 기준)이 없으면 지운다.

사용:
  ALLOW_TEST=1 python stage2/evaluate.py --list-units [--models ...] [--ks ...] [--sr ...]   (ALLOW_TEST 없이도 상태 일부 표시)
  ALLOW_TEST=1 python stage2/evaluate.py --run [--models ...] [--ks ...] [--only-ready] [--gpu 0]
  ALLOW_TEST=1 python stage2/evaluate.py --unit kpr__x4__swinir__on__swinir --gpu 1
  ALLOW_TEST=1 python stage2/evaluate.py --aggregate
  python stage2/evaluate.py --selftest-dist        (각 모델 env에서 공식 거리·평가 함수와 합성 특징으로 비교, GPU 수 초)
  python stage2/stats.py --selftest
"""
import argparse
import contextlib
import csv
import fcntl
import json
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 스크립트 폴더(stage2/)가 sys.path[0]이면 stage2/queue.py가 표준 라이브러리 queue를 가린다(DataLoader·urllib3가 깨짐). 뺀다.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()

import numpy as np  # noqa: E402

MODELS = ("bot", "clipreid", "kpr", "kpr_solider")
SR_CONDS = ("swinir", "pisa_sr_sem0.0", "osdhuman")
KS = (3, 4)
WEIGHT = {"bot": "model.pt", "clipreid": "model.pth", "kpr": "model.pth.tar", "kpr_solider": "model.pth.tar"}
MF = Path("/workspace/miniforge3/envs")
ENVPY = {"bot": MF / "srreid-core/bin/python", "clipreid": MF / "srreid-clipreid/bin/python",
         "kpr": MF / "srreid-kpr/bin/python", "kpr_solider": MF / "srreid-kpr/bin/python"}
USES_MEMMAP = {"bot": True, "clipreid": False, "kpr": True, "kpr_solider": True}
TEST_SPLITS = (("query", "query"), ("gallery", "bounding_box_test"))  # (특징 폴더 이름, 데이터 split)
EXPECTED_FILES = {"query": 3368, "bounding_box_test": 19732}           # 파일 수(junk 포함)
EXPECTED_ROWS = {"query": 3368, "gallery": 15913}                      # pid -1 제외 후(stage05/items.py N_OFFICIAL)
BLOCK = 512
KPR_PAIRWISE = 500  # torchreid default_config cfg.test.batch_size_pairwise_dist_matrix (gallery 분할 크기일 뿐 값과 무관)
TRAIN_ROOT = ROOT / "runs" / "stage2" / "train"
EVAL_ROOT = ROOT / "data" / "locked" / "stage2_eval"
RESULTS = ROOT / "results" / "stage2"
CELL_FIELDS = ("model", "k", "train_cond", "test_cond", "mAP", "R1", "R5", "R10", "n_query", "n_query_valid", "n_gallery",
               "seconds", "seconds_extract", "seconds_metrics", "gpu_peak_alloc_gib", "peak_rss_gib", "server", "conc_note", "finished")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------ unit 목록
class Unit:
    def __init__(self, model, k, train, test):
        self.model, self.k, self.train, self.test = model, k, train, test

    @property
    def id(self):
        return f"{self.model}__x{self.k}__{self.train}__on__{self.test}"

    @classmethod
    def parse(cls, s):
        m = re.fullmatch(r"(bot|clipreid|kpr|kpr_solider)__x([34])__(.+)__on__(.+)", s)
        if not m:
            raise SystemExit(f"unit 이름 형식이 아님: {s}")
        u = cls(m.group(1), int(m.group(2)), m.group(3), m.group(4))
        conds = ("bicubic",) + SR_CONDS
        if u.train not in conds or u.test not in conds or (u.train != "bicubic" and u.test not in ("bicubic", u.train)):
            raise SystemExit(f"설계에 없는 unit: {s}")
        return u

    def train_dir(self, cfg):
        return cfg.train_root / f"{self.model}__{self.train}__x{self.k}"

    def weights(self, cfg):
        return self.train_dir(cfg) / WEIGHT[self.model]

    def model_ready(self, cfg):
        d = self.train_dir(cfg)
        return (d / "DONE").exists() and self.weights(cfg).exists()

    def dir(self, cfg):
        return cfg.eval_root / f"{self.model}__x{self.k}" / f"{self.train}__on__{self.test}"

    def done(self, cfg):
        d = self.dir(cfg)
        return (d / "metrics.json").exists() and (d / "per_query.csv.gz").exists()

    def caches(self, cfg):
        if not USES_MEMMAP[self.model]:
            return set()
        return {(self.test, self.k, cfg.split_map[tag][0]) for tag, _ in TEST_SPLITS}


def all_units(models=MODELS, ks=KS, srs=SR_CONDS):
    out, seen = [], set()
    for k in ks:
        for m in models:
            cells = [("bicubic", "bicubic")]
            for s in srs:
                cells += [("bicubic", s), (s, s), (s, "bicubic")]
            for tr, te in cells:
                u = Unit(m, k, tr, te)
                if u.id not in seen:
                    seen.add(u.id)
                    out.append(u)
    # 캐시를 빨리 정리할 수 있게 (k, test 조건)으로 모은다
    order = {c: i for i, c in enumerate(("bicubic",) + SR_CONDS)}
    return sorted(out, key=lambda u: (u.k, order[u.test], MODELS.index(u.model), order[u.train]))


# ------------------------------------------------------------------ memmap test 캐시
def cache_files(cond, k, split):
    from stage05 import cache as C
    base = C.cache_path(cond, k, split)
    return base.with_name(base.name + ".u8"), base.with_name(base.name + ".json"), base.with_name(base.name + ".lock")


def registry():
    from stage05 import cache as C
    return Path(C.CACHE) / "stage2_eval_created.txt"


def cache_valid(cond, k, split, expected, newest):
    from stage05 import cache as C
    data, meta, _ = cache_files(cond, k, split)
    if not (data.exists() and meta.exists()):
        return False
    m = json.loads(meta.read_text())
    shape = tuple(m["shape"])
    return (m.get("source") == str(C.source_dir(cond, k, split)) and (expected is None or len(m["names"]) == expected)
            and shape[1:] == (128 * k, 64 * k, 3) and data.stat().st_size == int(np.prod(shape)) and data.stat().st_mtime >= newest)


@contextlib.contextmanager
def cache_in_use(cfg, cond, k, split):
    """캐시를 (필요하면) 만들고, 쓰는 동안 공유 잠금을 쥔다. 정리는 배타 잠금을 비차단으로 얻을 때만 지운다."""
    from stage05 import cache as C
    data, meta, lockp = cache_files(cond, k, split)
    if split == "bounding_box_train":  # 개발 시험 전용: 학습용 train 캐시는 만들지도, 잠그지도, 지우지도 않고 있는 것만 읽는다
        if not cfg.dev or not (data.exists() and meta.exists()):
            raise RuntimeError(f"train 캐시 없음(평가 드라이버는 만들지 않음): {data}")
        yield
        return
    Path(C.CACHE).mkdir(parents=True, exist_ok=True)
    expected = EXPECTED_FILES[split]
    lk = open(lockp, "a+")
    try:
        while True:
            fcntl.flock(lk, fcntl.LOCK_EX)
            src = C.source_dir(cond, k, split)
            files = [p for p in src.iterdir() if p.suffix in (".webp", ".png", ".jpg")]
            if expected is not None and len(files) != expected:
                raise RuntimeError(f"test 데이터 불완전: {src} {len(files)}장 (필요 {expected})")
            newest = max(p.stat().st_mtime for p in files)
            if not cache_valid(cond, k, split, expected, newest):
                h, w = 128 * k, 64 * k
                need = len(files) * h * w * 3
                free = shutil.disk_usage(C.CACHE).free
                if free < need + 2 * 2**30:
                    raise RuntimeError(f"캐시 공간 부족: {C.CACHE} 남음 {free / 2**30:.1f}GiB, 필요 {need / 2**30:.1f}GiB + 2GiB")
                with open(registry(), "a") as f:
                    f.write(f"{cond}\t{k}\t{split}\n")
                t0 = time.perf_counter()
                info = C.build(cond, k, split, workers=cfg.cache_workers, force=True)
                print(f"[{now()}] 캐시 생성 {cond} x{k} {split}: {info['bytes'] / 2**30:.2f}GiB {time.perf_counter() - t0:.0f}s", flush=True)
                if not cache_valid(cond, k, split, expected, newest):
                    raise RuntimeError(f"캐시 검증 실패: {data}")
            fcntl.flock(lk, fcntl.LOCK_SH)  # 강등(원자적이지 않음) → 다시 확인
            if data.exists() and meta.exists():
                break
            fcntl.flock(lk, fcntl.LOCK_UN)
        yield
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def created_caches():
    p = registry()
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().split("\n"):
        if line.strip():
            c, k, s = line.split("\t")
            out.add((c, int(k), s))
    return out


def cleanup_caches(cfg, force=False):
    """이 드라이버가 만든 test 캐시 중, 필요한 미완료 unit(학습 완료 모델)이 없고 아무도 쓰지 않는 것을 지운다."""
    from stage05 import cache as C
    created = created_caches()
    if not created:
        return []
    needed = set()
    if not force:
        for u in all_units():
            if u.model_ready(cfg) and not u.done(cfg):
                needed |= u.caches(cfg)
    removed = []
    for key in sorted(created):
        if key in needed or key[2] == "bounding_box_train":
            continue
        data, meta, lockp = cache_files(*key)
        if not (data.exists() or meta.exists()):
            continue
        with open(lockp, "a+") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            C.remove(*key)
            fcntl.flock(lk, fcntl.LOCK_UN)
        removed.append(key)
        print(f"[{now()}] 캐시 삭제 {key}", flush=True)
    still = {k for k in created if any(p.exists() for p in cache_files(*k)[:2])}
    with open(registry(), "w") as f:
        f.writelines(f"{c}\t{k}\t{s}\n" for c, k, s in sorted(still))
    return removed


# ------------------------------------------------------------------ 하위 프로세스
def child_env(cfg):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    if cfg.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    return env


def run_child(cmd, log, env):
    """하위 프로세스를 돌리고 (종료 코드, 초, 최대 RSS GiB(그 프로세스만, DataLoader 작업자 제외))."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log, "a") as f:
        f.write(f"\n[{now()}] {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, time.perf_counter() - t0, ru.ru_maxrss / 2**20


def check_gpu_env(cfg):
    vis = str(cfg.gpu) if cfg.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None or vis == "" or "," in vis:
        raise SystemExit("GPU를 정확히 1장 지정해야 함: --gpu N 또는 CUDA_VISIBLE_DEVICES=N (CLIP-ReID 추출이 1장을 요구)")


def run_unit(cfg, u):
    """한 unit을 끝까지 처리. 결과: 'done' | 'skip-done' | 'skip-locked' | 'skip-not-ready' | 'failed'."""
    if u.done(cfg):
        return "skip-done"
    if not u.model_ready(cfg):
        return "skip-not-ready"
    d = u.dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    lk = open(d / ".lock", "a+")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lk.close()
        return "skip-locked"
    try:
        if u.done(cfg):
            return "skip-done"
        env = child_env(cfg)
        t_unit = time.perf_counter()
        print(f"[{now()}] unit 시작 {u.id} (GPU {env.get('CUDA_VISIBLE_DEVICES')})", flush=True)
        run_log = []
        for tag, split in TEST_SPLITS:
            fdir = d / "feats" / tag
            if (fdir / "extract_meta.json").exists():
                continue
            data_split, names_file = cfg.split_map[tag]
            cmd = [ENVPY[u.model], ROOT / "stage2" / "eval_extract.py", "--model", u.model, "--weights", u.weights(cfg),
                   "--cond", u.test, "--k", u.k, "--split", data_split, "--out", d / "feats", "--tag", tag]
            if names_file:
                cmd += ["--names-file", names_file]
            ctx = cache_in_use(cfg, u.test, u.k, data_split) if USES_MEMMAP[u.model] else contextlib.nullcontext()
            with ctx:
                rc, sec, rss = run_child(cmd, d / "logs" / f"extract_{tag}.log", env)
            if rc != 0:
                print(f"[{now()}] 추출 실패 {u.id} {tag} rc={rc}, 로그 {d / 'logs' / f'extract_{tag}.log'}", flush=True)
                return "failed"
            run_log.append({"step": f"extract_{tag}", "seconds": round(sec, 2), "maxrss_gib": round(rss, 3)})
        cmd = [ENVPY[u.model], ROOT / "stage2" / "evaluate.py", "_metrics", "--model", u.model, "--unit-dir", d,
               "--device", cfg.metrics_device] + (["--dev"] if cfg.dev else [])
        rc, sec, rss = run_child(cmd, d / "logs" / "metrics.log", env)
        if rc != 0:
            print(f"[{now()}] 지표 실패 {u.id} rc={rc}, 로그 {d / 'logs' / 'metrics.log'}", flush=True)
            return "failed"
        run_log.append({"step": "metrics", "seconds": round(sec, 2), "maxrss_gib": round(rss, 3)})
        m = json.loads((d / "metrics.json").read_text())
        m.update(unit=u.id, model=u.model, k=u.k, train_cond=u.train, test_cond=u.test, weights=str(u.weights(cfg)),
                 run_log=m.get("run_log", []) + run_log, wall_seconds_this_process=round(time.perf_counter() - t_unit, 1))
        atomic_write(d / "metrics.json", json.dumps(m, indent=1, ensure_ascii=False))
        if cfg.drop_features:
            shutil.rmtree(d / "feats", ignore_errors=True)
        print(f"[{now()}] unit 완료 {u.id}: mAP {m['mAP']:.4f} R1 {m['rank1']:.4f} ({time.perf_counter() - t_unit:.0f}s)", flush=True)
        return "done"
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def load_stats():
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage2_stats", ROOT / "stage2" / "stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 지표 (모델 env 하위 프로세스)
def metrics_worker(a):
    import torch
    from stage05 import eval_core
    stats = load_stats()
    d = Path(a.unit_dir)
    t0 = time.perf_counter()
    Q, G = eval_core.load_split(d / "feats" / "query"), eval_core.load_split(d / "feats" / "gallery")
    mq = json.loads((d / "feats" / "query" / "extract_meta.json").read_text())
    mg = json.loads((d / "feats" / "gallery" / "extract_meta.json").read_text())
    for key in ("model", "weights", "cond", "k"):
        if mq[key] != mg[key]:
            raise SystemExit(f"query·gallery 추출 설정 불일치: {key} {mq[key]} vs {mg[key]}")
    if not a.dev:
        for tag, X in (("query", Q), ("gallery", G)):
            if len(X["pid"]) != EXPECTED_ROWS[tag] or (X["pid"] == -1).any():
                raise SystemExit(f"{tag} 행 수 {len(X['pid'])} != {EXPECTED_ROWS[tag]} 또는 junk 포함")
    dev = a.device
    if dev == "auto":
        dev = "cuda" if (a.model in ("bot", "kpr", "kpr_solider") and torch.cuda.is_available()) else "cpu"
    dist_block, info = make_dist(a.model, Q, G, dev, mq)
    res = eval_core.stream_eval(Q["pid"], Q["cam"], G["pid"], G["cam"], dist_block, BLOCK)
    s = eval_core.summary(res)
    stats.write_per_query(d / "per_query.csv.gz", Q["names"], Q["pid"], Q["cam"], res["valid"], res["ap"], res["cmc"])
    peaks = [x for x in (mq.get("gpu_peak_alloc_gib"), mg.get("gpu_peak_alloc_gib")) if x is not None]
    gpu = max(peaks) if peaks else None
    out = {"mAP": s["mAP"], "rank1": s["rank1"], "rank5": s["rank5"], "rank10": s["rank10"], "n_query": s["n_query"],
           "n_query_valid": s["n_valid"], "n_gallery": int(len(G["pid"])), "block": BLOCK, "metrics_device": dev, "distance": info,
           "seconds_extract": round(mq["seconds"] + mg["seconds"], 2), "seconds_metrics": round(time.perf_counter() - t0, 2),
           "gpu_peak_alloc_gib_extract": gpu, "extract_query": mq, "extract_gallery": mg, "server": server(), "finished": now(),
           "conc_note": os.environ.get("SRREID_CONC_NOTE")}
    atomic_write(d / "metrics.json", json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("n_query", "n_query_valid", "n_gallery", "seconds_metrics")}), flush=True)


def make_dist(model, Q, G, dev, meta):
    """모델별 공식 test 거리. dist_block(i0, i1) → (i1−i0)×G numpy."""
    import torch
    if model == "bot":
        # eval/metrics_reid.py: dist = 1 − g·q (특징은 추출 때 이미 L2 정규화, 다시 정규화하지 않음)
        if not meta.get("l2_normalized_in_extract"):
            raise SystemExit("BoT 특징이 정규화되지 않음")
        qf = torch.from_numpy(np.array(Q["feat"], np.float32)).to(dev)
        gf = torch.from_numpy(np.array(G["feat"], np.float32)).to(dev)

        def dist_block(i0, i1):
            with torch.no_grad():
                return (1 - qf[i0:i1] @ gf.T).cpu().numpy()
        return dist_block, {"kind": "1 - cosine (features L2-normalized at extraction)", "device": dev}
    if model == "clipreid":
        sys.path.insert(0, str(ROOT / "third_party" / "CLIP-ReID"))
        from utils.metrics import euclidean_distance  # 공식 거리 함수
        tm = meta["trainer_meta"]
        if tm.get("feat_norm_in_eval") != "yes":
            raise SystemExit(f"CLIP-ReID TEST.FEAT_NORM이 yes가 아님: {tm.get('feat_norm_in_eval')}")
        qf = torch.nn.functional.normalize(torch.from_numpy(np.array(Q["feat"])), dim=1, p=2)
        gf = torch.nn.functional.normalize(torch.from_numpy(np.array(G["feat"])), dim=1, p=2)
        if dev != "cpu":
            qf, gf = qf.to(dev), gf.to(dev)

        def dist_block(i0, i1):
            return euclidean_distance(qf[i0:i1], gf)
        return dist_block, {"kind": "R1_mAP_eval: L2 normalize + euclidean_distance", "neck_feat": tm.get("neck_feat"), "device": dev}
    # kpr, kpr_solider
    sys.path.insert(0, str(ROOT / "third_party" / "keypoint_promptable_reidentification"))
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features as fn
    tm = meta["trainer_meta"]
    combine, metric, norm = tm["combine"], tm["dist_metric"], tm["normalize_feature"]
    qf = torch.from_numpy(np.array(Q["feat"]))
    gf = torch.from_numpy(np.array(G["feat"]))
    if norm:  # engine.normalize = F.normalize(p=2, dim=-1)
        qf = torch.nn.functional.normalize(qf, p=2, dim=-1)
        gf = torch.nn.functional.normalize(gf, p=2, dim=-1)
    qv, gv = torch.from_numpy(np.array(Q["vis"])), torch.from_numpy(np.array(G["vis"]))
    use_gpu = dev != "cpu"

    def dist_block(i0, i1):
        with torch.no_grad():
            dm = fn(qf[i0:i1], gf, qv[i0:i1], gv, combine, KPR_PAIRWISE, use_gpu, metric, use_logger=False)[0]
        return dm.float().numpy() if torch.is_tensor(dm) else np.asarray(dm, np.float32)
    return dist_block, {"kind": "compute_distance_matrix_using_bp_features", "normalize_feature": bool(norm), "dist_metric": metric,
                        "dist_combine_strat": combine, "batch_size_pairwise_dist_matrix": KPR_PAIRWISE, "use_gpu": use_gpu,
                        "test_embeddings": tm.get("test_embeddings"), "vis_dtype": meta.get("vis_dtype")}


def server():
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i",
                              (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return os.environ.get("SRREID_SERVER") or f"{socket.gethostname()} {gpu or ''}".strip()


# ------------------------------------------------------------------ 집계
def aggregate(cfg):
    stats = load_stats()
    cfg.results.mkdir(parents=True, exist_ok=True)
    with open(cfg.results / ".aggregate.lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        units = all_units()
        cells = []
        done = {}
        for u in units:
            if not u.done(cfg):
                continue
            m = json.loads((u.dir(cfg) / "metrics.json").read_text())
            done[u.id] = u
            ex = [x for x in (m.get("extract_query", {}), m.get("extract_gallery", {}))]
            rss = [r.get("maxrss_gib") for r in m.get("run_log", []) if r.get("maxrss_gib") is not None]
            cells.append({"model": u.model, "k": u.k, "train_cond": u.train, "test_cond": u.test, "mAP": m["mAP"], "R1": m["rank1"],
                          "R5": m["rank5"], "R10": m["rank10"], "n_query": m["n_query"], "n_query_valid": m["n_query_valid"],
                          "n_gallery": m["n_gallery"], "seconds": round(m["seconds_extract"] + m["seconds_metrics"], 1),
                          "seconds_extract": m["seconds_extract"], "seconds_metrics": m["seconds_metrics"],
                          "gpu_peak_alloc_gib": m.get("gpu_peak_alloc_gib_extract"), "peak_rss_gib": max(rss) if rss else None,
                          "server": m.get("server"), "conc_note": m.get("conc_note") or ex[0].get("conc_note"), "finished": m.get("finished")})
        write_csv(cfg.results / "eval_cells.csv", CELL_FIELDS, cells)
        # judgment: 셀 4개가 모인 (모델, k, s)
        rows, counts, ids, sha, cache = [], None, None, None, {}

        def pq(u):
            if u.id not in cache:
                cache[u.id] = stats.read_per_query(u.dir(cfg) / "per_query.csv.gz")
            return cache[u.id]
        for m in MODELS:
            for k in KS:
                for s in SR_CONDS:
                    need = {(tr, te): Unit(m, k, tr, te) for tr, te in (("bicubic", "bicubic"), ("bicubic", s), (s, s), (s, "bicubic"))}
                    if not all(u.id in done for u in need.values()):
                        continue
                    cells_pq = {key: pq(u) for key, u in need.items()}
                    base = cells_pq[("bicubic", "bicubic")]
                    if counts is None:
                        ids = stats.query_ids(base)
                        counts, sha = stats.resample_counts(ids)
                    elif not np.array_equal(stats.query_ids(base), ids):
                        raise SystemExit(f"query 신원 집합이 셀마다 다름: {m} x{k}")
                    rows += stats.judge(m, k, s, cells_pq, counts, ids, sha)
        write_csv(cfg.results / "judgment.csv", stats.JUDGMENT_FIELDS, rows)
        if counts is not None:
            atomic_write(cfg.results / "bootstrap_meta.json", json.dumps({
                "n_rep": int(counts.shape[0]), "seed": stats.SEED, "n_id": int(len(ids)), "resample_sha256": sha,
                "method": "identity-level paired bootstrap over query pids, np.random.default_rng(42).integers(0, n_id, (n_rep, n_id)); "
                          "replicate metric = mean over all valid queries of sampled identities with multiplicity; percentile 95% CI (linear)",
                "updated": now()}, indent=1))
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"[{now()}] 집계: 셀 {len(cells)}개 → {cfg.results / 'eval_cells.csv'}, 판정 행 {len(rows)}개 → {cfg.results / 'judgment.csv'}", flush=True)


def write_csv(path, fields, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    os.replace(tmp, path)


# ------------------------------------------------------------------ 거리 자체 시험 (모델 env 하위 프로세스)
def selftest_dist_worker(a):
    """합성 특징으로 '블록 거리 + eval_core.stream_eval'과 각 모델 공식 경로(전체 거리 행렬 + 공식 평가 함수)를 비교한다."""
    import torch
    from stage05 import eval_core
    rng = np.random.default_rng(0)
    nq, ng, n_id = 97, 611, 30
    q_pid, g_pid = rng.integers(1, n_id + 1, nq), rng.integers(0, n_id + 1, ng)
    q_cam, g_cam = rng.integers(0, 6, nq), rng.integers(0, 6, ng)
    q_pid[:2] = 500  # 양성 없는 query
    dev = "cuda" if torch.cuda.is_available() and a.model != "clipreid" else "cpu"
    if a.model == "kpr" or a.model == "kpr_solider":
        P, D = 6, 64
        centers = rng.normal(size=(n_id + 1, P, D))
        qf = (centers[q_pid % (n_id + 1)] + rng.normal(scale=4.0, size=(nq, P, D))).astype(np.float32)
        gf = (centers[g_pid] + rng.normal(scale=4.0, size=(ng, P, D))).astype(np.float32)
        qv, gv = rng.random((nq, P)) > 0.15, rng.random((ng, P)) > 0.15
        qv[:, 0], gv[:, 0] = True, True  # 공식 test 임베딩은 bn_foreg가 첫 자리(항상 보임)
        qv[5, 0], gv[7, 0] = False, False
        qv[5, 1:] = False  # 모든 쌍이 무효인 query(채움값 경로)
        meta = {"trainer_meta": {"combine": "mean", "dist_metric": "euclidean", "normalize_feature": True}, "vis_dtype": "bool"}
        Q = {"feat": qf, "vis": qv, "pid": q_pid, "cam": q_cam}
        G = {"feat": gf, "vis": gv, "pid": g_pid, "cam": g_cam}
        sys.path.insert(0, str(ROOT / "third_party" / "keypoint_promptable_reidentification"))
        from torchreid.metrics.distance import compute_distance_matrix_using_bp_features as fn
        from torchreid import metrics
        nq_t = torch.nn.functional.normalize(torch.from_numpy(qf), p=2, dim=-1)
        ng_t = torch.nn.functional.normalize(torch.from_numpy(gf), p=2, dim=-1)
        dm = fn(nq_t, ng_t, torch.from_numpy(qv), torch.from_numpy(gv), "mean", KPR_PAIRWISE, dev != "cpu", "euclidean", use_logger=False)[0].numpy()
        r = metrics.evaluate_rank(dm, q_pid, g_pid, q_cam, g_cam, max_rank=20, use_cython=False)
        ref_ap, ref_cmc, ref_map = np.asarray(r["all_AP"]), np.asarray(r["cmc"]), float(r["mAP"])
    else:
        D = 2048 if a.model == "bot" else 1280
        centers = rng.normal(size=(n_id + 1, D))
        qf = (centers[q_pid % (n_id + 1)] + rng.normal(scale=9.0, size=(nq, D))).astype(np.float32)
        gf = (centers[g_pid] + rng.normal(scale=9.0, size=(ng, D))).astype(np.float32)
        if a.model == "bot":
            qf /= np.linalg.norm(qf, axis=1, keepdims=True)
            gf /= np.linalg.norm(gf, axis=1, keepdims=True)
            meta = {"l2_normalized_in_extract": True}
            from eval.metrics_reid import evaluate_query
            vi = [i for i in range(nq) if ((g_pid == q_pid[i]) & (g_cam != q_cam[i])).any()]
            ref_ap = np.array([evaluate_query(qf[i], q_pid[i], q_cam[i], gf, g_pid, g_cam)[0] for i in vi])
            ref_cmc, ref_map = None, float(ref_ap.mean())
        else:
            meta = {"trainer_meta": {"feat_norm_in_eval": "yes", "neck_feat": "before"}}
            sys.path.insert(0, str(ROOT / "third_party" / "CLIP-ReID"))
            from utils.metrics import R1_mAP_eval
            ev = R1_mAP_eval(nq, max_rank=50, feat_norm="yes")
            ev.reset()
            ev.update((torch.from_numpy(np.concatenate([qf, gf])), np.concatenate([q_pid, g_pid]), np.concatenate([q_cam, g_cam])))
            captured = {}

            def prof(frame, event, arg):
                if event == "return" and frame.f_code.co_name == "eval_func":
                    captured["ap"] = np.array(frame.f_locals["all_AP"])
            sys.setprofile(prof)
            ref_cmc, ref_map = ev.compute()[:2]
            sys.setprofile(None)
            ref_ap = captured["ap"]
        Q = {"feat": qf, "vis": None, "pid": q_pid, "cam": q_cam}
        G = {"feat": gf, "vis": None, "pid": g_pid, "cam": g_cam}
    dist_block, info = make_dist(a.model, Q, G, dev, meta)
    ok = True
    for block in (1, 13, 64, nq):
        res = eval_core.stream_eval(q_pid, q_cam, g_pid, g_cam, dist_block, block)
        ap = res["ap"][res["valid"]]
        same = len(ap) == len(ref_ap) and np.allclose(ap, ref_ap, rtol=0, atol=1e-9) and abs(res["mAP"] - ref_map) < 1e-9
        if ref_cmc is not None:
            n = min(len(ref_cmc), 20)
            same = same and np.allclose(res["cmc_curve"][:n], np.asarray(ref_cmc)[:n], atol=1e-6)
        ok &= bool(same)
        print(f"{'PASS' if same else 'FAIL'} {a.model} block={block} device={dev}: 경량 mAP {res['mAP']:.6f} / 공식 {ref_map:.6f}, "
              f"유효 query {res['n_valid']}/{nq}, AP 최대 차 {np.abs(ap - ref_ap).max() if len(ap) == len(ref_ap) else 'len mismatch'}", flush=True)
    sys.exit(0 if ok else 1)


def selftest_dist(cfg):
    env = child_env(cfg)
    ok = True
    for m in ("bot", "clipreid", "kpr"):
        r = subprocess.run([str(ENVPY[m]), str(ROOT / "stage2" / "evaluate.py"), "_selftest_dist", "--model", m], env=env, cwd=str(ROOT))
        ok &= r.returncode == 0
    print("거리 자체 시험:", "모두 통과" if ok else "실패 있음", flush=True)
    return ok


# ------------------------------------------------------------------ main
class Cfg:
    pass


def make_cfg(a):
    cfg = Cfg()
    cfg.train_root = Path(a.train_root)
    cfg.eval_root = Path(a.eval_root)
    cfg.results = Path(a.results_dir)
    cfg.gpu = a.gpu
    cfg.metrics_device = a.metrics_device
    cfg.drop_features = a.drop_features
    cfg.cache_workers = a.cache_workers
    cfg.dev = a.dev_pseudo is not None
    if cfg.dev:  # 개발 전용: train split의 이름 목록을 가짜 query/gallery로 쓴다(test 데이터 없음)
        nd = Path(a.dev_pseudo)
        cfg.split_map = {"query": ("bounding_box_train", str(nd / "query.txt")), "gallery": ("bounding_box_train", str(nd / "gallery.txt"))}
    else:
        cfg.split_map = {tag: (split, None) for tag, split in TEST_SPLITS}
    return cfg


def require_test_window(cfg):
    if cfg.dev:
        return
    if os.environ.get("ALLOW_TEST") != "1":
        raise SystemExit("거부: Stage 2 평가는 test split을 읽는다. 사용자가 승인한 test 창에서 ALLOW_TEST=1로만 실행한다.\n"
                         "  (test 데이터 없이 할 수 있는 것: --list-units, --selftest-dist, python stage2/stats.py --selftest)")


def list_units(cfg, units):
    can = os.environ.get("ALLOW_TEST") == "1" or cfg.dev
    for u in units:
        if not u.model_ready(cfg):
            st = "waiting_model"
        elif not can:
            st = "ready_or_done(locked)"
        elif u.done(cfg):
            st = "done"
        elif (u.dir(cfg) / ".lock").exists() and _locked(u.dir(cfg) / ".lock"):
            st = "running"
        else:
            st = "ready"
        print(f"{u.id}\t{st}\t{len(u.caches(cfg)) and 'memmap' or 'files'}", flush=True)


def _locked(p):
    with open(p, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("_metrics", "_selftest_dist"):
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("--model", required=True, choices=MODELS)
        ap.add_argument("--unit-dir")
        ap.add_argument("--device", default="auto")
        ap.add_argument("--dev", action="store_true")
        a = ap.parse_args()
        return metrics_worker(a) if a.mode == "_metrics" else selftest_dist_worker(a)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-units", action="store_true", help="unit 목록: <unit>\\t<상태>\\t<memmap|files>")
    g.add_argument("--run", action="store_true", help="고른 unit을 차례로 처리(다른 프로세스가 잡은 unit은 건너뜀)")
    g.add_argument("--unit", nargs="+", help="이 unit(들)만 처리")
    g.add_argument("--aggregate", action="store_true", help="eval_cells.csv, judgment.csv만 다시 만든다")
    g.add_argument("--cleanup-caches", action="store_true", help="이 드라이버가 만든 test 캐시를 지운다(쓰는 중인 것 제외)")
    g.add_argument("--selftest-dist", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS), choices=KS)
    ap.add_argument("--sr", nargs="+", default=list(SR_CONDS), choices=SR_CONDS)
    ap.add_argument("--only-ready", action="store_true", help="학습이 끝나지 않은 모델의 unit은 건너뜀(없으면 하나라도 안 끝났을 때 멈춤)")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES로 넘길 GPU 번호 1개(없으면 환경 변수)")
    ap.add_argument("--metrics-device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="auto: bot·kpr은 cuda, clipreid는 cpu(stage05 경량 평가 검증과 같은 장치)")
    ap.add_argument("--drop-features", action="store_true", help="지표 뒤 특징 파일 삭제(per-query는 남김). 기본은 잠긴 영역에 보관")
    ap.add_argument("--keep-caches", action="store_true", help="unit 뒤 test 캐시 정리를 하지 않음")
    ap.add_argument("--cache-workers", type=int, default=8)
    ap.add_argument("--train-root", default=str(TRAIN_ROOT))
    ap.add_argument("--eval-root", default=str(EVAL_ROOT), help=argparse.SUPPRESS)
    ap.add_argument("--results-dir", default=str(RESULTS), help=argparse.SUPPRESS)
    ap.add_argument("--dev-pseudo", default=None, help=argparse.SUPPRESS)  # 개발 시험 전용(train split 가짜 query/gallery)
    a = ap.parse_args()
    cfg = make_cfg(a)
    if a.selftest_dist:
        sys.exit(0 if selftest_dist(cfg) else 1)
    units = [u for u in all_units(a.models, a.ks, a.sr)]
    if a.list_units:
        return list_units(cfg, units)
    require_test_window(cfg)
    if a.aggregate:
        return aggregate(cfg)
    if a.cleanup_caches:
        print(cleanup_caches(cfg, force=True))
        return
    if a.unit:
        units = [Unit.parse(s) for s in a.unit]
    not_ready = [u.id for u in units if not u.model_ready(cfg)]
    if not_ready and not a.only_ready:
        raise SystemExit("학습이 끝나지 않은 모델의 unit이 있음(--only-ready로 건너뛸 수 있음):\n  " + "\n  ".join(not_ready))
    check_gpu_env(cfg)
    summary = {}
    for u in units:
        try:
            r = run_unit(cfg, u)
        except Exception as e:  # 한 unit 실패가 나머지를 막지 않게
            print(f"[{now()}] unit 오류 {u.id}: {type(e).__name__}: {e}", flush=True)
            r = "failed"
        summary[u.id] = r
        if r == "done":
            if not a.keep_caches:
                cleanup_caches(cfg)
            aggregate(cfg)
    if not a.keep_caches:
        cleanup_caches(cfg)
    counts = {}
    for r in summary.values():
        counts[r] = counts.get(r, 0) + 1
    print(f"[{now()}] 끝: {counts}", flush=True)
    for uid, r in summary.items():
        if r == "failed":
            print(f"  실패: {uid}", flush=True)
    sys.exit(1 if "failed" in counts else 0)


if __name__ == "__main__":
    main()
