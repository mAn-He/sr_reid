"""Stage 2 판정 통계(2026-11-09 사용자 결정, CLAUDE.md "Stage 2 판정 규칙"). numpy와 표준 라이브러리만 쓴다.

- 셀 지표: mAP = 유효 query AP의 평균(공식 Market 규칙), R1 = 유효 query rank-1 적중의 평균.
- 신원 단위 쌍 부트스트랩: query 신원(유효 query의 pid 고유값, 정렬)을 복원 추출로 n_id개 뽑는다. 10,000회, seed 42.
  모든 비교가 같은 재표본 행렬을 쓴다(np.random.default_rng(42).integers(0, n_id, (10000, n_id))).
  한 재표본의 지표 = 뽑힌 신원(중복 포함)의 모든 유효 query 값의 평균 = Σ_j count_j·S_j / Σ_j count_j·N_j
  (S_j: 신원 j의 유효 query 값 합, N_j: 신원 j의 유효 query 수). 두 셀을 같은 재표본에서 계산해 차이를 낸다.
- 차이 Δ = 셀 A − 셀 B(전체 표본 값), 95% CI = 재표본 차이의 2.5·97.5 백분위(numpy linear).
- 판정: CI 하한 > 0 useful, 상한 < 0 harmful, 그 밖 inconclusive. 주 비교(S 모델 + SR test − C 모델 + bicubic test)에만 붙인다.

per-query 파일(csv.gz): name,pid,cam,valid,ap,r1,r5,r10. 두 셀은 같은 query 이름 순서여야 한다(검사함).
자체 시험: python stage2/stats.py --selftest  (test 데이터 불필요, 합성 데이터)
"""
import argparse
import csv
import gzip
import hashlib
import io
import os
import sys

import numpy as np

N_REP = 10000
SEED = 42
ALPHA = 0.05
PQ_FIELDS = ("name", "pid", "cam", "valid", "ap", "r1", "r5", "r10")


# ------------------------------------------------------------------ per-query 입출력
def write_per_query(path, names, pids, cams, valid, ap, cmc):
    """cmc: N×max_rank (0/1). 원자적 쓰기."""
    tmp = str(path) + ".tmp"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(PQ_FIELDS)
    for i in range(len(names)):
        v = bool(valid[i])
        w.writerow([names[i], int(pids[i]), int(cams[i]), int(v), repr(float(ap[i])) if v else "",
                    int(cmc[i, 0]) if v else "", int(cmc[i, 4]) if v else "", int(cmc[i, 9]) if v else ""])
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def read_per_query(path):
    with gzip.open(path, "rt") as f:
        rows = list(csv.DictReader(f))
    valid = np.array([r["valid"] == "1" for r in rows])
    num = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])  # noqa: E731
    return {"name": [r["name"] for r in rows], "pid": np.array([int(r["pid"]) for r in rows]),
            "cam": np.array([int(r["cam"]) for r in rows]), "valid": valid, "ap": num("ap"), "r1": num("r1"),
            "r5": num("r5"), "r10": num("r10")}


# ------------------------------------------------------------------ 지표와 부트스트랩
def cell_metric(pq, metric):
    col = {"mAP": "ap", "R1": "r1"}[metric]
    return float(np.mean(pq[col][pq["valid"]]))


def query_ids(pq):
    return np.unique(pq["pid"][pq["valid"]])


def resample_counts(ids, n_rep=N_REP, seed=SEED):
    """n_rep × n_id 정수 행렬: 각 재표본에서 신원 j가 뽑힌 횟수. 행렬과 sha256을 돌려준다."""
    n = len(ids)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_rep, n))
    counts = np.zeros((n_rep, n), np.int32)
    np.add.at(counts, (np.repeat(np.arange(n_rep), n), idx.ravel()), 1)
    h = hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes() + np.asarray(ids, np.int64).tobytes()).hexdigest()
    return counts, h


def id_sums(pq, ids, metric):
    """신원별 (유효 query 값 합, 유효 query 수). ids 순서."""
    col = {"mAP": "ap", "R1": "r1"}[metric]
    v = pq["valid"]
    pos = np.searchsorted(ids, pq["pid"][v])
    if not np.all(ids[pos] == pq["pid"][v]):
        raise ValueError("셀의 유효 query 신원이 재표본 신원 목록에 없음")
    s = np.bincount(pos, weights=pq[col][v], minlength=len(ids))
    c = np.bincount(pos, minlength=len(ids)).astype(np.float64)
    return s, c


def replicate_values(counts, s, c):
    return (counts @ s) / (counts @ c)


def check_paired(a, b):
    if a["name"] != b["name"]:
        raise ValueError("두 셀의 query 이름·순서가 다름(쌍 비교 불가)")
    if not np.array_equal(a["pid"], b["pid"]):
        raise ValueError("두 셀의 query pid가 다름")


def compare(a, b, counts, ids, metric):
    """Δ = a − b. 같은 재표본(counts)에서 두 셀을 계산한 차이의 백분위 CI."""
    check_paired(a, b)
    sa, ca = id_sums(a, ids, metric)
    sb, cb = id_sums(b, ids, metric)
    rep = replicate_values(counts, sa, ca) - replicate_values(counts, sb, cb)
    lo, hi = np.percentile(rep, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)], method="linear")
    return {"delta": cell_metric(a, metric) - cell_metric(b, metric), "ci_lo": float(lo), "ci_hi": float(hi),
            "a_value": cell_metric(a, metric), "b_value": cell_metric(b, metric), "rep_mean": float(rep.mean())}


def label(lo, hi):
    if lo > 0:
        return "useful"
    if hi < 0:
        return "harmful"
    return "inconclusive"


# 비교 이름: (셀 A의 (train, test), 판정 여부). 기준 셀 B는 항상 (bicubic, bicubic).
COMPARISONS = (("S_on_S", "sr", "sr", True),        # 주 비교: S 모델 + SR test
               ("C_on_S", "bicubic", "sr", False),  # test만 SR
               ("S_on_C", "sr", "bicubic", False))  # train만 SR
JUDGMENT_FIELDS = ("model", "k", "sr_cond", "comparison", "train_cond", "test_cond", "metric", "delta", "ci_lo", "ci_hi", "label",
                   "a_value", "b_value", "n_id", "n_query_valid", "n_rep", "seed", "resample_sha256")


def judge(model, k, sr, cells, counts, ids, sha):
    """cells: {(train_cond, test_cond): per-query dict}. 기준은 (bicubic, bicubic)."""
    base = cells[("bicubic", "bicubic")]
    rows = []
    for name, tr, te, labeled in COMPARISONS:
        tr_c, te_c = (sr if tr == "sr" else "bicubic"), (sr if te == "sr" else "bicubic")
        a = cells[(tr_c, te_c)]
        for metric in ("mAP", "R1"):
            r = compare(a, base, counts, ids, metric)
            rows.append({"model": model, "k": k, "sr_cond": sr, "comparison": name, "train_cond": tr_c, "test_cond": te_c,
                         "metric": metric, "delta": r["delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "label": label(r["ci_lo"], r["ci_hi"]) if labeled else "",
                         "a_value": r["a_value"], "b_value": r["b_value"], "n_id": len(ids), "n_query_valid": int(a["valid"].sum()),
                         "n_rep": counts.shape[0], "seed": SEED, "resample_sha256": sha})
    return rows


# ------------------------------------------------------------------ 자체 시험
def _ref_market(dist, q_pids, g_pids, q_cams, g_cams, max_rank=50):
    """공식 eval_func(CLIP-ReID utils/metrics.py, reid-strong-baseline, torchreid eval_market1501)의 루프를 그대로 옮긴 참조 구현."""
    indices = np.argsort(dist, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)
    all_cmc, all_ap, valid_idx = [], [], []
    for q in range(dist.shape[0]):
        order = indices[q]
        keep = np.invert((g_pids[order] == q_pids[q]) & (g_cams[order] == q_cams[q]))
        orig = matches[q][keep]
        if not np.any(orig):
            continue
        cmc = orig.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        tmp = orig.cumsum() / (np.arange(1, orig.shape[0] + 1) * 1.0) * orig
        all_ap.append(tmp.sum() / orig.sum())
        valid_idx.append(q)
    return np.asarray(all_cmc, np.float32).sum(0) / len(valid_idx), float(np.mean(all_ap)), np.array(all_ap), np.array(valid_idx)


def _synthetic_pq(rng, n_ids=120, per_id=(3, 7), base=0.7, shift=0.0, noise=0.1, n_invalid=2, names=None):
    pids, ap = [], []
    ids = np.sort(rng.choice(np.arange(1, 1500), n_ids, replace=False))
    for p in ids:
        for _ in range(rng.integers(per_id[0], per_id[1] + 1)):
            pids.append(p)
    n = len(pids)
    pid = np.array(pids)
    id_eff = dict(zip(ids, rng.normal(0, 0.15, n_ids)))
    ap = np.clip(base + np.array([id_eff[p] for p in pid]) + rng.normal(0, noise, n), 0, 1)
    valid = np.ones(n, bool)
    valid[rng.choice(n, n_invalid, replace=False)] = False
    pq = {"name": names or [f"{p:04d}_c1s1_{i:06d}_00" for i, p in enumerate(pid)], "pid": pid, "cam": np.zeros(n, int),
          "valid": valid, "ap": np.where(valid, ap, np.nan), "r1": np.where(valid, (ap > 0.6).astype(float), np.nan)}
    pq["r5"], pq["r10"] = pq["r1"], pq["r1"]
    return pq


def _shifted(pq, shift, rng=None, noise=0.0):
    out = dict(pq)
    ap = pq["ap"] + shift + (rng.normal(0, noise, len(pq["ap"])) if rng is not None else 0)
    out["ap"] = np.where(pq["valid"], ap, np.nan)  # [0,1] 자르기는 하지 않는다(차이를 정확히 shift로 두기 위해)
    out["r1"] = np.where(pq["valid"], (out["ap"] > 0.6).astype(float), np.nan)
    return out


def selftest(verbose=True):
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage05 import eval_core
    rng = np.random.default_rng(0)
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        if verbose:
            print(("PASS " if cond else "FAIL ") + name, flush=True)

    # 1. 재표본 지표가 '신원을 중복 포함해 이어 붙인 query 집합의 평균'과 같은지(직접 계산과 비교)
    pq = _synthetic_pq(rng)
    ids = query_ids(pq)
    counts, sha = resample_counts(ids, n_rep=200, seed=SEED)
    idx = np.random.default_rng(SEED).integers(0, len(ids), size=(200, len(ids)))
    s, c = id_sums(pq, ids, "mAP")
    fast = replicate_values(counts, s, c)
    naive = []
    for r in range(200):
        vals = np.concatenate([pq["ap"][pq["valid"] & (pq["pid"] == ids[j])] for j in idx[r]])
        naive.append(vals.mean())
    check("재표본 mAP = 중복 포함 이어 붙인 query 평균", np.allclose(fast, naive, rtol=0, atol=1e-12))
    check("counts 행 합 = n_id", np.all(counts.sum(1) == len(ids)))
    c2, sha2 = resample_counts(ids, n_rep=200, seed=SEED)
    check("같은 seed → 같은 재표본(sha256)", sha == sha2 and np.array_equal(counts, c2))
    check("전체 표본 mAP는 무효 query 제외 평균", np.isclose(cell_metric(pq, "mAP"), np.nanmean(pq["ap"])))

    # 2. 판정: 일정한 +/− 이동, 0 이동, 잡음만
    counts, sha = resample_counts(ids)
    up = _shifted(pq, 0.03)
    r = compare(up, pq, counts, ids, "mAP")
    check(f"+0.03 이동 → Δ=0.03, CI=[0.03,0.03], useful ({r['delta']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}])",
          np.isclose(r["delta"], 0.03) and np.isclose(r["ci_lo"], 0.03) and np.isclose(r["ci_hi"], 0.03) and label(r["ci_lo"], r["ci_hi"]) == "useful")
    r = compare(_shifted(pq, -0.02), pq, counts, ids, "mAP")
    check("−0.02 이동 → harmful", label(r["ci_lo"], r["ci_hi"]) == "harmful" and np.isclose(r["delta"], -0.02))
    r = compare(pq, pq, counts, ids, "mAP")
    check("같은 셀 → Δ=0, CI=[0,0], inconclusive", r["delta"] == 0 and r["ci_lo"] == 0 and r["ci_hi"] == 0 and label(0, 0) == "inconclusive")
    noisy = _shifted(pq, 0.0, np.random.default_rng(5), noise=0.05)
    r = compare(noisy, pq, counts, ids, "mAP")
    check(f"잡음만(평균 0) → CI가 0 포함 ({r['ci_lo']:.4f},{r['ci_hi']:.4f})", r["ci_lo"] < 0 < r["ci_hi"])
    # 쌍 효과: 신원 효과가 커도 두 셀에 공통이면 CI가 좁아야 한다(쌍이 아닌 독립 부트스트랩보다 좁음)
    small = _shifted(pq, 0.005, np.random.default_rng(6), noise=0.01)
    r = compare(small, pq, counts, ids, "mAP")
    sa, ca = id_sums(small, ids, "mAP")
    sb, cb = id_sums(pq, ids, "mAP")
    indep_counts, _ = resample_counts(ids, seed=SEED + 1)
    indep = replicate_values(counts, sa, ca) - replicate_values(indep_counts, sb, cb)
    il, ih = np.percentile(indep, [2.5, 97.5])
    check(f"쌍 CI 폭({r['ci_hi'] - r['ci_lo']:.4f}) < 독립 CI 폭({ih - il:.4f}), 쌍은 useful",
          (r["ci_hi"] - r["ci_lo"]) < (ih - il) and label(r["ci_lo"], r["ci_hi"]) == "useful")
    # 백분위 정의 확인
    sa, ca = id_sums(noisy, ids, "R1")
    sb, cb = id_sums(pq, ids, "R1")
    rep = replicate_values(counts, sa, ca) - replicate_values(counts, sb, cb)
    r = compare(noisy, pq, counts, ids, "R1")
    check("R1 CI = 재표본 차이의 2.5·97.5 백분위", np.isclose(r["ci_lo"], np.percentile(rep, 2.5)) and np.isclose(r["ci_hi"], np.percentile(rep, 97.5)))
    # 쌍 검사: 이름이 다르면 거부
    bad = dict(pq)
    bad["name"] = list(reversed(pq["name"]))
    try:
        compare(bad, pq, counts, ids, "mAP")
        check("이름 순서가 다르면 거부", False)
    except ValueError:
        check("이름 순서가 다르면 거부", True)

    # 3. judge 표 모양과 라벨 위치
    cells = {("bicubic", "bicubic"): pq, ("bicubic", "swinir"): _shifted(pq, -0.01), ("swinir", "swinir"): _shifted(pq, 0.02),
             ("swinir", "bicubic"): _shifted(pq, -0.03)}
    rows = judge("bot", 3, "swinir", cells, counts, ids, sha)
    lab = {(r["comparison"], r["metric"]): r["label"] for r in rows}
    check("judge: 6행, 주 비교에만 라벨", len(rows) == 6 and lab[("S_on_S", "mAP")] == "useful" and lab[("C_on_S", "mAP")] == ""
          and lab[("S_on_C", "mAP")] == "")
    dl = {r["comparison"]: r["delta"] for r in rows if r["metric"] == "mAP"}
    check("judge: Δ 부호(S_on_S +0.02, C_on_S −0.01, S_on_C −0.03)", np.isclose(dl["S_on_S"], 0.02) and np.isclose(dl["C_on_S"], -0.01) and np.isclose(dl["S_on_C"], -0.03))

    # 4. per-query 파일 왕복
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pq.csv.gz")
        cmc = np.zeros((len(pq["pid"]), 50), np.float32)
        cmc[:, :] = np.nan_to_num(pq["r1"])[:, None]
        write_per_query(p, pq["name"], pq["pid"], pq["cam"], pq["valid"], pq["ap"], cmc)
        back = read_per_query(p)
        check("per-query csv.gz 왕복(AP 비트 일치)", back["name"] == pq["name"] and np.array_equal(back["valid"], pq["valid"])
              and np.array_equal(back["ap"][back["valid"]], pq["ap"][pq["valid"]]))

    # 5. eval_core.block_metrics/stream_eval = 공식 루프(참조 구현), 블록 크기 무관. 합성 특징, 양성 없는 query·distractor 포함
    rng = np.random.default_rng(1)
    n_id, D = 40, 32
    centers = rng.normal(size=(n_id + 1, D))
    def make(n, ids_):
        pid = rng.choice(ids_, n)
        cam = rng.integers(0, 6, n)
        f = centers[np.searchsorted(np.arange(n_id + 1), pid)] + rng.normal(scale=1.2, size=(n, D))
        return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32), pid, cam
    qf, qp, qc = make(150, np.arange(1, n_id + 1))
    gf, gp, gc = make(900, np.arange(0, n_id + 1))  # pid 0 = distractor
    qp[:3] = 999  # gallery에 없는 신원 → 유효하지 않은 query
    dist = 1 - qf @ gf.T
    cmc_ref, map_ref, ap_ref, vidx = _ref_market(dist, qp, gp, qc, gc)
    for block in (1, 7, 64, 150):
        res = eval_core.stream_eval(qp, qc, gp, gc, lambda i0, i1: dist[i0:i1], block)
        same = (np.array_equal(np.flatnonzero(res["valid"]), vidx) and np.allclose(res["ap"][res["valid"]], ap_ref, rtol=0, atol=1e-12)
                and np.allclose(res["cmc_curve"], cmc_ref, atol=1e-6) and np.isclose(res["mAP"], map_ref, atol=1e-12))
        check(f"stream_eval(block={block}) = 공식 Market 루프 (AP·CMC·valid)", same)
    check("양성 없는 query 3개는 valid=False", res["n_valid"] == 147)
    # eval/metrics_reid.py(BoT 자체 평가)와도 같은지
    try:
        from eval.metrics_reid import evaluate_query
        aps = [evaluate_query(qf[i], qp[i], qc[i], gf, gp, gc)[0] for i in vidx]
        check("stream_eval = eval/metrics_reid.evaluate_query (BoT 1−cos)", np.allclose(res["ap"][res["valid"]], aps, rtol=0, atol=1e-12))
    except ImportError as e:
        check(f"eval/metrics_reid 불러오기 실패 {e}", False)

    print(f"{sum(ok)}/{len(ok)} 통과", flush=True)
    return all(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
