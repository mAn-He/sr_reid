"""query 신원 단위 복원추출, 모든 조건에 같은 재표본을 쓰는 paired 부트스트랩. gallery는 고정."""
import argparse

import numpy as np

from common.util import load_cfg, read_csv, write_csv
from sr.run import conditions

METRICS = ("ap", "r1")
LABEL = {"ap": "map", "r1": "rank1"}


def verdict(lo, hi):
    if lo > 0:
        return "useful"
    if hi < 0:
        return "harmful"
    return "inconclusive"


def resample_means(per_id_sum, per_id_n, weights):
    """weights: B×신원 수 복원추출 횟수. 반환: B개 평균."""
    return (weights @ per_id_sum) / (weights @ per_id_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    bc = cfg["eval"]["bootstrap"]
    pq_dir = cfg["_run"] / "eval" / "per_query"
    tables = {p.stem: read_csv(p) for p in sorted(pq_dir.glob("*.csv"))}
    ids = sorted({int(r["pid"]) for t in tables.values() for r in t})
    index = {p: i for i, p in enumerate(ids)}
    rng = np.random.RandomState(bc["seed"])
    draws = rng.randint(0, len(ids), size=(bc["B"], len(ids)))
    weights = np.zeros((bc["B"], len(ids)))
    for b in range(bc["B"]):
        weights[b] = np.bincount(draws[b], minlength=len(ids))

    boots, point = {}, {}
    for cond, t in tables.items():
        n = np.zeros(len(ids))
        for r in t:
            n[index[int(r["pid"])]] += 1
        for m in METRICS:
            s = np.zeros(len(ids))
            for r in t:
                s[index[int(r["pid"])]] += float(r[m])
            boots[f"{cond}|{m}"] = resample_means(s, n, weights)
            point[f"{cond}|{m}"] = s.sum() / n.sum()
    np.savez(cfg["_run"] / "eval" / "boot.npz", **boots)

    q = (1 - bc["ci"]) / 2 * 100
    policy_of = {c["cid"]: c["policy"] for c in conditions(cfg)}
    rows = []
    for cond in tables:
        for m in METRICS:
            b = boots[f"{cond}|{m}"]
            lo, hi = np.percentile(b, [q, 100 - q])
            rows.append({"condition": cond, "metric": LABEL[m], "comparator": "", "estimate": point[f"{cond}|{m}"],
                         "ci_lo": lo, "ci_hi": hi, "verdict": ""})
            refs = ["path1_mixed", "path1_lr_only"]
            if cond in policy_of:
                refs.append(f"bicubic__{policy_of[cond]}")
            for ref in refs:
                if ref == cond or f"{ref}|{m}" not in boots:
                    continue
                d = b - boots[f"{ref}|{m}"]
                lo, hi = np.percentile(d, [q, 100 - q])
                rows.append({"condition": cond, "metric": LABEL[m], "comparator": ref,
                             "estimate": point[f"{cond}|{m}"] - point[f"{ref}|{m}"], "ci_lo": lo, "ci_hi": hi,
                             "verdict": verdict(lo, hi)})
    write_csv(cfg["_run"] / "eval" / "bootstrap.csv", rows,
              ["condition", "metric", "comparator", "estimate", "ci_lo", "ci_hi", "verdict"])
    print(f"부트스트랩 완료: 조건 {len(tables)}개, 신원 {len(ids)}명, B={bc['B']}", flush=True)


if __name__ == "__main__":
    main()
