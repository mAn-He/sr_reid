"""정책별 PSNR·LPIPS 순위와 mAP 순위의 Spearman ρ. lambda_sem은 대표점 하나만 포함한다.

대표 방법이 5개뿐이라 ρ만 기술 통계로 보고하고 p값은 내지 않는다.
"""
import argparse

import numpy as np
from scipy.stats import spearmanr

from common.util import load_cfg, read_csv, write_csv
from sr.run import conditions


def representatives(cfg, policy):
    sem = cfg["eval"]["correlate"]["sem_representative"]
    return [c for c in conditions(cfg) if c["policy"] == policy and c["lambda_sem"] in (None, sem)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    ev = cfg["_run"] / "eval"
    mean_ap = {r["condition"]: float(r["estimate"]) for r in read_csv(ev / "bootstrap.csv")
               if r["metric"] == "map" and r["comparator"] == ""}
    rows = []
    for policy in cfg["sr"]["preup_policies"]:
        xs = []
        for c in representatives(cfg, policy):
            f = ev / "fidelity" / f"{c['cid']}.csv"
            if c["cid"] in mean_ap and f.exists():
                t = read_csv(f)
                xs.append((np.mean([float(r["psnr"]) for r in t]), np.mean([float(r["lpips"]) for r in t]), mean_ap[c["cid"]]))
        n = len(xs)
        rho_p = rho_l = None
        if n >= 3:
            a = np.array(xs)
            rho_p = float(spearmanr(a[:, 0], a[:, 2]).statistic)
            rho_l = float(spearmanr(-a[:, 1], a[:, 2]).statistic)  # LPIPS는 낮을수록 좋아 부호를 뒤집는다.
        rows.append({"preup_policy": policy, "n_methods": n, "rho_psnr_map": rho_p, "rho_neg_lpips_map": rho_l})
    write_csv(cfg["_results"] / "correlation.csv", rows, ["preup_policy", "n_methods", "rho_psnr_map", "rho_neg_lpips_map"])
    print(rows, flush=True)


if __name__ == "__main__":
    main()
