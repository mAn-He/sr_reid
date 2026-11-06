"""C.6 등가성 표와 기준선 행. with_test.sh 아래에서 돈다. 공개 가중치 지표만 읽는다."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

FEAT = ROOT / "data" / "locked" / "features"
LOG = ROOT / "logs" / "stage05"
PUB = {"bot": (85.9, 94.5, "reid-strong-baseline README Market1501 ResNet50 (center loss 포함 모델, 공개 가중치 파일명 rank1_945)", "256x128"),
       "clipreid": (89.6, 95.5, "공개 가중치 테스트 로그 clipreid_vit_market_test_log.txt", "256x128"),
       "kpr": (89.58, 95.87, "공개 가중치 파일명 kpr_market_IN_89.58_95.87", "384x128"),
       "kpr_solider": (93.25, 96.59, "공개 가중치 파일명 kpr_market_SOLIDER_93.25_96.59", "384x128")}


def equal_features():
    rows = {}
    for m in ("kpr", "kpr_solider"):
        for s in ("query", "gallery"):
            a, b = FEAT / m / "public_full" / s, FEAT / m / "public_full_v2" / s
            for f in sorted(p.name for p in a.glob("*.npy")):
                x, y = np.load(a / f), np.load(b / f)
                rows[f"{m}/{s}/{f}"] = {"shape": list(x.shape), "dtype": str(x.dtype), "array_equal": bool(np.array_equal(x, y)), "max_abs_diff": float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())}
    return rows


def main():
    fe = equal_features()
    (ROOT / "results" / "stage05_light_feature_check.json").write_text(json.dumps(fe, indent=1, ensure_ascii=False))
    print(json.dumps(fe, ensure_ascii=False)[:1500])
    rows = []
    for m in PUB:
        d = FEAT / m / "public_sub"
        if not (d / "official_metrics.npz").exists():
            print("없음", m)
            continue
        o = np.load(d / "official_metrics.npz")
        for b in (256, 512, 1024):
            l = np.load(d / f"light_b{b}.npz")
            ap_o, ap_l = np.asarray(o["ap"], np.float64), np.asarray(l["ap"], np.float64)
            valid = np.asarray(l["valid"], bool)
            ap_l_v = ap_l[valid]
            cmc_o = np.asarray(o["cmc"], np.float64)
            cmc_l = np.asarray(l["cmc_curve"], np.float64)
            same_len = len(ap_o) == len(ap_l_v)
            rows.append({"model": m, "input": PUB[m][3], "n_query": len(ap_l), "n_gallery": 5000, "block": b, "n_valid_light": int(valid.sum()), "n_ap_official": len(ap_o),
                         "official_mAP": float(o["mAP"]), "light_mAP": float(l["mAP"]), "mAP_abs_diff": abs(float(o["mAP"]) - float(l["mAP"])),
                         "official_r1": cmc_o[0], "light_r1": cmc_l[0], "official_r5": cmc_o[4], "light_r5": cmc_l[4], "official_r10": cmc_o[9], "light_r10": cmc_l[9],
                         "cmc_1_5_10_max_abs_diff": max(abs(cmc_o[i] - cmc_l[i]) for i in (0, 4, 9)),
                         "ap_per_query_max_abs_diff": float(np.abs(ap_o - ap_l_v).max()) if same_len else "길이 다름",
                         "ap_per_query_n_diff_gt_1e-6": int((np.abs(ap_o - ap_l_v) > 1e-6).sum()) if same_len else "길이 다름"})
    with open(ROOT / "results" / "stage05_eval_equiv.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: r[k] for k in ("model", "block", "n_valid_light", "n_ap_official", "mAP_abs_diff", "cmc_1_5_10_max_abs_diff", "ap_per_query_max_abs_diff", "ap_per_query_n_diff_gt_1e-6")})
    base = []
    for m, (mp, r1, src, inp) in PUB.items():
        o = json.loads((LOG / f"baseline_{m}_light.out.json").read_text())["result"]["1024"]
        base.append({"model": m, "input": inp, "weights": "공개 가중치", "evaluator": "경량(스트리밍, 블록 256/512/1024 동일)", "mAP": round(o["mAP"] * 100, 2), "rank1": round(o["rank1"] * 100, 2), "rank5": round(o["rank5"] * 100, 2), "rank10": round(o["rank10"] * 100, 2),
                     "n_query": o["n_query"], "published_mAP": mp, "published_rank1": r1, "published_source": src, "note": "참고 행. 학습하지 않음. re-ranking 없음"})
    with open(ROOT / "results" / "baseline_public_weights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(base[0]))
        w.writeheader()
        w.writerows(base)
    for r in base:
        print({k: r[k] for k in ("model", "input", "mAP", "rank1", "rank5", "rank10", "published_mAP", "published_rank1")})


main()
