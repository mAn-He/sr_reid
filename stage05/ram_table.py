"""B.2 RAM 역산표. 코드 분석에서 나온 누적 항목의 shape×dtype×N. 측정이 아니라 계산이다. 결과: results/stage05_ram_estimate.csv.
N은 사양 값(23,100 = query 3,368 + 파일 기준 gallery 19,732)과 공식 로더 값(19,281 = pid −1 제외)을 둘 다 쓴다."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
LIMIT = 57.74
NQ = 3368
SIZES = {"bot": [(256, 128), (384, 192), (512, 256)], "clipreid": [(256, 128), (384, 192), (512, 256)], "kpr": [(384, 128), (384, 192), (512, 256)], "kpr_solider": [(384, 128), (384, 192), (512, 256)]}


def vector_model(nq, ng, dim, dist_bytes):
    """전역 특징 하나: 특징 누적 + cat 피크(원본 리스트와 합본 공존) + Q×G 거리."""
    per = dim * 4 + 16
    feats = (nq + ng) * per
    return {"features": feats, "collate_peak_extra": (nq + ng) * dim * 4, "distmat": nq * ng * dist_bytes, "evaluate_temporaries": nq * ng * 8}


def kpr_model(nq, ng, h, w):
    """part_based_engine._feature_extraction 누적(K=5 test_embeddings 조합 기준, f_ 12,288B/장). 해상도 항목 바이트는 H·W에 비례."""
    hw = h * w
    fixed = 12288 + 6 + 544 + 2720 + 16
    res = {"parts_masks": 1.5 * hw, "pxl_scores": 1.5 * hw, "anns_image": 12 * hw, "anns_prompt": 64 * hw, "anns_target": 1.5 * hw}
    per = fixed + sum(res.values())
    total = (nq + ng) * per
    return {"fixed_per_image": (nq + ng) * fixed, **{k: (nq + ng) * v for k, v in res.items()}, "collate_peak_extra": ng * per, "distmat": nq * ng * 4, "evaluate_temporaries": nq * ng * 8, "_total_acc": total}


def main():
    rows = []
    for ngname, ng in (("official_loader_N19281", 15913), ("spec_N23100", 19732)):
        for model, sizes in SIZES.items():
            for h, w in sizes:
                if model == "bot":
                    d = vector_model(NQ, ng, 2048, 4)
                elif model == "clipreid":
                    d = vector_model(NQ, ng, 1280, 4)
                else:
                    d = kpr_model(NQ, ng, h, w)
                    d.pop("_total_acc")
                total = sum(d.values())
                rows.append({"n_basis": ngname, "model": model, "input": f"{h}x{w}", "N": NQ + ng, **{k: round(v / GIB, 3) for k, v in d.items()}, "peak_total_gib": round(total / GIB, 2), "pod_limit_gib": LIMIT, "exceeds_limit": total / GIB > LIMIT})
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(ROOT / "results" / "stage05_ram_estimate.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fields)
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(r["n_basis"], r["model"], r["input"], r["peak_total_gib"], "초과" if r["exceeds_limit"] else "안에")


if __name__ == "__main__":
    main()
