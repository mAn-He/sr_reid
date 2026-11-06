"""중간 지시 3. results/stage05_kpr_memory.csv의 allocated 피크를 peak = a + b·B + c·B²로 적합해 배치 64를 외삽한다. numpy만 쓴다.
결과: results/stage05_vram_extrapolation.csv. 외삽값은 측정이 아니다. 범위는 2차식 95% 예측구간과 1차식의 합집합으로 잡는다.
실제 점유 = allocated × 비율 + 드라이버 몫. 비율 하한은 Stage 0 값 1.26, 상한은 max(1.26, reserved/allocated 실측 최대). 격자 nvidia-smi는 3스텝이라 1초 표본이 피크를 놓쳐 비율에 쓰지 않는다.
격자(3스텝)와 30 iteration 측정(stage05_vram.csv)의 일정한 차 0.356GiB를 더한다. 파라미터 1벌 크기와 같지만 원인은 확정하지 않았다."""
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "stage05_kpr_memory.csv"
OUT = ROOT / "results" / "stage05_vram_extrapolation.csv"
STAGE0_RATIO = 1.26
# t(0.975, dof). dof 1~6
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
GPUS = ((48, "48GB"), (80, "80GB"))
USABLE = 0.98  # 공칭 GB는 실제 GiB다. 4090 24GB에서 torch가 23.52GiB를 본다
OFFICIAL_OFFSET = 0.356  # stage05_vram − 격자: ×3 b48 15.13−14.776, ×4 b24 13.57−13.214
DRIVER_GIB = 0.88  # nvidia-smi − reserved, 30 iteration 측정: ×3 b48, ×4 b24 모두 0.88


def design(b, deg):
    return np.stack([b ** d for d in range(deg + 1)], 1).astype(np.float64)


def fit(b, y, deg, at=64.0):
    X = design(b, deg)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    dof = len(b) - deg - 1
    x0 = design(np.array([at]), deg)
    pred = float((x0 @ coef)[0])
    half = None
    if dof >= 1:
        s2 = float(res @ res) / dof
        var = s2 * (1 + float((x0 @ np.linalg.pinv(X.T @ X) @ x0.T).item()))
        half = T95.get(dof, 2.0) * math.sqrt(max(var, 0.0))
    return {"coef": coef.tolist(), "residuals": res.tolist(), "rmse": float(np.sqrt(np.mean(res ** 2))), "pred": pred, "half": half}


def verdict(occupy_hi, occupy_lo):
    """상한 기준으로 들어가는 가장 작은 GPU. 하한과 상한이 갈리면 함께 표시한다."""
    def tier(v):
        for cap, name in GPUS:
            if v <= cap * USABLE:
                return name
        return ">80GB"
    return tier(occupy_hi), tier(occupy_lo)


def reserved_ratio(m, inp, b, c, peak):
    k = {"384x192": 3, "512x256": 4}[inp]
    o = json.loads((ROOT / "logs" / "stage05" / f"h4_{m}_x{k}_b{b}_ckpt{c}.out.json").read_text())
    return max(v["reserved_gib"] for v in o["phases"].values()) / peak


def main():
    rows = list(csv.DictReader(SRC.open()))
    groups = {}
    for r in rows:
        if r["status"] != "ok" or not r["peak_allocated_gib"] or int(r["batch"]) == 64:  # 배치 64 실측은 검증용으로 적합에서 뺀다
            continue
        groups.setdefault((r["model"], r["input"], int(r["ckpt"])), []).append(r)
    out = []
    for (m, inp, c), rs in sorted(groups.items()):
        rs.sort(key=lambda x: int(x["batch"]))
        b = np.array([int(x["batch"]) for x in rs], np.float64)
        y = np.array([float(x["peak_allocated_gib"]) for x in rs])
        ratios = [reserved_ratio(m, inp, int(x["batch"]), c, float(x["peak_allocated_gib"])) for x in rs if not x["alloc_conf"]]  # max_split 점은 reserved가 24GB에 눌려 비율에 쓰지 않는다
        ratio_hi = max(ratios + [STAGE0_RATIO])
        row = {"model": m, "input": inp, "ckpt": c, "n_points": len(b), "batches": " ".join(str(int(x)) for x in b), "peaks_gib": " ".join(f"{v:.3f}" for v in y),
               "ratio_stage0": STAGE0_RATIO, "ratio_reserved_max": round(max(ratios), 3), "kind": "외삽(배치 64 미측정)"}
        if len(b) < 2:
            row["note"] = "측정점 부족"
            out.append(row)
            continue
        lin = fit(b, y, 1)
        row.update(lin_a=round(lin["coef"][0], 4), lin_b=round(lin["coef"][1], 5), lin_rmse=round(lin["rmse"], 4), lin_b64=round(lin["pred"], 2))
        lo, hi = lin["pred"], lin["pred"]
        if len(b) >= 4:
            q = fit(b, y, 2)
            row.update(a=round(q["coef"][0], 4), b=round(q["coef"][1], 5), c=round(q["coef"][2], 6), residuals=" ".join(f"{v:+.3f}" for v in q["residuals"]), rmse=round(q["rmse"], 4),
                       b64_alloc=round(q["pred"], 2), b64_ci95=round(q["half"], 2) if q["half"] is not None else "")
            band = q["half"] or 0.0
            lo, hi = min(lo, q["pred"] - band), max(hi, q["pred"] + band)
        else:
            row["note"] = f"측정점 {len(b)}개, 2차 적합 불가. 1차식만"
        lo, hi, mid = lo + OFFICIAL_OFFSET, hi + OFFICIAL_OFFSET, lin["pred"] + OFFICIAL_OFFSET
        occ_lo, occ_hi = lo * STAGE0_RATIO + DRIVER_GIB, hi * ratio_hi + DRIVER_GIB
        row.update(official_offset=OFFICIAL_OFFSET, b64_official_lin=round(mid, 2), b64_alloc_lo=round(lo, 2), b64_alloc_hi=round(hi, 2), occupy_lo_gib=round(occ_lo, 2), occupy_hi_gib=round(occ_hi, 2),
                   occupy_mid_gib=round(mid * ratio_hi + DRIVER_GIB, 2))
        vh, vl = verdict(occ_hi, occ_lo)
        row.update(verdict_upper=vh, verdict_lower=vl)
        out.append(row)
    fields = ["model", "input", "ckpt", "n_points", "batches", "peaks_gib", "a", "b", "c", "residuals", "rmse", "b64_alloc", "b64_ci95", "lin_a", "lin_b", "lin_rmse", "lin_b64",
              "b64_alloc_lo", "b64_alloc_hi", "ratio_stage0", "ratio_reserved_max", "official_offset", "b64_official_lin", "occupy_lo_gib", "occupy_mid_gib", "occupy_hi_gib", "verdict_lower", "verdict_upper", "kind", "note"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in fields})
    print(json.dumps(out, ensure_ascii=False, indent=0)[:6000])


if __name__ == "__main__":
    main()
