"""0단계 ×3 방식 비교 요약: 방법별 direct − from_x4 이미지 짝 차이의 평균과 이미지 단위 부트스트랩 95% CI(10,000회, seed 42, 100장).
입력 results/stage1/step0_x3cmp_per_image.csv, step0_x3cmp_embed.csv. 출력 results/stage1/step0_x3cmp_compare.csv."""
import csv
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / "results" / "stage1"
M = ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "lf_relerr_db", "hf_energy", "hf_frac", "slope_dev_hi", "grid_peak_db", "peak_db", "de2000"]
v = {}
for f in ("step0_x3cmp_per_image.csv", "step0_x3cmp_embed.csv"):
    for r in csv.DictReader(open(R / f)):
        d = v.setdefault((r["method"], r["way"], r["name"]), {})
        d.update({k: float(x) for k, x in r.items() if k not in ("name", "method", "way")})
names = sorted({k[2] for k in v})
rng = np.random.default_rng(42)
idx = rng.integers(0, len(names), (10000, len(names)))
rows = []
for m in ["osdhuman"] + [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]:
    for k in M:
        a = np.array([v[(m, "from_x4", n)][k] for n in names]); b = np.array([v[(m, "direct", n)][k] for n in names])
        d = b - a
        bs = d[idx].mean(1)
        rows.append({"method": m, "metric": k, "from_x4_mean": a.mean(), "direct_mean": b.mean(), "diff_direct_minus_from_x4": d.mean(),
                     "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "n": len(names)})
with open(R / "step0_x3cmp_compare.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["method"], r["metric"], *(f"{r[k]:.4g}" for k in ("from_x4_mean", "direct_mean", "diff_direct_minus_from_x4", "ci_lo", "ci_hi")))
