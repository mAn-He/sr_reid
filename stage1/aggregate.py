"""Stage 1 2단계 집계. srreid-quality env, CPU. 입력은 runs/stage1_metrics/(metrics_embed.py, metrics_fft.py 출력).
집계: 이미지 → 신원 평균 → 751명 평균. 신원 단위 부트스트랩 95% CI(10,000회, seed 42, 모든 방법·지표에 같은 재표본 사용, 백분위 구간).
출력(results/stage1/):
  per_image_metrics.csv     방법 × 배율 × 이미지, 모든 지표(원시 값)
  per_image_baselines.csv   해석 기준선 × 이미지(DINOv3·PersonViT 코사인)
  summary.csv               방법(기준선 포함) × 배율 × 지표: 평균, 95% CI, 이미지 중앙값, 방향
  distribution_distance.csv 모델 × 집합: Fréchet distance, KID(원본 집합 대비, L2 정규화 임베딩, 점추정)"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
import eval_models as em  # noqa: E402
from metrics_embed import MET, METHODS, MODELS, RES, sample  # noqa: E402

BASES = {"base_blur_s05": "기준선: 같은 사진 + 가우시안 블러 σ=0.5", "base_jpeg_q90": "기준선: 같은 사진 + JPEG 품질 90",
         "base_othercam": "기준선: 같은 사람 다른 카메라"}
# 지표: (방향, 설명). 방향 higher/lower는 A-FINE에서는 "좋음", 나머지는 적힌 뜻(원본과의 일치 등)으로 읽는다. neutral은 방향 없음.
DIR = {
    "dinov3_large_cos": ("higher", "높을수록 원본과 임베딩이 가까움"),
    "personvit_base_cos": ("higher", "높을수록 원본과 임베딩이 가까움"),
    "afine_all_scale": ("lower", "낮을수록 좋음(A-FINE 총점, 0~100)"),
    "afine_all": ("lower", "낮을수록 좋음(A-FINE 원값, 꼬리가 길어 중앙값만 보고)"),
    "fidelity": ("lower", "낮을수록 좋음(A-FINE 충실도 F)"),
    "natural_dis": ("lower", "낮을수록 좋음(A-FINE 자연스러움 N, SR 이미지)"),
    "natural_ref": ("lower", "낮을수록 좋음(A-FINE 자연스러움 N, 기준 이미지 = 원본 bicubic 확대, 방법과 무관)"),
    "lf_coh": ("higher", "높을수록 원본 대역에서 원본과 일치"),
    "lf_relerr_db": ("lower", "낮을수록 원본 대역에서 원본과 일치"),
    "hf_energy": ("neutral", "방향 없음: 원본 나이퀴스트 밖 에너지(추가된 고주파 양)"),
    "hf_frac": ("neutral", "방향 없음: 고주파 에너지 비율"),
    "hf_db_vs_bicubic": ("neutral", "방향 없음: 같은 이미지 bicubic 대비 고주파 에너지 dB"),
    "slope_o": ("neutral", "원본 스펙트럼 기울기(방법과 무관)"),
    "slope_s_lo": ("neutral", "SR 저대역 기울기"),
    "slope_s_hi": ("neutral", "SR 고대역 기울기"),
    "slope_dev_hi": ("zero", "0에 가까울수록 원본 기울기 유지(음수: 고주파 급감, 양수: 원본 경향보다 많음)"),
    "slope_dev_nat": ("zero", "0에 가까울수록 자연 이미지 평균 기울기(−2)와 같음"),
    "grid_peak_db": ("lower", "낮을수록 원본 표본화 격자 위치의 주기 피크가 약함"),
    "peak_db": ("lower", "낮을수록 원본 대역 밖 주기 피크가 약함"),
    "de2000": ("lower", "낮을수록 원본과 색 차이 작음(참고값)"),
}
COLS = list(DIR)


def read_csv(p):
    rows = list(csv.DictReader(open(p)))
    return {r["name"]: {k: float(v) for k, v in r.items() if k != "name"} for r in rows}


def boot_counts(n_id, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_id, (B, n_id))
    C = np.zeros((B, n_id), np.float32)
    np.add.at(C, (np.repeat(np.arange(B), n_id), idx.ravel()), 1)
    return C


def main():
    smp = sample()
    ns = [Path(r["name"]).stem for r in smp]
    pid = np.array([r["pid"] for r in smp])
    cam = [r["cam"] for r in smp]
    ids = sorted(set(pid))
    id_index = {p: i for i, p in enumerate(ids)}
    pid_i = np.array([id_index[p] for p in pid])
    emb = {mk: {} for mk in MODELS}
    for mk in MODELS:
        for p in sorted((MET / "emb" / mk).glob("*.npy")):
            emb[mk][p.stem] = np.load(p)
    per = {}  # (method, k) -> dict col -> array
    sets = [(m, k) for m in METHODS for k in (3, 4)]
    missing = []
    for m, k in sets:
        s = f"{m}__x{k}"
        d = {}
        for mk in MODELS:
            if s in emb[mk]:
                d[f"{mk}_cos"] = (emb[mk]["orig"] * emb[mk][s]).sum(1)
        fa, ff = MET / "afine" / f"{s}.csv", MET / "fft" / f"{s}.csv"
        for f in (fa, ff):
            if f.exists():
                v = read_csv(f)
                for c in next(iter(v.values())):
                    d[c] = np.array([v[n][c] for n in ns])
        if len(d) < 2:
            missing.append(s)
            continue
        per[(m, k)] = d
    for m, k in per:
        if "hf_energy" in per[(m, k)] and ("bicubic", k) in per and "hf_energy" in per[("bicubic", k)]:
            per[(m, k)]["hf_db_vs_bicubic"] = 10 * np.log10(per[(m, k)]["hf_energy"] / per[("bicubic", k)]["hf_energy"])
    base = {}
    for b in BASES:
        base[b] = {f"{mk}_cos": (emb[mk]["orig"] * emb[mk][b]).sum(1) for mk in MODELS if b in emb[mk]}
    # 이미지별 원시 값
    with open(RES / "per_image_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k", "name", "pid", "cam"] + COLS)
        for (m, k), d in per.items():
            for i, n in enumerate(ns):
                w.writerow([m, k, n, pid[i], cam[i]] + [repr(float(d[c][i])) if c in d else "" for c in COLS])
    with open(RES / "per_image_baselines.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "name", "pid", "cam"] + [f"{mk}_cos" for mk in MODELS])
        for b, d in base.items():
            for i, n in enumerate(ns):
                w.writerow([b, n, pid[i], cam[i]] + [repr(float(d[f"{mk}_cos"][i])) if f"{mk}_cos" in d else "" for mk in MODELS])
    # 신원 평균 + 부트스트랩
    C = boot_counts(len(ids))
    out = []

    def summarize(label, k, col, x):
        fin = np.isfinite(x)
        cnt = np.bincount(pid_i, weights=fin.astype(float), minlength=len(ids))
        sm_ = np.bincount(pid_i, weights=np.where(fin, x, 0.0), minlength=len(ids))
        ok = cnt > 0
        idm = np.where(ok, sm_ / np.maximum(cnt, 1), np.nan)
        bm = (C[:, ok] @ idm[ok]) / C[:, ok].sum(1)
        r = {"method": label, "k": k, "metric": col, "mean": float(idm[ok].mean()), "ci_lo": float(np.percentile(bm, 2.5)),
             "ci_hi": float(np.percentile(bm, 97.5)), "median_img": float(np.nanmedian(x)), "n_img": int(np.isfinite(x).sum()), "n_id": int(ok.sum()),
             "direction": DIR.get(col, ("", ""))[0], "direction_note": DIR.get(col, ("", ""))[1], "note": ""}
        if col == "afine_all":
            r.update(mean="", ci_lo="", ci_hi="", note="원값은 꼬리가 길어 이미지 중앙값만 보고")
        if label == "bicubic" and col in ("afine_all_scale", "afine_all", "fidelity", "natural_dis"):
            r["note"] = "기준 이미지와 동일, 비교 불가"
        out.append(r)

    for (m, k), d in per.items():
        for c in COLS:
            if c in d:
                summarize(m, k, c, d[c])
    for b, d in base.items():
        for c, x in d.items():
            summarize(b, "-", c, x)
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    # 분포 거리
    dist = []
    for mk in MODELS:
        eo = emb[mk].get("orig")
        if eo is None:
            continue
        for s, v in emb[mk].items():
            if s == "orig":
                continue
            ok = np.isfinite(v).all(1)
            kv, ks = em.kid(eo[ok], v[ok], n_subsets=100, subset=1000, seed=42, device="cuda")
            m, k = (s.split("__x")[0], int(s.split("__x")[1])) if "__x" in s else (s, "-")
            dist.append({"model": mk, "method": m, "k": k, "n": int(ok.sum()), "frechet": em.frechet(eo[ok], v[ok]), "kid": kv, "kid_std_subsets": ks, "kid_note": "cubic kernel, 100 subsets x 1000, seed 42, float64 GPU"})
    with open(RES / "distribution_distance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(dist[0]))
        w.writeheader()
        w.writerows(dist)
    print(f"집합 {len(per)}개, 빠진 집합 {missing}, summary {len(out)}행, 분포 거리 {len(dist)}행")


if __name__ == "__main__":
    main()
