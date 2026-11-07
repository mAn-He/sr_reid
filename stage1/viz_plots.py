"""Stage 1 3단계 그림(수치 그림). srreid-quality env. 입력 results/stage1/*.csv, runs/stage1_metrics/.
출력 results/stage1/figures/*.png (+ t-SNE 좌표 results/stage1/tsne_<model>_x<k>.csv.gz, 설정 results/stage1/tsne_config.json)
  dist_<metric>.png         지표별 이미지 단위 분포(×3, ×4 상자그림), 임베딩 코사인에는 해석 기준선 3종을 겹쳐 그림
  forest_x<k>.png           요약: 방법 × 지표 평균과 신원 부트스트랩 95% CI
  spearman_x<k>.png         지표 간 이미지 단위 Spearman 상관(방법 12종을 합쳐 계산 / 방법별 계산 후 평균)
  fft_radial_x<k>.png       방법별 평균 방사형 파워 스펙트럼(원본·bicubic 겹침)
  fft2d_<name>_x<k>.png     대표 이미지 2D 스펙트럼 비포·애프터
  tsne_<model>_x<k>.png, tsne_<model>_x<k>_facets.png
그림은 Claude가 열어 보지 않는다(수치로만 검사)."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
from aggregate import DIR  # noqa: E402
from metrics_embed import INP, MET, METHODS, MODELS, RES, SR, names  # noqa: E402

FIG = RES / "figures"
LAB = {m: m.replace("pisa_sr_sem", "pisa λ") for m in METHODS}
COL = {m: plt.get_cmap("tab20")(i % 20) for i, m in enumerate(METHODS)}
DIST_METRICS = ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "lf_relerr_db",
                "hf_energy", "hf_db_vs_bicubic", "slope_dev_hi", "grid_peak_db", "peak_db", "de2000"]
BASE_LAB = {"base_blur_s05": "same photo + blur σ0.5", "base_jpeg_q90": "same photo + JPEG q90", "base_othercam": "same person, other camera"}
DIRWORD = {"higher": "higher = closer to original", "lower": "lower = better" , "zero": "closer to 0 = keeps reference slope", "neutral": "no direction"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 110})


def random20():
    """비교용 무작위 20장(seed 42). 비포·애프터 d와 2D 스펙트럼 대표 이미지(앞 2장)에 같이 쓴다."""
    ns = names()
    rng = np.random.default_rng(42)
    return [ns[i] for i in sorted(rng.choice(len(ns), 20, replace=False))]


def dir_text(c):
    d = DIR[c][0]
    if c.startswith(("afine", "fidelity", "natural")):
        return "lower = better (A-FINE)"
    if c in ("lf_coh",):
        return "higher = closer to original (low band)"
    if c in ("lf_relerr_db",):
        return "lower = closer to original (low band)"
    if c in ("grid_peak_db", "peak_db"):
        return "lower = weaker periodic peak"
    if c == "de2000":
        return "lower = smaller colour difference"
    return DIRWORD[d]


def load():
    pi = pd.read_csv(RES / "per_image_metrics.csv")
    pb = pd.read_csv(RES / "per_image_baselines.csv")
    sm = pd.read_csv(RES / "summary.csv")
    return pi, pb, sm


def dist_plots(pi, pb):
    for c in DIST_METRICS:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
        for ax, k in zip(axs, (3, 4)):
            data, labels, cols = [], [], []
            for m in METHODS:
                v = pi[(pi.method == m) & (pi.k == k)][c].dropna().values
                if c.startswith(("afine", "fidelity", "natural")) and m == "bicubic":
                    v = np.array([])
                data.append(v if len(v) else [np.nan])
                labels.append(LAB[m] + (" (n/a: =ref)" if c.startswith(("afine", "fidelity", "natural")) and m == "bicubic" else ""))
                cols.append(COL[m])
            if c.endswith("_cos"):
                for b, bl in BASE_LAB.items():
                    v = pb[pb.baseline == b][c].dropna().values
                    data.append(v)
                    labels.append("[baseline] " + bl)
                    cols.append((0.6, 0.6, 0.6, 1))
            bp = ax.boxplot(data, showfliers=True, flierprops={"markersize": 1.5, "alpha": 0.3}, patch_artist=True, widths=0.6)
            for p, cc in zip(bp["boxes"], cols):
                p.set_facecolor(cc)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=60, ha="right")
            ax.set_title(f"x{k}")
            if c == "hf_energy":
                ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"{c}  ({dir_text(c)}) — per-image values, n=2,250 per box")
        fig.tight_layout()
        fig.savefig(FIG / f"dist_{c}.png")
        plt.close(fig)


def forest(sm):
    mets = [c for c in DIST_METRICS if c in set(sm.metric)]
    for k in (3, 4):
        fig, axs = plt.subplots(3, (len(mets) + 2) // 3, figsize=(18, 11))
        for ax, c in zip(axs.ravel(), mets):
            s = sm[(sm.metric == c) & (sm.k.astype(str) == str(k))].set_index("method")
            ys = []
            for i, m in enumerate(METHODS):
                if m not in s.index or pd.isna(s.loc[m, "mean"]) or (c.startswith(("afine", "fidelity", "natural")) and m == "bicubic"):
                    continue
                r = s.loc[m]
                ax.errorbar(float(r["mean"]), i, xerr=[[float(r["mean"]) - float(r["ci_lo"])], [float(r["ci_hi"]) - float(r["mean"])]], fmt="o", color=COL[m], ms=4, capsize=2)
                ys.append(i)
            if c.endswith("_cos"):
                b = sm[(sm.metric == c) & (sm.k.astype(str) == "-")]
                for _, r in b.iterrows():
                    ax.axvline(float(r["mean"]), color="gray", ls="--", lw=0.8)
                    ax.text(float(r["mean"]), len(METHODS) - 0.3, BASE_LAB[r["method"]], rotation=90, fontsize=6, color="gray", va="top")
            ax.set_yticks(range(len(METHODS)), [LAB[m] for m in METHODS], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{c}\n({dir_text(c)})", fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        for ax in axs.ravel()[len(mets):]:
            ax.axis("off")
        fig.suptitle(f"x{k}: identity-level mean (751 ids) with identity bootstrap 95% CI (10,000 resamples, seed 42). Bicubic omitted for A-FINE (= reference).")
        fig.tight_layout()
        fig.savefig(FIG / f"forest_x{k}.png")
        plt.close(fig)


def spearman(pi):
    from scipy.stats import spearmanr
    mets = ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "lf_relerr_db", "hf_db_vs_bicubic",
            "slope_dev_hi", "grid_peak_db", "peak_db", "de2000"]
    out = {}
    for k in (3, 4):
        d = pi[(pi.k == k) & (pi.method != "bicubic")]
        pooled = spearmanr(d[mets].values, nan_policy="omit").statistic
        within = np.nanmean([spearmanr(d[d.method == m][mets].values, nan_policy="omit").statistic for m in METHODS if m != "bicubic"], axis=0)
        out[k] = (pooled, within)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
        for ax, mat, t in zip(axs, (pooled, within), ("pooled over 11 methods (bicubic excluded: A-FINE n/a)", "computed per method, then averaged")):
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(mets)), mets, rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(len(mets)), mets, fontsize=7)
            for i in range(len(mets)):
                for j in range(len(mets)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
            ax.set_title(t)
        fig.colorbar(im, ax=axs, shrink=0.7)
        fig.suptitle(f"x{k}: image-level Spearman correlation between metrics")
        fig.savefig(FIG / f"spearman_x{k}.png", bbox_inches="tight")
        plt.close(fig)
        pd.DataFrame(pooled, index=mets, columns=mets).to_csv(RES / f"spearman_pooled_x{k}.csv")
        pd.DataFrame(within, index=mets, columns=mets).to_csv(RES / f"spearman_within_x{k}.csv")


def fft_radial():
    o = np.load(MET / "fft" / "orig__radial.npz")
    for k in (3, 4):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.loglog(o["f"], o["P"].mean(0), color="black", lw=2.2, label="original 128x64")
        for m in METHODS:
            p = MET / "fft" / f"{m}__x{k}__radial.npz"
            if not p.exists():
                continue
            z = np.load(p)
            ax.loglog(z["f"], z["P"].mean(0), color=COL[m], lw=2 if m == "bicubic" else 1.1, ls="--" if m in ("bicubic", "lanczos") else "-", label=LAB[m])
        ax.axvline(0.5, color="gray", ls=":", lw=1)
        ax.text(0.5, ax.get_ylim()[0] * 3, " original Nyquist (0.5)", color="gray", fontsize=8)
        for g in range(1, k // 2 + 1):
            ax.axvline(g, color="red", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("radial frequency (cycles per original pixel)")
        ax.set_ylabel("mean power (Y channel, DFT/pixel count)²")
        ax.set_title(f"x{k}: mean radial power spectrum over 2,250 images (red dotted: original sampling-frequency multiples)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG / f"fft_radial_x{k}.png")
        plt.close(fig)


def fft2d(sel):
    import fft_color as fc
    from PIL import Image

    def ld(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"))
    for n in sel:
        for k in (3, 4):
            o = ld(INP / f"{n}.png")
            panels = [("original", np.fft.fftshift(np.abs(fc.spectrum(fc.to_y(o))) ** 2), 1)]
            for m in METHODS:
                panels.append((LAB[m], np.fft.fftshift(np.abs(fc.spectrum(fc.to_y(ld(SR / m / f"x{k}" / f"{n}.webp")))) ** 2), k))
            vmin, vmax = -9, np.log10(max(p.max() for _, p, _ in panels))
            fig, axs = plt.subplots(1, len(panels), figsize=(len(panels) * 1.6, 3.6))
            for ax, (t, P, kk) in zip(axs, panels):
                h, w = P.shape
                ext = [-0.5 * kk, 0.5 * kk, 0.5 * kk, -0.5 * kk]
                ax.imshow(np.log10(P + 1e-12), cmap="magma", vmin=vmin, vmax=vmax, extent=ext, aspect="equal")
                ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="cyan", lw=0.6))
                ax.set_xlim(-0.5 * k, 0.5 * k)
                ax.set_ylim(0.5 * k, -0.5 * k)
                ax.set_title(t, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{n}  x{k}: log10 power spectrum (Y), axes in cycles/original px over ±{k / 2:g}; cyan box = original band (±0.5)", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG / f"fft2d_{n}_x{k}.png")
            plt.close(fig)


def tsne():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    cfg = {"input": "L2-normalized CLS embeddings (same as cosine metric)", "points": "original 2,250 + 12 methods x 2,250 per scale",
           "pca_components": 50, "pca_random_state": 42, "perplexity": 30, "init": "pca", "learning_rate": "auto", "max_iter": 1000,
           "metric": "euclidean", "random_state": 42, "method": "barnes_hut", "angle": 0.5, "n_jobs": 8, "sklearn": __import__("sklearn").__version__}
    (RES / "tsne_config.json").write_text(json.dumps(cfg, indent=1))
    for mk in MODELS:
        for k in (3, 4):
            out = RES / f"tsne_{mk}_x{k}.csv.gz"
            if out.exists():
                df = pd.read_csv(out)
            else:
                sets = ["orig"] + [f"{m}__x{k}" for m in METHODS]
                X = np.concatenate([np.load(MET / "emb" / mk / f"{s}.npy") for s in sets])
                lab = np.repeat(["original"] + METHODS, 2250)
                Z = PCA(50, random_state=42).fit_transform(X)
                Y = TSNE(2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000, random_state=42, n_jobs=8).fit_transform(Z)
                df = pd.DataFrame({"set": lab, "name": np.tile(names(), len(sets)), "x": Y[:, 0], "y": Y[:, 1]})
                df.to_csv(out, index=False)
            fig, ax = plt.subplots(figsize=(10, 9))
            for s in ["original"] + METHODS:
                d = df[df.set == s]
                ax.scatter(d.x, d.y, s=1.5, alpha=0.5, color="black" if s == "original" else COL[s], label=LAB.get(s, s))
            ax.legend(markerscale=6, fontsize=7)
            ax.set_title(f"t-SNE {mk} x{k} (original + 12 methods, seed 42, perplexity 30)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(FIG / f"tsne_{mk}_x{k}.png")
            plt.close(fig)
            fig, axs = plt.subplots(3, 5, figsize=(20, 12))
            for ax, s in zip(axs.ravel(), ["original"] + METHODS):
                ax.scatter(df.x, df.y, s=0.5, color=(0.85, 0.85, 0.85))
                if s != "original":
                    d = df[df.set == "original"]
                    ax.scatter(d.x, d.y, s=0.5, color=(0.45, 0.45, 0.45), alpha=0.4)
                d = df[df.set == s]
                ax.scatter(d.x, d.y, s=1.2, color="black" if s == "original" else COL[s])
                ax.set_title(f"{LAB.get(s, s)} (dark grey = original)", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axs.ravel()[13:]:
                ax.axis("off")
            fig.suptitle(f"t-SNE {mk} x{k}: each method highlighted")
            fig.tight_layout()
            fig.savefig(FIG / f"tsne_{mk}_x{k}_facets.png")
            plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["dist", "forest", "spearman", "fft", "tsne"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    pi, pb, sm = load()
    if "dist" in a.parts:
        dist_plots(pi, pb)
    if "forest" in a.parts:
        forest(sm)
    if "spearman" in a.parts:
        spearman(pi)
    if "fft" in a.parts:
        fft_radial()
        fft2d(random20()[:2])
    if "tsne" in a.parts:
        tsne()
    print("그림 완료")
