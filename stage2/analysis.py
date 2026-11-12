"""Stage 2 사후 분석 2b~2d (평가 완료 후 분석, Stage 2 판정과 별개). srreid-quality env(torch GPU로 부트스트랩). ALLOW_TEST=1에서만.
입력: data/locked/stage2_analysis/metrics/(test_metrics.py), data/locked/stage2_eval/*/per_query.csv.gz, results/stage1/per_image_metrics.csv·summary.csv,
      results/stage2/judgment.csv.
단계:
  table  test 이미지별 지표 표(코사인 = 원본·SR 임베딩 내적, hf_db_vs_bicubic = 같은 이미지 bicubic 대비 고주파 에너지 dB) → 잠긴 영역 csv.gz
  dist   Stage 1 train 표본 대 test의 지표 분포(방법·배율별 평균, 신원 단위 부트스트랩 95% CI, 상자그림)
  query  ΔAP_q = AP(S_s 모델, s test) − AP(C 모델, bicubic test). 설명 변수: query 지표, 같은 신원 정답 gallery(다른 카메라) 지표 평균.
         Spearman ρ(query 단위, 신원 단위 평균) + 신원 단위 부트스트랩 95% CI(10,000회, seed 42), 지표 10분위별 평균 ΔAP_q 그림
  summary 방법 단위 표: Stage 1 지표 평균(train 표본) + Stage 2 ΔmAP(S_on_S). 상관 계수는 내지 않음(SR 3개).
출력: 집계·그림 results/stage2_analysis/, query별 원시 값 data/locked/stage2_analysis/."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
LK = ROOT / "data" / "locked" / "stage2_analysis"
MET = LK / "metrics"
RES = ROOT / "results" / "stage2_analysis"
FIG = RES / "figures"
EVAL = ROOT / "data" / "locked" / "stage2_eval"
CONDS = ["bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman"]
SR = ["swinir", "pisa_sr_sem0.0", "osdhuman"]
MODELS = ["bot", "clipreid", "kpr", "kpr_solider"]
MNAME = {"bot": "BoT", "clipreid": "CLIP-ReID", "kpr": "KPR", "kpr_solider": "KPR+SOLIDER"}
SNAME = {"bicubic": "bicubic", "swinir": "swinir", "pisa_sr_sem0.0": "pisa λ0.0", "osdhuman": "osdhuman"}
METRICS = ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "hf_db_vs_bicubic", "slope_dev_hi", "grid_peak_db", "de2000"]
AFINE = ("afine_all_scale", "fidelity", "natural_dis")
MLAB = {"dinov3_large_cos": "DINOv3 코사인↑", "personvit_base_cos": "PersonViT 코사인↑", "afine_all_scale": "A-FINE 총점↓", "fidelity": "A-FINE 충실도↓",
        "natural_dis": "A-FINE 자연스러움↓", "lf_coh": "저주파 일치도↑", "hf_db_vs_bicubic": "고주파 추가량 dB(방향 없음)", "slope_dev_hi": "기울기 편차(0이 원본)",
        "grid_peak_db": "격자 피크 dB↓", "de2000": "ΔE2000↓"}
B, SEED = 10000, 42
PAT = __import__("re").compile(r"([-\d]+)_c(\d)")


def pidcam(n):
    m = PAT.search(n)
    return int(m.group(1)), int(m.group(2))


# ------------------------------------------------------------------ table
def build_table():
    rows = []
    for split in ("query", "bounding_box_test"):
        emb = {mk: np.load(MET / "emb" / mk / f"{split}__orig.npy") for mk in ("dinov3_large", "personvit_base")}
        fft_b = {}
        for k in (3, 4):
            fft_b[k] = pd.read_csv(MET / "fft" / f"{split}__bicubic__x{k}.csv").set_index("name")
        for c in CONDS:
            for k in (3, 4):
                s = f"{split}__{c}__x{k}"
                f = pd.read_csv(MET / "fft" / f"{s}.csv").set_index("name")
                a = pd.read_csv(MET / "afine" / f"{s}.csv").set_index("name")
                d = f.join(a)
                for mk in emb:
                    d[f"{mk}_cos"] = (emb[mk] * np.load(MET / "emb" / mk / f"{s}.npy")).sum(1)
                d["hf_db_vs_bicubic"] = 10 * np.log10(d["hf_energy"] / fft_b[k].loc[d.index, "hf_energy"])
                if c == "bicubic":
                    d[list(AFINE)] = np.nan  # 기준 이미지와 같아 비교 불가
                d = d.reset_index()
                d.insert(0, "split", split)
                d.insert(0, "k", k)
                d.insert(0, "cond", c)
                rows.append(d)
    T = pd.concat(rows, ignore_index=True)
    T.insert(T.columns.get_loc("name"), "file", T["name"])  # 실제 파일 이름(확장자 없음). Market 원본에 x.jpg.jpg 형식이 일부 있다
    T["name"] = T["name"].str.split(".").str[0]  # 평가 per_query와 같은 키
    pc = T["name"].map(pidcam)
    T["pid"], T["cam"] = [p for p, _ in pc], [c for _, c in pc]
    T.to_csv(LK / "test_per_image_metrics.csv.gz", index=False)
    print("table", T.shape, flush=True)
    return T


def load_table():
    return pd.read_csv(LK / "test_per_image_metrics.csv.gz")


# ------------------------------------------------------------------ bootstrap helpers
def boot_counts(n, dev):
    import torch
    g = np.random.default_rng(SEED)
    idx = torch.from_numpy(g.integers(0, n, (B, n))).to(dev)
    C = torch.zeros((B, n), device=dev)
    C.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    return C


def id_mean_ci(x, pid, dev):
    """이미지 → 신원 평균 → 신원 평균의 평균, 신원 부트스트랩 백분위 CI(Stage 1과 같은 방식)."""
    import torch
    ok = np.isfinite(x)
    x, pid = x[ok], pid[ok]
    ids, inv = np.unique(pid, return_inverse=True)
    m = np.bincount(inv, weights=x) / np.bincount(inv)
    C = boot_counts(len(ids), dev)
    bm = (C @ torch.from_numpy(m).float().to(dev)) / C.sum(1)
    bm = bm.cpu().numpy()
    return float(m.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), len(ids), len(x)


def wspearman(x, y, W):
    """가중(복원추출 횟수) Spearman: 가중 중간 순위의 가중 Pearson. x,y: (n,) 텐서, W: (B,n) 가중치. 동점은 중간 순위."""
    import torch

    def wrank(v):
        order = torch.argsort(v)
        vs = v[order]
        grp = torch.cat([torch.zeros(1, dtype=torch.long, device=v.device), (vs[1:] != vs[:-1]).long()]).cumsum(0)
        ng = int(grp[-1]) + 1
        Ws = W[:, order]
        gw = torch.zeros((W.shape[0], ng), device=v.device, dtype=W.dtype).index_add_(1, grp, Ws)
        before = torch.cumsum(gw, 1) - gw
        r_g = before + (gw + 1) / 2
        r = torch.empty_like(W)
        r[:, order] = r_g[:, grp]
        return r

    rx, ry = wrank(x), wrank(y)
    sw = W.sum(1, keepdim=True)
    mx, my = (W * rx).sum(1, keepdim=True) / sw, (W * ry).sum(1, keepdim=True) / sw
    cov = (W * (rx - mx) * (ry - my)).sum(1)
    return cov / torch.sqrt((W * (rx - mx) ** 2).sum(1) * (W * (ry - my) ** 2).sum(1))


def spearman_ci(x, y, groups, dev):
    """점추정(전체, 가중치 1)과 신원 부트스트랩 CI. groups: 행별 신원 번호(0..G-1)."""
    import torch
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    _, gi = np.unique(groups, return_inverse=True)
    G = gi.max() + 1
    xt, yt = torch.from_numpy(x).double().to(dev), torch.from_numpy(y).double().to(dev)
    rho = float(wspearman(xt, yt, torch.ones((1, len(x)), device=dev, dtype=torch.double))[0])
    C = boot_counts(G, dev).double()
    W = C[:, torch.from_numpy(gi).to(dev)]
    rs = []
    for i in range(0, B, 2500):
        rs.append(wspearman(xt, yt, W[i:i + 2500]).cpu().numpy())
    rs = np.concatenate(rs)
    return rho, float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5)), int(len(x)), int(G)


# ------------------------------------------------------------------ dist (2b)
def step_dist(T, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    for fp in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",):
        if os.path.exists(fp):
            from matplotlib import font_manager
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = ["NanumGothic"]
    s1 = pd.read_csv(ROOT / "results" / "stage1" / "summary.csv")
    p1 = pd.read_csv(ROOT / "results" / "stage1" / "per_image_metrics.csv")
    p1 = p1[p1.method.isin(CONDS)]
    tt = T[(T.pid > 0)]  # 사람 이미지만(junk −1, 방해 0000 제외)
    rows = []
    for c in CONDS:
        for k in (3, 4):
            x = tt[(tt.cond == c) & (tt.k == k)]
            for mtr in METRICS:
                if c == "bicubic" and mtr in AFINE:
                    continue
                a = s1[(s1.method == c) & (s1.k.astype(str) == str(k)) & (s1.metric == mtr)]
                mt, lo, hi, nid, nimg = id_mean_ci(x[mtr].values.astype(float), x.pid.values, dev)
                rows.append({"cond": c, "k": k, "metric": mtr, "train_mean": float(a["mean"].iloc[0]), "train_ci_lo": float(a.ci_lo.iloc[0]), "train_ci_hi": float(a.ci_hi.iloc[0]),
                             "train_n_id": int(a.n_id.iloc[0]), "test_mean": mt, "test_ci_lo": lo, "test_ci_hi": hi, "test_n_id": nid, "test_n_img": nimg,
                             "test_minus_train": mt - float(a["mean"].iloc[0]),
                             "ci_overlap": not (lo > float(a.ci_hi.iloc[0]) or hi < float(a.ci_lo.iloc[0]))})
    D = pd.DataFrame(rows)
    D.to_csv(RES / "train_vs_test_metrics.csv", index=False)
    FIG.mkdir(parents=True, exist_ok=True)
    for mtr in METRICS:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        data, labels, colors = [], [], []
        for c in CONDS:
            if c == "bicubic" and mtr in AFINE:
                continue
            for k in (3, 4):
                a = p1[(p1.method == c) & (p1.k == k)][mtr].dropna().values
                b = tt[(tt.cond == c) & (tt.k == k)][mtr].dropna().values
                data += [a, b]
                labels += [f"{SNAME[c]} x{k}\ntrain", f"{SNAME[c]} x{k}\ntest"]
                colors += ["#8fb3d9", "#e3a36f"]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for p, col in zip(bp["boxes"], colors):
            p.set_facecolor(col)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{MLAB[mtr]}: Stage 1 train sample (blue) vs test query+gallery (orange), image level, no outliers drawn")
        fig.tight_layout()
        fig.savefig(FIG / f"dist_train_test_{mtr}.png", dpi=110)
        plt.close(fig)
    print("dist", D.shape, flush=True)
    return D


# ------------------------------------------------------------------ query (2c)
def per_query(m, k, tr, te):
    return pd.read_csv(EVAL / f"{m}__x{k}" / f"{tr}__on__{te}" / "per_query.csv.gz")


def step_query(T, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_rows, qrows = [], []
    gal = T[(T.split == "bounding_box_test") & (T.pid > 0)]
    qry = T[T.split == "query"]
    for s in SR:
        for k in (3, 4):
            qm = qry[(qry.cond == s) & (qry.k == k)].set_index("name")
            g = gal[(gal.cond == s) & (gal.k == k)]
            # 정답 gallery: 같은 신원, 다른 카메라(Market 규칙). 신원·카메라별 합으로 '다른 카메라 평균'을 계산
            gsum = g.groupby(["pid", "cam"])[METRICS].agg(["sum", "count"])
            for m in MODELS:
                a = per_query(m, k, s, s).set_index("name")
                b = per_query(m, k, "bicubic", "bicubic").set_index("name")
                assert list(a.index) == list(b.index)
                v = (a.valid == 1) & (b.valid == 1)
                d = pd.DataFrame({"pid": a.pid, "cam": a.cam, "dAP": a.ap - b.ap, "ap_S_on_S": a.ap, "ap_C_on_C": b.ap})[v]
                for mtr in METRICS:
                    d[f"q_{mtr}"] = qm.loc[d.index, mtr].values
                    tot = gsum[(mtr, "sum")].groupby(level=0).sum()
                    cnt = gsum[(mtr, "count")].groupby(level=0).sum()
                    same_s = gsum[(mtr, "sum")]
                    same_c = gsum[(mtr, "count")]
                    keys = list(zip(d.pid.values, d.cam.values + 1))  # per_query cam은 0부터, 파일명 카메라는 1부터
                    ss = np.array([same_s.get(kk, 0.0) for kk in keys])
                    sc = np.array([same_c.get(kk, 0) for kk in keys])
                    num = tot.reindex(d.pid.values).values - ss
                    den = cnt.reindex(d.pid.values).values - sc
                    d[f"g_{mtr}"] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
                d.insert(0, "model", m)
                d.insert(1, "k", k)
                d.insert(2, "sr", s)
                qrows.append(d.reset_index())
                pid = d.pid.values
                for mtr in METRICS:
                    if s == "bicubic" and mtr in AFINE:
                        continue
                    for src in ("q", "g"):
                        x = d[f"{src}_{mtr}"].values.astype(float)
                        r, lo, hi, n, G = spearman_ci(x, d.dAP.values.astype(float), pid, dev)
                        # 신원 단위 평균 버전
                        ok = np.isfinite(x)
                        dd = pd.DataFrame({"pid": pid[ok], "x": x[ok], "y": d.dAP.values[ok]}).groupby("pid").mean()
                        r2, lo2, hi2, n2, _ = spearman_ci(dd.x.values, dd.y.values, np.arange(len(dd)), dev)
                        out_rows.append({"model": m, "k": k, "sr": s, "metric": mtr, "source": "query 이미지" if src == "q" else "정답 gallery 평균",
                                         "rho_query": r, "ci_lo_query": lo, "ci_hi_query": hi, "n_query": n, "n_id": G,
                                         "rho_id": r2, "ci_lo_id": lo2, "ci_hi_id": hi2, "n_id_avg": n2,
                                         "mean_dAP": float(d.dAP.mean())})
                print("query", m, k, s, flush=True)
    Q = pd.concat(qrows, ignore_index=True)
    Q.to_csv(LK / "query_dAP_with_metrics.csv.gz", index=False)
    S = pd.DataFrame(out_rows)
    S.to_csv(RES / "spearman_dAP_metrics.csv", index=False)
    # 10분위 그림: 지표마다 (배율 × SR) 격자, 모델별 선
    for mtr in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
        for i, k in enumerate((3, 4)):
            for j, s in enumerate(SR):
                ax = axes[i, j]
                for m in MODELS:
                    d = Q[(Q.model == m) & (Q.k == k) & (Q.sr == s)]
                    x = d[f"q_{mtr}"]
                    if x.notna().sum() < 50:
                        continue
                    dec = pd.qcut(x.rank(method="first"), 10, labels=False)
                    ax.plot(range(1, 11), 100 * d.groupby(dec).dAP.mean().values, marker="o", ms=3, label=MNAME[m])
                ax.axhline(0, color="gray", lw=0.6)
                ax.set_title(f"x{k} {SNAME[s]}", fontsize=9)
                if i == 1:
                    ax.set_xlabel(f"query {mtr} decile (1 = lowest)", fontsize=8)
                if j == 0:
                    ax.set_ylabel("mean ΔAP_q (%p)", fontsize=8)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"ΔAP_q by query-image {mtr} decile (S model on SR test − C model on bicubic test)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"decile_dAP_{mtr}.png", dpi=110)
        plt.close(fig)
    print("query 끝", S.shape, flush=True)
    return S


# ------------------------------------------------------------------ summary (2d)
def step_summary():
    s1 = pd.read_csv(ROOT / "results" / "stage1" / "summary.csv")
    j = pd.read_csv(ROOT / "results" / "stage2" / "judgment.csv")
    rows = []
    for s in SR:
        for k in (3, 4):
            r = {"SR": SNAME[s], "배율": k}
            for mtr in METRICS:
                a = s1[(s1.method == s) & (s1.k.astype(str) == str(k)) & (s1.metric == mtr)]
                r[MLAB[mtr]] = round(float(a["mean"].iloc[0]), 4)
            for m in MODELS:
                x = j[(j.model == m) & (j.k == k) & (j.sr_cond == s) & (j.comparison == "S_on_S") & (j.metric == "mAP")].iloc[0]
                r[f"ΔmAP {MNAME[m]} %p"] = f"{100 * x.delta:+.2f} ({x.label})"
            rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "method_summary.csv", index=False)
    print("summary", len(rows), flush=True)


def main():
    assert os.environ.get("ALLOW_TEST") == "1"
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[1:] or ["table", "dist", "query", "summary"]
    T = build_table() if "table" in steps else (load_table() if {"dist", "query"} & set(steps) else None)
    if "dist" in steps:
        step_dist(T, dev)
    if "query" in steps:
        step_query(T, dev)
    if "summary" in steps:
        step_summary()


if __name__ == "__main__":
    main()
