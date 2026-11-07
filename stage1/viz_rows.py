"""Stage 1 3단계 비포·애프터 사진 비교. srreid-quality env, CPU. 그림은 Claude가 열어 보지 않는다(수치로만 검사).
한 줄 = 한 이미지: 원본(최근접 확대) | bicubic | 나머지 SR 11종(METHODS 순서). 각 칸은 위에 전체 모습(높이 256, PIL BICUBIC 축소),
아래에 얼굴·상의 확대(이미지 위쪽 45% 영역을 SR 원 해상도 그대로 잘라 붙임, 원본 칸은 최근접 확대)를 둔다.
칸 아래에는 해당 지표 값을 적고, 대상 방법 칸은 빨간 테두리.

세트(배율별):
 a. worst/x<k>/<method>__<metric>.png: 방법·배율·지표마다 하위 10장(지표별 "나쁜" 쪽 정의는 WORST)
 b. common_worst_x<k>.png: 같은 방법에서 2개 이상 지표의 하위 10장에 든 이미지를 등장 횟수 순으로(최대 30줄). 목록 전부는 common_worst.csv.
    어떤 방법에도 공통 최악이 없으면 그 사실을 CSV에 적고 (a)를 본다.
 c. allbad_x<k>.png / specific_x<k>.png: 모든 방법에서 나쁜 이미지(원본 문제 후보)와 특정 방법에서만 나쁜 이미지. 정의는 classify().
 d. random20_x<k>.png: 무작위 20장(seed 42)
JPEG 썸네일(HTML 내장용)은 report.py가 만든다."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
from metrics_embed import INP, METHODS, RES, SR  # noqa: E402
from viz_plots import random20  # noqa: E402

FIG = RES / "figures"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LAB = {m: m.replace("pisa_sr_sem", "pisa λ") for m in METHODS}
# 하위(나쁜 쪽) 정의: low = 값이 작을수록 나쁨, high = 클수록 나쁨, abs = 0에서 멀수록
WORST = {"dinov3_large_cos": "low", "personvit_base_cos": "low", "afine_all_scale": "high", "fidelity": "high", "natural_dis": "high",
         "lf_coh": "low", "slope_dev_hi": "abs", "grid_peak_db": "high", "de2000": "high"}
AFINE = ("afine_all_scale", "fidelity", "natural_dis")
TH, CROP = 256, 0.45


def badness(x, how):
    return {"low": -x, "high": x, "abs": np.abs(x)}[how]


def ld(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def cell(n, m, k):
    if m is None:
        o = ld(INP / f"{n}.png")
        full = o.resize((64 * k, 128 * k), Image.NEAREST)
    else:
        full = ld(SR / m / f"x{k}" / f"{n}.webp")
    W, H = full.size
    thumb = full.resize((TH // 2, TH), Image.BICUBIC) if H != TH else full
    crop = full.crop((0, 0, W, int(round(H * CROP))))
    return thumb, crop


def compose(rows, k, path, title, metric=None, pi=None):
    """rows: [(name, row_label, target_method or None)]"""
    f12, f10 = ImageFont.truetype(FONT, 13), ImageFont.truetype(FONT, 11)
    cols = [None] + METHODS
    cw = 64 * k + 6
    ch = TH + int(round(128 * k * CROP)) + 6 + (16 if metric else 0) + 18
    head = 46
    canvas = Image.new("RGB", (cw * len(cols), head + ch * len(rows)), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 2), title, fill="black", font=f12)
    for j, m in enumerate(cols):
        dr.text((j * cw + 3, 24), "original (NN)" if m is None else LAB[m], fill="black", font=f12)
    vals = None
    if metric and pi is not None:
        vals = pi[pi.k == k].set_index(["method", "name"])[metric]
    for i, (n, lab, tgt) in enumerate(rows):
        y = head + i * ch
        dr.text((4, y), lab, fill="red", font=f10)
        for j, m in enumerate(cols):
            thumb, crop = cell(n, m, k)
            x = j * cw + 3
            canvas.paste(thumb, (x + (64 * k - thumb.size[0]) // 2, y + 16))
            canvas.paste(crop, (x, y + 16 + TH + 3))
            if vals is not None and m is not None:
                v = vals.get((m, n), np.nan)
                txt = "n/a (=ref)" if (metric in AFINE and m == "bicubic") else f"{v:.4g}"
                dr.text((x, y + 16 + TH + 3 + crop.size[1] + 1), txt, fill="black", font=f10)
            if tgt is not None and m == tgt:
                dr.rectangle([x - 2, y + 14, x + 64 * k + 1, y + ch - 3], outline="red", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=False, compress_level=6)
    return path


def bottom10(pi, m, k, c):
    d = pi[(pi.method == m) & (pi.k == k)][["name", c]].dropna()
    b = badness(d[c].values, WORST[c])
    o = np.argsort(-b, kind="stable")[:10]
    return list(d.name.values[o]), list(d[c].values[o])


def worst_sets(pi, only_tables=False):
    rows_csv, common = [], []
    for k in (3, 4):
        for m in METHODS:
            seen = {}
            for c in WORST:
                if c in AFINE and m == "bicubic":
                    continue
                ns, vs = bottom10(pi, m, k, c)
                for r, (n, v) in enumerate(zip(ns, vs)):
                    rows_csv.append({"k": k, "method": m, "metric": c, "worst_side": WORST[c], "rank": r + 1, "name": n, "value": v})
                    seen.setdefault(n, []).append(c)
                if not only_tables:
                    compose([(n, f"#{r + 1} {n}  {c}={v:.4g}", m) for r, (n, v) in enumerate(zip(ns, vs))], k,
                            FIG / "worst" / f"x{k}" / f"{m}__{c}.png", f"x{k} {LAB[m]}: 10 worst by {c} (worst side: {WORST[c]})", c, pi)
            multi = sorted([(len(v), n, v) for n, v in seen.items() if len(v) >= 2], key=lambda t: (-t[0], t[1]))
            if not multi:
                common.append({"k": k, "method": m, "name": "", "count": 0, "metrics": "공통 최악 없음 → (a) 지표별 최악 참조"})
            for cnt, n, v in multi:
                common.append({"k": k, "method": m, "name": n, "count": cnt, "metrics": " ".join(v)})
    pd.DataFrame(rows_csv).to_csv(RES / "worst10.csv", index=False)
    cw = pd.DataFrame(common)
    cw.to_csv(RES / "common_worst.csv", index=False)
    if not only_tables:
        for k in (3, 4):
            d = cw[(cw.k == k) & (cw["count"] >= 2)].sort_values(["count", "method", "name"], ascending=[False, True, True]).head(30)
            if len(d):
                compose([(r["name"], f"{LAB[r['method']]}: in bottom-10 of {r['count']} metrics ({r['metrics']})  {r['name']}", r["method"]) for _, r in d.iterrows()],
                        k, FIG / f"common_worst_x{k}.png", f"x{k}: images in the bottom-10 of >=2 metrics for the same method (sorted by count)")


def classify(pi):
    """이미지마다 방법 내 백분위(0 = 그 방법에서 가장 나쁨, WORST 방향 기준)를 구한다. 배율·지표별로:
    - 공통 나쁨(allbad): 비교 가능한 모든 방법에서 백분위 ≤ 10% (A-FINE 지표는 bicubic 제외 11종)
    - 특정 방법만 나쁨(specific): 그 방법에서 하위 10장 안이고, 나머지 방법의 백분위 중앙값 ≥ 50%"""
    out = []
    for k in (3, 4):
        for c in WORST:
            ms = [m for m in METHODS if not (c in AFINE and m == "bicubic")]
            d = pi[(pi.k == k) & pi.method.isin(ms)].pivot(index="name", columns="method", values=c)[ms]
            b = badness(d.values, WORST[c])
            pct = pd.DataFrame(b, index=d.index, columns=ms).rank(pct=True, ascending=False)  # 가장 나쁨 → 작은 백분위
            allbad = pct.index[(pct <= 0.10).all(1)]
            for n in allbad:
                out.append({"k": k, "metric": c, "kind": "allbad", "name": n, "method": "", "max_pct": float(pct.loc[n].max()), "median_pct_others": ""})
            for m in ms:
                worst10 = pct[m].nsmallest(10).index
                others = pct.drop(columns=m)
                for n in worst10:
                    med = float(others.loc[n].median())
                    if med >= 0.5:
                        out.append({"k": k, "metric": c, "kind": "specific", "name": n, "method": m, "max_pct": float(pct.loc[n, m]), "median_pct_others": med})
    df = pd.DataFrame(out)
    df.to_csv(RES / "allbad_vs_specific.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=["random", "worst", "classify"])
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    pi = pd.read_csv(RES / "per_image_metrics.csv")
    if "random" in a.parts and not a.tables_only:
        for k in (3, 4):
            compose([(n, f"random #{i + 1} {n}", None) for i, n in enumerate(random20())], k, FIG / f"random20_x{k}.png", f"x{k}: 20 random sample images (seed 42)")
    if "worst" in a.parts:
        worst_sets(pi, a.tables_only)
    if "classify" in a.parts:
        df = classify(pi)
        if not a.tables_only:
            for k in (3, 4):
                d = df[(df.k == k) & (df.kind == "allbad")]
                g = d.groupby("name").metric.apply(list).reset_index()
                g["cnt"] = g.metric.apply(len)
                g = g.sort_values(["cnt", "name"], ascending=[False, True]).head(20)
                if len(g):
                    compose([(r["name"], f"bad in ALL methods for {r['cnt']} metric(s): {' '.join(r['metric'])}  {r['name']}", None) for _, r in g.iterrows()],
                            k, FIG / f"allbad_x{k}.png", f"x{k}: images in the worst 10% of every method (candidate source-image problems)")
                d = df[(df.k == k) & (df.kind == "specific")].sort_values(["median_pct_others", "method"], ascending=[False, True])
                d = d.drop_duplicates(["name", "method"]).head(30)
                if len(d):
                    compose([(r["name"], f"{LAB[r['method']]} only: bottom-10 by {r['metric']}, other methods median percentile {r['median_pct_others']:.2f}  {r['name']}", r["method"])
                             for _, r in d.iterrows()], k, FIG / f"specific_x{k}.png", f"x{k}: images bad in one method only (others' median percentile >= 0.5)")
    print("비포·애프터 완료")


if __name__ == "__main__":
    main()
