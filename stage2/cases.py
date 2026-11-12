"""Stage 2 사후 분석 3: 성공·실패 사례(평가 완료 후 분석, 판정과 별개). srreid-quality env. ALLOW_TEST=1에서만. 그림은 열어 보지 않는다(수치로만 검사).
SR 조건·배율마다 4개 모델의 ΔAP_q 평균(4모델 모두 유효한 query)으로 가장 손해 본 20개(worst)와 이득 본 20개(best).
그림 한 줄: 원본 query(최근접 ×2) | bicubic query | SR query | BoT C top-5 | BoT S top-5 | KPR+SOLIDER C top-5 | KPR+SOLIDER S top-5
(C = bicubic 모델 + bicubic test gallery, S = SR 모델 + SR test gallery. 테두리 초록 = 같은 신원(정답), 빨강 = 오답. top-5는 Market 규칙: 같은 신원·같은 카메라 제외).
출력: results/stage2_analysis/cases.csv(지표 값 포함), figures/cases/<sr>_x<k>_<worst|best>.png(커밋 안 함)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
LK = ROOT / "data" / "locked" / "stage2_analysis"
RES = ROOT / "results" / "stage2_analysis"
FIG = RES / "figures" / "cases"
EVAL = ROOT / "data" / "locked" / "stage2_eval"
SR = ["swinir", "pisa_sr_sem0.0", "osdhuman"]
MODELS = ["bot", "clipreid", "kpr", "kpr_solider"]
SNAME = {"swinir": "swinir", "pisa_sr_sem0.0": "pisa l0.0", "osdhuman": "osdhuman"}
METRICS = ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "hf_db_vs_bicubic", "slope_dev_hi", "grid_peak_db", "de2000"]
PY = {"bot": "/workspace/miniforge3/envs/srreid-core/bin/python", "kpr_solider": "/workspace/miniforge3/envs/srreid-kpr/bin/python"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TH, TW = 128, 64


def topk(model, k, tr, te, names, tag):
    out = LK / "cases" / f"topk_{model}_x{k}_{tr}_on_{te}_{tag}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        nf = out.with_suffix(".txt")
        nf.write_text("\n".join(names))
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        rc = subprocess.call([PY[model], "stage2/topk.py", model, str(EVAL / f"{model}__x{k}" / f"{tr}__on__{te}"), str(nf), str(out)], env=env, cwd=ROOT)
        if rc:
            raise SystemExit(f"topk 실패 {model} x{k} {tr} on {te}")
    return json.loads(out.read_text())


def img(path, nn2=False):
    with Image.open(path) as im:
        im = im.convert("RGB")
    if nn2:
        im = im.resize((TW, TH), Image.NEAREST)
    return im.resize((TW, TH), Image.BICUBIC) if im.size != (TW, TH) else im


def _p(base, n, ext):
    p = base / f"{n}{ext}"
    return p if p.exists() else base / f"{n}.jpg{ext}"  # Market 원본의 x.jpg.jpg 형식


def qpath(cond, k, n):
    return _p(ROOT / "data" / cond / f"x{k}" / "query", n, ".webp")


def gpath(cond, k, n):
    return _p(ROOT / "data" / cond / f"x{k}" / "bounding_box_test", n, ".webp")


def compose(rows, s, k, tops, title, path):
    f11, f13 = ImageFont.truetype(FONT, 11), ImageFont.truetype(FONT, 13)
    cw, gap = TW + 6, 10
    groups = [("query: orig | bicubic | SR", 3)] + [(t, 5) for t in ("BoT C (bicubic)", "BoT S (SR)", "KPR+SOLIDER C (bicubic)", "KPR+SOLIDER S (SR)")]
    W = sum(n * cw + gap for _, n in groups)
    rh = TH + 30
    canvas = Image.new("RGB", (W, 44 + rh * len(rows)), "white")
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 2), title, fill="black", font=f13)
    x = 0
    for g, n in groups:
        dr.text((x + 2, 24), g, fill="black", font=f11)
        x += n * cw + gap
    for i, r in enumerate(rows):
        y = 44 + i * rh
        n = r["name"]
        dr.text((2, y), f"#{i + 1} {n}  mean dAP {100 * r['mean_dAP']:+.1f}%p  " + " ".join(f"{m}:{100 * r[f'dAP_{m}']:+.0f}" for m in MODELS), fill="black", font=f11)
        cells = [img(_p(ROOT / "data" / "locked" / "sr_input" / "query", n, ".png"), nn2=True), img(qpath("bicubic", k, n)), img(qpath(s, k, n))]
        x = 0
        for c in cells:
            canvas.paste(c, (x + 3, y + 14))
            x += cw
        x += gap
        for key in ("bot_C", "bot_S", "kpr_solider_C", "kpr_solider_S"):
            cond = "bicubic" if key.endswith("_C") else s
            for t in tops[key][n]:
                canvas.paste(img(gpath(cond, k, t["name"])), (x + 3, y + 14))
                dr.rectangle([x + 1, y + 12, x + TW + 4, y + 14 + TH + 1], outline="green" if t["correct"] else "red", width=2)
                x += cw
            x += gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, compress_level=6)


def main():
    assert os.environ.get("ALLOW_TEST") == "1"
    Q = pd.read_csv(LK / "query_dAP_with_metrics.csv.gz")
    T = pd.read_csv(LK / "test_per_image_metrics.csv.gz")
    out = []
    for s in SR:
        for k in (3, 4):
            d = Q[(Q.sr == s) & (Q.k == k)].pivot_table(index="name", columns="model", values="dAP").dropna()
            d.columns = [f"dAP_{c}" for c in d.columns]
            d["mean_dAP"] = d.mean(1)
            d = d.sort_values("mean_dAP", kind="stable")
            qm = T[(T.cond == s) & (T.k == k) & (T.split == "query")].set_index("name")[METRICS]
            for kind, sel in (("worst", d.head(20)), ("best", d.tail(20).iloc[::-1])):
                names = list(sel.index)
                tops = {}
                for m in ("bot", "kpr_solider"):
                    tops[f"{m}_C"] = topk(m, k, "bicubic", "bicubic", names, f"{kind}_{s}")
                    tops[f"{m}_S"] = topk(m, k, s, s, names, f"{kind}_{s}")
                rows = []
                for rank, n in enumerate(names):
                    r = {"sr": s, "k": k, "kind": kind, "rank": rank + 1, "name": n, "pid": int(n.split("_")[0]), **sel.loc[n].to_dict(), **qm.loc[n].to_dict()}
                    for key in tops:
                        r[f"top5_correct_{key}"] = sum(t["correct"] for t in tops[key][n])
                    rows.append(r)
                out += rows
                compose(rows, s, k, tops, f"x{k} {SNAME[s]}: 20 queries with the {'largest loss' if kind == 'worst' else 'largest gain'} in mean dAP_q over 4 models",
                        FIG / f"{s}_x{k}_{kind}.png")
                print("cases", s, k, kind, flush=True)
    pd.DataFrame(out).to_csv(RES / "cases.csv", index=False)


if __name__ == "__main__":
    main()
