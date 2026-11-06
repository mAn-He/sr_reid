"""0단계: osdhuman·pisa_sr ×3 두 방식 비교(표본 앞 100장).
  from_x4: ×4 출력을 PIL BICUBIC으로 ×3 크기로 축소.  direct: 공식 resize 경로에 upscale=3을 넣어 직접 생성.
FFT·ΔE2000 지표(stage1/fft_color.py)와 평가 모델 지표(있으면 results/stage1/step0_x3cmp_embed.csv를 합침)를 내고,
비포·애프터 그림(원본 | bicubic | 방법별 from_x4 | direct, 머리·상의 확대 포함)을 만든다. 그림은 Claude가 열어 보지 않는다.
출력: results/stage1/step0_x3cmp_per_image.csv, results/stage1/step0_x3cmp_summary.csv, results/stage1/figures/step0_x3cmp_*.png. srreid-core env."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
import fft_color as fc  # noqa: E402

S0 = ROOT / "runs" / "stage1_step0" / "x3cmp"
INP = ROOT / "data" / "sr_input" / "bounding_box_train"
OUT = ROOT / "results" / "stage1"
FIG = OUT / "figures"
METHODS = ["osdhuman"] + [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]


def names():
    return sorted(p.stem for p in (S0 / "bicubic" / "x3").glob("*.webp"))


def load(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def metrics():
    rows = []
    for n in names():
        o = load(INP / f"{n}.png")
        for m in ["bicubic"] + METHODS:
            for way, sub in (("from_x4", "x3"), ("direct", "x3_direct")):
                if m == "bicubic" and way == "direct":
                    continue
                p = S0 / m / sub / f"{n}.webp"
                if not p.exists():
                    continue
                s = load(p)
                r = {"name": n, "method": m, "way": way if m != "bicubic" else "pil", **fc.fft_metrics(o, s, 3), "de2000": fc.de2000(o, s)}
                rows.append(r)
    return rows


def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["method"], r["way"])].append(r)
    keys = [k for k in rows[0] if k not in ("name", "method", "way")]
    out = []
    for (m, w), rs in g.items():
        d = {"method": m, "way": w, "n": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs], float)
            d[k] = float(np.nanmean(v))
        out.append(d)
    return out


def tile(arr, scale_to):
    return Image.fromarray(arr).resize(scale_to, Image.NEAREST)


def figure(sel, tag):
    """한 줄 = 이미지 하나: 원본(최근접 확대) | bicubic ×3 | 방법별 from_x4, direct. 아래 줄에 머리·상의 확대(위 45% 영역)."""
    cols = [("original (NN zoom)", None, None), ("bicubic x3", "bicubic", "x3")]
    for m in METHODS:
        lab = m.replace("pisa_sr_sem", "pisa L")  # PIL 기본 글꼴은 latin-1만 된다
        cols += [(f"{lab} x4->x3", m, "x3"), (f"{lab} direct x3", m, "x3_direct")]
    W_, H_ = 192, 384
    pad, head = 4, 30
    crop_h = int(H_ * 0.45)
    canvas = Image.new("RGB", (len(cols) * (W_ + pad), head + len(sel) * (H_ + crop_h * 2 // 1 + 2 * pad)), "white")
    dr = ImageDraw.Draw(canvas)
    for j, (lab, _, _) in enumerate(cols):
        dr.text((j * (W_ + pad) + 2, 2), lab[:30], fill="black")
    y = head
    for n in sel:
        o = load(INP / f"{n}.png")
        for j, (lab, m, sub) in enumerate(cols):
            a = o if m is None else load(S0 / m / sub / f"{n}.webp")
            im = tile(a, (W_, H_))
            canvas.paste(im, (j * (W_ + pad), y))
            cr = im.crop((0, 0, W_, crop_h)).resize((W_, crop_h * 2), Image.NEAREST)  # 위 45%(머리·상의) 2배 확대
            canvas.paste(cr, (j * (W_ + pad), y + H_ + pad))
        dr.text((2, y + 2), n, fill="red")
        y += H_ + crop_h * 2 + 2 * pad
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"step0_x3cmp_{tag}.png"
    canvas.save(p)
    return p


def write(name, rows):
    keys = list(rows[0])
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = metrics()
    write("step0_x3cmp_per_image.csv", rows)
    summ = summarize(rows)
    write("step0_x3cmp_summary.csv", summ)
    for d in summ:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}, ensure_ascii=False))
    ns = names()
    rng = np.random.default_rng(42)
    sel = [ns[i] for i in sorted(rng.choice(len(ns), 6, replace=False))]
    print(figure(sel, "random6"))
