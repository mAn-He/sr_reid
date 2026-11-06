"""0단계 ×3 방식 비교의 평가 모델 지표(DINOv3-L, PersonViT-B, A-FINE). 잠정 설정(사용자 결정 전): DINOv3 timm 사본(RoPE 보정 없음), PersonViT 정규화 0.5/0.5, A-FINE pyiqa 사본.
출력: results/stage1/step0_x3cmp_embed.csv(이미지별), results/stage1/step0_x3cmp_embed_dist.csv(분포 거리, 100장이라 참고값). srreid-quality env, flock으로 실행."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
import eval_models as em  # noqa: E402

S0 = ROOT / "runs" / "stage1_step0" / "x3cmp"
INP = ROOT / "data" / "sr_input" / "bounding_box_train"
OUT = ROOT / "results" / "stage1"
METHODS = ["osdhuman"] + [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]


def pil(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def main():
    names = sorted(p.stem for p in (S0 / "bicubic" / "x3").glob("*.webp"))
    origs = [pil(INP / f"{n}.png") for n in names]
    sets = [("bicubic", "pil", "x3")] + [(m, w, sub) for m in METHODS for w, sub in (("from_x4", "x3"), ("direct", "x3_direct"))]
    imgs = {(m, w): [pil(S0 / m / sub / f"{n}.webp") for n in names] for m, w, sub in sets}
    rows = {(m, w, n): {"name": n, "method": m, "way": w} for m, w, _ in sets for n in names}
    dist = []
    for build in (em.dinov3, em.personvit):
        e = build()
        eo = e(origs)
        for (m, w), ims in imgs.items():
            es = e(ims)
            cos = (eo * es).sum(1)
            for n, c in zip(names, cos):
                rows[(m, w, n)][f"{e.name}_cos"] = float(c)
            kv, ks = em.kid(eo, es, n_subsets=100, subset=len(names))
            dist.append({"model": e.name, "method": m, "way": w, "n": len(names), "frechet": em.frechet(eo, es), "kid": kv, "kid_std": ks})
        del e
        import torch
        torch.cuda.empty_cache()
    af = em.AFINE()
    for (m, w), ims in imgs.items():
        r = af(origs, ims)
        for i, n in enumerate(names):
            for k, v in r.items():
                rows[(m, w, n)][k] = float(v[i])
    OUT.mkdir(parents=True, exist_ok=True)
    rl = list(rows.values())
    with open(OUT / "step0_x3cmp_embed.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(rl[0]))
        wr.writeheader()
        wr.writerows(rl)
    with open(OUT / "step0_x3cmp_embed_dist.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(dist[0]))
        wr.writeheader()
        wr.writerows(dist)
    import collections
    g = collections.defaultdict(list)
    for r in rl:
        g[(r["method"], r["way"])].append(r)
    for k, rs in g.items():
        print(k, {c: round(float(np.mean([r[c] for r in rs])), 4) for c in rs[0] if c not in ("name", "method", "way")})


if __name__ == "__main__":
    main()
