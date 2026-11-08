"""Stage 1 2단계: 평가 모델 지표(DINOv3-L, PersonViT-B 임베딩, A-FINE). srreid-quality env, GPU.
원본 128×64와 실제 SR만 쓴다(정답 없음). 설정은 2026-11-07 사용자 결정(stage1/eval_models.py 참조).

- 임베딩: 원본·SR·기준선 모두 PIL BICUBIC으로 256×128로 맞춘 뒤 모델별 정규화, 최종 norm 뒤 CLS 토큰, L2 정규화.
  DINOv3: timm 변환본 ViT-L, RoPE 주기 bf16 보정, ImageNet 평균·표준편차. PersonViT: ViT-B teacher backbone, mean/std 0.5/0.5.
- 해석 기준선(원본 128×64에서 만든 뒤 같은 방식으로 임베딩):
  blur_s05 = 가우시안 블러 σ=0.5 원본 픽셀(scipy.ndimage.gaussian_filter, 채널별, mode="reflect", 결과 반올림 uint8)
  jpeg_q90 = PIL JPEG 품질 90(PIL 기본 4:2:0 크로마 서브샘플링)으로 저장 후 다시 읽음
  othercam = 같은 신원의 다른 카메라 원본(우선 표본 안에서, 없으면 train 전체에서, seed 42로 하나 고름. 쌍 목록 results/stage1/baseline_othercam_pairs.csv)
- A-FINE: dis = SR(원 크기), ref = 원본을 SR 크기로 PIL BICUBIC 확대. 출력은 모두 낮을수록 좋다(공식 README "the lower, the better").
  bicubic 방법은 dis와 ref가 같은 이미지이므로 비교 불가(값은 저장하되 보고에서 표시).
출력(재시작 시 있는 것은 건너뜀):
  runs/stage1_metrics/emb/<model>/<set>.npy (float32, 표본 순서), runs/stage1_metrics/afine/<set>.csv
  set 이름: orig, base_blur_s05, base_jpeg_q90, base_othercam, <method>__x<k>
2,250장이 다 있는 SR 집합만 처리한다(생성 중에도 반복 실행 가능)."""
import argparse
import csv
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
import eval_models as em  # noqa: E402

SAMPLE = ROOT / "configs" / "stage1_sample.csv"
INP = ROOT / "data" / "sr_input" / "bounding_box_train"
SR = ROOT / "runs" / "stage1_sample"
MET = ROOT / "runs" / "stage1_metrics"
RES = ROOT / "results" / "stage1"
METHODS = ["bicubic", "lanczos", "realesrgan", "osdhuman", "swinir", "hat", "mambairv2"] + [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]
MODELS = {"dinov3_large": lambda: em.dinov3("large", rope_bf16=True), "personvit_base": lambda: em.personvit("base", norm=(0.5, 0.5))}


def sample():
    return list(csv.DictReader(open(SAMPLE)))


def names():
    return [Path(r["name"]).stem for r in sample()]


def pil(p):
    with Image.open(p) as im:
        return im.convert("RGB")


def othercam_pairs():
    """표본 이미지마다 같은 신원·다른 카메라 원본 하나. 표본 안 후보 우선, 없으면 train 전체. seed 42."""
    out = RES / "baseline_othercam_pairs.csv"
    if out.exists():
        return {r["name"]: r["pair"] for r in csv.DictReader(open(out))}
    rng = np.random.default_rng(42)
    rows = sample()
    allp = sorted(p.stem for p in INP.glob("*.png"))
    by_pid = {}
    for n in allp:
        by_pid.setdefault(n.split("_")[0], []).append(n)
    samp = {}
    for r in rows:
        samp.setdefault(r["pid"], []).append(Path(r["name"]).stem)
    res = []
    for r in rows:
        n = Path(r["name"]).stem
        pid, cam = n.split("_")[0], n.split("_")[1][:2]
        cand = sorted(x for x in samp[r["pid"]] if x.split("_")[1][:2] != cam)
        src = "sample"
        if not cand:
            cand = sorted(x for x in by_pid[pid] if x.split("_")[1][:2] != cam)
            src = "train"
        pair = cand[rng.integers(len(cand))] if cand else ""
        res.append({"name": n, "pair": pair, "source": src if pair else "none"})
    RES.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")  # GPU 두 작업이 동시에 만들 수 있으므로 원자적 교체(내용은 결정적)
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, ["name", "pair", "source"])
        w.writeheader()
        w.writerows(res)
    tmp.replace(out)
    return {r["name"]: r["pair"] for r in res}


def blur(im):
    from scipy.ndimage import gaussian_filter
    a = np.asarray(im, np.float64)
    b = np.stack([gaussian_filter(a[..., c], 0.5, mode="reflect") for c in range(3)], -1)
    return Image.fromarray(np.clip(np.round(b), 0, 255).astype(np.uint8))


def jpeg(im):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return pil(buf)


def sr_sets(ns, methods=None):
    out = []
    for m in (METHODS if methods is None else [x for x in methods if x in METHODS]):  # --methods none → SR 집합 없음
        for k in (3, 4):
            d = SR / m / f"x{k}"
            if all((d / f"{n}.webp").exists() for n in ns):
                out.append((f"{m}__x{k}", d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["emb", "afine"])
    ap.add_argument("--methods", nargs="*", default=None, help="이 방법들의 SR 집합만(A100 GPU 분담용). 기본: 전부")
    ap.add_argument("--no-base", action="store_true", help="orig·기준선 임베딩을 만들지 않는다(다른 GPU 작업이 맡을 때)")
    ap.add_argument("--emb-bs", type=int, default=256, help="임베딩 배치(A100 80GB, 2026-11-08 사용자 허용: VRAM이 허용하는 큰 배치)")
    ap.add_argument("--afine-bs", type=int, default=128)
    a = ap.parse_args()
    ns = names()
    origs = [pil(INP / f"{n}.png") for n in ns]
    pairs = othercam_pairs()
    sets = sr_sets(ns, a.methods)
    print(f"완성된 SR 집합 {len(sets)} (대상 방법 {a.methods or '전부'})", flush=True)
    if "emb" in a.parts:
        for mk, build in MODELS.items():
            od = MET / "emb" / mk
            od.mkdir(parents=True, exist_ok=True)
            base = [] if a.no_base else ["orig", "base_blur_s05", "base_jpeg_q90", "base_othercam"]
            todo = [s for s in base + [s for s, _ in sets] if not (od / f"{s}.npy").exists()]
            if not todo:
                continue
            e = build()
            for s in todo:
                if s == "orig":
                    ims = origs
                elif s == "base_blur_s05":
                    ims = [blur(o) for o in origs]
                elif s == "base_jpeg_q90":
                    ims = [jpeg(o) for o in origs]
                elif s == "base_othercam":
                    ims = [pil(INP / f"{pairs[n]}.png") if pairs[n] else Image.new("RGB", (64, 128)) for n in ns]
                else:
                    d = dict(sets)[s]
                    ims = [pil(d / f"{n}.webp") for n in ns]
                v = e(ims, bs=a.emb_bs)
                if s == "base_othercam":
                    v[[not pairs[n] for n in ns]] = np.nan
                np.save(od / f"{s}.npy.tmp.npy", v.astype(np.float32))
                (od / f"{s}.npy.tmp.npy").replace(od / f"{s}.npy")
                print(mk, s, v.shape, flush=True)
            del e
            import torch
            torch.cuda.empty_cache()
    if "afine" in a.parts:
        od = MET / "afine"
        od.mkdir(parents=True, exist_ok=True)
        todo = [(s, d) for s, d in sets if not (od / f"{s}.csv").exists()]
        if todo:
            af = em.AFINE()
            for s, d in todo:
                r = af(origs, [pil(d / f"{n}.webp") for n in ns], bs=a.afine_bs)
                tmp = od / f"{s}.csv.tmp"
                with open(tmp, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["name"] + list(r))
                    for i, n in enumerate(ns):
                        w.writerow([n] + [repr(float(r[k][i])) for k in r])
                tmp.replace(od / f"{s}.csv")
                print("afine", s, {k: float(np.median(v)) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
