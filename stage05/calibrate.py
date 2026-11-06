"""G.6 보정 실험. 원본(train 신원)을 ×1/k PIL BICUBIC 축소 → SR ×k → 원본과 full-reference 비교(PSNR, SSIM, LPIPS, DISTS, DreamSim, LR-PSNR, ΔE2000).
k=3은 128×64가 3으로 나뉘지 않으므로 126×63으로 가운데 크롭 후 축소한다. k=4는 파일럿과 같은 조건이면 재사용한다(--pilot-check로 확인, --pilot로 채점).
--prepare: 축소 입력을 쓴다. --score: 폴더 쌍을 채점해 JSON으로 저장한다. 수치만 쓴다. srreid-quality env."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

CAL = ROOT / "runs" / "stage05" / "calibration"


def reference(p, k):
    """비교 기준 원본. k로 나뉘도록 가운데 크롭."""
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    h2, w2 = h // k * k, w // k * k
    y, x = (h - h2) // 2, (w - w2) // 2
    return a[y:y + h2, x:x + w2]


def downscale(a, k):
    return np.asarray(Image.fromarray(a).resize((a.shape[1] // k, a.shape[0] // k), Image.BICUBIC))


def names(n, seed=42):
    """train 신원 전용. val 100명도 원본 이미지는 품질 보정에만 쓰고 ReID 판단에는 쓰지 않는다."""
    rows = list(csv.DictReader(open(ROOT / "data" / "processed" / "manifests" / "split.csv")))
    pool = sorted(r["name"] for r in rows if r["split"] == "train651" and r["role"] == "train")
    idx = np.random.default_rng(seed).permutation(len(pool))[:n]
    return [pool[i] for i in sorted(idx)]


def prepare(k, n):
    d = CAL / f"x{k}" / "lr"
    gt = CAL / f"x{k}" / "gt"
    d.mkdir(parents=True, exist_ok=True)
    gt.mkdir(parents=True, exist_ok=True)
    for name in names(n):
        ref = reference(ROOT / "data" / "market" / "bounding_box_train" / f"{name}.jpg", k)
        stem = name
        Image.fromarray(ref).save(gt / f"{stem}.png")
        Image.fromarray(downscale(ref, k)).save(d / f"{stem}.png")
    print(json.dumps({"k": k, "n": n, "lr": str(d), "gt": str(gt)}))


def pilot_check():
    """k=4 재사용 조건: 파일럿 LR(val 100명의 query 404장, train 신원)이 원본의 PIL BICUBIC ×1/4와 픽셀 단위로 같은가."""
    gt = CAL / "x4_pilot" / "gt"
    lr = CAL / "x4_pilot" / "lr"
    gt.mkdir(parents=True, exist_ok=True)
    lr.mkdir(parents=True, exist_ok=True)
    n, same = 0, 0
    for p in sorted((ROOT / "data" / "processed" / "val_query_lr").glob("*.png")):
        ref = reference(ROOT / "data" / "market" / "bounding_box_train" / f"{p.stem}.jpg", 4)
        with Image.open(p) as im:
            pl = np.asarray(im.convert("RGB"))
        n += 1
        same += int(np.array_equal(downscale(ref, 4), pl))
        Image.fromarray(ref).save(gt / p.name)
        Image.fromarray(pl).save(lr / p.name)
    out = {"n": n, "lr_pixel_equal": same, "reusable": n > 0 and n == same}
    (CAL / "x4_pilot" / "check.json").write_text(json.dumps(out))
    print(json.dumps(out))


def score(k, sr_dir, tag, dreamsim=True, pilot=False):
    from stage05 import quality_tools as Q
    per = Q.Perceptual(dreamsim=dreamsim)
    base = CAL / (f"x{k}_pilot" if pilot else f"x{k}")
    gt_dir, lr_dir = base / "gt", base / "lr"
    rows = []
    for g in sorted(gt_dir.glob("*.png")):
        cands = [Path(sr_dir) / f"{g.stem}{s}" for s in (".webp", ".png")]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            continue
        with Image.open(p) as im:
            sr = np.asarray(im.convert("RGB"))
        with Image.open(g) as im:
            ref = np.asarray(im.convert("RGB"))
        with Image.open(lr_dir / g.name) as im:
            lr = np.asarray(im.convert("RGB"))
        if sr.shape != ref.shape:
            rows.append({"name": g.stem, "error": f"크기 불일치 {sr.shape} {ref.shape}"})
            continue
        rows.append({"name": g.stem, **per(sr, ref), **Q.lr_psnr(sr, lr, k), **Q.color_delta_e(sr, ref)})
    ok = [r for r in rows if "error" not in r]
    keys = [c for c in ok[0] if c != "name"] if ok else []
    out = {"tag": tag, "k": k, "n": len(ok), "n_error": len(rows) - len(ok), **{c: float(np.mean([r[c] for r in ok])) for c in keys}}
    (base / f"{tag}.json").write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=0))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-dreamsim", action="store_true")
    ap.add_argument("--pilot-check", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.pilot_check:
        pilot_check()
    if a.prepare:
        prepare(a.k, a.n)
    if a.score:
        score(a.k, a.score, a.tag or Path(a.score).name, not a.no_dreamsim, a.pilot)
