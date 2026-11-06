"""G.3 신원 보존 코사인. 평가 ReID와 다른 모델의 특징으로 계산한다(Stage 1은 BoT 평가이므로 CLIP-ReID 공개 가중치). srreid-clipreid env.
입력은 같은 파일명의 두 폴더. 두 쪽 모두 공식 입력 256×128로 맞춘다(T.Resize, 공식 평가 변환)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from stage05.clip_run import setup  # noqa: E402


@torch.no_grad()
def features(model, tf, paths, batch=64):
    out = []
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            with Image.open(p) as im:
                xs.append(tf(im.convert("RGB")))
        out.append(torch.nn.functional.normalize(model(torch.stack(xs).cuda(), cam_label=None, view_label=None), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sr", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torchvision.transforms as T
    args = argparse.Namespace(h=256, w=128, batch=64, workers=0, public=1)
    cfg, model = setup(args)
    model = model.cuda().eval()
    tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    sr = {p.name.split(".")[0]: p for p in Path(a.sr).iterdir() if p.suffix in (".webp", ".png")}
    ref = {p.name.split(".")[0]: p for p in Path(a.ref).iterdir() if p.suffix in (".jpg", ".png", ".webp")}
    common = sorted(set(sr) & set(ref))
    if a.limit:
        common = common[:a.limit]
    fr, fs = features(model, tf, [str(ref[n]) for n in common]), features(model, tf, [str(sr[n]) for n in common])
    cos = (fr * fs).sum(1)
    print(json.dumps({"n": len(common), "mean_cosine": float(cos.mean()), "p05": float(np.percentile(cos, 5))}))


if __name__ == "__main__":
    main()
