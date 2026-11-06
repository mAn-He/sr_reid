"""G.4 속성 일치율. 원본 대비 SR의 예측 일치율을 색 / 비색으로 나눈다. 입력은 같은 파일명의 두 폴더. srreid-quality env."""
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

from stage05.attr_train import MEAN, SIZE, STD, build_model, forward, is_color  # noqa: E402


def predict(model, binary, paths, batch=128):
    out = []
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            with Image.open(p) as im:
                a = np.asarray(im.convert("RGB").resize((SIZE[1], SIZE[0]), Image.BICUBIC), np.float32) / 255
            xs.append(((a - MEAN) / STD).transpose(2, 0, 1))
        with torch.no_grad():
            b, ag = forward(model, torch.from_numpy(np.stack(xs)).cuda())
        out.append(np.concatenate([(b.sigmoid() > 0.5).cpu().numpy(), ag.argmax(1, keepdim=True).cpu().numpy()], 1))
    return np.concatenate(out)


def agreement(ref_paths, sr_paths):
    st = torch.load(ROOT / "weights" / "stage05" / "attr" / "model.pt", map_location="cpu")
    binary = st["binary"]
    model = build_model(len(binary), ckpt=True)
    model.load_state_dict(st["model"])
    model = model.cuda().eval()
    names = binary + ["age"]
    agree = (predict(model, binary, ref_paths) == predict(model, binary, sr_paths)).mean(0)
    per = dict(zip(names, agree.tolist()))
    return {"per_attribute": per, "color": float(np.mean([v for k, v in per.items() if is_color(k)])), "non_color": float(np.mean([v for k, v in per.items() if not is_color(k)]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sr", required=True)
    a = ap.parse_args()
    rd, sd = Path(a.ref), Path(a.sr)
    sr = {p.name.split(".")[0]: p for p in sd.iterdir() if p.suffix in (".webp", ".png")}
    ref = {p.name.split(".")[0]: p for p in rd.iterdir() if p.suffix in (".jpg", ".png", ".webp")}
    common = sorted(set(sr) & set(ref))
    print(json.dumps({"n": len(common), **agreement([str(ref[n]) for n in common], [str(sr[n]) for n in common])}, ensure_ascii=False))
