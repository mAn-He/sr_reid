"""특징 추출: eval 모드, flip TTA 없음, BNNeck 뒤 특징 L2 정규화."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from reid.model import BoT
from reid.trainer import MEAN, STD


def load_model(cfg, mode):
    st = torch.load(cfg["_run"] / "reid" / mode / "model.pt", map_location="cpu")
    model = BoT(st["num_classes"])
    model.load_state_dict(st["model"])
    return model.eval().cuda()


def to_tensor(path, cfg):
    h, w = cfg["reid"]["input_size"]
    img = Image.open(path).convert("RGB").resize((w, h), Image.BICUBIC)
    x = (np.asarray(img, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


@torch.no_grad()
def extract(model, paths, cfg):
    feats = []
    batch = cfg["reid"]["eval"]["batch"]
    for i in range(0, len(paths), batch):
        x = torch.stack([to_tensor(p, cfg) for p in paths[i:i + batch]]).cuda()
        feats.append(F.normalize(model(x), dim=1).cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 2048), np.float32)
