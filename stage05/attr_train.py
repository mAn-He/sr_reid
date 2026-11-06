"""G.4 속성 분류기. Market-1501 속성 주석(vana77/Market-1501_Attribute)으로 ResNet-50을 train 651명 원본에서 학습한다(val 100명 제외).
주석 파일의 test 구조체는 쓰지 않는다. srreid-quality env. seed 42. 결과: weights/stage05/attr/model.pt, results/stage05_attr_val.json."""
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

ATTR_MAT = ROOT / "data" / "attributes" / "market_attribute.mat"
SIZE = (256, 128)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
COLOR_PREFIX = ("up", "down")
BINARY_NON_COLOR = {"backpack", "bag", "handbag", "clothes", "down", "up", "hair", "hat", "gender"}


def load_labels():
    """train 구조체만 읽는다. 값 1,2 → 0,1. age는 1~4 → 0~3. 키는 네 자리 pid."""
    m = sio.loadmat(ATTR_MAT, squeeze_me=True, struct_as_record=False)["market_attribute"].train
    names = [n for n in m._fieldnames if n != "image_index"]
    ids = [f"{int(i):04d}" for i in np.atleast_1d(m.image_index)]
    labels = {}
    for j, pid in enumerate(ids):
        labels[pid] = {n: int(np.atleast_1d(getattr(m, n))[j]) - 1 for n in names}
    return names, labels


def is_color(name):
    return name.startswith(("up", "down")) and name not in ("up", "down")


def split_ids():
    rows = list(csv.DictReader(open(ROOT / "data" / "processed" / "manifests" / "split.csv")))
    return rows


class AttrSet(Dataset):
    def __init__(self, paths, targets, train):
        self.paths, self.targets, self.train = paths, targets, train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((SIZE[1], SIZE[0]), Image.BICUBIC)
        a = np.asarray(im, np.float32) / 255
        if self.train and np.random.random() < 0.5:
            a = a[:, ::-1]
        return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy()), torch.tensor(self.targets[i])


def build_model(n_binary, ckpt=None):
    net = torchvision.models.resnet50(weights=None)
    if ckpt is None:
        net.load_state_dict(torch.load(ROOT / "weights" / "resnet50-0676ba61.pth", map_location="cpu"))
    net.fc = nn.Identity()
    model = nn.ModuleDict({"backbone": net, "binary": nn.Linear(2048, n_binary), "age": nn.Linear(2048, 4)})
    return model


def forward(model, x):
    f = model["backbone"](x)
    return model["binary"](f), model["age"](f)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    names, labels = load_labels()
    binary = [n for n in names if n != "age"]
    rows = split_ids()
    train_rows = [r for r in rows if r["role"] == "train"]
    val_rows = [r for r in rows if r["role"] != "train"]
    pat = re.compile(r"^(\d{4})_")
    root = ROOT / "data" / "market" / "bounding_box_train"
    all_files = sorted(root.glob("*.jpg"))
    train_ids = {r["pid"].zfill(4) for r in train_rows}
    val_ids = {r["pid"].zfill(4) for r in val_rows}
    if train_ids & val_ids:
        raise SystemExit("train, val ID 겹침")

    def collect(ids):
        p, t = [], []
        for f in all_files:
            pid = pat.match(f.name).group(1)
            if pid in ids:
                p.append(str(f))
                t.append([labels[pid][n] for n in binary] + [labels[pid]["age"]])
        return p, t
    ptr, ttr = collect(train_ids)
    pva, tva = collect(val_ids)
    print(f"train {len(train_ids)}명 {len(ptr)}장, val {len(val_ids)}명 {len(pva)}장", flush=True)
    assert len(train_ids) == 651 and len(val_ids) == 100
    model = build_model(len(binary)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    epochs = 30
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [20, 25], 0.1)
    loader = DataLoader(AttrSet(ptr, ttr, True), batch_size=64, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            b, a = forward(model, x)
            loss = bce(b, y[:, :-1].float()) + ce(a, y[:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        print(f"epoch {ep} loss {total / len(loader):.4f}", flush=True)
    out = ROOT / "weights" / "stage05" / "attr"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "binary": binary, "epochs": epochs, "seed": 42}, out / "model.pt")
    model.eval()
    correct = np.zeros(len(binary) + 1)
    n = 0
    with torch.no_grad():
        for x, y in DataLoader(AttrSet(pva, tva, False), batch_size=128, num_workers=8):
            b, a = forward(model, x.cuda())
            pb = (b.sigmoid() > 0.5).cpu().numpy()
            correct[:-1] += (pb == y[:, :-1].numpy().astype(bool)).sum(0)
            correct[-1] += (a.argmax(1).cpu().numpy() == y[:, -1].numpy()).sum()
            n += len(x)
    acc = dict(zip(binary + ["age"], (correct / n).tolist()))
    color = [v for k, v in acc.items() if is_color(k)]
    other = [v for k, v in acc.items() if not is_color(k)]
    res = {"n_val_images": n, "acc": acc, "mean_color": float(np.mean(color)), "mean_non_color": float(np.mean(other)), "mean_all": float(np.mean(list(acc.values())))}
    (ROOT / "results" / "stage05_attr_val.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
