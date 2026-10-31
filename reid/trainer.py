"""train_hr.py, train_lr.py가 공유하는 학습 루프. 에폭 경계에서 재개해도 같은 난수 흐름을 쓴다."""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.util import EXIT_FATAL, EXIT_OOM, load_cfg, read_csv, sha256_file, wait_free_gb
from reid.model import BoT

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class TrainSet(Dataset):
    """mode: hr(HR만), lr_only(LR 경로만), mixed(샘플마다 p_lr 확률로 LR 경로)."""

    def __init__(self, cfg, mode):
        self.hr = np.load(cfg["_processed"] / "train_hr256.npy") if mode != "lr_only" else None
        self.lr = np.load(cfg["_processed"] / "train_lr256.npy") if mode != "hr" else None
        self.mode = mode
        self.p_lr = cfg["reid"]["path1"]["p_lr"]
        rows = read_csv(cfg["_manifests"] / "train_cache.csv")
        ids = sorted({r["pid"] for r in rows})
        remap = {p: i for i, p in enumerate(ids)}
        self.labels = np.array([remap[r["pid"]] for r in rows])
        self.num_classes = len(ids)
        self.aug = cfg["reid"]["augment"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        rng = self.rng
        if self.mode == "hr" or (self.mode == "mixed" and rng.random() >= self.p_lr):
            img = self.hr[i]
        else:
            img = self.lr[i]
        if rng.random() < self.aug["flip"]:
            img = img[:, ::-1]
        pad = self.aug["pad_crop"]
        h, w = img.shape[:2]
        padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.uint8)
        padded[pad:pad + h, pad:pad + w] = img
        top, left = rng.integers(0, 2 * pad + 1, size=2)
        x = (padded[top:top + h, left:left + w].astype(np.float32) / 255 - MEAN) / STD
        if rng.random() < self.aug["random_erasing"]:
            for _ in range(100):
                area = rng.uniform(0.02, 0.4) * h * w
                ratio = rng.uniform(0.3, 3.33)
                eh, ew = int(round((area * ratio) ** 0.5)), int(round((area / ratio) ** 0.5))
                if eh < h and ew < w:
                    y0, x0 = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
                    x[y0:y0 + eh, x0:x0 + ew] = 0  # 정규화 후 0 = ImageNet 평균
                    break
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), int(self.labels[i])


def worker_init(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def pk_batches(labels, p, k, rng):
    """신원별로 k장씩 묶고 신원 p개를 뽑아 배치를 만든다. k장 미만 신원은 복원추출."""
    by_id = defaultdict(list)
    for i, l in enumerate(labels):
        by_id[int(l)].append(i)
    chunks = {}
    for pid, idx in by_id.items():
        idx = rng.choice(idx, k, replace=True).tolist() if len(idx) < k else rng.permutation(idx).tolist()
        chunks[pid] = [idx[j:j + k] for j in range(0, len(idx) - k + 1, k)]
    batches = []
    alive = sorted(chunks)
    while len(alive) >= p:
        chosen = rng.choice(alive, p, replace=False).tolist()
        batch = []
        for pid in chosen:
            batch += chunks[pid].pop()
            if not chunks[pid]:
                alive.remove(pid)
        batches.append(batch)
    return batches


def triplet_hard(g, y, margin):
    d = (g.unsqueeze(1) - g.unsqueeze(0)).pow(2).sum(-1).clamp(min=1e-12).sqrt()
    same = y[:, None] == y[None, :]
    hardest_pos = (d * same).max(dim=1).values
    hardest_neg = d.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def lr_at(o, epoch):
    if epoch < o["warmup_epochs"]:
        t = epoch / o["warmup_epochs"]
        factor = o["warmup_factor"] * (1 - t) + t
    else:
        factor = o["gamma"] ** sum(epoch >= m for m in o["milestones"])
    return o["lr"] * factor


def train(config_path, mode):
    cfg = load_cfg(config_path)
    out = cfg["_run"] / "reid" / mode
    out.mkdir(parents=True, exist_ok=True)
    r, seed = cfg["reid"], cfg["seed"]
    if not wait_free_gb(cfg["gpu"]["min_free_gb_train"], cfg["gpu"]["wait_timeout_s"]):
        print("min_free_gb_train 미충족", flush=True)
        sys.exit(EXIT_FATAL)
    torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["fraction_train"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    seed_all(seed)
    ds = TrainSet(cfg, mode)
    ds.rng = np.random.default_rng(seed)
    model = BoT(ds.num_classes, cfg["reid"]["pretrained"]).cuda()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=r["optim"]["lr"],
                           weight_decay=r["optim"]["weight_decay"])
    start = 0
    last = out / "last.pt"
    if last.exists():
        st = torch.load(last, map_location="cuda")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = st["epoch"] + 1
        print(f"재개: epoch {start}", flush=True)

    p, k = r["sampler"]["p"], r["sampler"]["k"]
    log = open(out / "train_log.jsonl", "a")
    try:
        for epoch in range(start, r["epochs"]):
            seed_all(seed * 100003 + epoch)
            batches = pk_batches(ds.labels, p, k, np.random.default_rng(seed * 100003 + epoch))
            loader = DataLoader(ds, batch_sampler=batches, num_workers=r["num_workers"], worker_init_fn=worker_init,
                                pin_memory=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(r["optim"], epoch)
            model.train()
            t0, total = time.time(), 0.0
            for x, y in loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logit, g = model(x)
                loss = F.cross_entropy(logit, y, label_smoothing=r["loss"]["ce_label_smoothing"]) \
                    + triplet_hard(g, y, r["loss"]["triplet_margin"])
                if not torch.isfinite(loss):
                    print(f"loss가 유한하지 않음: epoch {epoch}", flush=True)
                    sys.exit(EXIT_FATAL)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            log.write(json.dumps({"epoch": epoch, "loss": total / max(len(batches), 1), "lr": opt.param_groups[0]["lr"],
                                  "seconds": time.time() - t0}) + "\n")
            log.flush()
            if (epoch + 1) % r["ckpt_every"] == 0 or epoch + 1 == r["epochs"]:
                tmp = str(last) + ".tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, tmp)
                os.replace(tmp, last)
    except torch.cuda.OutOfMemoryError as e:
        print("OOM:", e, flush=True)
        sys.exit(EXIT_OOM)

    final = out / "model.pt"
    torch.save({"model": model.state_dict(), "num_classes": ds.num_classes, "mode": mode}, final)
    (out / "model.pt.sha256").write_text(sha256_file(final) + "\n")
    meta = {"mode": mode, "seed": seed, "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(), "sha256": sha256_file(final),
            "pretrained_sha256": sha256_file(cfg["reid"]["pretrained"])}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    print("완료", meta, flush=True)
