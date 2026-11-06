"""Part D. Stage 0 백엔드를 상속해 단계별 시간과 결과 불변 최적화를 더한다. 외부 저장소는 수정하지 않는다(런타임 대체만).
최적화 플래그 (flags): memmap, copy_read, gpu_norm, worker_norm, no_timers, pin, no_pin, prefetch4, persistent, benchmark, channels_last, mask_f32, mask_f16_sim.
memmap만 주면 정규화·erasing은 GPU에서 한다. worker_norm을 더하면 기준선처럼 워커에서 한다(입력 텐서가 기준선과 비트 단위로 같은지 D.5 보강에서 확인)."""
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stage0 import backends as S0
from stage05 import cache as C

ROOT = Path(__file__).resolve().parents[1]
ALL = ["memmap", "gpu_norm", "no_timers", "pin", "prefetch4"]


def timed(t, key, sync, fn):
    t0 = time.perf_counter()
    out = fn()
    sync()
    t[key] += time.perf_counter() - t0
    return out


class BotSet(Dataset):
    """_BotSet과 같은 순서로 난수를 쓴다. 정규화와 erasing 적용만 GPU로 넘긴다."""

    def __init__(self, cond, k, aug, labels, seed, norm=None):
        self.cond, self.k, self.aug, self.labels, self.norm = cond, k, aug, labels, norm
        self.mm, self.rng = None, np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = C.load(self.cond, self.k, "bounding_box_train")[0]
        rng, aug = self.rng, self.aug
        img = self.mm[i]
        if rng.random() < aug["flip"]:
            img = img[:, ::-1]
        pad = aug["pad_crop"]
        h, w = img.shape[:2]
        padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.uint8)
        padded[pad:pad + h, pad:pad + w] = img
        top, left = rng.integers(0, 2 * pad + 1, size=2)
        x = padded[top:top + h, left:left + w]
        rect = (0, 0, 0, 0)
        if rng.random() < aug["random_erasing"]:
            for _ in range(100):
                area = rng.uniform(0.02, 0.4) * h * w
                ratio = rng.uniform(0.3, 3.33)
                eh, ew = int(round((area * ratio) ** 0.5)), int(round((area / ratio) ** 0.5))
                if eh < h and ew < w:
                    y0, x0 = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
                    rect = (int(y0), int(x0), eh, ew)
                    break
        if self.norm is not None:
            f = (x.astype(np.float32) / 255 - self.norm[0]) / self.norm[1]
            y0, x0, eh, ew = rect
            if ew:
                f[y0:y0 + eh, x0:x0 + ew] = 0
            return torch.from_numpy(f.transpose(2, 0, 1).copy()), self.labels[i]
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), self.labels[i], torch.tensor(rect)


class FastBot(S0.BotBackend):
    def __init__(self, scale, batch, ckpt, seed, flags, cond="bicubic"):
        super().__init__(scale, batch, ckpt, seed)
        self.flags = set(flags)
        if "memmap" in flags:
            C.build(cond, scale, "bounding_box_train")
            ds = BotSet(cond, scale, self.r["augment"], [i[1] for i in self.items], seed, (self.mean, self.std) if "worker_norm" in flags else None)
            kw = {"prefetch_factor": 4} if "prefetch4" in flags else {}
            self.loader = S0.cycle(DataLoader(ds, batch_sampler=self.batches, num_workers=self.r["num_workers"], worker_init_fn=S0._bot_worker,
                                              pin_memory="no_pin" not in flags, persistent_workers="persistent" in flags, **kw))
            self.mean_t = torch.tensor(self.mean).view(1, 3, 1, 1).cuda()
            self.std_t = torch.tensor(self.std).view(1, 3, 1, 1).cuda()
        if "benchmark" in flags:
            torch.backends.cudnn.benchmark = True
        if "channels_last" in flags:
            self.model = self.model.to(memory_format=torch.channels_last)

    def _input(self):
        batch = next(self.loader)
        if "memmap" not in self.flags or "worker_norm" in self.flags:
            x, y = batch
            return x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        u8, y, rects = batch
        x = (u8.cuda(non_blocking=True).float() / 255 - self.mean_t) / self.std_t
        for i in torch.nonzero(rects[:, 2]).flatten().tolist():
            y0, x0, eh, ew = rects[i].tolist()
            x[i, :, y0:y0 + eh, x0:x0 + ew] = 0
        return x, y.cuda(non_blocking=True)

    def train_step(self):
        self.model.train()
        x, y = self._input()
        if "channels_last" in self.flags:
            x = x.to(memory_format=torch.channels_last)
        logit, g = self.model(x)
        loss = torch.nn.functional.cross_entropy(logit, y, label_smoothing=self.r["loss"]["ce_label_smoothing"]) + self.triplet(g, y, self.r["loss"]["triplet_margin"])
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def profile_step(self, t, sync):
        self.model.train()
        if "memmap" in self.flags and "worker_norm" not in self.flags:
            batch = timed(t, "load", sync, lambda: next(self.loader))
            u8, y, rects = batch

            def h2d():
                xx = (u8.cuda(non_blocking=True).float() / 255 - self.mean_t) / self.std_t
                for i in torch.nonzero(rects[:, 2]).flatten().tolist():
                    y0, x0, eh, ew = rects[i].tolist()
                    xx[i, :, y0:y0 + eh, x0:x0 + ew] = 0
                return xx, y.cuda(non_blocking=True)
            x, yy = timed(t, "h2d", sync, h2d)
        else:
            xb, yb = timed(t, "load", sync, lambda: next(self.loader))
            x, yy = timed(t, "h2d", sync, lambda: (xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)))
        logit, g = timed(t, "forward", sync, lambda: self.model(x))
        loss = timed(t, "loss", sync, lambda: torch.nn.functional.cross_entropy(logit, yy, label_smoothing=self.r["loss"]["ce_label_smoothing"]) + self.triplet(g, yy, self.r["loss"]["triplet_margin"]))
        self.opt.zero_grad()
        timed(t, "backward", sync, loss.backward)
        timed(t, "opt", sync, self.opt.step)
        return float(loss.detach())


def clip_erase(re_obj, chan, img_h, img_w):
    """timm RandomErasing._erase와 같은 난수 순서. 값을 쓰는 대신 위치와 잡음을 돌려준다(mode=pixel)."""
    if random.random() > re_obj.probability:
        return None
    area = img_h * img_w
    count = re_obj.min_count if re_obj.min_count == re_obj.max_count else random.randint(re_obj.min_count, re_obj.max_count)
    out = None
    for _ in range(count):
        for _attempt in range(10):
            target = random.uniform(re_obj.min_area, re_obj.max_area) * area / count
            aspect = math.exp(random.uniform(*re_obj.log_aspect_ratio))
            h = int(round(math.sqrt(target * aspect)))
            w = int(round(math.sqrt(target / aspect)))
            if w < img_w and h < img_h:
                top = random.randint(0, img_h - h)
                left = random.randint(0, img_w - w)
                out = (top, left, torch.empty((chan, h, w), dtype=torch.float32).normal_())
                break
    return out


class ClipSet(Dataset):
    """공식 변환(Resize no-op, RandomHorizontalFlip, Pad, RandomCrop, ToTensor, Normalize, RandomErasing)과 같은 순서로 torch·random 난수를 쓴다."""

    def __init__(self, cond, k, rows, pad, flip_p, re_obj, norm=None):
        self.cond, self.k, self.rows, self.pad, self.flip_p, self.re, self.norm = cond, k, rows, pad, flip_p, re_obj, norm
        self.mm = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = C.load(self.cond, self.k, "bounding_box_train")[0]
        img = self.mm[i]
        pad = self.pad
        h, w = img.shape[:2]
        if torch.rand(1) < self.flip_p:
            img = img[:, ::-1]
        top = torch.randint(0, 2 * pad + 1, size=(1,)).item()
        left = torch.randint(0, 2 * pad + 1, size=(1,)).item()
        padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.uint8)
        padded[pad:pad + h, pad:pad + w] = img
        x = padded[top:top + h, left:left + w]
        e = clip_erase(self.re, 3, h, w)
        t = torch.from_numpy(x.transpose(2, 0, 1).copy())
        if self.norm is not None:
            t = t.float().div(255).sub_(self.norm[0]).div_(self.norm[1])
            if e is not None:
                t[:, e[0]:e[0] + e[2].shape[1], e[1]:e[1] + e[2].shape[2]] = e[2]
            e = None
        return t, self.rows[i][1], e


def clip_collate(batch):
    return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch]), [b[2] for b in batch]


class FastClip(S0.ClipBackend):
    def __init__(self, scale, batch, ckpt, seed, flags, stage=2, cond="bicubic"):
        super().__init__(scale, batch, ckpt, seed, stage)
        self.flags = set(flags)
        if "memmap" in flags:
            from timm.data.random_erasing import RandomErasing
            C.build(cond, scale, "bounding_box_train")
            cfg = self.cfg
            re_obj = RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu")
            norm = (torch.tensor(cfg.INPUT.PIXEL_MEAN).view(3, 1, 1), torch.tensor(cfg.INPUT.PIXEL_STD).view(3, 1, 1)) if "worker_norm" in flags else None
            ds = ClipSet(cond, scale, self.items, cfg.INPUT.PADDING, cfg.INPUT.PROB, re_obj, norm)
            kw = {"prefetch_factor": 4} if "prefetch4" in flags else {}
            self.loader = S0.cycle(DataLoader(ds, batch_size=batch, sampler=self.sampler, num_workers=self.nw, collate_fn=clip_collate,
                                              pin_memory="pin" in flags, persistent_workers="persistent" in flags, **kw))
            self.mean_t = torch.tensor(cfg.INPUT.PIXEL_MEAN).view(1, 3, 1, 1).cuda()
            self.std_t = torch.tensor(cfg.INPUT.PIXEL_STD).view(1, 3, 1, 1).cuda()

    def _h2d(self, batch):
        if "memmap" not in self.flags:
            img, vid, _, _ = batch
            return img.cuda(non_blocking=True), vid.cuda(non_blocking=True)
        u8, vid, erases = batch
        if "worker_norm" in self.flags:
            return u8.cuda(non_blocking=True), vid.cuda(non_blocking=True)
        x = (u8.cuda(non_blocking=True).float() / 255 - self.mean_t) / self.std_t
        for i, e in enumerate(erases):
            if e is not None:
                top, left, noise = e
                x[i, :, top:top + noise.shape[1], left:left + noise.shape[2]] = noise.cuda(non_blocking=True)
        return x, vid.cuda(non_blocking=True)

    def _step2(self):
        self.model.train()
        batch = next(self.loader)
        self.opt.zero_grad()
        self.opt_center.zero_grad()
        img, target = self._h2d(batch)
        with torch.cuda.amp.autocast(enabled=True):
            score, feat, image_features = self.model(x=img, label=target, cam_label=None, view_label=None)
            logits = image_features @ self.text_features.t()
            loss = self.loss_fn(score, feat, target, None, logits)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item()

    def profile_step(self, t, sync):
        self.model.train()
        batch = timed(t, "load", sync, lambda: next(self.loader))
        self.opt.zero_grad()
        self.opt_center.zero_grad()
        img, target = timed(t, "h2d", sync, lambda: self._h2d(batch))

        def fwd():
            with torch.cuda.amp.autocast(enabled=True):
                return self.model(x=img, label=target, cam_label=None, view_label=None)
        score, feat, image_features = timed(t, "forward", sync, fwd)

        def lossf():
            with torch.cuda.amp.autocast(enabled=True):
                return self.loss_fn(score, feat, target, None, image_features @ self.text_features.t())
        loss = timed(t, "loss", sync, lossf)
        timed(t, "backward", sync, lambda: self.scaler.scale(loss).backward())
        timed(t, "opt", sync, lambda: (self.scaler.step(self.opt), self.scaler.update()))
        return float(loss.detach())


def install_kpr_cache(scale, mask_dtype="f32", cond="bicubic", f16_roundtrip=False):
    """디코딩·JSON·npy 읽기를 memmap 조회로 대체한다. Stage 0 경로 규칙(data/kpr/x{k})과 무관하게 파일명으로 찾는다."""
    from torchreid.data.datasets import dataset as mod
    C.build(cond, scale, "bounding_box_train")
    state = {}

    def get(kind):
        if kind not in state:
            if kind == "img":
                state[kind] = C.load(cond, scale, "bounding_box_train")
            elif kind == "mask":
                d = ROOT / "data" / "kpr_cache" / "masks" / "bounding_box_train"
                state[kind] = (np.load(d / f"masks_{mask_dtype}.npy", mmap_mode="r"), {n.split(".")[0]: i for i, n in enumerate((d / "names.txt").read_text().split("\n"))})
            else:
                z = np.load(ROOT / "data" / "kpr_cache" / f"x{scale}" / "bounding_box_train" / "keypoints.npz")
                state[kind] = (z["kp"], z["offsets"], {n: i for i, n in enumerate(z["names"].tolist())})
        return state[kind]

    def read_image(path):
        mm, idx = get("img")
        return np.array(mm[idx[Path(path).name.split(".")[0]]])

    def read_masks(path):
        arr, idx = get("mask")
        m = arr[idx[Path(path).name.split(".")[0]]]
        m = m.astype(np.float32) / 255 if mask_dtype == "u8" else np.array(m)
        if f16_roundtrip:  # float16 저장본을 파일 없이 흉내 낸다. float32 → float16 → float32.
            m = m.astype(np.float16).astype(np.float32)
        return np.transpose(m, (1, 2, 0))

    def read_keypoints(path, bbox_ltwh=None):
        kp, off, idx = get("kp")
        i = idx[Path(path).name.split(".")[0]]
        return kp[off[i]:off[i + 1]].copy()

    mod.read_image, mod.read_masks, mod.read_keypoints = read_image, read_masks, read_keypoints


class FastKpr(S0.KprBackend):
    def __init__(self, scale, batch, ckpt, seed, flags, solider=False):
        self.flags = set(flags)
        sys.path.insert(0, str(S0.KPR_REPO))
        if "copy_read" in flags:  # 파일 경로는 그대로 두고 lru_cache된 배열만 복사해 돌려준다. BIPO의 제자리 쓰기가 캐시에 남는지 가르는 진단용.
            from torchreid.data.datasets import dataset as mod
            orig = mod.read_image
            mod.read_image = lambda path: orig(path).copy()
        if "memmap" in flags:
            install_kpr_cache(scale, "u8" if "mask_u8" in flags else "f32", f16_roundtrip="mask_f16_sim" in flags)
        super().__init__(scale, batch, ckpt, seed, solider=solider, tag=f"f{os.getpid()}")
        self.solider = solider
        tl = self.engine.train_loader
        if {"prefetch4", "persistent", "no_pin"} & self.flags:
            kw = {"prefetch_factor": 4} if "prefetch4" in flags else {}
            tl = DataLoader(tl.dataset, sampler=tl.sampler, batch_size=tl.batch_size, shuffle=False, num_workers=tl.num_workers, pin_memory="no_pin" not in flags, drop_last=True,
                            persistent_workers="persistent" in flags, **kw)
            self.loader = S0.cycle(tl)
        if "no_timers" in flags:
            class Null:
                def start(self):
                    pass

                def stop(self):
                    pass
            for name in ("feature_extraction_timer", "loss_timer", "optimizer_timer"):
                setattr(self.engine, name, Null())

    def profile_step(self, t, sync):
        from torch.cuda import amp
        from torchreid.utils.constants import PIXELS
        e = self.engine
        e.set_model_mode("train")
        data = timed(t, "load", sync, lambda: next(self.loader))
        imgs, tm, pm, kp, pids, _, cam = timed(t, "h2d", sync, lambda: e.parse_data_for_train(data))

        def fwd():
            with amp.autocast(enabled=e.mixed_precision):
                return self.model(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
        emb, vis, idc, pix, _, _ = timed(t, "forward", sync, fwd)

        def lossf():
            with amp.autocast(enabled=e.mixed_precision):
                return e.combine_losses(vis, emb, idc, pids, pix, tm, bpa_weight=e.losses_weights[PIXELS]["ce"])[0]
        loss = timed(t, "loss", sync, lossf)
        e.optimizer.zero_grad()

        def back():
            if e.scaler is None:
                loss.backward()
            else:
                e.scaler.scale(loss).backward()
        timed(t, "backward", sync, back)

        def opt():
            if e.scaler is None:
                e.optimizer.step()
            else:
                e.scaler.step(e.optimizer)
                e.scaler.update()
        timed(t, "opt", sync, opt)
        return float(loss.detach())

    def profile_samples(self, n):
        """표본 1장의 단계별 CPU 시간. 메인 프로세스에서 공식 __getitem__을 그대로 돌린다."""
        from torchreid.data.datasets import dataset as mod
        ds = self.engine.train_loader.dataset
        t = {"read_image": 0.0, "read_masks": 0.0, "read_keypoints": 0.0, "transforms": 0.0, "prompt_masks": 0.0, "target_preprocess": 0.0}

        def wrap(fn, key):
            def inner(*a, **k):
                t0 = time.perf_counter()
                out = fn(*a, **k)
                t[key] += time.perf_counter() - t0
                return out
            return inner
        for name in ("read_image", "read_masks", "read_keypoints"):
            f = getattr(mod, name)
            setattr(mod, name, wrap(getattr(f, "__wrapped__", f), name))
        tr = ds.transforms("train")
        tr_w = wrap(tr, "transforms")
        ds.transforms = lambda mode: tr_w
        class Timed:
            def __init__(self, obj, key):
                self.obj, self.call = obj, wrap(obj, key)

            def __call__(self, *a, **k):
                return self.call(*a, **k)

            def __getattr__(self, name):
                return getattr(self.obj, name)
        ds.keypoints_to_prompt_masks = Timed(ds.keypoints_to_prompt_masks, "prompt_masks")
        ds.mode = "train"
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        pids = [ds.train[int(j)]["pid"] for j in rng.integers(0, len(ds.train), 8)]  # BIPO batch_sampling이 읽는 배치 pid
        for i in rng.integers(0, len(ds.train), n):
            ds[(int(i), pids)] if ds.random_occlusions else ds[int(i)]
        total = time.perf_counter() - t0
        return {"n": n, "ms_per_sample": 1000 * total / n, **{f"{k}_ms": 1000 * v / n for k, v in t.items()}}


def build(backend, scale, batch, impl="baseline", ckpt=False, seed=42, workers=None, flags=()):
    if workers:
        os.environ["SRREID_WORKERS"] = str(workers)
    flags = list(flags) or (ALL if impl == "fast" else [])
    if impl == "fast" or flags:
        if backend == "bot":
            return FastBot(scale, batch, ckpt, seed, flags)
        if backend == "clipreid":
            return FastClip(scale, batch, ckpt, seed, flags)
        return FastKpr(scale, batch, ckpt, seed, flags, solider=backend == "kpr_solider")
    if backend == "bot":
        return FastBot(scale, batch, ckpt, seed, [])
    if backend == "clipreid":
        return FastClip(scale, batch, ckpt, seed, [])
    return FastKpr(scale, batch, ckpt, seed, [], solider=backend == "kpr_solider")
