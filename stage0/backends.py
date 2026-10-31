"""모델별 측정 대상. 외부 저장소는 수정하지 않고 import한다. 데이터는 ×k train만 쓴다(공식 로더는 query, gallery를 읽으므로 대체)."""
import copy
import glob
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights" / "stage0"
CLIP_REPO = ROOT / "third_party" / "CLIP-ReID"
KPR_REPO = ROOT / "third_party" / "keypoint_promptable_reidentification"
PATTERN = re.compile(r"([-\d]+)_c(\d)")
EVAL_POOL = 5120


def size_of(scale):
    return 128 * scale, 64 * scale


def train_items(scale):
    """(경로, 재번호 pid, camid). 전체 12,936장, 751 ID."""
    paths = sorted(glob.glob(str(ROOT / "data" / f"bicubic_x{scale}" / "*.png")))
    pids = sorted({int(PATTERN.search(os.path.basename(p)).group(1)) for p in paths})
    remap = {p: i for i, p in enumerate(pids)}
    out = []
    for p in paths:
        pid, cam = map(int, PATTERN.search(os.path.basename(p)).groups())
        out.append((p, remap[pid], cam - 1))
    return out


def cycle(loader):
    while True:
        for x in loader:
            yield x


class BotBackend:
    """직접 구현 BoT. configs/pilot.yaml의 reid 레시피. AMP 없음, 입력은 ×k를 그대로 쓴다."""
    amp = False

    def __init__(self, scale, batch, ckpt=False, seed=0):
        import yaml
        from reid.model import BoT
        from reid.trainer import MEAN, STD, pk_batches, triplet_hard
        self.r = yaml.safe_load(open(ROOT / "configs" / "pilot.yaml"))["reid"]
        self.items = train_items(scale)
        self.scale, self.batch = scale, batch
        self.triplet = triplet_hard
        self.mean, self.std = MEAN, STD
        self.k = self.r["sampler"]["k"]
        self.num_classes = len({i[1] for i in self.items})
        self.model = BoT(self.num_classes, str(ROOT / self.r["pretrained"])).cuda()
        self.opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.r["optim"]["lr"],
                                    weight_decay=self.r["optim"]["weight_decay"])
        labels = np.array([i[1] for i in self.items])
        rng = np.random.default_rng(seed)
        self.batches = pk_batches(labels, batch // self.k, self.k, rng)
        if ckpt:
            from torch.utils.checkpoint import checkpoint
            for name in ("4", "5", "6", "7"):  # layer1~4. BN 통계 갱신이 두 번 되는 점은 등가 검사에 기록된다.
                block = self.model.backbone[int(name)]
                orig = block.forward
                block.forward = lambda x, o=orig: checkpoint(o, x, use_reentrant=False, preserve_rng_state=True)
        ds = _BotSet(self.items, self.r["augment"], self.mean, self.std, seed)
        self.loader = cycle(DataLoader(ds, batch_sampler=self.batches, num_workers=self.r["num_workers"],
                                       worker_init_fn=_bot_worker, pin_memory=True))
        self.eval_ds = _BotEval(self.items[:EVAL_POOL], self.mean, self.std)

    def meta(self):
        o = self.r["optim"]
        return {"recipe_batch": self.r["sampler"]["p"] * self.r["sampler"]["k"], "k": self.k, "amp": self.amp,
                "optimizer": f"adam lr={o['lr']} wd={o['weight_decay']}", "epochs": self.r["epochs"],
                "iters_per_epoch": len(self.batches), "num_workers": self.r["num_workers"], "torch": torch.__version__}

    def train_step(self):
        self.model.train()
        x, y = next(self.loader)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logit, g = self.model(x)
        loss = torch.nn.functional.cross_entropy(logit, y, label_smoothing=self.r["loss"]["ce_label_smoothing"]) \
            + self.triplet(g, y, self.r["loss"]["triplet_margin"])
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def extract_loader(self, batch):
        return DataLoader(self.eval_ds, batch_size=batch, num_workers=self.r["num_workers"])

    @torch.no_grad()
    def extract_batch(self, x):
        self.model.eval()
        return self.model(x.cuda())

    def feature_shape(self):
        return {"dim": 2048}


class _BotSet(Dataset):
    def __init__(self, items, aug, mean, std, seed):
        self.items, self.aug, self.mean, self.std, self.rng = items, aug, mean, std, np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = self.rng
        img = np.asarray(Image.open(self.items[i][0]).convert("RGB"))
        if rng.random() < self.aug["flip"]:
            img = img[:, ::-1]
        pad = self.aug["pad_crop"]
        h, w = img.shape[:2]
        padded = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.uint8)
        padded[pad:pad + h, pad:pad + w] = img
        top, left = rng.integers(0, 2 * pad + 1, size=2)
        x = (padded[top:top + h, left:left + w].astype(np.float32) / 255 - self.mean) / self.std
        if rng.random() < self.aug["random_erasing"]:
            for _ in range(100):
                area = rng.uniform(0.02, 0.4) * h * w
                ratio = rng.uniform(0.3, 3.33)
                eh, ew = int(round((area * ratio) ** 0.5)), int(round((area / ratio) ** 0.5))
                if eh < h and ew < w:
                    y0, x0 = rng.integers(0, h - eh + 1), rng.integers(0, w - ew + 1)
                    x[y0:y0 + eh, x0:x0 + ew] = 0
                    break
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), self.items[i][1]


def _bot_worker(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


class _BotEval(Dataset):
    def __init__(self, items, mean, std):
        self.items, self.mean, self.std = items, mean, std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = (np.asarray(Image.open(self.items[i][0]).convert("RGB")).astype(np.float32) / 255 - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


class ClipBackend:
    """CLIP-ReID vit_clipreid.yml. train_clipreid.py와 processor_clipreid_stage{1,2}.py의 한 스텝을 그대로 옮겼다."""
    amp = True

    def __init__(self, scale, batch, ckpt=False, seed=0, stage=2):
        sys.path.insert(0, str(CLIP_REPO))
        from config import cfg
        from datasets.bases import ImageDataset
        from datasets.make_dataloader_clipreid import train_collate_fn
        from datasets.sampler import RandomIdentitySampler
        from loss.make_loss import make_loss
        from loss.supcontrast import SupConLoss
        from model.clip import clip
        from model.make_model_clipreid import make_model
        from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
        from timm.data.random_erasing import RandomErasing
        import torchvision.transforms as T

        clip._download = lambda url, root=None: str(W / "ViT-B-16.pt")  # 저장소 수정 없이 기록된 가중치를 쓴다.
        h, w = size_of(scale)
        cfg.merge_from_file(str(CLIP_REPO / "configs" / "person" / "vit_clipreid.yml"))
        cfg.merge_from_list(["INPUT.SIZE_TRAIN", [h, w], "INPUT.SIZE_TEST", [h, w], "DATASETS.NAMES", "market1501",
                             "SOLVER.STAGE2.IMS_PER_BATCH", batch, "SOLVER.STAGE1.IMS_PER_BATCH", batch])
        cfg.freeze()
        self.cfg, self.scale, self.batch, self.stage = cfg, scale, batch, stage
        items = [(p, pid, cam, 1) for p, pid, cam in train_items(scale)]
        self.items = items
        self.num_classes = len({i[1] for i in items})
        self.k = cfg.DATALOADER.NUM_INSTANCE
        train_tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3), T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                              T.Pad(cfg.INPUT.PADDING), T.RandomCrop(cfg.INPUT.SIZE_TRAIN), T.ToTensor(),
                              T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                              RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu")])
        self.val_tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(),
                                 T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
        self.ImageDataset = ImageDataset
        self.collate = train_collate_fn
        self.sampler = RandomIdentitySampler(items, batch, self.k)
        self.nw = cfg.DATALOADER.NUM_WORKERS
        self.model = make_model(cfg, num_class=self.num_classes, camera_num=6, view_num=1).cuda()
        self.scaler = torch.cuda.amp.GradScaler()
        self.xent = SupConLoss("cuda")
        if stage == 1:
            self.opt = make_optimizer_1stage(cfg, self.model)
            self.labels, self.feats = self._stage1_cache()
            self.perm = torch.randperm(len(self.items)).cuda()
            self.i = 0
            return
        self.loss_fn, center = make_loss(cfg, num_classes=self.num_classes)
        self.opt, self.opt_center = make_optimizer_2stage(cfg, self.model, center)
        if ckpt:
            from torch.utils.checkpoint import checkpoint
            for block in self.model.image_encoder.transformer.resblocks:
                orig = block.forward
                block.forward = lambda x, o=orig: checkpoint(o, x, use_reentrant=False, preserve_rng_state=True)
        self.text_features = self._text_features()
        ds = ImageDataset(items, train_tf)
        self.loader = cycle(DataLoader(ds, batch_size=batch, sampler=self.sampler, num_workers=self.nw,
                                       collate_fn=train_collate_fn))

    def _text_features(self):
        out = []
        with torch.no_grad():
            for i in range(0, self.num_classes, self.batch):
                ids = torch.arange(i, min(i + self.batch, self.num_classes))
                with torch.cuda.amp.autocast(enabled=True):
                    out.append(self.model(label=ids, get_text=True).cpu())
        return torch.cat(out, 0).cuda()

    def _stage1_cache(self):
        """공식 코드는 train 전체 특징을 GPU에 올린다. 측정용으로 3,200장만 추출해 전체 크기로 반복한다. 추출 시간은 extract로 따로 잰다."""
        ds = self.ImageDataset(self.items[:3200], self.val_tf)
        loader = DataLoader(ds, batch_size=self.batch, num_workers=self.nw, collate_fn=self.collate, shuffle=True)
        labels, feats = [], []
        with torch.no_grad():
            for img, vid, _, _ in loader:
                with torch.cuda.amp.autocast(enabled=True):
                    f = self.model(img.cuda(), vid.cuda(), get_image=True)
                labels.append(vid)
                feats.append(f.cpu())
        n = len(self.items)
        labels = torch.cat(labels).repeat(n // 3200 + 1)[:n].cuda()
        return labels, torch.cat(feats).repeat(n // 3200 + 1, 1)[:n].cuda()

    def meta(self):
        c = self.cfg.SOLVER
        s = c.STAGE1 if self.stage == 1 else c.STAGE2
        iters = len(self.items) // c.STAGE1.IMS_PER_BATCH + 1 if self.stage == 1 else len(self.sampler) // self.batch
        return {"recipe_batch": 64, "k": self.k, "amp": True, "optimizer": f"{s.OPTIMIZER_NAME} lr={s.BASE_LR}",
                "epochs": s.MAX_EPOCHS, "iters_per_epoch": iters, "num_workers": self.nw, "torch": torch.__version__}

    def train_step(self):
        return self._step1() if self.stage == 1 else self._step2()

    def _step1(self):
        self.model.train()
        n = len(self.perm)
        idx = self.perm[(self.i * self.batch) % n:(self.i * self.batch) % n + self.batch]
        self.i += 1
        self.opt.zero_grad()
        target, feats = self.labels[idx], self.feats[idx]
        with torch.cuda.amp.autocast(enabled=True):
            text = self.model(label=target, get_text=True)
        loss = self.xent(feats, text, target, target) + self.xent(text, feats, target, target)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item()

    def _step2(self):
        self.model.train()
        img, vid, _, _ = next(self.loader)
        self.opt.zero_grad()
        self.opt_center.zero_grad()
        img, target = img.cuda(), vid.cuda()
        with torch.cuda.amp.autocast(enabled=True):
            score, feat, image_features = self.model(x=img, label=target, cam_label=None, view_label=None)
            logits = image_features @ self.text_features.t()
            loss = self.loss_fn(score, feat, target, None, logits)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item()

    def extract_loader(self, batch):
        return DataLoader(self.ImageDataset(self.items[:EVAL_POOL], self.val_tf), batch_size=batch, num_workers=self.nw)

    @torch.no_grad()
    def extract_batch(self, data):
        self.model.eval()
        return self.model(data[0].cuda(), cam_label=None, view_label=None)

    def feature_shape(self):
        return {"dim": self.model.in_planes + self.model.in_planes_proj}


class KprBackend:
    """KPR, KPR+SOLIDER 공식 market train yaml. forward_backward는 공식 engine을 그대로 호출한다."""

    def __init__(self, scale, batch, ckpt=False, seed=0, solider=False, eval_batch=128, eval_pool=8, tag="m"):
        sys.path.insert(0, str(KPR_REPO))
        import torchreid
        from torchreid.data.datasets.image.market1501 import Market1501
        from torchreid.data.datasets.dataset import ImageDataset
        from torchreid.scripts.builder import build_config, build_torchreid_model_engine

        name = f"srreid_market_x{scale}"
        root = ROOT / "data" / "kpr" / f"x{scale}"

        class TrainOnly(Market1501):
            dataset_dir = str(root)

            def __init__(self, root="", masks_dir=None, **kw):
                self.kp_dir = kw["config"].model.kpr.keypoints.kp_dir
                self.masks_dir = masks_dir
                self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[masks_dir]
                self.dataset_dir = TrainOnly.dataset_dir
                train = self.process_dir(os.path.join(self.dataset_dir, "bounding_box_train"), relabel=True)
                ImageDataset.__init__(self, train, copy.deepcopy(train[:eval_pool]), copy.deepcopy(train[:eval_pool]), **kw)

            def process_dir(self, dir_path, relabel=False):
                paths = sorted(glob.glob(os.path.join(dir_path, "*.png")))
                pids = sorted({int(PATTERN.search(os.path.basename(p)).group(1)) for p in paths})
                remap = {p: i for i, p in enumerate(pids)}
                out = []
                for p in paths:
                    pid, cam = map(int, PATTERN.search(os.path.basename(p)).groups())
                    out.append({"img_path": p, "pid": remap[pid], "camid": cam - 1, "kp_path": self.infer_kp_path(p),
                                "masks_path": self.infer_masks_path(p, self.masks_dir, self.masks_suffix)})
                return out

        torchreid.data.register_image_dataset(name, TrainOnly)
        h, w = size_of(scale)
        pre = "solider" if solider else "imagenet"
        yaml_path = KPR_REPO / "configs" / "kpr" / pre / "kpr_market_train.yaml"
        save = ROOT / "runs" / "stage0" / "kpr_tmp"
        args = SimpleNamespace(root=str(ROOT / "data" / "kpr"), save_dir=str(save), inference_enabled=False, sources=[name],
                               targets=[name], transforms=None, job_id=f"{tag}_{os.getpid()}",
                               opts=["data.height", h, "data.width", w, "train.batch_size", batch, "test.visrank", False,
                                     "test.batch_size", eval_batch, "model.backbone_pretrained_path", str(W) + "/",
                                     "train.seed", seed])
        self.cfg = cfg = build_config(args=args, config_path=str(yaml_path))
        self.scale, self.batch = scale, batch
        self.engine, self.model = build_torchreid_model_engine(cfg)
        self.ckpt_blocks = 0
        if ckpt:
            self.set_ckpt(True)
        self.loader = cycle(self.engine.train_loader)
        self.iters = len(self.engine.train_loader)
        self.name = name

    def set_ckpt(self, on):
        n = 0
        for m in self.model.modules():
            if hasattr(m, "with_cp"):
                m.with_cp = on
                n += 1
        self.ckpt_blocks = n
        if on and n == 0:
            raise RuntimeError("with_cp 모듈을 찾지 못함")

    def meta(self):
        t = self.cfg.train
        return {"recipe_batch": 64, "k": self.cfg.sampler.num_instances, "amp": bool(t.mixed_precision),
                "optimizer": f"{t.optim} lr={t.lr}", "epochs": t.max_epoch, "iters_per_epoch": self.iters,
                "num_workers": self.cfg.data.workers, "torch": torch.__version__, "fixbase_epoch": t.fixbase_epoch,
                "ckpt_blocks": self.ckpt_blocks, "visrank": bool(self.cfg.test.visrank),
                "official_input": "384x128"}

    def train_step(self):
        self.engine.set_model_mode("train")
        loss, _ = self.engine.forward_backward(next(self.loader))
        return loss.item()

    def extract_loader(self, batch):
        return self.engine.test_loader[self.name]["query"]

    @torch.no_grad()
    def extract_batch(self, data):
        from torchreid.utils.tools import extract_test_embeddings
        self.engine.set_model_mode("eval")
        imgs, target_masks, prompt_masks, keypoints_xyc, pids, camids = self.engine.parse_data_for_eval(data)
        if self.engine.use_gpu:
            imgs = imgs.cuda()
            target_masks = target_masks.cuda() if target_masks is not None else None
            prompt_masks = prompt_masks.cuda() if prompt_masks is not None else None
        out = self.model(imgs, target_masks=target_masks, prompt_masks=prompt_masks, keypoints_xyc=keypoints_xyc,
                         cam_label=camids)
        features, vis, _, _ = extract_test_embeddings(out, self.cfg.model.kpr.test_embeddings)
        self.last_shape = {"parts": features.shape[1], "dim": features.shape[2], "vis_dtype": str(vis.dtype),
                           "test_embeddings": list(self.cfg.model.kpr.test_embeddings)}
        return features

    def feature_shape(self):
        return getattr(self, "last_shape", None)

    def equivalence(self, batch_size):
        """같은 seed, 같은 입력의 1스텝 손실과 gradient norm: off, off(잡음 기준), on."""
        from torch.cuda import amp
        from torchreid.utils.constants import PIXELS
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        state = copy.deepcopy(self.model.state_dict())
        data = next(self.loader)
        res = {}
        for tag, on in (("off", False), ("off2", False), ("on", True)):
            self.model.load_state_dict(state)
            self.set_ckpt(on)
            self.engine.set_model_mode("train")
            self.model.zero_grad(set_to_none=True)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            np.random.seed(0)
            imgs, tm, pm, kp, pids, _, cam = self.engine.parse_data_for_train(data)
            with amp.autocast(enabled=self.cfg.train.mixed_precision):
                emb, vis, idc, pix, _, _ = self.model(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
                loss, _ = self.engine.combine_losses(vis, emb, idc, pids, pix, tm,
                                                     bpa_weight=self.engine.losses_weights[PIXELS]["ce"])
            loss.backward()
            grads = [p.grad.detach().float().norm() for p in self.model.parameters() if p.grad is not None]
            res[tag] = {"loss": float(loss.item()), "grad_norm": float(torch.stack(grads).norm().item())}
        return {"batch": batch_size, **res, "loss_diff_on_off": abs(res["on"]["loss"] - res["off"]["loss"]),
                "grad_norm_diff_on_off": abs(res["on"]["grad_norm"] - res["off"]["grad_norm"]),
                "loss_diff_noise": abs(res["off2"]["loss"] - res["off"]["loss"]),
                "grad_norm_diff_noise": abs(res["off2"]["grad_norm"] - res["off"]["grad_norm"])}


def build(backend, scale, batch, ckpt=False, seed=0, stage=2, **kw):
    if backend == "bot":
        return BotBackend(scale, batch, ckpt, seed)
    if backend == "clipreid":
        return ClipBackend(scale, batch, ckpt, seed, stage)
    return KprBackend(scale, batch, ckpt, seed, solider=backend == "kpr_solider", **kw)
