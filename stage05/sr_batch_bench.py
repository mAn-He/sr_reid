"""Stage 1 준비. 학습형 SR 10종의 배치 추론 가능 여부와 배치 크기별 처리량·VRAM·픽셀 차이를 잰다.

각 방법의 env에서 실행한다(방법은 CLI 인자로 고른다). third_party 소스는 고치지 않고 공식 모듈·클래스를 그대로 불러,
공식 스크립트의 전처리·패딩·dtype·후처리·반올림·시드 처리를 같은 프로세스 안에서 그대로 재현하되 N장을 한 텐서로 묶어 넣는다.
공식과 다른 점은 배치 차원뿐이다. 공식 코드가 배치 1을 전제로 짠 곳(아래 '배치 어댑터')만 같은 연산을 배치로 다시 적었다.

기준점: 배치 1 출력이 runs/stage05/sr_smoke/<method>/x<k>/bounding_box_train/*.webp(공식 출력, 무손실)와 픽셀 단위로 같은지 본다.
측정: 배치 크기마다 100장 전체를 한 번 돌려 예열(모든 배치 모양의 cudnn 선택 포함)한 뒤, 같은 100장을 다시 돌려 시간·VRAM·픽셀 차이를 잰다.
마지막 배치는 나머지 크기 그대로 넣는다(패딩 없음).
확산 방법(osdhuman, pisa_sr)은 노이즈 두 방식을 모두 잰다.
  noise=global: 공식처럼 전역 CUDA RNG에서 배치 모양 그대로 한 번에 뽑는다(배치 구성에 따라 이미지별 노이즈가 달라진다).
  noise=replay: 배치 1에서 이미지마다 뽑던 순서·모양·dtype 그대로 미리 뽑아 넣는다(배치 1과 같은 노이즈). 하네스 수준 패치이며 소스는 그대로다.
결과: logs/stage1/sr_batch_bench/<tag>.json. --collect로 results/stage1_sr_batch_bench.csv를 만든다.
이미지는 열지 않고 수치로만 비교한다."""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"
W = ROOT / "weights"
W5 = W / "stage05" / "sr"
SEED = 42
INP = ROOT / "data" / "sr_input" / "bounding_box_train"
SMOKE = ROOT / "runs" / "stage05" / "sr_smoke"
OUT = ROOT / "logs" / "stage1" / "sr_batch_bench"
WORK = Path(os.environ.get("SRREID_RAW", "/root/srreid_raw")) / "sr_batch_bench"
SERVER = "새 서버(EPYC 7282, cgroup v2, 116.4GiB)"
CSV_PATH = ROOT / "results" / "stage1_sr_batch_bench.csv"
FIELDS = ["server", "method", "k", "batch", "n_images", "img_s", "max_alloc_gib", "max_reserved_gib", "n_diff_vs_b1", "max_abs_diff_vs_b1",
          "batch1_vs_official_n_diff", "batch1_vs_official_max_abs", "status", "note"]
REPO = {"realesrgan": "Real-ESRGAN", "osdhuman": "OSDHuman", "pisa_sr": "PiSA-SR", "swinir": "SwinIR", "hat": "HAT", "mambairv2": "MambaIR"}


def collect():
    rows = []
    for p in sorted(OUT.glob("*.json")):
        if "__diag" in p.name:  # 진단 실행(SRREID_DIAG_TAG)은 표에 넣지 않는다
            continue
        rows += json.loads(p.read_text())["rows"]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"{len(rows)}행 -> {CSV_PATH}")


# ---------------------------------------------------------------- 방법별 어댑터
# prep(name) -> 이미지 하나의 공식 입력(디코딩만, 시간 측정 밖). run(items) -> RGB uint8 HxWx3 리스트(시간 측정 안).

class RealESRGAN:
    """inference_realesrgan.py: RRDBNet x4plus, tile 0, tile_pad 10, pre_pad 0, half=True(--fp32 없음), outscale 4."""
    def __init__(self, a):
        import cv2
        import numpy as np
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        self.cv2, self.np, self.torch = cv2, np, torch
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.up = RealESRGANer(scale=4, model_path=str(W / "RealESRGAN_x4plus.pth"), dni_weight=None, model=model, tile=0, tile_pad=10, pre_pad=0, half=True, gpu_id=None)
        assert self.up.mod_scale is None and self.up.pre_pad == 0 and self.up.tile_size == 0

    def prep(self, name):
        return self.cv2.imread(str(INP / f"{name}.png"), self.cv2.IMREAD_UNCHANGED)

    def run(self, items):
        np, torch, cv2, up = self.np, self.torch, self.cv2, self.up
        ts = []
        for img in items:  # RealESRGANer.enhance 앞부분(utils.py:195-216)과 pre_process(91-94)를 이미지별로 그대로
            assert img.ndim == 3 and img.shape[2] == 3
            img = img.astype(np.float32)
            assert np.max(img) <= 256
            img = cv2.cvtColor(img / 255, cv2.COLOR_BGR2RGB)
            ts.append(torch.from_numpy(np.transpose(img, (2, 0, 1))).float())
        x = torch.stack(ts).to(up.device)
        if up.half:
            x = x.half()
        with torch.no_grad():
            out = up.model(x)  # process()
        out = out.data.float().cpu().clamp_(0, 1).numpy()  # post_process 이후 squeeze().float().cpu().clamp_ (225행), 배치 차원만 남김
        res = []
        for o in out:
            o = np.transpose(o[[2, 1, 0], :, :], (1, 2, 0))  # BGR
            o = (o * 255.0).round().astype(np.uint8)
            res.append(np.ascontiguousarray(o[:, :, ::-1]))  # cv2.imwrite(BGR) -> PIL RGB로 읽은 것과 같은 배열
        return res


class SwinIR:
    """main_test_swinir.py classical_sr: fp32, 창 크기 8 배수로 '항상' 한 칸 더 뒤집어 패딩(h//8+1), tile 없음."""
    def __init__(self, a):
        import cv2
        import numpy as np
        import torch
        import main_test_swinir as mts
        self.cv2, self.np, self.torch = cv2, np, torch
        self.k = a.k
        ns = argparse.Namespace(task="classical_sr", scale=a.k, noise=15, jpeg=40, training_patch_size=64, large_model=False,
                                model_path=str(W5 / f"001_classicalSR_DF2K_s64w8_SwinIR-M_x{a.k}.pth"), folder_lq=None, folder_gt=None, tile=None, tile_overlap=32)
        self.args = ns
        self.model = mts.define_model(ns).eval().to(torch.device("cuda"))
        self.test = mts.test
        self.window_size = 8

    def prep(self, name):
        return self.cv2.imread(str(INP / f"{name}.png"), self.cv2.IMREAD_COLOR)  # get_image_pair: lq 경로는 같은 PNG로의 링크

    def run(self, items):
        np, torch = self.np, self.torch
        xs = []
        for im in items:
            img_lq = im.astype(np.float32) / 255.
            img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
            xs.append(torch.from_numpy(img_lq).float())
        img_lq = torch.stack(xs).to("cuda")
        ws = self.window_size
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // ws + 1) * ws - h_old
            w_pad = (w_old // ws + 1) * ws - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.test(img_lq, self.model, self.args, ws)
            output = output[..., :h_old * self.k, :w_old * self.k]
        output = output.data.float().cpu().clamp_(0, 1).numpy()
        res = []
        for o in output:
            o = np.transpose(o[[2, 1, 0], :, :], (1, 2, 0))
            o = (o * 255.0).round().astype(np.uint8)
            res.append(np.ascontiguousarray(o[:, :, ::-1]))
        return res


class BasicSRBase:
    """HAT·MambaIRv2: sr_generate.py와 같은 옵션을 basicsr parse_options로 읽고 build_model로 만든다. test_pipeline처럼 cudnn.benchmark=True."""
    tag = ""

    def __init__(self, a):
        import numpy as np
        import torch
        import yaml
        self.np, self.torch = np, torch
        if a.method == "hat":
            import basicsr.utils.matlab_functions as mf
            import basicsr.utils.color_util as cu
            mf.rgb2ycbcr = cu.rgb2ycbcr  # sr_generate.py와 같은 이름 잇기(추론 경로에서는 호출되지 않음)
            import hat.archs, hat.data, hat.models  # noqa: F401
            base = TP / "HAT" / "options" / "test" / f"HAT_SRx{a.k}_ImageNet-pretrain.yml"
            weight = W5 / f"HAT_SRx{a.k}_ImageNet-pretrain.pth"
        else:
            base = TP / "MambaIR" / "options" / "test" / "mambairv2" / f"test_MambaIRv2_SR_x{a.k}.yml"
            weight = W5 / f"mambairv2_classicSR_Base_x{a.k}.pth"
        from basicsr.data import build_dataset
        from basicsr.models import build_model
        from basicsr.utils import tensor2img
        from basicsr.utils.options import parse_options
        self.tensor2img = tensor2img
        work = WORK / f"{a.method}_x{a.k}"
        (work / "in").mkdir(parents=True, exist_ok=True)
        for n in a.names:
            l = work / "in" / f"{n}.png"
            if not l.exists():
                l.symlink_to(INP / f"{n}.png")
        o = yaml.safe_load(open(base))
        tag = "hat" if a.method == "hat" else "mamba"
        o["name"] = f"{tag}_x{a.k}"
        o["num_gpu"] = 1
        o["manual_seed"] = SEED
        o["datasets"] = {"test_1": {"name": "custom", "type": "SingleImageDataset", "dataroot_lq": str(work / "in"), "io_backend": {"type": "disk"}}}
        o["path"]["pretrain_network_g"] = str(weight)
        o["path"]["strict_load_g"] = True
        o["val"] = {"save_img": True, "suffix": "x"}
        opt_path = work / f"{tag}_x{a.k}.yml"
        yaml.safe_dump(o, open(opt_path, "w"))
        argv = sys.argv
        sys.argv = ["test.py", "-opt", str(opt_path)]
        opt, _ = parse_options(str(work), is_train=False)
        sys.argv = argv
        torch.backends.cudnn.benchmark = os.environ.get("SRREID_DIAG_CUDNN_BENCH", "1") == "1"  # basicsr/test.py test_pipeline과 동일(진단 때만 0)
        ds = build_dataset(opt["datasets"]["test_1"])
        self.ds = {Path(ds.paths[i]).stem: i for i in range(len(ds))}
        self.dataset = ds
        self.model = build_model(opt)
        self.opt = opt

    def prep(self, name):
        return self.dataset[self.ds[name]]["lq"]  # SingleImageDataset.__getitem__: imfrombytes float32 -> img2tensor(bgr2rgb)

    def to_rgb(self, result):
        res = []
        for i in range(result.shape[0]):
            bgr = self.tensor2img([result[i:i + 1]])  # 공식: tensor2img([visuals['result']]) (배치 1). 슬라이스 한 장씩 같은 함수로
            res.append(self.np.ascontiguousarray(bgr[:, :, ::-1]))
        return res


class HAT(BasicSRBase):
    def run(self, items):
        m = self.model
        m.feed_data({"lq": self.torch.stack(items)})  # dataloader collate(배치 1)와 같은 stack
        m.pre_process()
        assert "tile" not in m.opt
        m.process()
        m.post_process()
        res = self.to_rgb(m.get_current_visuals()["result"])
        del m.lq
        del m.output
        self.torch.cuda.empty_cache()  # hat_model.py nondist_validation과 같음
        return res


class GumbelReplay:
    """MambaIRv2 추론 경로의 F.gumbel_softmax(mambairv2_arch.py:248, hard=True)는 매 호출 CUDA RNG에서 exponential 노이즈를 뽑는다.
    replay 모드에서는 배치 1이 이미지마다(층 순서대로) 뽑던 노이즈를 같은 모양·dtype으로 미리 뽑아 두고, 배치 forward의 j번째 호출에
    이미지별 j번째 노이즈를 이어 붙여 준다. 나머지 계산은 torch.nn.functional.gumbel_softmax 본문과 같다. eps가 None이면 원래 함수."""
    def __init__(self, torch):
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.orig = F.gumbel_softmax
        self.noise, self.j, self.shapes = None, 0, None
        rp = self

        def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
            if rp.shapes is not None and rp.noise is None:  # 모양 기록 모드
                rp.shapes.append((tuple(logits.shape[1:]), logits.dtype))
            if rp.noise is None:
                return rp.orig(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            e = rp.noise[rp.j]
            rp.j += 1
            assert e.shape == logits.shape and e.dtype == logits.dtype, (e.shape, logits.shape)
            gumbels = -e.log()
            gumbels = (logits + gumbels) / tau
            y_soft = gumbels.softmax(dim)
            if hard:
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
                return y_hard - y_soft.detach() + y_soft
            return y_soft
        F.gumbel_softmax = gumbel_softmax

    def draw(self, n):
        torch = self.torch
        per = [[torch.empty((1,) + shp, dtype=dt, device="cuda").exponential_() for shp, dt in self.shapes] for _ in range(n)]  # 이미지 순, 층 순
        self.noise = [torch.cat([per[i][j] for i in range(n)]) for j in range(len(self.shapes))]
        self.j = 0
        return torch.cuda.get_rng_state()

    def clear(self):
        assert self.noise is None or self.j == len(self.noise), (self.j, len(self.noise))
        self.noise = None


class MambaIRv2(BasicSRBase):
    def __init__(self, a):
        super().__init__(a)
        self.noise = a.noise
        self.replay = GumbelReplay(self.torch)
        if self.noise == "replay":  # 호출 모양을 0 입력 한 장으로 기록하고 RNG를 되돌린다(공식 첫 이미지 때 상태 유지)
            torch = self.torch
            st = torch.cuda.get_rng_state()
            self.replay.shapes = []
            x = torch.zeros_like(self.prep(a.names[0]))[None]
            self.model.feed_data({"lq": x})
            mambairv2_test_batched(self.model, torch)
            del self.model.lq, self.model.output
            torch.cuda.set_rng_state(st)
            print("gumbel 호출", len(self.replay.shapes), self.replay.shapes[:1], flush=True)

    def run(self, items):
        m = self.model
        m.feed_data({"lq": self.torch.stack(items)})
        post = self.replay.draw(len(items)) if self.noise == "replay" else None
        mambairv2_test_batched(m, self.torch)
        if post is not None:
            self.replay.clear()
            self.torch.cuda.set_rng_state(post)
        res = self.to_rgb(m.get_current_visuals()["result"])
        del m.lq
        del m.output
        self.torch.cuda.empty_cache()
        return res


def mambairv2_test_batched(self, torch):
    """배치 어댑터. MambaIR/basicsr/models/mambairv2_model.py:13-104 test()와 같은 연산이고, 병합 버퍼만
    torch.zeros(1, C, ...)(72·95행) 대신 torch.zeros(B, C, ...)로 만든다. 128×64 입력은 분할 1×1이라 그물망 한 번 통과와 같다."""
    import torch.nn.functional as F
    B, C, h, w = self.lq.size()
    split_token_h = h // 200 + 1
    split_token_w = w // 200 + 1
    mod_pad_h, mod_pad_w = 0, 0
    if h % split_token_h != 0:
        mod_pad_h = split_token_h - h % split_token_h
    if w % split_token_w != 0:
        mod_pad_w = split_token_w - w % split_token_w
    img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    _, _, H, W_ = img.size()
    split_h = H // split_token_h
    split_w = W_ // split_token_w
    shave_h = split_h // 10
    shave_w = split_w // 10
    scale = self.opt.get('scale', 1)
    ral = H // split_h
    row = W_ // split_w
    slices = []
    for i in range(ral):
        for j in range(row):
            if i == 0 and i == ral - 1:
                top = slice(i * split_h, (i + 1) * split_h)
            elif i == 0:
                top = slice(i * split_h, (i + 1) * split_h + shave_h)
            elif i == ral - 1:
                top = slice(i * split_h - shave_h, (i + 1) * split_h)
            else:
                top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)
            if j == 0 and j == row - 1:
                left = slice(j * split_w, (j + 1) * split_w)
            elif j == 0:
                left = slice(j * split_w, (j + 1) * split_w + shave_w)
            elif j == row - 1:
                left = slice(j * split_w - shave_w, (j + 1) * split_w)
            else:
                left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
            slices.append((top, left))
    img_chops = [img[..., top, left] for top, left in slices]
    net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
    net.eval()
    with torch.no_grad():
        outputs = [net(chop) for chop in img_chops]
        _img = torch.zeros(B, C, H * scale, W_ * scale)
        for i in range(ral):
            for j in range(row):
                top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                _top = slice(0, split_h * scale) if i == 0 else slice(shave_h * scale, (shave_h + split_h) * scale)
                _left = slice(0, split_w * scale) if j == 0 else slice(shave_w * scale, (shave_w + split_w) * scale)
                _img[..., top, left] = outputs[i * row + j][..., _top, _left]
        self.output = _img
    if not hasattr(self, 'net_g_ema'):
        self.net_g.train()  # 공식과 같음(다음 호출 첫 줄에서 다시 eval)
    _, _, h2, w2 = self.output.size()
    self.output = self.output[:, :, 0:h2 - mod_pad_h * scale, 0:w2 - mod_pad_w * scale]


class NoiseReplay:
    """확산 방법의 노이즈 재현. diffusers DiagonalGaussianDistribution.sample과 DDPMScheduler.step의 randn_tensor를
    하네스에서만 바꿔, 미리 뽑아 둔 텐서를 쓰게 한다(eps가 None이면 원래 함수 그대로)."""
    def __init__(self, torch):
        from diffusers.models.autoencoders import vae as dvae
        from diffusers.schedulers import scheduling_ddpm as sd
        self.torch = torch
        self.eps = None
        self.sched = None
        orig_sample = dvae.DiagonalGaussianDistribution.sample
        orig_randn = sd.randn_tensor
        rp = self

        def sample(self_, generator=None):
            if rp.eps is None:
                return orig_sample(self_, generator)
            assert rp.eps.shape == self_.mean.shape and rp.eps.dtype == self_.parameters.dtype, (rp.eps.shape, self_.mean.shape, rp.eps.dtype, self_.parameters.dtype)
            return self_.mean + self_.std * rp.eps

        def randn(shape, generator=None, device=None, dtype=None, layout=None):
            if rp.sched is None:
                return orig_randn(shape, generator=generator, device=device, dtype=dtype, layout=layout)
            assert tuple(rp.sched.shape) == tuple(shape) and rp.sched.dtype == dtype, (rp.sched.shape, shape, rp.sched.dtype, dtype)
            return rp.sched
        dvae.DiagonalGaussianDistribution.sample = sample
        sd.randn_tensor = randn

    def draw(self, n, shape, with_sched, dtype):
        """배치 1 공식 경로가 이미지마다 부르는 순서(sample -> [scheduler.step])대로 같은 모양·dtype으로 뽑는다."""
        torch = self.torch
        e, s = [], []
        for _ in range(n):
            e.append(torch.randn((1,) + shape, generator=None, device=torch.device("cuda"), dtype=dtype, layout=torch.strided))
            if with_sched:
                s.append(torch.randn((1,) + shape, generator=None, device=torch.device("cuda"), dtype=dtype, layout=torch.strided))
        self.eps = torch.cat(e)
        self.sched = torch.cat(s) if with_sched else None
        return torch.cuda.get_rng_state()

    def clear(self):
        self.eps = self.sched = None


class OSDHuman:
    """run_inference.py: 공식 parse_args에 sr_generate.py와 같은 인자를 넣어 쓴다(fp16, adain, ram_img_only, 타일 크기 기본값)."""
    def __init__(self, a):
        sys.argv = ["run_inference.py", "-i", str(INP), "-o", str(WORK / "osd_unused"), "--osediff_path", str(W / "osdhuman" / "model.pkl"),
                    "--embedding_change_path", str(W / "osdhuman" / "model_embedding_change.pkl"), "--pretrained_model_name_or_path", str(W / "sd21_base"),
                    "--ram_path", str(W / "ram_swin_large_14m.pth"), "--ram_img_only", "--process_size", "512", "--upscale", "4", "--seed", str(SEED)]
        import run_inference as ri
        args = ri.parse_args()
        ri.set_seed(args.seed)
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        import torchvision.transforms.functional as tvf
        from my_utils.wavelet_color_fix import adain_color_fix
        from osediff import OSEDiff_test
        from ram import inference_ram_img_embed
        from ram.inference import AttentionPooling
        from ram.models.ram_lora import ram
        self.np, self.torch, self.Image, self.transforms, self.tvf = np, torch, Image, transforms, tvf
        self.adain, self.embed = adain_color_fix, inference_ram_img_embed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
        self.ram_transforms = transforms.Compose([transforms.Resize((384, 384)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.model = OSEDiff_test(args)
        assert args.ram_img_only and not args.use_vre and args.align_method == "adain"
        self.ram_model = ram(ram_img_only=True, pretrained=args.ram_path, pretrained_condition=None, image_size=384, vit="swin_l")
        ec = AttentionPooling(args, 512, 77, 2)
        ec.load_state_dict(torch.load(args.embedding_change_path, map_location="cpu"))
        ec.eval().to("cuda", dtype=self.dtype)
        self.embedding_change = ec
        self.ram_model.eval().to("cuda", dtype=self.dtype)
        self.replay = NoiseReplay(torch)
        self.noise = a.noise

    def prep(self, name):
        return self.Image.open(INP / f"{name}.png").convert("RGB")

    def _resize(self, input_image):
        args, Image = self.args, self.Image
        ori_width, ori_height = input_image.size
        resize_flag = False
        if ori_width < args.process_size // args.upscale or ori_height < args.process_size // args.upscale:
            scale = (args.process_size // args.upscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0] * args.upscale, input_image.size[1] * args.upscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        return input_image, resize_flag, ori_width, ori_height

    def run(self, items):
        torch, args = self.torch, self.args
        pre = [self._resize(im) for im in items]
        with torch.no_grad():
            lq = torch.stack([self.tvf.to_tensor(p[0]) for p in pre]).cuda() * 2 - 1
            lq = lq.to(dtype=self.dtype)
            post_state = None
            if self.noise == "replay":
                post_state = self.replay.draw(lq.shape[0], (4, lq.shape[2] // 8, lq.shape[3] // 8), True, self.dtype)
            prompt_embedding = self.embed(self.ram_transforms(lq), self.ram_model)
            visual_prompt = self.embedding_change(prompt_embedding)
            output_image = self.model(lq, prompt=None, visual_prompt=visual_prompt)
            if post_state is not None:
                self.replay.clear()
                torch.cuda.set_rng_state(post_state)
            torch.cuda.synchronize()
        res = []
        for i, (input_image, resize_flag, ow, oh) in enumerate(pre):
            output_pil = self.transforms.ToPILImage()(output_image[i].cpu() * 0.5 + 0.5)
            output_pil = self.adain(target=output_pil, source=input_image)
            if resize_flag:
                output_pil = output_pil.resize((int(args.upscale * ow), int(args.upscale * oh)), self.Image.LANCZOS)
            res.append(self.np.asarray(output_pil.convert("RGB")).copy())
        return res


class PiSASR:
    """test_pisasr.py: 인자 기본값을 그대로 옮기고 sr_generate.py 인자(lambda_pix 1.0, lambda_sem, seed 42)를 넣는다. --default 없음.
    공식 스크립트는 args.seed를 쓰지 않으므로 시드는 runner.py가 건 것(random/np/torch/cuda 42)뿐이다. 여기서도 같게 건다."""
    def __init__(self, a):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        import torchvision.transforms.functional as F
        from pisasr import PiSASR_eval
        from src.my_utils.wavelet_color_fix import adain_color_fix
        self.np, self.torch, self.Image, self.transforms, self.F, self.adain = np, torch, Image, transforms, F, adain_color_fix
        args = argparse.Namespace(input_image=str(INP), output_dir=str(WORK / "pisa_unused"), pretrained_model_path=str(W / "sd21_base"),
                                  pretrained_path=str(W / "pisa_sr" / "pisa_sr.pkl"), seed=SEED, process_size=512, upscale=4, align_method="adain",
                                  lambda_pix=1.0, lambda_sem=a.lambda_sem, vae_decoder_tiled_size=224, vae_encoder_tiled_size=1024,
                                  latent_tiled_size=96, latent_tiled_overlap=32, mixed_precision="fp16", default=False)
        self.args = args
        self.model = PiSASR_eval(args)
        self.model.set_eval()
        self.replay = NoiseReplay(torch)
        self.noise = a.noise

    def prep(self, name):
        return self.Image.open(INP / f"{name}.png").convert("RGB")

    def _resize(self, input_image):
        args, Image = self.args, self.Image
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        return input_image, resize_flag, ori_width, ori_height

    def _forward_batched(self, c_t, prompt):
        """배치 어댑터. pisasr.py:474-493 PiSASR_eval.forward와 같은 연산이고, 프롬프트 임베딩(배치 1, 480행)만 배치 크기로 반복한다.
        빈 프롬프트 임베딩은 이미지와 무관한 같은 값이다. 배치 1이면 공식 forward를 그대로 부른다."""
        m = self.model
        if c_t.shape[0] == 1:
            return m(self.args.default, c_t, prompt=prompt)[1]
        c_t = c_t.to(dtype=m.weight_dtype)
        prompt_embeds = m.encode_prompt([prompt]).to(dtype=m.weight_dtype).repeat(c_t.shape[0], 1, 1)
        encoded_control = m.vae.encode(c_t).latent_dist.sample() * m.vae.config.scaling_factor
        model_pred = m._process_latents(encoded_control, prompt_embeds, self.args.default)
        x_denoised = encoded_control - model_pred
        return m.vae.decode(x_denoised / m.vae.config.scaling_factor).sample.clamp(-1, 1)

    def run(self, items):
        torch, args = self.torch, self.args
        pre = [self._resize(im) for im in items]
        with torch.no_grad():
            c_t = torch.stack([self.F.to_tensor(p[0]) for p in pre]).cuda() * 2 - 1
            post_state = None
            if self.noise == "replay":
                post_state = self.replay.draw(c_t.shape[0], (4, c_t.shape[2] // 8, c_t.shape[3] // 8), False, self.model.weight_dtype)
            output_image = self._forward_batched(c_t, "")
            if post_state is not None:
                self.replay.clear()
                torch.cuda.set_rng_state(post_state)
        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        res = []
        for i, (input_image, resize_flag, ow, oh) in enumerate(pre):
            output_pil = self.transforms.ToPILImage()(output_image[i].cpu())
            output_pil = self.adain(target=output_pil, source=input_image)
            if resize_flag:
                output_pil = output_pil.resize((int(args.upscale * ow), int(args.upscale * oh)))
            res.append(self.np.asarray(output_pil.convert("RGB")).copy())
        return res


ADAPTERS = {"realesrgan": RealESRGAN, "swinir": SwinIR, "hat": HAT, "mambairv2": MambaIRv2, "osdhuman": OSDHuman, "pisa_sr": PiSASR}


# ---------------------------------------------------------------- 측정

def diff_stats(np, a_list, b_list):
    n_diff, max_abs, s, cnt, tot = 0, 0, 0, 0, 0
    for a, b in zip(a_list, b_list):
        tot += a.size
        if a.shape != b.shape:
            return {"n_diff": len(a_list), "max_abs": -1, "mean_abs_diff_px": None, "shape_mismatch": [list(a.shape), list(b.shape)]}
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        if d.any():
            n_diff += 1
            max_abs = max(max_abs, int(d.max()))
            s += int(d.sum())
            cnt += int((d > 0).sum())
    return {"n_diff": n_diff, "max_abs": max_abs, "mean_abs_diff_px": (s / cnt) if cnt else 0.0, "frac_px": cnt / tot if tot else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(ADAPTERS))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lambda-sem", dest="lambda_sem", type=float, default=1.0)
    ap.add_argument("--noise", choices=["global", "replay"], default="global")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--anchor-only", action="store_true", help="배치 1 공식 대조만")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        return collect()

    import importlib.util
    spec = importlib.util.spec_from_file_location("srreid_guard", str(ROOT / "common" / "guard.py"))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install()
    repo = TP / REPO[a.method]
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]  # stage05/의 cache.py 등이 공식 모듈을 가리지 않게
    sys.path.insert(0, str(repo))  # runner.py와 같음(PYTHONPATH=repo도 함께 준다)
    os.chdir(WORK.parent if WORK.parent.exists() else "/tmp")
    import numpy as np
    import torch
    from PIL import Image
    # runner.py와 같은 시드·메모리 비율
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_per_process_memory_fraction(0.95)
    if os.environ.get("SRREID_DIAG_SDPA") == "nomem":  # 진단 전용(공식 수치 아님): torch 2.0.1 mem-efficient SDPA 끄기 -> flash 커널
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    mname = a.method if a.method != "pisa_sr" else f"pisa_sr_sem{a.lambda_sem}"
    off_dir = SMOKE / mname / f"x{a.k}" / "bounding_box_train"
    names = sorted(p.stem for p in off_dir.glob("*.webp"))
    assert len(names) == 100, len(names)
    a.names = names
    diffusion = a.method in ("osdhuman", "pisa_sr", "mambairv2")  # 추론에 난수를 쓰는 방법(MambaIRv2는 gumbel_softmax 라우팅)
    tag = f"{mname}__x{a.k}" + (f"__noise-{a.noise}" if diffusion else "") + ("__anchor" if a.anchor_only else "") + os.environ.get("SRREID_DIAG_TAG", "")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    ad = ADAPTERS[a.method](a)
    t_load = time.perf_counter() - t_load
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()

    def reset_rng():  # 모델 생성 직후 상태(공식 실행에서 첫 이미지를 넣을 때의 상태)로 되돌린다.
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)

    inputs = [ad.prep(n) for n in names]
    official = []
    for n in names:
        with Image.open(off_dir / f"{n}.webp") as im:
            official.append(np.asarray(im.convert("RGB")).copy())

    def full_pass(bs):
        reset_rng()
        outs, t_gpu = [], 0.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(inputs), bs):
            outs += ad.run(inputs[i:i + bs])
        torch.cuda.synchronize()
        return outs, time.perf_counter() - t0

    base = {"server": SERVER, "method": mname, "k": a.k, "n_images": len(names)}
    rows, log = [], {"tag": tag, "diag_env": {k: v for k, v in os.environ.items() if k.startswith("SRREID_DIAG")}, "load_seconds": t_load, "torch": torch.__version__, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "gpu": torch.cuda.get_device_name(0), "noise": a.noise if diffusion else None}
    torch.cuda.reset_peak_memory_stats()
    b1, t_b1 = full_pass(1)
    anchor = diff_stats(np, b1, official)
    import hashlib
    log["anchor_sha1"] = {n: hashlib.sha1(o.tobytes()).hexdigest() for n, o in zip(names, b1)}  # 프로세스 간 재현성 비교용
    log["anchor"] = dict(anchor, seconds=t_b1, peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    print(json.dumps({"anchor": log["anchor"]}, ensure_ascii=False), flush=True)
    noise_note = f"noise={a.noise}; " if diffusion else ""
    if a.anchor_only:
        rows.append(dict(base, batch=1, img_s=len(names) / t_b1, batch1_vs_official_n_diff=anchor["n_diff"], batch1_vs_official_max_abs=anchor["max_abs"],
                         n_diff_vs_b1=0, max_abs_diff_vs_b1=0, status="anchor_only",
                         note=f"{noise_note}배치1 공식 대조만(예열 없는 첫 실행 시간). 공식 대비 차이 픽셀 평균 {anchor['mean_abs_diff_px']}"))
    else:
        for bs in a.batches:
            r = dict(base, batch=bs, batch1_vs_official_n_diff=anchor["n_diff"], batch1_vs_official_max_abs=anchor["max_abs"])
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                full_pass(bs)  # 예열: 이 배치 크기의 모든 모양(마지막 나머지 배치 포함)
                warm_alloc, warm_res = torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                outs, sec = full_pass(bs)
                d = diff_stats(np, outs, b1)
                r.update(img_s=len(names) / sec, max_alloc_gib=torch.cuda.max_memory_allocated() / 2**30, max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                         n_diff_vs_b1=d["n_diff"], max_abs_diff_vs_b1=d["max_abs"], status="ok",
                         note=f"{noise_note}마지막 배치 {len(names) % bs or bs}장(패딩 없음); 차이 픽셀 평균 |d| {d['mean_abs_diff_px']:.4g}, 차이 값 비율 {d['frac_px']:.3g}(전체 H·W·C 값 중); "
                              f"예열 피크 alloc {warm_alloc:.3f}/reserved {warm_res:.3f} GiB; 시간=전처리(디코딩 제외)+추론+uint8 후처리")
                r["_mean_abs_diff_px"] = d["mean_abs_diff_px"]
                r["_frac_px"] = d["frac_px"]
                r["_seconds"] = sec
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                r.update(status="OOM", note=f"{noise_note}{type(e).__name__}: {str(e).splitlines()[0][:200]}")
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
                break
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    log["rows"] = rows
    (OUT / f"{tag}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print("저장:", OUT / f"{tag}.json", flush=True)


if __name__ == "__main__":
    main()
