"""Part G. SR 품질 지표. srreid-quality env. 이미지를 열람하지 않고 수치만 쓴다. Stage 1·2에서 재사용한다."""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

ROOT = Path(__file__).resolve().parents[1]
DREAMSIM_CACHE = ROOT / "weights" / "stage05" / "dreamsim"


def psnr(a, b, data_range=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def rgb_to_y(a):
    return 16.0 + (65.481 * a[..., 0] + 128.553 * a[..., 1] + 24.966 * a[..., 2]) / 255.0


def lr_psnr(sr, lr, k):
    """SR 출력을 PIL BICUBIC으로 ×1/k 축소해 원본과 비교한다. RGB와 Y."""
    h, w = lr.shape[:2]
    if sr.shape[0] != h * k or sr.shape[1] != w * k:
        raise ValueError(f"크기 불일치 {sr.shape} {lr.shape} k={k}")
    down = np.asarray(Image.fromarray(sr).resize((w, h), Image.BICUBIC))
    return {"lr_psnr_rgb": psnr(down, lr), "lr_psnr_y": psnr(rgb_to_y(down.astype(np.float64)), rgb_to_y(lr.astype(np.float64)))}


def region_lab(a):
    """상단/하단 절반의 평균 Lab. 상·하의 영역의 근사."""
    lab = rgb2lab(a)
    h = a.shape[0] // 2
    return lab[:h].mean((0, 1)), lab[h:].mean((0, 1))


def color_delta_e(sr, ref):
    """상·하의 근사 영역의 평균 Lab ΔE2000. 크기가 달라도 영역 평균끼리 비교한다."""
    t1, b1 = region_lab(sr)
    t2, b2 = region_lab(ref)
    return {"de2000_top": float(deltaE_ciede2000(t1, t2)), "de2000_bottom": float(deltaE_ciede2000(b1, b2))}


class Perceptual:
    """LPIPS(alex), DISTS는 pyiqa, SSIM·PSNR은 pyiqa Y 채널(파일럿 지표와 같은 구현). DreamSim은 HF_HUB_OFFLINE=1로 동작."""

    def __init__(self, device="cuda", dreamsim=True):
        import pyiqa
        self.device = device
        self.m = {"lpips": pyiqa.create_metric("lpips", device=device), "dists": pyiqa.create_metric("dists", device=device),
                  "psnr_y": pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device=device),
                  "ssim_y": pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr", device=device)}
        self.ds, self.pp = None, None
        if dreamsim:
            from dreamsim import dreamsim as make
            self.ds, self.pp = make(pretrained=True, cache_dir=str(DREAMSIM_CACHE), device=device)

    @staticmethod
    def _t(a, device):
        import torch
        return torch.from_numpy(a.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255

    def __call__(self, sr, ref):
        import torch
        x, y = self._t(sr, self.device), self._t(ref, self.device)
        with torch.no_grad():
            out = {k: float(f(x, y).item()) for k, f in self.m.items()}
            if self.ds is not None:
                d = self.ds(self.pp(Image.fromarray(sr)).to(self.device), self.pp(Image.fromarray(ref)).to(self.device))
                out["dreamsim"] = float(d.item())
        return out
