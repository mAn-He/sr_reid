"""G.2~G.5 단위 테스트. 합성 배열만 쓰고 데이터 이미지를 읽지 않는다. srreid-quality env."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stage05 import quality_tools as Q  # noqa: E402
from stage05.attr_train import is_color  # noqa: E402


def ok(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)
    if not cond:
        fails.append(name)


fails = []
rng = np.random.default_rng(0)
smooth = np.asarray(Image.fromarray(rng.integers(0, 255, (8, 4, 3), np.uint8)).resize((64, 128), Image.BICUBIC))
sr = np.asarray(Image.fromarray(smooth).resize((64 * 4, 128 * 4), Image.BICUBIC))
ok("psnr identical inf", Q.psnr(smooth, smooth) == float("inf"))
ok("psnr known mse", abs(Q.psnr(np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 1, np.uint8)) - 48.1308) < 1e-3)
lr = Q.lr_psnr(sr, smooth, 4)
ok("lr_psnr high for bicubic up then down", lr["lr_psnr_rgb"] > 35, str(lr))
try:
    Q.lr_psnr(sr[:-1], smooth, 4)
    ok("lr_psnr rejects wrong size", False)
except ValueError:
    ok("lr_psnr rejects wrong size", True)
ok("deltaE zero on identical", max(Q.color_delta_e(smooth, smooth).values()) < 1e-9)
red, blue = np.zeros((128, 64, 3), np.uint8), np.zeros((128, 64, 3), np.uint8)
red[..., 0], blue[..., 2] = 200, 200
d = Q.color_delta_e(red, blue)
ok("deltaE large red vs blue", d["de2000_top"] > 40 and d["de2000_bottom"] > 40, str(d))
half = red.copy()
half[64:] = blue[64:]
d = Q.color_delta_e(half, red)
ok("deltaE separates top and bottom", d["de2000_top"] < 1e-9 and d["de2000_bottom"] > 40, str(d))
ok("attribute split", is_color("upblack") and is_color("downblue") and not is_color("gender") and not is_color("backpack") and not is_color("up"))
per = Q.Perceptual(device=os.environ.get("SRREID_QUALITY_DEVICE", "cuda"), dreamsim=True)
same, diff = per(smooth, smooth), per(smooth, np.flipud(smooth).copy())
print(same, diff)
for k in ("lpips", "dists", "dreamsim"):
    ok(f"{k} zero-ish same < diff", same[k] < 1e-3 and diff[k] > same[k], f"{same[k]:.5f} {diff[k]:.5f}")
ok("psnr_y high for same", same["psnr_y"] > 60, str(same["psnr_y"]))
ok("ssim_y one for same", abs(same["ssim_y"] - 1) < 1e-3, str(same["ssim_y"]))
print("실패", fails)
sys.exit(1 if fails else 0)
