"""val query HR 대비 PSNR/SSIM(Y채널), LPIPS(alex). pyiqa 사용. 깨진 이미지 판정용 Y 표준편차도 기록한다."""
import argparse

import numpy as np
import pyiqa
import torch
from PIL import Image

from common.util import load_cfg, log_failure, read_csv, wait_free_gb, write_csv
from sr.run import cond_dir, conditions


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), np.float32).transpose(2, 0, 1) / 255)


def y_std(x):
    """BT.601 Y 표준편차, 0-255 스케일."""
    return (65.481 * x[0] + 128.553 * x[1] + 24.966 * x[2] + 16).std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    wait_free_gb(cfg["gpu"]["min_free_gb_eval"], cfg["gpu"]["wait_timeout_s"])
    torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["fraction_train"])
    psnr = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device="cuda")
    ssim = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr", device="cuda")
    lpips = pyiqa.create_metric("lpips", device="cuda")
    query = {f"{r['name']}.png": r for r in read_csv(cfg["_manifests"] / "split.csv") if r["role"] == "query"}
    out_dir = cfg["_run"] / "eval" / "fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = cfg["eval"]["fidelity"]["batch"]
    for c in conditions(cfg):
        out = out_dir / f"{c['cid']}.csv"
        d = cond_dir(cfg, c)
        if out.exists() or not (d / "outputs.csv").exists():
            continue
        try:
            names = [r["name"] for r in read_csv(d / "outputs.csv") if r["status"] == "ok"]
            rows = []
            for i in range(0, len(names), batch):
                part = names[i:i + batch]
                sr = [load(d / "final" / n) for n in part]
                hr = [load(query[n]["path"]) for n in part]
                x, y = torch.stack(sr).cuda(), torch.stack(hr).cuda()
                with torch.no_grad():
                    p, s, l = psnr(x, y), ssim(x, y), lpips(x, y)
                for j, n in enumerate(part):
                    rows.append({"name": n, "psnr": p[j].item(), "ssim": s[j].item(), "lpips": l[j].item(),
                                 "y_std": y_std(sr[j])})
            write_csv(out, rows, ["name", "psnr", "ssim", "lpips", "y_std"])
            print(f"{c['cid']}: PSNR={np.mean([r['psnr'] for r in rows]):.3f}", flush=True)
        except (RuntimeError, OSError, ValueError) as e:
            log_failure(cfg, "eval_fidelity", c["cid"], "", e)


if __name__ == "__main__":
    main()
