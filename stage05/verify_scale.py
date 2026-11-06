"""Part 0. 마스크·키포인트 스케일 재검증(×3, ×4). 기대 무게중심은 픽셀 중심 규약 (c+0.5)·k−0.5. srreid-kpr env, CPU. 수치만 쓴다."""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402

from stage0.backends import KPR_REPO  # noqa: E402
from stage0.scale import scale_coord  # noqa: E402

sys.path.insert(0, str(KPR_REPO))
from torchreid.data.masks_transforms.resize import Resize  # noqa: E402

ORIG = ROOT / "data" / "kpr" / "orig"
KP = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"
MASK = "pifpaf_maskrcnn_filtering"
N_KP, N_MASK = 2000, 200
THRESH = 1.0


def centroid(m):
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    t = m.sum()
    return (float((xs * m).sum() / t), float((ys * m).sum() / t)) if t > 1e-6 else None


def nearest_upscale_check(k):
    """수식 자체 점검: 픽셀 i가 [ik, (i+1)k)로 복제되는 정수배 nearest에서 무게중심이 (c+0.5)k−0.5와 일치해야 한다."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        m = rng.random((17, 9), dtype=np.float32)
        big = np.repeat(np.repeat(m, k, 0), k, 1)
        c0, c1 = centroid(m), centroid(big)
        worst = max(worst, abs(c1[0] - scale_coord(c0[0], k)), abs(c1[1] - scale_coord(c0[1], k)))
    return worst


def loader_mask(m, h, w):
    """공식 로더 경로: Resize(H, W) cv2 INTER_LINEAR(default_config.py:214, masks_transforms/resize.py:49) → ResizeMasks(H/4, W/4) bilinear(transforms.py:218).
    입력은 저장 마스크 36×17×9(채널 우선). 출력은 36×(H/4)×(W/4)."""
    import torch
    big = np.stack([Resize(h, w).apply_to_mask(c, interpolation=1) for c in m])
    t = torch.from_numpy(big).unsqueeze(0)
    return torch.nn.functional.interpolate(t, (h // 4, w // 4), mode="bilinear").squeeze(0).numpy()


def expected_from_stored(c, n_src, n_dst):
    """half-pixel 대응 x_src = (x_dst+0.5)·n_src/n_dst − 0.5의 역. cv2 INTER_LINEAR와 align_corners=False bilinear가 모두 이 대응을 쓴다."""
    return (c + 0.5) * n_dst / n_src - 0.5


def loader_path_check(mask_files, k):
    """로더 경로 ×k 마스크의 무게중심이 저장 마스크 무게중심의 half-pixel 사상과 얼마나 다른지. 픽셀 단위는 H/4 격자이며 이미지 좌표로는 4배다."""
    h, w = 128 * k, 64 * k
    gaps, gaps_vs_x1, edge = [], [], []
    for f in mask_files:
        m = np.load(f)
        out = loader_mask(m, h, w)
        base = loader_mask(m, 128, 64)
        for c in range(m.shape[0]):
            c0, c1, cb = centroid(m[c]), centroid(out[c]), centroid(base[c])
            if not (c0 and c1 and cb):
                continue
            gaps.append(max(abs(c1[0] - expected_from_stored(c0[0], m.shape[2], w // 4)), abs(c1[1] - expected_from_stored(c0[1], m.shape[1], h // 4))))
            gaps_vs_x1.append(max(abs(c1[0] - scale_coord(cb[0], k)), abs(c1[1] - scale_coord(cb[1], k))))
            edge.append(float(m[c][0].sum() + m[c][-1].sum() + m[c][:, 0].sum() + m[c][:, -1].sum()) / float(m[c].sum()))
    g, gx, e = np.array(gaps), np.array(gaps_vs_x1), np.array(edge)
    inner = e < 0.05
    return {"grid": f"{h // 4}x{w // 4}", "n_part_channels": len(g), "unit": "H/4 격자 픽셀(이미지 좌표는 ×4)",
            "vs_stored_half_pixel": {"mean": float(g.mean()), "p99": float(np.percentile(g, 99)), "max": float(g.max())},
            "vs_stored_half_pixel_mass_away_from_edge": {"n": int(inner.sum()), "mean": float(g[inner].mean()), "p99": float(np.percentile(g[inner], 99)), "max": float(g[inner].max())},
            "vs_x1_loader_(c+0.5)k-0.5": {"mean": float(gx.mean()), "p99": float(np.percentile(gx, 99)), "max": float(gx.max())},
            "within_threshold_mean": bool(g.mean() <= THRESH)}


def main():
    files = sorted(glob.glob(str(ORIG / "external_annotation" / KP / "bounding_box_train" / "*.json")))
    files = [f for f in files if json.load(open(f))]
    rng = np.random.default_rng(0)
    kp_files = [files[i] for i in rng.choice(len(files), min(N_KP, len(files)), replace=False)]
    masks = sorted(glob.glob(str(ORIG / "masks" / MASK / "bounding_box_train" / "*.npy")))
    mask_files = [masks[i] for i in rng.choice(len(masks), N_MASK, replace=False)]
    out = {"formula": "(c+0.5)*k-0.5", "old_wrong_formula": "c*k", "threshold_px": THRESH, "n_keypoint_files": len(kp_files),
           "n_mask_files": len(mask_files), "split": "bounding_box_train"}
    passed = True
    for k in (3, 4):
        err, gap, conf = 0.0, 0.0, 0.0
        for f in kp_files:
            for sa, sb in zip(json.load(open(f)), json.load(open(ROOT / "data" / "kpr" / f"x{k}" / "external_annotation" / KP / "bounding_box_train" / os.path.basename(f)))):
                a, b = np.array(sa["keypoints"]), np.array(sb["keypoints"])
                err = max(err, float(np.abs(b[:, :2] - scale_coord(a[:, :2], k)).max()))
                gap = max(gap, float(np.abs(b[:, :2] - a[:, :2] * k).max()))
                conf = max(conf, float(np.abs(b[:, 2] - a[:, 2]).max()))
        res = {"keypoints": {"max_err_vs_pixel_center_px": err, "max_gap_vs_old_c_times_k_px": gap, "theory_gap_(k-1)/2": (k - 1) / 2,
                             "max_confidence_change": conf},
               "pure_nearest_formula_check_px": nearest_upscale_check(k)}
        for name, interp in (("nearest", 0), ("bilinear", 1)):
            worst_new, worst_old, n = 0.0, 0.0, 0
            for f in mask_files:
                m = np.load(f)
                for c in range(m.shape[0]):
                    base = Resize(128, 64).apply_to_mask(m[c], interpolation=interp)
                    big = Resize(128 * k, 64 * k).apply_to_mask(m[c], interpolation=interp)
                    c0, c1 = centroid(base), centroid(big)
                    if c0 and c1:
                        worst_new = max(worst_new, abs(c1[0] - scale_coord(c0[0], k)), abs(c1[1] - scale_coord(c0[1], k)))
                        worst_old = max(worst_old, abs(c1[0] - c0[0] * k), abs(c1[1] - c0[1] * k))
                        n += 1
            res[f"masks_{name}"] = {"n_part_channels": n, "max_gap_vs_pixel_center_px": worst_new, "max_gap_vs_old_c_times_k_px": worst_old,
                                    "within_threshold": worst_new <= THRESH}
        # 판정은 로더 경로(공식 Resize, interpolation=1, default_config.py:214)와 정수배 nearest 수식 점검으로 한다.
        # 저장 해상도 17×9에서 직접 비정수배 nearest로 그린 값은 로더가 쓰지 않으므로 참고로만 남긴다.
        res["masks_nearest"]["used_by_loader"] = False
        res["masks_bilinear"]["used_by_loader"] = True
        res["masks_loader_path"] = loader_path_check(mask_files, k)
        ok = err < 1e-9 and res["pure_nearest_formula_check_px"] < 1e-3 and res["masks_bilinear"]["within_threshold"]
        res["passed"] = ok
        passed = passed and ok
        out[f"x{k}"] = res
        print(k, json.dumps(res, ensure_ascii=False), flush=True)
    out["passed"] = passed
    (ROOT / "results" / "stage05_scale_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
