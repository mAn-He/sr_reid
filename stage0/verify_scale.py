"""KPR 키포인트·마스크 스케일 수치 검증. srreid-kpr env, CPU로 돈다. 이미지를 보지 않고 수치만 쓴다. 결과는 results/stage0_scale_check.json."""
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
import torch  # noqa: E402

from stage0 import backends  # noqa: E402
from stage0.scale import scale_coord  # noqa: E402

ORIG = ROOT / "data" / "kpr" / "orig"
KP = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"
MASK = "pifpaf_maskrcnn_filtering"
N = 10


def centroid(m):
    ys, xs = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    t = m.sum()
    return (float((xs * m).sum() / t), float((ys * m).sum() / t)) if t > 1e-6 else None


def main():
    out = {"n_samples": N}
    files = sorted(glob.glob(str(ORIG / "external_annotation" / KP / "bounding_box_train" / "*.json")))
    files = [f for f in files if json.load(open(f))][:N]
    stems = [os.path.basename(f).split(".")[0] for f in files]
    for k in (2, 3):
        err_formula, gap_xk, conf_err = 0.0, 0.0, 0.0
        for f in files:
            a = np.array(json.load(open(f))[0]["keypoints"])
            b = np.array(json.load(open(ROOT / "data" / "kpr" / f"x{k}" / "external_annotation" / KP / "bounding_box_train" / os.path.basename(f)))[0]["keypoints"])
            err_formula = max(err_formula, float(np.abs(b[:, :2] - scale_coord(a[:, :2], k)).max()))
            gap_xk = max(gap_xk, float(np.abs(b[:, :2] - a[:, :2] * k).max()))
            conf_err = max(conf_err, float(np.abs(b[:, 2] - a[:, 2]).max()))
        # 마스크: 저장 해상도는 고정이고 로더의 Resize가 입력 크기로 맞춘다. 로더와 같은 함수로 ×1, ×k 렌더링을 비교한다.
        sys.path.insert(0, str(backends.KPR_REPO))
        from torchreid.data.masks_transforms.resize import Resize
        shapes = set()
        worst, worst_center, n_parts = 0.0, 0.0, 0
        for s in stems:
            m = np.load(ORIG / "masks" / MASK / "bounding_box_train" / f"{s}.npy")
            shapes.add(str(m.shape))
            for c in range(m.shape[0]):
                base = Resize(128, 64).apply_to_mask(m[c], interpolation=1)
                big = Resize(128 * k, 64 * k).apply_to_mask(m[c], interpolation=1)
                c0, c1 = centroid(base), centroid(big)
                if c0 and c1:
                    worst = max(worst, abs(c1[0] - c0[0] * k), abs(c1[1] - c0[1] * k))
                    worst_center = max(worst_center, abs(c1[0] - scale_coord(c0[0], k)), abs(c1[1] - scale_coord(c0[1], k)))
                    n_parts += 1
        # 실제 로더 통과: 로더가 원본 크기를 가정하는지 확인한다.
        b = backends.KprBackend(k, 64, tag=f"verify{k}", eval_batch=8, eval_pool=N)
        ds = b.engine.test_loader[b.name]["query"].dataset
        shapes_out = []
        for i in range(N):
            sample = ds[i]
            shapes_out.append({"image": list(sample["image"].shape), "target_masks": list(sample["target_masks"].shape),
                               "prompt_masks": list(sample["prompt_masks"].shape)})
        kp_stats = {"max_err_vs_pixel_center_formula": err_formula, "max_gap_vs_x_times_k": gap_xk,
                    "theoretical_gap_(k-1)/2": (k - 1) / 2, "max_confidence_change": conf_err}
        out[f"x{k}"] = {"keypoints": kp_stats,
                        "masks": {"stored_shapes": sorted(shapes), "scaled_on_disk": False, "n_part_channels": n_parts,
                                  "max_centroid_gap_vs_x_times_k_px": worst, "within_1px": worst <= 1.0,
                                  "max_centroid_gap_vs_pixel_center_formula_px": worst_center},
                        "loader_output_shapes": shapes_out[0], "loader_all_same_shape": all(s == shapes_out[0] for s in shapes_out)}
        print(k, json.dumps(out[f"x{k}"], ensure_ascii=False), flush=True)
    passed = all(out[f"x{k}"]["keypoints"]["max_err_vs_pixel_center_formula"] < 1e-9 and out[f"x{k}"]["masks"]["within_1px"]
                 and out[f"x{k}"]["loader_all_same_shape"] for k in (2, 3))
    out["passed"] = passed
    (ROOT / "results" / "stage0_scale_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
