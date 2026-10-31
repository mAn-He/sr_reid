"""×2, ×3 데이터 준비. bounding_box_train만 읽는다. 끝난 파일은 건너뛴다.
data/bicubic_x{k}: PIL BICUBIC 확대 PNG, data/kpr/x{k}: KPR 로더 구조(이미지·마스크 심볼릭 링크, 스케일한 키포인트 JSON)."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from common import guard
from common.util import ROOT, sha256_file, write_csv
from stage0.scale import scale_skeleton

SRC = ROOT / "data" / "market" / "bounding_box_train"
ORIG = ROOT / "data" / "kpr" / "orig"
KP = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"
MASK = "pifpaf_maskrcnn_filtering"
FIELDS = ["name", "input_sha256", "output_sha256", "height", "width"]


def upscale(args):
    src, dst, k = args
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if not dst.exists():
            tmp = dst.with_suffix(".tmp.png")
            im.resize((w * k, h * k), Image.BICUBIC).save(tmp)
            os.replace(tmp, dst)
    return {"name": dst.name, "input_sha256": sha256_file(src), "output_sha256": sha256_file(dst),
            "height": h * k, "width": w * k}


def scale_kp(args):
    src, dst, k = args
    if not dst.exists():
        dst.write_text(json.dumps([scale_skeleton(s, k) for s in json.loads(src.read_text())]))


def main():
    guard.install()
    srcs = sorted(SRC.glob("*.jpg"))
    if len(srcs) != 12936:
        raise SystemExit(f"bounding_box_train 이미지 수 이상: {len(srcs)}")
    for k in (2, 3):
        out = ROOT / "data" / f"bicubic_x{k}"
        out.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(16) as ex:
            rows = list(ex.map(upscale, [(p, out / (p.stem + ".png"), k) for p in srcs], chunksize=64))
        write_csv(ROOT / "data" / f"bicubic_x{k}_manifest.csv", rows, FIELDS)
        kpr = ROOT / "data" / "kpr" / f"x{k}"
        kp_out = kpr / "external_annotation" / KP / "bounding_box_train"
        kp_out.mkdir(parents=True, exist_ok=True)
        (kpr / "masks").mkdir(exist_ok=True)
        # 마스크는 17×9 고정 해상도라 스케일하지 않는다. 근거는 README.
        if not (kpr / "masks" / MASK).exists():
            (kpr / "masks" / MASK).symlink_to(ORIG / "masks" / MASK)
        if not (kpr / "bounding_box_train").exists():
            (kpr / "bounding_box_train").symlink_to(out)
        jsons = sorted((ORIG / "external_annotation" / KP / "bounding_box_train").glob("*.json"))
        with ProcessPoolExecutor(16) as ex:
            list(ex.map(scale_kp, [(j, kp_out / j.name, k) for j in jsons], chunksize=64))
        print(f"x{k}: 이미지 {len(rows)}, 키포인트 {len(jsons)}", flush=True)


if __name__ == "__main__":
    main()
