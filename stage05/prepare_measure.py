"""측정용 데이터 준비. 이미지를 열람하지 않는다. 끝난 파일은 건너뛴다.
train: data/bicubic_x4 PNG와 data/kpr/x4 (Stage 0 구조). test (ALLOW_TEST): data/locked/measure/bicubic_x{k}/{query,bounding_box_test} PNG,
data/locked/kpr_annot (공식 주석 zip의 test 항목), data/locked/kpr_root/{orig,x3,x4}/Market-1501-v15.09.15 (공식 로더 구조)."""
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from common import guard
from common.util import ROOT, sha256_file, write_csv
from stage0.prepare_data import FIELDS, KP, MASK, ORIG, SRC, scale_kp, upscale

LOCKED = ROOT / "data" / "locked"
ZIP = ROOT / "data" / "kpr_annotations" / "kpr_market_annotations.zip"
SPLITS = ("query", "bounding_box_test")
EXPECT = {"query": 3368, "bounding_box_test": 19732}
MARKET = "Market-1501-v15.09.15"


def link(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_symlink() and not dst.exists():
        dst.symlink_to(src)


def prepare_train():
    srcs = sorted(SRC.glob("*.jpg"))
    assert len(srcs) == 12936, len(srcs)
    out = ROOT / "data" / "bicubic_x4"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(upscale, [(p, out / (p.stem + ".png"), 4) for p in srcs], chunksize=64))
    write_csv(ROOT / "data" / "bicubic_x4_manifest.csv", rows, FIELDS)
    kpr = ROOT / "data" / "kpr" / "x4"
    kp_out = kpr / "external_annotation" / KP / "bounding_box_train"
    kp_out.mkdir(parents=True, exist_ok=True)
    link(kpr / "masks" / MASK, ORIG / "masks" / MASK)
    link(kpr / "bounding_box_train", out)
    jsons = sorted((ORIG / "external_annotation" / KP / "bounding_box_train").glob("*.json"))
    with ProcessPoolExecutor(12) as ex:
        list(ex.map(scale_kp, [(j, kp_out / j.name, 4) for j in jsons], chunksize=64))
    print(f"train x4: 이미지 {len(rows)}, 키포인트 {len(jsons)}", flush=True)


def prepare_test():
    if os.environ.get("ALLOW_TEST") != "1":
        raise SystemExit("test 준비는 scripts/with_test.sh로 실행")
    os.chmod(ZIP, 0o600)
    annot = LOCKED / "kpr_annot"
    n = 0
    prefixes = [f"{MARKET}/external_annotation/{KP}/{s}/" for s in SPLITS] + [f"{MARKET}/masks/{MASK}/{s}/" for s in SPLITS]
    with zipfile.ZipFile(ZIP) as z:
        for i in z.infolist():
            if i.is_dir() or not any(i.filename.startswith(p) for p in prefixes):
                continue
            dst = annot / Path(*Path(i.filename).parts[1:])
            if not (dst.exists() and dst.stat().st_size == i.file_size):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(i))
            n += 1
    os.chmod(ZIP, 0o000)
    print(f"test 주석 {n}개 추출", flush=True)
    for split in SPLITS:
        srcs = sorted((LOCKED / split).glob("*.jpg"))
        assert len(srcs) == EXPECT[split], (split, len(srcs))
        for k in (3, 4):
            out = LOCKED / "measure" / f"bicubic_x{k}" / split
            out.mkdir(parents=True, exist_ok=True)
            with ProcessPoolExecutor(12) as ex:
                rows = list(ex.map(upscale, [(p, out / (p.stem + ".png"), k) for p in srcs], chunksize=64))
            write_csv(LOCKED / "measure" / f"bicubic_x{k}_{split}_manifest.csv", rows, FIELDS)
            kp_out = LOCKED / "kpr_root" / f"x{k}" / MARKET / "external_annotation" / KP / split
            kp_out.mkdir(parents=True, exist_ok=True)
            jsons = sorted((annot / "external_annotation" / KP / split).glob("*.json"))
            with ProcessPoolExecutor(12) as ex:
                list(ex.map(scale_kp, [(j, kp_out / j.name, k) for j in jsons], chunksize=64))
            print(f"{split} x{k}: 이미지 {len(rows)}, 키포인트 {len(jsons)}", flush=True)
    # 공식 로더 구조. train은 ×k PNG와 스케일한 키포인트, 원본 크기는 원본 그대로.
    for k in (3, 4):
        r = LOCKED / "kpr_root" / f"x{k}" / MARKET
        link(r / "bounding_box_train", ROOT / "data" / f"bicubic_x{k}")
        link(r / "external_annotation" / KP / "bounding_box_train", ROOT / "data" / "kpr" / f"x{k}" / "external_annotation" / KP / "bounding_box_train")
        link(r / "masks" / MASK / "bounding_box_train", ORIG / "masks" / MASK / "bounding_box_train")
        for s in SPLITS:
            link(r / s, LOCKED / "measure" / f"bicubic_x{k}" / s)
            link(r / "masks" / MASK / s, annot / "masks" / MASK / s)
    r = LOCKED / "kpr_root" / "orig" / MARKET
    link(r / "bounding_box_train", ROOT / "data" / "market" / "bounding_box_train")
    link(r / "external_annotation" / KP / "bounding_box_train", ORIG / "external_annotation" / KP / "bounding_box_train")
    link(r / "masks" / MASK / "bounding_box_train", ORIG / "masks" / MASK / "bounding_box_train")
    for s in SPLITS:
        link(r / s, LOCKED / s)
        link(r / "external_annotation" / KP / s, annot / "external_annotation" / KP / s)
        link(r / "masks" / MASK / s, annot / "masks" / MASK / s)


if __name__ == "__main__":
    guard.install()
    prepare_test() if "--test" in sys.argv else prepare_train()
