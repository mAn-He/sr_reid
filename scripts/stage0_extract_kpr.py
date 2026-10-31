"""KPR 공식 주석 zip에서 bounding_box_train 항목만 푼다. query, bounding_box_test 항목은 내용을 읽지 않는다."""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "kpr_annotations" / "kpr_market_annotations.zip"
OUT = ROOT / "data" / "kpr" / "orig"
PREFIXES = ["Market-1501-v15.09.15/external_annotation/pifpaf_keypoints_pifpaf_maskrcnn_filtering/bounding_box_train/",
            "Market-1501-v15.09.15/masks/pifpaf_maskrcnn_filtering/bounding_box_train/"]


def main():
    os.chmod(ZIP, 0o600)
    n = 0
    with zipfile.ZipFile(ZIP) as z:
        for i in z.infolist():
            if i.is_dir() or not any(i.filename.startswith(p) for p in PREFIXES):
                continue
            rel = Path(*Path(i.filename).parts[1:])
            dst = OUT / rel
            if not (dst.exists() and dst.stat().st_size == i.file_size):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(i))
            n += 1
    os.chmod(ZIP, 0o000)  # test 주석이 함께 들어 있으므로 다시 잠근다.
    print(f"train 주석 {n}개 추출, zip 잠금", flush=True)


if __name__ == "__main__":
    main()
