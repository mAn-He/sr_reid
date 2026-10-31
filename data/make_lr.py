"""val query ×4 축소, Path 1 입력, 학습 캐시 생성. --verify는 LR 생성 게이트.

리사이즈 경로: LR 32×16 → 128×64(bicubic) → 256×128. Path 1 학습/평가와 "bicubic" SR 방법이 같은 경로를 쓴다.
크기 표기는 H×W이고 PIL.resize에는 (W, H)로 넘긴다.
"""
import argparse
import hashlib
import sys

import numpy as np
from PIL import Image

from common.util import EXIT_FATAL, load_cfg, read_csv, sha256_file, write_csv


def to_lr(img, cfg):
    h, w = cfg["lr"]["lr_size"]
    return img.resize((w, h), Image.BICUBIC, reducing_gap=None)


def to_hr_size(img, cfg):
    h, w = cfg["lr"]["hr_size"]
    return img.resize((w, h), Image.BICUBIC)


def to_input(img, cfg):
    h, w = cfg["reid"]["input_size"]
    return img.resize((w, h), Image.BICUBIC)


def make_query(cfg, rows):
    lr_dir = cfg["_processed"] / "val_query_lr"
    up_dir = cfg["_processed"] / "val_query_lrup"
    lr_dir.mkdir(exist_ok=True)
    up_dir.mkdir(exist_ok=True)
    out = []
    for r in rows:
        if r["role"] != "query":
            continue
        lr = to_lr(Image.open(r["path"]).convert("RGB"), cfg)
        lr_p, up_p = lr_dir / f"{r['name']}.png", up_dir / f"{r['name']}.png"
        lr.save(lr_p)
        to_hr_size(lr, cfg).save(up_p)
        out.append({"name": r["name"], "hr_sha256": r["sha256"], "lr_sha256": sha256_file(lr_p),
                    "lrup_sha256": sha256_file(up_p)})
    write_csv(cfg["_manifests"] / "val_query_lr.csv", out, ["name", "hr_sha256", "lr_sha256", "lrup_sha256"])
    print(f"val query LR {len(out)}장", flush=True)


def make_train_cache(cfg, rows):
    train = [r for r in rows if r["role"] == "train"]
    h, w = cfg["reid"]["input_size"]
    hr = np.zeros((len(train), h, w, 3), np.uint8)
    lr = np.zeros_like(hr)
    out = []
    for n, r in enumerate(train):
        img = Image.open(r["path"]).convert("RGB")
        hr[n] = np.asarray(to_input(img, cfg))
        lr[n] = np.asarray(to_input(to_hr_size(to_lr(img, cfg), cfg), cfg))
        out.append({"name": r["name"], "pid": r["pid"], "hr_sha256": r["sha256"],
                    "hr256_array_sha256": hashlib.sha256(hr[n].tobytes()).hexdigest(),
                    "lr256_array_sha256": hashlib.sha256(lr[n].tobytes()).hexdigest()})
    np.save(cfg["_processed"] / "train_hr256.npy", hr)
    np.save(cfg["_processed"] / "train_lr256.npy", lr)
    write_csv(cfg["_manifests"] / "train_cache.csv", out, list(out[0]))
    print(f"학습 캐시 {len(out)}장", flush=True)


def verify(cfg, rows):
    """게이트: 개수, 32×16, sha256, PIL BICUBIC 재현. 실패하면 전체 중단."""
    errors = []
    if cfg["lr"]["filter"] != "PIL.BICUBIC" or cfg["lr"]["reducing_gap"] is not None:
        errors.append(f"config 필터가 지시와 다름: {cfg['lr']}")
    query = [r for r in rows if r["role"] == "query"]
    m = {r["name"]: r for r in read_csv(cfg["_manifests"] / "val_query_lr.csv")}
    lr_dir = cfg["_processed"] / "val_query_lr"
    files = sorted(lr_dir.glob("*.png"))
    if not query or len(files) != len(query) or len(m) != len(query):
        errors.append(f"개수 불일치: query {len(query)}, 파일 {len(files)}, manifest {len(m)}")
    h, w = cfg["lr"]["lr_size"]
    lanczos_differs = False
    for r in query:
        p = lr_dir / f"{r['name']}.png"
        if not p.exists() or r["name"] not in m:
            errors.append(f"누락: {r['name']}")
            continue
        got = Image.open(p)
        if got.size != (w, h):
            errors.append(f"크기 오류 {r['name']}: {got.size}")
        if sha256_file(p) != m[r["name"]]["lr_sha256"]:
            errors.append(f"sha256 불일치: {r['name']}")
        src = Image.open(r["path"]).convert("RGB")
        if sha256_file(r["path"]) != r["sha256"]:
            errors.append(f"원본 sha256 불일치: {r['name']}")
        if np.asarray(got).tobytes() != np.asarray(to_lr(src, cfg)).tobytes():
            errors.append(f"PIL BICUBIC 재현 실패: {r['name']}")
        other = np.asarray(src.resize((w, h), Image.LANCZOS, reducing_gap=None))
        lanczos_differs = lanczos_differs or other.tobytes() != np.asarray(got).tobytes()
    if query and not lanczos_differs:
        errors.append("검증 무력: LANCZOS 결과와 구분되지 않음")
    if errors:
        print("LR 게이트 실패:", *errors[:30], sep="\n  ", flush=True)
        sys.exit(EXIT_FATAL)
    print(f"LR 게이트 통과: {len(query)}장, {h}x{w}, sha256 일치, PIL BICUBIC 재현", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    rows = read_csv(cfg["_manifests"] / "split.csv")
    if args.verify:
        verify(cfg, rows)
    else:
        make_query(cfg, rows)
        make_train_cache(cfg, rows)


if __name__ == "__main__":
    main()
