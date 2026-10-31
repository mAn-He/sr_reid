"""train 검증 후 train651 / val100 manifest를 만든다. 공식 test split은 읽지 않는다."""
import argparse
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

from common.util import EXIT_FATAL, load_cfg, sha256_file, write_csv

NAME = re.compile(r"^(\d{4})_c(\d)s\d_\d+_\d+\.jpg$")
FIELDS = ["split", "role", "name", "path", "pid", "camid", "sha256"]


def scan(cfg):
    train_dir = cfg["_market"] / cfg["split"]["train_dir"]
    items, bad = [], []
    for p in sorted(train_dir.iterdir()):
        m = NAME.match(p.name)
        if m is None:
            if p.suffix.lower() == ".jpg":
                bad.append(p.name)
            continue
        items.append({"name": p.stem, "path": str(p), "pid": int(m[1]), "camid": int(m[2])})
    return items, bad


def verify(cfg, items, bad):
    H, W = cfg["lr"]["hr_size"]
    n_ids = len({i["pid"] for i in items})
    errors = []
    if bad:
        errors.append(f"파일명 규칙 위반 {len(bad)}개: {bad[:5]}")
    if len(items) != cfg["split"]["expected_train_images"]:
        errors.append(f"이미지 수 {len(items)} != {cfg['split']['expected_train_images']}")
    if n_ids != cfg["split"]["expected_train_ids"]:
        errors.append(f"ID 수 {n_ids} != {cfg['split']['expected_train_ids']}")
    wrong = [i["name"] for i in items if Image.open(i["path"]).size != (W, H)]
    if wrong:
        errors.append(f"크기가 {H}x{W}이 아닌 이미지 {len(wrong)}개: {wrong[:5]}")
    return errors


def split(cfg, items):
    rng = np.random.RandomState(cfg["split"]["split_seed"])
    ids = sorted({i["pid"] for i in items})
    val_ids = set(rng.choice(ids, cfg["split"]["n_val_ids"], replace=False).tolist())
    by_id_cam = defaultdict(list)
    for i in items:
        by_id_cam[(i["pid"], i["camid"])].append(i)

    query_names = set()
    for key in sorted(by_id_cam):
        if key[0] in val_ids:
            group = by_id_cam[key]
            query_names.add(group[rng.randint(len(group))]["name"])

    for i in items:
        if i["pid"] not in val_ids:
            i.update(split="train651", role="train")
        else:
            i.update(split="val100", role="query" if i["name"] in query_names else "gallery")

    # 다른 카메라 gallery가 없는 query는 다시 뽑지 않고 제외하며 manifest에 남긴다.
    gallery_cams = defaultdict(set)
    for i in items:
        if i["role"] == "gallery":
            gallery_cams[i["pid"]].add(i["camid"])
    for i in items:
        if i["role"] == "query" and not (gallery_cams[i["pid"]] - {i["camid"]}):
            i["role"] = "excluded_query"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    items, bad = scan(cfg)
    errors = verify(cfg, items, bad)
    if errors:
        print("train 검증 실패:", *errors, sep="\n  ", flush=True)
        sys.exit(EXIT_FATAL)
    print(f"train 검증 통과: {len(items)}장, {len({i['pid'] for i in items})} ID", flush=True)
    items = split(cfg, items)
    for i in items:
        i["sha256"] = sha256_file(i["path"])
    write_csv(cfg["_manifests"] / "split.csv", items, FIELDS)
    roles = defaultdict(int)
    for i in items:
        roles[i["role"]] += 1
    print(dict(roles), flush=True)


if __name__ == "__main__":
    main()
