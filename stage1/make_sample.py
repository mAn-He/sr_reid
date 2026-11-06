"""Stage 1 표본: train 751명에서 신원마다 3장(seed 42). 가능하면 서로 다른 카메라에서 고른다. 3장 미만 신원은 전부 쓴다.
알고리즘(고정): 신원을 번호순으로 돌며 하나의 np.random.default_rng(42)를 이어 쓴다.
 1) 신원의 카메라 목록을 섞고 앞에서부터 최대 3개 카메라를 고른다. 각 카메라에서 이미지 1장을 무작위로 뽑는다.
 2) 카메라가 3개 미만이면 남은 이미지에서 무작위로 채운다.
출력: configs/stage1_sample.csv (name, pid, cam, seq, frame, bbox). test 분할은 읽지 않는다."""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "market" / "bounding_box_train"
OUT = ROOT / "configs" / "stage1_sample.csv"
PAT = re.compile(r"^(\d{4})_c(\d)s(\d)_(\d{6})_(\d{2})")
PER_ID, SEED = 3, 42


def main():
    by_id = defaultdict(list)
    for p in sorted(SRC.glob("*.jpg")):
        m = PAT.match(p.name)
        pid, cam, seq, frame, bb = m.groups()
        by_id[int(pid)].append({"name": p.name, "pid": int(pid), "cam": int(cam), "seq": int(seq), "frame": int(frame), "bbox": int(bb)})
    rng = np.random.default_rng(SEED)
    rows = []
    for pid in sorted(by_id):
        imgs = by_id[pid]
        if len(imgs) <= PER_ID:
            rows += imgs
            continue
        cams = sorted({r["cam"] for r in imgs})
        cams = [cams[i] for i in rng.permutation(len(cams))][:PER_ID]
        pick = []
        for c in cams:
            cand = [r for r in imgs if r["cam"] == c]
            pick.append(cand[rng.integers(len(cand))])
        rest = [r for r in imgs if r not in pick]
        while len(pick) < PER_ID:
            pick.append(rest.pop(int(rng.integers(len(rest)))))
        rows += sorted(pick, key=lambda r: r["name"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, ["name", "pid", "cam", "seq", "frame", "bbox"])
        w.writeheader()
        w.writerows(rows)
    per = Counter(r["pid"] for r in rows)
    ncam = Counter(len({r["cam"] for r in rows if r["pid"] == p}) for p in per)
    print(f"이미지 {len(rows)}장, 신원 {len(per)}명 (train 전체 신원 {len(by_id)}명, 전체 {sum(map(len, by_id.values()))}장)")
    print("신원당 장수 분포:", dict(sorted(Counter(per.values()).items())))
    print("신원당 서로 다른 카메라 수:", dict(sorted(ncam.items())))
    print("카메라 분포(표본):", dict(sorted(Counter(r["cam"] for r in rows).items())))
    allc = Counter(r["cam"] for v in by_id.values() for r in v)
    print("카메라 분포(train 전체):", dict(sorted(allc.items())))
    print("신원별 가용 카메라 수(train 전체):", dict(sorted(Counter(len({r['cam'] for r in v}) for v in by_id.values()).items())))


if __name__ == "__main__":
    sys.exit(main())
