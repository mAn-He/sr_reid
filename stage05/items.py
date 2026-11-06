"""공식 로더와 같은 규칙의 파일 목록. pid −1(junk)은 뺀다. 표준 라이브러리와 numpy만 쓴다."""
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"([-\d]+)_c(\d)")
N_OFFICIAL = {"query": 3368, "bounding_box_test": 15913}  # pid −1 제외 후 수. 파일 수는 3368, 19732.


def locked_dir(cond, k, split):
    if cond == "orig":
        return ROOT / "data" / "locked" / split
    if cond == "bicubic":
        return ROOT / "data" / "locked" / "measure" / f"bicubic_x{k}" / split
    return ROOT / "data" / cond / f"x{k}" / split


def list_split(d, drop_junk=True):
    """(경로, pid, camid). camid는 0부터."""
    out = []
    for p in sorted(os.listdir(d)):
        m = PATTERN.search(p)
        if not m or not p.endswith((".jpg", ".png", ".webp")):
            continue
        pid, cam = int(m.group(1)), int(m.group(2)) - 1
        if drop_junk and pid == -1:
            continue
        out.append((os.path.join(str(d), p), pid, cam))
    return out


def subset_indices(n, m, seed):
    if not m or m >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, m, replace=False))


def query_gallery(cond, k, sub_q=0, sub_g=0, seed=0):
    q = list_split(locked_dir(cond, k, "query"))
    g = list_split(locked_dir(cond, k, "bounding_box_test"))
    qi, gi = subset_indices(len(q), sub_q, seed), subset_indices(len(g), sub_g, seed + 1)
    return [q[i] for i in qi], [g[i] for i in gi]
