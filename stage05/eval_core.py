"""경량 평가기의 공통 부분. numpy만 쓴다. 전체 거리 행렬을 만들지 않고 query 블록 단위로 Market 규칙의 CMC·AP를 계산한다."""
import json
from pathlib import Path

import numpy as np

MAX_RANK = 50


def block_metrics(dist, q_pids, g_pids, q_cams, g_cams, max_rank=MAX_RANK):
    """torchreid eval_market1501과 같은 규칙. 같은 ID·같은 카메라를 뺀다. 양성이 없는 query는 valid=False."""
    order = np.argsort(dist, axis=1)
    n = len(q_pids)
    ap = np.full(n, np.nan)
    cmc = np.zeros((n, max_rank), np.float32)
    valid = np.zeros(n, bool)
    for i in range(n):
        ord_i = order[i]
        keep = ~((g_pids[ord_i] == q_pids[i]) & (g_cams[ord_i] == q_cams[i]))
        raw = (g_pids[ord_i][keep] == q_pids[i]).astype(np.int32)
        if not raw.any():
            continue
        c = raw.cumsum()
        hit = (c >= 1).astype(np.float32)
        cmc[i] = hit[:max_rank] if len(hit) >= max_rank else np.pad(hit, (0, max_rank - len(hit)), mode="edge")
        ap[i] = float(((c / np.arange(1, len(c) + 1)) * raw).sum() / raw.sum())
        valid[i] = True
    return ap, cmc, valid


def stream_eval(q_pids, q_cams, g_pids, g_cams, dist_block, block, on_block=None):
    """dist_block(i0, i1) -> (i1-i0)×G 거리. 블록 거리는 바로 버린다."""
    n = len(q_pids)
    aps, cmcs, valids = [], [], []
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d = dist_block(i0, i1)
        a, c, v = block_metrics(d, q_pids[i0:i1], g_pids, q_cams[i0:i1], g_cams)
        if on_block is not None:
            on_block(i0, i1, d)
        aps.append(a)
        cmcs.append(c)
        valids.append(v)
        del d
    ap, cmc, valid = np.concatenate(aps), np.concatenate(cmcs), np.concatenate(valids)
    return {"ap": ap, "cmc": cmc, "valid": valid, "mAP": float(np.nanmean(ap)), "cmc_curve": cmc[valid].mean(0),
            "n_valid": int(valid.sum()), "n_query": n}


def summary(res):
    c = res["cmc_curve"]
    return {"mAP": res["mAP"], "rank1": float(c[0]), "rank5": float(c[4]), "rank10": float(c[9]), "n_valid": res["n_valid"], "n_query": res["n_query"]}


def save_split(d, feat, vis, pid, cam, names):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "feat.npy", feat)
    if vis is not None:
        np.save(d / "vis.npy", vis)
    np.save(d / "pid.npy", pid)
    np.save(d / "cam.npy", cam)
    (d / "names.txt").write_text("\n".join(names))


def load_split(d):
    d = Path(d)
    vis = np.load(d / "vis.npy", mmap_mode="r") if (d / "vis.npy").exists() else None
    return {"feat": np.load(d / "feat.npy", mmap_mode="r"), "vis": vis, "pid": np.load(d / "pid.npy"), "cam": np.load(d / "cam.npy"),
            "names": (d / "names.txt").read_text().split("\n")}


def subset(n_q, n_g, sub_q, sub_g, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_q, sub_q, replace=False)), np.sort(rng.choice(n_g, sub_g, replace=False))


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
