"""cosine 거리, Rank-1, AP (Market 규칙: 같은 ID·같은 카메라 gallery 제외)."""
import numpy as np


def evaluate_query(q_feat, q_pid, q_cam, g_feats, g_pids, g_cams):
    dist = 1 - g_feats @ q_feat
    order = np.argsort(dist, kind="stable")
    keep = ~((g_pids[order] == q_pid) & (g_cams[order] == q_cam))
    match = (g_pids[order] == q_pid)[keep]
    n_pos = int(match.sum())
    if n_pos == 0:
        raise ValueError("유효 gallery가 없는 query")
    r1 = float(match[0])
    precision = np.cumsum(match) / (np.arange(len(match)) + 1)
    ap = float((precision * match).sum() / n_pos)
    return ap, r1


def evaluate(q_feats, q_pids, q_cams, valid, g_feats, g_pids, g_cams):
    """출력이 없거나 크기가 틀린 query(valid=False)는 빼지 않고 AP=0, R1=0으로 계산한다."""
    ap, r1 = np.zeros(len(q_pids)), np.zeros(len(q_pids))
    for i in range(len(q_pids)):
        if valid[i]:
            ap[i], r1[i] = evaluate_query(q_feats[i], q_pids[i], q_cams[i], g_feats, g_pids, g_cams)
    return ap, r1
