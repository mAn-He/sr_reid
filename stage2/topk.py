"""Stage 2 사후 분석 3 보조: 고른 query들의 top-5 gallery(모델별 공식 거리, Market 규칙으로 같은 신원·같은 카메라 제외). 모델 env에서 실행.
python stage2/topk.py <model> <unit_dir> <names.txt> <out.json>  (ALLOW_TEST=1)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path[:] = [p for p in sys.path if not p.endswith("/stage2")]
from common import guard  # noqa: E402

guard.install()
import importlib.util  # noqa: E402

from stage05 import eval_core  # noqa: E402

spec = importlib.util.spec_from_file_location("s2eval", ROOT / "stage2" / "evaluate.py")
EV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EV)


def main():
    model, ud, nf, out = sys.argv[1:5]
    ud = Path(ud)
    Q, G = eval_core.load_split(ud / "feats" / "query"), eval_core.load_split(ud / "feats" / "gallery")
    meta = json.loads((ud / "feats" / "query" / "extract_meta.json").read_text())
    want = [l.strip() for l in open(nf) if l.strip()]
    pos = {n: i for i, n in enumerate(Q["names"])}
    idx = np.array([pos[n] for n in want])
    sub = {"feat": np.array(Q["feat"])[idx], "vis": None if Q["vis"] is None else np.array(Q["vis"])[idx], "pid": Q["pid"][idx], "cam": Q["cam"][idx], "names": want}
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dist_block, _ = EV.make_dist(model, sub, G, dev, meta)
    d = dist_block(0, len(idx))
    res = {}
    for i, n in enumerate(want):
        di = d[i].copy()
        di[(G["pid"] == sub["pid"][i]) & (G["cam"] == sub["cam"][i])] = np.inf  # Market 규칙
        o = np.argsort(di, kind="stable")[:5]
        res[n] = [{"name": G["names"][j], "pid": int(G["pid"][j]), "correct": bool(G["pid"][j] == sub["pid"][i]), "dist": float(di[j])} for j in o]
    Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
