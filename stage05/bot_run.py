"""BoT 평가. srreid-core env. baseline: 공식 reid-strong-baseline 코드와 공개 가중치(256×128). inhouse: 직접 구현 BoT(×k 입력, RAM 측정용).
task: extract(특징 저장) / official(공식 eval_func 전체 행렬) / light(블록 스트리밍)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
import torchvision.transforms as T  # noqa: E402

from stage05 import eval_core, items  # noqa: E402

BOT = ROOT / "third_party" / "reid-strong-baseline"
WEIGHTS = ROOT / "weights" / "stage05" / "public" / "market_resnet50_model_120_rank1_945.pth"


class Eval(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i][0]) as im:
            return self.tf(im.convert("RGB"))


def official_model():
    sys.path.insert(0, str(BOT))
    from modeling.baseline import Baseline
    m = Baseline(751, 1, "", "bnneck", "after", "resnet50", "self")
    st = torch.load(WEIGHTS, map_location="cpu")
    if not isinstance(st, dict):  # 공개 파일이 모델 객체 통째로 저장된 경우
        st = st.state_dict()
    missing = m.load_state_dict(st, strict=True)
    print("가중치 로드", missing, flush=True)
    return m


def inhouse_model():
    from reid.model import BoT
    return BoT(751, str(ROOT / "weights" / "resnet50-0676ba61.pth"))


def extract(a):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if a.arch == "official":
        model = official_model()
        tf = T.Compose([T.Resize([256, 128]), T.ToTensor(), T.Normalize(mean, std)])  # 공식 data/transforms/build.py 평가 변환
    else:
        model = inhouse_model()
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    model = model.cuda().eval()
    q, g = items.query_gallery(a.cond, a.k, a.sub_q, a.sub_g, a.seed)
    for split, rows in (("query", q), ("gallery", g)):
        loader = DataLoader(Eval(rows, tf), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
        out = []
        with torch.no_grad():
            for x in loader:
                out.append(model(x.cuda(non_blocking=True)).cpu().numpy())
        eval_core.save_split(Path(a.features) / split, np.concatenate(out), None, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                             [os.path.basename(r[0]) for r in rows])
    return {"batch": a.batch, "amp": False, "n_query": len(q), "n_gallery": len(g), "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def official(a):
    """공식 R1_mAP와 같게: L2 정규화 후 제곱 유클리드 전체 행렬 → eval_func. query별 AP는 함수가 끝날 때 지역 변수에서 읽는다."""
    sys.path.insert(0, str(BOT))
    from data.datasets.eval_reid import eval_func
    Q, G = eval_core.load_split(Path(a.features) / "query"), eval_core.load_split(Path(a.features) / "gallery")
    qf = torch.nn.functional.normalize(torch.from_numpy(np.array(Q["feat"])), dim=1)
    gf = torch.nn.functional.normalize(torch.from_numpy(np.array(G["feat"])), dim=1)
    m, n = qf.shape[0], gf.shape[0]
    dist = torch.pow(qf, 2).sum(1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = dist - 2 * qf @ gf.t()
    captured = {}

    def prof(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "eval_func":
            captured["ap"] = np.array(frame.f_locals["all_AP"])
    sys.setprofile(prof)
    cmc, mAP = eval_func(dist.numpy(), Q["pid"], G["pid"], Q["cam"], G["cam"])
    sys.setprofile(None)
    np.savez(Path(a.features) / "official_metrics.npz", ap=captured["ap"], cmc=cmc, mAP=mAP)
    return {"mAP": float(mAP), "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9])}


def light(a):
    Q, G = eval_core.load_split(Path(a.features) / "query"), eval_core.load_split(Path(a.features) / "gallery")
    gf = torch.nn.functional.normalize(torch.from_numpy(np.array(G["feat"])), dim=1).cuda()
    qf = torch.nn.functional.normalize(torch.from_numpy(np.array(Q["feat"])), dim=1).cuda()

    def dist_block(i0, i1):
        with torch.no_grad():
            return (2 - 2 * qf[i0:i1] @ gf.T).cpu().numpy()
    out = {}
    for block in a.blocks:
        t0 = time.perf_counter()
        res = eval_core.stream_eval(Q["pid"], Q["cam"], G["pid"], G["cam"], dist_block, block)
        out[str(block)] = {**eval_core.summary(res), "seconds": time.perf_counter() - t0}
        np.savez(Path(a.features) / f"light_b{block}.npz", ap=res["ap"], cmc=res["cmc"], valid=res["valid"], cmc_curve=res["cmc_curve"], mAP=res["mAP"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["extract", "official", "light"])
    ap.add_argument("--arch", default="official")
    ap.add_argument("--cond", default="orig")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--features", required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sub-q", type=int, default=0)
    ap.add_argument("--sub-g", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    res = {"extract": extract, "official": official, "light": light}[a.task](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"task": a.task, "args": vars(a), "seconds": time.perf_counter() - t0, "result": res}, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False)[:800], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
