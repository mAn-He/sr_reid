"""CLIP-ReID 평가. srreid-clipreid env. task: extract / official(공식 do_inference, R1_mAP_eval) / light. 외부 저장소는 수정하지 않는다."""
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
from torch.utils.data import DataLoader  # noqa: E402

from stage05 import eval_core, items  # noqa: E402

CLIP_REPO = ROOT / "third_party" / "CLIP-ReID"
W0 = ROOT / "weights" / "stage0"
PUBLIC = ROOT / "weights" / "stage05" / "public" / "clipreid_vit_market.pth"
sys.path.insert(0, str(CLIP_REPO))


def setup(a):
    from config import cfg
    from model.clip import clip
    from model.make_model_clipreid import make_model
    clip._download = lambda url, root=None: str(W0 / "ViT-B-16.pt")
    with open(CLIP_REPO / "configs" / "person" / "vit_clipreid.yml") as f:
        recipe = type(cfg).load_cfg(f)
    del recipe["DATASETS"]
    cfg.merge_from_other_cfg(recipe)
    h, w = a.h, a.w
    cfg.merge_from_list(["INPUT.SIZE_TRAIN", [h, w], "INPUT.SIZE_TEST", [h, w], "DATASETS.NAMES", "market1501", "TEST.IMS_PER_BATCH", a.batch,
                         "DATALOADER.NUM_WORKERS", a.workers])
    cfg.freeze()
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    if a.public:
        model.load_param(str(PUBLIC))
    return cfg, model


def loader(cfg, a):
    import torchvision.transforms as T
    from datasets.bases import ImageDataset
    from datasets.make_dataloader_clipreid import val_collate_fn
    tf = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    q, g = items.query_gallery(a.cond, a.k, a.sub_q, a.sub_g, a.seed)
    rows = [(p, pid, cam, 1) for p, pid, cam in q + g]
    return DataLoader(ImageDataset(rows, tf), batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=val_collate_fn), q, g


def official(a):
    from processor.processor_clipreid_stage2 import do_inference
    cfg, model = setup(a)
    val, q, g = loader(cfg, a)
    captured = {}

    def prof(frame, event, arg):
        if event == "return" and frame.f_code.co_name == "eval_func":
            captured["ap"] = np.array(frame.f_locals["all_AP"])
            captured["cmc"], captured["mAP"] = arg
    sys.setprofile(prof)
    do_inference(cfg, model, val, len(q))
    sys.setprofile(None)
    if a.record_metrics:
        Path(a.features).mkdir(parents=True, exist_ok=True)
        np.savez(Path(a.features) / "official_metrics.npz", **captured)
        c = captured["cmc"]
        return {"mAP": float(captured["mAP"]), "rank1": float(c[0]), "rank5": float(c[4]), "rank10": float(c[9]), "n_query": len(q), "n_gallery": len(g)}
    return {"n_query": len(q), "n_gallery": len(g), "metrics_recorded": False}


def extract(a):
    cfg, model = setup(a)
    model = model.cuda().eval()
    val, q, g = loader(cfg, a)
    out = []
    with torch.no_grad():
        for img, *_ in val:  # 공식 do_inference와 같게 AMP 없음, SIE 끔
            out.append(model(img.cuda(), cam_label=None, view_label=None).cpu().numpy())
    feats = np.concatenate(out)
    nq = len(q)
    eval_core.save_split(Path(a.features) / "query", feats[:nq], None, np.array([r[1] for r in q]), np.array([r[2] for r in q]), [os.path.basename(r[0]) for r in q])
    eval_core.save_split(Path(a.features) / "gallery", feats[nq:], None, np.array([r[1] for r in g]), np.array([r[2] for r in g]), [os.path.basename(r[0]) for r in g])
    return {"batch": a.batch, "amp": False, "n_query": nq, "n_gallery": len(g), "dim": feats.shape[1], "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def light(a):
    from utils.metrics import euclidean_distance
    Q, G = eval_core.load_split(Path(a.features) / "query"), eval_core.load_split(Path(a.features) / "gallery")
    qf = torch.nn.functional.normalize(torch.from_numpy(np.array(Q["feat"])), dim=1, p=2)
    gf = torch.nn.functional.normalize(torch.from_numpy(np.array(G["feat"])), dim=1, p=2)

    def dist_block(i0, i1):
        return euclidean_distance(qf[i0:i1], gf)  # 공식 거리 함수
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
    ap.add_argument("--cond", default="orig")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--public", type=int, default=1)
    ap.add_argument("--record-metrics", type=int, default=1)
    ap.add_argument("--features", required=True)
    ap.add_argument("--batch", type=int, default=64)
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
