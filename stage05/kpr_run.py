"""KPR, KPR+SOLIDER 평가. srreid-kpr env. 외부 저장소는 수정하지 않고 런타임 대체만 쓴다.
task: official(공식 engine.run, visrank 설정대로, 지표 캡처) / extract(경량: 임베딩·가시성·pid·camid·파일명만 저장) / light(블록 스트리밍)."""
import argparse
import glob as _glob
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage05 import eval_core, items  # noqa: E402

KPR_REPO = ROOT / "third_party" / "keypoint_promptable_reidentification"
sys.path.insert(0, str(KPR_REPO))
PUBLIC = ROOT / "weights" / "stage05" / "public"
W_NAME = {"kpr": "kpr_market_IN_89.58_95.87_41898449.pth.tar", "kpr_solider": "kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar"}
DATA_NAME = "srreid_market"


def register(a):
    import torchreid
    from torchreid.data.datasets.image import market1501 as mod
    from torchreid.data.datasets.image.market1501 import Market1501

    mod.glob = SimpleNamespace(glob=lambda pat: sorted(f for f in _glob.glob(pat.replace("*.jpg", "*.*")) if f.endswith((".jpg", ".png", ".webp"))))  # 공식 process_dir을 그대로 쓰되 png, webp도 읽게 한다.
    orig = Market1501.process_dir

    def process_dir(self, dir_path, relabel=False):
        rows = orig(self, dir_path, relabel=relabel)
        rows = sorted(rows, key=lambda r: os.path.basename(r["img_path"]))
        split = os.path.basename(dir_path)
        if split == "query" and a.q_has_match:  # 공식 visrank는 양성 없는 query에서 IndexError를 낸다. 부분집합 측정에서만 쓴다.
            g = sorted(orig(self, str(Path(dir_path).parent / "bounding_box_test"), relabel=False), key=lambda r: os.path.basename(r["img_path"]))
            g = [g[i] for i in items.subset_indices(len(g), a.sub_g, a.seed + 1)] if a.sub_g else g
            have = {}
            for r in g:
                have.setdefault(r["pid"], set()).add(r["camid"])
            rows = [r for r in rows if have.get(r["pid"], set()) - {r["camid"]}]
        if split == "query" and a.sub_q:
            rows = [rows[i] for i in items.subset_indices(len(rows), a.sub_q, a.seed)]
        if split == "bounding_box_test" and a.sub_g:
            rows = [rows[i] for i in items.subset_indices(len(rows), a.sub_g, a.seed + 1)]
        return rows

    class Market(Market1501):
        pass
    Market.process_dir = process_dir
    torchreid.data.register_image_dataset(DATA_NAME, Market)


def install_cache(a):
    """E.3, E.4: 디코딩·JSON·npy 읽기를 memmap 조회로 대체. 공식 transforms는 그대로 지나간다."""
    from torchreid.data.datasets import dataset as mod
    from stage05 import cache as C
    k = a.k
    state = {}

    def get(kind, split):
        key = (kind, split)
        if key not in state:
            if kind == "img":
                state[key] = C.load(a.cond, k, split)
            elif kind == "mask":
                d = ROOT / "data" / "kpr_cache" / "masks" / split
                arr = np.load(d / f"masks_{a.mask_dtype}.npy", mmap_mode="r")
                state[key] = (arr, {n.split(".")[0]: i for i, n in enumerate((d / "names.txt").read_text().split("\n"))})  # 공식 주석에 x.jpg.npy 24개가 섞여 있다
            else:
                z = np.load(ROOT / "data" / "kpr_cache" / f"x{k}" / split / "keypoints.npz")
                state[key] = (z["kp"], z["offsets"], {n: i for i, n in enumerate(z["names"].tolist())})
        return state[key]

    def split_of(path):
        return os.path.basename(os.path.dirname(path))

    def read_image(path):
        mm, idx = get("img", split_of(path))
        return np.array(mm[idx[Path(path).name.split(".")[0]]])

    def read_masks(path):
        arr, idx = get("mask", split_of(path))
        m = arr[idx[Path(path).name.split(".")[0]]]
        m = m.astype(np.float32) / 255 if a.mask_dtype == "u8" else np.array(m)
        return np.transpose(m, (1, 2, 0))

    def read_keypoints(path, bbox_ltwh=None):
        kp, off, idx = get("kp", split_of(path))
        i = idx[Path(path).name.split(".")[0]]
        return kp[off[i]:off[i + 1]].copy()

    mod.read_image, mod.read_masks, mod.read_keypoints = read_image, read_masks, read_keypoints


def build(a):
    from torchreid.scripts.builder import build_config, build_torchreid_model_engine
    from torchreid.data.datasets import dataset as dmod
    register(a)
    if a.no_occluders:  # 평가 변환에는 BIPO가 없다. 데이터셋마다 occluder를 메모리에 올리는 것만 막는다.
        dmod.load_human_occluders = lambda *p, **k: {}
    pre = "solider" if a.model == "kpr_solider" else "imagenet"
    yaml_path = KPR_REPO / "configs" / "kpr" / pre / "kpr_market_test.yaml"
    save = Path(a.save_root) / f"{a.model}_{a.task}_{os.getpid()}"
    opts = ["data.height", a.h, "data.width", a.w, "test.batch_size", a.batch, "data.workers", a.workers, "test.visrank", bool(a.visrank),
            "model.load_weights", str(PUBLIC / W_NAME[a.model]), "model.backbone_pretrained_path", str(ROOT / "weights" / "stage0") + "/"]
    args = SimpleNamespace(root=str(ROOT / "data" / "locked" / "kpr_root" / ("orig" if a.k == 1 else f"x{a.k}")), save_dir=str(save),
                           inference_enabled=False, sources=[DATA_NAME], targets=[DATA_NAME], transforms=None, job_id="j", opts=opts)
    cfg = build_config(args=args, config_path=str(yaml_path))
    engine, model = build_torchreid_model_engine(cfg)
    if a.no_pin:  # 핀 메모리 캐시가 처리한 이미지 수에 비례해 메인 RSS에 남는다(runs/stage05/diag). 계산 결과와 무관.
        for name in ("query", "gallery"):
            engine.test_loader[DATA_NAME][name].pin_memory = False
    return cfg, engine, model, save


def official(a):
    """공식 경로. 지표는 evaluate_rank를 감싸 읽는다. visrank 이미지는 data/locked 아래에 쓰이고 측정 후 지운다."""
    from torchreid import metrics
    from torchreid.scripts.default_config import engine_run_kwargs
    cfg, engine, model, save = build(a)
    captured = {}
    orig = metrics.evaluate_rank

    def wrap(*x, **kw):
        out = orig(*x, **kw)
        captured.update({k: out[k] for k in ("cmc", "mAP", "all_AP", "all_cmc") if k in out})
        return out
    metrics.evaluate_rank = wrap
    engine.run(**engine_run_kwargs(cfg))
    res = {"visrank": bool(a.visrank), "batch": a.batch, "n_query": len(engine.datamanager.test_dataset[DATA_NAME]["query"]), "n_gallery": len(engine.datamanager.test_dataset[DATA_NAME]["gallery"])}
    if a.record_metrics and captured:
        Path(a.features).mkdir(parents=True, exist_ok=True)
        np.savez(Path(a.features) / "official_metrics.npz", ap=captured["all_AP"], cmc=captured["cmc"], mAP=captured["mAP"])
        c = captured["cmc"]
        res.update(mAP=float(captured["mAP"]), rank1=float(c[0]), rank5=float(c[4]), rank10=float(c[9]))
    os.system(f"rm -rf '{save}'")
    return res


def extract(a):
    import resource
    from torchreid.utils.tools import extract_test_embeddings
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)  # 512×256 b32에서 Too many open files(텐서 공유 fd). 결과와 무관.
    if a.cached:
        install_cache(a)
    cfg, engine, model, save = build(a)
    engine.set_model_mode("eval")
    t = {}
    for name in ("query", "gallery"):
        loader = engine.test_loader[DATA_NAME][name]
        feats, vis, pids, cams, names = [], [], [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():  # 공식 _feature_extraction과 같게 autocast 없음
            for data in loader:
                imgs, tm, pm, kp, pid, cam = engine.parse_data_for_eval(data)
                imgs = imgs.cuda()
                tm = tm.cuda() if tm is not None else None
                pm = pm.cuda() if pm is not None else None
                out = model(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
                f, v, _, _ = extract_test_embeddings(out, cfg.model.kpr.test_embeddings)
                feats.append(f.cpu().numpy().astype(np.float32))
                vis.append(v.cpu().numpy())
                pids.append(pid.numpy())
                cams.append(cam.numpy())
                names += [os.path.basename(p) for p in data["img_path"]]
                del data, imgs, tm, pm, kp, out, f, v  # 나머지는 즉시 폐기
        eval_core.save_split(Path(a.features) / name, np.concatenate(feats), np.concatenate(vis), np.concatenate(pids), np.concatenate(cams), names)
        t[name] = {"seconds": time.perf_counter() - t0, "n": len(names), "feat_shape": list(np.concatenate(feats).shape) if len(feats) < 2 else None}
    os.system(f"rm -rf '{save}'")
    return {"batch": a.batch, "amp": False, "timing": t, "visibility_dtype": str(vis[0].dtype), "test_embeddings": list(cfg.model.kpr.test_embeddings),
            "normalize_feature": bool(cfg.test.normalize_feature), "dist_metric": cfg.test.dist_metric, "combine": cfg.test.part_based.dist_combine_strat,
            "pairwise_batch": cfg.test.batch_size_pairwise_dist_matrix, "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def light(a):
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features as fn
    Q, G = eval_core.load_split(Path(a.features) / "query"), eval_core.load_split(Path(a.features) / "gallery")
    qf = torch.nn.functional.normalize(torch.from_numpy(np.array(Q["feat"])), p=2, dim=-1)
    gf = torch.nn.functional.normalize(torch.from_numpy(np.array(G["feat"])), p=2, dim=-1)
    qv, gv = torch.from_numpy(np.array(Q["vis"])), torch.from_numpy(np.array(G["vis"]))
    if a.force_invalid:  # C.4: 실제 특징에는 무효 쌍이 없다. 가시성을 일부 끄면 채움값 경로가 돌아 블록 간 불변을 확인할 수 있다. 지표는 기준선에 쓰지 않는다.
        rng = np.random.default_rng(a.seed)
        qv[torch.from_numpy(rng.random(len(qv)) < a.force_invalid)] = False
        gv[torch.from_numpy(rng.random(len(gv)) < a.force_invalid)] = False
    out = {}
    for block in a.blocks:
        stats = {"blocks": 0, "rows_with_invalid": 0, "invalid_pairs": 0, "rows_fill_not_above_valid": 0, "min_fill_margin": None, "fills": []}

        def dist_block(i0, i1):
            with torch.no_grad():
                d = fn(qf[i0:i1], gf, qv[i0:i1], gv, a.combine, a.pairwise, True, a.metric, use_logger=False)[0]
            return d.float().numpy() if torch.is_tensor(d) else np.asarray(d, np.float32)

        def on_block(i0, i1, d):
            stats["blocks"] += 1
            q_vis = np.asarray(qv[i0:i1])
            g_vis = np.asarray(gv)
            valid = (q_vis.astype(np.float32) @ g_vis.T.astype(np.float32)) > 0
            for r in range(d.shape[0]):
                inv = ~valid[r]
                if inv.any():
                    stats["rows_with_invalid"] += 1
                    stats["invalid_pairs"] += int(inv.sum())
                    fill = float(d[r][inv].min())
                    top = float(d[r][valid[r]].max()) if valid[r].any() else -np.inf
                    margin = fill - top
                    stats["min_fill_margin"] = margin if stats["min_fill_margin"] is None else min(stats["min_fill_margin"], margin)
                    stats["rows_fill_not_above_valid"] += int(margin <= 0)
                    stats["fills"].append(fill)
        t0 = time.perf_counter()
        res = eval_core.stream_eval(Q["pid"], Q["cam"], G["pid"], G["cam"], dist_block, block, on_block)
        fills = stats.pop("fills")
        stats["fill_min"], stats["fill_max"] = (min(fills), max(fills)) if fills else (None, None)
        out[str(block)] = {**eval_core.summary(res), "seconds": time.perf_counter() - t0, "fill_check": stats}
        np.savez(Path(a.features) / f"light{'_forced' if a.force_invalid else ''}_b{block}.npz", ap=res["ap"], cmc=res["cmc"], valid=res["valid"], cmc_curve=res["cmc_curve"], mAP=res["mAP"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["official", "extract", "light"])
    ap.add_argument("--model", default="kpr")
    ap.add_argument("--cond", default="orig")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--features", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--visrank", type=int, default=1)
    ap.add_argument("--record-metrics", type=int, default=1)
    ap.add_argument("--cached", type=int, default=1)
    ap.add_argument("--mask-dtype", default="f32")
    ap.add_argument("--force-invalid", type=float, default=0.0)
    ap.add_argument("--no-pin", type=int, default=0)
    ap.add_argument("--no-occluders", type=int, default=0)
    ap.add_argument("--q-has-match", type=int, default=0)
    ap.add_argument("--sub-q", type=int, default=0)
    ap.add_argument("--sub-g", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--combine", default="mean")
    ap.add_argument("--pairwise", type=int, default=500)
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--save-root", default=str(ROOT / "data" / "locked" / "measure" / "kpr_tmp"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    res = {"official": official, "extract": extract, "light": light}[a.task](a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"task": a.task, "args": vars(a), "seconds": time.perf_counter() - t0, "result": res}, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False)[:900], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
