"""KPR 경량 추출 메인 프로세스 RSS 원인 분리. 지표 산출 없음, 수치 기록만."""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv, argv = sys.argv[:1], sys.argv[1:]
from stage05 import kpr_run  # noqa: E402
import torch  # noqa: E402


def smaps():
    out = {}
    for line in open("/proc/self/smaps_rollup"):
        p = line.split()
        if p[0] in ("Rss:", "Pss:", "Anonymous:", "Pss_Anon:", "Pss_File:", "Pss_Shmem:"):
            out[p[0][:-1]] = round(int(p[1]) / 2**20, 2)
    return out


def alive():
    n, b = 0, 0
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.device.type == "cpu":
                n += 1
                b += o.numel() * o.element_size()
        except Exception:
            pass
    return {"cpu_tensors": n, "cpu_gib": round(b / 2**30, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-occluders", type=int, default=0)
    ap.add_argument("--no-pin", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sub-q", type=int, default=50)
    ap.add_argument("--sub-g", type=int, default=2500)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--out", required=True)
    x = ap.parse_args(argv)
    a = SimpleNamespace(task="diag", model="kpr", cond="bicubic" if x.k > 1 else "orig", k=x.k, h=x.h, w=x.w, batch=x.batch, workers=x.workers, visrank=0, mask_dtype="u8", sub_q=x.sub_q, sub_g=x.sub_g, seed=0, save_root=str(ROOT / "data" / "locked" / "measure" / "kpr_tmp"))
    kpr_run.install_cache(a)
    from torchreid.data.datasets import dataset as dmod
    if x.no_occluders:
        dmod.load_human_occluders = lambda *p, **k: {}
    rows = []

    def mark(tag, full=False):
        r = {"tag": tag, "t": round(time.time() - t0, 1), **smaps()}
        if full:
            r.update(alive())
        rows.append(r)
        print(json.dumps(r), flush=True)
    t0 = time.time()
    mark("start")
    from torchreid.utils.tools import extract_test_embeddings
    cfg, engine, model, save = kpr_run.build(a)
    mark("built", True)
    engine.set_model_mode("eval")
    for name in ("query", "gallery"):
        loader = engine.test_loader[kpr_run.DATA_NAME][name]
        if x.no_pin:
            loader.pin_memory = False
        with torch.no_grad():
            for i, data in enumerate(loader):
                imgs, tm, pm, kp, pid, cam = engine.parse_data_for_eval(data)
                imgs = imgs.cuda()
                tm = tm.cuda() if tm is not None else None
                pm = pm.cuda() if pm is not None else None
                out = model(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
                f, v, _, _ = extract_test_embeddings(out, cfg.model.kpr.test_embeddings)
                _ = f.cpu().numpy(), v.cpu().numpy()
                del data, imgs, tm, pm, kp, out, f, v
                if i % 8 == 0:
                    mark(f"{name}_{i}", i % 16 == 0)
        mark(f"{name}_end", True)
    Path(x.out).write_text(json.dumps(rows, indent=0))
    os.system(f"rm -rf '{save}'")
    os._exit(0)


main()
