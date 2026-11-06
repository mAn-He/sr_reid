"""Part H. checkpointing 등가성. 모델 env. train만 읽는다.
task=step: 같은 프로세스, 같은 배치로 off/on을 번갈아 반복해 1스텝 loss·grad norm·grad 벡터를 비교한다.
task=curve: 프로세스 1개 = ckpt 설정 1개. 같은 seed로 N iteration의 loss·grad norm 곡선을 남긴다.
mode: stage0(cudnn.deterministic만, Stage 0 측정과 같음) / warn(use_deterministic_algorithms(True, warn_only=True)) / cpu_up(warn에 더해 학습 경로의 interpolate를 CPU에서 계산)."""
import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from stage05 import backends as B  # noqa: E402
from stage05.mem_probe import limit_occluders  # noqa: E402


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def set_mode(mode):
    torch.backends.cudnn.deterministic = True
    if mode in ("warn", "cpu_up"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if mode == "cpu_up":
        orig = F.interpolate

        def interp(x, *a, **k):
            if x.is_cuda and x.requires_grad:
                return orig(x.cpu(), *a, **k).to(x.device)
            return orig(x, *a, **k)
        F.interpolate = interp


def grad_vector(model):
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])


class ClipToggle:
    """S0.ClipBackend의 checkpoint 감싸기와 같은 호출을 플래그로 켜고 끈다."""

    def __init__(self, b):
        from torch.utils.checkpoint import checkpoint
        self.on = False
        for block in b.model.image_encoder.transformer.resblocks:
            orig = block.forward
            block.forward = lambda x, o=orig: checkpoint(o, x, use_reentrant=False, preserve_rng_state=True) if self.on else o(x)


def one_step(a, b, data, toggle):
    """옵티마이저를 밟지 않는 1스텝. 같은 배치와 같은 가중치에서 시작한다."""
    from torch.cuda import amp
    b.model.zero_grad(set_to_none=True)
    seed(0)
    if a.backend == "clipreid":
        b.model.train()
        img, target = b._h2d(data)
        with amp.autocast(enabled=True):
            score, feat, image_features = b.model(x=img, label=target, cam_label=None, view_label=None)
            loss = b.loss_fn(score, feat, target, None, image_features @ b.text_features.t())
    else:
        from torchreid.utils.constants import PIXELS
        e = b.engine
        e.set_model_mode("train")
        imgs, tm, pm, kp, pids, _, cam = e.parse_data_for_train(data)
        with amp.autocast(enabled=e.mixed_precision):
            emb, vis, idc, pix, _, _ = b.model(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
            loss, _ = e.combine_losses(vis, emb, idc, pids, pix, tm, bpa_weight=e.losses_weights[PIXELS]["ce"])
    loss.backward()
    return float(loss.item()), grad_vector(b.model).cpu()


def step_task(a, res):
    b = B.build(a.backend, a.scale, a.batch, "fast", False, 42, None, ["memmap"])
    torch.backends.cudnn.benchmark = False
    state = {k: v.clone() for k, v in b.model.state_dict().items()}
    data = next(b.loader)
    toggle = ClipToggle(b) if a.backend == "clipreid" else None
    runs, vecs = [], {"off": [], "on": []}
    for i in range(a.repeats * 2):
        tag = "off" if i % 2 == 0 else "on"
        b.model.load_state_dict(state)
        if toggle:
            toggle.on = tag == "on"
        else:
            b.set_ckpt(tag == "on")
        torch.cuda.reset_peak_memory_stats()
        loss, vec = one_step(a, b, data, toggle)
        runs.append({"tag": tag, "loss": loss, "grad_norm": float(vec.norm()), "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
        vecs[tag].append(vec)
    off, on = vecs["off"], vecs["on"]
    noise_off = max(float((v - off[0]).abs().max()) for v in off)
    noise_on = max(float((v - on[0]).abs().max()) for v in on)
    diff_max = max(float((x - y).abs().max()) for x in off for y in on)
    norms = {t: [r["grad_norm"] for r in runs if r["tag"] == t] for t in ("off", "on")}
    losses = {t: [r["loss"] for r in runs if r["tag"] == t] for t in ("off", "on")}
    res.update(runs=runs, grad_vec_noise_off=noise_off, grad_vec_noise_on=noise_on, grad_vec_diff_on_off=diff_max,
               grad_norm_range_off=max(norms["off"]) - min(norms["off"]), grad_norm_range_on=max(norms["on"]) - min(norms["on"]),
               grad_norm_mean_gap=abs(float(np.mean(norms["on"]) - np.mean(norms["off"]))), grad_norm_mean=float(np.mean(norms["off"])),
               loss_range=max(losses["off"] + losses["on"]) - min(losses["off"] + losses["on"]))


def curve_task(a, res):
    flags = [] if a.flags == ["none"] else a.flags
    b = B.build(a.backend, a.scale, a.batch, "baseline" if not flags else "fast", bool(a.ckpt), 42, a.workers, flags)
    torch.backends.cudnn.benchmark = False
    losses, norms = [], []
    for _ in range(a.iters):
        losses.append(float(b.train_step()))
        norms.append(float(grad_vector(b.model).norm()))
    res.update(losses=losses, grad_norms=norms, peak_alloc_gib=torch.cuda.max_memory_allocated() / 2 ** 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["step", "curve"])
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["stage0", "warn", "cpu_up"])
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--occluder-samples", type=int, default=400)  # 0이면 공식 그대로
    ap.add_argument("--flags", nargs="*", default=["memmap"])  # curve만. none이면 Stage 0 경로
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"status": "ok", "args": vars(a), "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "torch": torch.__version__}
    seed(42)
    set_mode(a.mode)
    code = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if a.backend.startswith("kpr") and a.occluder_samples:
                limit_occluders(a.occluder_samples)
            {"step": step_task, "curve": curve_task}[a.task](a, res)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            res["status"] = "oom" if "out of memory" in str(e).lower() else "crashed"
            res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
            code = 1
    res["nondeterministic_ops"] = sorted({str(w.message).split(" does not have")[0] for w in caught if "deterministic" in str(w.message)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
