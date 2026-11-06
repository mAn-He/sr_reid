"""Part H. 메모리 재진단. srreid-kpr env, train만 읽는다.
배치별 forward/loss/backward/step 단계 피크와 피크 위치, OOM이면 그 지점 스냅샷과 실패한 할당의 스택. checkpointing 등가성은 ckpt_equiv.py."""
import argparse
import collections
import json
import os
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
import torch  # noqa: E402

from stage05 import backends as B  # noqa: E402

GIB = 2**30


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def snapshot_top(n=25):
    """살아 있는 블록을 프레임별로 합산한다. 저장소와 우리 코드 프레임 가운데 가장 안쪽을 키로 쓴다. torch 1.13의 _snapshot은 세그먼트 리스트를 돌려준다."""
    try:
        snap = torch.cuda.memory._snapshot()
    except Exception as e:
        return {"error": str(e)}
    segs = snap if isinstance(snap, list) else snap.get("segments", [])
    per = collections.Counter()
    total, inactive, largest_free = 0, 0, 0
    largest = []
    for seg in segs:
        for block in seg.get("blocks", []):
            size = block["size"]
            if block.get("state") == "inactive":
                inactive += size
                largest_free = max(largest_free, size)
            if block.get("state") != "active_allocated":
                continue
            total += size
            history = block.get("history") or []
            frames = history[0].get("frames", []) if history else block.get("frames", [])
            where = "unknown"
            for fr in frames:
                fn = fr.get("filename", "")
                if "third_party" in fn or "stage05" in fn or "stage0" in fn:
                    where = f"{fn.split('third_party/')[-1]}:{fr.get('line')}:{fr.get('name')}"
                    break
            per[where] += size
            largest.append((size, where))
    largest.sort(key=lambda x: -x[0])
    return {"active_gib": total / GIB, "inactive_gib": inactive / GIB, "largest_free_gib": largest_free / GIB,
            "largest_blocks": [{"gib": a / GIB, "where": b} for a, b in largest[:8]],
            "top": [{"where": k, "gib": v / GIB} for k, v in per.most_common(n)]}


class PeakTracker:
    """모듈 forward와 출력 텐서의 grad 훅에서 allocated 피크가 1% 넘게 늘면 위치와 스냅샷을 남긴다. 훅은 값을 바꾸지 않는다."""

    def __init__(self, model):
        self.best, self.where, self.snap, self.phase = 0, "", None, "load"
        self.handles = []
        for name, m in model.named_modules():
            self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_m, _i, out):
            self.check(f"forward:{name}")
            tensors = [out] if torch.is_tensor(out) else [t for t in out if torch.is_tensor(t)] if isinstance(out, (tuple, list)) else []
            for t in tensors:
                if t.requires_grad:
                    t.register_hook(lambda _g, nm=name: self.check(f"grad:{nm}"))
        return fn

    def check(self, label):
        cur = torch.cuda.max_memory_allocated()
        if cur > self.best * 1.01:
            self.best = cur
            self.where = f"{self.phase}/{label}"
            self.snap = snapshot_top(12)


def ckpt_modules(model):
    """with_cp를 가진 모듈 종류와 수, 그리고 체크포인트 밖에서 파라미터를 가진 최상위 모듈."""
    kinds = collections.Counter(type(m).__name__ for m in model.modules() if hasattr(m, "with_cp"))
    top = {}
    for name, child in model.named_children():
        n_cp = sum(1 for m in child.modules() if hasattr(m, "with_cp"))
        n_p = sum(p.numel() for p in child.parameters())
        top[name] = {"with_cp_modules": n_cp, "params": n_p, "type": type(child).__name__}
    return {"with_cp_kinds": dict(kinds), "top_level": top}


def limit_occluders(n):
    """BIPO 가림 이미지 로드를 표본 n장의 신원으로 줄인다. 배치 텐서 크기가 그대로라 GPU 메모리 측정에 영향이 없고, 실행마다 약 2분을 아낀다. 0이면 공식 그대로."""
    if not n:
        return
    sys.path.insert(0, str(B.S0.KPR_REPO))
    from torchreid.data.datasets import dataset as dmod
    orig = dmod.load_human_occluders
    dmod.load_human_occluders = lambda samples, **k: orig(samples[:n], **k)


def phases(a):
    from torch.cuda import amp
    res = {"model": a.backend, "scale": a.scale, "batch": a.batch, "ckpt": bool(a.ckpt), "phases": {}, "occluder_samples": a.occluder_samples}
    limit_occluders(a.occluder_samples)
    b = B.build(a.backend, a.scale, a.batch, "fast", bool(a.ckpt), 42, None, ["memmap"])
    from torchreid.utils.constants import PIXELS  # build가 sys.path를 잡은 뒤에 import한다
    res["ckpt_structure"] = ckpt_modules(b.model)
    e = b.engine
    e.set_model_mode("train")
    try:
        torch.cuda.memory._record_memory_history(True)
    except Exception as ex:
        res["history_error"] = str(ex)
    phase = "load"
    tracker = PeakTracker(b.model)
    peak_all = 0

    def mark(name, fn):
        nonlocal phase, peak_all
        phase = name
        tracker.phase = name
        peak_all = max(peak_all, torch.cuda.max_memory_allocated())
        tracker.best = max(tracker.best, peak_all)
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = fn()
        torch.cuda.synchronize()
        tracker.check("end")
        res["phases"].setdefault(name, {"before_gib": before / GIB, "peak_gib": torch.cuda.max_memory_allocated() / GIB, "after_gib": torch.cuda.memory_allocated() / GIB,
                                        "reserved_gib": torch.cuda.max_memory_reserved() / GIB})
        res["phases"][name] = {"before_gib": before / GIB, "peak_gib": max(res["phases"][name]["peak_gib"], torch.cuda.max_memory_allocated() / GIB),
                               "after_gib": torch.cuda.memory_allocated() / GIB, "reserved_gib": torch.cuda.max_memory_reserved() / GIB}
        return out
    try:
        for step in range(a.steps):
            data = next(b.loader)
            imgs, tm, pm, kp, pids, _, cam = mark("h2d", lambda: e.parse_data_for_train(data))

            def fwd():
                with amp.autocast(enabled=e.mixed_precision):
                    return b.model(imgs, target_masks=tm, prompt_masks=pm, keypoints_xyc=kp, cam_label=cam)
            emb, vis, idc, pix, _, _ = mark("forward", fwd)
            if step == a.steps - 1:
                res["snapshot_after_forward"] = snapshot_top()

            def lossf():
                with amp.autocast(enabled=e.mixed_precision):
                    return e.combine_losses(vis, emb, idc, pids, pix, tm, bpa_weight=e.losses_weights[PIXELS]["ce"])[0]
            loss = mark("loss", lossf)
            e.optimizer.zero_grad()
            mark("backward", lambda: e.scaler.scale(loss).backward())
            mark("step", lambda: (e.scaler.step(e.optimizer), e.scaler.update()))
            del emb, vis, idc, pix, loss
        res["status"] = "ok"
    except RuntimeError as ex:
        if "out of memory" not in str(ex).lower():
            raise
        res["status"] = "oom"
        res["oom_phase"] = phase
        res["oom_message"] = str(ex)[:300]
        res["snapshot_at_oom"] = snapshot_top()
        tb = traceback.extract_tb(ex.__traceback__)
        res["oom_stack"] = [f"{f.filename.split('third_party/')[-1]}:{f.lineno}:{f.name}" for f in tb if "third_party" in f.filename or "stage05" in f.filename][-6:]
    res["peak_allocated_gib"] = max(peak_all, torch.cuda.max_memory_allocated()) / GIB
    res["peak_where"], res["snapshot_at_peak"] = tracker.where, tracker.snap
    st = torch.cuda.memory_stats()
    res["reserved_peak_gib"] = st.get("reserved_bytes.all.peak", 0) / GIB
    res["inactive_split_peak_gib"] = st.get("inactive_split_bytes.all.peak", 0) / GIB
    res["num_alloc_retries"] = st.get("num_alloc_retries", 0)
    res["num_ooms"] = st.get("num_ooms", 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="phases")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--occluder-samples", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed(42)
    try:
        res = phases(a)
    except BaseException as e:
        traceback.print_exc()
        res = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:600]}", "args": vars(a)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
