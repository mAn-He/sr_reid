"""SR env 안에서 공식 스크립트를 수정 없이 실행하고 시간, VRAM, 단독 여부를 기록한다.

runpy로 같은 프로세스에서 돌려야 torch.cuda.max_memory_allocated를 읽을 수 있다.
이미지별 시간은 저장 함수에 런타임 훅을 걸어 저장 간격으로 잰다(소스 파일은 건드리지 않음).
"""
import argparse
import importlib.util
import json
import os
import random
import runpy
import subprocess
import sys
import threading
import time

EXIT_OOM = 77
EXIT_GUARD = 3


def load_guard(root):
    spec = importlib.util.spec_from_file_location("srreid_guard", os.path.join(root, "common", "guard.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install()
    return mod


def used_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return float(out.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("script_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    guard = load_guard(a.root)
    sys.path.insert(0, a.repo)

    import numpy as np
    import torch
    from PIL import Image
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.cuda.set_per_process_memory_fraction(a.fraction)

    out_dir = os.path.abspath(a.out_dir)
    saves = []
    pil_save = Image.Image.save

    def timed_pil(self, fp, *args, **kw):
        res = pil_save(self, fp, *args, **kw)
        if isinstance(fp, (str, os.PathLike)) and os.path.abspath(fp).startswith(out_dir):
            saves.append(time.perf_counter())
        return res
    Image.Image.save = timed_pil
    try:
        import cv2
        cv_write = cv2.imwrite

        def timed_cv(path, *args, **kw):
            res = cv_write(path, *args, **kw)
            if os.path.abspath(path).startswith(out_dir):
                saves.append(time.perf_counter())
            return res
        cv2.imwrite = timed_cv
    except ImportError:
        pass

    foreign = [0.0]
    stop = threading.Event()

    def sampler():
        while not stop.wait(2):
            own = torch.cuda.memory_reserved() / 2**20
            foreign[0] = max(foreign[0], used_mib() - own)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    t0 = time.perf_counter()
    start_used = used_mib()
    rc, err = 0, None
    sys.argv = [a.script] + [x for x in a.script_args if x != "--"]
    try:
        runpy.run_path(a.script, run_name="__main__")
    except torch.cuda.OutOfMemoryError as e:
        rc, err = EXIT_OOM, str(e)
    except guard.GuardViolation as e:
        rc, err = EXIT_GUARD, str(e)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as e:  # 실패는 결과로 기록한다. 호출자가 failures.csv에 남긴다.
        rc, err = 1, f"{type(e).__name__}: {e}"
        if "out of memory" in str(e).lower():
            rc = EXIT_OOM
    stop.set()
    th.join(timeout=5)
    intervals = [b - a_ for a_, b in zip(saves, saves[1:])]
    with open(a.timing, "w") as f:
        json.dump({"rc": rc, "error": err, "n_saved": len(saves), "per_image_seconds": intervals,
                   "first_save_seconds": saves[0] - t0 if saves else None, "total_seconds": time.perf_counter() - t0,
                   "max_memory_allocated": torch.cuda.max_memory_allocated(),
                   "max_memory_reserved": torch.cuda.max_memory_reserved(),
                   "start_used_mib": start_used, "max_foreign_mib": foreign[0]}, f, indent=1)
    if err:
        print("runner 오류:", err, file=sys.stderr, flush=True)
    os._exit(rc)  # 외부 스크립트의 남은 스레드가 종료를 막지 못하게 한다. 위에서 파일을 닫았다.


if __name__ == "__main__":
    main()
