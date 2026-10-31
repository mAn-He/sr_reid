"""config, 경로, sha256, manifest, 실패 기록, GPU 상태. 모든 core 단계가 쓴다."""
import csv
import hashlib
import os
import subprocess
import time
from pathlib import Path

import yaml

from common import guard

ROOT = Path(guard.ROOT)
EXIT_FATAL = 2
EXIT_GUARD = 3
EXIT_OOM = 77


def load_cfg(path, install_guard=True):
    if install_guard:
        guard.install()
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = str(path)
    p = cfg["paths"]
    cfg["_run"] = ROOT / p["runs"] / cfg["run_id"]
    cfg["_results"] = ROOT / p["results"]
    cfg["_market"] = ROOT / p["market_root"]
    cfg["_processed"] = ROOT / p["processed"]
    cfg["_manifests"] = cfg["_processed"] / "manifests"
    for d in (cfg["_run"], cfg["_results"], cfg["_processed"], cfg["_manifests"]):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows, fields):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def log_failure(cfg, stage, condition, image, reason):
    path = cfg["_results"] / "failures.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "run_id", "stage", "condition", "image", "reason"])
        w.writerow([time.strftime("%F %T"), cfg["run_id"], stage, condition, image, str(reason)[:300]])


def gpu_memory_gb():
    """(used, free, total) GiB. RunPod 컨테이너에서는 compute-apps 목록이 비어 나오므로 메모리량으로 판단한다."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    used, free, total = (float(x) / 1024 for x in out.strip().splitlines()[0].split(","))
    return used, free, total


def wait_free_gb(min_free_gb, timeout_s, poll_s=20):
    """min_free_gb 검사. 시간 안에 비지 않으면 False."""
    t0 = time.time()
    while True:
        if gpu_memory_gb()[1] >= min_free_gb:
            return True
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(poll_s)


def wait_exclusive(max_used_gb, timeout_s, poll_s=20):
    t0 = time.time()
    while True:
        if gpu_memory_gb()[0] <= max_used_gb:
            return True
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(poll_s)
