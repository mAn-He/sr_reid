"""파드 cgroup 메모리 1초 기록기(표준 라이브러리만). cgroup v1·v2 자동 판별.
v2: memory.current(현재 사용, v1 usage_in_bytes에 해당), memory.peak(있으면), memory.stat, memory.events(oom, oom_kill, max, high).
v1: memory.usage_in_bytes, memory.max_usage_in_bytes, memory.stat(total_*), memory.oom_control.
열은 v1·v2 공통 이름으로 쓴다. 측정에 영향이 없도록 작은 파일만 읽는다.
사용: python stage05/cgroup_log.py OUT.csv"""
import os
import sys
import time

V2_ROOT = "/sys/fs/cgroup"
V1_ROOT = "/sys/fs/cgroup/memory"
# 공통 열 이름 -> (v1 memory.stat 키, v2 memory.stat 키)
STAT_KEYS = {"rss": ("total_rss", "anon"), "cache": ("total_cache", "file"), "shmem": ("total_shmem", "shmem"), "mapped_file": ("total_mapped_file", "file_mapped"),
             "inactive_file": ("total_inactive_file", "inactive_file"), "active_file": ("total_active_file", "active_file")}
EVENT_KEYS = ("oom", "oom_kill", "max", "high")


def detect():
    """'v2'·'v1'·None. v2는 통합 계층의 memory.current 존재로 판별한다."""
    if os.path.exists(f"{V2_ROOT}/memory.current"):
        return "v2"
    if os.path.exists(f"{V1_ROOT}/memory.usage_in_bytes"):
        return "v1"
    return None


def _read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def _kv(p):
    out = {}
    for line in _read(p).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def limit_bytes(ver=None):
    """cgroup 메모리 상한(바이트). 제한 없음이면 None."""
    ver = ver or detect()
    if ver == "v2":
        v = _read(f"{V2_ROOT}/memory.max")
        return None if v in ("", "max") else int(v)
    if ver == "v1":
        v = _read(f"{V1_ROOT}/memory.limit_in_bytes")
        return int(v) if v and int(v) < (1 << 62) else None
    return None


def sample(ver):
    """한 번 읽기. dict(공통 열 -> 문자열)."""
    if ver == "v2":
        stat = _kv(f"{V2_ROOT}/memory.stat")
        ev = _kv(f"{V2_ROOT}/memory.events")
        row = {"usage": _read(f"{V2_ROOT}/memory.current"), "peak": _read(f"{V2_ROOT}/memory.peak")}
        row.update({k: stat.get(v2, "") for k, (_, v2) in STAT_KEYS.items()})
        row.update({f"ev_{k}": ev.get(k, "") for k in EVENT_KEYS})
        return row
    stat = _kv(f"{V1_ROOT}/memory.stat")
    oc = _kv(f"{V1_ROOT}/memory.oom_control")
    row = {"usage": _read(f"{V1_ROOT}/memory.usage_in_bytes"), "peak": _read(f"{V1_ROOT}/memory.max_usage_in_bytes")}
    row.update({k: stat.get(v1, "") for k, (v1, _) in STAT_KEYS.items()})
    row.update({"ev_oom": oc.get("under_oom", ""), "ev_oom_kill": oc.get("oom_kill", ""), "ev_max": _read(f"{V1_ROOT}/memory.failcnt"), "ev_high": ""})
    return row


COLS = ["usage", "peak", *STAT_KEYS, *(f"ev_{k}" for k in EVENT_KEYS)]


def main():
    out = sys.argv[1]
    ver = detect()
    if ver is None:
        sys.exit("cgroup 메모리 인터페이스를 찾지 못함")
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    with open(out, "a", buffering=1) as f:
        if new:
            f.write(f"# cgroup={ver} limit_bytes={limit_bytes(ver)}\n")
            f.write("t," + ",".join(COLS) + "\n")
        while True:
            t = time.time()
            row = sample(ver)
            f.write(f"{t:.1f}," + ",".join(row.get(k, "") for k in COLS) + "\n")
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


if __name__ == "__main__":
    main()
