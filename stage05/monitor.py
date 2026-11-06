"""명령 하나를 프로세스 그룹으로 실행하고 1초 간격으로 메인 + 모든 DataLoader 워커의 RSS 합계를 잰다. /usr/bin/time -v 병행.
memmap 공유 페이지는 RSS 합계에 워커 수만큼 겹치므로 PSS 합계도 같이 남긴다. cgroup 한도(57.7GiB)에 닿기 전에 감시 한도에서 끊고 결과로 기록한다."""
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import psutil

GIB = 2**30


def session_procs(sid):
    out = []
    for p in psutil.process_iter(["pid"]):
        try:
            if os.getsid(p.pid) == sid:
                out.append(p)
        except (ProcessLookupError, PermissionError):
            pass
    return out


def pss(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            m = re.search(r"^Pss:\s+(\d+) kB", f.read(), re.M)
        return int(m.group(1)) * 1024 if m else 0
    except OSError:
        return 0


def gpu_used_mib():
    try:
        return float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0.0


def parse_time(path):
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line.startswith("Maximum resident set size"):
                out["time_v_max_rss_gib"] = int(line.rsplit(":", 1)[1]) * 1024 / GIB
            elif line.startswith("Elapsed (wall clock)"):
                out["time_v_elapsed"] = line.rsplit(": ", 1)[1]
            elif line.startswith("Exit status"):
                out["time_v_exit"] = int(line.rsplit(":", 1)[1])
    except OSError:
        pass
    return out


def run(cmd, log, env=None, limit_gib=48.0, timeout=None, cwd=None):
    """cmd 전체를 새 세션에서 돌린다. 끝나면 그룹 전체를 죽여 고아 워커가 GPU·RAM을 잡지 못하게 한다."""
    time_file = str(log) + ".time"
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(["/usr/bin/time", "-v", "-o", time_file] + [str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                env=env, cwd=cwd, start_new_session=True)
    res = {"max_rss_sum_gib": 0.0, "max_pss_sum_gib": 0.0, "max_nprocs": 0, "max_gpu_mib": 0.0, "killed_over_limit": False, "timeout": False,
           "limit_gib": limit_gib}
    tick = 0
    while proc.poll() is None:
        try:
            ps = session_procs(proc.pid)
            rss = 0
            for p in ps:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            total_pss = sum(pss(p.pid) for p in ps)
        except psutil.Error:
            rss, total_pss, ps = 0, 0, []
        res["max_rss_sum_gib"] = max(res["max_rss_sum_gib"], rss / GIB)
        res["max_pss_sum_gib"] = max(res["max_pss_sum_gib"], total_pss / GIB)
        res["max_nprocs"] = max(res["max_nprocs"], len(ps))
        if tick % 2 == 0:
            res["max_gpu_mib"] = max(res["max_gpu_mib"], gpu_used_mib())
        tick += 1
        # 공유 페이지 중복이 없는 PSS로 한도를 판단한다. RSS 합계는 memmap 워커에서 실제보다 크게 나온다.
        if total_pss / GIB > limit_gib:
            res["killed_over_limit"] = True
            break
        if timeout and time.time() - t0 > timeout:
            res["timeout"] = True
            break
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    time.sleep(3)
    res["rc"] = proc.returncode
    res["seconds"] = time.time() - t0
    res.update(parse_time(time_file))
    return res


if __name__ == "__main__":
    import sys
    print(json.dumps(run(sys.argv[2:], sys.argv[1]), indent=1))
