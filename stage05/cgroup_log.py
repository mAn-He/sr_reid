"""파드 cgroup 메모리 1초 기록기(표준 라이브러리만). v1 memory.usage_in_bytes가 v2의 memory.current에 해당한다. 측정에 영향이 없도록 파일 두 개만 읽는다."""
import os
import sys
import time

D = "/sys/fs/cgroup/memory"
KEYS = ("total_rss", "total_cache", "total_shmem", "total_mapped_file", "total_inactive_file", "total_active_file")


def main():
    out = sys.argv[1]
    new = not os.path.exists(out)
    with open(out, "a", buffering=1) as f:
        if new:
            f.write("t,usage," + ",".join(KEYS) + ",max_usage\n")
        while True:
            t = time.time()
            usage = open(f"{D}/memory.usage_in_bytes").read().strip()
            peak = open(f"{D}/memory.max_usage_in_bytes").read().strip()
            stat = dict(line.split() for line in open(f"{D}/memory.stat"))
            f.write(f"{t:.1f},{usage}," + ",".join(stat.get(k, "") for k in KEYS) + f",{peak}\n")
            time.sleep(max(0.0, 1.0 - (time.time() - t)))


if __name__ == "__main__":
    main()
