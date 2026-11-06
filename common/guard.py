"""test split 접근 가드. 경로 검사와 audit hook을 함께 쓴다. ALLOW_TEST=1일 때만 허용.

chmod 000은 root에게 효력이 없으므로(파드는 root 실행) 이 가드가 실제 차단 수단이다.
표준 라이브러리만 쓴다. SR env의 runner.py도 이 파일을 직접 읽어 설치한다.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCKED_NAMES = ("query", "bounding_box_test", "gt_query", "gt_bbox")
LOCKED_ROOT = os.path.join(ROOT, "data", "locked")
MARKET_ROOT = os.path.join(ROOT, "data", "market")
DATA_ROOT = os.path.join(ROOT, "data")
CACHE_ROOT = os.environ.get("SRREID_CACHE", "/root/srreid_cache")  # 실행 중에만 있는 로컬 NVMe memmap 캐시
FLAG = os.path.join(ROOT, "logs", "GUARD_VIOLATION")
EVENTS = {"open", "os.listdir", "os.scandir", "glob.glob", "os.rename", "os.remove", "os.chmod"}

_in_hook = False


class GuardViolation(RuntimeError):
    pass


def is_locked(path):
    try:
        p = os.path.abspath(os.fsdecode(path))
    except TypeError:
        return False
    if p == LOCKED_ROOT or p.startswith(LOCKED_ROOT + os.sep):
        return True
    if any(p == t or p.startswith(t + os.sep) for t in (os.path.join(MARKET_ROOT, n) for n in LOCKED_NAMES)):
        return True
    # data/<condition>/x<k>/<split>/ 형식과 캐시 파일도 같은 규칙으로 막는다. 캐시 파일명은 <condition>__x<k>__<split>.*
    if p.startswith(DATA_ROOT + os.sep) and any(n in p[len(DATA_ROOT):].split(os.sep) for n in LOCKED_NAMES):
        return True
    if p.startswith(CACHE_ROOT + os.sep):
        name = os.path.basename(p)
        return any(f"__{n}." in name or f"__{n}__" in name for n in LOCKED_NAMES)
    return False


def check(path):
    if os.environ.get("ALLOW_TEST") == "1" or not is_locked(path):
        return
    global _in_hook
    _in_hook = True
    try:
        with open(FLAG, "a") as f:
            f.write(f"pid={os.getpid()} path={path}\n")
    finally:
        _in_hook = False
    raise GuardViolation(f"test split 접근 차단: {path} (ALLOW_TEST=1일 때만 허용)")


def _hook(event, args):
    if _in_hook or event not in EVENTS or not args or isinstance(args[0], int) or args[0] is None:
        return
    check(args[0])


def install():
    sys.addaudithook(_hook)
