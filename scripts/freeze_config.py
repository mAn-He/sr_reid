"""config를 run 폴더에 고정한다. 같은 run_id에 내용이 다른 config를 쓰면 거부한다. commit hash도 기록한다."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.util import EXIT_FATAL, load_cfg, sha256_file  # noqa: E402


def commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    cfg = load_cfg(sys.argv[1])
    frozen = cfg["_run"] / "config.yaml"
    if frozen.exists() and sha256_file(frozen) != sha256_file(sys.argv[1]):
        print(f"{cfg['run_id']}의 config가 고정본과 다릅니다. 새 run_id와 change_reason으로 실행하세요.", flush=True)
        sys.exit(EXIT_FATAL)
    if not frozen.exists():
        shutil.copy(sys.argv[1], frozen)
        repos = {m: {"commit": commit(ROOT / c["repo"]),
                     "dirty": subprocess.check_output(["git", "-C", str(ROOT / c["repo"]), "status", "--porcelain"],
                                                      text=True).strip()}
                 for m, c in cfg["sr"]["methods"].items() if "repo" in c}
        (cfg["_run"] / "provenance.json").write_text(json.dumps({"code_commit": commit(ROOT), "repos": repos}, indent=1))


if __name__ == "__main__":
    main()
