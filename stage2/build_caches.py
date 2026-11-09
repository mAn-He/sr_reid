"""Stage 2 학습 전 train memmap 캐시를 모든 조건에 미리 만든다(SRREID_CACHE, 기본 /dev/shm/srreid_cache로 실행). 크기·시간 기록."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
from stage05 import cache as C  # noqa: E402

rows = []
for cond in ("bicubic", "swinir", "pisa_sr_sem0.0", "osdhuman"):
    for k in (3, 4):
        r = C.build(cond, k, "bounding_box_train", workers=16)
        meta = json.loads(C.cache_path(cond, k, "bounding_box_train").with_suffix(".json").read_text()) if False else None
        rows.append({"cond": cond, "k": k, **r})
        print(rows[-1], flush=True)
out = ROOT / "results" / "stage2" / "train_cache_build.json"
out.write_text(json.dumps({"cache_root": str(C.CACHE), "rows": rows, "total_gib": sum(r["bytes"] for r in rows) / 2**30}, indent=1))
