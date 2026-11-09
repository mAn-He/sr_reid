"""Stage 2 평가용 특징 추출 래퍼(2026-11-09). 모델 env에서 실행한다. stage2/evaluate.py가 부른다.

학습기의 --extract 경로를 그대로 호출한다(같은 모델 생성·가중치 로드·전처리·배치·특징 정의). 바꾸는 것은 추출할 행 목록뿐이다:
공식 Market 로더처럼 pid -1(junk)을 뺀다(stage05/items.py:list_split과 같은 규칙). gallery의 0000(distractor)은 남는다.
  - bot:      train_bot.py --extract --cache-cond. stage05.cache.load가 돌려주는 memmap을 선택 행만 보이는 보기로 바꾼다.
  - clipreid: train_clipreid.py --extract --images <목록>. 공식 ImageDataset + val_transforms(Resize는 크기가 같아 no-op).
  - kpr*:     train_kpr.py --extract. stage05.cache.load가 돌려주는 이름→행 사전에서 선택하지 않은 이름을 뺀다(memmap 행 번호는 그대로).
              test 마스크·키포인트 캐시에 junk가 없을 수 있어 처음부터 빼고 추출한다.
출력(eval_core 형식): <out>/<tag>/{feat,vis(kpr만),pid,cam}.npy, names.txt(확장자 없는 파일명), extract_meta.json.
test split은 가드가 막는다(ALLOW_TEST=1일 때만).

사용: eval_extract.py --model M --weights W --cond C --k K --split S --out DIR [--tag T] [--names-file F] [--limit N]
"""
import argparse
import importlib.util
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 스크립트 폴더(stage2/)가 sys.path[0]이면 stage2/queue.py가 표준 라이브러리 queue를 가린다(DataLoader·urllib3가 깨짐). 뺀다.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from stage05 import cache as C  # noqa: E402
from stage05 import eval_core  # noqa: E402

PATTERN = re.compile(r"([-\d]+)_c(\d)")
SCRIPT = {"bot": "train_bot.py", "clipreid": "train_clipreid.py", "kpr": "train_kpr.py", "kpr_solider": "train_kpr.py"}
EXTS = (".webp", ".png", ".jpg")


def key(name):
    """Market 원본에 x.jpg.jpg 형식이 있어 첫 점 앞까지를 이름으로 쓴다(stage05/cache.py:load와 같음)."""
    return os.path.basename(str(name)).split(".")[0]


def parse(name):
    pid, cam = map(int, PATTERN.search(key(name)).groups())
    return pid, cam - 1


def select(names, names_file, limit):
    """names(원래 순서)에서 쓸 행 번호. pid -1 제외, 목록이 있으면 목록 안의 이름만."""
    keep = None
    if names_file:
        keep = {s.strip() for s in Path(names_file).read_text().split("\n") if s.strip()}
    rows = [i for i, n in enumerate(names) if parse(n)[0] != -1 and (keep is None or key(n) in keep)]
    if keep is not None and len(rows) != len(keep):
        raise SystemExit(f"--names-file의 {len(keep)}개 중 {len(rows)}개만 찾음")
    return rows[:limit] if limit else rows


def load_trainer(model):
    path = ROOT / "stage2" / SCRIPT[model]
    spec = importlib.util.spec_from_file_location(f"s2eval_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RowView:
    """memmap의 선택 행만 0..n-1로 보이는 보기. train_bot.extract는 len(mm)과 mm[i]만 쓴다."""

    def __init__(self, mm, rows):
        self.mm, self.rows = mm, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.mm[self.rows[i]]


class _Exit(Exception):
    pass


def call_main(mod, argv):
    """학습기 main()을 argv로 부른다. train_kpr.main 끝의 os._exit는 예외로 바꿔 받아 이 프로세스가 결과를 마저 쓴다."""
    real_exit = os._exit
    me = os.getpid()

    def fake_exit(code=0):
        if os.getpid() != me:  # fork된 DataLoader 작업자(popen_fork가 os._exit로 끝냄)는 진짜로 끝낸다
            real_exit(code)
        raise _Exit(code)
    os._exit = fake_exit
    old = sys.argv
    sys.argv = [mod.__file__] + [str(x) for x in argv]
    try:
        mod.main()
    except _Exit as e:
        if e.args and e.args[0]:
            raise SystemExit(e.args[0])
    finally:
        sys.argv = old
        os._exit = real_exit


def cache_names(cond, k, split):
    C.build(cond, k, split)
    base = C.cache_path(cond, k, split)
    meta = json.loads(base.with_name(base.name + ".json").read_text())
    return meta["names"], meta


def run(a):
    t0 = time.perf_counter()
    tmp = Path(a.out) / f"_tmp_{a.tag}"
    tmp.mkdir(parents=True, exist_ok=True)
    mod = load_trainer(a.model)
    extra = {}
    if a.model == "bot":
        names, meta = cache_names(a.cond, a.k, a.split)
        n_all = len(names)
        rows = select(names, a.names_file, a.limit)
        sel = [key(names[i]) for i in rows]
        orig_load = C.load

        def load(cond, k, split):
            mm, _ = orig_load(cond, k, split)
            return RowView(mm, rows), {n: j for j, n in enumerate(sel)}
        C.load = load
        call_main(mod, ["--extract", "--model-pt", a.weights, "--out", tmp / "feats.npy",
                        "--cache-cond", a.cond, "--cache-k", a.k, "--cache-split", a.split])
        C.load = orig_load
        feat, vis, got = np.load(tmp / "feats.npy"), None, sel
        extra = {"source": meta["source"], "via": "memmap RowView", "l2_normalized_in_extract": True}
    elif a.model == "clipreid":
        src = C.source_dir(a.cond, a.k, a.split)
        files = sorted(p for p in src.iterdir() if p.suffix in EXTS)
        n_all = len(files)
        rows = select([p.name for p in files], a.names_file, a.limit)
        sel = [key(files[i].name) for i in rows]
        lst = tmp / "images.txt"
        lst.write_text("\n".join(str(files[i]) for i in rows) + "\n")
        call_main(mod, ["--extract", "--k", a.k, "--model-pth", a.weights, "--images", lst, "--out", tmp])
        feat, vis = np.load(tmp / "feats.npy"), None
        fm = json.loads((tmp / "feats_meta.json").read_text())
        got = [key(n) for n in fm.pop("names")]
        fm.pop("pids", None)
        fm.pop("camids", None)
        extra = {"source": str(src), "via": "official ImageDataset + val_transforms", "trainer_meta": fm}
    else:
        names, meta = cache_names(a.cond, a.k, a.split)
        n_all = len(names)
        rows = select(names, a.names_file, a.limit)
        sel = [key(names[i]) for i in rows]
        orig_load = C.load

        def load(cond, k, split):
            mm, idx = orig_load(cond, k, split)
            if (cond, k, split) == (a.cond, a.k, a.split):
                keep = set(sel)
                idx = {n: i for n, i in idx.items() if n in keep}
            return mm, idx
        C.load = load
        call_main(mod, ["--extract", "--model", a.model, "--model-pth", a.weights, "--cond", a.cond, "--k", a.k,
                        "--split", a.split, "--out", tmp])
        C.load = orig_load
        d = eval_core.load_split(tmp / a.split)
        feat, vis, got = np.array(d["feat"]), np.array(d["vis"]), [key(n) for n in d["names"]]
        extra = {"source": meta["source"], "via": "memmap idx filter", "trainer_meta": json.loads((tmp / "extract.json").read_text())}
    if got != sel:
        raise SystemExit(f"추출 순서·이름 불일치: {len(got)} vs {len(sel)}")
    if len(feat) != len(sel):
        raise SystemExit(f"특징 수 {len(feat)} != 선택 행 수 {len(sel)}")
    pc = np.array([parse(n) for n in sel], dtype=np.int64).reshape(-1, 2)
    out = Path(a.out) / a.tag
    eval_core.save_split(out, feat, vis, pc[:, 0], pc[:, 1], sel)
    peak = None
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_initialized():
            peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    w = Path(a.weights)
    meta_out = {"model": a.model, "weights": str(w.resolve()), "weights_bytes": w.stat().st_size, "weights_mtime": w.stat().st_mtime,
                "cond": a.cond, "k": a.k, "split": a.split, "tag": a.tag, "names_file": a.names_file, "limit": a.limit,
                "n": len(sel), "n_source_rows": n_all, "feat_shape": list(feat.shape), "feat_dtype": str(feat.dtype),
                "vis_shape": None if vis is None else list(vis.shape), "vis_dtype": None if vis is None else str(vis.dtype),
                "seconds": round(time.perf_counter() - t0, 2), "gpu_peak_alloc_gib": peak,
                "maxrss_gib_main_proc": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "conc_note": os.environ.get("SRREID_CONC_NOTE"),
                **extra}
    tmp_meta = out / "extract_meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta_out, indent=1, ensure_ascii=False))
    os.replace(tmp_meta, out / "extract_meta.json")  # 완료 표시. 이것이 있으면 이 split 추출은 끝난 것
    os.system(f"rm -rf '{tmp}'")
    print(json.dumps({k: meta_out[k] for k in ("model", "cond", "k", "split", "n", "feat_shape", "seconds", "gpu_peak_alloc_gib")}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=list(SCRIPT))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--k", type=int, required=True, choices=(3, 4))
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="출력 하위 폴더 이름(기본: split)")
    ap.add_argument("--names-file", default=None, help="이 이름(확장자 없음)만 추출. 개발용 가짜 query/gallery")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.tag = a.tag or a.split
    code = 0
    try:
        run(a)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int):
            print(e.code, file=sys.stderr, flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # DataLoader 작업자 정리 지연 방지(train_kpr.py와 같은 이유)


if __name__ == "__main__":
    main()
