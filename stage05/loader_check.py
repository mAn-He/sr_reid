"""E.5 어댑터와 공식 로더 대조. 공식 입력 크기의 원본 조건에서 파일 목록·pid·camid·키포인트·픽셀을 비교하고 sampler 1 epoch를 끝까지 소비한다.
모델 env로 돌린다. query·gallery를 읽으므로 scripts/with_test.sh 아래에서 실행한다. 이미지는 수치 비교만 하고 열람하지 않는다."""
import argparse
import csv
import importlib
import importlib.util
import os
import random
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stage05 import cache as C  # noqa: E402
from stage05 import items as I  # noqa: E402

RES = ROOT / "results" / "stage05_loader_check.csv"
FIELDS = ["model", "check", "split", "n", "mismatch", "max_abs_diff", "note"]
SPLITS = ("bounding_box_train", "query", "bounding_box_test")
BOT = ROOT / "third_party" / "reid-strong-baseline"
CLIP = ROOT / "third_party" / "CLIP-ReID"
KPR = ROOT / "third_party" / "keypoint_promptable_reidentification"
MARKET = "Market-1501-v15.09.15"
KP_DIR = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"
MASK_DIR = "pifpaf_maskrcnn_filtering"


def key(path):
    return os.path.basename(path).split(".")[0]


def image_dir(split):
    return ROOT / "data" / "market" / split if split == "bounding_box_train" else ROOT / "data" / "locked" / split


def load_market(repo, name):
    """저장소 패키지 __init__을 거치지 않고 market1501과 bases만 올린다. 무관한 데이터셋의 의존성을 피한다."""
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(repo / "data" / "datasets" if name == "botds" else repo / "datasets")]
    sys.modules[name] = pkg
    return importlib.import_module(f"{name}.market1501").Market1501


def official_rows(model, split):
    """공식 process_dir 출력. 반환: 이름 정렬한 (경로, pid, camid). train은 relabel."""
    relabel = split == "bounding_box_train"
    d = str(image_dir(split))
    if model == "bot":
        Market = load_market(BOT, "botds")
        rows = object.__new__(Market)._process_dir(d, relabel=relabel)
    elif model == "clipreid":
        Market = load_market(CLIP, "clipds")
        obj = object.__new__(Market)
        obj.pid_begin = 0
        rows = [r[:3] for r in obj._process_dir(d, relabel=relabel)]
    else:
        sys.path.insert(0, str(KPR))
        from torchreid.data.datasets.image.market1501 import Market1501
        obj = object.__new__(Market1501)
        obj.dataset_dir, obj.kp_dir, obj.masks_dir = str(kpr_root(split)), KP_DIR, MASK_DIR
        obj.masks_parts_numbers, obj.has_background, obj.masks_suffix = Market1501.masks_dirs[MASK_DIR]
        rows = obj.process_dir(d, relabel=relabel)
        return sorted(rows, key=lambda r: os.path.basename(r["img_path"]))
    return sorted(rows, key=lambda r: os.path.basename(r[0]))


def kpr_root(split):
    if split == "bounding_box_train":
        return ROOT / "data" / "kpr" / "orig"
    return ROOT / "data" / "locked" / "kpr_root" / "orig" / MARKET


def adapter_rows(split):
    """어댑터 목록. train은 공식과 같은 규칙으로 재번호한다(stage0 train_items와 같은 식)."""
    rows = I.list_split(image_dir(split))
    if split == "bounding_box_train":
        remap = {p: i for i, p in enumerate(sorted({r[1] for r in rows}))}
        rows = [(r[0], remap[r[1]], r[2]) for r in rows]
    return rows


def _pixels(args):
    path, decoder, mm_args, i = args
    if decoder == "cv2":
        import cv2
        a = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"))
    mm = np.memmap(mm_args[0], np.uint8, "r", shape=mm_args[1])
    d = np.abs(a.astype(np.int16) - np.asarray(mm[i]).astype(np.int16))
    return int(d.max())


def check_pixels(paths, split, decoder):
    """memmap 항목과 공식 디코더 출력의 픽셀 전수 비교."""
    C.build("orig", 1, split)
    mm, idx = C.load("orig", 1, split)
    base = C.cache_path("orig", 1, split)
    mm_args = (str(base.with_name(base.name + ".u8")), mm.shape)
    with ProcessPoolExecutor(12) as ex:
        diffs = list(ex.map(_pixels, [(p, decoder, mm_args, idx[key(p)]) for p in paths], chunksize=64))
    return sum(d > 0 for d in diffs), max(diffs)


def _kpr_annot(args):
    kp_path, mask_path, kp, u8_args, f32_args, i = args
    sys.path.insert(0, str(KPR))
    from torchreid.utils.tools import read_keypoints, read_masks
    off = read_keypoints(kp_path, (0, 0, 64, 128))
    kp_bad = not (off.shape == kp.shape and np.array_equal(off.astype(np.float64), kp))
    m = read_masks(mask_path)
    m = np.transpose(m, (2, 0, 1)) if m.shape[0] != 36 else m  # read_masks 출력을 저장 축(36×17×9)에 맞춘다
    f32 = np.load(f32_args, mmap_mode="r")[i]
    u8 = np.load(u8_args, mmap_mode="r")[i]
    return kp_bad, bool(np.array_equal(m, f32)), float(np.abs(m - u8.astype(np.float32) / 255).max())


def check_kpr_annot(rows, split):
    z = np.load(ROOT / "data" / "kpr_cache" / "x1" / split / "keypoints.npz")
    kp_all, offsets = z["kp"], z["offsets"]  # npz 키 접근은 매번 전체를 다시 푼다. 한 번만 읽는다.
    kp_idx = {n: i for i, n in enumerate(z["names"].tolist())}
    d = ROOT / "data" / "kpr_cache" / "masks" / split
    m_idx = {n.split(".")[0]: i for i, n in enumerate((d / "names.txt").read_text().split("\n"))}
    jobs = []
    for r in rows:
        k = key(r["img_path"])
        i = kp_idx[k]
        jobs.append((r["kp_path"], r["masks_path"], kp_all[offsets[i]:offsets[i + 1]].copy(), str(d / "masks_u8.npy"), str(d / "masks_f32.npy"), m_idx[k]))
    with ProcessPoolExecutor(12) as ex:
        out = list(ex.map(_kpr_annot, jobs, chunksize=64))
    return sum(o[0] for o in out), sum(not o[1] for o in out), max(o[2] for o in out)


def sampler_epoch(model, off, ada, batch=64, inst=4, seed=42):
    """공식 sampler를 같은 seed로 1 epoch 끝까지 소비한다. 반환: (공식 이름열, 어댑터 이름열). BoT는 어댑터가 직접 구현 sampler다."""
    random.seed(seed)
    np.random.seed(seed)
    if model == "bot":
        sys.path.insert(0, str(BOT))
        spec = importlib.util.spec_from_file_location("bot_triplet_sampler", BOT / "data" / "samplers" / "triplet_sampler.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        o = [key(off[i][0]) for i in mod.RandomIdentitySampler(off, batch, inst)]
        from reid.trainer import pk_batches
        batches = pk_batches(np.array([r[1] for r in ada]), batch // inst, inst, np.random.default_rng(seed))
        return o, [key(ada[i][0]) for b in batches for i in b]
    if model == "clipreid":
        sys.path.insert(0, str(CLIP))
        spec = importlib.util.spec_from_file_location("clip_sampler", CLIP / "datasets" / "sampler.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        o = [key(off[i][0]) for i in mod.RandomIdentitySampler([r + (0,) for r in off], batch, inst)]  # 공식 sampler는 (경로, pid, camid, trackid)를 푼다
        random.seed(seed)
        np.random.seed(seed)
        return o, [key(ada[i][0]) for i in mod.RandomIdentitySampler([tuple(r) + (0,) for r in ada], batch, inst)]
    from torchreid.data.sampler import RandomIdentitySampler
    first = lambda i: i[0] if isinstance(i, tuple) else i  # 공식 sampler는 BIPO용으로 (index, batch_pids)를 낸다
    o = [key(off[first(i)]["img_path"]) for i in RandomIdentitySampler(off, batch, inst)]
    random.seed(seed)
    np.random.seed(seed)
    dicts = [{"img_path": r[0], "pid": r[1], "camid": r[2]} for r in ada]
    return o, [key(dicts[first(i)]["img_path"]) for i in RandomIdentitySampler(dicts, batch, inst)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bot", "clipreid", "kpr"])
    a = ap.parse_args()
    out = []

    def rec(check, split, n, mismatch, diff="", note=""):
        out.append({"model": a.model, "check": check, "split": split, "n": n, "mismatch": mismatch, "max_abs_diff": diff, "note": note})
        print(out[-1], flush=True)
    kpr = a.model == "kpr"
    train_off = train_ada = None
    for split in SPLITS:
        off = official_rows(a.model, split)
        ada = adapter_rows(split)
        if kpr:
            triples = {key(r["img_path"]): (r["pid"], r["camid"]) for r in off}
            paths = [r["img_path"] for r in off]
        else:
            triples = {key(r[0]): (r[1], r[2]) for r in off}
            paths = [r[0] for r in off]
        ada_t = {key(r[0]): (r[1], r[2]) for r in ada}
        rec("file_list", split, len(off), len(set(triples) ^ set(ada_t)), note=f"어댑터 {len(ada)}장")
        common = set(triples) & set(ada_t)
        rec("pid", split, len(common), sum(triples[n][0] != ada_t[n][0] for n in common))
        rec("camid", split, len(common), sum(triples[n][1] != ada_t[n][1] for n in common))
        bad, mx = check_pixels(paths, split, "cv2" if kpr else "pil")
        rec("pixels_memmap_vs_official_decoder", split, len(paths), bad, mx, "cv2.imread" if kpr else "PIL")
        if kpr:
            kp_bad, f_bad, u_max = check_kpr_annot(off, split)
            rec("keypoints", split, len(off), kp_bad, note="공식 read_keypoints와 x1 캐시 array_equal")
            rec("masks_f32", split, len(off), f_bad, note="공식 read_masks와 array_equal")
            rec("masks_u8_quant", split, len(off), 0, round(u_max, 6), "오차 상한 1/510≈0.00196")
        if split == "bounding_box_train":
            train_off, train_ada = off, ada
    o_seq, a_seq = sampler_epoch(a.model, train_off, train_ada)
    if a.model == "bot":
        rec("sampler_epoch", "bounding_box_train", len(a_seq), "", note=f"직접 구현 pk_batches. 공식 sampler와 순서 비교 대상 아님. 공식 epoch {len(o_seq)}장, 어댑터 {len(a_seq)}장, 커버 이미지 공식 {len(set(o_seq))} 어댑터 {len(set(a_seq))}")
    else:
        rec("sampler_epoch", "bounding_box_train", len(o_seq), sum(x != y for x, y in zip(o_seq, a_seq)) + abs(len(o_seq) - len(a_seq)), note="같은 seed, 이름 정렬 목록에서 공식 sampler 1 epoch")
    new = not RES.exists()
    with RES.open("a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
