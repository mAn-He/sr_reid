"""Stage 2 사후 분석 1: 공개 가중치 기준선 재확인. ALLOW_TEST=1(사용자 잠금 해제 중)에서만.
- 특징: 공개 가중치를 공식 입력 크기(BoT·CLIP-ReID 256×128, KPR 계열 384×128)로 원본 test에서 새로 추출. 추출기는 Stage 0.5 기준선 추출기
  (stage05/bot_run.py·clip_run.py·kpr_run.py --task extract, Stage 0.5 run_eval.py와 같은 인자). 공개 BoT는 공식 reid-strong-baseline 구조라
  Stage 2의 자체 구현 BoT 추출기로는 읽을 수 없어 추출만 Stage 0.5 경로를 쓴다.
- 지표: Stage 2 평가 코드 그대로(stage2/evaluate.py make_dist = 모델별 공식 거리, stage05/eval_core.stream_eval 블록 512, query별 AP).
- KPR+SOLIDER 로드 수정 확인: Stage 2 추출기(stage2/train_kpr.py --extract, semantic_embed 키 허용 수정 포함)와 Stage 0.5 추출기를
  같은 공개 가중치·같은 입력(×3 bicubic query)에 돌려 특징·가시성이 같은지 비교.
출력: data/locked/stage2_analysis/baseline/<model>/{query,gallery}/ (특징), results/stage2_analysis/baseline_recheck.csv(집계)"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MF = "/workspace/miniforge3/envs"
ENVPY = {"bot": f"{MF}/srreid-core/bin/python", "clipreid": f"{MF}/srreid-clipreid/bin/python", "kpr": f"{MF}/srreid-kpr/bin/python", "kpr_solider": f"{MF}/srreid-kpr/bin/python"}
SCRIPT = {"bot": "stage05/bot_run.py", "clipreid": "stage05/clip_run.py", "kpr": "stage05/kpr_run.py", "kpr_solider": "stage05/kpr_run.py"}
SIZE = {"bot": (256, 128), "clipreid": (256, 128), "kpr": (384, 128), "kpr_solider": (384, 128)}
BATCH = {"bot": 128, "clipreid": 64, "kpr": 64, "kpr_solider": 64}
REF = {"bot": (85.95, 94.63), "clipreid": (89.66, 95.43), "kpr": (89.59, 95.87), "kpr_solider": (93.25, 96.59)}
OUT = ROOT / "data" / "locked" / "stage2_analysis" / "baseline"
RES = ROOT / "results" / "stage2_analysis"


def extract(model, gpu):
    d = OUT / model
    if (d / "extract.done").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    h, w = SIZE[model]
    args = ["--task", "extract", "--features", str(d), "--batch", str(BATCH[model]), "--workers", "8", "--sub-q", "0", "--sub-g", "0"]
    if model == "bot":
        args += ["--cond", "orig", "--k", "1", "--arch", "official"]
    elif model == "clipreid":
        args += ["--cond", "orig", "--k", "1", "--h", str(h), "--w", str(w), "--public", "1"]
    else:
        args += ["--model", model, "--cond", "orig", "--k", "1", "--h", str(h), "--w", str(w), "--no-pin", "1", "--no-occluders", "1"]
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    rc = subprocess.call([ENVPY[model], SCRIPT[model]] + args + ["--out", str(d / "extract.out.json")], env=env, cwd=ROOT,
                         stdout=open(d / "extract.log", "w"), stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"{model} 추출 실패 rc={rc}, {d / 'extract.log'}")
    (d / "extract.done").write_text("ok")


def build_unit(model):
    """Stage 2 metrics_worker 입력 형식(<unit>/feats/{query,gallery} + extract_meta.json)으로 묶는다. pid −1은 추출기에서 이미 빠짐."""
    from stage05 import eval_core
    d, u = OUT / model, OUT / model / "unit"
    for split, tag in (("query", "query"), ("gallery", "gallery")):
        X = eval_core.load_split(d / split)
        feat = np.array(X["feat"], np.float32)
        meta = {"model": model, "weights": "public", "cond": "orig", "k": 1, "split": split, "seconds": 0.0, "gpu_peak_alloc_gib": None}
        if model == "bot":  # 공식 BoT test: feat_norm yes + 유클리드. 정규화 특징에서 1−cos와 순위가 같다(d² = 2 − 2cos) → Stage 2 BoT 거리 경로 사용
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            meta["l2_normalized_in_extract"] = True
        elif model == "clipreid":
            meta["trainer_meta"] = {"feat_norm_in_eval": "yes", "neck_feat": "before"}
        else:
            meta["trainer_meta"] = {"combine": "mean", "dist_metric": "euclidean", "normalize_feature": True, "test_embeddings": ["bn_foreg", "parts"]}
            meta["vis_dtype"] = str(np.array(X["vis"]).dtype)
        eval_core.save_split(u / "feats" / tag, feat, None if X.get("vis") is None else np.array(X["vis"]), np.array(X["pid"]), np.array(X["cam"]), list(X["names"]))
        (u / "feats" / tag / "extract_meta.json").write_text(json.dumps(meta))
    return u


def main():
    assert os.environ.get("ALLOW_TEST") == "1"
    model, gpu = sys.argv[1], sys.argv[2]
    extract(model, gpu)
    u = build_unit(model)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4")
    rc = subprocess.call([ENVPY[model], "stage2/evaluate.py", "_metrics", "--model", model, "--unit-dir", str(u)], env=env, cwd=ROOT)
    if rc:
        raise SystemExit(f"{model} 지표 실패 rc={rc}")
    m = json.loads((u / "metrics.json").read_text())
    r = {"model": model, "input": "x".join(map(str, SIZE[model])), "mAP": round(100 * m["mAP"], 2), "rank1": round(100 * m["rank1"], 2),
         "ref_mAP": REF[model][0], "ref_rank1": REF[model][1], "n_query_valid": m["n_query_valid"], "n_gallery": m["n_gallery"]}
    r["pass"] = r["mAP"] == r["ref_mAP"] and r["rank1"] == r["ref_rank1"]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"baseline_recheck_{model}.json").write_text(json.dumps(r, ensure_ascii=False))
    print(r, flush=True)


if __name__ == "__main__":
    main()
