"""특징 추출과 ReID 지표. 이미 계산한 조건은 건너뛰고, SR 출력은 재생성하지 않는다.

--only bicubic이면 bicubic 조건과 기준선만 계산한다(평가 코드 end-to-end 점검).
"""
import argparse
import json

import numpy as np
import torch

from common.util import load_cfg, log_failure, read_csv, sha256_file, wait_free_gb, write_csv
from eval.metrics_reid import evaluate
from reid.extract import extract, load_model
from sr.run import cond_dir, conditions


def gallery(cfg, model, mode, rows):
    path = cfg["_run"] / "features" / mode / "gallery.npz"
    g = [r for r in rows if r["role"] == "gallery"]
    if path.exists():
        return np.load(path)["feats"], g
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = extract(model, [r["path"] for r in g], cfg)
    np.savez(path, feats=feats)
    return feats, g


def score(cfg, cond, mode, model, query, paths, g_feats, g_rows):
    out = cfg["_run"] / "eval" / "per_query" / f"{cond}.csv"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    valid = np.array([p is not None for p in paths])
    feats = np.zeros((len(paths), g_feats.shape[1]), np.float32)
    if valid.any():
        feats[valid] = extract(model, [p for p in paths if p is not None], cfg)
    fp = cfg["_run"] / "features" / mode / f"{cond}.npz"
    np.savez(fp, feats=feats, valid=valid)
    q_pids = np.array([int(r["pid"]) for r in query])
    q_cams = np.array([int(r["camid"]) for r in query])
    g_pids = np.array([int(r["pid"]) for r in g_rows])
    g_cams = np.array([int(r["camid"]) for r in g_rows])
    ap, r1 = evaluate(feats, q_pids, q_cams, valid, g_feats, g_pids, g_cams)
    write_csv(out, [{"name": r["name"], "pid": r["pid"], "camid": r["camid"], "valid": int(v), "ap": a, "r1": b}
                    for r, v, a, b in zip(query, valid, ap, r1)], ["name", "pid", "camid", "valid", "ap", "r1"])
    print(f"{cond}: mAP={ap.mean():.4f} R1={r1.mean():.4f} valid={valid.sum()}/{len(valid)} 특징 sha256={sha256_file(fp)[:12]}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    wait_free_gb(cfg["gpu"]["min_free_gb_eval"], cfg["gpu"]["wait_timeout_s"])
    torch.cuda.set_per_process_memory_fraction(cfg["gpu"]["fraction_train"])
    rows = read_csv(cfg["_manifests"] / "split.csv")
    query = sorted((r for r in rows if r["role"] == "query"), key=lambda r: r["name"])
    up = cfg["_processed"] / "val_query_lrup"
    lr_paths = [up / f"{r['name']}.png" for r in query]

    # Path 1 입력과 bicubic official 출력이 바이트 단위로 같은지 기록한다(리사이즈 경로 통일 확인).
    bic = next(c for c in conditions(cfg) if c["cid"] == "bicubic__official")
    bic_rows = {r["name"]: r for r in read_csv(cond_dir(cfg, bic) / "outputs.csv")}
    same = sum(bic_rows.get(p.name, {}).get("final_sha256") == sha256_file(p) for p in lr_paths)
    (cfg["_run"] / "path1_bicubic_identity.json").write_text(json.dumps({"identical": same, "n": len(lr_paths)}))

    for mode, cond, paths in (("mixed", "path1_mixed", lr_paths), ("lr_only", "path1_lr_only", lr_paths),
                              ("hr", "hr_oracle", [r["path"] for r in query])):
        try:
            model = load_model(cfg, mode)
            g_feats, g_rows = gallery(cfg, model, mode, rows)
            score(cfg, cond, mode, model, query, paths, g_feats, g_rows)
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            log_failure(cfg, "eval_reid", cond, "", e)
            print(f"{cond} 실패: {e}", flush=True)

    model = load_model(cfg, "hr")
    g_feats, g_rows = gallery(cfg, model, "hr", rows)
    for c in conditions(cfg):
        if a.only and c["method"] != a.only:
            continue
        d = cond_dir(cfg, c)
        out_csv = d / "outputs.csv"
        if not out_csv.exists():
            log_failure(cfg, "eval_reid", c["cid"], "", "SR 출력 없음")
            continue
        done = {r["name"] for r in read_csv(out_csv) if r["status"] == "ok"}
        paths = [d / "final" / f"{r['name']}.png" if f"{r['name']}.png" in done else None for r in query]
        score(cfg, c["cid"], "hr", model, query, paths, g_feats, g_rows)


if __name__ == "__main__":
    main()
