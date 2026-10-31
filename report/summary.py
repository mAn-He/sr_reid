"""results/pilot_summary.csv, grid.png, dose_response.png를 만든다. 이미지는 로컬 저장만 한다."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from common.util import load_cfg, read_csv, write_csv
from sr.run import cond_dir, conditions

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz 기본 팔레트 slot 1-3
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
FIELDS = ["method", "lambda_sem", "preup_policy", "rank1", "rank1_ci_lo", "rank1_ci_hi", "map", "map_ci_lo", "map_ci_hi",
          "vs_path1_mixed_map", "vs_path1_mixed_ci_lo", "vs_path1_mixed_ci_hi", "verdict_vs_path1_mixed",
          "vs_path1_lr_only_map", "vs_path1_lr_only_ci_lo", "vs_path1_lr_only_ci_hi", "verdict_vs_path1_lr_only",
          "vs_bicubic_map", "vs_bicubic_ci_lo", "vs_bicubic_ci_hi", "verdict_vs_bicubic",
          "vs_path1_mixed_rank1", "verdict_rank1_vs_path1_mixed", "vs_bicubic_rank1", "verdict_rank1_vs_bicubic",
          "psnr", "ssim", "lpips", "sec_per_image", "exclusive_gpu", "peak_vram_allocated_gb", "n_output_ok", "n_query"]


def get(boot, cond, metric, comp=""):
    return boot.get((cond, metric, comp))


def fill_compare(row, boot, cond, ref, key, verdict_key):
    b = get(boot, cond, "map", ref)
    if b:
        row[f"{key}_map"], row[f"{key}_ci_lo"], row[f"{key}_ci_hi"] = b["estimate"], b["ci_lo"], b["ci_hi"]
        row[verdict_key] = b["verdict"]


def build_summary(cfg):
    ev = cfg["_run"] / "eval"
    bpath = ev / "bootstrap.csv"
    boot = {(r["condition"], r["metric"], r["comparator"]): r for r in read_csv(bpath)} if bpath.exists() else {}
    n_query = sum(r["role"] == "query" for r in read_csv(cfg["_manifests"] / "split.csv"))
    rows = []

    def metric_cols(row, cond):
        for m in ("rank1", "map"):
            b = get(boot, cond, m)
            if b:
                row[m], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = b["estimate"], b["ci_lo"], b["ci_hi"]

    for name, cond in (("hr_oracle", "hr_oracle"), ("path1_mixed(주)", "path1_mixed"), ("path1_lr_only(보조)", "path1_lr_only")):
        row = {"method": name, "preup_policy": "reference", "n_query": n_query}
        metric_cols(row, cond)
        rows.append(row)
    for c in conditions(cfg):
        cid = c["cid"]
        row = {"method": c["method"], "lambda_sem": c["lambda_sem"], "preup_policy": c["policy"], "n_query": n_query}
        metric_cols(row, cid)
        fill_compare(row, boot, cid, "path1_mixed", "vs_path1_mixed", "verdict_vs_path1_mixed")
        fill_compare(row, boot, cid, "path1_lr_only", "vs_path1_lr_only", "verdict_vs_path1_lr_only")
        fill_compare(row, boot, cid, f"bicubic__{c['policy']}", "vs_bicubic", "verdict_vs_bicubic")
        for ref, key in (("path1_mixed", "path1_mixed"), (f"bicubic__{c['policy']}", "bicubic")):
            b = get(boot, cid, "rank1", ref)
            if b:
                row[f"vs_{key}_rank1"], row[f"verdict_rank1_vs_{key}"] = b["estimate"], b["verdict"]
        f = ev / "fidelity" / f"{cid}.csv"
        if f.exists():
            t = read_csv(f)
            for k in ("psnr", "ssim", "lpips"):
                row[k] = float(np.mean([float(r[k]) for r in t])) if t else None
        d = cond_dir(cfg, c)
        tp = d / "timing.json"
        if tp.exists():
            t = json.loads(tp.read_text())
            row["sec_per_image"] = t.get("sec_per_image_exclusive")
            row["exclusive_gpu"] = t.get("exclusive_gpu")
            peak = t.get("max_memory_allocated_exclusive")
            row["peak_vram_allocated_gb"] = peak / 2**30 if peak is not None else None
        o = d / "outputs.csv"
        row["n_output_ok"] = sum(r["status"] == "ok" for r in read_csv(o)) if o.exists() else 0
        rows.append(row)
    write_csv(cfg["_results"] / "pilot_summary.csv", rows, FIELDS)


def build_grid(cfg):
    """val 10개 ID × 방법 비교. official 정책의 대표 방법과 HR, LR을 나란히 놓는다."""
    query = sorted((r for r in read_csv(cfg["_manifests"] / "split.csv") if r["role"] == "query"), key=lambda r: r["name"])
    first = {}
    for r in query:
        first.setdefault(r["pid"], r)
    pids = sorted(first)
    rng = np.random.RandomState(cfg["seed"])
    pick = sorted(rng.choice(pids, min(10, len(pids)), replace=False).tolist())
    sem = cfg["eval"]["correlate"]["sem_representative"]
    columns = [("HR", None), ("LR 32x16", "lr")]
    for c in conditions(cfg):
        if c["policy"] == "official" and c["lambda_sem"] in (None, sem):
            columns.append((c["base"], c))
    cw, ch, label = 64 * 2 + 8, 128 * 2 + 8, 24
    canvas = Image.new("RGB", (cw * len(columns), label + ch * len(pick)), SURFACE)
    draw = ImageDraw.Draw(canvas)
    for j, (name, _) in enumerate(columns):
        draw.text((j * cw + 4, 6), name, fill=INK)
    for i, pid in enumerate(pick):
        r = first[pid]
        for j, (_, src) in enumerate(columns):
            if src is None:
                p = Path(r["path"])
            elif src == "lr":
                p = cfg["_processed"] / "val_query_lr" / f"{r['name']}.png"
            else:
                p = cond_dir(cfg, src) / "final" / f"{r['name']}.png"
            if p.exists():
                canvas.paste(Image.open(p).convert("RGB").resize((128, 256), Image.NEAREST), (j * cw + 4, label + i * ch + 4))
            else:
                draw.text((j * cw + 8, label + i * ch + 120), "출력 없음".encode("ascii", "ignore").decode() or "missing", fill=MUTED)
    canvas.save(cfg["_results"] / "grid.png")


def build_dose(cfg):
    ev = cfg["_run"] / "eval" / "bootstrap.csv"
    if not ev.exists():
        return
    boot = {(r["condition"], r["metric"], r["comparator"]): r for r in read_csv(ev)}
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    plotted = False
    for color, policy in zip(SERIES, cfg["sr"]["preup_policies"]):
        pts = []
        for c in conditions(cfg):
            if c["lambda_sem"] is not None and c["policy"] == policy:
                b = boot.get((c["cid"], "map", ""))
                if b:
                    pts.append((c["lambda_sem"], float(b["estimate"]), float(b["ci_lo"]), float(b["ci_hi"])))
        if pts:
            a = np.array(pts)
            ax.errorbar(a[:, 0], a[:, 1], yerr=[a[:, 1] - a[:, 2], a[:, 3] - a[:, 1]], color=color, lw=2, marker="o", ms=6,
                        capsize=3, label=policy, markeredgecolor=SURFACE, markeredgewidth=1.5)
            plotted = True
    ax.set_xlabel("lambda_sem (lambda_pix = 1.0)", color=MUTED)
    ax.set_ylabel("mAP (val100, 95% CI)", color=MUTED)
    ax.set_title("PiSA-SR: lambda_sem vs mAP", color=INK, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if plotted:
        ax.legend(frameon=False, title="preup_policy")
    fig.tight_layout()
    fig.savefig(cfg["_results"] / "dose_response.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    build_summary(cfg)
    build_grid(cfg)
    build_dose(cfg)
    print("리포트 산출물 작성 완료", flush=True)


if __name__ == "__main__":
    main()
