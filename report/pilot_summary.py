"""PILOT_SUMMARY.md 작성. LR-PSNR, 신원 보존 코사인, 이미지 단위 AUC는 파일럿 결과를 본 뒤 추가한 사후 지표다.

새로 읽는 것은 SR 출력, val query LR, 저장된 HR 모델 특징뿐이다. SR과 학습은 다시 돌리지 않으며 GPU도 쓰지 않는다.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.stats import rankdata, spearmanr

from common.util import ROOT, load_cfg, read_csv, write_csv
from eval.correlate import representatives
from sr.run import cond_dir, conditions

METRICS = [("psnr", "PSNR"), ("ssim", "SSIM"), ("neg_lpips", "−LPIPS"), ("lr_psnr", "LR-PSNR"), ("id_cos", "신원 코사인")]
HEADER = "구 설계(합성 ×4 LR, HR 고정 모델, val 100명, seed 1) 탐색 결과. 본 실험 결론에 쓰지 않음."
IMPLICATIONS = [
    "- 측정: 신원 코사인은 PiSA-SR λ=1.0(`official` 0.672)과 OSDHuman(`fixed8x` 0.671, `official` 0.665)이 높고, bicubic(0.591)과 Real-ESRGAN(`official` 0.591)이 낮았습니다.",
    "- 측정: 이미지 단위 AUC 평균은 PSNR 0.53, LPIPS 0.59로 낮아 픽셀·지각 충실도만으로 SR을 고르기 어렵습니다. 신원 코사인은 0.75지만 ReID 평가와 같은 모델의 특징이라 높게 나오기 쉽습니다.",
    "- 추측: `native`처럼 작은 입력을 그대로 확산 SR에 넣으면 크게 나빠지므로 새 설계도 입력 크기를 고정해야 할 수 있습니다. 합성 ×4 결과라 ×2·×3 설계로 옮겨진다는 근거는 없습니다.",
]


def y_channel(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size[1], size[0]), Image.BICUBIC, reducing_gap=None)  # data/make_lr.py와 같은 축소
    x = np.asarray(img, np.float64) / 255
    return 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2] + 16


def lr_psnr(sr_path, lr_path, size):
    mse = np.mean((y_channel(sr_path, size) - y_channel(lr_path)) ** 2)
    return float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else float("inf")


def per_image(cfg, query):
    """조건별 이미지 표. 계산이 끝난 CSV가 있으면 다시 읽기만 한다."""
    out = cfg["_results"] / "pilot_posthoc_per_image.csv"
    if out.exists():
        rows = read_csv(out)
        for r in rows:
            for k in ("psnr", "ssim", "lpips", "lr_psnr", "id_cos", "ap", "r1"):
                r[k] = float(r[k])
        return rows
    size = tuple(cfg["lr"]["lr_size"])
    lr_dir = cfg["_processed"] / "val_query_lr"
    hr = np.load(cfg["_run"] / "features" / "hr" / "hr_oracle.npz")["feats"]
    rows = []
    with ThreadPoolExecutor(16) as pool:
        for c in conditions(cfg):
            cid, d = c["cid"], cond_dir(cfg, c)
            fid = {r["name"]: r for r in read_csv(cfg["_run"] / "eval" / "fidelity" / f"{cid}.csv")}
            pq = {r["name"]: r for r in read_csv(cfg["_run"] / "eval" / "per_query" / f"{cid}.csv")}
            feats = np.load(cfg["_run"] / "features" / "hr" / f"{cid}.npz")["feats"]
            cos = (feats * hr).sum(1) / np.maximum(np.linalg.norm(feats, axis=1) * np.linalg.norm(hr, axis=1), 1e-12)
            names = [f"{r['name']}.png" for r in query]
            psnrs = pool.map(lambda n: lr_psnr(d / "final" / n, lr_dir / n, size), names)
            for r, n, v, s in zip(query, names, psnrs, cos):
                rows.append({"condition": cid, "name": r["name"], "pid": r["pid"], "psnr": float(fid[n]["psnr"]), "ssim": float(fid[n]["ssim"]),
                             "lpips": float(fid[n]["lpips"]), "lr_psnr": v, "id_cos": float(s), "ap": float(pq[r["name"]]["ap"]), "r1": float(pq[r["name"]]["r1"])})
            print(f"{cid}: 사후 지표 계산", flush=True)
    write_csv(out, rows, ["condition", "name", "pid", "psnr", "ssim", "lpips", "lr_psnr", "id_cos", "ap", "r1"])
    return rows


def score(row, metric):
    return -row["lpips"] if metric == "neg_lpips" else row[metric]


def auc(s, y):
    """y=1(Rank-1 성공)의 점수가 y=0보다 높을 확률. 동점은 평균 순위로 처리한다."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    return float((rankdata(s)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auc_table(cfg, rows):
    """조건별 AUC와 query ID 단위 부트스트랩 CI. 단위, B, seed는 파일럿 평가 설정을 그대로 쓴다."""
    bs = cfg["eval"]["bootstrap"]
    q = (1 - bs["ci"]) / 2 * 100
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    out = []
    for cid, part in by.items():
        pids = np.array([r["pid"] for r in part])
        y = np.array([r["r1"] for r in part])
        groups = [np.flatnonzero(pids == p) for p in np.unique(pids)]
        rng = np.random.default_rng(bs["seed"])
        draws = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]) for _ in range(bs["B"])]
        for m, _ in METRICS:
            s = np.array([score(r, m) for r in part])
            boot = np.array([auc(s[i], y[i]) for i in draws])
            out.append({"condition": cid, "metric": m, "auc": auc(s, y), "ci_lo": float(np.nanpercentile(boot, q)),
                        "ci_hi": float(np.nanpercentile(boot, 100 - q)), "n_hit": int(y.sum()), "n_miss": int((1 - y).sum())})
    write_csv(cfg["_results"] / "pilot_posthoc_auc.csv", out, ["condition", "metric", "auc", "ci_lo", "ci_hi", "n_hit", "n_miss"])
    return out


def method_means(rows):
    means = {}
    for r in rows:
        means.setdefault(r["condition"], []).append(r)
    return {c: {**{m: float(np.mean([score(r, m) for r in p])) for m, _ in METRICS}, "lpips": float(np.mean([r["lpips"] for r in p])),
                "map": float(np.mean([r["ap"] for r in p])), "r1": float(np.mean([r["r1"] for r in p]))} for c, p in means.items()}


def spearman_table(cfg, means):
    out = []
    for policy in cfg["sr"]["preup_policies"]:
        reps = [c["cid"] for c in representatives(cfg, policy)]
        row = {"preup_policy": policy, "n_methods": len(reps)}
        for m, _ in METRICS:
            row[f"rho_{m}_map"] = float(spearmanr([means[c][m] for c in reps], [means[c]["map"] for c in reps]).statistic)
        out.append(row)
    write_csv(cfg["_results"] / "pilot_posthoc_spearman.csv", out, ["preup_policy", "n_methods"] + [f"rho_{m}_map" for m, _ in METRICS])
    return out


def label(c):
    return c["method"] + (f" λ={c['lambda_sem']}" if c["lambda_sem"] is not None else "")


def f(v, d=3):
    return "-" if v in (None, "") else f"{float(v):.{d}f}"


def build(cfg, rows, aucs, spear):
    means = method_means(rows)
    summary = read_csv(cfg["_results"] / "pilot_summary.csv")
    sm = {(r["method"], r["lambda_sem"], r["preup_policy"]): r for r in summary}
    conds = conditions(cfg)
    L = ["# 파일럿 결과 요약 (pilot_001)", "", f"**{HEADER}**", "",
         "값은 `results/pilot_summary.csv`, `runs/pilot_001/`, `results/pilot_posthoc_*.csv`에서 계산했습니다. 공식 test split은 쓰지 않았습니다. "
         "val query는 404장입니다. SD2.1-base 사본은 공식 해시 조회가 실패해(HTTP 401) 미검증 상태이며, PiSA-SR과 OSDHuman 수치는 그 전제 위에 있습니다.", ""]

    L += ["## 1. SR 처리 시간", "", "exclusive_gpu=true 측정(조건당 50장)만 씁니다. 27개 조건 모두 해당합니다. 모델 로드 시간은 제외했습니다. 피크 VRAM은 PyTorch `max_memory_allocated`이며 bicubic, lanczos는 CPU 처리라 0입니다.", "",
          "| 방법 | 초/장 official | 초/장 native | 초/장 fixed8x | 피크 VRAM GiB official | native | fixed8x |", "|---|---:|---:|---:|---:|---:|---:|"]
    bases = list(dict.fromkeys(label(c) for c in conds))
    for b in bases:
        t = {}
        for c in conds:
            if label(c) == b:
                j = json.loads((cond_dir(cfg, c) / "timing.json").read_text())
                assert j["exclusive_gpu"] is True, c["cid"]
                t[c["policy"]] = (j["sec_per_image_exclusive"], j["max_memory_allocated_exclusive"] / 2**30)
        L.append(f"| {b} | {t['official'][0]:.4f} | {t['native'][0]:.4f} | {t['fixed8x'][0]:.4f} | {t['official'][1]:.2f} | {t['native'][1]:.2f} | {t['fixed8x'][1]:.2f} |")
    L += ["", "PiSA-SR의 피크 VRAM은 15개 조건 모두 6.35GiB로 같습니다. 입력 크기가 달라도 같으므로 추론이 아니라 모델 로드 시점의 피크일 수 있습니다. 확인하지는 않았습니다.", ""]

    L += ["## 2. SR 품질", "", "이미지별 값의 평균입니다. PSNR, SSIM, LPIPS는 파일럿 사전 지표(HR 대비, PSNR·SSIM은 Y 채널, LPIPS는 alex)입니다.", "",
          "**사후 지표(결과를 본 뒤 계산, 사전 등록 아님):**",
          "- LR-PSNR: SR 출력을 LR 생성과 같은 PIL BICUBIC으로 32×16까지 줄여 val query LR과 비교한 Y 채널 PSNR입니다.",
          "- 신원 코사인: HR 고정 BoT 모델에서 SR 출력 특징과 같은 이미지의 HR 원본 특징 사이 코사인입니다. 파일럿에 저장된 특징을 그대로 썼습니다.",
          "- **속성 일치율: 미측정.** 속성 분류 모델과 속성 라벨이 파일럿에도 로컬에도 없습니다. 값을 대체 지표로 채우지 않았습니다.", "",
          "| 방법 | 정책 | PSNR | SSIM | LPIPS | LR-PSNR | 신원 코사인 | 속성 일치율 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for c in conds:
        m = means[c["cid"]]
        L.append(f"| {label(c)} | {c['policy']} | {m['psnr']:.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['lr_psnr']:.2f} | {m['id_cos']:.4f} | 미측정 |")
    L += ["", "PiSA-SR은 `official`과 `fixed8x`의 값이 표에서 같게 보입니다. 바이트는 달라도 λ=1.0 기준 평균 절대 픽셀 차이가 0.03/255에 그칩니다(OSDHuman은 0.41/255).", ""]

    L += ["## 3. ReID 결과", "", f"**{HEADER}**", "",
          "CI는 query ID 단위 부트스트랩 95%(B=2000)입니다. 판정은 mAP 차이의 CI 기준이며, 파일럿의 사전 등록 규칙을 그대로 옮겼습니다. 주 기준선은 `path1_mixed`, 보조 기준선은 `path1_lr_only`입니다.", "",
          "| 방법 | 정책 | Rank-1 | Rank-1 95% CI | mAP | mAP 95% CI | vs mixed | vs lr_only | vs bicubic |", "|---|---|---:|---|---:|---|---|---|---|"]
    for r in summary:
        name = r["method"] + (f" λ={r['lambda_sem']}" if r["lambda_sem"] else "")
        L.append(f"| {name} | {r['preup_policy']} | {f(r['rank1'])} | [{f(r['rank1_ci_lo'])}, {f(r['rank1_ci_hi'])}] | {f(r['map'])} | [{f(r['map_ci_lo'])}, {f(r['map_ci_hi'])}] | "
                 f"{r['verdict_vs_path1_mixed'] or '-'} | {r['verdict_vs_path1_lr_only'] or '-'} | {r['verdict_vs_bicubic'] or '-'} |")
    L.append("")

    L += ["## 4. 품질 지표와 ReID 성능의 상관", "", "전부 사후 분석의 기술 통계이며 다중 비교를 보정하지 않았습니다. 신원 코사인은 ReID 평가와 같은 모델의 특징에서 나오므로, 이 지표의 상관이 높은 것은 어느 정도 예상되는 결과입니다.", "",
          "### 방법 단위 Spearman ρ (대상: 평균 품질 지표와 mAP)", "", "대표 방법 5개(PiSA-SR은 λ=1.0)뿐이라 p값은 내지 않습니다. LPIPS는 부호를 뒤집어 높을수록 좋게 맞췄습니다.", "",
          "| 정책 | 방법 수 | " + " | ".join(n for _, n in METRICS) + " |", "|---|---:|" + "---:|" * len(METRICS)]
    for r in spear:
        L.append(f"| {r['preup_policy']} | {r['n_methods']} | " + " | ".join(f"{r[f'rho_{m}_map']:+.2f}" for m, _ in METRICS) + " |")
    idx = {(a["condition"], a["metric"]): a for a in aucs}
    L += ["", "### 이미지 단위 AUC", "", "조건 안에서 품질 점수로 해당 query의 Rank-1 성공과 실패를 구분하는 AUC입니다. 0.5는 구분 못 함을 뜻합니다. 괄호는 query ID 단위 부트스트랩 95% CI입니다. 아래 표는 `official` 정책의 대표 방법입니다. 27개 조건 전체는 `results/pilot_posthoc_auc.csv`에 있습니다.", "",
          "| 방법 | 성공/실패 | " + " | ".join(n for _, n in METRICS) + " |", "|---|---:|" + "---|" * len(METRICS)]
    for c in representatives(cfg, "official"):
        a0 = idx[(c["cid"], "psnr")]
        L.append(f"| {label(c)} | {a0['n_hit']}/{a0['n_miss']} | " + " | ".join(
            f"{idx[(c['cid'], m)]['auc']:.2f} ({idx[(c['cid'], m)]['ci_lo']:.2f}, {idx[(c['cid'], m)]['ci_hi']:.2f})" for m, _ in METRICS) + " |")
    all_mean = {m: np.nanmean([a["auc"] for a in aucs if a["metric"] == m]) for m, _ in METRICS}
    L += [f"| 27개 조건 평균 | | " + " | ".join(f"{all_mean[m]:.2f}" for m, _ in METRICS) + " |", ""]

    L += ["## 5. 새 설계에 시사하는 점", ""] + IMPLICATIONS + [""]
    (ROOT / "PILOT_SUMMARY.md").write_text("\n".join(L))
    print("PILOT_SUMMARY.md 작성", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    query = sorted((r for r in read_csv(cfg["_manifests"] / "split.csv") if r["role"] == "query"), key=lambda r: r["name"])
    rows = per_image(cfg, query)
    aucs = auc_table(cfg, rows)
    means = method_means(rows)
    spear = spearman_table(cfg, means)
    build(cfg, rows, aucs, spear)


if __name__ == "__main__":
    main()
