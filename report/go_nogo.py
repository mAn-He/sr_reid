"""사전 등록한 go/no-go 기준을 그대로 계산해 REPORT.md를 쓴다."""
import argparse
import itertools
import json

import numpy as np
from scipy.stats import kendalltau

from common.util import ROOT, load_cfg, read_csv
from eval.correlate import representatives
from sr.run import cond_dir, conditions


def load_boot(cfg):
    p = cfg["_run"] / "eval" / "boot.npz"
    return dict(np.load(p)) if p.exists() else {}


def broken_rates(cfg, n_query):
    """누락, 디코딩 실패, 크기 불일치, Y 표준편차 또는 PSNR 기준 미달을 깨진 이미지로 친다."""
    g = cfg["go_nogo"]
    out = {}
    for c in conditions(cfg):
        if c["kind"] not in ("pisa", "osd"):
            continue
        o = cond_dir(cfg, c) / "outputs.csv"
        f = cfg["_run"] / "eval" / "fidelity" / f"{c['cid']}.csv"
        if not o.exists():
            out[c["cid"]] = None
            continue
        ok = {r["name"] for r in read_csv(o) if r["status"] == "ok"}
        bad = n_query - len(ok)
        if f.exists():
            bad += sum(float(r["y_std"]) < g["broken_min_y_std"] or float(r["psnr"]) < g["broken_min_psnr_db"]
                       for r in read_csv(f))
        out[c["cid"]] = bad / n_query
    return out


def item1(cfg, n_query):
    rates = broken_rates(cfg, n_query)
    lines = ["| 조건 | 깨진 비율 |", "|---|---|"]
    lines += [f"| {k} | {'출력 없음' if v is None else f'{v:.4f}'} |" for k, v in rates.items()]
    official = {k: v for k, v in rates.items() if k.endswith("__official")}
    if not official or any(v is None for v in official.values()):
        return "판정 불가", "확산 모델 `official` 출력이 없는 조건이 있습니다.\n\n" + "\n".join(lines)
    go = all(v <= cfg["go_nogo"]["broken_rate_max"] for v in official.values())
    note = f"기준: 확산 모델별 `official` 깨진 비율 {cfg['go_nogo']['broken_rate_max']} 이하. 수치 기준이며, 육안 확인은 grid.png로 합니다."
    return ("GO" if go else "NO-GO"), note + "\n\n" + "\n".join(lines)


def item2(cfg, boot):
    q = (1 - cfg["eval"]["bootstrap"]["ci"]) / 2 * 100
    policies = cfg["sr"]["preup_policies"]
    point = {r["condition"]: float(r["estimate"]) for r in read_csv(cfg["_run"] / "eval" / "bootstrap.csv")
             if r["metric"] == "map" and r["comparator"] == ""}
    reps = {p: {c["base"]: c["cid"] for c in representatives(cfg, p) if c["cid"] in point} for p in policies}
    lines = ["| 정책 쌍 | 공통 방법 수 | Kendall τ | 뒤집힌 쌍 | CI가 0을 벗어난 뒤집힘 |", "|---|---:|---:|---:|---:|"]
    any_flip = any_sig = False
    compared = 0
    for a, b in itertools.combinations(policies, 2):
        common = sorted(set(reps[a]) & set(reps[b]))
        if len(common) < 2:
            lines.append(f"| {a} / {b} | {len(common)} | 비교 불가 | | |")
            continue
        compared += 1
        xa, xb = [point[reps[a][m]] for m in common], [point[reps[b][m]] for m in common]
        tau = kendalltau(xa, xb).statistic
        flips = sig = 0
        for m, n in itertools.combinations(common, 2):
            da, db = point[reps[a][m]] - point[reps[a][n]], point[reps[b][m]] - point[reps[b][n]]
            if da * db < 0:
                flips += 1
                for p in (a, b):
                    d = boot[f"{reps[p][m]}|ap"] - boot[f"{reps[p][n]}|ap"]
                    lo, hi = np.percentile(d, [q, 100 - q])
                    if lo > 0 or hi < 0:
                        sig += 1
                        break
        any_flip, any_sig = any_flip or flips > 0, any_sig or sig > 0
        lines.append(f"| {a} / {b} | {len(common)} | {tau:.3f} | {flips} | {sig} |")
    if compared == 0:
        return "판정 불가", "비교할 공통 방법이 부족합니다.\n\n" + "\n".join(lines)
    verdict = "변함" if any_sig else ("불확실" if any_flip else "변하지 않음")
    note = "정책별 출력이 바이트 단위로 같은 조합은 아래 정책 동일성 표에 있습니다."
    if verdict == "변함":
        note = "본 실험에서 선확대 정책을 고정 변수로 사전 등록해야 합니다. " + note
    return verdict, note + "\n\n" + "\n".join(lines)


def item3(cfg):
    n = cfg["go_nogo"]["official_query_count"]
    lines = ["| 조건 | 초/장 (exclusive_gpu=true) | query당 외삽 시간(시간) |", "|---|---:|---:|"]
    total, missing = 0.0, []
    for c in conditions(cfg):
        p = cond_dir(cfg, c) / "timing.json"
        sec = json.loads(p.read_text()).get("sec_per_image_exclusive") if p.exists() else None
        if sec is None:
            missing.append(c["cid"])
            lines.append(f"| {c['cid']} | 측정값 없음 | |")
            continue
        total += sec * n
        lines.append(f"| {c['cid']} | {sec:.4f} | {sec * n / 3600:.3f} |")
    head = f"공식 query {n}장(문헌값, 파일은 읽지 않음) × 조건 {len(conditions(cfg)) - len(missing)}개 기준 합계 **{total / 3600:.2f}시간**. 모델 로드 시간은 제외했습니다."
    if missing:
        head += f" 단독 측정값이 없는 조건 {len(missing)}개는 합계에서 빠졌으므로 실제 소요는 더 깁니다."
    return ("계산됨" if not missing else "부분 계산"), head + "\n\n" + "\n".join(lines)


def item4(cfg, boot):
    q = (1 - cfg["eval"]["bootstrap"]["ci"]) / 2 * 100
    sems = cfg["sr"]["methods"]["pisa_sr"]["lambda_sem"]
    lines = ["| 정책 | 점추정 단조 감소 | mAP(λ=최대) − mAP(λ=0) | 95% CI | 신호 |", "|---|---|---:|---|---|"]
    verdict = "판정 불가"
    for policy in cfg["sr"]["preup_policies"]:
        keys = [f"pisa_sr_sem{s}__{policy}|ap" for s in sems]
        if not all(k in boot for k in keys):
            lines.append(f"| {policy} | 출력 부족 | | | 판정 불가 |")
            continue
        est = [float(np.mean(boot[k])) for k in keys]
        mono = all(b < a for a, b in zip(est, est[1:]))
        d = boot[keys[-1]] - boot[keys[0]]
        lo, hi = np.percentile(d, [q, 100 - q])
        signal = "신호 있음" if mono and hi < 0 else "신호 없음"
        lines.append(f"| {policy} | {'예' if mono else '아니오'} | {est[-1] - est[0]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {signal} |")
        if policy == "official":
            verdict = signal
    return verdict, "주 판정은 `official` 정책입니다. 가설 신호일 뿐 결론이 아닙니다. 점추정은 부트스트랩 평균입니다.\n\n" + "\n".join(lines)


def identity_table(cfg):
    lines = ["| 방법 | 정책 쌍 | 바이트 동일 / 공통 |", "|---|---|---:|"]
    by = {}
    for c in conditions(cfg):
        p = cond_dir(cfg, c) / "outputs.csv"
        if p.exists():
            by.setdefault(c["base"], {})[c["policy"]] = {r["name"]: r["final_sha256"] for r in read_csv(p) if r["status"] == "ok"}
    for base, pol in by.items():
        for a, b in itertools.combinations(sorted(pol), 2):
            common = set(pol[a]) & set(pol[b])
            lines.append(f"| {base} | {a} / {b} | {sum(pol[a][n] == pol[b][n] for n in common)} / {len(common)} |")
    return "\n".join(lines)


def section(title, res):
    return f"### {title}\n\n**판정: {res[0]}**\n\n{res[1]}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    rows = read_csv(cfg["_manifests"] / "split.csv")
    n_query = sum(r["role"] == "query" for r in rows)
    excluded = sum(r["role"] == "excluded_query" for r in rows)
    boot = load_boot(cfg)
    have_boot = (cfg["_run"] / "eval" / "bootstrap.csv").exists()
    na = ("판정 불가", "부트스트랩 결과가 없습니다.")
    sd = ROOT / "weights" / "sd21_verification.json"
    sd_status = json.loads(sd.read_text())["status"] if sd.exists() else "기록 없음"
    ident = cfg["_run"] / "path1_bicubic_identity.json"
    ident_txt = json.loads(ident.read_text()) if ident.exists() else None
    text = [f"# 파일럿 리포트 ({cfg['run_id']})\n",
            "모든 수치는 `results/pilot_summary.csv`와 `runs/` 산출물에서 계산했습니다. 공식 test split은 사용하지 않았습니다.\n",
            f"- val query {n_query}장 사용, 다른 카메라 gallery가 없어 제외한 query {excluded}장.",
            f"- 주 기준선은 `path1_mixed`, 보조 기준선은 `path1_lr_only` (결과 확인 전 등록).",
            f"- Path 1 입력과 bicubic official 출력의 바이트 동일: {ident_txt}.",
            f"- SD2.1-base 사본 상태: **{sd_status}**.\n",
            "## go/no-go 판정\n",
            section("1. 확산 계열 `official` 출력 안정성", item1(cfg, n_query)),
            section("2. 선확대 정책에 따른 mAP 순위 변화", item2(cfg, boot) if have_boot else na),
            section("3. 본 실험 소요 외삽", item3(cfg)),
            section("4. lambda_sem 증가에 따른 mAP 단조 감소", item4(cfg, boot) if have_boot else na),
            "## 정책 간 출력 동일성\n", identity_table(cfg), "",
            "## 한계\n",
            "- 신뢰구간은 query 표본 변동만 반영합니다. 파일럿의 ReID 학습 seed는 하나입니다.",
            "- 본 실험에서는 HR ReID 모델 3 seed로 학습 변동을 반영할 예정 (SR 출력은 재사용, 특징 추출만 반복).",
            "- HR은 상대적 정의(Market 원본 128×64)이며, LR은 bicubic 합성 축소로 실제 카메라 열화와 다르다. 실제 LR 검증은 별도 데이터셋으로 후속 단계에서 수행.",
            "- 다중 비교를 보정하지 않았습니다.",
            "- Spearman ρ는 대표 방법 5개 기준의 기술 통계입니다. p값은 내지 않습니다.",
            "- test split을 쓰지 않아 ReID 구현을 BoT 공개 수치와 대조할 수 없습니다. `hr_oracle` 행이 상한선 점검용입니다.",
            "- val gallery는 100 ID뿐이라 mAP가 공식 평가보다 높게 나오며, 공식 수치와 직접 비교할 수 없습니다.",
            "- 깨진 이미지 판정은 수치 기준입니다. 환각 같은 시각 오류는 grid.png 육안 확인이 필요합니다.",
            f"- SD2.1-base 사본 상태는 '{sd_status}'이며 이 결과의 확산 모델 수치는 그 전제 위에 있습니다.\n"]
    (ROOT / cfg["paths"].get("report", "REPORT.md")).write_text("\n".join(text))
    print("REPORT 작성 완료", flush=True)


if __name__ == "__main__":
    main()
