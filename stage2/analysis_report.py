"""Stage 2 사후 분석 보고서: STAGE2_ANALYSIS.md, STAGE2_ANALYSIS.html(그림 축소 JPEG 내장). srreid-quality env.
입력은 results/stage2_analysis/의 집계 표와 그림만(잠긴 영역의 query별 원시 값은 읽지 않는다)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
from report import Doc  # noqa: E402

RES = ROOT / "results" / "stage2_analysis"
FIG = RES / "figures"
SR = ["swinir", "pisa_sr_sem0.0", "osdhuman"]
CONDS = ["bicubic"] + SR
MODELS = ["bot", "clipreid", "kpr", "kpr_solider"]
MNAME = {"bot": "BoT", "clipreid": "CLIP-ReID", "kpr": "KPR", "kpr_solider": "KPR+SOLIDER"}
SNAME = {"bicubic": "bicubic", "swinir": "swinir", "pisa_sr_sem0.0": "pisa λ0.0", "osdhuman": "osdhuman"}
METRICS = ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "hf_db_vs_bicubic", "slope_dev_hi", "grid_peak_db", "de2000"]
MLAB = {"dinov3_large_cos": "DINOv3 코사인 (높을수록 원본과 가까움)", "personvit_base_cos": "PersonViT 코사인 (높을수록 원본과 가까움)", "afine_all_scale": "A-FINE 총점 (낮을수록 좋음)",
        "fidelity": "A-FINE 충실도 (낮을수록 좋음)", "natural_dis": "A-FINE 자연스러움 (낮을수록 좋음)", "lf_coh": "저주파 일치도 (높을수록 일치)",
        "hf_db_vs_bicubic": "고주파 추가량 dB (방향 없음)", "slope_dev_hi": "스펙트럼 기울기 편차 (0이 원본)", "grid_peak_db": "격자 피크 dB (낮을수록 약함)", "de2000": "ΔE2000 (낮을수록 색 차이 작음)"}


def f4(v):
    return f"{v:.4g}"


def main():
    d = Doc()
    d.h(1, "Stage 2 사후 분석: 기준선 재확인, test SR 품질 지표, query 단위 분석, 성공·실패 사례")
    d.p("**평가 완료 후 사후 분석이며 Stage 2 판정(`STAGE2_REPORT.md`, `results/stage2/judgment.csv`)과 별개다. 판정은 바꾸지 않았다.** "
        "서버 A100 SXM4 80GB ×2(EPYC 7713, cgroup v2), 2026-11-12. test 접근은 사용자 잠금 해제 창에서만 했고 `logs/test_access.log`에 기록했다. "
        "query별 원시 값과 특징은 잠긴 영역 `data/locked/stage2_analysis/`, 집계·그림은 `results/stage2_analysis/`.")
    d.raw_html('<nav><a href="#b">1 기준선</a><a href="#d">2b 분포</a><a href="#q">2c query 분석</a><a href="#s">2d 방법 요약</a><a href="#c">3 사례</a><a href="#l">한계</a></nav>')

    # 1
    d.h(2, "1. 기준선 재확인", "b")
    rows = [json.loads((RES / f"baseline_recheck_{m}.json").read_text()) for m in MODELS]
    t = pd.DataFrame([{"모델": MNAME[r["model"]], "입력": r["input"], "이번 mAP / R1 (A100 재추출)": f"{r['mAP']:.2f} / {r['rank1']:.2f}",
                       "Stage 0.5 (4090)": f"{r['ref_mAP']:.2f} / {r['ref_rank1']:.2f}", "소수 둘째 자리 일치": "통과" if r["pass"] else "불일치"} for r in rows])
    d.table(t, "공개 가중치, 공식 입력 크기, 원본 test(query 3,368, gallery 15,913). 특징은 Stage 0.5 기준선 추출기로 새로 뽑고(공개 BoT는 공식 reid-strong-baseline 구조라 Stage 2 자체 BoT 추출기로 읽을 수 없음), 지표는 Stage 2 평가 코드(`stage2/evaluate.py` make_dist + `eval_core.stream_eval`).")
    eq = json.loads((RES / "baseline_recheck_cause.json").read_text())
    d.ul(eq["bullets"])

    # 2b
    d.h(2, "2b. Stage 1 train 표본 대 test 지표 분포", "d")
    d.p("test 지표는 Stage 1과 같은 모델·전처리·설정(원본 test 이미지 대비)으로 {bicubic, swinir, pisa λ0.0, osdhuman} × {×3, ×4} × (query 3,368 + gallery 19,732)에 계산했다. "
        "분포 비교는 사람 이미지(pid > 0, junk·방해 이미지 제외) 기준, 집계는 Stage 1과 같다(이미지 → 신원 평균 → 신원 평균의 평균, 신원 단위 부트스트랩 95% CI 10,000회, seed 42). bicubic의 A-FINE은 기준 이미지와 같아 비교 불가.")
    D = pd.read_csv(RES / "train_vs_test_metrics.csv")
    for mtr in METRICS:
        x = D[D.metric == mtr]
        if not len(x):
            continue
        t = pd.DataFrame([{"방법": SNAME[r.cond], "배율": r.k, "train 표본 평균 [CI]": f"{f4(r.train_mean)} [{f4(r.train_ci_lo)}, {f4(r.train_ci_hi)}]",
                           "test 평균 [CI]": f"{f4(r.test_mean)} [{f4(r.test_ci_lo)}, {f4(r.test_ci_hi)}]", "test − train": f"{r.test_minus_train:+.4g}",
                           "CI 겹침": "예" if r.ci_overlap else "아니오"} for r in x.itertuples()])
        d.table(t, MLAB[mtr])
    n_no = int((~D.ci_overlap).sum())
    d.p(f"train 표본과 test의 CI가 겹치지 않는 조합: {n_no}/{len(D)}. 차이의 크기는 'test − train' 열, 방향은 지표 설명을 따른다.")
    for mtr in METRICS:
        d.img(FIG / f"dist_train_test_{mtr}.png", f"{MLAB[mtr]}: train 표본(파랑) 대 test(주황) 이미지 분포")

    # 2c
    d.h(2, "2c. query 단위 분석: ΔAP_q와 지표의 Spearman 상관", "q")
    d.p("ΔAP_q = (SR로 학습한 모델 + SR test의 query AP) − (bicubic 모델 + bicubic test의 query AP). 설명 변수는 그 query SR 이미지의 지표 값, 그리고 같은 신원·다른 카메라 정답 gallery SR 이미지들의 지표 평균. "
        "ρ는 query 단위와 신원 단위 평균(같은 사람의 query끼리 상관된 문제를 줄임) 두 가지, 95% CI는 신원 단위 부트스트랩(10,000회, seed 42, 복원추출 횟수를 가중치로 한 가중 Spearman).")
    S = pd.read_csv(RES / "spearman_dAP_metrics.csv")
    rows = []
    for mtr in METRICS:
        for src in ("query 이미지", "정답 gallery 평균"):
            x = S[(S.metric == mtr) & (S.source == src)]
            if not len(x):
                continue
            rows.append({"지표": MLAB[mtr], "설명 변수": src, "조합 수": len(x),
                         "query ρ 중앙값 [최소, 최대]": f"{x.rho_query.median():+.3f} [{x.rho_query.min():+.3f}, {x.rho_query.max():+.3f}]",
                         "query ρ CI>0 / CI<0": f"{int((x.ci_lo_query > 0).sum())} / {int((x.ci_hi_query < 0).sum())}",
                         "신원 ρ 중앙값 [최소, 최대]": f"{x.rho_id.median():+.3f} [{x.rho_id.min():+.3f}, {x.rho_id.max():+.3f}]",
                         "신원 ρ CI>0 / CI<0": f"{int((x.ci_lo_id > 0).sum())} / {int((x.ci_hi_id < 0).sum())}"})
    d.table(pd.DataFrame(rows), "지표별 요약(모델 4 × 배율 2 × SR 3 = 최대 24조합에 걸친 ρ 분포). 'CI>0 / CI<0'은 95% CI가 0을 벗어난 조합 수(보정하지 않음).")
    for m in MODELS:
        x = S[(S.model == m) & (S.source == "query 이미지")]
        t = x.pivot_table(index="metric", columns=["k", "sr"], values="rho_query").reindex(METRICS)
        t.columns = [f"x{k} {SNAME[s]}" for k, s in t.columns]
        t = t.map(lambda v: f"{v:+.3f}" if pd.notna(v) else "").reset_index().rename(columns={"metric": "지표"})
        t["지표"] = t["지표"].map(MLAB)
        d.table(t, f"{MNAME[m]}: query 이미지 지표와 ΔAP_q의 Spearman ρ(query 단위). CI는 `results/stage2_analysis/spearman_dAP_metrics.csv`.")
    for mtr in METRICS:
        d.img(FIG / f"decile_dAP_{mtr}.png", f"query 이미지 {MLAB[mtr]} 10분위별 평균 ΔAP_q")

    # 2d
    d.h(2, "2d. 방법 단위 요약 (상관 계수 없음)", "s")
    M = pd.read_csv(RES / "method_summary.csv")
    d.table(M, "Stage 1 지표 평균(train 표본 2,250장, 신원 평균)과 Stage 2 ΔmAP(S 모델 + SR test − C 모델 + bicubic test, 판정). SR이 3개뿐이라 상관 계수는 내지 않았다.")

    # 3
    d.h(2, "3. 성공·실패 사례", "c")
    d.p("SR 조건·배율마다 4개 모델 ΔAP_q 평균(4모델 모두 유효한 query)으로 가장 손해 본 20개와 이득 본 20개. 그림 한 줄: 원본 query(최근접 확대) | bicubic query | SR query | BoT C top-5 | BoT S top-5 | KPR+SOLIDER C top-5 | KPR+SOLIDER S top-5. "
        "C = bicubic 모델이 bicubic gallery에서, S = SR 모델이 SR gallery에서 찾은 결과. 테두리 초록 = 정답(같은 신원), 빨강 = 오답. Market 규칙대로 같은 신원·같은 카메라 gallery는 제외.")
    C = pd.read_csv(RES / "cases.csv")
    rows = []
    for s in SR:
        for k in (3, 4):
            for kind in ("worst", "best"):
                x = C[(C.sr == s) & (C.k == k) & (C.kind == kind)]
                r = {"SR": SNAME[s], "배율": k, "구분": "손해" if kind == "worst" else "이득", "평균 ΔAP_q %p": f"{100 * x.mean_dAP.mean():+.1f}"}
                for mtr in ("dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "lf_coh", "de2000"):
                    r[MLAB[mtr].split(" (")[0]] = f4(x[mtr].mean())
                rows.append(r)
    d.table(pd.DataFrame(rows), "사례 20개의 query SR 이미지 지표 평균(전체 query별 값은 `results/stage2_analysis/cases.csv`).")
    for s in SR:
        for k in (3, 4):
            for kind in ("worst", "best"):
                d.img(FIG / "cases" / f"{s}_x{k}_{kind}.png", f"×{k} {SNAME[s]}: {'가장 손해 본' if kind == 'worst' else '가장 이득 본'} query 20개")

    # 한계
    d.h(2, "한계", "l")
    d.ul(["사후 분석이며 Stage 2 판정과 별개다. 상관은 인과가 아니다.",
          "다중 비교: 지표 10 × 설명 변수 2 × 모델 4 × 배율 2 × SR 3 = 480개 상관을 보정 없이 보고한다. 개별 CI가 0을 벗어나는 것만으로 의미를 두지 않는다.",
          "seed 1개: ΔAP_q에는 두 모델(S, C) 학습 변동이 그대로 들어 있다. 특히 KPR+SOLIDER ×3 bicubic 모델과 KPR ×3 swinir 모델은 약하게 수렴해 그 조합의 ΔAP_q 분포 전체가 한쪽으로 치우친다(`STAGE2_REPORT.md` 한계).",
          "query AP는 0~1 사이에 몰리고 동점(둘 다 AP 1)이 많아 ρ의 크기가 작게 나오기 쉽다. 동점은 중간 순위로 처리했다.",
          "정답 gallery 지표 평균은 같은 신원·다른 카메라 이미지만(Market 규칙) 사용했다.",
          "train 표본은 2,250장(751명), test는 사람 이미지 전체(750명)로 표본 크기가 다르다. 두 분포의 차이는 데이터 분할(신원·장면 차이)의 영향을 포함한다.",
          "기준선 재확인의 미세 차이(CLIP-ReID, KPR)는 GPU 종류(fp16·커널)에 따른 특징 차이이며 Stage 2 셀은 모두 같은 A100에서 추출했다.",
          "사례 그림은 Claude가 열어 보지 않았다(수치로만 생성·검사)."])
    d.write(ROOT / "STAGE2_ANALYSIS.md", ROOT / "STAGE2_ANALYSIS.html", "Stage 2 사후 분석")
    for p in ("STAGE2_ANALYSIS.md", "STAGE2_ANALYSIS.html"):
        print(p, f"{(ROOT / p).stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
