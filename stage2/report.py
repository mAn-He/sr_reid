"""Stage 2 보고서(STAGE2_REPORT.md). srreid-core env. 입력은 results/stage2/*.csv와 runs/stage2/train/*/train_meta.json(잠금 밖 집계만).
test 이미지·이미지별 결과는 읽지 않는다(eval_cells.csv, judgment.csv는 집계값)."""
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "stage2"
TR = ROOT / "runs" / "stage2" / "train"
MODELS = ["bot", "clipreid", "kpr", "kpr_solider"]
MNAME = {"bot": "BoT", "clipreid": "CLIP-ReID", "kpr": "KPR", "kpr_solider": "KPR+SOLIDER"}
SR = ["swinir", "pisa_sr_sem0.0", "osdhuman"]
SNAME = {"bicubic": "bicubic", "swinir": "swinir", "pisa_sr_sem0.0": "pisa λ0.0", "osdhuman": "osdhuman"}
SERVER = "A100 서버(A100 SXM4 80GB ×2, EPYC 7713, cgroup v2)"
PRICE = 3.59


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def pct(v):
    return f"{100 * v:.2f}"


def meta_row(m, c, k):
    d = json.loads((TR / f"{m}__{c}__x{k}" / "train_meta.json").read_text())
    if m == "bot":
        cud = f"det={d['cudnn_deterministic']}, bench={d['cudnn_benchmark']}"
        sec, amp, ckpt, gpu = d["train_seconds"], "없음", d["ckpt"], d["cuda_visible_devices"]
        peak = d["peak_reserved_gib"]
    elif m == "clipreid":
        cc = d["cudnn_after_set_seed"]
        cud = f"det={cc['deterministic']}, bench={cc['benchmark']}"
        sec = sum(d["wall_s"].values())
        amp, ckpt = "fp16 autocast(공식)", d.get("ckpt", 0)
        gpu = d["env"].get("CUDA_VISIBLE_DEVICES") if isinstance(d["env"], dict) else ""
        peak = d["mem"]["stage2"]["peak_reserved_gib"] if isinstance(d["mem"], dict) else ""
    else:
        e = d["env"] if isinstance(d["env"], dict) else {}
        cc = e.get("cudnn", {})
        cud = f"det={cc.get('deterministic', '')}, bench={cc.get('benchmark', '')}"
        sec, amp, ckpt = d["train_seconds"], "mixed_precision(공식)", d["ckpt"]
        gpu = e.get("CUDA_VISIBLE_DEVICES", "")
        peak = d["peak_reserved_gib_this_session"]
    return {"모델": MNAME[m], "학습 조건": SNAME[c], "배율": k, "GPU": gpu, "학습 시간 h(병렬 실행 중)": f"{sec / 3600:.2f}",
            "AMP": amp, "cudnn": cud, "checkpointing": "켬" if ckpt else "끔", "peak 예약 GiB": peak if peak == "" else f"{float(peak):.1f}",
            "재개": len(d.get("resume_events", [])) if m != "bot" else (1 if d.get("resumed_from_epoch") else 0)}


def main():
    cells = pd.read_csv(R / "eval_cells.csv")
    judg = pd.read_csv(R / "judgment.csv")
    q = pd.read_csv(R / "train_queue_log.csv")
    L = ["# Stage 2 보고서: SR 3종 학습·평가 (ReID 4모델 × {bicubic, swinir, pisa λ0.0, osdhuman} × {×3, ×4})", ""]
    L += [f"작성 자동(`stage2/report.py`). 서버 {SERVER}. 모든 시간은 **병렬 실행 중 측정**(GPU 2장에 여러 학습·평가 동시)이다. seed 42 하나.", ""]
    # 1. 생성
    L += ["## 1. 이미지 생성", ""]
    mt, ms = pd.read_csv(R / "manifest_summary_train.csv"), pd.read_csv(R / "manifest_summary_test.csv")
    L += [f"- train: SR 3종 × ×3·×4 × 12,936장. 누락 {int(mt.missing.sum())}, 크기 불일치 {int(mt.wrong_size.sum())}, 로그 SHA256 불일치 {int(mt.log_sha_mismatch.sum())}. bicubic train은 기존 `data/bicubic_x{{3,4}}` 재사용.",
          f"- test: SR 3종 + bicubic × ×3·×4 × (query 3,368 + gallery 19,732). 누락 {int(ms.missing.sum())}, 크기 불일치 {int(ms.wrong_size.sum())}, 로그 SHA256 불일치 {int(ms.log_sha_mismatch.sum())}. 실제 파일은 잠긴 영역 `data/locked/sr/`.",
          "- 설정: Stage 1과 같음(swinir ×3 공식 ×3 가중치, pisa·osdhuman ×3 = ×4 후 PIL BICUBIC 축소, 확산 2종 결정적 설정 + 예열 + 파일명 seed, b1). GPU 2장 × 4 프로세스 조각 대기열."]
    sb = list(csv.DictReader(open(R / "gen_solo_bitcheck_train.csv")))
    st = list(csv.DictReader(open(R / "gen_solo_bitcheck_test_summary.csv")))
    ov = list(csv.DictReader(open(R / "gen_stage1_overlap_check.csv")))
    L += [f"- 조각별 단독 실행 비트 비교: train {sum(int(r['n_bit_identical']) for r in sb)}/{sum(int(r['n']) for r in sb)}, test {sum(int(r['n_bit_identical']) for r in st)}/{sum(int(r['n']) for r in st)}.",
          f"- Stage 1 표본 2,250장과 비트 비교(SR 3종·bicubic × ×3·×4): {sum(int(r['n_bit_identical']) for r in ov)}/{sum(int(r['n_compared']) for r in ov)}.", ""]
    # 2. 학습
    L += ["## 2. 학습 (32회)", "",
          "- 각 모델 공식 레시피 기본 설정(BoT는 저장소 자체 구현 `reid/trainer.py` 레시피). 배치 64, epoch·학습률·스케줄·옵티마이저·정밀도·증강 변경 없음. 입력은 SR 크기 그대로(×3 384×192, ×4 512×256).",
          "- 확정 설정: memmap(학습 전 /dev/shm에 8조건 29.6GiB), BoT worker_norm, KPR copy_read + float32 마스크(원본 키포인트 기반, 모든 조건 동일), BIPO는 학습에서만, 핀 메모리 켬. DataLoader worker 수는 레시피 값(BoT 4, CLIP 8, KPR 4) 그대로(바꾸면 증강 난수가 달라짐).",
          "- cudnn: 공식 코드 기본값(KPR·CLIP-ReID benchmark=True, 자체 BoT는 원래 코드대로 deterministic=True). 등가성 검증용 결정론 모드는 쓰지 않음. gradient checkpointing은 32회 모두 끔(80GB 안에 들어감).",
          "- KPR BIPO 가림 이미지: 공식 코드는 학습 이미지 자체에서 가림 조각을 가져오므로, SR 조건에서는 그 조건의 이미지를 쓰도록 했다(공식 코드가 bicubic PNG 경로를 직접 읽는 부분만 대체).",
          "- 배치: 200 iteration 시험(`results/stage2/train_trial_summary.csv`) 뒤 KPR 계열 ×4는 GPU 단독(약 73GiB), 나머지는 GPU당 최대 3개 자동 대기열(`results/stage2/train_plan.json`).", ""]
    rows = [meta_row(m, c, k) for m in MODELS for k in (3, 4) for c in ["bicubic"] + SR if (TR / f"{m}__{c}__x{k}" / "DONE").exists()]
    L += [md_table(pd.DataFrame(rows)), ""]
    t0 = pd.to_datetime(q[q.event == "start"].time.min())
    t1 = pd.to_datetime(q[q.event.isin(["end", "fail"])].time.max())
    hrs = (t1 - t0).total_seconds() / 3600
    L += [f"- 학습 대기열 {t0} → {t1}, 벽시계 {hrs:.1f}시간, 학습 구간 비용 약 ${hrs * PRICE:.0f}(A100 2장 ${PRICE}/시간). 실패 {int((q.event == 'fail').sum())}건.", ""]
    # 3. 평가
    L += ["## 3. 평가", "",
          "- 경량 평가기(`stage05/eval_core.py`, 공식 AP·CMC와 일치 검증), 모델별 공식 거리(BoT 1−cos, CLIP-ReID L2 정규화 유클리드, KPR 공식 part 거리). query 3,368(pid −1 제외), gallery 15,913(junk 제외, 0000 포함).",
          "- **주 지표 mAP, Rank-1은 보조.** 판정: (S 모델 + SR test) − (C 모델 + bicubic test), 신원 단위 쌍 부트스트랩 95% CI(query 신원 재표본 10,000회, seed 42, 모든 비교에 같은 재표본). CI 하한 > 0 useful, 상한 < 0 harmful, 그 밖 inconclusive.",
          "- 2×2의 나머지 두 칸(train만 SR = S 모델 + bicubic test, test만 SR = C 모델 + SR test)은 같은 기준선 대비 차이와 CI만 적고 판정하지 않는다.", ""]
    for k in (3, 4):
        L += [f"### ×{k}: mAP / Rank-1 (%)", ""]
        rows = []
        for m in MODELS:
            for tc in ["bicubic"] + SR:
                r = {"모델": MNAME[m], "학습 조건": SNAME[tc]}
                for te in ["bicubic"] + SR:
                    x = cells[(cells.model == m) & (cells.k == k) & (cells.train_cond == tc) & (cells.test_cond == te)]
                    r[f"test {SNAME[te]}"] = f"{pct(x.iloc[0].mAP)} / {pct(x.iloc[0].R1)}" if len(x) else ""
                rows.append(r)
        L += [md_table(pd.DataFrame(rows)), "", "빈칸은 평가 대상이 아닌 조합(SR 모델은 자기 SR test와 bicubic test만).", ""]
    L += ["### 판정 (주 비교 S 모델 + SR test vs C 모델 + bicubic test)", ""]
    rows = []
    for m in MODELS:
        for k in (3, 4):
            for s in SR:
                r = {"모델": MNAME[m], "배율": k, "SR": SNAME[s]}
                for met, nm in (("mAP", "mAP"), ("R1", "Rank-1(보조)")):
                    x = judg[(judg.model == m) & (judg.k == k) & (judg.sr_cond == s) & (judg.comparison == "S_on_S") & (judg.metric == met)]
                    if len(x):
                        x = x.iloc[0]
                        r[f"Δ{nm} %p [95% CI]"] = f"{100 * x.delta:+.2f} [{100 * x.ci_lo:+.2f}, {100 * x.ci_hi:+.2f}]"
                        r[f"{nm} 판정"] = x.label
                rows.append(r)
    L += [md_table(pd.DataFrame(rows)), ""]
    jm = judg[(judg.comparison == "S_on_S") & (judg.metric == "mAP")]
    L += [f"mAP 판정 집계(24회): useful {int((jm.label == 'useful').sum())}, harmful {int((jm.label == 'harmful').sum())}, inconclusive {int((jm.label == 'inconclusive').sum())}.", ""]
    L += ["### 2×2 나머지 칸 (mAP, C 모델 + bicubic test 대비 차이 %p [95% CI], 판정 없음)", ""]
    rows = []
    for m in MODELS:
        for k in (3, 4):
            for s in SR:
                r = {"모델": MNAME[m], "배율": k, "SR": SNAME[s]}
                for comp, nm in (("C_on_S", "test만 SR (C 모델 + SR test)"), ("S_on_C", "train만 SR (S 모델 + bicubic test)")):
                    x = judg[(judg.model == m) & (judg.k == k) & (judg.sr_cond == s) & (judg.comparison == comp) & (judg.metric == "mAP")]
                    if len(x):
                        x = x.iloc[0]
                        r[nm] = f"{100 * x.delta:+.2f} [{100 * x.ci_lo:+.2f}, {100 * x.ci_hi:+.2f}]"
                rows.append(r)
    L += [md_table(pd.DataFrame(rows)), ""]
    # 4. 한계
    L += ["## 4. 한계", "",
          "- **다중 비교**: 주 판정만 24회(모델 4 × 배율 2 × SR 3)이며 보정하지 않은 95% CI다. 우연히 0을 벗어나는 판정이 섞일 수 있다(모두 귀무가설이 참이라도 기대 약 1.2건).",
          "- **seed 1개**: 모든 학습이 seed 42 한 번이다. CI는 test query 신원 재표본만 반영하며 학습 변동(초기화·데이터 순서·cudnn 비결정성)을 반영하지 못한다. 작은 차이(특히 ±0.5%p 안팎)는 학습을 다시 하면 바뀔 수 있다.",
          "- KPR·CLIP-ReID는 공식 기본값(cudnn.benchmark, fp16)이라 같은 seed로 다시 돌려도 비트 단위로 같지 않다. BoT만 결정적이다.",
          "- 판정의 기준선은 C 모델 + bicubic test 하나다. SR 모델을 bicubic test에 쓰는 경우(train만 SR)에는 큰 하락이 있을 수 있으며(예: BoT osdhuman), 이는 별도 칸으로만 보고했다.",
          "- **기준선 학습 하나의 영향(관찰)**: KPR+SOLIDER ×3 bicubic 모델(C 모델)은 bicubic test mAP 87.56으로 같은 모델의 SR 조건 대각 칸(90.8~91.8)과 ×4 bicubic(91.45)보다 낮고, 마지막 학습 손실도 6.42로 다른 KPR+SOLIDER 학습(5.37~5.52)보다 높다. KPR+SOLIDER ×3의 useful 3건(+3.3~+4.2%p)은 대부분 이 기준선 학습이 약하게 수렴한 데서 온 것일 수 있다. 반대로 KPR ×3 swinir 모델은 84.77로 같은 모델의 다른 조건(89.2~90.0)보다 낮아 harmful(−5.21%p)로 나왔다. seed 1개라 학습 변동과 SR 효과를 가를 수 없다.",
          "- 시간은 병렬 실행 중 측정값이라 단독 처리량이 아니다.",
          "- 평가 개발 중 에이전트가 잠금 해제 창에서 `ls data/locked`를 한 번 실행했다(최상위 폴더 이름만 출력, 파일 열람 없음, ALLOW_TEST 미설정).", ""]
    (ROOT / "STAGE2_REPORT.md").write_text("\n".join(L))
    print("STAGE2_REPORT.md", len(L), "줄")


if __name__ == "__main__":
    main()
