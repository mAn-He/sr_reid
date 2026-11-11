# Stage 2 보고서: SR 3종 학습·평가 (ReID 4모델 × {bicubic, swinir, pisa λ0.0, osdhuman} × {×3, ×4})

작성 자동(`stage2/report.py`). 서버 A100 서버(A100 SXM4 80GB ×2, EPYC 7713, cgroup v2). 모든 시간은 **병렬 실행 중 측정**(GPU 2장에 여러 학습·평가 동시)이다. seed 42 하나.

## 1. 이미지 생성

- train: SR 3종 × ×3·×4 × 12,936장. 누락 0, 크기 불일치 0, 로그 SHA256 불일치 0. bicubic train은 기존 `data/bicubic_x{3,4}` 재사용.
- test: SR 3종 + bicubic × ×3·×4 × (query 3,368 + gallery 19,732). 누락 0, 크기 불일치 0, 로그 SHA256 불일치 0. 실제 파일은 잠긴 영역 `data/locked/sr/`.
- 설정: Stage 1과 같음(swinir ×3 공식 ×3 가중치, pisa·osdhuman ×3 = ×4 후 PIL BICUBIC 축소, 확산 2종 결정적 설정 + 예열 + 파일명 seed, b1). GPU 2장 × 4 프로세스 조각 대기열.
- 조각별 단독 실행 비트 비교: train 600/600, test 1200/1200.
- Stage 1 표본 2,250장과 비트 비교(SR 3종·bicubic × ×3·×4): 18000/18000.

## 2. 학습 (32회)

- 각 모델 공식 레시피 기본 설정(BoT는 저장소 자체 구현 `reid/trainer.py` 레시피). 배치 64, epoch·학습률·스케줄·옵티마이저·정밀도·증강 변경 없음. 입력은 SR 크기 그대로(×3 384×192, ×4 512×256).
- 확정 설정: memmap(학습 전 /dev/shm에 8조건 29.6GiB), BoT worker_norm, KPR copy_read + float32 마스크(원본 키포인트 기반, 모든 조건 동일), BIPO는 학습에서만, 핀 메모리 켬. DataLoader worker 수는 레시피 값(BoT 4, CLIP 8, KPR 4) 그대로(바꾸면 증강 난수가 달라짐).
- cudnn: 공식 코드 기본값(KPR·CLIP-ReID benchmark=True, 자체 BoT는 원래 코드대로 deterministic=True). 등가성 검증용 결정론 모드는 쓰지 않음. gradient checkpointing은 32회 모두 끔(80GB 안에 들어감).
- KPR BIPO 가림 이미지: 공식 코드는 학습 이미지 자체에서 가림 조각을 가져오므로, SR 조건에서는 그 조건의 이미지를 쓰도록 했다(공식 코드가 bicubic PNG 경로를 직접 읽는 부분만 대체).
- 배치: 200 iteration 시험(`results/stage2/train_trial_summary.csv`) 뒤 KPR 계열 ×4는 GPU 단독(약 73GiB), 나머지는 GPU당 최대 3개 자동 대기열(`results/stage2/train_plan.json`).

| 모델 | 학습 조건 | 배율 | GPU | 학습 시간 h(병렬 실행 중) | AMP | cudnn | checkpointing | peak 예약 GiB | 재개 |
|---|---|---|---|---|---|---|---|---|---|
| BoT | bicubic | 3 | 0 | 2.63 | 없음 | det=True, bench=False | 끔 | 10.6 | 0 |
| BoT | swinir | 3 | 0 | 2.62 | 없음 | det=True, bench=False | 끔 | 10.6 | 0 |
| BoT | pisa λ0.0 | 3 | 0 | 2.71 | 없음 | det=True, bench=False | 끔 | 10.6 | 0 |
| BoT | osdhuman | 3 | 0 | 2.61 | 없음 | det=True, bench=False | 끔 | 10.6 | 0 |
| BoT | bicubic | 4 | 0 | 3.71 | 없음 | det=True, bench=False | 끔 | 18.8 | 0 |
| BoT | swinir | 4 | 0 | 3.81 | 없음 | det=True, bench=False | 끔 | 18.8 | 0 |
| BoT | pisa λ0.0 | 4 | 0 | 4.11 | 없음 | det=True, bench=False | 끔 | 18.8 | 0 |
| BoT | osdhuman | 4 | 0 | 4.22 | 없음 | det=True, bench=False | 끔 | 18.8 | 0 |
| CLIP-ReID | bicubic | 3 | 0 | 2.47 | fp16 autocast(공식) | det=True, bench=True | 끔 | 14.4 | 0 |
| CLIP-ReID | swinir | 3 | 0 | 2.45 | fp16 autocast(공식) | det=True, bench=True | 끔 | 14.4 | 0 |
| CLIP-ReID | pisa λ0.0 | 3 | 0 | 2.51 | fp16 autocast(공식) | det=True, bench=True | 끔 | 14.4 | 0 |
| CLIP-ReID | osdhuman | 3 | 0 | 2.59 | fp16 autocast(공식) | det=True, bench=True | 끔 | 14.4 | 0 |
| CLIP-ReID | bicubic | 4 | 0 | 2.87 | fp16 autocast(공식) | det=True, bench=True | 끔 | 30.6 | 0 |
| CLIP-ReID | swinir | 4 | 0 | 2.91 | fp16 autocast(공식) | det=True, bench=True | 끔 | 30.6 | 0 |
| CLIP-ReID | pisa λ0.0 | 4 | 0 | 2.86 | fp16 autocast(공식) | det=True, bench=True | 끔 | 30.6 | 0 |
| CLIP-ReID | osdhuman | 4 | 0 | 2.55 | fp16 autocast(공식) | det=True, bench=True | 끔 | 30.6 | 0 |
| KPR | bicubic | 3 | 1 | 6.15 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.6 | 0 |
| KPR | swinir | 3 | 0 | 7.20 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.6 | 0 |
| KPR | pisa λ0.0 | 3 | 1 | 7.09 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.6 | 0 |
| KPR | osdhuman | 3 | 1 | 4.70 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.6 | 0 |
| KPR | bicubic | 4 | 1 | 5.15 | mixed_precision(공식) | det=False, bench=True | 끔 | 68.7 | 0 |
| KPR | swinir | 4 | 1 | 5.20 | mixed_precision(공식) | det=False, bench=True | 끔 | 68.7 | 0 |
| KPR | pisa λ0.0 | 4 | 0 | 6.14 | mixed_precision(공식) | det=False, bench=True | 끔 | 68.7 | 0 |
| KPR | osdhuman | 4 | 1 | 5.17 | mixed_precision(공식) | det=False, bench=True | 끔 | 68.7 | 0 |
| KPR+SOLIDER | bicubic | 3 | 0 | 8.72 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.5 | 0 |
| KPR+SOLIDER | swinir | 3 | 0 | 8.81 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.5 | 0 |
| KPR+SOLIDER | pisa λ0.0 | 3 | 0 | 12.48 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.5 | 0 |
| KPR+SOLIDER | osdhuman | 3 | 0 | 6.67 | mixed_precision(공식) | det=False, bench=True | 끔 | 37.5 | 0 |
| KPR+SOLIDER | bicubic | 4 | 1 | 5.15 | mixed_precision(공식) | det=False, bench=True | 끔 | 73.3 | 0 |
| KPR+SOLIDER | swinir | 4 | 1 | 5.50 | mixed_precision(공식) | det=False, bench=True | 끔 | 73.3 | 0 |
| KPR+SOLIDER | pisa λ0.0 | 4 | 1 | 5.19 | mixed_precision(공식) | det=False, bench=True | 끔 | 73.3 | 0 |
| KPR+SOLIDER | osdhuman | 4 | 1 | 5.14 | mixed_precision(공식) | det=False, bench=True | 끔 | 73.3 | 0 |

- 학습 대기열 2026-11-09 08:42:29 → 2026-11-11 15:51:46, 벽시계 55.2시간, 학습 구간 비용 약 $198(A100 2장 $3.59/시간). 실패 0건.

## 3. 평가

- 경량 평가기(`stage05/eval_core.py`, 공식 AP·CMC와 일치 검증), 모델별 공식 거리(BoT 1−cos, CLIP-ReID L2 정규화 유클리드, KPR 공식 part 거리). query 3,368(pid −1 제외), gallery 15,913(junk 제외, 0000 포함).
- **주 지표 mAP, Rank-1은 보조.** 판정: (S 모델 + SR test) − (C 모델 + bicubic test), 신원 단위 쌍 부트스트랩 95% CI(query 신원 재표본 10,000회, seed 42, 모든 비교에 같은 재표본). CI 하한 > 0 useful, 상한 < 0 harmful, 그 밖 inconclusive.
- 2×2의 나머지 두 칸(train만 SR = S 모델 + bicubic test, test만 SR = C 모델 + SR test)은 같은 기준선 대비 차이와 CI만 적고 판정하지 않는다.

### ×3: mAP / Rank-1 (%)

| 모델 | 학습 조건 | test bicubic | test swinir | test pisa λ0.0 | test osdhuman |
|---|---|---|---|---|---|
| BoT | bicubic | 85.52 / 93.88 | 83.91 / 93.26 | 68.52 / 86.19 | 57.43 / 78.21 |
| BoT | swinir | 84.89 / 93.88 | 85.31 / 94.03 |  |  |
| BoT | pisa λ0.0 | 77.91 / 90.80 |  | 85.32 / 93.82 |  |
| BoT | osdhuman | 18.77 / 43.71 |  |  | 86.25 / 93.62 |
| CLIP-ReID | bicubic | 89.91 / 95.10 | 89.40 / 94.89 | 84.54 / 92.04 | 80.41 / 90.88 |
| CLIP-ReID | swinir | 89.65 / 95.19 | 89.68 / 95.13 |  |  |
| CLIP-ReID | pisa λ0.0 | 86.71 / 93.56 |  | 89.64 / 95.10 |  |
| CLIP-ReID | osdhuman | 80.65 / 90.91 |  |  | 89.34 / 94.83 |
| KPR | bicubic | 89.98 / 95.75 | 89.69 / 95.58 | 85.11 / 93.38 | 82.86 / 92.79 |
| KPR | swinir | 84.70 / 94.06 | 84.77 / 93.94 |  |  |
| KPR | pisa λ0.0 | 87.87 / 95.28 |  | 89.88 / 95.55 |  |
| KPR | osdhuman | 83.68 / 93.05 |  |  | 89.24 / 95.43 |
| KPR+SOLIDER | bicubic | 87.56 / 94.30 | 87.31 / 94.51 | 84.13 / 92.40 | 80.17 / 89.99 |
| KPR+SOLIDER | swinir | 91.80 / 96.05 | 91.78 / 96.11 |  |  |
| KPR+SOLIDER | pisa λ0.0 | 89.88 / 95.46 |  | 91.35 / 95.78 |  |
| KPR+SOLIDER | osdhuman | 86.90 / 93.74 |  |  | 90.83 / 95.25 |

빈칸은 평가 대상이 아닌 조합(SR 모델은 자기 SR test와 bicubic test만).

### ×4: mAP / Rank-1 (%)

| 모델 | 학습 조건 | test bicubic | test swinir | test pisa λ0.0 | test osdhuman |
|---|---|---|---|---|---|
| BoT | bicubic | 83.96 / 93.11 | 81.58 / 92.37 | 63.08 / 82.60 | 50.75 / 74.20 |
| BoT | swinir | 83.28 / 92.67 | 83.68 / 93.02 |  |  |
| BoT | pisa λ0.0 | 75.38 / 89.55 |  | 84.53 / 93.79 |  |
| BoT | osdhuman | 9.78 / 28.27 |  |  | 85.51 / 93.74 |
| CLIP-ReID | bicubic | 89.21 / 95.16 | 88.48 / 94.39 | 83.18 / 91.45 | 77.82 / 89.85 |
| CLIP-ReID | swinir | 88.88 / 94.66 | 88.94 / 95.07 |  |  |
| CLIP-ReID | pisa λ0.0 | 85.80 / 93.59 |  | 89.24 / 94.77 |  |
| CLIP-ReID | osdhuman | 77.72 / 90.32 |  |  | 89.18 / 94.69 |
| KPR | bicubic | 89.79 / 95.52 | 89.56 / 95.46 | 86.05 / 94.24 | 84.35 / 93.14 |
| KPR | swinir | 89.90 / 95.55 | 90.01 / 95.64 |  |  |
| KPR | pisa λ0.0 | 87.87 / 95.22 |  | 89.95 / 95.58 |  |
| KPR | osdhuman | 82.72 / 92.67 |  |  | 89.47 / 95.40 |
| KPR+SOLIDER | bicubic | 91.45 / 95.72 | 91.09 / 95.61 | 85.67 / 92.99 | 76.61 / 87.56 |
| KPR+SOLIDER | swinir | 91.36 / 96.08 | 91.36 / 95.78 |  |  |
| KPR+SOLIDER | pisa λ0.0 | 89.82 / 94.89 |  | 91.52 / 95.93 |  |
| KPR+SOLIDER | osdhuman | 86.38 / 93.50 |  |  | 90.84 / 95.61 |

빈칸은 평가 대상이 아닌 조합(SR 모델은 자기 SR test와 bicubic test만).

### 판정 (주 비교 S 모델 + SR test vs C 모델 + bicubic test)

| 모델 | 배율 | SR | ΔmAP %p [95% CI] | mAP 판정 | ΔRank-1(보조) %p [95% CI] | Rank-1(보조) 판정 |
|---|---|---|---|---|---|---|
| BoT | 3 | swinir | -0.22 [-0.48, +0.05] | inconclusive | +0.15 [-0.42, +0.71] | inconclusive |
| BoT | 3 | pisa λ0.0 | -0.20 [-0.53, +0.12] | inconclusive | -0.06 [-0.65, +0.56] | inconclusive |
| BoT | 3 | osdhuman | +0.73 [+0.24, +1.20] | useful | -0.27 [-0.98, +0.44] | inconclusive |
| BoT | 4 | swinir | -0.28 [-0.57, +0.00] | inconclusive | -0.09 [-0.69, +0.48] | inconclusive |
| BoT | 4 | pisa λ0.0 | +0.57 [+0.22, +0.91] | useful | +0.68 [+0.06, +1.32] | useful |
| BoT | 4 | osdhuman | +1.55 [+1.04, +2.04] | useful | +0.62 [-0.15, +1.40] | inconclusive |
| CLIP-ReID | 3 | swinir | -0.24 [-0.42, -0.05] | harmful | +0.03 [-0.47, +0.51] | inconclusive |
| CLIP-ReID | 3 | pisa λ0.0 | -0.27 [-0.56, +0.02] | inconclusive | +0.00 [-0.53, +0.53] | inconclusive |
| CLIP-ReID | 3 | osdhuman | -0.57 [-0.96, -0.17] | harmful | -0.27 [-0.89, +0.33] | inconclusive |
| CLIP-ReID | 4 | swinir | -0.27 [-0.50, -0.06] | harmful | -0.09 [-0.54, +0.39] | inconclusive |
| CLIP-ReID | 4 | pisa λ0.0 | +0.03 [-0.27, +0.32] | inconclusive | -0.39 [-0.96, +0.18] | inconclusive |
| CLIP-ReID | 4 | osdhuman | -0.03 [-0.48, +0.41] | inconclusive | -0.48 [-1.16, +0.21] | inconclusive |
| KPR | 3 | swinir | -5.21 [-5.80, -4.65] | harmful | -1.81 [-2.51, -1.14] | harmful |
| KPR | 3 | pisa λ0.0 | -0.11 [-0.40, +0.19] | inconclusive | -0.21 [-0.76, +0.33] | inconclusive |
| KPR | 3 | osdhuman | -0.74 [-1.13, -0.37] | harmful | -0.33 [-0.89, +0.24] | inconclusive |
| KPR | 4 | swinir | +0.22 [-0.03, +0.45] | inconclusive | +0.12 [-0.39, +0.62] | inconclusive |
| KPR | 4 | pisa λ0.0 | +0.16 [-0.10, +0.42] | inconclusive | +0.06 [-0.43, +0.56] | inconclusive |
| KPR | 4 | osdhuman | -0.32 [-0.70, +0.03] | inconclusive | -0.12 [-0.75, +0.53] | inconclusive |
| KPR+SOLIDER | 3 | swinir | +4.22 [+3.65, +4.81] | useful | +1.81 [+1.15, +2.48] | useful |
| KPR+SOLIDER | 3 | pisa λ0.0 | +3.79 [+3.26, +4.36] | useful | +1.48 [+0.78, +2.21] | useful |
| KPR+SOLIDER | 3 | osdhuman | +3.27 [+2.68, +3.89] | useful | +0.95 [+0.21, +1.73] | useful |
| KPR+SOLIDER | 4 | swinir | -0.09 [-0.33, +0.16] | inconclusive | +0.06 [-0.39, +0.51] | inconclusive |
| KPR+SOLIDER | 4 | pisa λ0.0 | +0.07 [-0.19, +0.34] | inconclusive | +0.21 [-0.30, +0.71] | inconclusive |
| KPR+SOLIDER | 4 | osdhuman | -0.61 [-1.00, -0.24] | harmful | -0.12 [-0.72, +0.50] | inconclusive |

mAP 판정 집계(24회): useful 6, harmful 6, inconclusive 12.

### 2×2 나머지 칸 (mAP, C 모델 + bicubic test 대비 차이 %p [95% CI], 판정 없음)

| 모델 | 배율 | SR | test만 SR (C 모델 + SR test) | train만 SR (S 모델 + bicubic test) |
|---|---|---|---|---|
| BoT | 3 | swinir | -1.61 [-1.83, -1.41] | -0.63 [-0.91, -0.36] |
| BoT | 3 | pisa λ0.0 | -17.00 [-17.91, -16.10] | -7.61 [-8.20, -7.04] |
| BoT | 3 | osdhuman | -28.10 [-29.23, -26.97] | -66.75 [-68.02, -65.43] |
| BoT | 4 | swinir | -2.38 [-2.66, -2.12] | -0.69 [-1.00, -0.39] |
| BoT | 4 | pisa λ0.0 | -20.88 [-21.85, -19.93] | -8.59 [-9.21, -7.98] |
| BoT | 4 | osdhuman | -33.21 [-34.39, -32.03] | -74.18 [-75.43, -72.87] |
| CLIP-ReID | 3 | swinir | -0.51 [-0.64, -0.39] | -0.27 [-0.45, -0.09] |
| CLIP-ReID | 3 | pisa λ0.0 | -5.38 [-5.93, -4.84] | -3.20 [-3.58, -2.85] |
| CLIP-ReID | 3 | osdhuman | -9.50 [-10.26, -8.74] | -9.26 [-9.95, -8.59] |
| CLIP-ReID | 4 | swinir | -0.73 [-0.90, -0.57] | -0.33 [-0.55, -0.12] |
| CLIP-ReID | 4 | pisa λ0.0 | -6.03 [-6.60, -5.48] | -3.41 [-3.81, -3.03] |
| CLIP-ReID | 4 | osdhuman | -11.39 [-12.24, -10.54] | -11.49 [-12.27, -10.73] |
| KPR | 3 | swinir | -0.29 [-0.39, -0.20] | -5.28 [-5.87, -4.71] |
| KPR | 3 | pisa λ0.0 | -4.88 [-5.36, -4.40] | -2.11 [-2.45, -1.77] |
| KPR | 3 | osdhuman | -7.13 [-7.76, -6.52] | -6.30 [-6.89, -5.72] |
| KPR | 4 | swinir | -0.23 [-0.33, -0.14] | +0.11 [-0.14, +0.35] |
| KPR | 4 | pisa λ0.0 | -3.74 [-4.17, -3.32] | -1.92 [-2.23, -1.61] |
| KPR | 4 | osdhuman | -5.44 [-5.97, -4.91] | -7.07 [-7.73, -6.44] |
| KPR+SOLIDER | 3 | swinir | -0.25 [-0.36, -0.15] | +4.24 [+3.66, +4.84] |
| KPR+SOLIDER | 3 | pisa λ0.0 | -3.43 [-3.85, -3.01] | +2.33 [+1.76, +2.90] |
| KPR+SOLIDER | 3 | osdhuman | -7.39 [-8.04, -6.74] | -0.66 [-1.29, -0.03] |
| KPR+SOLIDER | 4 | swinir | -0.36 [-0.48, -0.23] | -0.09 [-0.34, +0.16] |
| KPR+SOLIDER | 4 | pisa λ0.0 | -5.78 [-6.38, -5.20] | -1.62 [-1.95, -1.30] |
| KPR+SOLIDER | 4 | osdhuman | -14.84 [-15.83, -13.87] | -5.07 [-5.64, -4.53] |

## 4. 한계

- **다중 비교**: 주 판정만 24회(모델 4 × 배율 2 × SR 3)이며 보정하지 않은 95% CI다. 우연히 0을 벗어나는 판정이 섞일 수 있다(모두 귀무가설이 참이라도 기대 약 1.2건).
- **seed 1개**: 모든 학습이 seed 42 한 번이다. CI는 test query 신원 재표본만 반영하며 학습 변동(초기화·데이터 순서·cudnn 비결정성)을 반영하지 못한다. 작은 차이(특히 ±0.5%p 안팎)는 학습을 다시 하면 바뀔 수 있다.
- KPR·CLIP-ReID는 공식 기본값(cudnn.benchmark, fp16)이라 같은 seed로 다시 돌려도 비트 단위로 같지 않다. BoT만 결정적이다.
- 판정의 기준선은 C 모델 + bicubic test 하나다. SR 모델을 bicubic test에 쓰는 경우(train만 SR)에는 큰 하락이 있을 수 있으며(예: BoT osdhuman), 이는 별도 칸으로만 보고했다.
- **기준선 학습 하나의 영향(관찰)**: KPR+SOLIDER ×3 bicubic 모델(C 모델)은 bicubic test mAP 87.56으로 같은 모델의 SR 조건 대각 칸(90.8~91.8)과 ×4 bicubic(91.45)보다 낮고, 마지막 학습 손실도 6.42로 다른 KPR+SOLIDER 학습(5.37~5.52)보다 높다. KPR+SOLIDER ×3의 useful 3건(+3.3~+4.2%p)은 대부분 이 기준선 학습이 약하게 수렴한 데서 온 것일 수 있다. 반대로 KPR ×3 swinir 모델은 84.77로 같은 모델의 다른 조건(89.2~90.0)보다 낮아 harmful(−5.21%p)로 나왔다. seed 1개라 학습 변동과 SR 효과를 가를 수 없다.
- 시간은 병렬 실행 중 측정값이라 단독 처리량이 아니다.
- 평가 개발 중 에이전트가 잠금 해제 창에서 `ls data/locked`를 한 번 실행했다(최상위 폴더 이름만 출력, 파일 열람 없음, ALLOW_TEST 미설정).
