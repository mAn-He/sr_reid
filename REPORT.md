# 파일럿 리포트 (pilot_001)

모든 수치는 `results/pilot_summary.csv`와 `runs/` 산출물에서 계산했습니다. 공식 test split은 사용하지 않았습니다.

- val query 404장 사용, 다른 카메라 gallery가 없어 제외한 query 9장.
- 주 기준선은 `path1_mixed`, 보조 기준선은 `path1_lr_only` (결과 확인 전 등록).
- Path 1 입력과 bicubic official 출력의 바이트 동일: {'identical': 404, 'n': 404}.
- SD2.1-base 사본 상태: **미검증**.

## go/no-go 판정

### 1. 확산 계열 `official` 출력 안정성

**판정: GO**

기준: 확산 모델별 `official` 깨진 비율 0.01 이하. 수치 기준이며, 육안 확인은 grid.png로 합니다.

| 조건 | 깨진 비율 |
|---|---|
| pisa_sr_sem0.0__official | 0.0000 |
| pisa_sr_sem0.0__native | 0.0000 |
| pisa_sr_sem0.0__fixed8x | 0.0000 |
| pisa_sr_sem0.5__official | 0.0000 |
| pisa_sr_sem0.5__native | 0.0000 |
| pisa_sr_sem0.5__fixed8x | 0.0000 |
| pisa_sr_sem1.0__official | 0.0000 |
| pisa_sr_sem1.0__native | 0.0000 |
| pisa_sr_sem1.0__fixed8x | 0.0000 |
| pisa_sr_sem1.5__official | 0.0000 |
| pisa_sr_sem1.5__native | 0.0000 |
| pisa_sr_sem1.5__fixed8x | 0.0000 |
| pisa_sr_sem2.0__official | 0.0000 |
| pisa_sr_sem2.0__native | 0.0000 |
| pisa_sr_sem2.0__fixed8x | 0.0000 |
| osdhuman__official | 0.0000 |
| osdhuman__native | 0.0000 |
| osdhuman__fixed8x | 0.0000 |

### 2. 선확대 정책에 따른 mAP 순위 변화

**판정: 변함**

본 실험에서 선확대 정책을 고정 변수로 사전 등록해야 합니다. 정책별 출력이 바이트 단위로 같은 조합은 아래 정책 동일성 표에 있습니다.

| 정책 쌍 | 공통 방법 수 | Kendall τ | 뒤집힌 쌍 | CI가 0을 벗어난 뒤집힘 |
|---|---:|---:|---:|---:|
| official / native | 5 | -0.400 | 7 | 7 |
| official / fixed8x | 5 | 1.000 | 0 | 0 |
| native / fixed8x | 5 | -0.400 | 7 | 7 |

### 3. 본 실험 소요 외삽

**판정: 계산됨**

공식 query 3368장(문헌값, 파일은 읽지 않음) × 조건 27개 기준 합계 **8.82시간**. 모델 로드 시간은 제외했습니다.

| 조건 | 초/장 (exclusive_gpu=true) | query당 외삽 시간(시간) |
|---|---:|---:|
| bicubic__official | 0.0255 | 0.024 |
| bicubic__native | 0.0228 | 0.021 |
| bicubic__fixed8x | 0.0903 | 0.084 |
| lanczos__official | 0.0236 | 0.022 |
| lanczos__native | 0.0238 | 0.022 |
| lanczos__fixed8x | 0.1074 | 0.100 |
| realesrgan_x4plus__official | 0.0575 | 0.054 |
| realesrgan_x4plus__native | 0.0523 | 0.049 |
| realesrgan_x4plus__fixed8x | 0.0991 | 0.093 |
| pisa_sr_sem0.0__official | 0.5381 | 0.503 |
| pisa_sr_sem0.0__native | 0.1656 | 0.155 |
| pisa_sr_sem0.0__fixed8x | 0.7489 | 0.701 |
| pisa_sr_sem0.5__official | 0.5341 | 0.500 |
| pisa_sr_sem0.5__native | 0.1668 | 0.156 |
| pisa_sr_sem0.5__fixed8x | 0.8207 | 0.768 |
| pisa_sr_sem1.0__official | 0.5377 | 0.503 |
| pisa_sr_sem1.0__native | 0.1657 | 0.155 |
| pisa_sr_sem1.0__fixed8x | 0.8511 | 0.796 |
| pisa_sr_sem1.5__official | 0.5389 | 0.504 |
| pisa_sr_sem1.5__native | 0.1634 | 0.153 |
| pisa_sr_sem1.5__fixed8x | 0.8568 | 0.802 |
| pisa_sr_sem2.0__official | 0.5429 | 0.508 |
| pisa_sr_sem2.0__native | 0.1645 | 0.154 |
| pisa_sr_sem2.0__fixed8x | 0.7953 | 0.744 |
| osdhuman__official | 0.4713 | 0.441 |
| osdhuman__native | 0.1239 | 0.116 |
| osdhuman__fixed8x | 0.7382 | 0.691 |

### 4. lambda_sem 증가에 따른 mAP 단조 감소

**판정: 신호 없음**

주 판정은 `official` 정책입니다. 가설 신호일 뿐 결론이 아닙니다. 점추정은 부트스트랩 평균입니다.

| 정책 | 점추정 단조 감소 | mAP(λ=최대) − mAP(λ=0) | 95% CI | 신호 |
|---|---|---:|---|---|
| official | 아니오 | -0.0012 | [-0.0240, +0.0211] | 신호 없음 |
| native | 아니오 | -0.3579 | [-0.4010, -0.3168] | 신호 없음 |
| fixed8x | 아니오 | -0.0024 | [-0.0252, +0.0200] | 신호 없음 |

## 정책 간 출력 동일성

| 방법 | 정책 쌍 | 바이트 동일 / 공통 |
|---|---|---:|
| bicubic | fixed8x / native | 0 / 404 |
| bicubic | fixed8x / official | 0 / 404 |
| bicubic | native / official | 404 / 404 |
| lanczos | fixed8x / native | 0 / 404 |
| lanczos | fixed8x / official | 0 / 404 |
| lanczos | native / official | 404 / 404 |
| realesrgan_x4plus | fixed8x / native | 0 / 404 |
| realesrgan_x4plus | fixed8x / official | 0 / 404 |
| realesrgan_x4plus | native / official | 404 / 404 |
| pisa_sr_sem0.0 | fixed8x / native | 0 / 404 |
| pisa_sr_sem0.0 | fixed8x / official | 0 / 404 |
| pisa_sr_sem0.0 | native / official | 0 / 404 |
| pisa_sr_sem0.5 | fixed8x / native | 0 / 404 |
| pisa_sr_sem0.5 | fixed8x / official | 0 / 404 |
| pisa_sr_sem0.5 | native / official | 0 / 404 |
| pisa_sr_sem1.0 | fixed8x / native | 0 / 404 |
| pisa_sr_sem1.0 | fixed8x / official | 0 / 404 |
| pisa_sr_sem1.0 | native / official | 0 / 404 |
| pisa_sr_sem1.5 | fixed8x / native | 0 / 404 |
| pisa_sr_sem1.5 | fixed8x / official | 0 / 404 |
| pisa_sr_sem1.5 | native / official | 0 / 404 |
| pisa_sr_sem2.0 | fixed8x / native | 0 / 404 |
| pisa_sr_sem2.0 | fixed8x / official | 0 / 404 |
| pisa_sr_sem2.0 | native / official | 0 / 404 |
| osdhuman | fixed8x / native | 0 / 404 |
| osdhuman | fixed8x / official | 0 / 404 |
| osdhuman | native / official | 0 / 404 |

## 한계

- 신뢰구간은 query 표본 변동만 반영합니다. 파일럿의 ReID 학습 seed는 하나입니다.
- 본 실험에서는 HR ReID 모델 3 seed로 학습 변동을 반영할 예정 (SR 출력은 재사용, 특징 추출만 반복).
- HR은 상대적 정의(Market 원본 128×64)이며, LR은 bicubic 합성 축소로 실제 카메라 열화와 다르다. 실제 LR 검증은 별도 데이터셋으로 후속 단계에서 수행.
- 다중 비교를 보정하지 않았습니다.
- Spearman ρ는 대표 방법 5개 기준의 기술 통계입니다. p값은 내지 않습니다.
- test split을 쓰지 않아 ReID 구현을 BoT 공개 수치와 대조할 수 없습니다. `hr_oracle` 행이 상한선 점검용입니다.
- val gallery는 100 ID뿐이라 mAP가 공식 평가보다 높게 나오며, 공식 수치와 직접 비교할 수 없습니다.
- 깨진 이미지 판정은 수치 기준입니다. 환각 같은 시각 오류는 grid.png 육안 확인이 필요합니다.
- SD2.1-base 사본 상태는 '미검증'이며 이 결과의 확산 모델 수치는 그 전제 위에 있습니다.
