# Stage 1·2 계획 (2026-11-06 설계 변경, 초안)

이전 Stage 1 설계(검증 분할 651/100, BoT·CLIP-ReID 12종 학습, seed 추가, 두 모델 순위 비교, Stage 1 병렬화)는 **전부 취소**했다. seed는 42 하나다. 시간에는 측정 서버를 표시한다([새] = RTX 4090, EPYC 7282, cgroup v2).

## 1. 전체 구조
1. **Stage 1, SR 기법 비교 (학습 없음)**
   - train 표본(751명 × 3장, 2,250장)에 12종 × {×2, ×3, ×4} SR을 적용한다.
   - SR 전후 비교 지표로 상위 2개를 고른다. 규칙은 `STAGE1_SELECTION_RULE.md`이고 결과 계산 전에 고정한다. test는 쓰지 않는다.
2. **생성**: 선정 2개 방법의 train·test 전체 ×3·×4(`STAGE05_PART_F_PLAN.md`). bicubic은 기존 데이터를 재사용한다. test는 사용자가 잠금을 풀고 다시 잠그는 절차로 만든다.
3. **Stage 2**: 모델 4종 × {bicubic, SR1, SR2} × {×3, ×4} = 학습 24회, seed 42.
   - 평가는 2×2로 (bicubic 학습 모델, SR 학습 모델) × (bicubic test, SR test)다.
   - 모델·배율마다 평가 7회: C 모델로 bicubic·SR1·SR2 test, SR1 모델로 SR1·bicubic test, SR2 모델로 SR2·bicubic test.

## 2. Stage 1 세부
- 표본: `configs/stage1_sample.csv`(`stage1/make_sample.py`, seed 42).
  - 2,250장, 751명. 신원당 3장이 748명, 2장이 3명이다.
  - 서로 다른 카메라 3대가 701명, 2대가 50명이다(그 50명은 train에서 카메라가 2대뿐).
  - 카메라 분포 c1~c6: 460 / 377 / 491 / 164 / 390 / 368.
- 지표(정의와 용도는 규칙 문서 2절)
  - a. 정보 보존: LR 일관성, 색. LR 일관성은 1단계 거르기에 쓴다.
  - b. 주파수: 보고만 한다.
  - c. 사람 특징 보존: 공개 BoT·CLIP-ReID ID 코사인, 속성 일치율.
  - d. 복원력: 1/k 축소 → SR → PSNR, SSIM, LPIPS, DreamSim.
  - e. 무참조 화질: MUSIQ, CLIP-IQA, MANIQA, NIQE. 보고만 한다.
- SR 실행량: 12종 × 3배율 × (표본 2,250장 + 복원용 축소 입력 2,250장).
  - ×4 전용 방법(osdhuman, pisa_sr ×5)은 ×4 한 번 실행의 출력을 줄여 ×2·×3을 만든다. 실제로는 원본 입력 1회 + 축소 입력 배율별 3회다.
  - 예상 GPU 시간 [새, 4090]. SR은 배치 측정(`results/stage1_sr_batch_bench.csv`)과 스모크 중앙값에서 계산했다. 축소 입력도 원본과 같은 시간이 든다고 보고 계산한 상한이다.
    - SR, 전부 배치 1: ×4 약 5.0h, ×3 약 2.7h, ×2 약 2.2h, 합계 약 10h. PiSA λ 5단계가 7.5h를 차지한다.
    - SR, 배치 사용(realesrgan b4, swinir b32, hat b8~16, 확산 b8~16, mambairv2는 b1): ×4 약 3.3h, ×3 약 1.8h, ×2 약 1.5h, 합계 약 6.6h.
    - 지표: d(복원)는 이미지당 0.106초 기준 약 2.4h다. e(무참조)는 측정하지 않았고 0.1~0.2초로 가정해 약 2.3~4.5h다. a·b·c는 수십 분이다. 합계 약 5~7h다.
    - Stage 1 합계는 약 12~17h다(×2 공식 가중치를 쓰면 swinir·hat·mambairv2 ×2분 약 0.3h가 늘어난다).
- 산출물: `results/stage1_metrics.csv`, `results/stage1_ranking.csv`, `STAGE1_REPORT.md`. 사용자 승인 뒤 생성 단계로 넘어간다.

## 3. Stage 2 시간 추정 (학습 1회, 레시피 epoch)
| 모델 | ×3 | ×4 | 근거 |
|---|---:|---:|---|
| BoT b64 | 약 1.0h | 약 1.8h | 3e [새] 384.6 / 218.4 img/s, 183 × 120 iteration |
| CLIP-ReID b64 | 약 1.1h | 약 2.2h | stage2 274.4 / 111.2 img/s, 185 × 60 + stage1·특징 캐시 |
| KPR b64 | 약 4.4~6.2h | 약 7.2h | 4090에서 레시피 배치 64 불가 → 대여. ×3 [새] 63~69 img/s([이전] 90), ×4는 55 img/s 가정 |
| KPR+SOLIDER b64 | 같음 | 같음 | 같음 |

- 합계(24회)
  - BoT·CLIP-ReID 12회: 약 18h(4090).
  - KPR 계열 12회: 약 70~80h(대여).
- 평가는 경량 평가기를 쓴다. 회당 수 분이고 KPR ×4 PSS는 약 15GiB다.
- 대여 비용: A100 80GB $1.39~1.59/h면 약 $100~130. H100은 $1.99/h부터(RunPod 공식, `docs/HW_SOURCES.md`). 대여 뒤 300 iteration으로 처리량을 다시 잰다.

## 4. GPU·서버 권장 (VRAM은 Part H [이전] 값 유지)
- KPR ×3 b64: allocated 19.5GiB(실측). L40S 48GB로 충분하다.
- KPR ×4 b64: allocated 34.0GiB, 점유 44~49GiB 추정. A100/H100 80GB가 안전하고 L40S는 경계다.
- RAM은 64GiB 이상(KPR 학습 PSS 최대 23~27GiB, cgroup 최대 54.7GiB [새])이 필요하다. CPU 할당은 16코어 이상을 권장한다.

## 5. 유지되는 확정 결정
memmap 4종, BoT worker_norm, KPR copy_read·float32 마스크, BIPO는 학습에서만, 핀 메모리 켬, 스레드 제한(OMP/MKL 4), WebP method 2, 다른 프로젝트 삭제 금지.

## 6. 남은 질문
- Stage 1 선정 규칙 질문 4개: `STAGE1_SELECTION_RULE.md` 7절.
- KPR 학습 서버(L40S 대 A100/H100)와 시점.
