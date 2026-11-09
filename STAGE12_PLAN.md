# Stage 1·2 계획 (2026-11-06 설계 변경, 2026-11-09 Stage 2 대상 확정)

이전 Stage 1 설계(검증 분할 651/100, BoT·CLIP-ReID 12종 학습, seed 추가, 두 모델 순위 비교, Stage 1 병렬화)는 **전부 취소**했다. seed는 42 하나다. 시간에는 측정 서버를 표시한다([새] = RTX 4090, EPYC 7282, cgroup v2).

## 1. 전체 구조
1. **Stage 1, SR 기법 비교 (학습 없음)**
   - train 표본(751명 × 3장, 2,250장)에 12종 × {×3, ×4} SR을 적용한다.
   - 정답 없이 원본 대비 지표와 시각화만 만든다. 선정(2개)은 사용자가 직접 한다. test는 쓰지 않는다.
2. **생성** (2026-11-09 확정): SR 3개 = **swinir, pisa_sr λ0.0(pisa_l0), osdhuman**. train 전체 12,936장 + test(query 3,368, gallery 19,732) × {×3, ×4}. bicubic train은 기존 `data/bicubic_x{3,4}` 재사용, test ×3·×4는 새로 생성. Stage 1과 같은 설정(swinir ×3 공식 가중치, pisa·osdhuman ×3 = ×4 후 PIL BICUBIC 축소, 결정적 설정 + 예열 + 파일명 seed, b1). test는 사용자가 잠금을 풀고 다시 잠그는 절차로 만들고 출력은 잠긴 영역(`data/locked/sr/`)에 둔다.
   - 가속: GPU 작업 4개(swinir ×3·×4, pisa ×4, osdhuman ×4)를 조각으로 나눠 GPU 2장 × 4 프로세스 공유 대기열. 조각별 20장 단독 실행 비트 비교, Stage 1 표본과 비트 비교, condition·split별 manifest.
3. **Stage 2**: 모델 4종 × {bicubic, swinir, pisa_l0, osdhuman} × {×3, ×4} = 학습 32회, seed 42.
   - 평가는 SR 조건마다 2×2로 (bicubic 학습 모델, SR 학습 모델) × (bicubic test, SR test). mAP·Rank-1, 신원 단위 부트스트랩 95% CI, bicubic 대비 useful / harmful / inconclusive. 경량 평가기.
   - 모델·배율마다 평가 10회: C 모델로 bicubic·SR 3종 test, SR 모델 3개로 각자 SR test와 bicubic test.
   - 학습: 공식 레시피 기본 설정(결정론 모드 끔, cudnn.benchmark 등 공식 기본값), 배치·정밀도·compile 변경 금지, gradient checkpointing은 메모리가 허용하면 끔(비트 동일 확인됨).
   - 가속: KPR 계열 ×4 먼저, 남는 자리에 BoT·CLIP-ReID(GPU당 2~3개), 200 iteration 조합 시험, 자원 기반 자동 대기열, memmap 미리 준비, 평가 겹치기.

## 2. Stage 1 세부 (2026-11-06 2차 교체)
- 이전 Stage 1 세부(선정 규칙, ReID 모델 기반 지표, 속성 모델, ×2 등)는 취소했다. `STAGE1_SELECTION_RULE.md`는 "취소됨"으로 남긴다.
- Stage 1은 순위·선정 없이 지표와 시각화만 만든다. 선정은 사용자가 한다. 정의와 단계는 `CLAUDE.md` 1절에 있고, 결과 보고서는 `STAGE1_REPORT.md`·`STAGE1_REPORT.html`이다.
- 표본: `configs/stage1_sample.csv`(`stage1/make_sample.py`, seed 42). 2,250장, 751명(3장 748명, 2장 3명). 카메라 3대 701명, 2대 50명. 카메라 분포 c1~c6 460 / 377 / 491 / 164 / 390 / 368.
- 0단계 결과(모델 로드, 재현성, ×3 방식 비교, 예상 시간·용량)는 `results/stage1/step0_*.csv`에 있다.

## 3. Stage 2 시간 추정 (학습 1회, 레시피 epoch, 4090 기준 초안. A100 실측은 학습 시작 전 시험으로 다시 잰다)
| 모델 | ×3 | ×4 | 근거 |
|---|---:|---:|---|
| BoT b64 | 약 1.0h | 약 1.8h | 3e [새] 384.6 / 218.4 img/s, 183 × 120 iteration |
| CLIP-ReID b64 | 약 1.1h | 약 2.2h | stage2 274.4 / 111.2 img/s, 185 × 60 + stage1·특징 캐시 |
| KPR b64 | 약 4.4~6.2h | 약 7.2h | 4090에서 레시피 배치 64 불가 → 대여. ×3 [새] 63~69 img/s([이전] 90), ×4는 55 img/s 가정 |
| KPR+SOLIDER b64 | 같음 | 같음 | 같음 |

- 합계(초안은 24회 기준. 확정 32회는 A100 시험 뒤 갱신)
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
- (해결) 학습 서버: A100 SXM4 80GB × 2 (2026-11-08~), $3.59/시간.
