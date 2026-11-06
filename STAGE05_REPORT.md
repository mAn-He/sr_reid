# Stage 0.5 보고서 (재작성)

2026-11-06에 `results/`의 CSV·JSON과 `logs/stage05/`, `runs/stage05/done/`에서 다시 작성했다. 이전 판은 서버 이전 때 남지 않았다.

**서버 표시:** [이전]은 이전 서버(RTX 4090, cgroup v1, 57.7GiB), [새]는 새 서버(RTX 4090, EPYC 7282, cgroup v2, 116.4GiB)다. 표시가 없는 값은 서버와 무관하다(정확도, 픽셀·비트 비교, 파일 크기, VRAM). VRAM은 같은 GPU 모델이라 [이전] 값을 유지한다. 시간·처리량·RAM은 [새]에서 다시 재서 나란히 적는다.

## 0. 요약
- 기준선(공개 가중치, 공식 입력)은 공개 수치와 0.1%p 이내로 재현된다(1절).
- 키포인트는 픽셀 중심 규약 `(x+0.5)k−0.5`에서 오차 0이다. 마스크는 float32로 확정했다(2절).
- KPR 공식 평가기는 RAM 피크 약 128GiB(코드 계산)라 쓸 수 없다. 경량 스트리밍 평가기가 공식과 AP·CMC 일치하며 RAM PSS 8.9GiB(×4 약 15GiB)다(3절).
- memmap은 4개 모델 모두 채택했다. KPR 계열은 공식 로더의 lru_cache 오염을 고친 copy_read 로더를 기준으로 한다(5절, `results/stage05_3abcd_summary.md`).
- WebP 무손실 method 2를 채택했다(6절).
- KPR 배치 64는 ×3이면 48GB, ×4이면 48GB 경계 또는 80GB가 필요하다(7절).
- 3e [새]: 학습 RAM은 모든 최종 구성이 57.7GiB 기준을 통과한다(PSS 최대 23.0GiB). 처리량은 KPR 계열 ×3만 [이전] 대비 −30%이고, CPU 스로틀링 탓으로 본다(4.3절).

## 1. 기준선 (공개 가중치, 공식 입력, 경량 평가기, re-ranking 없음)
출처: `results/baseline_public_weights.csv`. 학습하지 않은 참고 행이다.

| 모델 | 입력 | mAP | R1 | 공개 mAP / R1 |
|---|---|---:|---:|---|
| BoT | 256×128 | 85.95 | 94.63 | 85.9 / 94.5 |
| CLIP-ReID | 256×128 | 89.66 | 95.43 | 89.6 / 95.5 |
| KPR | 384×128 | 89.59 | 95.87 | 89.58 / 95.87 |
| KPR+SOLIDER | 384×128 | 93.25 | 96.59 | 93.25 / 96.59 |

## 2. Part 0: 키포인트·마스크 스케일
출처: `results/stage05_scale_check.json`(train 키포인트 2,000 파일, 마스크 200 파일, 파트 채널 7,200), `STAGE0_REPORT.md` 정정 절.
- 키포인트: ×3·×4 모두 픽셀 중심 공식 대비 최대 오차 0.0px, confidence 변화 0.
- 마스크(로더 경로, bilinear): 저장 해상도 반픽셀 기준 평균 차이 ×3 0.020px, ×4 0.025px. ×1 로더 `(c+0.5)k−0.5` 대비 평균 차이 ×3 0.032px, ×4 0.041px. 전체 bilinear 최대 차이 0.092px(×3), 0.104px(×4)로 1px 기준 통과.
- nearest 직접 확대(×3 3.44px, ×4 6.06px)는 로더가 쓰지 않는 경로라 참고값이다.
- 마스크 dtype: float32 확정. u8은 target_masks 차이 최대 0.999, loss 차이 iteration 0부터라 기각했다(5.2절).

## 3. 평가 메모리 (Part B·C, [이전])
- KPR 공식 평가 경로의 RAM 피크는 384×128에서 약 128GiB다(코드 계산 127.8GiB, `docs/PIPELINE_kpr.md:40-41`, §6.2). 원인은 두 가지다.
  - `_feature_extraction`의 `anns.append(data)`가 샘플당 `prompt_masks`(float64)와 `image`를 쌓는다. 이 둘은 이후 읽히지 않는다.
  - gallery `collate`가 concat으로 입력 리스트와 결과를 동시에 들고 있어 피크가 두 배가 된다.
  - 같은 계산으로 384×192는 191.6GiB, 512×256은 340.3GiB다. 부분집합 실측은 감시 한도(48GiB PSS)에서 끊겼다(`results/stage05_ram.csv`의 official_subset 행).
- 경량 평가기(특징만 추출 → 스트리밍 거리·AP)는 공식 평가기와 AP·CMC가 일치한다. 차이는 최대 1.1e-16이고 블록 256/512/1024 모두 같다(`results/stage05_eval_equiv.csv`). 특징 배열도 비트 동일이다(`results/stage05_light_feature_check.json`).

| 경량 평가기 RAM (핀 메모리·occluder 끔) [이전] | RSS 합계 | PSS 합계 |
|---|---:|---:|
| KPR 384×128 추출 | 30.9GiB | 8.85GiB |
| KPR+SOLIDER 384×128 추출 | 30.6GiB | 8.20GiB |
| KPR 384×192 추출 | 34.8GiB | 12.4GiB |
| KPR 512×256 추출 | 36.8GiB | 14.7GiB |
| KPR+SOLIDER 512×256 추출 | 37.0GiB | 14.9GiB |
| 스트리밍 평가 단계 | 4~5GiB | 4~5GiB |

RSS는 워커 간 공유 페이지가 중복 집계된 값이다. 판정은 PSS로 한다. 57.7GiB와 116.4GiB 모두 여유가 있다.

## 4. 학습 RAM (Part B3, 3e)
### 4.1 공식 로더 500 iteration [이전]
출처: `results/stage05_ram.csv`(train 행), cgroup은 `results/stage05_ram_cgroup.csv`(usage = memory.current 등가, v1 usage_in_bytes).

| 모델 | 입력 | 배치 | 워커 | RSS 합계 | PSS 합계 | cgroup usage 피크 | 57.7GiB 기준 | 116.4GiB 기준 |
|---|---|---:|---:|---:|---:|---:|---|---|
| BoT | ×3 / ×4 | 64 | 4 | 4.5 / 5.2 | 1.9 / 2.7 | — | 여유 | 여유 |
| CLIP-ReID | ×3 / ×4 | 64 | 4 | 15.2 / 15.4 | 4.2 / 4.5 | — | 여유 | 여유 |
| KPR | ×3 | 48 | 4 | 59.5 | 21.0 | — | 통과 | 통과 |
| KPR | ×3 | 48 | 8 | 86.0 | 26.6 | 51.2 | 통과(근접) | 통과 |
| KPR | ×4 | 24 | 4 | 76.5 | 27.0 | 51.1 | 통과(근접) | 통과 |
| KPR | ×4 | 24 | 8 | 233.5 | 42.5 | **57.74(한도)** | **한도 도달** | 통과 |
| KPR+SOLIDER | ×3 | 48 | 4 | 49.3 | 20.5 | 40.6 | 통과 | 통과 |
| KPR+SOLIDER | ×4 | 24 | 8 | 120.3 | 30.5 | 51.5 | 통과(근접) | 통과 |

cgroup usage에는 회수 가능한 페이지 캐시가 포함된다. 작업 집합(usage − inactive_file)은 `stage05_ram_cgroup.csv`의 working_set 열에 있다. KPR 계열은 워커 4가 57.7GiB 기준 보수 판정에 맞는다.

### 4.2 핀 메모리 on/off [이전]
출처: `results/stage05_throughput.csv`(D 핀 메모리 비교, ×3, memmap). 핀 메모리 끄면 처리량이 줄었다: BoT −6.5%, CLIP-ReID −9.2%, KPR −19.2%, KPR+SOLIDER −20.8%. PSS는 KPR 20.1 → 13.5GiB로 줄었다.

### 4.3 3e 새 서버 결과 [새]
출처: `results/stage05_srv2_throughput.csv`, `results/stage05_ram_cgroup.csv`(server=새 서버, cgroup v2 `logs/stage05/cgroup_mem_v2.csv`). 2026-11-06 07:31~09:31에 단독으로 실행했다(tmux `s3e`, rc=0). 300 iteration(워밍업 20), 워커 4, 최종 플래그로 돌렸다. KPR 계열은 공식 BIPO를 켰다.

**학습 RAM [새] (GiB)**

| 단계 | 배치·ckpt | PSS 합계 | RSS 합계 | cgroup memory.current 최대 | anon+shmem 최대 | 작업 집합 최대 | 57.7 기준 | 116.4 기준 |
|---|---|---:|---:|---:|---:|---:|---|---|
| BoT ×3 / ×4 | 64·0 | 4.7 / 7.5 | 7.2 / 9.6 | 28.0 / 34.0 | 2.3 / 3.6 | 9.5 / 15.5 | 여유 | 여유 |
| CLIP-ReID ×3 / ×4 | 64·0 / 64·1 | 7.5 / 10.3 | 18.5 / 20.5 | 35.8 / 37.1 | 5.1 / 6.4 | 17.3 / 18.6 | 여유 | 여유 |
| KPR ×3 | 48·1 | 19.7 | 47.5 | 47.8 | 16.3 | 29.1 | 통과 | 여유 |
| KPR ×4 | 24·1 | 23.0 | 58.8 | 50.3 | 18.8 | 31.6 | 통과 | 여유 |
| KPR+SOLIDER ×3 | 48·1 | 20.5 | 52.5 | 49.4 | 17.9 | 30.7 | 통과 | 여유 |
| KPR+SOLIDER ×4 | 24·1 | 22.2 | 58.3 | 50.0 | 18.6 | 31.3 | 통과 | 여유 |
| (참고) KPR ×4 b16 ckpt0 (d12) | 16·0 | 24.5 | 89.8 | **54.7** | 23.3 | 36.0 | 통과(근접) | 여유 |

- memory.current에는 회수 가능한 페이지 캐시가 들어간다. 새 서버에서는 학습 전부터 캐시가 약 23GiB 차 있었다(cgroup 기록 첫 줄). anon+shmem과 작업 집합은 57.7GiB보다 한참 아래다. OOM·oom_kill·max 이벤트는 0이다.
- 보수 판정(57.7GiB, limit_gib=48 PSS): 모든 최종 구성이 통과한다. [이전] 워커 4 PSS(KPR ×3 21.0, ×4 27.0)와 같은 수준이다.

**최종 구성 처리량 (img/s, 워커 4). [이전] 값은 D.6 fast(같은 플래그, BoT·CLIP은 worker_norm 없음)다.**

| 모델 | ×3 [이전] → [새] | GPU 사용률 [새] | ×4 [이전] → [새] | GPU 사용률 [새] |
|---|---|---:|---|---:|
| BoT b64 | 393.6 → 384.6 (−2.3%) | 99% | 223.3 → 218.4 (−2.2%) | 98% |
| CLIP-ReID b64 (+pin) | 250.9 → 274.4 (+9.3%) | 87% | 105.5 → 111.2 (+5.4%) | 96% |
| KPR b48 / b24 | 90.0 → **63.3 (−29.7%)** | 58% | 55.2 → 53.9 (−2.4%) | 87% |
| KPR+SOLIDER b48 / b24 | 88.8 → **62.1 (−30.0%)** | 59% | 54.2 → 56.0 (+3.1%) | 89% |

CLIP-ReID가 빨라진 것은 최종 구성에 worker_norm과 pin을 넣은 효과로 보인다(같은 플래그끼리 비교하면 아래 d8에서 −1.1%).

**핀 메모리 on/off [새] vs [이전] (×3, 플래그 memmap+pin / memmap+no_pin, 이전 서버 d8과 같은 설정)**

| 모델 | pin [이전] → [새] | no_pin [이전] → [새] | pin 끔 효과 [이전] / [새] |
|---|---|---|---|
| BoT | 394.1 → 391.1 | 368.4 → 363.2 | −6.5% / −7.1% |
| CLIP-ReID | 272.6 → 269.6 | 247.5 → 241.7 | −9.2% / −10.4% |
| KPR | 87.4 → 63.3 | 70.6 → 59.6 | −19.2% / −5.8% |
| KPR+SOLIDER | 88.4 → 62.1 | 70.0 → 57.6 | −20.8% / −7.2% |

핀 메모리는 계속 켠다. 새 서버 KPR 계열은 다른 병목 때문에 pin 효과가 작게 보인다.

**checkpointing 속도 비 [새] (같은 배치, 150 iteration, ckpt 켬 / 끔 img/s)**

| 조건 | 끔 | 켬 | 비 |
|---|---:|---:|---:|
| CLIP-ReID ×4 b32 | 136.2 | 107.5 | 0.79 |
| KPR ×3 b32 | 72.8 | 67.9 | 0.93 |
| KPR ×4 b16 | 57.6 | 55.4 | 0.96 |
| KPR+SOLIDER ×3 b32 | 70.4 | 68.0 | 0.97 |
| KPR+SOLIDER ×4 b16 | 61.3 | 52.7 | 0.86 |

KPR 계열은 [새]에서 CPU 병목이 끼어 ckpt 비용이 실제보다 작게 보일 수 있다. CLIP-ReID 0.79가 GPU 쪽 비용에 가까운 값이다.

**KPR ×3 저하 진단 [새] (d13, 3e 뒤 단독 실행)**
- 새 서버 cgroup은 CPU 13.6코어(`cpu.max 1360000 100000`)이고 `nproc`은 64다. torch 기본 intra-op 스레드가 64개다. 저장소에는 스레드를 제한하는 코드가 없다.
- 컨테이너 시작 이후 누적 `cpu.stat`: nr_throttled 42,637 / nr_periods 154,874, throttled 54,318초.
- OMP/MKL 스레드를 4·8로 제한한 KPR ×3 300 iteration 결과
  - KPR: 63.3 → 69.3 (omp4) · 68.5 (omp8) img/s.
  - KPR+SOLIDER: 62.1 → 68.5 (omp4) · 68.7 (omp8) img/s.
  - 약 +9~10%이고 GPU 사용률은 66~68%다. [이전] 90.0에는 못 미친다.
- 제한 후에도 구간의 약 48%가 스로틀링됐다. system 시간이 user의 약 2~2.4배라 커널 쪽(페이지 폴트·공유 메모리 전달 추정) 부담이 남은 원인으로 보인다. 확정은 못 했다.
- ×4는 GPU 쪽이 병목이라(사용률 87~89%) 영향이 거의 없다.
- 학습 레시피와는 무관하다. 스레드 제한 채택 여부는 사용자 결정 사항이다(문서에 없음).

## 5. Part D: 처리량·등가성
### 5.1 D.5 등가성 (200 iteration, ×3)
D.5 원판정은 비결정 모드였다(`results/stage05_train_equiv.csv`). 결정론 모드 재판정이 `results/stage05_train_equiv_det.csv`다. 판정은 결정론 모드 값으로 한다 [이전].

| 모델 | 비교 | 결정론 최대 loss 차이 (첫 차이 iter) | 비결정 D.5 원값 | 판정 |
|---|---|---|---|---|
| CLIP-ReID | 기준선 vs memmap | 0 (없음) | 0.084 (잡음 0.225 이내) | 통과 |
| BoT | 기준선 vs memmap | 0.2325 (0) | 0.2214 (0) | 불통과 → worker_norm으로 0 |
| KPR | 기준선 vs memmap+f32 | 1.0943 (30) | 1.1331 (12) | 공식 로더 오염 탓, copy_read 대비 0 |
| KPR+SOLIDER | 기준선 vs memmap+f32 | 1.8359 (2) | 2.3200 (2) | 공식 로더 오염 탓, copy_read 대비 0 |

### 5.2 3a~3d 판정 (상세: `results/stage05_3abcd_summary.md`) [이전]
- memmap 채택 확정: BoT(memmap+worker_norm), CLIP-ReID(memmap), KPR·KPR+SOLIDER(memmap+mask_f32, copy_read 기준). 모두 200 iteration 비트 동일이다.
- KPR lru_cache 오염 원인
  - `tools.py:104` `read_image`가 lru_cache로 같은 배열 객체를 돌려준다.
  - 입력이 목표 크기와 같으면 albumentations `functional.py:389-390`의 resize가 같은 객체를 반환한다.
  - BIPO `paste_over`가 그 배열을 제자리 수정한다(`batch_wise_inter_person_occlusion.py:66-68`, `:443-446`).
  - copy_read는 `read_image` 반환값만 복사한다.
- ×4 checkpointing(CLIP-ReID b32, KPR·KPR+SOLIDER b16)은 켬과 끔이 loss·grad 비트 동일이다. allocated 피크는 CLIP 15.19 → 3.40GiB, KPR 15.44 → 9.42GiB로 줄었다.

### 5.3 D.6 처리량, 최적화 전후 (워커 4) [이전]
출처: `results/stage05_throughput.csv`. fast = memmap+no_timers+prefetch4(KPR은 u8 마스크 아님, D.6 당시 구성).

| 모델 | ×3 기준 → fast (img/s) | 변화 | ×4 기준 → fast | 변화 |
|---|---|---:|---|---:|
| BoT b64 | 348.6 → 393.6 | +12.9% | 214.3 → 223.3 | +4.2% |
| CLIP-ReID b64 | 233.4 → 250.9 | +7.5% | 102.5 → 105.5 (ckpt) | +3.0% |
| KPR b48/b24 | 79.3 → 90.0 | +13.6% | 49.3 → 55.2 | +12.0% |
| KPR+SOLIDER b48/b24 | 72.6 → 88.8 | +22.4% | 47.9 → 54.2 | +13.3% |

워커 8은 BoT +8.7%, CLIP-ReID +5.7%, KPR +6.4%, KPR+SOLIDER +10.0%(×3)다. ×4 워커 8에서 KPR은 +20.3%, KPR+SOLIDER는 +15.4%다.

### 5.4 3e 최종 구성 처리량 [새]
4.3절 표에 있다. 요약: BoT·CLIP-ReID와 KPR 계열 ×4는 [이전]과 ±10% 이내다. KPR 계열 ×3은 CPU 스로틀링 탓에 −30%다(스레드 제한 시 −23%).

## 6. Part E: 저장 형식
### 6.1 WebP 무손실 [이전, bicubic 100장, 13프로세스 병렬]
출처: `results/storage_format.csv`. 모든 형식에서 픽셀 불일치 0이다.

| 배율 | method | 평균 KB | PNG 대비 | 병렬 처리량 (장/s) |
|---|---:|---:|---:|---:|
| ×3 | 2 (채택) | 40.27 | 0.602 | 86.6 |
| ×3 | 6 | 40.20 | 0.601 | 3.0 |
| ×4 | 2 (채택) | 62.92 | 0.622 | 83.3 |
| ×4 | 6 | 62.15 | 0.614 | 1.37 |

method 2는 method 6 대비 용량이 0.2~1.2% 크고 28~61배 빠르다. 실제 SR 이미지 100장 단독 재측정 [새]: 12개 방법 ×3·×4 모두 픽셀 불일치 0이다. PNG 대비 0.60~0.74이고 ×4 평균 62.9~207.3KB, 13프로세스 처리량은 ×4 119~162장/s다(`results/stage05_webp_m2_recheck_s2.csv`, `STAGE05_PART_F_PLAN.md` 4절).

### 6.2 memmap 캐시
- ×3 train memmap은 12,936×384×192×3 = 2.86GB(계산 일치)이고 위치는 `/root/srreid_cache`(로컬 NVMe)다. 학습 시작 때 없으면 만든다.
- 빌드 시간 11.5초는 근거 로그를 찾지 못했다(아래 대조표). [새] 재측정(워커 12, 원본 PNG가 페이지 캐시에 있는 상태): ×3 train 16.7초, ×4 train 20.7초(`results/stage05_memmap_build_s2.csv`).
- KPR 주석 캐시(`data/kpr_cache`, 영구): 마스크 float32 36×17×9, train·query·gallery 합 0.74GiB다. 배율·SR 방법과 무관하게 한 벌이다. 키포인트는 배율별 약 25MB(`results/stage05_kpr_mask_storage.csv`).

## 7. Part H: KPR VRAM (같은 GPU 모델이라 [이전] 값 유지)
출처: `results/stage05_vram.csv`, `results/stage05_kpr_memory.csv`, `results/stage05_vram_extrapolation.csv`(3d 반영 재적합).
- 4090 24GB 측정 범위: KPR ×3는 ckpt+배치 48, ×4는 ckpt+배치 24(기본 할당기)다. `max_split_size_mb:512`이면 ×3 배치 64(ckpt), ×4 배치 40(ckpt)까지 들어간다.
- 배치 64 allocated 피크(적합식 a+bB+cB²)
  - ×3: ckpt 켬 19.46GiB(실측 19.457로 검증), 끔 32.4GiB.
  - ×4: ckpt 켬 34.0GiB, 끔 57.4GiB.
- 점유 추정과 판정
  - ×3: ckpt 켬 26~29GiB, 끔 42~43GiB → 48GB.
  - ×4: ckpt 켬 44~49GiB → 48GB 경계(max_split 필요) 또는 80GB. 끔 71~76GiB → 80GB.
- 권장: ×3은 L40S 48GB. ×4 배치 64는 A100/H100 80GB가 안전하다. L40S는 ckpt 켬 + `max_split_size_mb:512` 조건에서 경계다.

## 8. Part G: SR 품질 지표 도구
- 도구: `stage05/quality_tools.py`(srreid-quality env). LPIPS(alex)·DISTS·PSNR-Y·SSIM-Y는 pyiqa, DreamSim(ensemble), LR 일관성 PSNR, 상·하 영역 ΔE2000을 쓴다.
- 단위 테스트 `stage05/test_quality.py` [새] 전부 통과.
- 소량 동작 확인 [새, train 20장, bicubic ×4 vs 원본 LANCZOS ×4 대용 참조]: 지표 모두 유한값, 이미지당 0.106초(GPU), 초기화 24.6초, 피크 1.25GiB(`results/stage05_quality_smoke_s2.json`). 참조가 대용이라 품질 판단에는 쓰지 않는다.
- DreamSim 가중치는 2026-11-06 `weights/stage05/dreamsim`(실제 디렉터리, 3.0GB, `SHA256.txt`)에 다시 받았다. DISTS·vgg16은 `TORCH_HOME=/workspace/weights/torch`에 받았다(파일명 해시 일치).

## 9. Part F: SR 생성
계획은 `STAGE05_PART_F_PLAN.md`에 있다. 새 서버 SR 스모크(train 100장, 측정 전용, `runs/stage05/sr_smoke`)에서 12개 방법이 전부 성공했다. F1(train ×4) GPU 시간은 14.8~20.5h, 용량은 16.5GiB다 [새]. HAT ×3·×4 가중치는 strict 로드와 forward를 확인했다 [새](`results/stage05_hat_load_check_s2.json`). 일괄 생성은 승인 전 시작하지 않는다.

## 10. 사용자 제시 값과의 대조
| 항목 | 제시 값 | 결과 파일 값 | 판정 |
|---|---|---|---|
| 기준선 mAP/R1 4종 | 85.95/94.63, 89.66/95.43, 89.59/95.87, 93.25/96.59 | 같음 | 일치 |
| KPR 공식 평가 RAM 피크 | 약 128GiB | 127.8GiB(코드 계산, `PIPELINE_kpr.md:41`). N 기준을 달리한 역산표는 130.8GiB(`stage05_ram_estimate.csv`) | 일치 |
| 경량 평가기 RSS/PSS | 30.9 / 8.9GiB | 30.935 / 8.848GiB (KPR 384×128) | 일치 |
| ×4 평가 PSS | 약 15GiB | 14.70(KPR), 14.92(SOLIDER) | 일치 |
| Part 0 키포인트 오차 | 0 | 0.0px | 일치 |
| 마스크 bilinear | 0.02~0.04px | 평균 0.020~0.041px(기준별) | 일치 |
| 마스크 dtype | float32 | float32 | 일치 |
| memmap ×3 train 용량 | 2.86GB | 2.861GB(계산) | 일치 |
| memmap ×3 train 빌드 시간 | 11.5초 | **근거 로그 없음** | 확인 불가. [새] 재측정 16.7초 |
| 학습 처리량 효과 | +14~23% | ×3 워커 4: BoT +12.9, CLIP +7.5, KPR +13.6, SOLIDER +22.4%. ×4: +3.0~13.3% | **부분 불일치**: 14~23%는 KPR 계열 ×3 범위에 가깝다. BoT·CLIP-ReID와 ×4는 3~13% |
| WebP m2 ×3 / ×4 | 40.27KB·86.6장/s / 62.92KB·83.3장/s | 40.27·86.59 / 62.92·83.31 | 일치 |
| WebP m6 ×3 / ×4 | 40.20KB·3.0장/s / 62.15KB·1.37장/s | 40.20·3.02 / 62.15·1.37 | 일치 |
| D.5 CLIP-ReID | 통과 | 통과 | 일치 |
| D.5 BoT | 0.233 (iter 0) | 0.2325 (0), 결정론 모드 | 일치(비결정 원값은 0.2214) |
| D.5 KPR | 1.094 (iter 30) | 1.0943 (30), 결정론 모드 | 일치(비결정 원값 1.1331 @12) |
| D.5 KPR+SOLIDER | 1.836 (iter 2) | 1.8359 (2), 결정론 모드 | 일치(비결정 원값 2.3200 @2) |

## 11. 알려진 기록
- `runs/stage05/done/d9_data_kpr_memmap.json`은 2026-11-01 01:47 쿼터 사건으로 0바이트였다. `stage05_data_equiv.csv`에 KPR memmap 행이 없고 기준선 반복 행이 두 번 있는 것이 그 흔적이다. KPR memmap 등가성은 d10 copy_read 비교로 대신 확정했다. `run_train.py`는 시작 때 이런 표시를 `logs/stage05/script_bug_rows/corrupt_done/`로 옮긴다.
- `data/kpr/x2/bounding_box_train`는 깨진 링크다(→ 11-01 정리 때 지운 `data/bicubic_x2`). 기록만 하고 그대로 둔다.
- 3e 첫 실행(11-06 07:28)은 새 컨테이너에 `/usr/bin/time`이 없어 바로 실패했다(측정 0건, `logs/bg/s3e_try1_no_usr_bin_time.log`). 설치 후 다시 실행했다. setup 스크립트가 이제 설치를 맡는다.
