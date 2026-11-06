# Stage 0.5 지시 3a~3d 결과 정리 (이전 서버, 재실행 없음)

모든 수치는 **이전 서버**(RTX 4090 24GB, cgroup v1, 57.7GiB) 측정값이다. 비교 양쪽이 같은 서버에서 돌았으므로 판정은 유효하다.
체인 로그: `logs/stage05/run_train.log` ("run_3abcd 종료"). 완료 표시: `runs/stage05/done/{h7_*,d9_*_wnorm,d10_det_*,h4_*_split512}.json`.
정리일 2026-11-06. 3d 측정점을 반영해 `results/stage05_vram_extrapolation.csv`를 다시 적합했다(CPU만, `stage05/fit_vram.py`). 3d 이전 적합 결과는 세션 스크래치에만 보관했다.

## 1. 모델별 등가성 판정과 memmap 채택 (이전 서버)
결정론 모드, ×3(384×192), 200 iteration, 워커 4. 출처: `results/stage05_train_equiv_det.csv`, `results/stage05_data_equiv.csv`.

| 모델 | 비교 | 최대 loss 차이 | 첫 차이 iteration | 판정 |
|---|---|---:|---:|---|
| BoT | 기준선 vs memmap (GPU 정규화) | 0.2325 | 0 | 불통과: GPU 정규화의 입력 차이 최대 7.15e-7 |
| BoT | 기준선 vs memmap+worker_norm | 0 | 없음 | **통과, 채택** |
| BoT | 기준선 vs 최종(memmap+worker_norm+no_timers+prefetch4) | 0 | 없음 | 통과 |
| CLIP-ReID | 기준선 vs memmap | 0 | 없음 | **통과, 채택** (입력 차이 1.19e-7, loss 차이 0) |
| CLIP-ReID | 기준선 vs memmap+worker_norm | 0 | 없음 | 통과 |
| CLIP-ReID | 기준선 vs 최종(+no_timers+prefetch4+pin) | 0 | 없음 | 통과 |
| KPR | 공식 로더 vs memmap+f32 | 1.0943 | 30 | 불통과: 공식 로더 캐시 오염 탓 (2절) |
| KPR | copy_read vs copy_read 반복 | 0 | 없음 | 잡음 0 |
| KPR | copy_read vs memmap+f32 | 0 | 없음 | **통과, 채택** (copy_read 기준) |
| KPR | 공식 로더 vs memmap+u8 마스크 | 0.9013 | 0 | 불통과: u8 마스크 기각 (target_masks 차이 최대 0.999) |
| KPR+SOLIDER | 공식 로더 vs memmap+f32 | 1.8359 | 2 | 불통과: 캐시 오염 탓 |
| KPR+SOLIDER | copy_read vs memmap+f32 | 0 | 없음 | **통과, 채택** (copy_read 기준) |
| KPR+SOLIDER | 공식 로더 vs memmap+u8 마스크 | 1.3421 | 0 | 불통과: u8 마스크 기각 |

memmap 채택 확정:
| 모델 | 채택 구성 | 기준 로더 |
|---|---|---|
| BoT | memmap + worker_norm | 공식 경로와 입력·loss 비트 동일 |
| CLIP-ReID | memmap (+worker_norm 동일) | 공식 경로와 loss 비트 동일 |
| KPR | memmap + mask_f32 | copy_read (공식 로더의 캐시 오염을 고친 로더) |
| KPR+SOLIDER | memmap + mask_f32 | copy_read |

KPR 입력 비교(`stage05_data_equiv.csv`): memmap+f32의 마스크·키포인트·pid·camid는 공식 로더와 200배치 모두 같고, 이미지는 KPR iteration 30의 1배치와 SOLIDER iteration 2의 6배치에서만 다르다. 이 배치들이 오염된 캐시를 다시 읽은 배치다. no_timers와 prefetch4는 입력을 바꾸지 않는다(all_f32 결과가 memmap 결과와 같음).

## 2. KPR lru_cache 오염 원인
- `third_party/keypoint_promptable_reidentification/torchreid/utils/tools.py:104` `@lru_cache(maxsize=None) def read_image`: 워커 안에서 디코딩된 numpy 배열 객체를 캐시하고 그 객체를 그대로 돌려준다. 같은 패턴이 `:130` read_masks, `:159` read_keypoints에도 있다.
- `.../torchreid/data/datasets/dataset.py:398` `transf_args["image"] = read_image(sample["img_path"])`: 캐시 객체가 albumentations 파이프라인으로 들어간다.
- `albumentations/augmentations/geometric/functional.py:389-390` (srreid-kpr env): 목표 크기와 같으면 `resize`가 `return img`로 같은 객체를 돌려준다. SR 입력(×3 384×192)이 정확히 이 경우다. 공식 128×64→384×128은 새 배열이 생겨 해당하지 않는다.
- `.../torchreid/data/data_augmentation/batch_wise_inter_person_occlusion.py:66-68` (`apply` → `paste_over`), `:443-446` `im_dst[...] = alpha*color_src + (1-alpha)*region_dst`: BIPO가 제자리에서 쓴다. 그래서 캐시된 원본 이미지에 가림 사람이 남고, 같은 워커가 그 이미지를 다시 읽으면 오염된 이미지가 나온다.
- 수정(copy_read, `stage05/backends.py:304-307`): `read_image`의 반환값만 `.copy()`한다. 캐시는 유지되고 BIPO는 복사본에만 쓴다. 공식 로더 대비 변경점은 이것 하나다.
- 결과: copy_read 반복 간 잡음 0. copy_read와 memmap+f32는 200 iteration 비트 동일. 공식 로더와 copy_read의 차이(KPR 1.094 @30, SOLIDER 1.836 @2)는 공식 로더와 memmap의 차이와 정확히 같다. 즉 D.5의 KPR 불통과는 memmap 탓이 아니라 공식 로더의 캐시 오염 탓이다.

## 3. CLIP-ReID ×4 checkpointing 등가성 (3a, 이전 서버)
cpu_up 모드, 200 iteration. 출처: `runs/stage05/done/h7_curve_*`, `h7_step_*`.

| 모델 | ×4 배치 | off vs off2 | off vs on (loss) | grad norm | allocated 피크 off → on |
|---|---:|---|---|---|---|
| CLIP-ReID | 32 | 비트 동일 | 비트 동일, 첫 차이 없음 | 유한값 모두 동일 | 15.19 → 3.40 GiB |
| KPR | 16 | 비트 동일 | 비트 동일 | 동일 | 15.44 → 9.42 GiB |
| KPR+SOLIDER | 16 | 비트 동일 | 비트 동일 | 동일 | 15.44 → 9.42 GiB |

초기 iteration의 비유한 grad norm(CLIP 0~4, KPR 0~8, SOLIDER 13개)은 AMP grad scaler 초기 스킵이다. on과 off에서 위치가 같다. 1스텝 비교(h7_step, ×3·×4)도 grad 벡터 차이 0, loss 범위 0이다. 피크가 줄었으므로 checkpointing이 실제로 동작했다. 판정: **×4 checkpointing 등가, 사용 가능**.

## 4. Part H: KPR 배치 64 VRAM (이전 서버, 3d 반영 재적합)
적합식: allocated 피크 = a + b·B + c·B² (배치 64 실측은 검증용으로 적합에서 제외). 실제 점유 = allocated(+0.356 오프셋) × 비율(하한 1.26, 상한 reserved/allocated 실측 최대) + 드라이버 0.88GiB. 사용 가능 비율 0.98.

| 모델 | 입력 | ckpt | 측정 배치 | a, b, c | b64 allocated (95% 구간) | 점유 하한~상한 GiB | 판정 |
|---|---|---|---|---|---|---|---|
| KPR | ×3 | 1 | 16~48 | 0.759, 0.2914, 1.3e-5 | 19.46 ±0.04 | 25.8 ~ 28.7 | 48GB |
| KPR | ×3 | 0 | 12~40 | 0.965, 0.4854, 9.1e-5 | 32.40 ±0.39 | 41.7 ~ 42.6 | 48GB |
| KPR | ×4 | 1 | 12~40 | 0.725, 0.5208, −1.6e-5 | 33.99 ±0.06 | 44.1 ~ 49.3 | 48GB(하한) / 80GB(상한) |
| KPR | ×4 | 0 | 8~22 | 0.929, 0.8843, −3.3e-5 | 57.39 ±1.70 | 71.5 ~ 75.8 | 80GB |
| KPR+SOLIDER | ×3 | 1 | 16~40 | 0.745, 0.2925, −8e-6 | 19.43 ±0.06 | 25.8 ~ 28.5 | 48GB |
| KPR+SOLIDER | ×4 | 1 | 12~40 | 0.735, 0.5200, −3e-6 | 34.00 ±0.08 | 44.1 ~ 49.2 | 48GB(하한) / 80GB(상한) |
| KPR+SOLIDER | ×4 | 0 | 8~22 | 0.928, 0.8842, −1.6e-5 | 57.45 ±2.11 | 71.1 ~ 76.4 | 80GB |

검증: KPR ×3 b64 ckpt1을 `max_split_size_mb:512`로 4090에서 직접 측정했다. allocated 19.457GiB로 예측 19.46과 맞는다. 같은 설정의 100 iteration 학습(h6)도 KPR 0.864 it/s, SOLIDER 0.870 it/s로 끝까지 돌았지만, nvidia-smi 피크가 23,868MiB(24,564MiB 중)로 여유가 거의 없다.

**GPU 권장 (이전 서버 측정 근거)**
- ×3 배치 64: L40S 48GB로 충분(ckpt 켬 26~29GiB, 끔 42~43GiB). 4090도 ckpt+max_split 512로 돌지만 여유가 0.7GiB 수준이라 장시간 학습에는 권하지 않는다.
- ×4 배치 64: ckpt 켬은 allocated 34.0GiB로 확정적이다. 점유는 단편화 비율에 따라 44~49GiB이므로 L40S(사용 가능 47.0GiB)에서는 `max_split_size_mb:512`가 필요하고 경계에 걸린다. A100/H100 80GB는 ckpt 끔(71~76GiB)까지 들어간다.
- 결론: ×3만이면 L40S, ×4를 배치 64·ckpt 끔으로 돌리거나 여유를 원하면 A100/H100 80GB. 최종 선택은 사용자 결정 사항이다.
