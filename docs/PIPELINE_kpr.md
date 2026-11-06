# KPR (ImageNet Swin) 파이프라인 정적 분석: Market-1501

코드만 읽어 작성했습니다. 학습과 GPU 실행은 하지 않았고, 이미지 파일과 test 분할은 열지 않았습니다. 이 문서의 메모리 값은 전부 코드에서 유도한 계산값이며 측정값이 아닙니다.

- 대상 commit: `e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa` (`git -C third_party/keypoint_promptable_reidentification rev-parse HEAD`, 작업 트리 변경 0건)
- 대상 config: `configs/kpr/imagenet/kpr_market_train.yaml`, `kpr_market_test.yaml`
- 크기 표기는 H×W입니다. 계획 입력은 ×3 = 384×192, ×4 = 512×256이고, 384×128은 baseline 전용입니다.
- 실행 환경 라이브러리: `env/srreid-kpr.yml` 기준 torch 1.13.0, albumentations 1.3.1, numpy 1.23.5, opencv 4.8.0

경로 약어:

- `K/` = `third_party/keypoint_promptable_reidentification/`
- `A/` = `/workspace/miniforge3/envs/srreid-kpr/lib/python3.10/site-packages/albumentations/`
- `IN-train` = `K/configs/kpr/imagenet/kpr_market_train.yaml`, `IN-test` = `K/configs/kpr/imagenet/kpr_market_test.yaml`
- `dflt` = `K/torchreid/scripts/default_config.py`

인용 형식은 `파일:함수:줄`입니다. 한 번 전체 경로를 쓴 파일은 이후 파일명만 씁니다. 짧은 이름은 아래와 같습니다.

| 짧은 이름 | 경로 (`K/torchreid/` 기준) |
|---|---|
| `builder.py`, `dflt` | `scripts/builder.py`, `scripts/default_config.py` |
| `dataset.py` | `data/datasets/dataset.py` |
| `datamanager.py`, `sampler.py`, `transforms.py` | `data/datamanager.py`, `data/sampler.py`, `data/transforms.py` |
| `mask_transform.py`, `resize.py` | `data/masks_transforms/mask_transform.py`, `data/masks_transforms/resize.py` |
| `pifpaf_mask_transform.py`, `coco_keypoints_transforms.py` | `data/masks_transforms/` 아래 |
| `batch_wise_inter_person_occlusion.py` | `data/data_augmentation/` 아래 |
| `keypoints_to_masks.py`, `market1501.py` | `data/datasets/`, `data/datasets/image/` 아래 |
| `engine.py`, `part_based_engine.py` | `engine/engine.py`, `engine/image/part_based_engine.py` |
| `kpr.py`, `promptable_solider.py`, `promptable_transformer_backbone.py` | `models/` 아래 |
| `swin_transformer.py` | `models/solider/backbones/` 아래 |
| `distance.py`, `rank.py` | `metrics/` 아래 |
| `tools.py`, `torchtools.py`, `tensortools.py`, `writer.py`, `avgmeter.py`, `distribution.py`, `engine_state.py` | `utils/` 아래 |
| `logger.py` | `utils/logging/` 아래 |
| `feature_map_visualization.py`, `visualize_query_gallery_rankings.py` | `utils/visualization/` 아래 |
| `GiLt_loss.py`, `part_averaged_triplet_loss.py`, `body_part_attention_loss.py` | `losses/` 아래 |
| `optimizer.py`, `lr_scheduler.py` | `optim/` 아래 |

## 핵심 요약

1. 평가 때 RAM이 터질 수 있는 원인은 코드상 `_feature_extraction`의 `anns.append(data)`입니다. 샘플당 `prompt_masks` 64·H·W 바이트(float64)와 `image` 12·H·W 바이트가 누적되는데, 이 두 배열은 저장 후 어디에서도 읽지 않습니다 (§6).
2. 계산상 RAM 피크는 384×192에서 약 191.6 GiB, 512×256에서 약 340.3 GiB입니다. 파드 RAM 한도는 cgroup 기준 57.7 GiB입니다 (`free -g`의 251 GB는 호스트 값). 공식 입력 384×128(127.8 GiB)을 포함한 세 크기 모두 공식 평가 경로 그대로면 gallery `collate` 전에 한도를 넘습니다 (§6.2).
3. Stage 0의 KPR "평가 메모리"는 이 누적을 재지 않았습니다. `KprBackend.extract_batch`는 특징을 반환만 하고 저장하지 않습니다 (§6.7).
4. Swin에는 절대 위치 임베딩이 없어 입력 크기를 바꿔도 보간이 필요 없습니다. 윈도우 패딩은 384×128, 384×192, 512×256의 모든 stage에서 발생합니다 (§3.1).
5. `display_feature_maps`는 매 스텝 호출되지만 `test.vis_feature_maps=False`라 학습 config에서 비활성입니다 (§6.6).
6. 기본 설정은 체크포인트를 저장하지 않습니다 (`model.save_model_flag=False`, `dflt:69`). 공식 train yaml은 이 값을 바꾸지 않습니다 (§4).

---

## 1. 학습 경로

### 1.1 진입과 구성

| 단계 | 근거 |
|---|---|
| 진입 | `K/main.py:main:45-54` |
| config 병합 순서: default → yaml → args/opts → 파트 수 계산 → `load_weights`의 config | `K/torchreid/scripts/builder.py:build_config:137-202` |
| datamanager → model → optimizer → scheduler → engine 순 생성 | `builder.py:build_torchreid_model_engine:230-247` |
| `spatial_feature_shape`을 모델 생성 후 cfg에 기록. dataset은 같은 cfg 객체를 공유하고, 작업자는 iteration 때 fork되므로 이후 값을 읽음 | `builder.py:build_model:262`, `dataset.py:Dataset.__init__:92`, `dataset.py:ImageDataset.getitem:386` |
| DataLoader: `RandomIdentitySampler`, `num_workers=workers`, `pin_memory=use_gpu`, `drop_last=True`. `persistent_workers` 인자 없음 | `K/torchreid/data/datamanager.py:ImageDataManager.__init__:228-241` |
| 샘플러가 `(idx, 같은 배치의 다른 pid 집합)`을 내보냄 | `K/torchreid/data/sampler.py:RandomIdentitySampler.__iter__:72-75` |

`persistent_workers`가 없으므로 작업자는 epoch(및 test loader 순회)마다 새로 fork됩니다. 뒤에 나오는 `lru_cache`의 효과와 직결됩니다 (§6.5).

학습 초기화 때 BIPO occluder를 메인 프로세스에 미리 올립니다. `dataset.py:Dataset.__init__:111-114` → `K/torchreid/data/data_augmentation/batch_wise_inter_person_occlusion.py:load_human_occluders:219-329`. ID당 최대 5장을 config 크기로 리사이즈하고 RGBA로 저장합니다 (같은 파일 `:233`, `:246`, `:322`). query, gallery dataset은 `random_occlusions`를 넘기지 않아 올리지 않습니다 (`datamanager.py:305-321`, `:332-349`).

### 1.2 `__getitem__`과 로딩

`K/torchreid/data/datasets/dataset.py:ImageDataset.__getitem__:365-372` → `ImageDataset.getitem:376-533`

| 대상 | 코드 | 형태 | 비고 |
|---|---|---|---|
| 이미지 | `K/torchreid/utils/tools.py:read_image:104-127` (`dataset.py:398`) | H_img×W_img×3 uint8 | `cv2.imread` + BGR→RGB, `@lru_cache(maxsize=None)` |
| 마스크 | `tools.py:read_masks:130-154` (`dataset.py:445`) | 디스크 36×17×9 → HWC 17×9×36 | `np.load`, `@lru_cache(maxsize=None)` |
| 키포인트 | `tools.py:read_keypoints:159-206` (`dataset.py:412-415`) | (S, 17, 3) → (17S, 3) | JSON 파싱, `@lru_cache(maxsize=None)`. 이미지 밖 좌표는 0으로 지움 (`K/torchreid/data/datasets/keypoints_to_masks.py:kp_img_to_kp_bbox:201-207`) |
| 경로 추론 | `dataset.py:Dataset.infer_masks_path:50-52`, `infer_kp_path:54-56` | | 키포인트 파일명은 확장자와 무관하게 `.jpg_keypoints.json` |

train 실측 숫자 (`data/kpr/orig/**/bounding_box_train`, 내용은 수치만 확인):

| 항목 | 값 | 표본 |
|---|---|---|
| `.npy` 개수 | 12,936 | 전체 |
| `.npy` shape, dtype | (36, 17, 9), float32 | 무작위 200개 전부 동일 |
| `.npy` 파일 크기 | 22,160 바이트 | 무작위 200개 전부 동일 |
| `.npy` 값 범위 | 0.0 ~ 0.978 | 무작위 200개 |
| JSON 개수 | 12,936 | 전체 |
| 스켈레톤 수 분포 | 0개 163, 1개 12,521, 2개 250, 3개 2 | 전체 |
| 스켈레톤 keypoints shape | (17, 3) | 전체 |
| `is_target` 키 | 비어 있지 않은 12,773개 모두 있음 | 전체 |
| JSON 크기 | 최소 2, 평균 479, 최대 1,439 바이트 | 전체 |
| 파일명 기준 ID 수 | 751, ID당 2~72장 | 전체 |
| ID당 min(5, n) 합 (occluder 상한) | 3,710 | 전체 |
| 4장 미만 ID | 15개 | 전체 |

### 1.3 augmentation 목록

조립 코드는 `K/torchreid/data/transforms.py:build_transforms:28-224`입니다. `IN-train`은 `data.transforms`를 지정하지 않아 기본값 `["rc", "re", "bipo"]`를 씁니다 (`dflt:186-190`).

실제 적용 순서 (앞에 삽입되는 `:211`까지 반영):

| # | 변환 | 종류 | 확률 | 파라미터 | 근거 |
|---|---|---|---|---|---|
| 1 | `CombinePifPafIntoFiveVerticalParts` | 마스크 그룹핑 36→5, `max` 후 clip | 1 | `five_v` | `transforms.py:209-211`, `K/torchreid/data/masks_transforms/pifpaf_mask_transform.py:211-232`, `mask_transform.py:MaskTransform.__init__:10`, `MaskGroupingTransform.apply_to_mask:41-48` |
| 2 | `Resize` | 이미지, 마스크, 키포인트 | 1 | H×W, `interpolation=1` (INTER_LINEAR) | `transforms.py:83`, `dflt:214`, `masks_transforms/resize.py:26-53` |
| 3 | `BIPO` | 사람 복사 붙여넣기 가림 | **0.05** | `path=""`, `n=1`, `pid_sampling_from_batch=True`, 중심 y 범위 0.55~0.75H | `transforms.py:85-94`, `IN-train:8-9`, `dflt:194-200`, `batch_wise_inter_person_occlusion.py:56-57`, `:143` |
| 4 | `PadIfNeeded` | 패딩 | 1 | `min_height=H+20`, `min_width=W+20`, `BORDER_CONSTANT`, `value=0`, `mask_value=0` | `transforms.py:102-103` |
| 5 | `RandomCrop` | 크롭 | 1 | H×W | `transforms.py:104` |
| 6 | `Normalize` | 이미지만 | 1 (라이브러리 기본) | mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225], max 255 | `transforms.py:73`, `:119`, `IN-train:6-7`, `A/augmentations/transforms.py:199-207` |
| 7 | `CoarseDropout` | random erasing | **0.5** | 구멍 1개, 높이 `int(H*0.15)`~`int(H*0.65)`, 폭 `int(W*0.15)`~`int(W*0.65)`, `fill_value=norm_mean`, `mask_fill_value=0` | `transforms.py:121-126`, `A/augmentations/dropout/coarse_dropout.py:109-157` |
| 8 | `ToTensorV2` | 이미지 CHW 변환. 마스크는 HWC 유지 | 1 | `transpose_mask=False` | `transforms.py:132`, `A/pytorch/transforms.py:75`, `:83-95` |
| 9 | `PermuteMasksDim` | 마스크 CHW | 1 | | `transforms.py:193`, `mask_transform.py:68-73` |
| 10 | `AddBackgroundMask` | 배경 채널 + softmax | 1 | `threshold`, `softmax_weight=15`, `mask_filtering_threshold=0.5` | `transforms.py:215`, `dflt:130-134`, `mask_transform.py:102-122` |
| 11 | `ResizeMasks` | 마스크 해상도를 Hf×Wf로 | 1 | `bilinear` | `transforms.py:218`, `dflt:215`, `mask_transform.py:85-87` |

쓰이지 않는 항목: `rf`(flip, `transforms.py:96-98`), `cj`(color jitter, `:106-116`), `drk`, `dak`(`:160-165`). **좌우 반전은 켜져 있지 않습니다.**

키포인트 처리 설정: `KeypointParams(format='xy', label_fields=['kp_vis_score','kp_indices'], remove_invisible=True, check_each_transform=True)` (`transforms.py:170-174`). 변환마다 이미지 밖 키포인트를 제거합니다 (`A/core/composition.py:202-212`).

주의할 동작:

- `CoarseDropout`은 구멍 안 키포인트를 제거합니다 (`A/augmentations/dropout/coarse_dropout.py:167-175`). 코드에도 FIXME가 있습니다 (`transforms.py:123`).
- `CoarseDropout`은 Normalize 뒤에 있어 `fill_value=norm_mean`은 정규화된 공간에 그대로 들어갑니다 (`transforms.py:119-126`).
- BIPO는 occluder 스켈레톤을 키포인트 리스트에 덧붙이고 (`batch_wise_inter_person_occlusion.py:apply_to_keypoints:106-110`), 이것이 negative 스켈레톤이 됩니다 (`dataset.py:465-471`).

테스트 변환은 `[five_v, Resize(H,W), Normalize, ToTensorV2, PermuteMasksDim, AddBackgroundMask, ResizeMasks]`입니다 (`transforms.py:139-142`, `:155`, `:194`, `:212`, `:216`, `:219`). 크롭, 드롭아웃, BIPO가 없습니다.

### 1.4 프롬프트 마스크 생성

| 항목 | 내용 | 근거 |
|---|---|---|
| 어디서 | `ImageDataset.getitem` 안, DataLoader 작업자 | `dataset.py:479-513` |
| 언제 | `transforms(**transf_args)` 다음, 즉 변환된 키포인트로 매 샘플 매 epoch 새로 생성. 캐시 없음 | `dataset.py:454`, `:483-513` |
| 해상도 | `pose_encoding_strategy=embed_heatmaps_patches`라 **입력 이미지 크기 H×W** | `dflt:152`, `dataset.py:474-477`, `:456` |
| 방식 | `keypoints_gaussian`, `vis_thresh=0.3`, `vis_continous=False`, `g_scale=11` | `dflt:108-110`, `dataset.py:95-98`, `keypoints_to_masks.py:KeypointsToMasks.__init__:270`, `_compute_keypoints_gaussian_heatmaps:319-339` |
| 가우시안 반지름 | `int(W/11)`, 작업자 인스턴스에 1회 캐시 | `keypoints_to_masks.py:get_gaussian_kernel:369-374` |
| 그룹핑 | `cck6`: head, left_arm, right_arm, left_leg, right_leg, torso | `dflt:111`, `K/torchreid/data/masks_transforms/coco_keypoints_transforms.py:81-92` |
| negative 채널 | 다른 스켈레톤별 17×H×W를 만들어 max. 없으면 0 채널 | `dataset.py:486-500` |
| 배경 채널 | `sum`, `softmax_weight=-1` (softmax 없음) | `transforms.py:180-190`, `mask_transform.py:103-106`, `:120-121` |
| 최종 | `[8, H, W]` **float64**, 채널 순서 [배경, negative, 파트 6개] | `dataset.py:500`, `:512`, `K/torchreid/utils/visualization/visualize_query_gallery_rankings.py:324-327` |

dtype 근거: `np.zeros`의 기본 float64 (`keypoints_to_masks.py:321`). `torch.cat`에서 float32 + float64는 float64로 승격합니다. 이 승격은 env Python에서 한 줄로 확인했습니다 (CPU, 파일 입출력 없음).

CPU 비용 힌트 (코드 읽기, 측정 아님). 스켈레톤 1개 샘플 기준:

| 연산 | 근거 | 384×128 | 384×192 | 512×256 |
|---|---|---:|---:|---:|
| 17×H×W float64 생성 | `keypoints_to_masks.py:321` | 6.4 MiB | 9.6 MiB | 17.0 MiB |
| 그룹핑의 fancy index 복사 (5+2+2+2+2+4 = 17채널) | `mask_transform.py:55` | 6.4 MiB | 9.6 MiB | 17.0 MiB |
| 최종 `[8,H,W]` | `dataset.py:512` | 3.0 MiB | 4.5 MiB | 8.0 MiB |

이 외에 파트별 `max`, `clamp`, `stack`, `cat`, `sum`의 중간 결과가 더 있습니다. 전부 H·W에 비례합니다. negative 스켈레톤마다 17×H×W가 한 번 더 듭니다 (`dataset.py:491`, `:496`).

### 1.5 target 마스크 생성

`dataset.py:515-519`에서 `masks[0]`이 `target_masks`가 됩니다. `keypoints.target_masks='none'`이라 키포인트에서 만들지 않습니다 (`dflt:112`, `dataset.py:479`).

경로: `(17,9,36) → (17,9,5) → cv2 linear → (H,W,5) → [BIPO] → 패딩 → 크롭 → [erasing] → (5,H,W) → 배경 + softmax×15 → (6,H,W) → bilinear → (6,Hf,Wf)`

### 1.6 모델

| 단계 | 형태 | 근거 |
|---|---|---|
| 패치 임베딩 (kernel 4, stride 4) | [N, Hs1·Ws1, 128] | `K/torchreid/models/promptable_solider.py:PromptableSoliderSwinTransformer.forward:120`, `:26-28` |
| 프롬프트 토큰을 더함 (conv 8→128, 0 초기화) | 동일 | `promptable_solider.py:121`, `:55-66`, `K/torchreid/models/promptable_transformer_backbone.py:_mask_embed:110-148`, `dflt:172` |
| 카메라 임베딩 (`sie_coe=3`) | 동일 | `promptable_solider.py:123-124`, `:211-213`, `promptable_transformer_backbone.py:_cam_embed:88-104`, `dflt:164-165` |
| Swin 4 stage, depths (2,2,18,2), heads (4,8,16,32), window 7, drop_path 0.1 | 채널 128, 256, 512, 1024 | `promptable_solider.py:33-50`, `:131-141`, `dflt:138` |
| `semantic_weight=0.2` 적용 (기본값, ImageNet 경로에도 적용) | | `promptable_solider.py:133-136`, `:221`, `dflt:170` |
| MSF: stage 2~4를 stage 1 크기로 bilinear 후 concat, 1×1 conv 1920→1024 | [N, 1024, Hf, Wf], Hf=H/4, Wf=W/4 | `K/torchreid/models/kpr.py:MultiStageFusion.forward:351-368`, `kpr.py:KPR.__init__:71-79` |
| 픽셀 분류 | [N, 6, Hf, Wf] | `kpr.py:190-191`, `PixelToPartClassifier:459-468` |
| 가시성 (이진) | [N, 5] bool | `kpr.py:234-242`, `dflt:96-97` |
| 임베딩 풀링 (gwap, 전경, 배경, 전역) | [N, 5, 1024] 등 | `kpr.py:247-254`, `GlobalWeightedAveragePoolingHead.forward:573-586` |
| 풀링 후 차원 축소 | D=512 | `kpr.py:257-261`, `dflt:83-86` |
| BN + 분류기 | | `kpr.py:267-271`, `BNClassifier:481-498` |

사전학습 가중치 파일은 `SOLIDER/swin_base_patch4_window7_224_22k.pth`이고 `convert_weights=True`입니다 (`promptable_solider.py:204-205`).

`semantic_weight`는 ImageNet 경로에서도 양수라 `semantic_embed_*` 층이 만들어지고 쓰입니다 (`K/torchreid/models/solider/backbones/swin_transformer.py:SwinTransformer.__init__:1245-1262`). ImageNet 체크포인트에 이 키가 있는지는 파일을 열지 않아 확인하지 못했습니다.

### 1.7 손실과 가중치

호출: `K/torchreid/engine/image/part_based_engine.py:ImagePartBasedEngine.forward_backward:117-125` → `combine_losses:142-182` → `K/torchreid/losses/GiLt_loss.py:GiLtLoss.forward:47-95`

| 항 | 대상 임베딩 | 가중치 | 근거 |
|---|---|---|---|
| CE (label smoothing) | 전역 `globl` | 1.0 | `dflt:338`, `GiLt_loss.py:44`, `:64-72` |
| CE | 전경 `foreg` | 1.0 | `dflt:341` |
| CE | 파트 concat `conct` | 1.0 | `dflt:344` |
| part-averaged triplet, margin 0.3 | `parts` | 1.0 | `dflt:331`, `:348`, `:354`, `GiLt_loss.py:77-86`, `K/torchreid/losses/part_averaged_triplet_loss.py:35-65` |
| 픽셀 CE (body part attention, label smoothing 0.1, 전 픽셀) | `pixels_cls_scores` vs `target_masks.argmax(1)` | 0.35 | `dflt:332-333`, `:350`, `part_based_engine.py:124`, `:174-179`, `K/torchreid/losses/body_part_attention_loss.py:18`, `:49-61` |

`tr` 가중치는 앞의 세 임베딩에서 0, `parts`의 `id` 가중치는 0입니다 (`dflt:339`, `:342`, `:345`, `:347`). 학습 중 triplet에는 가시성을 쓰지 않습니다 (`mask_filtering_training=False`, `dflt:78`, `GiLt_loss.py:98-101`).

유효 triplet이 없으면 triplet loss가 `None`을 반환합니다 (`part_averaged_triplet_loss.py:161-163`). Stage 0의 배치 4 실패가 이 경로입니다 (`STAGE0_REPORT.md:158`).

### 1.8 옵티마이저와 스케줄러

| 항목 | 값 | 근거 |
|---|---|---|
| 옵티마이저 | SGD, momentum 0.9, dampening 0, nesterov False | `dflt:288`, `:316-318`, `K/torchreid/optim/optimizer.py:162-170` |
| lr | 0.008 | `dflt:289` |
| weight decay | 1e-4, bias 1e-4 | `dflt:291-292` |
| 파라미터 그룹 | `transreid_lr=True`: 파라미터마다 그룹, `bias`만 lr×2 | `dflt:301`, `:312`, `optimizer.py:129-141` |
| 스케줄러 | `CosineAnnealingWarmupRestarts`, `first_cycle_steps=120`, `max_lr=0.008`, `min_lr=1.6e-05`(하드코딩), `warmup_steps=5` | `dflt:293`, `:302`, `K/torchreid/optim/lr_scheduler.py:103-111` |
| step 단위 | epoch | `K/torchreid/engine/engine.py:Engine.train:280`, `update_lr:122-128` |

스케줄러는 모든 그룹의 lr을 하나의 `max_lr` 기반 값으로 덮어씁니다 (`site-packages/cosine_annealing_warmup/scheduler.py:init_lr:46-50`, `get_lr:52-61`, `step:87-88`). 따라서 `optimizer.py:138`의 bias ×2는 실제 lr에 남지 않습니다.

epoch e(0부터)의 lr은 e<5에서 `1.6e-5 + (0.008-1.6e-5)·e/5`, e≥5에서 `1.6e-5 + (0.008-1.6e-5)·(1+cos(π(e-5)/115))/2`입니다.

---

## 2. 평가 경로

### 2.1 특징 추출

`engine.py:Engine.test:285-389` → `part_based_engine.py:_evaluate:237-486` → `_feature_extraction:184-235`

- `@torch.no_grad`이고 autocast가 없습니다 (`part_based_engine.py:237`, `:203`). 평가 텐서는 float32입니다.
- 사용하는 임베딩은 `test_embeddings=["bn_foreg","parts"]`이고 결과는 `[N, 6, 512]`입니다 (`dflt:91-94`, `K/torchreid/utils/tools.py:extract_test_embeddings:224-246`).
- 프롬프트도 평가에 입력됩니다 (`part_based_engine.py:203`, `dflt:147`). 마스크도 test에서 디스크에서 읽습니다 (`datamanager.py:320`, `:348`, `dataset.py:444-445`).
- `load_masks`가 test에서도 켜져 있으므로, README의 "training time only"(`K/README.md:293`)와 달리 평가 경로에도 test 마스크 파일이 필요합니다.

### 2.2 거리

`part_based_engine.py:299-315`

1. L2 정규화: `test.normalize_feature=True` (`dflt:365-367`, `part_based_engine.py:299-302`, `engine.py:normalize:579-580`).
2. 임베딩별 euclidean: `dist_metric="euclidean"` (`dflt:364`, `K/torchreid/metrics/distance.py:_compute_body_parts_dist_matrices:228-237`).
3. 서로 보이는 임베딩만 평균: `mask_filtering_testing=True`, `dist_combine_strat="mean"` (`dflt:79`, `:387`, `distance.py:_compute_distance_matrix_using_bp_features_and_masks:151`, `:157-160`, `K/torchreid/utils/tensortools.py:masked_mean:12-21`).
4. 공통 가시 임베딩이 없는 쌍은 -1 표시 후 최댓값+1로 채움 (`distance.py:174-177`).

### 2.3 지표

`part_based_engine.py:351-362` → `K/torchreid/metrics/rank.py:evaluate_rank:533-574` → `evaluate_py:515-516` → `eval_market1501:99-205`

- `use_cython=False`로 호출해 Python 경로를 탑니다 (`part_based_engine.py:361`).
- 같은 pid, 같은 camid 제거 (`rank.py:127-128`). CMC와 mAP 계산은 표준입니다 (`rank.py:138-158`).
- 출력 rank는 `[1,3,5,10,20]`입니다 (`dflt:368`, `part_based_engine.py:369-370`). `Writer.report_performance`는 `cmc[0..3]`을 r1, r5, r10, r20으로 기록하므로 logger의 이름과 실제 rank가 다릅니다 (`K/torchreid/utils/writer.py:76-79`).

### 2.4 re-ranking

기본값 `test.rerank=False`입니다 (`dflt:371`). 학습 뒤 최종 test는 `rerank` 인자를 넘기지 않아 값과 상관없이 꺼집니다 (`engine.py:236-249`). test 전용 경로만 넘깁니다 (`engine.py:190`). 켜지면 qq, gg 행렬을 추가로 계산합니다 (`part_based_engine.py:316-344`).

### 2.5 학습 중 주기적 평가

| key | 값 | 근거 |
|---|---|---|
| `train.eval_freq` | -1 | `dflt:306-308`, yaml 미지정 |
| `test.start_eval` | 0 | `dflt:370` |
| 판정 | `eval_freq>0`이어야 함. 공식 train yaml은 학습 중 평가 없음 | `writer.py:intermediate_evaluate:97-99`, `engine.py:215-230` |

### 2.6 마지막에 평가되는 것 전부

`engine.py:234-250`, `visrank=cfg.test.visrank=True` (`dflt:372-374`, `:530`)

| 순서 | 항목 | 게이트 | 근거 |
|---|---|---|---|
| 1 | query, gallery 특징 추출 | 없음 | `part_based_engine.py:257-278` |
| 2 | 거리 + `qg_pairwise_dist_statistics` 그림 4개 | `Writer` 존재, `use_logger=True`(기본값) | `distance.py:171-172`, `writer.py:172-186` |
| 3 | CMC, mAP | 없음 | `part_based_engine.py:351-370` |
| 4 | 파트별 ranking 표 | `test.detailed_ranking=False` | `dflt:383-385`, `part_based_engine.py:377-387` |
| 5 | 파트별 거리 분포 그림 | **게이트 없음** | `part_based_engine.py:389-391` |
| 6 | 전체 거리 분포 그림, SSMD | **게이트 없음** | `part_based_engine.py:395-403` |
| 7 | 픽셀 정확도 | `target_masks`, `pxl_scores`가 있으면 실행 | `part_based_engine.py:427-448` |
| 8 | 순위 격자 이미지 10장 | `test.visrank=True` | `part_based_engine.py:450-471`, `writer.py:298-303` |
| 9 | 파트별 순위 격자 | `test.visrank_per_body_part=False` | `dflt:380`, `writer.py:305` |
| 10 | 임베딩 투영 | `test.vis_embedding_projection=False` | `dflt:381`, `writer.py:293-296` |
| 11 | 결과 보고, 체크포인트 저장 | 저장은 `save_model_flag=False`로 무동작 | `engine.py:355`, `:384`, `:250`, `:87` |

---

## 3. 입력 크기 변경 시 영향받는 지점

`data.height`, `data.width`만 바꾸면 변환과 모델이 그 크기를 따릅니다 (따라가지 않는 고정값은 §3.7). 모델 안에서 입력 이미지를 리사이즈하는 코드는 없습니다. `_mask_embed`의 보간은 프롬프트 크기가 입력과 다를 때만 실행되고, 이 경로에서는 같은 크기입니다 (`promptable_transformer_backbone.py:112-118`).

### 3.1 Swin 윈도우와 패딩

패딩 공식: `pad_b=(7-H%7)%7`, `pad_r=(7-W%7)%7` (`swin_transformer.py:ShiftWindowMSA.forward:789-791`). 블록마다 패딩하고 뒤에서 잘라냅니다 (`:850-851`). stage 출력은 다운샘플 전 값입니다 (`SwinBlockSequence.forward:1060-1062`).

아래 표의 패딩 면적은 `1 - HW/(H_pad·W_pad)`입니다.

| 입력 | stage | 특징맵 | pad_b, pad_r | 패딩 후 | 윈도우 수 | 패딩 면적 |
|---|---|---:|---:|---:|---:|---:|
| 384×128 (baseline) | 1 | 96×32 | 2, 3 | 98×35 | 14×5 = 70 | 10.4% |
| | 2 | 48×16 | 1, 5 | 49×21 | 7×3 = 21 | 25.4% |
| | 3 | 24×8 | 4, 6 | 28×14 | 4×2 = 8 | 51.0% |
| | 4 | 12×4 | 2, 3 | 14×7 | 2×1 = 2 | 51.0% |
| 384×192 (×3) | 1 | 96×48 | 2, 1 | 98×49 | 14×7 = 98 | 4.0% |
| | 2 | 48×24 | 1, 4 | 49×28 | 7×4 = 28 | 16.0% |
| | 3 | 24×12 | 4, 2 | 28×14 | 4×2 = 8 | 26.5% |
| | 4 | 12×6 | 2, 1 | 14×7 | 2×1 = 2 | 26.5% |
| 512×256 (×4) | 1 | 128×64 | 5, 6 | 133×70 | 19×10 = 190 | 12.0% |
| | 2 | 64×32 | 6, 3 | 70×35 | 10×5 = 50 | 16.4% |
| | 3 | 32×16 | 3, 5 | 35×21 | 5×3 = 15 | 30.3% |
| | 4 | 16×8 | 5, 6 | 21×14 | 3×2 = 6 | 56.5% |

정리:

- **세 크기 모두 모든 stage에서 패딩이 발생합니다.**
- 패딩 영역은 0으로 채워지고, 이를 가리는 attention mask는 없습니다. `attn_mask`는 shift 경계만 처리합니다 (`swin_transformer.py:802-822`). W-MSA 블록은 mask 자체가 `None`입니다 (`:823-825`).
- `PatchEmbed`, `PatchMerging`의 adaptive padding은 작동하지 않습니다. H, W가 4로, 그 뒤 stage 크기가 2로 나누어떨어지기 때문입니다 (`swin_transformer.py:AdaptivePadding.get_pad_shape:353-363`).
- 512×256 stage 4의 패딩 면적 56.5%는 baseline 51.0%보다 크고, 384×192는 26.5%로 작습니다.

### 3.2 위치 임베딩

| 항목 | 결론 | 근거 |
|---|---|---|
| 절대 위치 임베딩 | **없음.** 보간 불필요 | `SwinTransformer` 생성 인자 `use_abs_pos_embed=False`이고 (`swin_transformer.py:1138`, `:1187-1192`), `promptable_solider.py:33-50`은 이 인자를 넘기지 않음. `:206`의 값은 kwargs로 사라짐 |
| `forward` 게이트 | False | `promptable_solider.py:126-127`, `dflt:154` |
| 상대 위치 편향 | 윈도우 7 고정, 입력 크기와 무관. 표 크기 169 | `promptable_solider.py:36`, `swin_transformer.py:WindowMSA.__init__:664-666` |
| 상대 위치 표 보간 코드 | 체크포인트와 윈도우가 다를 때만 실행, 입력 크기와 무관 | `swin_transformer.py:1345-1360` |
| 프롬프트 토크나이저 | conv라 크기 무관 | `promptable_solider.py:55-63` |
| 생성자의 `pretrain_img_size=[H,W]` | 절대 위치 임베딩이 없어 쓰이지 않음 | `promptable_solider.py:34`, `swin_transformer.py:1150-1157` |

CLIP-ReID처럼 위치 임베딩을 보간해야 하는 구조가 아닙니다 (`STAGE0_REPORT.md:29-34`와 비교).

### 3.3 마스크 해상도

| 항목 | 내용 | 근거 |
|---|---|---|
| 디스크 | 36×17×9 고정. ×k 데이터도 원본 심볼릭 링크 | §1.2, `stage0/prepare_data.py:55-57`, `STAGE0_REPORT.md:131-134` |
| 1차 리사이즈 | 5채널을 H×W로, cv2 INTER_LINEAR. 5채널은 4+1 chunk로 처리 | `resize.py:49-53`, `dflt:214`, `A/augmentations/geometric/functional.py:386-392`, `A/augmentations/utils.py:_maybe_process_in_chunks:190-203` |
| 2차 리사이즈 | softmax 후 `F.interpolate(mode='bilinear')`로 mask_size. `align_corners` 미지정 | `mask_transform.py:85-87`, `dflt:215` |
| mask_size | `spatial_feature_shape[:2]` = (H/4, W/4) | `dataset.py:400`, `promptable_solider.py:79-83` |
| argmax 대상 | `target_masks.argmax(dim=1)` | `part_based_engine.py:174` |
| 코드 주석과 실제 | 주석은 nearest가 최고 성능이라지만 기본값은 bilinear | `mask_transform.py:87`, `dflt:215` |

part attention target 크기와 픽셀 손실 수:

| 입력 | Hf×Wf | 배치 64의 픽셀 수 |
|---|---:|---:|
| 384×128 | 96×32 | 196,608 |
| 384×192 | 96×48 | 294,912 |
| 512×256 | 128×64 | 524,288 |

`body_part_attention_loss`는 `best_pred_ratio=1.0`에서도 전체 픽셀에 `topk`를 돌립니다 (`body_part_attention_loss.py:56`, `dflt:333`). 이 비용도 픽셀 수에 비례합니다.

### 3.4 키포인트 스케일

| 단계 | 처리 | 근거 |
|---|---|---|
| JSON | 이미지 자신의 좌표계라고 간주. 범위 밖은 0 처리 | `dataset.py:412-413`, `keypoints_to_masks.py:201-207` |
| `Resize` | `x·W/W_img`, `y·H/H_img` (단순 비율) | `resize.py:39-44`, `A/augmentations/geometric/functional.py:402-415` |
| 패딩, 크롭 | 이동, 밖이면 제거 | `transforms.py:102-104`, `:172-174` |
| 프롬프트 | `rescale_keypoints`가 img_size→output_size. 같은 크기라 값 불변, assert만 수행 | `keypoints_to_masks.py:211-230`, `:296-298` |

SR 입력에서 유의할 점:

- 파일 크기와 입력 크기가 같으면 `Resize`의 scale은 1.0입니다. JSON이 이미 입력 좌표계여야 합니다. Stage 0은 ×k JSON을 따로 만들었습니다 (`stage0/prepare_data.py:34-37`, `:60-62`, `stage0/scale.py:4-5`).
- Stage 0의 변환은 `(x+0.5)k−0.5`, 공식 `Resize`는 `x·k`입니다. ×3에서 1 px, ×4에서 1.5 px 차이가 납니다 (`stage0/scale.py:4-5`, `STAGE0_REPORT.md:131-134`).
- `stage0/prepare_data.py:45`는 ×2, ×3만 만듭니다. ×4 train 데이터(`data/bicubic_x4`, `data/kpr/x4`)는 `stage05/prepare_measure.py:33-46`이 만듭니다. 이 문서를 쓰는 중에 생긴 파일이라 내용은 확인하지 않았습니다.
- 공식 `process_dir`은 `*.jpg`만 찾습니다 (`K/torchreid/data/datasets/image/market1501.py:85`). PNG는 Stage 0처럼 subclass가 필요합니다 (`stage0/backends.py:317-326`).

### 3.5 프롬프트 마스크 크기

| 입력 | prompt_masks | 샘플당 | 배치 64 (GPU 전송 전 float64) | 가우시안 반지름, 커널 |
|---|---:|---:|---:|---:|
| 384×128 | 8×384×128 | 3.00 MiB | 192 MiB | 11, 23 |
| 384×192 | 8×384×192 | 4.50 MiB | 288 MiB | 17, 35 |
| 512×256 | 8×512×256 | 8.00 MiB | 512 MiB | 23, 47 |

채널 수 8의 근거: `cck6`의 6 + 배경 1 + negative 1 (`promptable_solider.py:195-201`, `K/torchreid/data/__init__.py:compute_parts_num_and_names:22-23`, `dflt:148`, `:114`). 모델 입력 직전에 `prompt_masks.type(image_features.dtype)`로 변환합니다 (`promptable_transformer_backbone.py:125`). 그 전까지 GPU에서도 float64 그대로입니다 (`part_based_engine.py:566-567`, `:199-200`).

가우시안 반지름을 원본 px로 환산하면 baseline은 가로 5.5 px, 세로 3.67 px이고, ×3은 5.67 px, ×4는 5.75 px입니다. baseline은 128×64→384×128이라 가로 2배, 세로 3배로 비등방이기 때문입니다 (코드 값에서 계산).

### 3.6 크롭, 패딩, 드롭아웃 값

| 값 | 384×128 | 384×192 | 512×256 | 크기 연동 | 근거 |
|---|---:|---:|---:|---|---|
| 패딩 | 10 px | 10 px | 10 px | **고정** | `transforms.py:102` |
| 패딩의 상대 폭 (10/W) | 7.81% | 5.21% | 3.91% | | 계산 |
| 패딩의 상대 높이 (10/H) | 2.60% | 2.60% | 1.95% | | 계산 |
| 드롭아웃 높이 | 57~249 | 57~249 | 76~332 | 비율 연동 | `transforms.py:124` |
| 드롭아웃 폭 | 19~83 | 28~124 | 38~166 | 비율 연동 | `transforms.py:125` |

### 3.7 크기에 따라 개념상 달라지지만 코드에는 고정된 값 (변경 제안 아님)

| 값 | 위치 | 비고 |
|---|---|---|
| `pad_size=10` | `transforms.py:102` | 위 표처럼 상대 이동량이 줄어듦 |
| 가우시안 `g_scale=11` | `keypoints_to_masks.py:270`, `:372` | 폭에만 연동, 높이와는 무관 |
| occluder 팽창 커널 5, blur (9,9) | `batch_wise_inter_person_occlusion.py:266`, `:296` | config 크기의 픽셀 단위 고정 |
| `max_occ_per_id=5` | `batch_wise_inter_person_occlusion.py:219` | 개수 고정, 메모리는 H·W 비례 |
| 순위 격자 표시 크기 128×256 | `visualize_query_gallery_rankings.py:36-37` | 표시용 |
| `display_feature_maps` 리사이즈 128×256 | `feature_map_visualization.py:140` | 비활성 |
| joints 히트맵 내부 32×64 | `keypoints_to_masks.py:342` | 이 config에서 미사용 |

그 밖에 occluder 리사이즈의 `cv2.resize(img, img_size, cv2.INTER_CUBIC)`는 세 번째 위치 인자가 interpolation이 아닙니다 (`batch_wise_inter_person_occlusion.py:246`). 이 판단은 cv2 시그니처 일반 지식이며 실행해 확인하지 않았습니다.

### 3.8 입력 크기와 함께 커지는 VRAM, RAM 항목

| 항목 | 식 | 384×128 | 384×192 | 512×256 | 근거 |
|---|---|---:|---:|---:|---|
| 파트 풀링 임시 텐서, 배치 64 (float32 기준) | N·K·1024·Hf·Wf·4 | 3.75 GiB | 5.62 GiB | 10.00 GiB | `kpr.py:574-576` |
| 같은 항목, 배치 128 | | 7.50 GiB | 11.25 GiB | 20.00 GiB | |
| 전경, 배경 풀링 임시, 배치 128 (각각) | N·1024·Hf·Wf·4 | 1.50 GiB | 2.25 GiB | 4.00 GiB | `kpr.py:250-251`, `:544` |
| BIPO occluder RAM (상한 3,710장) | 16·H·W·3710 | 2.72 GiB | 4.08 GiB | 7.25 GiB | `batch_wise_inter_person_occlusion.py:322` |

autocast 아래 `torch.mul` 결과의 dtype은 코드만으로 확정하지 못했습니다. float32로 계산했습니다. 384×192 배치 128 OOM(`STAGE0_REPORT.md:100`)은 이 임시 텐서가 주요인일 가능성이 높습니다. 이는 코드 읽기에서 나온 추정입니다.

occluder는 uint8 RGB와 float32 alpha를 concat하므로 float32입니다 (dtype은 env에서 확인).

---

## 4. 실제 config 값

병합 우선순위는 default → yaml → opts입니다 (`builder.py:137-151`). "yaml"은 값을 yaml이 지정한다는 뜻입니다.

| 항목 | 값 | 출처 |
|---|---|---|
| 입력 H×W | 384×128 | `IN-train:4-5` (기본은 256×128, `dflt:182-183`) |
| 정규화 | mean [0.485,0.456,0.406], std [0.229,0.224,0.225] | `IN-train:6-7` |
| backbone | `imagenet_swin_base_patch4_window7_224` | `IN-train:13` |
| transforms | `["rc","re","bipo"]` | `dflt:186-190` |
| `bipo.p` | 0.05 | `IN-train:8-9` (기본 0.2) |
| 배치 | 64 | `dflt:296` |
| P×K | 16×4 | `dflt:274-276`, `sampler.py:30` |
| 샘플러 | `RandomIdentitySampler` | `dflt:268-270` |
| epoch | 120 | `dflt:294` |
| 1 epoch iteration | 185 (Stage 0 기록) | `STAGE0_REPORT.md:57` |
| `fixbase_epoch` | 0 | `dflt:297` |
| lr | 0.008 | `dflt:289` |
| `reduced_lr` | 0.0002, `staged_lr=False`라 미사용 | `dflt:290`, `:299`, `optimizer.py:80` |
| 스케줄 | cosine warmup 5, min 1.6e-5 | `dflt:293`, `:302`, `lr_scheduler.py:103-111` |
| 옵티마이저 | SGD, momentum 0.9, wd 1e-4 | `dflt:288`, `:291`, `:316` |
| AMP | `mixed_precision=True`. 학습 `forward_backward`의 forward와 loss 구간만. 평가는 없음 | `dflt:311`, `part_based_engine.py:71`, `:99`, `:135-137`, `:203` |
| 파트 수 K (`parts_num`) | 5 | `IN-train:19`, `pifpaf_mask_transform.py:211-229`, `K/torchreid/data/__init__.py:44-48` |
| 프롬프트 파트 수 | 6 | `IN-train:16`, `coco_keypoints_transforms.py:81-92` |
| 프롬프트 채널 | 8, `[8,H,W]`, float64 | §3.5 |
| `test_embeddings` | `["bn_foreg","parts"]`, 합계 6개 | `dflt:91-94` |
| 마스크 해상도와 dtype (코드) | 디스크 그대로 읽고 float32 유지, target은 `[6,H/4,W/4]` | `tools.py:146-147`, `mask_transform.py:85-87` |
| 마스크 해상도와 dtype (train 실측) | (36,17,9), float32, 22,160 바이트 | §1.2 |
| `model.kpr.keypoints.prompt_masks` | `keypoints_gaussian` | `IN-train:15` |
| workers | 4 | `dflt:180` |
| test 배치 | 128 | `dflt:360` |
| 거리 블록 | 500 | `dflt:361` |
| `eval_freq` | -1 | `dflt:306-308` |
| `test.visrank` | True, topk 10, count 10, `display_worst_rand` | `dflt:372-377` |
| `test.vis_feature_maps` | False | `dflt:379` |
| 체크포인트 저장 | `save_model_flag=False`: **저장 없음** | `dflt:69`, `engine.py:save_model:86-87` |
| save-best | 옵션 없음. `is_best`는 기본 False이고 호출부가 넘기지 않음 | `engine.py:86`, `:230`, `:250` |
| `model.resume`, `load_weights` | 빈 문자열 | `dflt:60`, `:68` |
| `transreid.cam_num` | yaml 6이지만 코드가 읽지 않음. 실제 값은 dataset 속성 6 | `IN-train:21`, `dflt:158`, `builder.py:234`, `market1501.py:27` |
| `sie_camera`, `sie_coe` | True, 3 | `dflt:164-165` |
| `solider.semantic_weight` | 0.2 | `dflt:170` |
| 파일 저장 | `save_disk=True`, 그림 PNG와 이미지 JPG | `dflt:49`, `K/torchreid/utils/logging/logger.py:104-108`, `:129-133` |
| seed | 1 | `dflt:305` |
| drop_path | 0.1 | `dflt:138` |

test yaml의 차이 (`IN-test`):

| 키 | 값 | 줄 |
|---|---|---|
| `model.load_weights` | `pretrained_models/kpr_market_IN_89.58_95.87_41898449.pth.tar` | `:12` |
| `model.pretrained` | False | `:13` |
| `test.evaluate` | True | `:26` |
| `data.bipo.p` | 지정 없음 | 학습 전용 |

`load_config=True`(`dflt:61-63`)면 체크포인트 안의 `model.kpr`, `model.transreid` config가 병합됩니다 (`builder.py:162-200`). 체크포인트 파일은 열지 않았습니다.

Stage 0 래퍼가 덮어쓴 값은 다음과 같습니다 (`stage0/backends.py:333-337`): `data.height`, `data.width`, `train.batch_size`, `test.visrank=False`, `test.batch_size`, `model.backbone_pretrained_path`, `train.seed`.

---

## 5. 공식 입력 크기와 공개 가중치

### 5.1 입력 크기

| 모델 | 파일 | 값 | 줄 |
|---|---|---|---:|
| KPR | `configs/kpr/imagenet/kpr_market_train.yaml` | 384×128 | 4-5 |
| KPR | `configs/kpr/imagenet/kpr_market_test.yaml` | 384×128 | 6-7 |
| KPR+SOLIDER | `configs/kpr/solider/kpr_market_train.yaml` | 384×128 | 6-7 |
| KPR+SOLIDER | `configs/kpr/solider/kpr_market_test.yaml` | 384×128 | 6-7 |

**KPR+SOLIDER의 입력 설정은 KPR과 같습니다.** 둘 다 Market-1501에서 384×128입니다.

### 5.2 README의 Market-1501 공개 가중치

**README에는 model zoo 표가 없습니다.** 확인한 내용:

| 항목 | 내용 | 근거 |
|---|---|---|
| 가중치 링크 | 폴더 링크 하나: `https://drive.google.com/drive/folders/1t4wXc2c3qlFaqUCifAlc_OPrFwvb7peD?usp=sharing` | `K/README.md:297` |
| 모델별 개별 링크 | 없음 | README 전체 |
| Market-1501 mAP, Rank-1 수치 | README 본문에 없음. PWC 배지만 있음 | `K/README.md:45` |
| 어떤 config인지 | 표기는 없고, `configs/kpr/<pretraining>/kpr_<target_dataset>_test.yaml`을 쓰라는 안내 | `K/README.md:306` |
| 주석(마스크, 키포인트) 링크 | `https://drive.google.com/drive/folders/15_RdnS1nr3iAYcnCibXmbT1LxU2n8PHZ?usp=sharing` | `K/README.md:267` |
| 개별 파일 링크가 있는 예 | Occluded-PoseTrack 가중치만 | `K/demo.py:30-31` |

config 파일명에서 읽을 수 있는 대응:

| 모델 | test yaml | 가중치 파일명 | 파일명 속 수치 | 줄 |
|---|---|---|---|---:|
| KPR (ImageNet) | `configs/kpr/imagenet/kpr_market_test.yaml` | `kpr_market_IN_89.58_95.87_41898449.pth.tar` | 89.58, 95.87 | 12 |
| KPR+SOLIDER | `configs/kpr/solider/kpr_market_test.yaml` | `kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar` | 93.25, 96.59 | 13 |

파일명 수치는 mAP, Rank-1로 보입니다. 저장소 어디에도 명시되어 있지 않으므로 추정입니다. 같은 이름의 파일이 `weights/stage05/public/`에 있습니다 (384,344,696 바이트, 384,266,425 바이트). 파일 내용은 열지 않았습니다.

README도 학습 config가 불완전할 수 있다고 적고 있습니다 (`K/README.md:331`, `:378`).

---

## 6. RAM 역추적 (코드 근거)

N_q = 3,368, N_g = 15,913은 `market1501.py:21` docstring 값입니다. test 디렉터리는 열지 않았습니다. K = 5, 임베딩 수 K+1 = 6입니다.

### 6.1 가설: `_feature_extraction`이 입력 텐서까지 누적한다 → **확인**

`part_based_engine.py:185-193`에서 리스트를 만들고 `:216`, `:221`, `:222`, `:223`, `:226`에서 추가합니다.

`data`는 dataset이 반환한 sample dict를 default_collate한 것 전체입니다. `imgs.cuda()`는 새 GPU 텐서를 만들고 CPU 원본은 `data` 안에 남습니다 (`:196-201`). sample dict의 구성은 `dataset.py:505-513`, `:519`, `:531`입니다.

| 배열 | 형태 | dtype | 샘플당 바이트 | 근거 |
|---|---|---|---:|---|
| `f_` | [N, K+1, 512] | float32 | 4·(K+1)·512 = 12,288 | `part_based_engine.py:219-221` |
| `parts_visibility_` | [N, K+1] | bool | K+1 = 6 | `:213-216`, `kpr.py:237` |
| `p_masks_` | [N, K+1, H/4, W/4] | float32 | 1.5·H·W | `:220-222`, `tools.py:237-244` |
| `pxl_scores_` | [N, K+1, H/4, W/4] | float32 | 1.5·H·W | `:223`, `kpr.py:190` |
| `anns.image` | [N, 3, H, W] | float32 | 12·H·W | `dataset.py:531`, `A/pytorch/transforms.py:90` |
| `anns.prompt_masks` | [N, 8, H, W] | **float64** | **64·H·W** | `dataset.py:512` |
| `anns.target_masks` | [N, K+1, H/4, W/4] | float32 | 1.5·H·W | `dataset.py:519` |
| `anns.keypoints_xyc` | [N, 17, 4] | float64 | 544 | `dataset.py:513`, `mask_transform.py:59` |
| `anns.negative_keypoints_xyc` | [N, 5, 17, 4] | float64 | 2,720 | `dataset.py:503-509` |
| `anns.pid`, `anns.camid` | [N] | int64 | 16 | `market1501.py:108-112` |
| `anns.img_path`, `masks_path`, `kp_path` | 문자열 리스트 | | 무시 가능 | `market1501.py:108-112` |
| `anns.mask_size` | 텐서 2개 × 배치 수 | int64 | 무시 가능 | `dataset.py:400`, `torchtools.py:404-406` |

입력 크기별 샘플당 합계와 전체 크기:

| 입력 | image (GiB, q+g) | prompt_masks (GiB, q+g) | target/p_masks/pxl 각각 (GiB, q+g) | 샘플당 anns | 샘플당 그 외 |
|---|---:|---:|---:|---:|---:|
| 384×128 | 10.59 | 56.49 | 1.32 | 3,812,560 B | 159,750 B |
| 384×192 | 15.89 | 84.73 | 1.99 | 5,717,200 B | 233,478 B |
| 512×256 | 28.24 | 150.63 | 3.53 | 10,161,360 B | 405,510 B |

`data`는 `pin_memory=True` 로더가 돌려준 값이라 누적 원본은 핀 메모리입니다 (`datamanager.py:327`, `:355`). 핀 메모리가 해제 뒤 OS로 얼마나 돌아가는지는 이 저장소 코드로 확인할 수 없어 미확인으로 둡니다.

**읽는 곳 확인.** `anns` 사용처를 `engine`, `metrics`, `utils`에서 전부 찾았습니다.

| 필드 | 사용처 | 판정 |
|---|---|---|
| `image` | 없음 | **미사용** |
| `prompt_masks` | 없음. 순위 격자는 dataset에서 다시 읽음 (`visualize_query_gallery_rankings.py:96-97`, `:130-131`) | **미사용** |
| `target_masks` | 픽셀 정확도 | `part_based_engine.py:429-439` |
| `keypoints_xyc`, `negative_keypoints_xyc` | 순위 격자 | `visualize_query_gallery_rankings.py:101-108`, `:133-140` |
| 나머지 | Market에서 미사용 | §6.4 |

즉 샘플당 누적량의 약 94%를 차지하는 두 배열이 저장만 되고 읽히지 않습니다. 비율: 384×192 기준 (884,736 + 4,718,592) / (5,717,200 + 233,478) = 94.2%.

### 6.2 가설: `collate`가 concat으로 피크를 두 배로 만든다 → **확인**

`K/torchreid/utils/torchtools.py:collate:325-422`

- 호출: `part_based_engine.py:234`. `anns = collate(anns)`이므로 반환 전까지 입력 리스트가 살아 있습니다.
- dict는 키별로 재귀합니다 (`torchtools.py:396-398`). 텐서는 `torch.cat(batch, dim=0, out=out).numpy()`입니다 (`:371-379`). 메인 프로세스라 `out=None`이고 새 메모리를 잡습니다 (`:373`).
- 따라서 끝나기 직전에 anns의 원본 전체와 concat 전체가 동시에 있습니다.
- 다른 배열의 `torch.cat`도 순간적으로 두 배지만 차례로 처리되고 크기가 작습니다 (`part_based_engine.py:228-231`).

피크 식 (gallery `collate` 직전):

`RAM ≈ N_q·(anns+그 외) + N_g·그 외 + 2·N_g·anns + occluder + 프로세스 기본`

| 입력 | 전 배열 합 (q+g) | gallery collate 피크 (앞 세 항) | occluder 상한 추가 | 파드 한도 57.7 GiB 대비 |
|---|---:|---:|---:|---|
| 384×128 | 71.3 GiB | 127.8 GiB | 2.72 GiB | **초과** |
| 384×192 | 106.9 GiB | 191.6 GiB | 4.08 GiB | **초과** |
| 512×256 | 189.7 GiB | **340.3 GiB** | 7.25 GiB | **초과** |

이 표의 값은 계산이며 측정이 아닙니다. 다른 프로세스, 작업자, 디스크 캐시는 포함하지 않았습니다.

`q_anns`, `g_anns`는 `_evaluate`가 끝날 때까지 살아 있습니다 (`part_based_engine.py:264`, `:276`, `:486`). §6.1의 누적 배열은 거리 계산, 그림, 순위 격자 내내 그대로 유지됩니다.

### 6.3 가설: `compute_distance_matrix_using_bp_features`의 배치, 장치, 할당

`distance.py:compute_distance_matrix_using_bp_features:87-99` → `_compute_distance_matrix_using_bp_features_and_masks:132-179` (`visibility`가 bool이라 이 분기, `kpr.py:237`)

| 질문 | 답 | 근거 |
|---|---|---|
| 배치 | **gallery만** 500개씩 나눔. query는 나누지 않음 | `distance.py:133-134`, `:143`, `dflt:361`, `builder.py:81`, `part_based_engine.py:310` |
| GPU | `qf` 전체를 GPU에 올리고 gallery 블록별 내적, sqrt, masked mean | `:137-139`, `:144-158`, `:229-237` |
| CPU | 블록 결과를 CPU로 옮겨 리스트에 쌓고 마지막에 concat | `:164-169` |
| GPU 크기 | 블록당 `[6, 3368, 500]` float32 ≈ 38.5 MiB, 같은 크기의 임시 텐서 여러 개 | `:231-237` |

CPU 할당 (입력 크기와 무관, N_q, N_g, K에만 의존):

| 항목 | 형태 | dtype | 크기 | 근거 |
|---|---|---|---:|---|
| 파트별 거리 | [6, Q, G] | float32 | 1.20 GiB | `:169` |
| 위 concat 순간 추가 | | | +1.20 GiB | `:165`, `:169` |
| 샘플 거리 | [Q, G] | float32 | 0.20 GiB | `:168` |
| `replace_values` 임시 (파트별) | `input*(~mask)`, `mask*value`, 합 | float32 | 각 1.20 GiB | `:177`, `tensortools.py:5` |
| 마스크 | [6, Q, G] | bool | 각 0.30 GiB | `:177` |

`:177` 실행 중 파트별 거리 관련 메모리는 일시적으로 약 5.4 GiB입니다 (이전 값 1.20 + 임시 3×1.20 + 마스크 2×0.30).

Stage 0 기록과 대조할 때의 주의점:

- `STAGE0_REPORT.md:123`의 "블록 분할 불일치"는 공식 경로가 아니라 `stage0/worker.py:task_dist:130-133`이 **query를 나눠 함수를 여러 번 호출**해서 생긴 값입니다. `max_value`는 한 호출 안의 최댓값이며 (`distance.py:174`), 공식 경로는 query 전체로 한 번만 호출합니다 (`part_based_engine.py:304-313`). 이 함수의 gallery 블록은 최댓값 계산에 영향을 주지 않습니다.
- Stage 0은 `N_GALLERY = 19732`를 썼습니다 (`stage0/worker.py:21`). 공식 코드는 pid -1을 빼고 docstring 기준 15,913입니다 (`market1501.py:21`, `:91-92`, `:99-100`).
- Stage 0은 `use_logger=False`라 `qg_pairwise_dist_statistics`가 빠졌습니다 (`stage0/worker.py:115`, `distance.py:171`).

### 6.4 가설: `evaluate_rank`가 Market에서 q_anns, g_anns를 쓴다 → **부정**

- `use_cython=False`이므로 anns를 넘기지만 (`rank.py:570-574`, `part_based_engine.py:357-361`) `eval_market1501`은 **앞부분에서 쓰지 않습니다** (`rank.py:103-158`).
- 유일한 사용은 `if ... "occ_level" in q_anns` 블록입니다 (`rank.py:168-203`). Market sample에는 이 키가 없습니다 (`market1501.py:108-112`, `dataset.py:505-531`). 따라서 **실행되지 않습니다.**

`eval_market1501`의 임시 배열: `np.argsort` [Q,G] int64 0.40 GiB (`:112`), `matches` [Q,G] int32 0.20 GiB (`:113`), 쿼리별 Python 리스트 (`:148`).

### 6.5 가설: `plot_*`, `writer.visualize_*`가 임시 배열을 만든다 → **확인, 대부분 게이트 없음**

| 함수 | 임시 배열 | 게이트 | 근거 |
|---|---|---|---|
| `plot_body_parts_pairs_distance_distribution` | 파트(6)마다 아래를 반복 | **없음** | `part_based_engine.py:389-391`, `K/torchreid/utils/distribution.py:10-25` |
| `plot_pairs_distance_distribution` | 아래를 1회 | **없음** (SSMD 반환값에 쓰임) | `part_based_engine.py:395-403`, `distribution.py:28-32` |
| `compute_distance_distribution` | bool [Q,G] 2개 (각 0.05 GiB), `neg_p` ≈ [Q·G] float32 (≈0.20 GiB), `pos_p` 소량 | 위와 같음 | `distribution.py:36-37` |
| `plot_distributions` | `np.ones_like(neg_p)`와 나눗셈 결과 (각 ≈0.20 GiB), `ax.hist` 100 bin | 위와 같음 | `distribution.py:58-60` |
| `qg_body_part_distances_boxplot` | `np.random.choice(Q·G, 2000, replace=False)`, 샘플 (약 2,012×6) | `use_logger=True` 기본값 | `writer.py:188-200`, `distance.py:171` |
| `qg_body_part_pairs_availability_barplot` | bool [6,Q,G] 0.30 GiB | 위와 같음 | `writer.py:203` |
| `qg_body_part_availability_barplot`, `..._histogram` | [N,K+1] 수준 소량 | 위와 같음 | `writer.py:233-291` |
| `visualize_rank` | 재정렬 `np.argsort` 0.40 GiB + `matches` 0.20 GiB, 격자 4082×2170×3 ≈ 25.3 MiB × 10장, 쿼리당 getitem 13회 | `test.visrank=True`이고 `visrank` 인자도 True (최종 test만) | `part_based_engine.py:450`, `writer.py:300`, `visualize_query_gallery_rankings.py:53-54`, `:96`, `:128-130`, `:174-181`, `engine.py:221`, `:240` |
| `visualize_embeddings` | 현재 경로 미실행 | `test.vis_embedding_projection=False` | `writer.py:294`, `dflt:381` |

- `np.random.choice(replace=False)`는 numpy 내부에서 전체 순열을 만들 수 있습니다. 그러면 Q·G int64로 0.40 GiB입니다. 이것은 numpy 일반 동작에 대한 지식이고 코드로 확인하지 못했습니다.
- `Logger.add_figure`의 출력은 `save_disk=True`만 작동합니다 (`logger.py:86-108`, `dflt:45-49`). 그래도 위 계산은 logger와 상관없이 실행됩니다.
- 이 임시 배열은 대부분 입력 크기와 무관합니다. 입력 크기와 연동되는 RAM은 §6.1입니다.

**작업자 쪽 RAM (같은 절의 보충).** `read_image`, `read_masks`, `read_keypoints`의 무제한 `lru_cache`는 작업자별로 찹니다 (`tools.py:104`, `:130`, `:159`). 작업자가 매 순회 다시 fork되므로 (§1.1) epoch를 넘겨 재사용되지 않습니다. 한 epoch 안에서는 이미지가 거의 한 번씩만 쓰여 (`sampler.py:48-60`) 적중이 드물고 메모리만 찹니다.

| 입력 | 샘플당 캐시 (이미지 3·H·W + 마스크 22,032 B) | train, 작업자 1개 (12,936/4) | gallery, 작업자 1개 (15,913/4) |
|---|---:|---:|---:|
| 파일 128×64 | 46,608 B | 0.14 GiB | 0.17 GiB |
| 파일 384×192 | 243,216 B | 0.73 GiB | 0.90 GiB |
| 파일 512×256 | 415,248 B | 1.25 GiB | 1.54 GiB |

여기에 JSON 파싱 결과(작음)가 더해집니다. Stage 0의 RSS는 `RUSAGE_SELF`라 작업자 메모리를 포함하지 않습니다 (`stage0/worker.py:30`).

**캐시 오염 주의 (SR 입력에서만).** 파일 크기와 입력 크기가 같으면 `functional.resize`가 같은 객체를 돌려줍니다 (`A/augmentations/geometric/functional.py:389-390`). BIPO `paste_over`는 제자리 수정이므로 (`batch_wise_inter_person_occlusion.py:66-68`, `:445-446`) 캐시된 원본 배열이 변경됩니다. baseline(128×64→384×128)은 리사이즈가 새 배열을 만들어 해당하지 않습니다. 현재는 작업자가 매 epoch 새로 fork되고 같은 epoch 내 재사용이 4장 미만 ID로 드물어 영향은 작습니다. 다만 `persistent_workers=True`를 쓰면 누적될 수 있습니다.

### 6.6 `display_feature_maps`와 스텝당 호출

**`display_feature_maps`: 매 스텝 호출되지만 실행은 안 됩니다.**

- 호출: `part_based_engine.py:110-112`
- 게이트: `epoch == max_epoch-1 and batch < 10 and cfg.test.vis_feature_maps` (`K/torchreid/utils/visualization/feature_map_visualization.py:148`)
- 값: `test.vis_feature_maps=False` (`dflt:379`). `IN-train`, `IN-test`에 키가 없습니다.
- 비용: 함수 호출과 내부 함수 정의뿐입니다. GPU 동기화나 복사가 없습니다.

`forward_backward`와 이를 둘러싼 `Engine.train`의 스텝당 호출 전체:

| # | 호출 | 위치 | 횟수 | 제어 config |
|---|---|---|---:|---|
| 1 | `data_loading_timer.stop` → `torch.cuda.synchronize` | `engine.py:268`, `K/torchreid/utils/avgmeter.py:270-274` | 1 | 없음 (CUDA 가능 시 항상) |
| 2 | `batch_timer.start` → `cuda.Event` 생성, record | `engine.py:269`, `avgmeter.py:259-262` | 1 | 없음 |
| 3 | `.cuda()`: imgs, cam_id, target_masks, prompt_masks, keypoints_xyc, pids | `part_based_engine.py:561-570` | 6 | `use_gpu` |
| 4 | `feature_extraction_timer.start`, `stop` → synchronize | `:101`, `:113` | 1쌍 | 없음 |
| 5 | `display_feature_maps` | `:110-112` | 1 (무동작) | `test.vis_feature_maps=False` |
| 6 | `loss_timer.start`, `stop` → synchronize | `:116`, `:126` | 1쌍 | 없음 |
| 7 | `pred_accuracy`(torchmetrics) 전역, 전경, concat | `GiLt_loss.py:120` | 3 | `weights.*.id` |
| 8 | `update_invalid_pairwise_distances_count` → `.item()` | `part_averaged_triplet_loss.py:138`, `writer.py:102`, `avgmeter.py:115-116` | 1 | 없음 |
| 9 | `pred_accuracy` 픽셀, `accuracy.item()` | `body_part_attention_loss.py:60-61` | 1 | `weights.pixls.ce>0` |
| 10 | `optimizer_timer.start`, `stop` → synchronize | `part_based_engine.py:129`, `:138` | 1쌍 | 없음 |
| 11 | `batch_timer.stop` → synchronize | `engine.py:272` | 1 | 없음 |
| 12 | `writer.losses.update` → `.item()` | `engine.py:274`, `avgmeter.py:347-348` | 10 | 없음 |
| 13 | `writer.loss.update(loss)` → `.item()` | `engine.py:275`, `avgmeter.py:115-116` | 1 | 없음 |
| 14 | `data_loading_timer.start` | `engine.py:276` | 1 | 없음 |
| 15 | `engine_state.batch_completed` (리스너 no-op) | `engine.py:278`, `engine_state.py:57-61` | 1 | 없음 |

합계: `synchronize` 5회 (1, 4, 6, 10, 11), `.item()` 13회 (8, 9, 12, 13). #12의 10회는 전역 {c,a}, 전경 {c,a}, concat {c,a}, 파트 {t,tt,vt}, 픽셀 {c}입니다 (`GiLt_loss.py:71-72`, `:84-86`, `body_part_attention_loss.py:45-46`).

torchmetrics와 `GradScaler` 내부의 호출은 세지 않았습니다.

스텝 단위 logger 호출과 시각화는 없습니다.

- `Writer`는 `batch_completed`를 구현하지 않습니다 (`writer.py:320-373`).
- `train.batch_log_freq`는 코드 어디에서도 읽지 않습니다 (`dflt:310`).
- `train.batch_debug_freq=0`은 `visualize_triplets`에서만 읽고, 그 함수는 호출하는 곳이 없습니다 (`dflt:309`, `writer.py:134-135`).
- epoch 단위 `add_scalar`는 `tensorboard`, `wandb`, `clearml`이 꺼져 있어 무동작입니다 (`dflt:45-47`, `logger.py:80-84`).

Stage 0의 `train_step`은 `forward_backward`만 호출하므로 위 #1, #2, #11~#15가 빠져 있습니다 (`stage0/backends.py:366-369`). 대신 `loss.item()`과 측정용 `sync`가 들어갑니다 (`stage0/worker.py:50-51`). #4, #6, #10의 synchronize는 포함되어 있습니다.

### 6.7 Stage 0 RSS와의 대조 (추정)

Stage 0 "extract"는 공식 `_feature_extraction`을 부르지 않습니다. `extract_batch`가 특징을 만들고 버립니다 (`stage0/backends.py:375-388`, `stage0/worker.py:67-77`). 그래서 `STAGE0_REPORT.md:100-102`의 RSS에는 §6.1의 누적이 없습니다.

두 ×3 값으로 선형 근사하면 (RSS = 기본 + a·배치):

- a = (31.03−18.72)/128 = 0.0962 GiB/샘플
- 샘플당 anns 5,717,200 B = 0.00532 GiB이므로 a는 약 18 배치 분량입니다
- 기본 = 18.72 − 0.0962×128 = 6.4 GiB. 이 중 occluder 상한이 4.08 GiB입니다

18 배치 분량은 prefetch 대기 큐(작업자 4 × 기본 prefetch 2)와 핀 메모리 복사본을 합친 값으로 해석됩니다. 같은 모델로 ×2 배치 128을 예측하면 약 9.6 GiB이고 측정은 8.81 GiB로 약 9% 차이입니다. 두 점 근사이며 큐 해석은 추정입니다.

### 6.8 데이터 로더 CPU 시간이 쓰이는 곳 (코드 읽기, 미측정)

단일 샘플, 384×192 기준의 바이트 처리량입니다. 시간은 재지 않았습니다.

| 순위 | 구간 | 처리 내용 | 근거 |
|---:|---|---|---|
| 1 | 프롬프트 생성 | 17×H×W float64 할당 (9.6 MiB), fancy index 복사 (9.6 MiB), max, clamp, stack, cat, sum, cat (최종 4.5 MiB). 매 epoch 재계산, 캐시 없음. 합계 수십 MiB | `keypoints_to_masks.py:321`, `mask_transform.py:55-56`, `dataset.py:484-512` |
| 2 | 파일 읽기와 디코드 | 샘플당 이미지 + npy + JSON 3개 파일. epoch 간 캐시 무효 (§6.5). Stage 0 데이터는 PNG이고 네트워크 볼륨 | `tools.py:119-120`, `:146`, `:169-170`, `STAGE0_REPORT.md:142` |
| 3 | BIPO (적용된 샘플만, p=0.05) | 키포인트마다 `scipy.ndimage.shift`를 H×W 전체에 실행 (17회 이상), `paste_over` | `batch_wise_inter_person_occlusion.py:93-95`, `:469-474`, `:411-446` |
| 4 | target 마스크 | 5채널을 H×W로 업샘플, 패딩, softmax, 6채널 bilinear | §1.5, `mask_transform.py:117`, `:87` |
| 5 | 이미지 | Normalize, 패딩, cutout 복사 (0.84 MiB 단위) | `transforms.py:103`, `:119`, `coarse_dropout.py:89-96` |
| 6 | collate | 배치 64에 prompt 288 MiB + image 54 MiB를 공유 메모리에 stack, 메인에서 핀 메모리로 복사 | `datamanager.py:228-241` |

근거로 삼을 수 있는 측정은 Stage 0의 "CPU/데이터 로딩 병목 의심" 표시뿐입니다 (`STAGE0_REPORT.md:13`, GPU 사용률 52%). 이 표의 순위 자체는 측정된 적이 없습니다. 작업자는 4개입니다 (`dflt:180`). `worker_init_fn`이나 cv2 스레드 설정은 데이터 코드에 없습니다.
