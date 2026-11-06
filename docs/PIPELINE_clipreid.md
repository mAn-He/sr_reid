# CLIP-ReID 파이프라인 (정적 분석)

학습, GPU 실행, 코드 수정 없이 코드만 읽고 작성했습니다. 이미지와 `data/locked`, `data/market/{query,bounding_box_test,gt_query,gt_bbox}`는 열지 않았습니다. 크기는 H×W입니다.

- 분석 대상: 커밋 `eb1898b72c882875f478bebfc6d41644eece0a5d`(2023-11-21), `git status` 변경 없음.
- Stage 0이 쓴 변형: `configs/person/vit_clipreid.yml`, stride 16, SIE 꺼짐(`stage0/backends.py:ClipBackend.__init__:174-179`).

| 경로 표기 | 실제 위치 |
|---|---|
| `CR/` | `third_party/CLIP-ReID/` |
| (접두사 없음) | `/workspace/sr-reid/` 기준 상대 경로 |
| `ENV_CLIP/` | `/workspace/miniforge3/envs/srreid-clipreid/lib/python3.10/site-packages/` |
| `yml:n` | `CR/configs/person/vit_clipreid.yml`의 n번째 줄 |
| `lock:n` | `env/srreid-clipreid.lock.txt`의 n번째 줄 |

**주의:** 학습은 두 단계입니다. 이 문서의 "stage 1, stage 2"는 CLIP-ReID 학습 단계이고, 연구 일정의 "Stage 0"과 다릅니다.

## 1. 학습 경로

진입점은 `CR/train_clipreid.py`입니다. 로더 → 모델 → 손실 → stage 1 → stage 2 순서로 실행됩니다(`:66-100`).

### 1.1 공통 데이터 경로

| 단계 | 내용 | 근거 |
|---|---|---|
| 파일 목록 | `*.jpg`만 수집, pid -1 제외, train은 재번호 | `CR/datasets/market1501.py:_process_dir:64-84` |
| `__getitem__` | PIL RGB로 읽고 transform 적용. (img, pid, camid, trackid, 파일명) 반환 | `CR/datasets/bases.py:ImageDataset.__getitem__:78-85`, `read_image:10-23` |
| 배치 | img, pid, camid, viewid로 묶음 | `CR/datasets/make_dataloader_clipreid.py:train_collate_fn:26-34` |

### 1.2 Stage 1: 프롬프트 학습

| 단계 | 내용 | 근거 |
|---|---|---|
| 데이터 | `train_set_normal`. **증강 없음**(`val_transforms`), shuffle, 배치 64, PK 없음 | `CR/datasets/make_dataloader_clipreid.py:make_dataloader:65,104-107` |
| 전처리 | `T.Resize(SIZE_TEST)` 기본 bilinear → ToTensor → Normalize(0.5, 0.5) | 같은 파일 `:54-58` |
| 이미지 특징 캐시 | 학습 전 1회, no_grad, autocast. 투영 CLS 특징(512차원)을 전량 GPU에 올림 | `CR/processor/processor_clipreid_stage1.py:do_train_stage1:43-53`, `CR/model/make_model_clipreid.py:build_transformer.forward:113-118` |
| 텍스트 | 매 배치 `"A photo of a X X X X person."`의 학습 토큰 4개로 인코딩 | `CR/processor/processor_clipreid_stage1.py:do_train_stage1:75-76`, `CR/model/make_model_clipreid.py:PromptLearner:197,209-212` |
| 손실 | `SupConLoss(이미지, 텍스트) + SupConLoss(텍스트, 이미지)`, 온도 1.0 | `CR/processor/processor_clipreid_stage1.py:do_train_stage1:77-80`, `CR/loss/supcontrast.py:14-30` |
| 배치 구성 | 캐시 N개의 `randperm`을 64개씩. 에폭당 12936//64+1 = 203회 | `CR/processor/processor_clipreid_stage1.py:do_train_stage1:55-57,65-71` |
| 옵티마이저 | Adam. `prompt_learner` 파라미터만, lr 3.5e-4, wd 1e-4 | `CR/solver/make_optimizer_prompt.py:make_optimizer_1stage:4-19` |
| 스케줄 | Cosine. 5에폭 warmup(1e-5부터), lr_min 1e-6, `scheduler.step(epoch)` | `CR/train_clipreid.py:73-74`, `CR/solver/scheduler_factory.py:7-25`, `CR/solver/cosine_lr.py:_get_lr:67-94`, `CR/processor/processor_clipreid_stage1.py:do_train_stage1:62` |
| 길이 | 120에폭, 평가 없음, 120에폭에 저장 | `CR/configs/person/vit_clipreid.yml:38-39`, `CR/processor/processor_clipreid_stage1.py:do_train_stage1:95-102` |

에폭별 lr(코드에서 계산, 실행 안 함): 1에폭 7.8e-5, 4에폭 2.82e-4, 5에폭 3.4851e-4, 60에폭 1.755e-4, 119에폭 1.06e-6, 120에폭 1e-6.

### 1.3 Stage 2: 이미지 인코더 학습

**데이터와 증강.** 샘플러는 `RandomIdentitySampler`(P=64//4=16, K=4)입니다(`CR/datasets/make_dataloader_clipreid.py:84-88`, `CR/datasets/sampler.py:17-63`). 증강은 `make_dataloader:43-52`에 아래 순서로 정의되어 있습니다.

| 순서 | 종류 | 확률 | 값 | 줄 | config |
|---:|---|---:|---|---:|---|
| 1 | Resize | 1 | SIZE_TRAIN, `interpolation=3`(BICUBIC) | 44 | `yml:15` |
| 2 | 좌우 뒤집기 | 0.5 | - | 45 | `yml:17` |
| 3 | 패딩 | 1 | 사방 10px, 값 0, constant | 46 | `yml:19` |
| 4 | 랜덤 크롭 | 1 | SIZE_TRAIN으로 | 47 | `yml:15` |
| 5 | ToTensor | 1 | [0, 1] | 48 | - |
| 6 | 정규화 | 1 | mean 0.5, std 0.5 | 49 | `yml:20-21` |
| 7 | Random Erasing | 0.5 | timm. `mode='pixel'`(픽셀별 정규분포 잡음), `max_count=1`, 면적 0.02~1/3, 비율 0.3~1/0.3 로그 균등, 최대 10회 | 50 | `yml:18` |

Random Erasing 기본값은 `ENV_CLIP/timm/data/random_erasing.py:RandomErasing.__init__:46-58`, `_erase:78-100`입니다(timm 0.9.12, `env/srreid-clipreid.lock.txt:63`). README가 timm 버전을 고정하지 않으므로(`CR/README.md:15`) 공식 실행의 기본값과 같은지는 확인 못함입니다.

**모델.**

| 항목 | 내용 | 근거 |
|---|---|---|
| 이미지 인코더 | CLIP ViT-B-16 비전 부분. 위치 임베딩을 입력 크기에 맞춰 보간 | `CR/model/make_model_clipreid.py:build_transformer.__init__:82-88`, 3절 |
| 출력 | 11층 토큰(`x11`), 12층 토큰(`x12`), 투영(`xproj`) | `CR/model/clip/model.py:VisionTransformer.forward:218-240` |
| 특징 | 각 CLS: 768, 768, 512차원 | `CR/model/make_model_clipreid.py:build_transformer.forward:136-138` |
| BNNeck | 768, 512 각각. bias 고정 | `CR/model/make_model_clipreid.py:build_transformer.__init__:75-80`, `CR/model/make_model_clipreid.py:build_transformer.forward:140-141` |
| 분류기 | bias 없는 Linear 두 개 | `CR/model/make_model_clipreid.py:build_transformer.__init__:70-73`, `CR/model/make_model_clipreid.py:build_transformer.forward:144-146` |
| SIE | 꺼짐. `target_cam=None`이라 `cv_embed` 없음 | `CR/processor/processor_clipreid_stage2.py:do_train_stage2:87-94`, `CR/model/make_model_clipreid.py:build_transformer.forward:127-134` |
| 텍스트 특징 | 시작 때 751 ID 전부 1회 추출해 고정 | `CR/processor/processor_clipreid_stage2.py:do_train_stage2:56-71` |

**손실.** `CR/loss/make_loss.py:make_loss:39-56`입니다.

| 항 | 입력 | 가중치 | 근거 |
|---|---|---:|---|
| ID | 두 분류기 logit의 label smoothing CE 합(ε=0.1) | 0.25 | `CR/configs/person/vit_clipreid.yml:8`, `CR/loss/softmax_loss.py:17-35` |
| Triplet | `x11`, `x12`, `xproj` CLS의 합. hard mining, margin 0.3 | 1.0 | `yml:9`, `CR/config/defaults.py:109`, `CR/loss/triplet_loss.py:106-134` |
| I2T | `이미지 특징 @ 텍스트 특징`의 label smoothing CE | 1.0 | `yml:10`, `CR/processor/processor_clipreid_stage2.py:do_train_stage2:97-98` |

Center loss 객체는 생성되지만 쓰이지 않습니다(`CR/loss/make_loss.py:16`, `CR/processor/processor_clipreid_stage2.py:do_train_stage2:105`).

**옵티마이저와 스케줄러.**

| 항목 | 내용 | 근거 |
|---|---|---|
| 대상 | `text_encoder`, `prompt_learner`를 제외한 전부. 이 시점에 둘 다 동결 | `CR/solver/make_optimizer_prompt.py:make_optimizer_2stage:26-32` |
| lr, wd | 5e-6, 1e-4. 키에 `bias`가 있으면 lr ×2 | 같은 함수 `:35-39`, `yml:46,50-51,57` |
| 스케줄 | WarmupMultiStepLR. warmup 10, factor 0.1, [30, 50], gamma 0.1 | `CR/train_clipreid.py:86-87`, `CR/solver/lr_scheduler.py:43-55`, `yml:47-49,59-60` |
| 호출 | 에폭 시작 때 `scheduler.step()` | `CR/processor/processor_clipreid_stage2.py:do_train_stage2:79` |
| 길이 | 60에폭 | `yml:53` |

일반 파라미터의 에폭별 lr(코드에서 계산, 실행 안 함): 1에폭 9.5e-7, 9에폭 4.55e-6, 10~29에폭 5e-6, 30~49에폭 5e-7, 50~60에폭 5e-8.

## 2. 평가 경로

| 항목 | 값 | 근거 |
|---|---|---|
| 데이터 | query + gallery를 순서대로. `val_transforms`, 배치 64 | `CR/datasets/make_dataloader_clipreid.py:98-103`, `yml:64` |
| AMP | 평가 forward에 autocast 없음 | `CR/processor/processor_clipreid_stage2.py:do_train_stage2:163-176`, `CR/processor/processor_clipreid_stage2.py:do_inference:210-222` |
| 특징 | `NECK_FEAT: 'before'`. `x12` CLS와 `xproj` CLS 연결, **BN 앞**, 1280차원 | `yml:67`, `CR/model/make_model_clipreid.py:build_transformer.forward:149-153` |
| 정규화 | `FEAT_NORM: 'yes'`, L2. `if self.feat_norm:`이라 빈 문자열이 아니면 `'no'`도 정규화됨 | `yml:68`, `CR/utils/metrics.py:R1_mAP_eval.compute:111-113` |
| Flip test | 없음 | `CR/processor/processor_clipreid_stage2.py:do_train_stage2:163-176` |
| 거리 | 제곱 유클리드(정규화 후 2-2cos와 같음). CPU에서 계산 | `CR/utils/metrics.py:euclidean_distance:7-13`, `CR/utils/metrics.py:R1_mAP_eval.update:105` |
| 지표 | CMC(max_rank 50), mAP. 같은 ID·카메라 제외, 양성 없는 query는 건너뜀 | `CR/utils/metrics.py:eval_func:28-87` |
| **Re-ranking** | **꺼짐. `TEST.RE_RANKING` 값과 무관.** 평가기 생성 시 `reranking` 인자를 넘기지 않아 기본값 False | `yml:65`, `CR/processor/processor_clipreid_stage2.py:do_train_stage2:46`, `CR/processor/processor_clipreid_stage2.py:do_inference:197`, `CR/utils/metrics.py:91,123-130` |
| 주기 평가 | `SOLVER.STAGE2.EVAL_PERIOD: 60`, MAX 60 → **에폭 [60]** 한 번. stage 1은 평가 없음 | `yml:53,56`, `CR/processor/processor_clipreid_stage2.py:do_train_stage2:139` |

### 평가 중 메모리에 쌓이는 배열

Market 공식 수치 Q=3,368, G=15,913 기준입니다(`CR/datasets/market1501.py:24`, 파일은 세지 않음). 평가 특징의 float32는 평가 때 autocast가 없고 모델이 float32인 점에서 나온 추론이며, 실행으로 확인하지 않았습니다.

| 배열 | shape | dtype | 크기 | 위치 | 근거 |
|---|---|---|---:|---|---|
| `self.feats` | 배치별 [B, 1280] 목록 | float32 | 합계 94.1 MiB | CPU | `CR/utils/metrics.py:update:105` |
| `feats` 결합, 정규화 | [19281, 1280] | float32 | 94.1 MiB, 임시로 추가 사본 | CPU | `CR/utils/metrics.py:R1_mAP_eval.compute:110,113` |
| `self.pids`, `self.camids` | 길이 19,281 목록 | NumPy 정수 스칼라 | - | CPU | `CR/utils/metrics.py:R1_mAP_eval.update:106-107` |
| `q_pids`, `q_camids`, `g_pids`, `g_camids` | [3368], [15913] | int64 추정 | - | CPU | `CR/utils/metrics.py:R1_mAP_eval.compute:116-122` |
| `distmat` | [3368, 15913] | float32 | 204.4 MiB | CPU | `CR/utils/metrics.py:euclidean_distance:10-13`, `CR/utils/metrics.py:R1_mAP_eval.compute:130` |
| `indices` | [3368, 15913] | int64 | 408.9 MiB | CPU | `CR/utils/metrics.py:eval_func:39` |
| `matches` | [3368, 15913] | int32 | 204.4 MiB | CPU | `CR/utils/metrics.py:eval_func:42` |
| `all_cmc`, `all_AP` | 유효 query 수만큼, `all_cmc`는 각 길이 50 | 정수 배열, float | - | CPU | `CR/utils/metrics.py:eval_func:67,79` |
| `img_path_list` | 이미지 수 | 문자열 | - | CPU, `do_inference`만 | `CR/processor/processor_clipreid_stage2.py:do_inference:208,223` |

특징 차원은 입력 크기와 무관합니다. 입력 크기는 추출 시 GPU 메모리에만 영향을 줍니다. `evaluator.reset()`이 매 에폭 시작에 호출됩니다(`CR/processor/processor_clipreid_stage2.py:do_train_stage2:77`).

Stage 0의 거리 측정은 무작위 특징 [3368, 19732]입니다(`stage0/worker.py:21,99`). 공식 데이터 클래스는 gallery 15,913장을 쓰므로 실제보다 큽니다. 추출 측정 배치는 128, 256으로(`stage0/run_probe.py:250`), yml의 64와 다릅니다.

### 학습 중 메모리 누적 (stage 1 캐시, stage 2 텍스트)

autocast 안에서 만들어지므로 dtype은 float16으로 추정합니다(정적 추론, 실행 확인 안 함).

| 배열 | shape | dtype | 크기 | 위치 | 근거 |
|---|---|---|---:|---|---|
| `image_features_list` | [12936, 512] | float16 추정 | 12.6 MiB | GPU | `CR/processor/processor_clipreid_stage1.py:do_train_stage1:47-53` |
| `labels_list` | [12936] | int64 | - | GPU | `CR/processor/processor_clipreid_stage1.py:do_train_stage1:52` |
| `text_features` | [751, 512] | float16 추정 | 0.7 MiB | GPU | `CR/processor/processor_clipreid_stage2.py:do_train_stage2:68-71` |

## 3. 입력 크기 변경 시 영향받는 지점

### 3.1 위치 임베딩과 토큰 수

| 단계 | 내용 | 근거 |
|---|---|---|
| 해상도 계산 | `h_resolution = int((SIZE_TRAIN[0]-16)//STRIDE_SIZE[0] + 1)`, `w_resolution = int((SIZE_TRAIN[1]-16)//STRIDE_SIZE[1] + 1)` | `CR/model/make_model_clipreid.py:build_transformer.__init__:82-83` |
| 패치 합성곱 | kernel은 가중치에서 읽어 16, stride는 `STRIDE_SIZE[0]`만 사용 | `CR/model/clip/model.py:build_model:408`, `VisionTransformer.__init__:206`, `CR/model/make_model_clipreid.py:84` |
| 새 임베딩 크기 | `h_resolution*w_resolution + 1` | `CR/model/clip/model.py:VisionTransformer.__init__:210` |
| 보간 | `resize_pos_embed`. CLS 분리 → 14×14 격자 → `F.interpolate(size=(h, w), mode='bilinear')` → 연결 | `CR/model/clip/model.py:resize_pos_embed:449-465`, 호출 `build_model:434` |
| 시점 | 모델 생성 때 1회 | `CR/model/make_model_clipreid.py:load_clip_to_cpu:175-189` |
| 적용 | `x + self.positional_embedding` | `CR/model/clip/model.py:VisionTransformer.forward:225` |

| 입력 | stride 16 (`vit_clipreid.yml`) 격자 | 토큰(CLS 포함) | stride 12 (OLP) 격자 | 토큰(CLS 포함) |
|---|---:|---:|---:|---:|
| 256×128 | 16×8 | 129 | 21×10 | 211 |
| 384×192 | 24×12 | 289 | 31×15 | 466 |
| 512×256 | 32×16 | 513 | 42×21 | 883 |

위 표의 값은 공식에서 계산했습니다. stride 16의 384×192는 Stage 0 로그의 `197 → 289`, `height:24 width:12`와 일치합니다(`STAGE0_REPORT.md:31-34`). 나머지 값은 실행으로 확인하지 않았습니다.

### 3.2 크기에 의존하는 지점

| 지점 | 근거 | 입력 크기를 바꿀 때의 동작 |
|---|---|---|
| `INPUT.SIZE_TRAIN` | `yml:15`, `CR/model/make_model_clipreid.py:82-83`, `CR/datasets/make_dataloader_clipreid.py:44,47` | 위치 임베딩, Resize, RandomCrop을 결정 |
| `INPUT.SIZE_TEST` | `yml:16`, `CR/datasets/make_dataloader_clipreid.py:55` | Resize에만 쓰임. 모델은 SIZE_TRAIN으로 만들어지므로 둘이 다르면 `CR/model/clip/model.py:VisionTransformer.forward:225`에서 shape 오류가 날 것으로 보임(정적 추론) |
| **학습 Resize** | `CR/datasets/make_dataloader_clipreid.py:44` | BICUBIC. **입력이 목표와 같으면 no-op**: `F.resize`가 먼저 원본을 그대로 반환(`ENV_CLIP/torchvision/transforms/functional.py:resize:465-468`). 정수 3의 변환은 `:37-41,441-446` |
| **평가·stage 1 Resize** | `CR/datasets/make_dataloader_clipreid.py:55,65` | 기본 BILINEAR(`ENV_CLIP/torchvision/transforms/transforms.py:Resize.__init__:317`). **같은 크기면 no-op**. 다를 때는 학습(BICUBIC)과 필터가 다름 |
| `T.Pad`, `T.RandomCrop` | `CR/datasets/make_dataloader_clipreid.py:46-47`, `yml:19` | 패딩 10px 고정, 크롭은 SIZE_TRAIN. 크기 H+20, W+20에서 잘라내므로 이동폭 절대값 ±10px 유지 |
| Random Erasing | `CR/datasets/make_dataloader_clipreid.py:50`, `ENV_CLIP/timm/data/random_erasing.py:81-92` | 면적 비율 기준이라 H×W를 따라감 |
| **코드의 고정값 16** | `CR/model/make_model_clipreid.py:82-83` | 패치 크기 16을 가정. H, W는 config에서 읽음 |
| `STRIDE_SIZE[1]` | `CR/model/make_model_clipreid.py:83,84` | 격자 계산에만 쓰임. 합성곱 stride는 `[0]`뿐이라 두 값이 다르면 맞지 않음 |
| **공개 가중치 로드** | `CR/model/make_model_clipreid.py:load_param:156-160` | `copy_`로 그대로 복사, 보간 없음. 256×128 가중치(129토큰)는 384×192(289토큰) 모델에 그대로 들어가지 않음(정적 추론, 실행 안 함) |
| 공식 데이터 클래스 | `CR/datasets/market1501.py:26,65` | 폴더명과 `*.jpg`가 고정. PNG 입력은 수집되지 않음. Stage 0은 이 클래스를 쓰지 않음 |
| Stage 0 크기 | `stage0/backends.py:size_of:23-24`, `ClipBackend.__init__:178-179`, `train_items:29` | `data/bicubic_x{k}` PNG와 override로 입력 지정 |
| Stage 0 측정 범위 | `stage0/run_probe.py:285`, `stage0/prepare_data.py:45` | k=2, 3만. **512×256은 Stage 0 측정값이 없음**. `data/bicubic_x4`는 이 문서 작성 중(2026-10-31 10:41) 별도 작업으로 생겼으며, 이 분석의 측정 근거에는 쓰지 않음 |

`CR/model/make_model_clipreid.py`, `CR/processor/processor_clipreid_stage{1,2}.py`, `CR/datasets/make_dataloader_clipreid.py`, `CR/datasets/bases.py`, `CR/utils/metrics.py`, `CR/solver/make_optimizer_prompt.py`, `CR/loss/make_loss.py`에서 256, 128, 384, 224, 192, 512를 검색한 결과 H×W를 고정한 곳은 없었습니다. 나온 값은 특징 차원 512(`CR/model/make_model_clipreid.py:61,199`)뿐입니다. `CR/model/clip/model.py:100`의 224는 RN50용 인자 기본값입니다.

**두 경로의 크기 의존 차이:** stage 1의 학습 스텝은 캐시와 텍스트만 쓰므로 입력 크기와 무관합니다. 크기는 캐시 추출에만 영향을 줍니다. Stage 2의 스텝은 매번 이미지 인코더를 통과합니다.

### 3.3 개념상 크기와 함께 달라지는 값 (목록만, 변경 제안 아님)

| 값 | 현재 | 근거 | 256×128 | 384×192 | 512×256 |
|---|---|---|---:|---:|---:|
| `INPUT.PADDING` 이동폭, H 대비 | 10px | `yml:19` | 3.9% | 2.6% | 2.0% |
| `INPUT.PADDING` 이동폭, W 대비 | 10px | `yml:19` | 7.8% | 5.2% | 3.9% |
| 패치 한 변의 H 대비 비율 | 16px | `CR/model/make_model_clipreid.py:82` | 6.3% | 4.2% | 3.1% |
| `MODEL.STRIDE_SIZE` 절대 px | 16 | `yml:7` | 16 | 16 | 16 |
| 위치 임베딩 원본 격자 대비 확대 | 14×14 | `CR/model/clip/model.py:459` | 16×8 | 24×12 | 32×16 |

뒤집기 확률과 Random Erasing 면적 비율은 크기에 따라 달라지지 않습니다. lr, 배치, 에폭도 config상 크기와 연결되어 있지 않습니다.

## 4. 실제 config 값

인용은 `CR/configs/person/vit_clipreid.yml`의 줄입니다.

| 항목 | Stage 1 | 인용 | Stage 2 | 인용 |
|---|---|---:|---|---:|
| 입력 H×W | [256, 128] | 15-16 | [256, 128] | 15-16 |
| 배치 | 64 | 30 | 64 | 44 |
| P, K | 없음(무작위 순열) | - | P=16, K=4 | 25, 44 |
| 에폭 | 120 | 38 | 60 | 53 |
| 옵티마이저 | Adam | 31 | Adam | 45 |
| lr | 3.5e-4 | 32 | 5e-6, bias ×2 | 46, 57 |
| 스케줄 | cosine, warmup 5, 시작 1e-5, 최소 1e-6 | 33-35, 41 | multistep [30, 50], γ 0.1, warmup 10, factor 0.1 | 47-49, 59-60 |
| wd, bias wd | 1e-4, 1e-4 | 36-37 | 1e-4, 1e-4 | 50-51 |
| 체크포인트 주기 | 120 | 39 | 60 | 54 |
| 평가 주기 | 없음 | - | 60 | 56 |

| 항목 | 값 | 인용 |
|---|---|---|
| 증강 | flip 0.5, RE 0.5, padding 10 | `yml:17-19` |
| 평균, 표준편차 | 0.5, 0.5 | `yml:20-21` |
| 샘플러 | `softmax_triplet` | `yml:24` |
| K | 4 | `yml:25` |
| num_workers | 8 | `yml:26` |
| 손실 가중치 | ID 0.25, triplet 1.0, I2T 1.0 | `yml:8-10` |
| **Stride** | [16, 16], OLP 꺼짐 | `yml:7` |
| **SIE** | 주석 처리되어 꺼짐. 기본값 `SIE_CAMERA=False`, `SIE_VIEW=False`, `SIE_COE=3.0` | `yml:11-12`, `CR/config/defaults.py:61-63` |
| 평가 배치 | 64 | `yml:64` |
| Re-ranking 키 | False. 실제로 읽히지 않음(2절) | `yml:65` |
| 평가 특징 | `before`, 정규화 `yes` | `yml:67-68` |
| 시드 | 1234 | `CR/config/defaults.py:108` |
| Margin | 0.3 | `CR/config/defaults.py:109` |
| **AMP** | 항상 켜짐. config 키 없음 | stage 1 `CR/processor/processor_clipreid_stage1.py:33,47,75,82-85`, stage 2 `CR/processor/processor_clipreid_stage2.py:47,68,95,100-103` |
| dtype | 모델 파라미터 float32, 학습 forward는 autocast(fp16 혼합), 평가는 autocast 없음 | `CR/model/clip/model.py:convert_weights:378-399`(이름과 달리 `.float()`), `CR/processor/processor_clipreid_stage2.py:do_train_stage2:95` |
| `DATASETS` | 값이 전부 주석이라 yml 그대로는 yacs가 거부. Stage 0은 키를 지우고 병합 | `yml:70-85`, `stage0/backends.py:176`, `STAGE0_REPORT.md:149` |

## 5. Stage 구분, Stage 0 측정, 환경, 변형, 공개 가중치

### 5.1 Stage 1과 Stage 2

| 항목 | Stage 1 | Stage 2 |
|---|---|---|
| 목적 | ID별 프롬프트 토큰(`cls_ctx`) 학습 | 이미지 인코더, BNNeck, 분류기 학습 |
| 학습 대상 | `prompt_learner`만 | `text_encoder`, `prompt_learner` 제외 전부 |
| 이미지 인코더 | 고정. 시작 때 특징 1회 추출 | 매 스텝 forward와 backward |
| 텍스트 인코더 | 매 스텝 forward | 시작 때 1회 추출해 고정 |
| 데이터 | 증강 없음, 무작위 순열 | 증강 있음, PK 샘플러 |
| 손실 | SupCon 양방향 | ID + triplet + I2T |
| 에폭당 스텝 | 203 | 약 185 (`stage0/backends.py:meta:244`) |
| 입력 크기의 스텝 비용 영향 | 없음(캐시 추출에만) | 직접 |

근거는 1.2절, 1.3절입니다. 코드는 stage 1이 끝난 모델을 메모리에서 그대로 stage 2로 넘깁니다(`CR/train_clipreid.py:76-100`).

### 5.2 Stage 0이 측정한 것

**보고서 표의 it/s는 stage 2 값입니다**(`STAGE0_REPORT.md:18`, `stage0/backends.py:ClipBackend.__init__:158` 기본 `stage=2`, `stage0/run_probe.py:211`). Stage 1도 따로 쟀습니다(`stage0/run_probe.py:214-215`, `stage0/worker.py:39`). 측정은 warmup 10스텝, 측정 40스텝입니다(`stage0/run_probe.py:161`).

| 입력 | 단계 | it/s | img/s | allocated GiB | 에폭당 스텝 | 에폭 | 인용 |
|---|---|---:|---:|---:|---:|---:|---|
| 256×128 | stage 2 | 8.63 | 552.2 | 5.84 | 185 | 60 | `results/stage0_probe.csv:4`, `STAGE0_REPORT.md:11` |
| 256×128 | stage 1 | 19.40 | 1241.7 | 2.27 | 203 | 120 | `results/stage0_probe.csv:5` |
| 384×192 | stage 2 | 3.95 | 252.7 | 13.41 | 185 | 60 | `results/stage0_probe.csv:6`, `STAGE0_REPORT.md:12` |
| 384×192 | stage 1 | 22.33 | 1429.2 | 2.26 | 203 | 120 | `results/stage0_probe.csv:7` |
| 512×256 | 둘 다 | 없음 | - | - | - | - | `stage0/run_probe.py:285` |

**Stage 1 측정 방식과 한계**

| 항목 | 내용 | 근거 |
|---|---|---|
| 캐시 외삽 | **`_stage1_cache`는 앞 3,200장만 추출해 12,936개로 반복**합니다. 공식은 전량을 추출합니다 | `stage0/backends.py:ClipBackend._stage1_cache:226-239`, `CR/processor/processor_clipreid_stage1.py:do_train_stage1:43-53` |
| ID 편중 | 경로가 정렬되어 있어(`stage0/backends.py:train_items:29`) 3,200장은 앞쪽 ID에 몰립니다. 속도에는 영향이 없을 것으로 보이나 측정 중 손실 값은 공식 학습의 값이 아닙니다 | `stage0/backends.py:ClipBackend._stage1_cache:228` |
| 잰 구간 | 캐시 뒤의 프롬프트 스텝만. 캐시 시간은 포함되지 않음 | `stage0/worker.py:task_train:43-52` |
| 전체 캐시 시간 | 전량 추출을 재지 않고 `12936 ÷ 추출 img/s`로 계산. 추출 측정은 배치 128, autocast 없음이고, 공식 stage 1은 배치 64, autocast 있음 | `stage0/report.py:90`, `stage0/backends.py:extract_batch:285-287`, `CR/processor/processor_clipreid_stage1.py:do_train_stage1:47`, `STAGE0_REPORT.md:55-56` |
| 크기 효과 | 스텝에 이미지 인코더가 없어 22.33과 19.40의 차이는 입력 크기 효과가 아니라 측정 변동으로 해석합니다(해석) | `stage0/backends.py:_step1:251-264` |
| 메모리 | `max_memory_allocated`에 3,200장 추출 피크가 포함될 수 있음 | `stage0/backends.py:_stage1_cache:231-236`, `stage0/worker.py:memory:29` |

**Stage 2 측정과 공식 경로의 일치 여부**

| 항목 | 내용 | 근거 |
|---|---|---|
| 증강 | 공식 `train_transforms`와 같은 순서·값으로 다시 작성 | `stage0/backends.py:186-189`, `CR/datasets/make_dataloader_clipreid.py:43-52` |
| 스텝 | autocast, 손실, scaler가 공식과 같음. 스케줄러는 돌지 않음 | `stage0/backends.py:_step2:266-279`, `CR/processor/processor_clipreid_stage2.py:do_train_stage2:95-103` |
| 데이터 | `data/bicubic_x{k}` PNG. 공식 데이터 클래스 대신 train만 읽는 목록 | `stage0/backends.py:train_items:27-36` |
| 시간 추정 | 스텝 수 185는 `len(sampler)//batch`. 실제 에폭 배치 수는 샘플러 실행마다 달라질 수 있음 | `stage0/backends.py:meta:244`, `CR/datasets/sampler.py:29-35,55-63` |
| 합계 | 학습 1회 추정 256×128 0.71 h, 384×192 1.10 h. 검증·저장 시간 미포함 | `STAGE0_REPORT.md:55-56,138` |

### 5.3 측정 환경과 공식 요구 사항

README 인용(`CR/README.md:13`): `conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch`

| 패키지 | README | 측정 환경 | 일치 |
|---|---|---|---|
| Python | 3.8 (`CR/README.md:11`) | 3.10.13 (`env/srreid-clipreid.yml:6`, `env/srreid-clipreid.lock.txt:30`) | **아니오** |
| PyTorch | 1.8.0 (`:13`) | 1.13.0+cu117 (`lock:65`, `results/stage0_probe.csv:4-7`) | **아니오** |
| torchvision | 0.9.0 (`:13`) | 0.14.0+cu117 (`lock:66`) | **아니오** |
| CUDA | 10.2 (`:13`) | 11.7 빌드 (`lock:65`) | **아니오** |
| timm | 버전 없음 (`:15`) | 0.9.12 (`lock:63`) | 판단 불가 |
| yacs, scikit-image, tqdm, ftfy, regex | 버전 없음 (`:14,16-19`) | 0.1.8, 0.22.0, 4.66.1, 6.1.3, 2023.12.25 (`lock:72,60,67,43,57`) | 판단 불가 |

**측정 환경은 공식 요구 사항과 다릅니다.** 이유는 torch 1.8 + CUDA 10.2가 RTX 4090을 지원하지 않기 때문이라고 기록되어 있습니다(`env/srreid-clipreid.yml:1`, `STAGE0_REPORT.md:141`). 이 환경에서 공개 수치가 재현되는지는 확인 못함입니다.

### 5.4 변형과 사용 가능한 config

`configs/person`에는 아래 4개뿐입니다. **`*_12x12` 같은 파일은 없습니다.** OLP는 명령행으로 켭니다(`CR/README.md:61`).  
표의 `:n`은 같은 행 파일의 줄입니다.

| 파일 | 진입점 | NAME | STRIDE_SIZE | SIE_CAMERA, SIE_COE | ID / I2T | 입력 | 에폭 |
|---|---|---|---|---|---|---|---|
| `vit_clipreid.yml` **(Stage 0 사용)** | `train_clipreid.py` (`:87`) | ViT-B-16 (`:6`) | [16, 16] (`:7`) | 주석 (`:11-12`) | 0.25 / 1.0 (`:8,10`) | [256, 128] (`:15-16`) | S1 120 (`:38`), S2 60 (`:53`) |
| `vit_base.yml` | `train.py` (`:71`) | ViT-B-16 (`:6`) | [16, 16] (`:7`) | 주석 (`:10-11`) | 1.0 / 없음 (`:8`) | [256, 128] (`:14-15`) | 60 (`:37`) |
| `cnn_clipreid.yml` | `train_clipreid.py` (`:84`) | RN50 (`:6`) | 키 없음, 기본 [16, 16] (`CR/config/defaults.py:58`) | 키 없음, 기본 False, 3.0 (`CR/config/defaults.py:61-62`) | 1.0 / 1.0 (`:7,9`) | [256, 128] (`:12-13`) | S1 60 (`:35`), S2 120 (`:50`) |
| `cnn_base.yml` | `train.py` (`:68`) | RN50 (`:6`) | 키 없음, 기본 [16, 16] (`CR/config/defaults_base.py:58`) | 키 없음, 기본 False, 3.0 (`CR/config/defaults_base.py:61-62`) | 1.0 / 없음 (`:7`) | [256, 128] (`:11-12`) | 120 (`:34`) |

SIE+OLP 실행 예(`CR/README.md:61`): `MODEL.SIE_CAMERA True MODEL.SIE_COE 1.0 MODEL.STRIDE_SIZE '[12, 12]'`

**Stage 0이 잰 것은 SIE·OLP가 없는 기본 ViT-CLIP-ReID입니다.** OLP(stride 12)는 토큰이 더 많고(3.1절) 측정된 적이 없습니다.

### 5.5 Market-1501 공개 가중치

README 표에는 링크만 있고 **mAP, Rank-1 숫자가 없습니다**(`CR/README.md:78-86`). 표 아래에는 재순위 없는 결과라는 설명과 MSMT17 수치만 있습니다(`CR/README.md:88,90`). 그래서 수치는 표에 링크된 공개 test 로그(텍스트 약 4 KB)에서 읽었습니다. 가중치 자체는 내려받지 않았습니다.

| README 행 | 대응 config | Market mAP | Market R1 | 모델 | test 로그 |
|---|---|---:|---:|---|---|
| CNN-baseline (`:82`) | `cnn_base.yml`. 로그 표기 `res_base.yml` | 88.1 | 94.7 | [model](https://drive.google.com/file/d/15E4K9eGXMlqOGE1RAgXQjF4MzrFobGim/view?usp=share_link) | [test](https://drive.google.com/file/d/1CxzntZ8531NWmnp6AUrZh8GCWgunF2XA/view?usp=share_link) |
| CNN-CLIP-ReID (`:83`) | `cnn_clipreid.yml`. 로그 표기 `res_prom.yml` | 89.8 | 95.7 | [model](https://drive.google.com/file/d/1sBqCr5LxKcO9J2V0IvLQPb0wzwVzIZUp/view?usp=share_link) | [test](https://drive.google.com/file/d/1u2x5_c5iNYaQW6sL5SazP4NUMBnCNZb9/view?usp=share_link) |
| ViT-baseline (`:84`) | `vit_base.yml` | 86.4 | 93.3 | [model](https://drive.google.com/file/d/1XKUcP4LEpWr4Ah6sVdXNveUo4bAsVyjt/view?usp=share_link) | [test](https://drive.google.com/file/d/18xkr609oK_TdOzVZviZYMq48ZVANO5S6/view?usp=share_link) |
| **ViT-CLIP-ReID** (`:85`) | `vit_clipreid.yml`에 해당. 로그 표기 `vit_prom.yml`, **`I2T_LOSS_WEIGHT` 0.5** | **89.6** | **95.5** | [model](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view?usp=share_link) | [test](https://drive.google.com/file/d/1SKtpls1rtcuC-Xul-uVhEOtFKf8a1zDt/view?usp=share_link) |
| **ViT-CLIP-ReID-SIE-OLP** (`:86`) | `vit_clipreid.yml` + `CR/README.md:61` override. 로그 표기 `vit_prom.yml`, stride [12, 12], SIE True, COE 1.0, I2T 1.0 | **90.5** | **95.4** | [model](https://drive.google.com/file/d/1K32xrosw0gPrxYCWXER81mhWObEW5-d4/view?usp=share_link) | [test](https://drive.google.com/file/d/1UqE0zCTSaob4NMgKN_wjBEdtJJPSb3hW/view?usp=share_link) |

모든 로그의 입력은 [256, 128]이고 re-ranking은 꺼져 있습니다. ViT-CLIP-ReID와 SIE-OLP 로그의 평가 배치는 64, 특징은 `before`입니다. 세 CLIP-ReID 행 수치는 논문 표(arXiv 2211.13977 HTML을 요약 도구로 읽음)와 같습니다.

**주의할 점**

| 항목 | 내용 | 근거 |
|---|---|---|
| I2T 가중치 불일치 | ViT-CLIP-ReID test 로그의 yml은 `I2T_LOSS_WEIGHT: 0.5`인데 저장소 yml은 1.0입니다. 그 외 레시피 값은 일치합니다. 이 값은 학습에만 쓰이고 test 실행 때의 yml이므로 **학습 당시 값은 확인 못함**입니다 | test 로그 13, 103행, `yml:10` |
| 파일명 불일치 | 로그의 config 이름은 저장소에 없는 `configs/person/vit_prom.yml` 등입니다. 공개 후 이름이 바뀐 것으로 보이나 추정입니다 | test 로그 1행 |
| 크기 제약 | 공개 가중치는 256×128(129토큰)이고 SIE-OLP는 211토큰입니다. 보간 없이 복사하므로 384×192, 512×256에 그대로 쓸 수 없습니다(정적 추론) | `CR/model/make_model_clipreid.py:load_param:156-160` |
| 링크 상태 | ViT-CLIP-ReID, SIE-OLP 모델 페이지는 HTTP 200. 다운로드 가능 여부는 미확인 | 2026-10-31 HTTP 요청 |

## 6. 공식 입력 크기 확인

| 모델 | H×W | 인용 |
|---|---|---|
| CLIP-ReID ViT (`vit_clipreid.yml`) | 256×128 | `CR/configs/person/vit_clipreid.yml:15-16`: `SIZE_TRAIN: [256, 128]`, `SIZE_TEST: [256, 128]` |
| CLIP-ReID 나머지 3개 config | 256×128 | `CR/configs/person/vit_base.yml:14-15`, `CR/configs/person/cnn_clipreid.yml:12-13`, `CR/configs/person/cnn_base.yml:11-12` |

코드 기본값은 [384, 128]이지만(`CR/config/defaults.py:70,72`) yml이 덮어씁니다. yml 없이 실행하면 384×128이 되므로 "기본값"을 공식 크기로 읽으면 안 됩니다.

## 7. 확인 못함

| 항목 | 이유 |
|---|---|
| 공개 가중치의 학습 당시 `I2T_LOSS_WEIGHT` | test 로그만 있음. 학습 로그는 공개되지 않음 |
| 공개 가중치의 내용과 실제 재현 수치 | 내려받거나 실행하지 않음 |
| 공개 가중치를 만든 torch 버전 | test 로그에 버전이 찍히지 않음 |
| torch 1.13과 1.8의 학습 결과 차이 | 학습을 돌리지 않음 |
| 공식 실행 때의 timm `RandomErasing` 값 | README에 버전이 없음 |
| 512×256의 메모리와 속도, stride 12의 메모리와 속도 | Stage 0에 없음 |
| float16 추정, shape 오류 추정, LR 값 | 코드에서 추론·계산했고 실행으로 확인하지 않음 |
