# 자체 구현 BoT 파이프라인 (정적 분석)

학습, GPU 실행, 코드 수정 없이 코드만 읽고 작성했습니다. 이미지와 `data/locked`, `data/market/{query,bounding_box_test,gt_query,gt_bbox}`는 열지 않았습니다. 크기는 H×W입니다.

**이 저장소의 BoT는 공식 reid-strong-baseline이 아니라 직접 구현입니다.** 공식 BoT와의 차이는 7절, 공개 가중치는 8절에 있습니다.

| 경로 표기 | 실제 위치 |
|---|---|
| (접두사 없음) | `/workspace/sr-reid/` 기준 상대 경로 |
| `OFF/` | 공식 `michuanhaohao/reid-strong-baseline` 커밋 `3da7e6f03164a92e696cb6da059b1cd771b0346d`(2020-04-23). 작업용 임시 clone이며 프로젝트 안에는 없고, 실행하지 않았습니다 |
| `ENV_CORE/` | `/workspace/miniforge3/envs/srreid-core/lib/python3.10/site-packages/` |
| `ENV_CLIP/` | `/workspace/miniforge3/envs/srreid-clipreid/lib/python3.10/site-packages/` |

행 수는 `data/processed/manifests/split.csv`와 `train_cache.csv`의 role별 집계입니다. 이미지 내용은 보지 않았습니다.

## 1. 학습 경로

### 1.1 데이터

| 단계 | 내용 | 근거 |
|---|---|---|
| 사전 캐시 | 원본 → PIL BICUBIC → `reid.input_size`, uint8 배열을 `train_hr256.npy`로 저장 | `data/make_lr.py:to_input:26-28`, `make_train_cache:52-58,63` |
| LR 캐시 | 원본 → 32×16 → 128×64 → `input_size`, 모두 BICUBIC. `train_lr256.npy`로 저장 | `data/make_lr.py:make_train_cache:59,64`, `to_lr:16-18`, `to_hr_size:21-23` |
| 현재 캐시 크기 | 파일 크기가 11,342×256×128×3 + 헤더 128바이트와 일치(1,114,964,096바이트). 배열을 메모리에 올려 확인한 값은 아님 | `data/processed/train_hr256.npy` 파일 크기 |
| 로드 | `np.load`로 전량 RAM에 올림(mmap 아님). `mixed`는 두 배열 모두 | `reid/trainer.py:TrainSet.__init__:31-32` |
| 라벨 | train651을 0부터 재번호. 11,342장, 651 ID | `reid/trainer.py:TrainSet.__init__:35-39`, `data/prepare_split.py:split:59-61`, `split.csv` |
| 경로 선택 | `hr`는 HR만, `lr_only`는 LR만, `mixed`는 샘플마다 p(LR)=0.5 | `reid/trainer.py:TrainSet.__getitem__:47-50`, `configs/pilot.yaml:35` |
| 샘플러 | ID별로 K=4장씩 묶어 P=16 ID 선택. K장 미만이면 복원추출. 에폭마다 다시 만듦 | `reid/trainer.py:pk_batches:76-95`, `train:147` |
| 로더 | 에폭마다 `DataLoader`를 새로 만듦. `num_workers=4`, `pin_memory=True` | `reid/trainer.py:train:148-149` |

### 1.2 증강 (적용 순서대로)

모두 `reid/trainer.py:TrainSet.__getitem__`에 있습니다. 값은 `configs/pilot.yaml:31`입니다.

| 순서 | 종류 | 확률 | 값 | 줄 |
|---:|---|---:|---|---:|
| 1 | 좌우 뒤집기 | 0.5 | - | 51-52 |
| 2 | 패딩 | 1 | 사방 10px, 값 0 | 53-56 |
| 3 | 랜덤 크롭 | 1 | 원래 H×W로. top, left ∈ {0..20} | 57-58 |
| 4 | 정규화 | 1 | /255 후 ImageNet mean, std (`reid/trainer.py:17-18`) | 58 |
| 5 | Random Erasing | 0.5 | 면적 U(0.02, 0.4)×H×W, 비율 U(0.3, 3.33), 최대 100회 시도, 채움값 0 | 59-67 |
| 6 | CHW 변환 | 1 | - | 68 |

색 변환과 추가 리사이즈는 없습니다. 리사이즈는 캐시 생성 때 끝납니다.

### 1.3 모델

| 항목 | 값 | 근거 |
|---|---|---|
| 백본 | torchvision `resnet50`, `weights/resnet50-0676ba61.pth` 로드 | `reid/model.py:BoT.__init__:10-12`, `configs/pilot.yaml:23` |
| Last stride | 1. `conv2`와 `downsample[0]`의 stride를 코드에서 직접 지정 | `reid/model.py:BoT.__init__:13-14` |
| 첫 단 ReLU | 있음. `avgpool`, `fc`만 잘라 `conv1, bn1, relu, maxpool, layer1~4` 유지 | `reid/model.py:BoT.__init__:15`, `ENV_CORE/torchvision/models/resnet.py:_forward_impl:270` |
| 풀링 | 전역 평균. 2048차원 | `reid/model.py:BoT.forward:22` |
| BNNeck | `BatchNorm1d(2048)`, bias 고정 | `reid/model.py:BoT.__init__:16-17` |
| 분류기 | bias 없는 Linear, std=0.001 초기화 | `reid/model.py:BoT.__init__:18-19` |
| 출력 | 학습은 (BN 뒤 logit, BN 앞 특징 g), 평가는 BN 뒤 특징 f | `reid/model.py:BoT.forward:23-26` |

### 1.4 손실, 옵티마이저, 스케줄러

| 항목 | 값 | 근거 |
|---|---|---|
| 손실 | CE(label smoothing 0.1) + batch-hard triplet(margin 0.3), 가중치 1:1. **center loss 없음** | `reid/trainer.py:train:157-158`, `configs/pilot.yaml:27` |
| Triplet | BN 앞 특징 g의 유클리드 거리, 정규화 없음, 최원 양성과 최근 음성 | `reid/trainer.py:triplet_hard:98-103` |
| 옵티마이저 | Adam, 단일 그룹(`requires_grad`만), lr 3.5e-4, wd 5e-4, beta 기본값 | `reid/trainer.py:train:131-132`, `configs/pilot.yaml:29` |
| 스케줄 | 에폭 단위. warmup 10에폭(factor 0.1), 0-based epoch 40과 70에서 ×0.1 | `reid/trainer.py:lr_at:106-112`, `train:150-151` |
| 길이 | 120에폭 고정, 조기 종료 없음 | `configs/pilot.yaml:30` |
| AMP | 없음. `autocast`, `GradScaler` 호출이 없음 | `reid/trainer.py:train:154-164` |

0-based 에폭별 lr은 e0 3.5e-5, e9 3.185e-4, e10~39 3.5e-4, e40~69 3.5e-5, e70~119 3.5e-6입니다(`lr_at`에서 계산).

## 2. 평가 경로

| 항목 | 값 | 근거 |
|---|---|---|
| 평가 대상 | val100: query 404장, gallery 1,181장(99 ID). 제외 query 9장. 공식 test split이 아님 | `eval/run_reid.py:main:58-59`, `split.csv` |
| 전처리 | PIL로 읽어 `input_size`로 BICUBIC 리사이즈 후 정규화 | `reid/extract.py:to_tensor:18-22` |
| 추출 방식 | 메인 프로세스에서 순차 디코딩, `DataLoader` 없음, 배치 128 | `reid/extract.py:extract:26-32`, `configs/pilot.yaml:32` |
| 특징 | BN 뒤 2048차원, L2 정규화 | `reid/model.py:BoT.forward:26`, `reid/extract.py:extract:31` |
| Flip test | 없음. 코드에 뒤집기 추출이 없음 | `reid/extract.py:extract:26-32` |
| 거리 | query별 `1 - g·q`. 전체 행렬을 만들지 않음 | `eval/metrics_reid.py:evaluate_query:6` |
| 지표 | Rank-1, AP. 같은 ID·같은 카메라 gallery 제외. CMC-5/10 없음 | `eval/metrics_reid.py:evaluate_query:8-15` |
| 예외 처리 | 양성 gallery가 없으면 `ValueError`. 출력 없는 query는 AP=0, R1=0으로 평균에 포함 | `eval/metrics_reid.py:evaluate_query:11-12`, `evaluate:19-25` |
| Re-ranking | 없음. 구현 자체가 없음 | `eval/metrics_reid.py` 전체 |
| 학습 중 평가 | **없음. 해당 config 키도, 에폭 목록도 없음.** 평가는 `eval/run_reid.py`를 따로 실행 | `reid/trainer.py:train:145-172` |

### 평가 중 메모리에 쌓이는 배열

Q=404, G=1,181 기준입니다. 평가에 쓰는 정수 배열의 dtype은 Linux의 NumPy 기본 정수(int64)로 추정했습니다.

| 배열 | shape | dtype | 크기 | 근거 |
|---|---|---|---:|---|
| `feats` 목록 → 결합 | [N, 2048] | float32 | - | `reid/extract.py:extract:27,31-32` |
| `g_feats` | [1181, 2048] | float32 | 9.2 MiB | `eval/run_reid.py:gallery:23` |
| `feats` | [404, 2048] | float32 | 3.2 MiB | `eval/run_reid.py:score:34,36` |
| `valid` | [404] | bool | - | `eval/run_reid.py:score:33` |
| `q_pids`, `q_cams` | [404] | int64 추정 | - | `eval/run_reid.py:score:39-40` |
| `g_pids`, `g_cams` | [1181] | int64 추정 | - | `eval/run_reid.py:score:41-42` |
| `ap`, `r1` | [404] | float64 | - | `eval/metrics_reid.py:evaluate:21` |
| query당 임시 `dist`, `order`, `keep`, `match`, `precision` | [1181] | float32, int64, bool, bool, float64 | - | `eval/metrics_reid.py:evaluate_query:6-14` |

gallery 특징은 `features/<mode>/gallery.npz`에 저장해 재사용합니다(`eval/run_reid.py:gallery:18-24`). Stage 0의 BoT 거리 측정은 이 경로가 아니라 무작위 특징 [3368, 19732] 전체 행렬이므로 실제 평가보다 큽니다(`stage0/worker.py:21,99,106`).

## 3. 입력 크기 변경 시 영향받는 지점

### 3.1 특징맵 크기

ResNet-50 last stride 1의 총 다운샘플은 16배입니다(첫 합성곱 2, maxpool 2, layer2 2, layer3 2, `ENV_CORE/torchvision/models/resnet.py`, `reid/model.py:13-14`). GAP 때문에 특징 차원은 입력과 무관합니다.

| 입력 | layer4 특징맵 | 공간 위치 수 | GAP 뒤 특징 | train651 캐시 1개 |
|---|---|---:|---:|---:|
| 256×128 | 16×8 | 128 | 2048 | 1.04 GiB |
| 384×192 | 24×12 | 288 | 2048 | 2.34 GiB |
| 512×256 | 32×16 | 512 | 2048 | 4.15 GiB |

캐시는 11,342×H×W×3 uint8로 계산했습니다. `mixed`는 HR, LR 두 개를 올리므로 2배입니다(`reid/trainer.py:31-32`).

### 3.2 크기에 의존하는 지점

| 지점 | 근거 | 입력 크기를 바꿀 때의 동작 |
|---|---|---|
| 입력 크기 키 | `configs/pilot.yaml:25` | 캐시와 평가 전처리가 읽음. 모델은 읽지 않음 |
| 학습 캐시 생성 | `data/make_lr.py:make_train_cache:52-59` | 크기가 바뀌면 다시 생성해야 함 |
| **하드코딩된 파일명** | `data/make_lr.py:63-64`, `reid/trainer.py:31-32` | `train_hr256.npy`, `train_lr256.npy` 이름은 크기와 무관하게 고정 |
| **크기 불일치 검사 없음** | `reid/trainer.py:TrainSet.__getitem__:54` | 학습 H×W는 캐시 배열에서, 평가 H×W는 config에서 옴. 캐시를 다시 만들지 않으면 학습은 옛 크기, 평가는 새 크기로 경고 없이 진행됨 |
| 평가 리사이즈 | `reid/extract.py:to_tensor:19-20` | PIL BICUBIC. 입력이 목표 크기와 같으면 복사만 함(`ENV_CORE/PIL/Image.py:2163-2164`) |
| SR 출력 크기 | `configs/pilot.yaml:48`, `sr/run.py:146`, `eval/run_reid.py:main:90` | 현재 `final/*.png`는 128×64. `to_tensor`가 이를 `input_size`로 BICUBIC 확대하므로, 현재 코드 그대로면 "모델 입력에 리사이즈 없음"이 성립하지 않음 |
| Path 1 LR 입력 | `data/make_lr.py:43`, `eval/run_reid.py:main:60-61` | `val_query_lrup`도 128×64 고정(`hr_size`). `to_tensor`에서 확대됨 |
| 패딩·크롭 | `reid/trainer.py:53-57`, `configs/pilot.yaml:31` | 절대값 10px |
| Random Erasing | `reid/trainer.py:59-65` | 면적 비율 기준이라 H×W를 따라감 |
| **GPU 메모리 상한** | `configs/pilot.yaml:40`, `reid/trainer.py:train:123`, `eval/run_reid.py:main:57` | 상한은 0.25×23.52 ≈ 5.88 GiB. Stage 0의 384×192 reserved는 10.11 GiB(`STAGE0_REPORT.md:10`). 설정 그대로면 384×192에서 OOM이 날 것으로 보임(정적 추론, 실행 안 함). Stage 0은 이 상한을 걸지 않음 |
| 시작 조건 | `configs/pilot.yaml:37`, `reid/trainer.py:train:120` | 여유 메모리 6 GB 기준. 큰 입력의 요구량보다 작음 |
| Stage 0 크기 | `stage0/backends.py:size_of:23-24`, `train_items:29` | 128×k, 64×k PNG를 그대로 사용. `pilot.yaml`의 `input_size`는 읽지 않음 |
| Stage 0 측정 범위 | `stage0/run_probe.py:285`, `stage0/prepare_data.py:45`, `stage0/report.py:11` | k=2, 3만. **512×256은 Stage 0 측정값이 없음**. `data/bicubic_x4`는 이 문서 작성 중(2026-10-31 10:41) 별도 작업으로 생겼으며, 이 분석의 측정 근거에는 쓰지 않음 |

모델 코드(`reid/model.py`)에는 입력 크기 의존 지점이나 하드코딩된 H×W가 없습니다.

### 3.3 개념상 크기와 함께 달라지는 값 (목록만, 변경 제안 아님)

| 값 | 현재 | 근거 | 256×128 | 384×192 | 512×256 |
|---|---|---|---:|---:|---:|
| 패딩·크롭 이동폭(절대 px), H 대비 | 10px | `configs/pilot.yaml:31` | 3.9% | 2.6% | 2.0% |
| 패딩·크롭 이동폭(절대 px), W 대비 | 10px | `configs/pilot.yaml:31` | 7.8% | 5.2% | 3.9% |
| 백본 수용영역의 입력 대비 비율 | 절대 px 고정 | `reid/model.py:10-15` | 기준 | 감소 | 더 감소 |

뒤집기 확률과 Random Erasing 면적 비율은 크기에 따라 달라지지 않습니다.

## 4. 실제 config 값

| 항목 | 값 | 인용 |
|---|---|---|
| 입력 H×W | [256, 128] | `configs/pilot.yaml:25` |
| 배치, PK | 64, P=16, K=4 | `configs/pilot.yaml:28` |
| 에폭 | 120 | `configs/pilot.yaml:30` |
| lr, wd | 3.5e-4, 5e-4 | `configs/pilot.yaml:29` |
| 스케줄 | warmup 10, factor 0.1, milestones [40, 70], gamma 0.1 | `configs/pilot.yaml:29` |
| 옵티마이저 | adam. 코드가 Adam으로 고정 | `configs/pilot.yaml:29`, `reid/trainer.py:131` |
| 손실 | LS 0.1, margin 0.3, center false | `configs/pilot.yaml:27` |
| 증강 | flip 0.5, pad_crop 10, RE 0.5 | `configs/pilot.yaml:31` |
| AMP | 사용 안 함(config 키 없음) | `reid/trainer.py:154-164`, `stage0/backends.py:BotBackend:47` |
| dtype | float32 | `reid/trainer.py:TrainSet.__getitem__:58` |
| num_workers | 4 | `configs/pilot.yaml:33` |
| Last stride | 1 | `configs/pilot.yaml:24`, 실제 동작은 `reid/model.py:13-14` |
| SIE, OLP, 패치 stride | 해당 없음(CNN) | `reid/model.py` |
| 체크포인트 | 5에폭마다 `last.pt` 덮어쓰기, 끝에 `model.pt` | `configs/pilot.yaml:34`, `reid/trainer.py:train:169-172,177-178` |
| 학습 중 평가 주기 | 없음 | `reid/trainer.py:train:145-172` |
| 평가 설정 | after_bnneck, flip_tta false, l2_norm true, batch 128 | `configs/pilot.yaml:32` |
| 시드 | 0, `cudnn.deterministic=True`, `benchmark=False` | `configs/pilot.yaml:3`, `reid/trainer.py:train:124-125` |
| GPU 비율 | 0.25 | `configs/pilot.yaml:40` |

**코드가 읽지 않는 키:** `reid.arch`(22줄), `reid.last_stride`(24줄), `reid.resize_filter`(26줄), `reid.loss.center`(27줄), `reid.optim.name`(29줄), `reid.eval.feature`, `flip_tta`, `l2_norm`(32줄), `reid.path1.primary`, `secondary`(35줄). `reid`, `eval`, `data`, `common`, `stage0`, `scripts`, `sr`의 Python 파일에서 해당 키의 첨자 접근과 `.get` 호출을 검색해 찾지 못했습니다. 이 값을 바꿔도 동작은 바뀌지 않고, 실제 동작은 코드에 고정되어 있습니다. 다른 방식으로 읽는 경로까지 배제하지는 못했습니다.

## 5. CLIP-ReID 전용 항목: 해당 없음. BoT의 Stage 0 측정 범위 기록

| 항목 | Stage 0 `BotBackend` | 실제 `reid/trainer.py` | 근거 |
|---|---|---|---|
| 데이터 | `data/bicubic_x{k}` PNG를 매번 디코딩 | npy 캐시 | `stage0/backends.py:_BotSet.__getitem__:116`, `reid/trainer.py:31-32` |
| 규모 | 12,936장, 751 ID | 11,342장, 651 ID | `stage0/backends.py:train_items:28`, `split.csv` |
| GPU 메모리 상한 | 없음 | 0.25 | `reid/trainer.py:123` |
| `cudnn.deterministic` | 설정 안 함 | True | `stage0/backends.py:BotBackend`, `stage0/worker.py`, `reid/trainer.py:124` |
| 배치 목록 | 한 번 만들어 반복 | 에폭마다 재생성 | `stage0/backends.py:65,73`, `reid/trainer.py:147` |
| torch | 2.1.2+cu118 | 같은 env로 추정 | `results/stage0_probe.csv:2-3`, `env/srreid-core.lock.txt:106` |

측정값은 256×128이 7.90 it/s, reserved 4.79 GiB이고 384×192가 5.09 it/s, reserved 10.11 GiB입니다(`STAGE0_REPORT.md:9-10`). `reid/trainer.py` 자체의 속도와 메모리를 잰 값이 아닙니다. 512×256은 없습니다.

## 6. 공식 입력 크기 확인

| 모델 | H×W | 인용 |
|---|---|---|
| 공식 BoT | 256×128 | `OFF/configs/softmax_triplet_with_center.yml:12-13`: `SIZE_TRAIN: [256, 128]`, `SIZE_TEST: [256, 128]` |
| 공식 BoT, center 없는 변형 | 256×128 | `OFF/configs/softmax_triplet.yml:12-13` |
| 자체 구현 BoT | 256×128 | `configs/pilot.yaml:25`: `input_size: [256, 128]` |

공식 코드의 기본값은 [384, 128]이지만(`OFF/config/defaults.py:52,54`) yml이 덮어씁니다.

## 7. 공식 BoT와 다른 점

공식 저장소를 clone해 양쪽 코드를 읽었으므로 아래 "양쪽 확인"은 기억이 아니라 코드로 대조한 결과입니다. 기준은 `OFF/configs/softmax_triplet_with_center.yml`입니다. 저장소만으로 확정할 수 없는 항목은 "공식 저장소 미확인"으로 표시했습니다.

### 7.1 레시피와 구조의 차이

| # | 항목 | 공식 | 자체 구현 | 상태 | 공식 근거 | 자체 근거 |
|---:|---|---|---|---|---|---|
| 1 | **Center loss** | 사용. weight 0.0005, center lr 0.5 | **없음** | 양쪽 확인 | `OFF/configs/softmax_triplet_with_center.yml:4,6,33-34` | `configs/pilot.yaml:27`, `reid/trainer.py:157-158`, `README.md:50` |
| 2 | **첫 단 ReLU** | **`conv1, bn1` 뒤 ReLU가 주석 처리되어 없음** | 있음 | 양쪽 확인 | `OFF/modeling/backbones/resnet.py:97,125` | `reid/model.py:10,15`, `ENV_CORE/torchvision/models/resnet.py:270` |
| 3 | **리사이즈 필터** | `T.Resize` 기본값(bilinear), 학습과 평가 모두 | PIL **BICUBIC**, 학습과 평가 모두 | 코드는 양쪽 확인. 공식 실험 당시 torchvision 버전은 공식 저장소 미확인 | `OFF/data/transforms/build.py:16,26` | `data/make_lr.py:26-28`, `reid/extract.py:20` |
| 4 | **학습 ID·장수** | train 전체(751 ID, 12,936장) | **train651(651 ID, 11,342장)** | 양쪽 확인 | `OFF/data/datasets/market1501.py:24,37`, `OFF/data/build.py:25` | `configs/pilot.yaml:11-13`, `data/prepare_split.py:59-61` |
| 5 | **평가 집합** | 공식 query 3,368장, gallery 15,913장 | **val100(query 404, gallery 1,181)** | 양쪽 확인 | `OFF/data/datasets/market1501.py:24,38-39`, `OFF/data/build.py:40` | `eval/run_reid.py:58-59`, `split.csv` |
| 6 | **Warmup factor** | 0.01 | 0.1 | 양쪽 확인 | `OFF/configs/softmax_triplet_with_center.yml:50-52` | `configs/pilot.yaml:29` |
| 7 | Warmup·감쇠 시점 | 첫 에폭 전에 `scheduler.step()`이 돌아 t=에폭(1-based). 10번째 에폭에 lr 최대, 40·70번째 에폭부터 감쇠 | t=0-based 에폭. 11번째 에폭에 최대, 41·71번째 에폭부터 감쇠 | 양쪽 확인(실행 안 함) | `OFF/engine/trainer.py:254-256`, `OFF/solver/lr_scheduler.py:43-55` | `reid/trainer.py:106-112,150-151` |
| 8 | **Random Erasing 채움값** | 정규화 뒤 텐서에 (0.485, 0.456, 0.406)을 기록. 실제 ImageNet 평균이 아님 | 0(정규화 공간의 평균) | 양쪽 확인 | `OFF/data/transforms/build.py:21-22`, `OFF/data/transforms/transforms.py:48-50` | `reid/trainer.py:66` |
| 9 | RE 비율 상한 | 1/0.3 = 3.333… | 3.33 | 양쪽 확인. 차이 미미 | `OFF/data/transforms/transforms.py:39` | `reid/trainer.py:62` |
| 10 | ImageNet 가중치 파일 | `resnet50-19c8e357.pth` | `resnet50-0676ba61.pth` | 파일명 양쪽 확인. 두 파일의 가중치가 수치로 같은지는 공식 저장소 미확인 | `OFF/configs/softmax_triplet_with_center.yml:3` | `configs/pilot.yaml:23`, `ENV_CORE/torchvision/models/resnet.py:358` |

두 warmup 설정의 차이는 factor 값만 보면 10배지만, 7번의 시점 차이 때문에 첫 에폭의 lr은 거의 같습니다.

| 1-based 에폭 | 공식 lr | 자체 lr |
|---:|---:|---:|
| 1 | 3.815e-5 | 3.5e-5 |
| 2 | 7.28e-5 | 6.65e-5 |
| 5 | 1.7675e-4 | 1.61e-4 |
| 10 | 3.5e-4 | 3.185e-4 |
| 11~39 | 3.5e-4 | 3.5e-4 |
| 40 | 3.5e-5 | 3.5e-4 |
| 41~69 | 3.5e-5 | 3.5e-5 |
| 70 | 3.5e-6 | 3.5e-5 |
| 71~120 | 3.5e-6 | 3.5e-6 |

코드에서 계산한 값이며 실행으로 확인하지 않았습니다.

### 7.2 평가와 운영의 차이

| # | 항목 | 공식 | 자체 구현 | 공식 근거 | 자체 근거 |
|---:|---|---|---|---|---|
| 11 | 학습 중 평가 | `EVAL_PERIOD: 40`, 에폭 40, 80, 120 | 없음 | `OFF/configs/softmax_triplet_with_center.yml:56`, `OFF/engine/trainer.py:281-283` | `reid/trainer.py:145-172` |
| 12 | 체크포인트 | 40에폭마다, 최대 10개 | 5에폭마다 덮어쓰기 | `OFF/configs/softmax_triplet_with_center.yml:54`, `OFF/engine/trainer.py:235` | `configs/pilot.yaml:34`, `reid/trainer.py:169-172` |
| 13 | 거리 | 제곱 유클리드 전체 행렬 | query별 `1 - 내적`. L2 정규화 후에는 순위가 같음(동점 제외) | `OFF/utils/reid_metric.py:47-50` | `eval/metrics_reid.py:6` |
| 14 | 정렬 | `argsort` 기본 | `kind="stable"`. 동점에서만 차이 | `OFF/data/datasets/eval_reid.py:18` | `eval/metrics_reid.py:7` |
| 15 | 양성 없는 query | 건너뜀 | `ValueError` | `OFF/data/datasets/eval_reid.py:38-40` | `eval/metrics_reid.py:11-12` |
| 16 | 출력 없는 query | 개념 없음 | AP=0, R1=0으로 평균에 포함 | - | `eval/metrics_reid.py:evaluate:20-25` |
| 17 | 지표 | CMC(max_rank 50), mAP | Rank-1, mAP | `OFF/data/datasets/eval_reid.py:10,59-61` | `eval/metrics_reid.py:13-16` |
| 18 | 특징 추출 | `DataLoader`, 8 workers | 메인 프로세스 순차 | `OFF/data/build.py:41-44` | `reid/extract.py:29-31` |
| 19 | num_workers | 8 | 4 | `OFF/configs/softmax_triplet_with_center.yml:24` | `configs/pilot.yaml:33` |
| 20 | 시드·cudnn | 시드 없음, `benchmark=True` | 시드 고정, `deterministic=True`, `benchmark=False` | `OFF/tools/train.py:153` | `reid/trainer.py:124-127,146` |
| 21 | 프레임워크 | pytorch>=0.4, ignite 0.1.2 | torch 2.1.2, 자체 루프 | `OFF/README.md:119-121` | `env/srreid-core.lock.txt:106` |
| 22 | 추가 학습 모드 | 없음 | `lr_only`, `mixed` | - | `reid/trainer.py:47-50`, `reid/train_lr.py` |

### 7.3 대조했고 같은 항목

| 항목 | 값 | 공식 근거 | 자체 근거 |
|---|---|---|---|
| 입력 크기 | 256×128 | `OFF/configs/softmax_triplet_with_center.yml:12-13` | `configs/pilot.yaml:25` |
| 배치, PK | 64, 16×4 | `:23,45` | `configs/pilot.yaml:28` |
| 샘플러 논리 | K 미만 복원추출, P개 무작위 선택. 묶음을 꺼내는 쪽만 다름(`pop(0)`, `pop()`) | `OFF/data/samplers/triplet_sampler.py:45-72` | `reid/trainer.py:76-95` |
| 옵티마이저 | Adam, 3.5e-4, wd 5e-4. bias lr 배수 1, bias wd 5e-4라 사실상 단일 그룹과 같음 | `:27,29,42-44`, `OFF/solver/build.py:make_optimizer_with_center:28-44` | `configs/pilot.yaml:29` |
| 에폭, 감쇠율 | 120, [40, 70], 0.1 | `:28,47-48` | `configs/pilot.yaml:29-30` |
| 증강 순서·확률 | 뒤집기 0.5, 패딩 10, 크롭, 정규화, RE 0.5 | `OFF/data/transforms/build.py:15-23` | `reid/trainer.py:51-59` |
| RE 면적, 시도 횟수 | 0.02~0.4, 100회 | `OFF/data/transforms/transforms.py:23,35-44` | `reid/trainer.py:60-64` |
| 평균, 표준편차 | ImageNet | `OFF/config/defaults.py:60,62` | `reid/trainer.py:17-18` |
| Last stride | 1 | `OFF/config/defaults.py:27` | `reid/model.py:13-14` |
| BNNeck, 분류기 | BN bias 고정, bias 없는 Linear, std 0.001 | `OFF/modeling/baseline.py:146-151` | `reid/model.py:16-19` |
| Triplet | BN 앞 특징, 유클리드, margin 0.3, hard mining | `OFF/config/defaults.py:101`, `OFF/layers/triplet_loss.py:38-116`, `OFF/modeling/baseline.py:165` | `reid/trainer.py:98-103,158` |
| Label smoothing | ε=0.1, 수식 동일(K만 751 대 651) | `OFF/layers/triplet_loss.py:129-147` | `reid/trainer.py:157` |
| 평가 특징 | BN 뒤, L2 정규화 | `:62-63` | `reid/model.py:26`, `reid/extract.py:31` |
| Re-ranking, flip test | 끔, 없음 | `:60` | `reid/extract.py:26-32` |
| AP 계산, 같은 ID·카메라 제외 | 동일 수식 | `OFF/data/datasets/eval_reid.py:32-33,50-54` | `eval/metrics_reid.py:8,14-15` |
| AMP | 없음 | `OFF/engine/trainer.py:77-91` | `reid/trainer.py:154-164` |

표에서 `:n`은 `OFF/configs/softmax_triplet_with_center.yml`의 줄입니다.

### 7.4 "BoT"라는 이름으로 쓸 때의 문제

해석이 들어간 부분이며 수치 효과는 실험으로 확인하지 않았습니다.

| 영향 | 내용 |
|---|---|
| 큼 | **공개 수치와 대조된 적이 없습니다.** 학습 ID와 평가 집합이 모두 달라 `hr_oracle`의 Rank-1 0.970, mAP 0.941(`PILOT_SUMMARY.md:74`)은 공식 94.1/85.7과 비교할 수 없습니다. `REPORT.md:138`도 대조 불가를 명시합니다. 구현이 BoT 수준을 재현하는지는 현재 근거가 없습니다 |
| 큼 | **연구 변수와 겹칩니다.** 리사이즈 필터(3번)는 SR 연구에서 입력 품질을 직접 바꾸는 요소입니다. 공식의 bilinear 경로와 다르고, 현재 평가 코드는 SR 출력을 BICUBIC으로 다시 확대합니다(3.2절) |
| 중간 | **정확한 대응 레시피는 center 없는 변형입니다.** 대조값은 `OFF/README.md:90`의 94.1(85.7)이며 `OFF/configs/softmax_triplet.yml:4,6`에 해당합니다. center 포함 94.5(85.9)(`OFF/README.md:91`)와 비교하면 안 됩니다 |
| 중간 | **첫 단 ReLU(2번)는 구조 차이입니다.** 정확도 영향은 확인 못함. 공식 수치는 ReLU 없는 구조에서 나왔습니다 |
| 작음 | Warmup(6, 7번): lr 차이는 warmup 구간과 감쇠 시점 1에폭에 한정됩니다. RE 채움값(8번)은 정규화 공간에서 0과 약 0.45의 차이입니다 |

논문이나 보고서에서는 "BoT 기반 자체 구현, center loss 제외"처럼 차이를 명시해야 정확합니다.

## 8. 공식 저장소의 학습된 가중치

| 항목 | 내용 | 근거 |
|---|---|---|
| 공개 여부 | 있음 | `OFF/README.md:107` |
| README 링크 | https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A | `OFF/README.md:107` |
| 실제 대상 | 위 링크는 Google Drive **폴더**로 리디렉션됨(HTTP 307): https://drive.google.com/drive/folders/1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A | 2026-10-31 HTTP 요청 |
| 폴더 내용 | `market_resnet50_model_120_rank1_945.zip.001`부터 `.045`까지 분할 압축. 첫 조각: https://drive.google.com/file/d/1cYQJ0_Kv-oPbeh0nwJPQUMoUHlOhZ1wb/view | 폴더 HTML 목록. 내려받지 않음 |
| 어떤 변형인가 | 파일명의 `120`과 `rank1_945`가 `+Center loss 94.5 (85.9)`(`OFF/README.md:91`)와 맞음. **파일명 기반 추정이며 README에는 config가 명시되어 있지 않음** | `OFF/README.md:107` 주변 |
| 입력 크기 | README, config 기준 256×128. 압축 내용은 확인 못함 | `OFF/configs/softmax_triplet_with_center.yml:12-13` |
| 자체 BoT 로드 | 그대로는 불가로 판단(실행 안 함). 공식 키는 `base.*`, `bottleneck.*`, `classifier.*`이고 자체 키는 `backbone.*`, `bnneck.*`, `classifier.*`이며, 활성화 구조도 다름(7.1의 2번) | `OFF/modeling/baseline.py:54,146-148`, `reid/model.py:15-18` |

## 9. 확인 못함

| 항목 | 이유 |
|---|---|
| 폴더의 전체 조각 수 | HTML 목록에서 45개까지 보였고, 목록이 잘렸는지 확인하지 못함 |
| 공개 zip의 내용, 조각 복원 가능 여부, 입력 크기, 실제 성능 | 내려받지 않음(지시 범위 밖) |
| `resnet50-19c8e357.pth`와 `resnet50-0676ba61.pth`의 가중치 일치 여부 | 앞의 파일이 프로젝트에 없고 내려받지 않음 |
| 차이 항목별 정확도 영향 | 학습을 돌리지 않음 |
| `reid/trainer.py`의 384×192, 512×256 실제 메모리·속도 | Stage 0이 다른 경로를 측정했고 512×256은 측정이 없음 |
| LR 스케줄의 실제 값 | 코드에서 계산했고 실행으로 확인하지 않음 |
