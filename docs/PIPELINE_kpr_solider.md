# KPR+SOLIDER 파이프라인 정적 분석: Market-1501

코드만 읽어 작성했습니다. 학습과 GPU 실행은 하지 않았고, 이미지 파일과 test 분할, 가중치 파일 내용은 열지 않았습니다. 메모리 값은 코드에서 유도한 계산값입니다.

공통 경로는 `docs/PIPELINE_kpr.md`(이하 **KPR 문서**)를 참조합니다. 이 문서는 차이를 전부 적습니다. commit, 경로 약어, 짧은 파일명, 인용 형식은 KPR 문서와 같습니다.

추가 약어:

- `SO-train` = `K/configs/kpr/solider/kpr_market_train.yaml`
- `SO-test` = `K/configs/kpr/solider/kpr_market_test.yaml`
- `IN-train`, `IN-test`, `dflt`는 KPR 문서와 같음

## 핵심 요약

1. **입력 크기는 KPR과 같은 384×128입니다** (`SO-train:6-7`, `SO-test:6-7`).
2. 코드 경로는 KPR과 같습니다. 같은 클래스 `PromptableSoliderSwinTransformer`, 같은 Swin 구조, 같은 데이터 변환 코드입니다. 따라서 RAM 역추적 결과(§6)도 입력 크기와 파트 수가 같아 그대로 적용됩니다.
3. train yaml의 실제 차이는 정규화, BIPO 확률(0.2 대 0.05), backbone 가중치, lr(0.0002 대 0.008), `fixbase_epoch`(20 대 0)입니다.
4. `fixbase_epoch=20`은 **백본이 아니라 프롬프트 토크나이저 `masks_patch_embed`만** 얼립니다. `STAGE0_REPORT.md:139`의 "백본을 고정"은 코드와 다릅니다.
5. test yaml과 train yaml이 `sie_camera`, `semantic_weight`에서 다릅니다. 공개 가중치를 재현하려는 경우 train yaml만으로 같은 설정이 된다고 볼 수 없습니다 (§4.3).

---

## 차이 전체 목록 (train yaml)

`diff IN-train SO-train` 결과 전부입니다.

| # | 키 | KPR | KPR+SOLIDER | 근거 | 동작 영향 |
|---:|---|---|---|---|---|
| 1 | 주석 | 없음 | `# Override default ...` | `SO-train:1` | 없음 |
| 2 | `data.norm_mean` | [0.485, 0.456, 0.406] | [0.5, 0.5, 0.5] | `IN-train:6`, `SO-train:8` | Normalize |
| 3 | `data.norm_std` | [0.229, 0.224, 0.225] | [0.5, 0.5, 0.5] | `IN-train:7`, `SO-train:9` | Normalize |
| 4 | `data.bipo.p` | 0.05 | 미지정 → **기본 0.2** | `IN-train:8-9`, `dflt:195` | BIPO 빈도 4배, CPU |
| 5 | `data.transforms` | 미지정 → 기본 | `["rc","re","bipo"]` | `SO-train:10`, `dflt:186-190` | 값이 기본과 같아 없음 |
| 6 | `model.kpr.backbone` | `imagenet_swin_...` | `solider_swin_...` | `IN-train:13`, `SO-train:14` | 가중치, 변환 인자 |
| 7 | `train.lr` | 미지정 → 0.008 | 0.0002 | `dflt:289`, `SO-train:25` | lr, scheduler `max_lr` |
| 8 | `train.reduced_lr` | 미지정 → 0.0002 | 0.0002 | `dflt:290`, `SO-train:26` | 없음 (값 같고, `staged_lr=False`라 미사용, `optimizer.py:80`) |
| 9 | `train.fixbase_epoch` | 미지정 → 0 | 20 | `dflt:297`, `SO-train:27` | 앞 20 epoch `masks_patch_embed` 고정 |

그 외 모든 값은 `dflt`에서 같습니다. `keypoints.prompt_masks`, `prompt_preprocess`, `masks.dir`, `masks.preprocess`, `transreid.cam_num`도 양쪽이 같습니다 (`IN-train:14-21`, `SO-train:15-22`).

---

## 1. 학습 경로

### 1.1 진입, 로딩

KPR 문서 §1.1, §1.2와 같습니다. `SO-train`을 `--config-file`로 넘기는 점만 다릅니다. BIPO occluder 적재도 같습니다 (`dataset.py:111-114`, `datamanager.py:200`, `SO-train:10`).

### 1.2 augmentation

종류와 순서는 KPR 문서 §1.3과 같습니다. **값의 차이는 다음 세 곳입니다.**

| # | 변환 | 차이 | 근거 |
|---:|---|---|---|
| 3 | `BIPO` | `p=0.2` (KPR 0.05) | `dflt:195`, `transforms.py:89` |
| 6 | `Normalize` | mean, std 0.5 | `SO-train:8-9`, `transforms.py:73` |
| 7 | `CoarseDropout` | `fill_value=[0.5,0.5,0.5]` (`norm_mean` 재사용) | `SO-train:8`, `transforms.py:126` |

### 1.3 프롬프트, target 마스크

KPR 문서 §1.4, §1.5와 같습니다. `[8, H, W]` float64, `cck6`, `five_v`입니다 (`SO-train:16-20`). BIPO가 더 자주 적용되므로 BIPO occluder 스켈레톤에서 오는 negative 채널 계산(`dataset.py:488-496`)도 평균적으로 더 자주 돕니다.

### 1.4 모델

구조는 같은 클래스이고 생성 함수만 다릅니다 (`K/torchreid/models/promptable_solider.py:228-231`).

| 항목 | KPR | KPR+SOLIDER | 근거 |
|---|---|---|---|
| 생성 함수 | `imagenet_swin` | `solider_swin` | `promptable_solider.py:188`, `:148` |
| 사전학습 파일 | `SOLIDER/swin_base_patch4_window7_224_22k.pth` | `SOLIDER/swin_base_reid.pth` | `:204`, `:164` |
| `convert_weights` | True | False | `:205`, `:165` |
| Swin 구조 | 같음 (embed 128, depths 2,2,18,2, window 7) | 같음 | `:26-50` |
| `semantic_weight` | 0.2 | 0.2 (train yaml 미지정) | `:221`, `:181`, `dflt:170` |
| `mask_path_emb_init_zeros` | True | True | `:222`, `:182`, `dflt:172` |
| `sie_camera`, `sie_coe` | True, 3 | True, 3 (train yaml 미지정) | `:211`, `:171`, `dflt:164-165` |

가중치 출처 기록: `swin_base_reid.pth`는 KPR README 공식 GDrive, id `1j9P2d_7ABjyItVaHwQHO87o_PlPtQ3_E`입니다 (`weights/stage0/SHA256.csv:4`). 파일은 열지 않았습니다.

### 1.5 `fixbase_epoch=20`의 실제 동작

`K/torchreid/engine/engine.py:Engine.two_stepped_transfer_learning:536-577`

- `model.backbone_appearance_feature_extractor`가 `PromptableTransformerBackbone`이면 `is_promptable_transformer=True`입니다 (`engine.py:552-554`).
- epoch+1 ≤ 20: `masks_patch_embed.eval()`, `requires_grad=False` (`:557-562`).
- 그 뒤: `masks_patch_embed.train()`, `requires_grad=True` (`:571-575`).
- `open_layers`는 기본 `[]`로 None이 아니어서 조건을 통과합니다 (`dflt:298`, `:526`, `engine.py:557`).
- **Swin 본체는 처음부터 학습됩니다.** `dflt:297`의 주석과 README도 같은 내용입니다 (`K/README.md:376-377`).

`STAGE0_REPORT.md:139`의 "처음 20 epoch 동안 백본을 고정합니다"는 코드와 다릅니다. 같은 줄은 그 구간을 "보수적으로 잡은 값"이라고 설명하지만, 실제 차이는 conv 하나의 gradient 수준으로 보입니다 (코드 읽기 추정).

`masks_patch_embed`는 0으로 초기화됩니다 (`promptable_solider.py:65-66`, `swin_transformer.py:PatchEmbed.zero_weights:498-501`). 얼어 있는 동안 프롬프트 토큰이 0 근처로 유지될 가능성이 높습니다. 이는 추정이며, LayerNorm을 거치는 값을 실행으로 확인하지 않았습니다. 데이터 로더의 프롬프트 생성과 GPU 전송은 이 구간에도 그대로 실행됩니다 (`dataset.py:483-513`, `part_based_engine.py:566-567`).

Stage 0 측정은 `two_stepped_transfer_learning`을 호출하지 않습니다 (`stage0/backends.py:366-369`). 즉 이 고정 구간은 반영되지 않았습니다.

### 1.6 손실, 옵티마이저, 스케줄러

손실은 KPR 문서 §1.7과 같습니다. 옵티마이저와 스케줄러는 값만 다릅니다.

| 항목 | KPR | KPR+SOLIDER | 근거 |
|---|---|---|---|
| lr | 0.008 | 0.0002 | `dflt:289`, `SO-train:25` |
| bias lr 지정값 (스케줄러가 덮어씀) | 0.016 | 0.0004 | `optimizer.py:138`, KPR 문서 §1.8 |
| scheduler `max_lr` | 0.008 | 0.0002 | `lr_scheduler.py:107`, `default_config.py:lr_scheduler_kwargs:517` |
| scheduler `min_lr` | 1.6e-05 | 1.6e-05 (같은 하드코딩) | `lr_scheduler.py:108` |
| `min_lr/max_lr` | 0.2% | 8% | 계산 |

epoch e의 lr은 e<5에서 `1.6e-5 + (2e-4-1.6e-5)·e/5`, e≥5에서 `1.6e-5 + (2e-4-1.6e-5)·(1+cos(π(e-5)/115))/2`입니다.

---

## 2. 평가 경로

KPR 문서 §2와 같습니다. 학습 뒤 최종 test는 **학습 설정 그대로** `sie_camera=True`, `semantic_weight=0.2`를 씁니다. `test_embeddings`, 거리, 지표, re-ranking, 주기 평가, 마지막 항목 전부 같고, `SO-train`은 이 키들을 지정하지 않습니다.

test 전용 실행(`test.evaluate=True`)은 §4.3의 차이가 있습니다.

---

## 3. 입력 크기 변경 시 영향받는 지점

KPR 문서 §3 전체가 그대로 적용됩니다. 근거는 같은 클래스(`promptable_solider.py:9`), 같은 `window_size=7`(`:36`), 같은 패치 크기(`:27`), 같은 변환 코드입니다.

| 항목 | 384×128 | 384×192 | 512×256 |
|---|---:|---:|---:|
| stage 1 특징맵 → 패딩 후 | 96×32 → 98×35 | 96×48 → 98×49 | 128×64 → 133×70 |
| stage 2 | 48×16 → 49×21 | 48×24 → 49×28 | 64×32 → 70×35 |
| stage 3 | 24×8 → 28×14 | 24×12 → 28×14 | 32×16 → 35×21 |
| stage 4 | 12×4 → 14×7 | 12×6 → 14×7 | 16×8 → 21×14 |
| 절대 위치 임베딩 | 없음 | 없음 | 없음 |
| part attention target (Hf×Wf) | 96×32 | 96×48 | 128×64 |
| prompt_masks | 8×384×128 | 8×384×192 | 8×512×256 |

SOLIDER 고유 사항 두 가지:

- 파일 `swin_base_reid.pth`의 `absolute_pos_embed`, 상대 위치 표 내용은 열지 않았습니다. 코드상 상대 위치 표는 크기가 다를 때만 보간되고 (`swin_transformer.py:1345-1360`), 입력 크기와 무관합니다.
- `semantic_weight`는 입력 크기와 관계없는 스칼라 임베딩입니다 (`promptable_solider.py:115-118`, `:133-136`).

크기에 연동되지 않는 값도 KPR 문서 §3.7과 같습니다. 변경을 제안하지 않습니다. 단 BIPO `p`가 더 크므로 occluder 관련 CPU 비용은 KPR보다 더 자주 발생합니다.

---

## 4. 실제 config 값

### 4.1 값 표

KPR 문서 §4와 값이 같은 항목은 한 칸에 모아 적었습니다.

| 항목 | KPR+SOLIDER | 출처 | KPR과 비교 |
|---|---|---|---|
| 입력 H×W | 384×128 | `SO-train:6-7` | 같음 |
| 정규화 | 0.5, 0.5 | `SO-train:8-9` | **다름** |
| backbone | `solider_swin_base_patch4_window7_224` | `SO-train:14` | **다름** |
| transforms | `["rc","re","bipo"]` | `SO-train:10` | 값 같음 |
| `bipo.p` | 0.2 | `dflt:195` | **다름** |
| 배치 | 64 | `dflt:296` | 같음 |
| P×K | 16×4 | `dflt:274-276`, `sampler.py:30` | 같음 |
| epoch | 120 | `dflt:294` | 같음 |
| `fixbase_epoch` | 20 | `SO-train:27` | **다름** |
| lr | 0.0002 | `SO-train:25` | **다름** |
| `reduced_lr` | 0.0002, 미사용 | `SO-train:26`, `dflt:299` | 같음 |
| 스케줄 | cosine warmup 5, max 0.0002, min 1.6e-5 | `dflt:293`, `:302`, `lr_scheduler.py:103-111` | **max 다름** |
| 옵티마이저 | SGD, momentum 0.9, wd 1e-4 | `dflt:288`, `:291`, `:316` | 같음 |
| AMP | True, 학습 `forward_backward`만 | `dflt:311`, `part_based_engine.py:99`, `:203` | 같음 |
| K (`parts_num`) | 5 | `SO-train:20` | 같음 |
| 프롬프트 파트 수 | 6 | `SO-train:17` | 같음 |
| 프롬프트 채널 | 8 | `promptable_solider.py:155-161` | 같음 |
| `test_embeddings` | `["bn_foreg","parts"]` | `dflt:91-94` | 같음 |
| 마스크 (코드) | target `[6,H/4,W/4]` float32 | `mask_transform.py:85-87` | 같음 |
| 마스크 (train 실측) | (36,17,9), float32, 22,160 바이트 | KPR 문서 §1.2, 같은 디렉터리 `SO-train:19` | 같음 |
| workers | 4 | `dflt:180` | 같음 |
| test 배치 | 128 | `dflt:360` | 같음 |
| `eval_freq` | -1 | `dflt:306-308` | 같음 |
| `test.visrank` | True | `dflt:372-374` | 같음 |
| `test.vis_feature_maps` | False | `dflt:379` | 같음 |
| save-best, 체크포인트 | 저장 없음 (`save_model_flag=False`), save-best 옵션 없음 | `dflt:69`, `engine.py:86-87` | 같음 |
| `semantic_weight` | 0.2 (train), -1.0 (test yaml) | `dflt:170`, `SO-test:27` | train 값은 같음 |
| `sie_camera` | True (train), False (test yaml) | `dflt:165`, `SO-test:25` | train 값은 같음 |

### 4.2 Stage 0 값과의 일치

Stage 0 표의 "KPR+SOLIDER sgd lr=0.0002"는 `SO-train:25`와 일치합니다 (`STAGE0_REPORT.md:15-16`). `fixbase_epoch`는 `meta()`로 기록되지만 측정 동작에 쓰이지 않습니다 (`stage0/backends.py:362`, `:366-369`).

### 4.3 `SO-train`과 `SO-test`의 차이

| 키 | train | test | 줄 |
|---|---|---|---:|
| `model.load_weights` | 없음 | `pretrained_models/kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar` | `SO-test:13` |
| `model.pretrained` | True (기본) | False | `SO-test:14` |
| `model.transreid.sie_camera` | True (기본) | **False** | `SO-test:25` |
| `model.solider.semantic_weight` | 0.2 (기본) | **-1.0** | `SO-test:27` |
| `test.evaluate` | False (기본) | True | `SO-test:30` |
| `train.*` | 지정 | 없음 | |

세 가지 사실을 함께 봐야 합니다.

- `build_config`는 체크포인트의 `model.kpr`, `model.transreid` config를 yaml 위에 병합합니다 (`builder.py:162-200`). `model.solider`는 병합하지 않으므로 `semantic_weight=-1.0`은 yaml 값이 유지됩니다 (코드상).
- 가중치 파일은 `sie_camera=False`, `semantic_weight<0`으로 학습됐을 가능성이 있습니다. 파일을 열지 않아 추정이며, README에 학습 config가 불완전할 수 있다는 언급이 있습니다 (`K/README.md:331`, `:378`).
- 따라서 `SO-train` 그대로의 학습과 공개 가중치의 설정이 같다고 단정할 수 없습니다.

KPR의 test yaml에는 이 두 키가 없습니다 (`IN-test:11-23`). KPR 쪽 train/test는 이 두 값이 같습니다.

---

## 5. 공식 입력 크기와 공개 가중치

### 5.1 입력 크기

| 모델 | 파일 | 값 | 줄 |
|---|---|---|---:|
| KPR | `configs/kpr/imagenet/kpr_market_train.yaml` | 384×128 | 4-5 |
| KPR | `configs/kpr/imagenet/kpr_market_test.yaml` | 384×128 | 6-7 |
| KPR+SOLIDER | `configs/kpr/solider/kpr_market_train.yaml` | 384×128 | 6-7 |
| KPR+SOLIDER | `configs/kpr/solider/kpr_market_test.yaml` | 384×128 | 6-7 |

**KPR+SOLIDER의 입력 설정은 KPR과 같습니다.**

### 5.2 공개 가중치

README에는 model zoo 표가 없습니다. 폴더 링크 하나만 있습니다.

- 가중치 폴더: `https://drive.google.com/drive/folders/1t4wXc2c3qlFaqUCifAlc_OPrFwvb7peD?usp=sharing` (`K/README.md:297`)
- README 본문에 Market-1501 mAP, Rank-1 수치 없음. PWC 배지만 있음 (`K/README.md:45`)
- 모델별 개별 링크 없음
- 실행 안내: `python main.py --config-file configs/kpr/<pretraining>/kpr_<target_dataset>_test.yaml` (`K/README.md:306`)

test yaml의 파일명:

| 모델 | test yaml | 가중치 파일명 | 파일명 속 수치 (mAP, Rank-1로 보임, 명시 없음) | 줄 |
|---|---|---|---|---:|
| KPR (ImageNet) | `configs/kpr/imagenet/kpr_market_test.yaml` | `kpr_market_IN_89.58_95.87_41898449.pth.tar` | 89.58, 95.87 | 12 |
| KPR+SOLIDER | `configs/kpr/solider/kpr_market_test.yaml` | `kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar` | 93.25, 96.59 | 13 |

같은 이름의 파일이 `weights/stage05/public/`에 있습니다. KPR+SOLIDER는 384,266,425 바이트입니다. 내용은 열지 않았습니다.

---

## 6. RAM 역추적 (코드 근거)

KPR 문서 §6이 모두 적용됩니다. `_feature_extraction`, `collate`, 거리, 지표, 그림 코드는 backbone에 분기하지 않습니다. 관련 값(`SO-train:17`, `:20`, `dflt:91-94`)도 같습니다.

### 6.1 판정 요약

| 가설 | 판정 | 근거 |
|---|---|---|
| `anns.append(data)`가 image, target_masks, prompt_masks, keypoints까지 누적 | **확인** | `part_based_engine.py:226`, `dataset.py:505-531` |
| `collate`가 concat으로 피크를 두 배로 | **확인** | `torchtools.py:371-379`, `:396-398`, `part_based_engine.py:234` |
| 거리 함수: gallery만 500개 블록, GPU 계산, CPU concat | **확인** | `distance.py:133-169`, `dflt:361` |
| `evaluate_rank`가 Market에서 anns 사용 | **부정** | `rank.py:168-169`, `market1501.py:108-112` |
| `plot_*`가 게이트 없이 Q×G 임시 배열 생성 | **확인** | `part_based_engine.py:389-403`, `distribution.py:36-37` |
| `writer.visualize_rank`의 임시 배열 | **확인**, `test.visrank` 게이트 | `writer.py:300`, `dflt:372-374` |
| `writer.visualize_embeddings` 활성 | **부정**, `test.vis_embedding_projection=False` | `writer.py:294`, `dflt:381` |
| `display_feature_maps` 활성 | **부정**, `test.vis_feature_maps=False` | `feature_map_visualization.py:148`, `dflt:379` |

### 6.2 배열 식

| 배열 | 형태 | dtype | 샘플당 바이트 |
|---|---|---|---:|
| `f_` | [N, K+1, 512] | float32 | 12,288 |
| `parts_visibility_` | [N, K+1] | bool | 6 |
| `p_masks_` | [N, K+1, H/4, W/4] | float32 | 1.5·H·W |
| `pxl_scores_` | [N, K+1, H/4, W/4] | float32 | 1.5·H·W |
| `anns.image` | [N, 3, H, W] | float32 | 12·H·W |
| `anns.prompt_masks` | [N, 8, H, W] | float64 | 64·H·W |
| `anns.target_masks` | [N, K+1, H/4, W/4] | float32 | 1.5·H·W |
| `anns.keypoints_xyc` | [N, 17, 4] | float64 | 544 |
| `anns.negative_keypoints_xyc` | [N, 5, 17, 4] | float64 | 2,720 |
| `anns.pid`, `camid` | [N] | int64 | 16 |

피크 (계산, N_q=3,368, N_g=15,913, `market1501.py:21`):

| 입력 | 전 배열 합 | gallery collate 피크 | occluder 상한 |
|---|---:|---:|---:|
| 384×128 | 71.3 GiB | 127.8 GiB | 2.72 GiB |
| 384×192 | 106.9 GiB | 191.6 GiB | 4.08 GiB |
| 512×256 | 189.7 GiB | 340.3 GiB | 7.25 GiB |

Stage 0의 두 모델 RSS가 거의 같은 것도 코드상 같은 경로와 일치합니다 (`STAGE0_REPORT.md:100-102`, `:111-113`).

### 6.3 스텝당 호출

KPR 문서 §6.6과 같습니다. `synchronize` 5회, `.item()` 13회이고, 스텝 단위 logger, 시각화는 없습니다. `fixbase` 구간에는 스텝이 아니라 epoch 시작 때 `print`와 `requires_grad` 설정이 추가됩니다 (`engine.py:559-562`, `:572-575`). `two_stepped_transfer_learning`은 `Engine.train`의 epoch 초입에서 불립니다 (`engine.py:262-264`).

### 6.4 데이터 로더 CPU 시간 (코드 읽기, 미측정)

순위는 KPR 문서 §6.8과 같고, **BIPO만 빈도가 4배**입니다 (p=0.2 대 0.05). 적용된 샘플에서 키포인트마다 `scipy.ndimage.shift`를 H×W 전체에 실행합니다 (`batch_wise_inter_person_occlusion.py:93-95`, `:469-474`). 그 비용 비중은 KPR보다 커집니다. 실제 비중은 측정된 적이 없습니다.

Stage 0의 KPR+SOLIDER 256×128 it/s(1.75)가 KPR(1.87)보다 낮은 것은 이 차이와 방향이 맞지만, 원인으로 확정하지는 않습니다 (`STAGE0_REPORT.md:13`, `:15`).
