# Stage 0: RTX 4090 실행 가능 범위 실측

학습이 아니라 측정입니다. test 이미지는 읽지 않았습니다. 크기 표기는 H×W입니다.

## 1. 모델·배율별 판정

| 모델 | 입력 | 판정 | 레시피 배치 | AMP | 옵티마이저 | allocated GiB | reserved GiB | nvidia-smi 피크 GiB | it/s | img/s | GPU 사용률 % |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| BoT | 256×128 | 4090 가능 | 64 | False | adam lr=0.00035 wd=0.0005 | 4.25 | 4.79 | 5.29 | 7.90 | 505.8 | 89 |
| BoT | 384×192 | 4090 가능 | 64 | False | adam lr=0.00035 wd=0.0005 | 9.11 | 10.11 | 10.60 | 5.09 | 325.8 | 87 |
| CLIP-ReID | 256×128 | 4090 가능 · CPU/데이터 로딩 병목 의심 | 64 | True | Adam lr=5e-06 | 5.84 | 6.35 | 7.21 | 8.63 | 552.2 | 63 |
| CLIP-ReID | 384×192 | 4090 가능 | 64 | True | Adam lr=5e-06 | 13.41 | 14.19 | 15.06 | 3.95 | 252.7 | 80 |
| KPR | 256×128 | 4090 가능 · CPU/데이터 로딩 병목 의심 | 64 | True | sgd lr=0.008 | 16.74 | 20.19 | 21.06 | 1.87 | 119.5 | 52 |
| KPR | 384×192 | 4090 불가, 필요 VRAM 추정치 39.8GiB (최대 배치 측정값의 선형 외삽) | 64 | True | sgd lr=0.008 | 21.69 | 22.24 | 22.96 | - | - | - |
| KPR+SOLIDER | 256×128 | 4090 가능 · CPU/데이터 로딩 병목 의심 | 64 | True | sgd lr=0.0002 | 16.74 | 20.19 | 21.06 | 1.75 | 111.8 | 48 |
| KPR+SOLIDER | 384×192 | 4090 불가, 필요 VRAM 추정치 39.8GiB (최대 배치 측정값의 선형 외삽) | 64 | True | sgd lr=0.0002 | 21.69 | 22.24 | 22.96 | - | - | - |

CLIP-ReID는 두 스테이지 중 이미지 인코더를 학습하는 stage 2의 값입니다. stage 1은 `results/stage0_probe.csv`의 `A_stage1` 행에 있습니다.

OOM 행의 메모리 값은 실패 직전까지의 피크입니다.

### checkpointing 등가성 (같은 seed, 같은 입력, 배치 8, 1스텝)

| 모델 | 입력 | loss on/off 차이 | grad norm off | grad norm on/off 차이 | 같은 설정 반복 차이 |
|---|---|---:|---:|---:|---:|
| KPR | 384×192 | 0.0 | 3573.33 | 2.476 | 1.186 |
| KPR+SOLIDER | 384×192 | 0.0 | 63010.40 | 0.156 | 0.742 |

### CLIP-ReID ×3 위치 임베딩 보간 확인

- `clipreid_x3_extract_b128.log`: Resized position embedding: %s to %s torch.Size([197, 768]) torch.Size([289, 768]) / Position embedding resize to height:24 width: 12
- `clipreid_x3_extract_b256.log`: Resized position embedding: %s to %s torch.Size([197, 768]) torch.Size([289, 768]) / Position embedding resize to height:24 width: 12
- `clipreid_x3_stage1_b64_w10m40.log`: Resized position embedding: %s to %s torch.Size([197, 768]) torch.Size([289, 768]) / Position embedding resize to height:24 width: 12
- `clipreid_x3_train_b64_w10m40.log`: Resized position embedding: %s to %s torch.Size([197, 768]) torch.Size([289, 768]) / Position embedding resize to height:24 width: 12

## 2. KPR·KPR+SOLIDER 최대 배치

정보용이며 채택안이 아닙니다. K=4의 배수만 탐색했습니다.

| 모델 | 입력 | 최대 배치 | allocated GiB | reserved GiB | it/s | img/s | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| KPR | 256×128 | 72 | 18.70 | 22.51 | 1.69 | 121.4 |  |
| KPR | 384×192 | 32 | 16.93 | 20.40 | 3.05 | 97.5 |  |
| KPR+SOLIDER | 256×128 | 72 | 18.70 | 22.51 | 1.75 | 126.2 |  |
| KPR+SOLIDER | 384×192 | 32 | 16.93 | 20.40 | 2.72 | 87.2 |  |

## 3. 학습 1회 예상 시간과 Stage 2 총 GPU 시간

모두 외삽입니다. 평가 시간은 특징 추출(배치 128)과 거리 계산이며 순위·mAP 계산은 빠져 있습니다.

| 모델 | 입력 | 학습 h | 평가 h | 근거 |
|---|---|---:|---:|---|
| BoT | 256×128 | 0.77 | 0.02 | 183×120÷7.90 |
| BoT | 384×192 | 1.20 | 0.02 | 183×120÷5.09 |
| CLIP-ReID | 256×128 | 0.71 | 0.01 | 185×60÷8.63 + stage1 203×120÷19.40 + 특징 캐시 12936÷621 |
| CLIP-ReID | 384×192 | 1.10 | 0.02 | 185×60÷3.95 + stage1 203×120÷22.33 + 특징 캐시 12936÷318 |
| KPR | 256×128 | 3.30 | 0.06 | 185×120÷1.87 |
| KPR | 384×192 | - | - | 레시피 배치의 속도 측정값 없음 |
| KPR+SOLIDER | 256×128 | 3.53 | 0.06 | 185×120÷1.75 |
| KPR+SOLIDER | 384×192 | - | - | 레시피 배치의 속도 측정값 없음 |

**Stage 2 전체(칸당 {bicubic + SR 2종} × seed 3 = 9회, 총 72회): 측정된 칸 합계 97.2 GPU 시간 (4.1일)**. 값이 없어 제외한 칸: KPR ×3, KPR+SOLIDER ×3.
한 번에 한 작업만 돌린다고 가정했으며 SR 생성 시간은 포함하지 않았습니다.

레시피 배치가 맞지 않는 칸의 참고값입니다. 위 합계에 넣지 않았습니다.

- KPR 384×192: 배치 32, 371×120÷3.05 = 학습 4.06시간 (배치를 바꾼 조건이라 레시피와 다르고, 학습 결과가 같다는 뜻이 아닙니다). 같은 배치에 checkpointing을 켜면 allocated 10.46GiB, 3.04 it/s입니다
- KPR+SOLIDER 384×192: 배치 32, 371×120÷2.72 = 학습 4.54시간 (배치를 바꾼 조건이라 레시피와 다르고, 학습 결과가 같다는 뜻이 아닙니다). 같은 배치에 checkpointing을 켜면 allocated 10.46GiB, 3.06 it/s입니다

### 평가 메모리 요약

상세는 `results/stage0_eval_mem.csv`입니다.

OOM 행의 메모리와 시간은 실패 시점까지의 값입니다.

| 모델 | 항목 | 장치 | 블록/배치 | 상태 | allocated GiB | RSS GiB | 초 | img/s | 전체 대비 최대 절대 오차 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| BoT | extract×3 | gpu | 128 | ok | 2.03 | 0.82 | 32.4 | 320.8 | - |
| BoT | extract×3 | gpu | 256 | ok | 3.96 | 0.92 | 27.4 | 406.4 | - |
| BoT | extract×2 | gpu | 128 | ok | 0.95 | 0.76 | 20.4 | 418.3 | - |
| BoT | dist | gpu | full | ok | 0.68 | 0.76 | 0.3 | - | 0.0 |
| BoT | dist | gpu | 256 | ok | 0.22 | 0.98 | 0.3 | - | 0.0 |
| BoT | dist | gpu | 512 | ok | 0.26 | 0.98 | 0.3 | - | 2.384185791015625e-07 |
| BoT | dist | gpu | 1024 | ok | 0.34 | 0.98 | 0.3 | - | 3.5762786865234375e-07 |
| BoT | dist | cpu | full | ok | 0.00 | 1.19 | 1.1 | - | 0.0 |
| BoT | dist | cpu | 256 | ok | 0.00 | 1.25 | 2.1 | - | 0.0 |
| BoT | dist | cpu | 512 | ok | 0.00 | 1.20 | 1.6 | - | 0.0 |
| BoT | dist | cpu | 1024 | ok | 0.00 | 1.20 | 1.2 | - | 0.0 |
| CLIP-ReID | extract×3 | gpu | 128 | ok | 2.29 | 3.42 | 41.5 | 317.9 | - |
| CLIP-ReID | extract×3 | gpu | 256 | ok | 4.09 | 3.52 | 29.5 | 407.2 | - |
| CLIP-ReID | extract×2 | gpu | 128 | ok | 1.29 | 3.37 | 20.4 | 620.5 | - |
| CLIP-ReID | dist | gpu | full | ok | 0.36 | 2.72 | 1.4 | - | 0.0 |
| CLIP-ReID | dist | gpu | 256 | ok | 0.20 | 2.95 | 1.5 | - | 0.0 |
| CLIP-ReID | dist | gpu | 512 | ok | 0.20 | 2.98 | 1.5 | - | 7.152557373046875e-07 |
| CLIP-ReID | dist | gpu | 1024 | ok | 0.20 | 2.95 | 1.5 | - | 7.152557373046875e-07 |
| CLIP-ReID | dist | cpu | full | ok | 0.00 | 0.75 | 1.1 | - | 0.0 |
| CLIP-ReID | dist | cpu | 256 | ok | 0.00 | 0.96 | 2.0 | - | 0.0 |
| CLIP-ReID | dist | cpu | 512 | ok | 0.00 | 0.98 | 1.6 | - | 0.0 |
| CLIP-ReID | dist | cpu | 1024 | ok | 0.00 | 0.98 | 1.5 | - | 0.0 |
| KPR | extract×3 | gpu | 128 | oom | 16.20 | 18.72 | 210.4 | - | - |
| KPR | extract×3 | gpu | 256 | oom | 19.09 | 31.03 | 226.0 | - | - |
| KPR | extract×2 | gpu | 128 | ok | 8.83 | 8.81 | 180.5 | 107.0 | - |
| KPR | dist | gpu | full | ok | 0.26 | 11.30 | 7.5 | - | 0.0 |
| KPR | dist | gpu | 256 | ok | 0.03 | 3.88 | 6.5 | - | 0.022093772888183594 |
| KPR | dist | gpu | 512 | ok | 0.04 | 4.44 | 6.8 | - | 0.01792740821838379 |
| KPR | dist | gpu | 1024 | ok | 0.09 | 5.70 | 6.6 | - | 0.01792740821838379 |
| KPR | dist | cpu | full | ok | 0.00 | 9.29 | 23.1 | - | 0.0 |
| KPR | dist | cpu | 256 | ok | 0.00 | 1.94 | 18.9 | - | 0.022093772888183594 |
| KPR | dist | cpu | 512 | ok | 0.00 | 2.56 | 16.7 | - | 0.01792764663696289 |
| KPR | dist | cpu | 1024 | ok | 0.00 | 3.82 | 16.1 | - | 0.01792764663696289 |
| KPR+SOLIDER | extract×3 | gpu | 128 | oom | 16.20 | 18.54 | 187.5 | - | - |
| KPR+SOLIDER | extract×3 | gpu | 256 | oom | 19.09 | 29.46 | 220.2 | - | - |
| KPR+SOLIDER | extract×2 | gpu | 128 | ok | 8.83 | 8.79 | 181.1 | 105.6 | - |
| KPR+SOLIDER | dist | gpu | full | ok | 0.26 | 11.30 | 13.0 | - | 0.0 |
| KPR+SOLIDER | dist | gpu | 256 | ok | 0.03 | 4.04 | 6.6 | - | 0.022093772888183594 |
| KPR+SOLIDER | dist | gpu | 512 | ok | 0.04 | 4.44 | 6.9 | - | 0.01792740821838379 |
| KPR+SOLIDER | dist | gpu | 1024 | ok | 0.09 | 5.69 | 6.6 | - | 0.01792740821838379 |
| KPR+SOLIDER | dist | cpu | full | ok | 0.00 | 9.65 | 24.1 | - | 0.0 |
| KPR+SOLIDER | dist | cpu | 256 | ok | 0.00 | 1.91 | 20.0 | - | 0.022093772888183594 |
| KPR+SOLIDER | dist | cpu | 512 | ok | 0.00 | 2.49 | 15.8 | - | 0.01792764663696289 |
| KPR+SOLIDER | dist | cpu | 1024 | ok | 0.00 | 3.83 | 15.1 | - | 0.01792764663696289 |

**KPR 거리는 블록 분할 결과가 전체 계산과 일치하지 않습니다.** 원인은 공식 함수(`torchreid/metrics/distance.py`)입니다. 공동으로 보이는 파트가 없는 쌍에 `블록 내 최댓값 + 1`을 넣기 때문에 그 값이 블록마다 달라집니다. CPU 소규모 재현(query 512, gallery 3,000, 블록 128)에서 공동 파트가 있는 쌍의 오차는 0.0, 없는 쌍의 오차만 최대 0.0246이었고, query 512개 모두 gallery 정렬 순서가 같았습니다. 이 소규모 재현은 무작위 특징과 무작위 가시성 기준이며, 실제 특징으로 확인한 것은 아닙니다.

## 4. 키포인트·마스크 스케일 검증

전체 통과 여부: **False** (`results/stage0_scale_check.json`). 원본 10장 기준입니다.

| 배율 | 공식 대비 최대 오차 | x·k와의 최대 차이 | 이론값 (k−1)/2 | 마스크 무게중심과 c·k 최대 차이(px) | 1px 이내 | 픽셀 중심 공식과 차이(px) | 저장 마스크 형태 |
|---:|---:|---:|---:|---:|---|---:|---|
| ×2 | 0.00e+00 | 0.500 | 0.5 | 0.522 | True | 0.027 | (36, 17, 9) |
| ×3 | 0.00e+00 | 1.000 | 1.0 | 1.037 | False | 0.059 | (36, 17, 9) |

키포인트는 픽셀 중심 규약이라 요청하신 `(x+0.5)·k−0.5`를 썼고, `x·k` 비교는 그 차이가 이론값과 같다는 확인입니다. 마스크는 17×9 고정 해상도라 스케일하지 않았습니다. 근거는 README입니다.

## 5. 확인하지 못한 항목과 그 이유

- 학습 시간은 50스텝 측정의 외삽입니다. 전체 학습을 돌리지 않았고, 검증 주기와 체크포인트 저장 시간은 넣지 않았습니다.
- KPR+SOLIDER 레시피는 처음 20 epoch 동안 백본을 고정합니다. 측정은 백본을 푼 상태라 그 구간은 메모리와 시간을 보수적으로 잡은 값입니다.
- KPR 공식 입력은 384×128입니다. 요청하신 매트릭스(256×128, 384×192)로 쟀으며 공식 입력은 재지 않았습니다.
- CLIP-ReID는 README의 torch 1.8 + CUDA 10.2가 RTX 4090을 지원하지 않아 torch 1.13.0 cu117로 쟀습니다. BoT는 파일럿과 같은 torch 2.1.2입니다.
- 데이터 로더는 네트워크 볼륨(MooseFS)에서 읽습니다. 로컬 디스크로 옮기면 병목 표시와 it/s가 달라질 수 있습니다.
- 평가 메모리의 거리 계산은 무작위 특징으로 쟀습니다. 실제 특징이어도 메모리는 같지만 시간은 조금 다를 수 있습니다. KPR 추출은 train 이미지로 쟀습니다.
- ×2 특징 추출(배치 128)은 지시 항목이 아니며, 학습 1회 시간 외삽에 필요해서 추가로 쟀습니다.
- KPR 계열 ×3 특징 추출은 배치 128과 256 모두 OOM이라 ×3 추출 속도와 평가 시간이 없습니다. 공식 평가 경로와 같이 autocast 없이 쟀습니다. 더 작은 배치는 지시에 없어 재지 않았습니다.
- 최대 배치 탐색에서 배치 4는 ID가 하나뿐이라 재지 못합니다. 하한은 8입니다.
- Swin ImageNet-22k 가중치는 KPR Google Drive에 없어 Microsoft 공식 릴리스에서 받았습니다.
- KPR 주석 zip에는 query, test 주석도 들어 있습니다. train 주석만 추출했고 zip은 chmod 000으로 잠갔습니다. 이미지는 들어 있지 않습니다.
- CLIP-ReID 공식 yml의 `DATASETS:`는 값이 전부 주석이라 yacs가 거부합니다. 저장소는 수정하지 않고 실행 시 그 키만 빼고 병합했으며, 나머지 값은 yml 그대로입니다.
- 측정 실패 기록: clipreid_x2_train_b64_w10m40 (수정 후 재측정 결과: ok): ValueError: Type mismatch (<class 'yacs.config.CfgNode'> vs. <class 'NoneType'>) with values (NAMES: market1501
ROOT_DIR: ../data vs. None) for config key: DATASETS
- 측정 실패 기록: clipreid_x2_stage1_b64_w10m40 (수정 후 재측정 결과: ok): ValueError: Type mismatch (<class 'yacs.config.CfgNode'> vs. <class 'NoneType'>) with values (NAMES: market1501
ROOT_DIR: ../data vs. None) for config key: DATASETS
- 측정 실패 기록: clipreid_x3_train_b64_w10m40 (수정 후 재측정 결과: ok): ValueError: Type mismatch (<class 'yacs.config.CfgNode'> vs. <class 'NoneType'>) with values (NAMES: market1501
ROOT_DIR: ../data vs. None) for config key: DATASETS
- 측정 실패 기록: clipreid_x3_stage1_b64_w10m40 (수정 후 재측정 결과: ok): ValueError: Type mismatch (<class 'yacs.config.CfgNode'> vs. <class 'NoneType'>) with values (NAMES: market1501
ROOT_DIR: ../data vs. None) for config key: DATASETS
- 측정 실패 기록: kpr_x3_train_b4_w3m7 (배치 4는 ID가 하나뿐이라 triplet loss 불가. 탐색 하한을 8로 고쳐 다시 탐색함): TypeError: cannot unpack non-iterable NoneType object
- 측정 실패 기록: kpr_x3_B (배치 4는 ID가 하나뿐이라 triplet loss 불가. 탐색 하한을 8로 고쳐 다시 탐색함): 시험 실패: TypeError: cannot unpack non-iterable NoneType object
- 측정 실패 기록: kpr_solider_x3_train_b4_w3m7 (배치 4는 ID가 하나뿐이라 triplet loss 불가. 탐색 하한을 8로 고쳐 다시 탐색함): TypeError: cannot unpack non-iterable NoneType object
- 측정 실패 기록: kpr_solider_x3_B (배치 4는 ID가 하나뿐이라 triplet loss 불가. 탐색 하한을 8로 고쳐 다시 탐색함): 시험 실패: TypeError: cannot unpack non-iterable NoneType object

## 출처

```
{
 "commits": {
  "CLIP-ReID": "eb1898b72c882875f478bebfc6d41644eece0a5d",
  "keypoint_promptable_reidentification": "e3e6ee2ffb74fd86a39518ce9a25ff91fbd973fa"
 },
 "licenses": {
  "CLIP-ReID": "MIT (third_party/CLIP-ReID/LICENSE)",
  "keypoint_promptable_reidentification": "Hippocratic License HL3-LAW-MEDIA-MIL-SOC-SV"
 },
 "solider_license": "tinyvision/SOLIDER Apache-2.0 (저장소 LICENSE 기준)",
 "kpr_annotation_zip": {
  "gdrive_id": "13ZPsAqKEHfSbL9luHZAZ6tALc_pFgjH7",
  "bytes": 1911198261
 }
}
```

가중치 sha256은 `weights/stage0/SHA256.csv`, 데이터 sha256은 `data/bicubic_x{2,3}_manifest.csv`에 있습니다.
