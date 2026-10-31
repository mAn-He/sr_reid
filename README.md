# SR → 사람 ReID 파일럿 (Phase 0)

저해상도 query를 SR로 복원하면, HR로 학습해 고정한 ReID 모델의 성능이 나아지는지 Market-1501 val split에서 점검합니다. 이 단계의 목적은 결론이 아니라 본 실험 설계의 go/no-go입니다.

## 실행

```bash
cd /workspace/sr-reid
tmux new -d -s srreid -c /workspace/sr-reid "bash scripts/run_pipeline.sh 2>&1 | tee -a logs/run_all.log"
```

- 합성 데이터 스모크(`configs/smoke.yaml`)를 먼저 돌리고, `scripts/smoke_gate.py`가 통과시켜야 본 실행(`scripts/run_all.sh`)을 시작합니다. 가중치 확보 실패는 통과를 막지 않으며 본 실행에서 해당 방법의 실패로 기록합니다.
- 재실행하면 `runs/<run_id>/done/`에 표식이 있는 단계와 sha256이 맞는 SR 출력을 건너뜁니다.
- 결과: `logs/STATUS.md`, `REPORT.md`, `results/pilot_summary.csv`, `results/grid.png`, `results/dose_response.png`, `results/failures.csv`.
- 전체 중단 조건은 두 가지입니다. 데이터 확보·검증과 LR 생성 게이트 실패, test 가드 위반입니다. 나머지 실패는 `failures.csv`에 기록하고 계속합니다.

## 연구 규칙

1. ReID 모델은 HR로만 학습해 고정합니다. SR 방법별 재학습은 없습니다.
2. 공식 test split은 이 단계에서 쓰지 않습니다. `query/`, `bounding_box_test/`, `gt_query/`, `gt_bbox/`는 `data/locked/`에 격리합니다.
3. train 751 ID 중 seed 0으로 100 ID를 떼어 val로 씁니다. 나머지 651 ID로 학습합니다.
4. 모든 입출력 이미지의 sha256을 manifest에 기록합니다. 학습용 HR·LR 배열은 행별 배열 sha256을 `train_cache.csv`에 기록합니다.
5. 외부 SR 저장소는 수정하지 않습니다.
6. 보행자 이미지를 외부 서비스로 보내지 않습니다. 추론은 `HF_HUB_OFFLINE=1`로 실행합니다.

**test 가드 주의:** 파드가 root로 돌아 `chmod 000`은 실제로 접근을 막지 못합니다. 실제 차단은 `common/guard.py`의 경로 검사와 audit hook이 담당하며, `ALLOW_TEST=1`일 때만 허용합니다.

## 데이터 출처

`Market-1501-v15.09.15.zip`은 사용자가 사실상 표준 배포 링크로 등록해 허가한 Google Drive 파일에서 받았습니다(layumi Person_reID_baseline_pytorch 튜토리얼이 안내하는 링크). **파일 소유자는 미확인입니다.** zip sha256은 `data/market_zip.sha256`에 있습니다. 데이터셋 readme 조건에 따라 연구 목적 외 사용과 재배포를 금지합니다.

## 결과 확인 전 등록한 결정

### Path 1 기준선
- **주 기준선: mixed (HR/LR 혼합, p(LR)=0.5 고정). 보조 기준선: LR-only.** 둘 다 처음부터 학습합니다.
- 이유: 평가 gallery가 HR이라 LR-only 모델은 학습 때 본 적 없는 HR 이미지로 gallery 특징을 뽑게 되어 약한 기준선이 됩니다.

### 리사이즈 경로
Path 1의 LR 입력은 학습과 평가 모두 32×16 → 128×64(bicubic) → 256×128을 씁니다. "bicubic" SR 방법과 같은 2단계 경로입니다. 바이트 동일 여부는 `runs/<run_id>/path1_bicubic_identity.json`에 기록합니다. 크기 표기는 H×W입니다.

### go/no-go 기준
`report/go_nogo.py`가 다음 기준을 그대로 계산합니다.
1. 안정성: 누락, 디코딩 실패, 크기 불일치, Y 표준편차 2 미만, PSNR 10 dB 미만을 깨진 이미지로 칩니다. 확산 모델별 `official` 비율이 1% 이하면 GO입니다.
2. 순위 변화: 대표 5개 방법의 정책 간 Kendall τ. 뒤집힌 쌍 중 차이의 CI가 0을 벗어나는 쌍이 있으면 "변함", 뒤집히기만 했으면 "불확실"입니다.
3. 소요 외삽: `exclusive_gpu=true` 측정값 × 3,368장(문헌상 공식 query 수, 파일은 읽지 않음) × 조건 수.
4. 용량-반응: 점추정이 단조 감소하고 `mAP(λ=2) − mAP(λ=0)`의 CI 상한이 0 미만이면 "신호 있음"입니다. 가설 신호일 뿐 결론이 아닙니다.

## ReID 구현 선택

fast-reid 대신 **직접 구현**했습니다. 의존성이 torch와 torchvision뿐이라 fast-reid의 yacs, Cython 평가 모듈 등이 필요 없고, 입력 경로와 혼합 비율을 저장소 수정 없이 통제할 수 있습니다. 설정은 ResNet-50, BNNeck, last stride 1, 입력 256×128, CE(label smoothing 0.1) + batch-hard triplet(margin 0.3)입니다. 원 BoT의 center loss는 지시에 따라 뺐습니다. 120 에폭 고정이며 val 기반 조기 종료는 없습니다.

## SR 실행과 선확대 정책

`sr/run.py`의 `run(in_dir, out_dir, cfg)`가 공통 인터페이스이고 `timing.json` 경로를 돌려줍니다. 외부 스크립트는 `sr/runner.py`가 같은 프로세스에서 `runpy`로 실행합니다. 그래야 `torch.cuda.max_memory_allocated`를 읽을 수 있습니다. 이미지별 시간은 저장 함수에 런타임 훅을 걸어 저장 간격으로 재며, 첫 장은 모델 로드 영향이 있어 제외합니다. 소스 파일은 수정하지 않습니다.

| 정책 | 고전 보간, Real-ESRGAN | PiSA-SR, OSDHuman |
|---|---|---|
| `official` | 32×16을 그대로 ×4 | `process_size=512`: 내부에서 256×128 → 1024×512로 올린 뒤 처리하고 128×64로 축소 |
| `native` | 32×16을 그대로 ×4 | `process_size=64`: 조건문을 통과해 선확대 없음. 네트워크 입력은 bicubic ×4의 128×64 |
| `fixed8x` | bicubic 8×(256×128) → ×4 → bicubic으로 128×64 | 같음 |

### OSDHuman의 작은 입력 처리 (커밋 e2723510, `run_inference.py`)

- `:70` `--upscale` 기본값은 1입니다.
- `:216-219` 입력 변이 `process_size // upscale`보다 작으면 짧은 변이 그 값이 되도록 확대하고 `resize_flag`를 켭니다.
- `:221-226` 다시 `upscale`배 하고 8의 배수로 맞춥니다.
- `:263-267` `resize_flag`이면 출력을 `upscale × 원본 크기`로 LANCZOS 축소합니다.

**따라서 기본값 그대로면 32×16 입력이 1024×512로 처리된 뒤 32×16으로 되돌아옵니다.** 최종 128×64를 내기 위해 `--upscale 4`만 기본값에서 바꿨습니다.

### PiSA-SR (커밋 9c76e12e, `test_pisasr.py`)

- `:38-41` 짧은 변이 `process_size // upscale`보다 작으면 확대하고, `:43` `upscale`배 합니다.
- `:69-70` `resize_flag`이면 원본 ×4로 bicubic 축소합니다.
- `:87` `--seed`는 인자로 받지만 이 스크립트에서 쓰이지 않습니다. `sr/runner.py`가 seed를 고정하고 `determinism.csv`로 재현성을 확인합니다.
- 조절 가능한 λ를 쓰기 위해 `--default`는 넘기지 않습니다.

## GPU 규칙

- `min_free_gb` 검사 후 시작하고 `set_per_process_memory_fraction`으로 상한을 둡니다.
- `run_all.sh`가 순서를 보장합니다. 조건별 처음 50장을 `--require-exclusive`로 하나씩 단독 실행한 **다음에** 학습 3개와 나머지 SR을 병렬로 시작합니다.
- `exclusive_gpu`는 시작 시점과 실행 중 메모리 관측으로 판정합니다. 소요 외삽에는 `true` 값만 씁니다.
- 학습이 OOM이면 SR을 멈추고 학습이 재시도합니다. SR OOM은 재시도하며, 마지막 재시도는 학습이 끝난 뒤 넉넉한 메모리로 합니다. 끝내 안 되면 이미지별로 `failures.csv`에 남깁니다.

## 가중치

| 항목 | 출처 | 비고 |
|---|---|---|
| ResNet-50 IMAGENET1K_V1 | torchvision 공식 URL | |
| RealESRGAN_x4plus | 공식 릴리스(기존 사본) | |
| pisa_sr.pkl | PiSA-SR README의 공식 Google Drive | |
| OSDHuman model, embedding | OSDHuman README의 공식 Google Drive | 저장소 루트에 LICENSE 없음. 연구 범위에서만 사용 |
| RAM swin large | recognize-anything Space(기존 사본) | |
| SD2.1-base | 기존 로컬 사본 | HF 공식 저장소 메타데이터와 대조. 결과는 `weights/sd21_verification.json` |

sha256은 `weights/SHA256.csv`, 외부 저장소 commit은 `runs/<run_id>/provenance.json`에 있습니다.

## 환경

conda env 4개를 `/workspace/miniforge3`에 새로 만듭니다(`env/setup_envs.sh`). 구성은 `env/*.yml`, 잠금은 `env/*.lock.txt`입니다. Pillow는 리샘플링과 디코딩 차이를 막기 위해 전 env 9.5.0으로 통일합니다.

| env | 용도 |
|---|---|
| `srreid-core` | 데이터, ReID, 평가, bicubic, lanczos |
| `srreid-realesrgan` | Real-ESRGAN |
| `srreid-pisasr` | PiSA-SR |
| `srreid-osdhuman` | OSDHuman |
