# CLAUDE.md: SR→ReID 프로젝트 작업 기준

다음 세션은 이 파일만 읽고 이어간다. 단계가 끝날 때마다 맨 아래 "현재 진행 단계"를 갱신한다.

## 0. 시작 규칙
- **Claude Code는 반드시 `/workspace/sr-reid`에서 실행한다.** 다른 폴더(예: `/workspace/rs-sr-detect`)에서 시작했으면 바로 종료하고 다시 연다.
- 첫 확인: `pwd`가 `/workspace/sr-reid`이고 `locale`이 UTF-8(C.UTF-8 또는 en_US.UTF-8)이어야 한다. 아니면 멈추고 보고한다.
- 재시작이나 서버 이동 뒤에는 `source /workspace/setup_after_restart.sh`를 먼저 실행한다. Claude Code 재설치, `~/.claude → /workspace/.claude` 링크, locale, TORCH_HOME 등 환경 변수, 깨진 링크 복구, git 로컬 user 확인을 이 스크립트가 한다. 이어서 `tmux -u new -s srreid` → `cd /workspace/sr-reid` → `claude --continue`.
- 긴 GPU·CPU 작업은 `bash scripts/run_bg.sh <세션이름> <명령...>`으로 별도 tmux 세션에 띄운다. Claude Code가 끊겨도 계속 돈다. 로그는 `logs/bg/<이름>.log`, 종료 코드는 `logs/bg/<이름>.exit`. 실행 스크립트는 완료 표시(`runs/stage05/done/*.json`, 완성 WebP 파일)로 끝난 단계를 건너뛴다.
- 문서에 없는 결정이 필요하면 추측하지 말고 멈춰서 사용자에게 묻는다.
- git push는 사용자가 직접 한다. 커밋은 요청이 있을 때만 하고, 저장소 로컬 user.name/email(mAn-He)을 쓴다.

## 1. 연구 설계 (요약, 원문은 `docs/DESIGN.md`)
- Market-1501 원본 128×64 기준 배율 k. ×3 = 384×192, ×4 = 512×256. SR 출력 크기가 그대로 네트워크 입력이고 모델 안에서 다시 리사이즈하지 않는다.
- 조건: C_k(train·test 모두 bicubic ×k), S_k(train·test 모두 SR ×k). 평가 전용: S_k 모델 + bicubic test, C_k 모델 + SR test.
- 모델 4종: BoT, CLIP-ReID, KPR, KPR+SOLIDER. **seed 42 하나만 쓴다. seed를 추가하지 않는다.**
- **전체 구조 (2026-11-06 설계 변경, 이전 Stage 1 설계인 검증 분할 651/100, BoT·CLIP-ReID 12종 학습, seed 추가, 두 모델 순위 비교, 병렬화는 전부 취소)**
  - Stage 1 (2026-11-06 2차 교체): ReID 학습 없이 SR 12종을 비교한다. 정답(HR) 없이 모든 지표를 "실제 SR vs 원본 128×64"로만 계산한다. **순위·추천·선정은 하지 않는다. 결과와 시각화만 만들고 선정은 사용자가 한다.**
    - 표본 `configs/stage1_sample.csv`(751명, 2,250장, seed 42), 배율 ×3·×4만(×2 없음), bicubic·lanczos도 같은 지표로 계산.
    - 지표: DINOv3·PersonViT(불가 시 PLIP) 임베딩 코사인과 분포 거리(Fréchet, KID), A-FINE(원본 기준), FFT 4종(`stage1/fft_color.py`), ΔE2000(참고).
    - 해석 기준선(약한 변형, 같은 사람 다른 카메라)은 그림에만 겹쳐 그린다. 집계: 이미지 → 신원 평균 → 751명 평균, 신원 단위 부트스트랩 95% CI(10,000회, seed 42).
    - 취소된 것: 선정 규칙(`STAGE1_SELECTION_RULE.md`는 "취소됨" 표시로 보존), 가중 평균 순위, 파레토 선정, 탈락 단계, 스모크 파일럿, 보정 실험, ReID 모델 기반 지표, 속성 모델, 자기 찾기 비율, 사람 영역 패치·어텐션 비교, ×2.
    - 단계: 0단계(모델 로드·재현성·×3 방식 확인, 보고 후 멈춤) → 1단계 SR 생성 → 2단계 지표 → 3단계 시각화. 산출물 `results/stage1/`, `results/stage1/figures/`, `STAGE1_REPORT.html`, `STAGE1_REPORT.md`.
  - **Stage 2 대상 확정(2026-11-09 사용자 결정): SR 3개 = swinir, pisa_sr λ0.0(`pisa_l0`, 폴더 이름 `pisa_sr_sem0.0`), osdhuman + bicubic 기준.**
  - 생성: SR 3개의 train 전체(12,936장)와 test(query 3,368 + gallery 19,732)를 ×3·×4로 만든다. bicubic train은 기존 `data/bicubic_x{3,4}`를 재사용하고 test ×3·×4는 새로 만든다.
  - Stage 2: 모델 4종 × {bicubic, swinir, pisa_l0, osdhuman} × {×3, ×4} = 학습 32회, seed 42. 평가는 SR 조건마다 2×2로 (bicubic 학습 모델, SR 학습 모델) × (bicubic test, SR test), mAP·Rank-1, 신원 단위 부트스트랩 95% CI, bicubic 대비 useful / harmful / inconclusive.
- 기준선은 공개 가중치를 공식 입력으로 평가한 참고 행이다(학습하지 않음). mAP/R1: BoT 85.95/94.63, CLIP-ReID 89.66/95.43, KPR 89.59/95.87, KPR+SOLIDER 93.25/96.59 (`results/baseline_public_weights.csv`).

## 2. 확정된 결정
- **볼륨**: /workspace 350GB, `configs/volume_limit_gib` = 325. 큰 작업 전 `bash scripts/disk_guard.sh 20`.
- **WebP**: 무손실 method 2 (`stage05/sr_generate.py`의 `WEBP`). 저장 직후 픽셀 전수 비교.
- **SR 생성 범위 (2026-11-06 변경)**
  - 이전 F1(train ×4 전체 12종)은 취소했다. 시작한 적 없음, `data/`에 생성물 0장.
  - Stage 1: 표본 2,250장 × 12종 × {×3, ×4}만 `runs/stage1_sample/<method>/x<k>/<원본파일명>.webp`에 만든다(`stage1/sr_gen.py`, 실행 `stage1/run_gen.sh`). ×3: swinir·hat·mambairv2는 공식 ×3 가중치, **realesrgan·osdhuman·pisa_sr은 ×4 출력을 PIL BICUBIC으로 384×192 축소(세 방법 같은 커널, 2026-11-07 결정)**.
  - Stage 1 배치 정책 (가): realesrgan b4(배치 1과 비트 일치), 나머지 b1. 이미지마다 파일명 기반 고정 seed(sha256("42:<파일명>")).
  - Stage 1 결정적 설정(2026-11-07 결정): osdhuman·pisa_sr·mambairv2·hat은 `--det --warmup`(cudnn.benchmark=False, cudnn.deterministic=True, SDPA math만, 예열 1회) + 파일명 seed. swinir·realesrgan은 공식 설정. 공식 설정과의 차이는 `results/stage1/det_vs_official_*.csv`.
- **A100 가속 방침 (2026-11-08 결정)**: 결과 픽셀이 바뀌는 방법은 쓰지 않는다.
  - SR 생성: 배치 정책 (가) 유지(배치를 키우지 않음). GPU당 여러 프로세스 동시 실행, 프로세스 = 방법 하나(pisa_sr는 λ별). 한 방법의 ×3·×4 생성과 그 방법의 지표는 같은 GPU. 측정 결과 GPU당 4개(두 GPU 합계 7.09장/s, 1개 5.70장/s, `results/stage1/a100_concurrency_bench.csv`), 동시 실행 출력은 단독 실행과 비트 일치(`a100_concurrency_bitcheck.csv`). 프로세스당 OMP/MKL 4(8개 × 4 = 32 ≤ cgroup 54.4).
  - 지표: DINOv3·PersonViT·A-FINE은 큰 배치 허용(임베딩 256, A-FINE 128). FFT·ΔE는 CPU 코어 수만큼 병렬(48 프로세스).
  - 동시 실행 중 시간·처리량은 "병렬 실행 중 측정"으로 표시한다. 실행 `stage1/run_a100.sh`(main, check20, post), 사전 확인 `stage1/run_a100_check.sh`(repro, bench).
  - **Stage 2: 학습 배치 크기는 레시피의 일부라 절대 바꾸지 않는다. 동시 실행만 허용한다.**
- **Stage 2 학습 설정 (2026-11-09 결정)**
  - 확정 설정 유지: memmap, BoT worker_norm, KPR copy_read + float32 마스크(원본 키포인트 기반 스케일 마스크를 모든 조건에 동일하게), BIPO는 학습에서만, 핀 메모리 켬.
  - 각 모델 공식 레시피 기본 설정. 등가성 검증 때의 결정론 모드는 끄고 공식 코드의 cudnn.benchmark 등 기본값 그대로. 실제 설정을 학습마다 기록.
  - 배치 크기, 정밀도(AMP·TF32 행렬곱), torch.compile 등 수치가 바뀌는 변경 금지. gradient checkpointing은 켜고 끄는 것이 비트 단위로 같음을 확인했으므로 A100 80GB에서 메모리가 허용하면 끈다(모델·배율별 기록).
  - 가속: KPR·KPR+SOLIDER ×4를 먼저 GPU 2장에 배치하고 남는 자리를 BoT·CLIP-ReID로(GPU당 2~3개까지). 200 iteration 시험으로 GPU별 조합 처리량 비교 후 선택. VRAM·CPU 여유 기반 자동 대기열, 완료분 건너뜀. 학습 전 모든 조건 memmap 캐시, RAM 여유 시 /dev/shm. 스레드·DataLoader worker는 cgroup CPU 한도 안에서 배분. 평가는 다음 학습과 겹쳐 진행. 동시 실행 중 시간은 "병렬 실행 중 측정".
- **Stage 1 평가 설정 (2026-11-07 결정)**
  - DINOv3: timm 변환본 ViT-L(`vit_large_patch16_dinov3.lvd1689m`), 256×128 PIL BICUBIC, CLS, RoPE 주기 bf16 보정. DINOv3 License 사용 표기 의무를 보고서에 기록.
  - PersonViT: ViT-B LUPerson 자기지도(checkpoint0260), mean/std 0.5/0.5. 한계: 사전학습 정규화 확인 불가, 체크포인트를 ReID 정확도로 고름.
  - A-FINE: pyiqa 사본. 보고 지표 afine_all_scale(0~100)·충실도·자연스러움, 원값 afine_all은 중앙값만. 모두 낮을수록 좋음. bicubic 행은 "기준 이미지와 동일, 비교 불가".
  - 분포 거리(Fréchet, KID): L2 정규화 임베딩, 고윳값 분해. 기준선: 블러 σ=0.5, JPEG 90, 같은 사람 다른 카메라.
  - env: 새 env를 만들지 않는다. scikit-learn 1.7.2를 srreid-quality에 설치(충돌 없음). 다른 패키지도 기존 env에 설치하고 충돌 시 멈추고 보고. srreid-mambair의 python 링크(→ srreid-realesrgan)는 그대로 둔다.
  - HTML: 그림은 축소 JPEG(품질 85) 내장, 원본 PNG는 `results/stage1/figures/`(커밋 안 함). 최악 10장은 별도 HTML(배율별).
  - **Stage 2 생성 설정(2026-11-09)**: Stage 1과 같은 설정(swinir ×3 공식 ×3 가중치, pisa_sr·osdhuman ×3 = ×4 후 PIL BICUBIC 축소, osdhuman·pisa_sr `--det --warmup` + 파일명 seed, swinir 공식 설정, b1). GPU 작업(swinir ×3, swinir ×4, pisa ×4, osdhuman ×4)을 이미지 목록 조각(shard)으로 나눠 GPU 2장 × 4 프로세스 공유 대기열로 돌린다(pisa·osdhuman은 더 잘게). ×3 축소·bicubic test는 CPU 작업으로 겹쳐 돈다. 조각마다 20장을 단독 실행과 비트 비교, Stage 1 표본 2,250장은 Stage 1 출력과 비트 비교. 실행 `stage2/run_gen.sh`.
  - 저장: train은 `data/<condition>/x<k>/bounding_box_train/<이름>.webp`. test는 실제 파일을 잠긴 영역 `data/locked/sr/<condition>/x<k>/<split>/`에 두고 `data/<condition>/x<k>/<split>`는 그곳을 가리키는 링크(재잠금 때 chmod 000이 test 출력까지 덮음, 가드도 경로의 query·bounding_box_test를 막음). test 로그·manifest도 `data/locked/sr/_logs/`.
  - 스모크 출력 `runs/stage05/sr_smoke/`(train 100장 × 12종)는 지우지 않는다.
  - 일괄 생성은 완성 WebP를 즉시 /workspace(`data/<method>/x<k>/<split>/`)에 쓰고, 재시작하면 남은 파일부터 이어서 생성한다(`sr_generate.py`가 완성 파일을 건너뜀).
- **memmap**: 4개 모델 모두 채택 확정(`results/stage05_3abcd_summary.md`).
  - BoT: memmap + worker_norm(정규화·erasing을 CPU 워커에서).
  - CLIP-ReID: memmap(+worker_norm 동일).
  - KPR·KPR+SOLIDER: memmap + mask_f32. 기준 로더는 copy_read다.
- **KPR 기준 로더 = copy_read**: 공식 로더 `tools.py:104`의 `lru_cache`된 `read_image` 배열을 BIPO가 제자리 수정한다(SR 입력 크기에서 albumentations resize가 같은 객체를 반환). 이 오염을 막으려고 `read_image` 반환값만 `.copy()`한다. 공식 로더 대비 변경점은 이것뿐이며 `stage05/backends.py:304-307`에 있다.
- **KPR 마스크 float32**(u8 기각). **BIPO는 학습에서만 켠다**(평가 변환에는 없음).
- **핀 메모리 켬. 학습 프로세스 스레드 제한**: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`(d13, 새 서버 KPR ×3 +9~10%, `STAGE05_REPORT.md` 4.3절). 값 4는 d13에서 4와 8의 차이가 잡음 수준이라 KPR 최고값을 골랐다.
- **test SR 생성 절차**: 잠금 해제와 재잠금은 사용자가 직접 한다. 그 사이 Claude는 생성 스크립트만 실행하고 출력도 잠긴 영역에 저장한다. 이미지와 지표는 열람하지 않는다.
- **RAM 판정**: 피크를 절대값(PSS 합계, cgroup memory.current)으로 기록하고 57.7GiB·116.4GiB 기준을 모두 표기한다. 판정은 보수적으로 57.7GiB 기준이다(`run_train.py` limit_gib=48 유지). KPR용 대여 서버가 정해지면 그 서버 기준으로 다시 판정한다.
- **서버 변경 원칙**: VRAM 결과는 유지한다. 처리량·학습 시간·RAM은 새 서버에서 다시 재서 이전 값과 나란히 기록한다. 새 서버 결과를 이전 서버 로그와 비트 단위로 비교하지 않는다. 모든 시간·처리량 수치에 측정 서버를 표시한다.
- **평가기**: KPR 공식 평가는 RAM 피크 약 128GiB(코드 계산 127.8GiB, `docs/PIPELINE_kpr.md:41`)라 경량 스트리밍 평가기를 쓴다(공식과 AP·CMC 일치 검증됨, `results/stage05_eval_equiv.csv`).
- **캐시 위치**
  - 재생성 가능한 임시 캐시(memmap `SRREID_CACHE=/root/srreid_cache`, eqdump, SR raw `SRREID_RAW=/root/srreid_raw`)는 /root 로컬 NVMe에 둔다. 실행 스크립트가 없으면 다시 만든다.
  - 가중치·env·HF/torch 캐시는 /workspace에 둔다: `TORCH_HOME=/workspace/weights/torch`, `HF_HOME=/workspace/.cache/huggingface`, DreamSim은 `weights/stage05/dreamsim`(실제 디렉터리). /root를 가리키는 가중치 링크는 만들지 않는다.

## 3. 금지 사항
- (해제됨 2026-11-09) SR 일괄 생성은 Stage 1 선정 뒤 시작한다. 선정 완료: swinir, pisa_sr λ0.0, osdhuman.
- test 이미지(query, bounding_box_test, gt_*)는 SR 생성만 허용하고 평가·열람하지 않는다. 예외는 공개 가중치 기준선·등가성 측정(이미 끝남)뿐이다. `data/locked`의 권한 변경·열람 금지. 가드는 `common/guard.py`(ALLOW_TEST=1일 때만 허용)다.
- 학습 레시피 변경 금지(입력 크기 제외): 배치, PK 샘플링, 학습률, 스케줄, epoch, 옵티마이저, 정밀도, 증강.
- 외부 모델·SR 저장소(`third_party/`) 소스 수정 금지. config, 래퍼, 서브클래스, 어댑터로만 바꾼다.
- 이미지 외부 전송 금지. 이미지는 열지 않고 수치로만 검사한다. 추론은 `HF_HUB_OFFLINE=1`.
- 다른 프로젝트 삭제 금지: `/workspace/Vla_check`, `/workspace/.venvs`, `/workspace/rs-sr-detect`, geo·osediff·sr env. `/workspace/weights` 공용 가중치(realesrgan, osediff/ram_swin_large, sd21_base, bert-base-uncased)는 sr-reid가 링크로 쓰므로 건드리지 않는다.
- 3e(학습 처리량 측정) 동안 다른 CPU·GPU 작업을 돌리지 않는다.
- `data/kpr/x2/bounding_box_train` 깨진 링크(→ 삭제된 `data/bicubic_x2`)는 그대로 둔다(기록만).

## 4. 서버 변경 이력
| 항목 | 이전 서버 (~2026-11-01) | 새 서버 (2026-11-06~11-07) | A100 서버 (2026-11-08~) |
|---|---|---|---|
| GPU | RTX 4090 24GB | RTX 4090 24GB (같은 모델) | A100 SXM4 80GB × 2 (Ampere sm_80) |
| CPU | (기록 없음) | AMD EPYC 7282, cgroup CPU 13.6코어 | AMD EPYC 7713, cgroup CPU 54.4코어 (cpu.max 5440000/100000), nproc 255 |
| cgroup | v1 | v2 | v2 |
| 메모리 한도 | 57.7GiB | memory.max 116.4GiB (124,999,999,488 B) | memory.max 233.8GiB (250,999,996,416 B) |
| 드라이버/CUDA | (기록 없음) | 595.91.07 / CUDA 13.2 | 595.91.07 / CUDA 13.2 |
| /workspace | 같은 네트워크 볼륨(MooseFS), 데이터 체크섬 37,793건 전수 일치 | 동일 | 동일 볼륨(EUR-IS-1), / 오버레이 80GB |

마이그레이션 때 사라진 것(모두 /root): DreamSim 가중치(링크 깨짐, 11-06 재다운로드), `/root/srreid_cache`, `/root/srreid_raw`(자동 재생성), `/root/.cache/torch`(pyiqa DISTS, 11-06 `TORCH_HOME`으로 재다운로드), Claude Code 바이너리, `/root/.gitconfig`(저장소 로컬 설정으로 대체).

A100 이동(2026-11-08, 4090 pod 크레딧 소진): 8개 env(core, quality, clipreid, kpr, realesrgan, mambair, osdhuman, pisasr) 모두 A100에서 import·GPU 연산 통과. SR 6종 3장 추론(mambairv2 CUDA 확장 포함), BoT·CLIP-ReID·KPR·KPR+SOLIDER 학습 5스텝 통과(`logs/a100diag/`). KPR 계열은 프로세스 종료 뒤 DataLoader 작업자가 남아 GPU 메모리를 잡을 수 있으니 연속 실행 시 프로세스 그룹째 정리한다. setup_after_restart.sh는 이번에 빠뜨린 것 없음. 비트 재현성은 GPU 종류 안에서만 보장되므로 Stage 1은 A100에서 다시 하고 4090 결과는 `*_4090`에 보관.

cgroup 기록 도구는 v1·v2를 자동 판별한다: `stage05/cgroup_log.py`, `stage05/proc_watch.sh`. 결합 스크립트는 `stage05/ram_join.py`(이전 서버 `logs/stage05/cgroup_mem.csv`, 새 서버 `logs/stage05/cgroup_mem_v2.csv`).

## 5. 주요 파일
- 결과 표: `results/stage05_*.csv`, 3a~3d 요약 `results/stage05_3abcd_summary.md`, 새 서버 3e `results/stage05_srv2_{throughput}.csv`
- 보고서: `STAGE0_REPORT.md`, `STAGE05_REPORT.md`, `STAGE05_PART_F_PLAN.md`, `STAGE12_PLAN.md`(초안)
- 실행: `stage05/run_train.py <phase...>`(완료 표시 `runs/stage05/done/`, 깨진 표시는 시작 때 `logs/stage05/script_bug_rows/corrupt_done/`로 격리), `stage05/sr_generate.py`, `stage05/run_eval.py`
- 모델 파이프라인 근거: `docs/PIPELINE_<model>.md`, 입력 크기 `docs/INPUT_SIZE_POLICY.md`
- 알려진 기록: `runs/stage05/done/d9_data_kpr_memmap.json`은 2026-11-01 01:47 쿼터 사건으로 0바이트였다. KPR memmap 등가성은 d10 copy_read 비교로 대신 확정했다.

## 6. 현재 진행 단계 (갱신: 2026-11-09)
- Stage 0, 파일럿, Stage 0.5 측정: 완료. 보고서는 `STAGE05_REPORT.md`, 3e 포함.
- SR 배치 추론 측정 완료: `results/stage1_sr_batch_bench.csv`. 배치 1과 비트 일치는 realesrgan b≤4뿐. mambairv2·osdhuman·pisa_sr은 실행 안 처리 순서(RNG)에 따라 출력이 달라진다.
- Stage 1 0단계 완료(`e553f53`), 사용자 결정 2026-11-07 반영.
- Stage 1 1~3단계 4090 실행(2026-11-07): 1·2단계 완료(`db4e188`, `3ce6209`), 3단계 그림 도중 pod 중단. 결과는 `runs/stage1_*_4090/`, `results/stage1_4090/`에 보관(A100 결과와 섞지 않음).
- **Stage 1 A100 재실행 완료 (2026-11-08)**: 1단계 `95093ef`(48집합, 실패 0), 2단계 `14e1d5b`, 3단계 보고서 `STAGE1_REPORT.html`·`.md`(최악 10장 `STAGE1_REPORT_worst_x{3,4}.html`은 100MB 넘어 gitignore, 그림 PNG `results/stage1/figures/`도 커밋 안 함). 동시 실행 출력 = 단독 재생성 13설정 20/20 비트 일치.
- **Stage 2 진행 중 (2026-11-09)**: SR 3개(swinir, pisa_sr λ0.0, osdhuman) 선정. 순서: train 생성 → (test 생성: 사용자 잠금 해제 대기) → 학습 32회(train 끝나면 시작) → 평가(잠금 해제 요청).
- 문서: `STAGE12_PLAN.md`, `STAGE05_PART_F_PLAN.md`.
