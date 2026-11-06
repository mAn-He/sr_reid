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
  - Stage 1: SR 기법 비교. ReID 학습 없이 train 표본(`configs/stage1_sample.csv`, 751명 × 3장, 2,250장)에서 SR 전후 비교 지표로 12종을 평가해 상위 2개를 고른다. 규칙은 `STAGE1_SELECTION_RULE.md`이고 결과 계산 전에 고정한다. test는 쓰지 않는다. 표본 SR 출력은 `runs/stage1_sample/`에 둔다.
  - 생성: 선정된 SR 2개 + bicubic만 train·test 전체를 ×3·×4로 만든다. bicubic은 기존 `data/bicubic_x{3,4}`를 재사용한다.
  - Stage 2: 모델 4종 × {bicubic, SR1, SR2} × {×3, ×4}, seed 42. 평가는 2×2로 (bicubic 학습 모델, SR 학습 모델) × (bicubic test, SR test)다.
- 기준선은 공개 가중치를 공식 입력으로 평가한 참고 행이다(학습하지 않음). mAP/R1: BoT 85.95/94.63, CLIP-ReID 89.66/95.43, KPR 89.59/95.87, KPR+SOLIDER 93.25/96.59 (`results/baseline_public_weights.csv`).

## 2. 확정된 결정
- **볼륨**: /workspace 350GB, `configs/volume_limit_gib` = 325. 큰 작업 전 `bash scripts/disk_guard.sh 20`.
- **WebP**: 무손실 method 2 (`stage05/sr_generate.py`의 `WEBP`). 저장 직후 픽셀 전수 비교.
- **SR 생성 범위 (2026-11-06 변경)**
  - 이전 F1(train ×4 전체 12종)은 취소했다. 시작한 적 없음, `data/`에 생성물 0장.
  - Stage 1: 표본 2,250장 × 12종 × {×2, ×3, ×4}만 `runs/stage1_sample/`에 만든다.
  - Stage 1 승인 뒤: 선정 2개 방법의 train·query·gallery ×3·×4. bicubic은 기존 데이터를 재사용한다.
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
- **SR 일괄 생성(선정 방법의 train·test 전체)은 사용자가 Stage 1 결과를 승인하기 전까지 시작하지 않는다.** Stage 1 표본 SR도 선정 규칙과 보고를 사용자가 승인한 뒤에 돌린다.
- test 이미지(query, bounding_box_test, gt_*)는 SR 생성만 허용하고 평가·열람하지 않는다. 예외는 공개 가중치 기준선·등가성 측정(이미 끝남)뿐이다. `data/locked`의 권한 변경·열람 금지. 가드는 `common/guard.py`(ALLOW_TEST=1일 때만 허용)다.
- 학습 레시피 변경 금지(입력 크기 제외): 배치, PK 샘플링, 학습률, 스케줄, epoch, 옵티마이저, 정밀도, 증강.
- 외부 모델·SR 저장소(`third_party/`) 소스 수정 금지. config, 래퍼, 서브클래스, 어댑터로만 바꾼다.
- 이미지 외부 전송 금지. 이미지는 열지 않고 수치로만 검사한다. 추론은 `HF_HUB_OFFLINE=1`.
- 다른 프로젝트 삭제 금지: `/workspace/Vla_check`, `/workspace/.venvs`, `/workspace/rs-sr-detect`, geo·osediff·sr env. `/workspace/weights` 공용 가중치(realesrgan, osediff/ram_swin_large, sd21_base, bert-base-uncased)는 sr-reid가 링크로 쓰므로 건드리지 않는다.
- 3e(학습 처리량 측정) 동안 다른 CPU·GPU 작업을 돌리지 않는다.
- `data/kpr/x2/bounding_box_train` 깨진 링크(→ 삭제된 `data/bicubic_x2`)는 그대로 둔다(기록만).

## 4. 서버 변경 이력
| 항목 | 이전 서버 (~2026-11-01) | 새 서버 (2026-11-06~) |
|---|---|---|
| GPU | RTX 4090 24GB | RTX 4090 24GB (같은 모델) |
| CPU | (기록 없음) | AMD EPYC 7282, cgroup CPU 13.6코어 |
| cgroup | v1 | v2 |
| 메모리 한도 | 57.7GiB | memory.max 116.4GiB (124,999,999,488 B) |
| 드라이버/CUDA | (기록 없음) | 595.91.07 / CUDA 13.2 |
| /workspace | 같은 네트워크 볼륨(MooseFS), 데이터 체크섬 37,793건 전수 일치 | 동일 |

마이그레이션 때 사라진 것(모두 /root): DreamSim 가중치(링크 깨짐, 11-06 재다운로드), `/root/srreid_cache`, `/root/srreid_raw`(자동 재생성), `/root/.cache/torch`(pyiqa DISTS, 11-06 `TORCH_HOME`으로 재다운로드), Claude Code 바이너리, `/root/.gitconfig`(저장소 로컬 설정으로 대체).

cgroup 기록 도구는 v1·v2를 자동 판별한다: `stage05/cgroup_log.py`, `stage05/proc_watch.sh`. 결합 스크립트는 `stage05/ram_join.py`(이전 서버 `logs/stage05/cgroup_mem.csv`, 새 서버 `logs/stage05/cgroup_mem_v2.csv`).

## 5. 주요 파일
- 결과 표: `results/stage05_*.csv`, 3a~3d 요약 `results/stage05_3abcd_summary.md`, 새 서버 3e `results/stage05_srv2_{throughput}.csv`
- 보고서: `STAGE0_REPORT.md`, `STAGE05_REPORT.md`, `STAGE05_PART_F_PLAN.md`, `STAGE12_PLAN.md`(초안)
- 실행: `stage05/run_train.py <phase...>`(완료 표시 `runs/stage05/done/`, 깨진 표시는 시작 때 `logs/stage05/script_bug_rows/corrupt_done/`로 격리), `stage05/sr_generate.py`, `stage05/run_eval.py`
- 모델 파이프라인 근거: `docs/PIPELINE_<model>.md`, 입력 크기 `docs/INPUT_SIZE_POLICY.md`
- 알려진 기록: `runs/stage05/done/d9_data_kpr_memmap.json`은 2026-11-01 01:47 쿼터 사건으로 0바이트였다. KPR memmap 등가성은 d10 copy_read 비교로 대신 확정했다.

## 6. 현재 진행 단계 (갱신: 2026-11-06, 설계 변경 반영)
- Stage 0, 파일럿, Stage 0.5 측정: 완료. 보고서는 `STAGE05_REPORT.md`, 3e 포함.
- 설계 변경(위 1절): Stage 1은 학습 없는 SR 비교다. 이전 F1은 취소했다(시작한 적 없음).
- Stage 1 준비
  - 완료: 표본 `configs/stage1_sample.csv`(`stage1/make_sample.py`, seed 42, 2,250장, 751명).
  - 완료: 선정 규칙 초안 `STAGE1_SELECTION_RULE.md` v0(사용자 승인 대기, 질문 4개).
  - 진행/완료 확인: 방법별 SR 배치 추론 측정 `stage05/sr_batch_bench.py` → `results/stage1_sr_batch_bench.csv`.
- **다음:** 사용자가 선정 규칙 v1, 속성 모델, ×2 가중치 정책, 배치 사용을 승인하면 표본 SR(`runs/stage1_sample/`)과 지표 계산을 시작한다.
- 문서: `STAGE12_PLAN.md`(새 구조), `STAGE05_PART_F_PLAN.md`(새 범위).
