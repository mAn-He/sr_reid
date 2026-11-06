# 입력 크기 정책

규칙은 `docs/DESIGN.md` 3절입니다. 파일·라인 근거는 `docs/PIPELINE_<model>.md` 3절과 6절에 있습니다.

## 공식 입력 크기
| 모델 | H×W | 근거 |
|---|---|---|
| BoT (공식) | 256×128 | `third_party/reid-strong-baseline/configs/softmax_triplet_with_center.yml:12-13` |
| BoT (직접 구현) | 256×128 | `configs/pilot.yaml:25` |
| CLIP-ReID ViT, stride 16 | 256×128 | `third_party/CLIP-ReID/configs/person/vit_clipreid.yml:15-16`, stride는 `:7` |
| KPR | 384×128 | `third_party/keypoint_promptable_reidentification/configs/kpr/imagenet/kpr_market_train.yaml:4-5`, 평가 `kpr_market_test.yaml:6-7` |
| KPR+SOLIDER | 384×128 | `.../configs/kpr/solider/kpr_market_train.yaml:6-7`, 평가 `kpr_market_test.yaml:6-7` |

KPR+SOLIDER의 입력 설정은 KPR과 같습니다.

## 384×192, 512×256에서의 config 변경점
| 항목 | BoT | CLIP-ReID | KPR, KPR+SOLIDER |
|---|---|---|---|
| 입력 키 | 어댑터가 ×k 이미지를 그대로 사용 | `INPUT.SIZE_TRAIN`, `INPUT.SIZE_TEST` | `data.height`, `data.width` |
| 특징맵 | 24×12, 32×16 (layer4) | 패치 격자 24×12, 32×16 | stage 4 12×6, 16×8 |
| 위치 임베딩 | 없음 | 보간 필요 (129 → 289, 513 토큰, 모델 생성 때 1회) | 절대 위치 임베딩 없음, 상대 위치 표는 윈도우 7 고정 |
| Swin 윈도우 | 해당 없음 | 해당 없음 | 나누어떨어지지 않아 모든 stage에서 패딩 발생 (표 아래) |
| 랜덤 크롭·패딩 | 10px 고정 | 10px 고정 | `pad_size=10` 고정 |
| 입력 리사이즈 | 없음 | `T.Resize` no-op | `Resize` 동일 크기면 같은 배열 반환 |
| 마스크 | 해당 없음 | 해당 없음 | 저장 36×17×9 고정. 로더가 5채널 H×W로 bilinear 후 H/4, W/4로 축소 |
| 키포인트 | 해당 없음 | 해당 없음 | 픽셀 중심 규약 `(x+0.5)k−0.5`로 미리 스케일 |

Swin stage별 패딩 면적 (`docs/PIPELINE_kpr.md` 3.1):

| 입력 | stage 1 | stage 2 | stage 3 | stage 4 |
|---|---:|---:|---:|---:|
| 384×128 | 10.4% | 25.4% | 51.0% | 51.0% |
| 384×192 | 4.0% | 16.0% | 26.5% | 26.5% |
| 512×256 | 12.0% | 16.4% | 30.3% | 56.5% |

패딩 영역은 0으로 채워지고 이를 가리는 attention mask는 없습니다. 입력 크기마다 패딩 면적이 달라, 배율 효과와 섞일 수 있습니다.

## 입력 크기에 따라 바뀌어야 할 수 있는 증강 값 (목록만, 변경하지 않음)
| 값 | 현재 | 256×128 기준 H 대비 | 384×192 | 512×256 |
|---|---|---:|---:|---:|
| 패딩·크롭 이동폭 (BoT, CLIP-ReID) | 10px | 3.9% | 2.6% | 2.0% |
| 패딩·크롭 이동폭 (KPR, 공식 384×128 기준) | 10px | 2.6% (384) | 2.6% | 2.0% |
| CLIP 패치 크기 대비 | 16px | 6.3% | 4.2% | 3.1% |

Random Erasing, CoarseDropout, 뒤집기 확률은 비율 기준이라 크기를 따라갑니다. 모델 수용영역은 절대 px이라 입력 대비 비율이 줄어듭니다.
