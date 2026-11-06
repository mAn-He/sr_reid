# 대여 GPU 근거 메모 (2026-10-31 검색, 추정의 출처)

공개 자료는 모두 추정 근거이며 실측이 아니다. 단일 GPU ResNet·ViT 혼합 정밀도 비교는 찾지 못했다.

## H100 대 A100 연산 성능
- ViT, 2~4노드, bf16: H100이 약 2.1~2.2배. https://docs.nvidia.com/nemo-framework/user-guide/24.12/multimodalmodels/visionlanguagefoundation/vit/perf.html
- ResNet-50, 4 GPU: 약 1.54배. https://thundercompute.com/blog/nvidia-a100-vs-h100
- 단일 GPU 작은 transformer 학습 스텝 비교: https://handbook.eng.kempnerinstitute.harvard.edu/technical_blog/choosing_gpu_partition.html

## 시간당 가격 (단일 GPU, on-demand, USD)
2026-10-08 표: https://inventivehq.com/tools/developer/gpu-rental-cost-calculator
- RTX 4090 24GB: RunPod Community 0.34, Secure 0.74
- A40 48GB: Community 0.35, Secure 0.49
- RTX 6000 Ada 48GB: Community 0.74, Secure 0.84
- A100 80GB: PCIe 1.19~1.59, SXM 1.39~1.59
- H100 80GB: PCIe 1.99~3.29, SXM 2.69~6.88

제조사 관련 또는 집계 사이트가 섞여 있어 값이 서로 어긋난다. 대여 전에 실제 가격을 확인한다.

## 2026-11-06 재검색 (RunPod, on-demand, USD/h)
- RunPod 공식 가격 페이지(2026-07 기준 표기): L40S 48GB 0.79, A100 80GB PCIe 1.39 / SXM 1.59, H100 80GB 1.99부터. https://www.runpod.io/gpu-cloud/pricing
- 제3자 추적: L40S 0.86~0.99, H100 2.69~3.99. https://www.spheron.network/blog/runpod-h100-pricing-2026/ , https://computeprices.com/providers/runpod/gpus/a100-pcie-80gb , https://gpu-cost.pages.dev/provider/runpod/
- 비용 추정(`STAGE12_PLAN.md`)은 공식 페이지 값을 쓴다. 대여 직전에 다시 확인한다.
- 대여 서버 CPU 참고: 새 서버(EPYC 7282, cgroup 13.6코어)에서 KPR ×3 b48 학습이 CPU 스로틀링으로 [이전] 대비 −30%였다(`STAGE05_REPORT.md` 4.3절). CPU 할당을 확인하고 고른다.
