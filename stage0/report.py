"""STAGE0_REPORT.md 작성. 측정 CSV만 읽는다. 판정 규칙은 측정 전 고정한 내용이다."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
BACKENDS = ["bot", "clipreid", "kpr", "kpr_solider"]
NAMES = {"bot": "BoT", "clipreid": "CLIP-ReID", "kpr": "KPR", "kpr_solider": "KPR+SOLIDER"}
SCALES = [2, 3]
TOTAL_GB = 24.0
N_EVAL = 3368 + 19732
N_TRAIN = 12936


def read(path):
    return list(csv.DictReader(open(path, newline=""))) if path.exists() else []


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pick(rows, backend, scale, setting, batch=None):
    out = [r for r in rows if r["backend"] == backend and r["scale"] == str(scale) and r["setting"] == setting
           and (batch is None or r["batch"] == str(batch))]
    return out[-1] if out else None


def linear_vram(rows, backend, scale):
    """맞은 배치들의 allocated로 선형 적합해 배치 64를 외삽한다. 점이 하나면 비례로 계산한다."""
    pts = sorted({(int(r["batch"]), num(r["max_memory_reserved_gb"])) for r in rows if r["backend"] == backend and r["scale"] == str(scale)
                  and r["setting"] in ("B", "B_trial") and r["status"] == "ok" and r["ckpt"] == "False" and num(r["max_memory_reserved_gb"])})
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1] / pts[0][0] * 64
    (b0, m0), (b1, m1) = pts[0], pts[-1]
    return m1 + (64 - b1) * (m1 - m0) / (b1 - b0)


def verdict(rows, backend, scale):
    a, c = pick(rows, backend, scale, "A"), pick(rows, backend, scale, "C")
    if not a:
        return "측정 안 됨"
    if a["status"] == "ok":
        v = "4090 가능"
    elif a["status"] != "oom":
        return f"측정 실패({a['status']})"
    elif c and c["status"] == "ok":
        s = num(c["slowdown_pct"])
        v = f"4090 가능(checkpointing, 속도 −{s:.0f}%)" if s is not None else "4090 가능(checkpointing, 속도 저하율 미측정)"
    else:
        need = linear_vram(rows, backend, scale)
        v = f"4090 불가, 필요 VRAM 추정치 {need:.1f}GB (최대 배치 측정값의 선형 외삽)" if need else "4090 불가, VRAM 추정 불가(맞는 배치 없음)"
    used = c if a["status"] == "oom" and c and c["status"] == "ok" else a
    if used["bottleneck_flag"] == "True":
        v += " · CPU/데이터 로딩 병목 의심"
    return v


def speed_row(rows, backend, scale):
    a, c = pick(rows, backend, scale, "A"), pick(rows, backend, scale, "C")
    if a and a["status"] == "ok":
        return a
    return c if c and c["status"] == "ok" else None


def train_hours(rows, ev, backend, scale):
    """(학습 시간, 평가 시간, 근거). 값이 없으면 None."""
    sp = speed_row(rows, backend, scale)
    if not sp:
        return None, None, "레시피 배치의 속도 측정값 없음"
    its, iters, epochs = num(sp["it_s"]), num(sp["iters_per_epoch"]), num(sp["epochs"])
    if not (its and iters and epochs):
        return None, None, "속도 또는 epoch 값 없음"
    ext = [r for r in ev if r["backend"] == backend and r["task"] == "extract" and r["scale"] == str(scale) and r["status"] == "ok"]
    ext128 = next((r for r in ext if r["batch"] == "128"), None)
    img_s = num(ext128["img_s"]) if ext128 else None
    seconds = iters * epochs / its
    basis = f"{int(iters)}×{int(epochs)}÷{its:.2f}"
    if backend == "clipreid":
        s1 = pick(rows, backend, scale, "A_stage1")
        if not (s1 and s1["status"] == "ok" and img_s):
            return None, None, "CLIP-ReID stage1 또는 특징 추출 측정값 없음"
        seconds += num(s1["iters_per_epoch"]) * num(s1["epochs"]) / num(s1["it_s"]) + N_TRAIN / img_s
        basis += f" + stage1 {int(num(s1['iters_per_epoch']))}×{int(num(s1['epochs']))}÷{num(s1['it_s']):.2f} + 특징 캐시 {N_TRAIN}÷{img_s:.0f}"
    dist = [num(r["seconds"]) for r in ev if r["backend"] == backend and r["task"] == "dist" and r["status"] == "ok" and r["block"] == "full" and r["device"] == "gpu"]
    if not (img_s and dist):
        return seconds / 3600, None, basis + "; 평가 측정값 없음"
    return seconds / 3600, (N_EVAL / img_s + dist[0]) / 3600, basis


def fmt(v, d=2):
    n = num(v)
    return f"{n:.{d}f}" if n is not None else "-"


def main():
    rows, ev = read(R / "stage0_probe.csv"), read(R / "stage0_eval_mem.csv")
    fails = read(R / "stage0_failures.csv")
    scale = json.loads((R / "stage0_scale_check.json").read_text()) if (R / "stage0_scale_check.json").exists() else None
    prov = json.loads((R / "stage0_provenance.json").read_text()) if (R / "stage0_provenance.json").exists() else {}
    L = ["# Stage 0: RTX 4090 실행 가능 범위 실측", "", "학습이 아니라 측정입니다. test 이미지는 읽지 않았습니다. 크기 표기는 H×W입니다.", ""]
    L += ["## 1. 모델·배율별 판정", "", "| 모델 | 입력 | 판정 | 레시피 배치 | AMP | 옵티마이저 | allocated GB | reserved GB | nvidia-smi 피크 GB | it/s | img/s | GPU 사용률 % |",
          "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for b in BACKENDS:
        for s in SCALES:
            a = pick(rows, b, s, "A") or {}
            L.append(f"| {NAMES[b]} | {128 * s}×{64 * s} | {verdict(rows, b, s)} | {a.get('recipe_batch', '-')} | {a.get('amp', '-')} | {a.get('optimizer', '-')} | "
                     f"{fmt(a.get('max_memory_allocated_gb'))} | {fmt(a.get('max_memory_reserved_gb'))} | {fmt(a.get('nvidia_smi_peak_gb'))} | {fmt(a.get('it_s'))} | {fmt(a.get('img_s'), 1)} | {fmt(a.get('gpu_util_mean'), 0)} |")
    L += ["", "CLIP-ReID는 두 스테이지 중 이미지 인코더를 학습하는 stage 2의 값입니다. stage 1은 `results/stage0_probe.csv`의 `A_stage1` 행에 있습니다.", ""]
    L += ["## 2. KPR·KPR+SOLIDER 최대 배치", "", "정보용이며 채택안이 아닙니다. K=4의 배수만 탐색했습니다.", "",
          "| 모델 | 입력 | 최대 배치 | allocated GB | reserved GB | it/s | img/s | 비고 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for b in ("kpr", "kpr_solider"):
        for s in SCALES:
            cands = [r for r in rows if r["backend"] == b and r["scale"] == str(s) and r["setting"] == "B" and r["status"] == "ok"]
            if cands:
                r = max(cands, key=lambda x: int(x["batch"]))
                L.append(f"| {NAMES[b]} | {128 * s}×{64 * s} | {r['batch']} | {fmt(r['max_memory_allocated_gb'])} | {fmt(r['max_memory_reserved_gb'])} | {fmt(r['it_s'])} | {fmt(r['img_s'], 1)} | {r['note']} |")
            else:
                L.append(f"| {NAMES[b]} | {128 * s}×{64 * s} | 없음 | - | - | - | - | 측정 실패 또는 맞는 배치 없음 |")
    L += ["", "## 3. 학습 1회 예상 시간과 Stage 2 총 GPU 시간", "", "모두 외삽입니다. 평가 시간은 특징 추출(배치 128)과 거리 계산이며 순위·mAP 계산은 빠져 있습니다.", "",
          "| 모델 | 입력 | 학습 h | 평가 h | 근거 |", "|---|---|---:|---:|---|"]
    total, missing = 0.0, []
    for b in BACKENDS:
        for s in SCALES:
            t, e, basis = train_hours(rows, ev, b, s)
            L.append(f"| {NAMES[b]} | {128 * s}×{64 * s} | {fmt(t)} | {fmt(e)} | {basis} |")
            if t is None or e is None:
                missing.append(f"{NAMES[b]} ×{s}")
            else:
                total += (t + e) * 9
    L += ["", f"**Stage 2 전체(칸당 {{bicubic + SR 2종}} × seed 3 = 9회, 총 72회): 측정된 칸 합계 {total:.1f} GPU 시간 ({total / 24:.1f}일)**"
          + (f". 값이 없어 제외한 칸: {', '.join(missing)}." if missing else "."),
          "한 번에 한 작업만 돌린다고 가정했으며 SR 생성 시간은 포함하지 않았습니다.", ""]
    L += ["### 평가 메모리 요약", "", "상세는 `results/stage0_eval_mem.csv`입니다.", "",
          "| 모델 | 항목 | 장치 | 블록/배치 | allocated GB | RSS GB | 초 | img/s | 전체 대비 최대 절대 오차 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in ev:
        L.append(f"| {NAMES.get(r['backend'], r['backend'])} | {r['task']}{'×' + r['scale'] if r['scale'] else ''} | {r['device']} | {r['block'] or r['batch']} | {fmt(r['max_memory_allocated_gb'])} | "
                 f"{fmt(r['max_rss_gb'])} | {fmt(r['seconds'], 1)} | {fmt(r['img_s'], 1)} | {r['max_abs_error_vs_full'] or '-'} |")
    L += ["", "## 4. 키포인트·마스크 스케일 검증", ""]
    if scale:
        L.append(f"전체 통과 여부: **{scale['passed']}** (`results/stage0_scale_check.json`). 원본 {scale['n_samples']}장 기준입니다.")
        L += ["", "| 배율 | 공식 대비 최대 오차 | x·k와의 최대 차이 | 이론값 (k−1)/2 | 마스크 무게중심 최대 차이(px) | 저장 마스크 형태 |", "|---:|---:|---:|---:|---:|---|"]
        for k in SCALES:
            c = scale[f"x{k}"]
            L.append(f"| ×{k} | {c['keypoints']['max_err_vs_pixel_center_formula']:.2e} | {c['keypoints']['max_gap_vs_x_times_k']:.3f} | {c['keypoints']['theoretical_gap_(k-1)/2']} | "
                     f"{c['masks']['max_centroid_gap_vs_x_times_k_px']:.3f} | {', '.join(c['masks']['stored_shapes'])} |")
        L += ["", "키포인트는 픽셀 중심 규약이라 요청하신 `(x+0.5)·k−0.5`를 썼고, `x·k` 비교는 그 차이가 이론값과 같다는 확인입니다. 마스크는 17×9 고정 해상도라 스케일하지 않았습니다. 근거는 README입니다.", ""]
    else:
        L += ["검증 결과가 없습니다.", ""]
    L += ["## 5. 확인하지 못한 항목과 그 이유", ""]
    notes = ["학습 시간은 50스텝 측정의 외삽입니다. 전체 학습을 돌리지 않았고, 검증 주기와 체크포인트 저장 시간은 넣지 않았습니다.",
             "KPR+SOLIDER 레시피는 처음 20 epoch 동안 백본을 고정합니다. 측정은 백본을 푼 상태라 그 구간은 메모리와 시간을 보수적으로 잡은 값입니다.",
             "KPR 공식 입력은 384×128입니다. 요청하신 매트릭스(256×128, 384×192)로 쟀으며 공식 입력은 재지 않았습니다.",
             "CLIP-ReID는 README의 torch 1.8 + CUDA 10.2가 RTX 4090을 지원하지 않아 torch 1.13.0 cu117로 쟀습니다. BoT는 파일럿과 같은 torch 2.1.2입니다.",
             "데이터 로더는 네트워크 볼륨(MooseFS)에서 읽습니다. 로컬 디스크로 옮기면 병목 표시와 it/s가 달라질 수 있습니다.",
             "평가 메모리의 거리 계산은 무작위 특징으로 쟀습니다. 실제 특징이어도 메모리는 같지만 시간은 조금 다를 수 있습니다. KPR 추출은 train 이미지로 쟀습니다."]
    miss = prov.get("missing") or []
    notes += [f"가중치 확보 실패: {m}" for m in miss]
    notes += [f"측정 실패 기록: {f['key']}: {f['reason'][:300]}" for f in fails]
    if not rows:
        notes.insert(0, "측정 결과 CSV가 없습니다.")
    L += [f"- {n}" for n in notes]
    if prov:
        L += ["", "## 출처", "", "```", json.dumps({k: prov.get(k) for k in ("commits", "licenses", "solider_license", "kpr_annotation_zip")}, indent=1, ensure_ascii=False), "```",
              "", "가중치 sha256은 `weights/stage0/SHA256.csv`, 데이터 sha256은 `data/bicubic_x{2,3}_manifest.csv`에 있습니다."]
    (ROOT / "STAGE0_REPORT.md").write_text("\n".join(L) + "\n")
    print("STAGE0_REPORT.md 작성", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
