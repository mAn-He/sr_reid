"""Stage 1 보고서 생성: STAGE1_REPORT.md, STAGE1_REPORT.html(그림 JPEG 품질 85 내장), 지표별 최악 10장은 STAGE1_REPORT_worst_x3.html, _x4.html로 나눔.
순위·추천·선정은 쓰지 않는다. 방법은 항상 고정 순서(METHODS)로 나열한다. srreid-quality env.
그림은 Claude가 열어 보지 않는다. JPEG 변환은 PIL로 수치 처리만 한다."""
import base64
import csv
import html
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage1"))
from aggregate import DIR  # noqa: E402
from metrics_embed import METHODS, RES, SR  # noqa: E402
from viz_rows import WORST  # noqa: E402

FIG = RES / "figures"
JPG_W, JPG_Q = 1600, 85
Image.MAX_IMAGE_PIXELS = None
LAB = {m: m.replace("pisa_sr_sem", "pisa_sr λ") for m in METHODS}
SERVER = "새 서버(RTX 4090, EPYC 7282, cgroup v2)"
DIRKO = {
    "dinov3_large_cos": "높을수록 원본과 가까움", "personvit_base_cos": "높을수록 원본과 가까움",
    "afine_all_scale": "낮을수록 좋음", "afine_all": "낮을수록 좋음", "fidelity": "낮을수록 좋음", "natural_dis": "낮을수록 좋음",
    "lf_coh": "높을수록 원본 대역 일치", "lf_relerr_db": "낮을수록 원본 대역 일치", "hf_db_vs_bicubic": "방향 없음", "hf_energy": "방향 없음",
    "slope_dev_hi": "0에 가까울수록 원본 기울기 유지", "grid_peak_db": "낮을수록 격자 피크 약함", "peak_db": "낮을수록 주기 피크 약함",
    "de2000": "낮을수록 색 차이 작음",
}
NAMEKO = {
    "dinov3_large_cos": "DINOv3-L 코사인", "personvit_base_cos": "PersonViT-B 코사인", "afine_all_scale": "A-FINE 총점(0~100)",
    "afine_all": "A-FINE 원값(중앙값)", "fidelity": "A-FINE 충실도", "natural_dis": "A-FINE 자연스러움", "lf_coh": "저주파 일치도",
    "lf_relerr_db": "저주파 상대오차 dB", "hf_db_vs_bicubic": "고주파 추가량(bicubic 대비 dB)", "hf_energy": "고주파 에너지",
    "slope_dev_hi": "스펙트럼 기울기 편차", "grid_peak_db": "격자 피크 dB", "peak_db": "주기 피크 dB", "de2000": "ΔE2000(참고)",
}


class Doc:
    def __init__(self):
        self.md, self.ht = [], []

    def h(self, lvl, t, anchor=None):
        self.md.append("#" * lvl + " " + t + "\n")
        a = f' id="{anchor}"' if anchor else ""
        self.ht.append(f"<h{lvl}{a}>{html.escape(t)}</h{lvl}>")

    def p(self, t):
        self.md.append(t + "\n")
        s = html.escape(t)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        self.ht.append(f"<p>{s}</p>")

    def ul(self, items):
        self.md.append("\n".join("- " + i for i in items) + "\n")
        li = []
        for i in items:
            s = re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(i))
            s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
            li.append(f"<li>{s}</li>")
        self.ht.append("<ul>" + "".join(li) + "</ul>")

    def table(self, df, caption=None):
        if caption:
            self.p(caption)
        cols = list(df.columns)
        self.md.append("| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n" +
                       "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n")
        th = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
        trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
        self.ht.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')

    def img(self, path, caption):
        path = Path(path)
        rel = path.relative_to(ROOT)
        self.md.append(f"![{caption}]({rel})\n")
        if not path.exists():
            self.ht.append(f"<p><i>(그림 없음: {html.escape(str(rel))})</i></p>")
            return
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > JPG_W:
                im = im.resize((JPG_W, max(1, round(im.height * JPG_W / im.width))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=JPG_Q, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.ht.append(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)} '
                       f'<span class="src">(원본 PNG: {html.escape(str(rel))})</span></figcaption></figure>')

    def raw_html(self, s):
        self.ht.append(s)

    def write(self, md_path, html_path, title):
        if md_path:
            Path(md_path).write_text("\n".join(self.md), encoding="utf-8")
        css = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--head:#f4f4f6}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--head:#202024}}
body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,"Segoe UI","Noto Sans KR",sans-serif;max-width:1680px;margin:0 auto;padding:16px}
table{border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}th,td{border:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}
th{background:var(--head);position:sticky;top:0}td:first-child,th:first-child{text-align:left}.tw{overflow-x:auto;margin:8px 0 18px}
figure{margin:12px 0 26px}figure img{max-width:100%;height:auto;background:#fff}figcaption{color:var(--mut);font-size:12px}.src{opacity:.7}
code{font-size:12px}h1{font-size:24px}h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:36px}nav a{margin-right:10px}"""
        Path(html_path).write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                   f"<title>{html.escape(title)}</title><style>{css}</style></head><body>" + "\n".join(self.ht) + "</body></html>", encoding="utf-8")


def fmt(v, metric):
    if v == "" or v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    v = float(v)
    if metric.endswith("_cos") or metric == "lf_coh":
        return f"{v:.4f}"
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def summary_table(sm, k, metrics):
    s = sm[sm.k.astype(str) == str(k)]
    rows = []
    for m in METHODS:
        r = {"방법": LAB[m]}
        for c in metrics:
            x = s[(s.method == m) & (s.metric == c)]
            col = f"{NAMEKO[c]} ({DIRKO[c]})"
            if not len(x):
                r[col] = ""
                continue
            x = x.iloc[0]
            if m == "bicubic" and c in ("afine_all_scale", "afine_all", "fidelity", "natural_dis"):
                r[col] = "기준 이미지와 동일, 비교 불가"
            elif c == "afine_all":
                r[col] = f"{fmt(x['median_img'], c)}"
            else:
                r[col] = f"{fmt(x['mean'], c)} [{fmt(x['ci_lo'], c)}, {fmt(x['ci_hi'], c)}]"
        rows.append(r)
    return pd.DataFrame(rows)


def baseline_table(sm):
    s = sm[sm.k.astype(str) == "-"]
    rows = []
    for b, lab in (("base_blur_s05", "같은 사진 + 가우시안 블러 σ=0.5"), ("base_jpeg_q90", "같은 사진 + JPEG 품질 90"), ("base_othercam", "같은 사람 다른 카메라")):
        r = {"기준선": lab}
        for c in ("dinov3_large_cos", "personvit_base_cos"):
            x = s[(s.method == b) & (s.metric == c)]
            r[NAMEKO[c]] = f"{fmt(x.iloc[0]['mean'], c)} [{fmt(x.iloc[0]['ci_lo'], c)}, {fmt(x.iloc[0]['ci_hi'], c)}]" if len(x) else ""
        rows.append(r)
    return pd.DataFrame(rows)


def gen_table():
    rows = []
    for m in METHODS:
        for k in (3, 4):
            f = SR / "_logs" / f"{m}__x{k}__summary.json"
            if not f.exists():
                rows.append({"방법": LAB[m], "배율": k, "생성 경로": "summary 없음"})
                continue
            d = json.loads(f.read_text())
            if m in ("bicubic", "lanczos"):
                path = f"PIL {'LANCZOS' if m == 'lanczos' else 'BICUBIC'} (CPU)"
            elif d.get("from_x4"):
                path = "×4 출력 → PIL BICUBIC 축소 384×192 (CPU)"
            else:
                path = ("공식 ×%d 가중치" % k if m in ("swinir", "hat", "mambairv2") else "공식 ×4") + (", 결정적+예열" if d.get("det") else ", 공식 설정")
            rows.append({"방법": LAB[m], "배율": k, "생성 경로": path, "배치": d.get("batch") or "", "이번 실행 장수": d.get("n_done"),
                         "벽시계 분": f"{d['wall'] / 60:.1f}", "추론 s/장(중앙값)": fmt(d.get("infer_s_per_img_median"), "x") if d.get("infer_s_per_img_median") else "",
                         "피크 VRAM 할당 GiB": fmt(d.get("peak_alloc_gib"), "x") if d.get("peak_alloc_gib") else ""})
    return pd.DataFrame(rows)


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def main():
    sm = read("summary.csv")
    dist = read("distribution_distance.csv")
    man = read("stage1_sample_manifest_summary.csv")
    pix, met, rep = read("det_vs_official_pixels.csv"), read("det_vs_official_metrics.csv"), read("det20_repro.csv")
    s0rep = read("step0_repro.csv")
    fail = {}
    lg = ROOT / "logs" / "bg" / "s1gen.log"
    if lg.exists():
        bad = [l.strip() for l in lg.read_text().splitlines() if re.search(r"종료 \S+ rc=[1-9]", l)]
        fail["text"] = (f"생성 작업 비정상 종료 {len(bad)}건" + (": " + "; ".join(bad) if bad else "") +
                        (" — " + (RES / "stage1_failures_note.txt").read_text().strip() if (RES / "stage1_failures_note.txt").exists() else ""))
    pi = read("per_image_metrics.csv")
    d = Doc()
    d.h(1, "Stage 1 보고서: SR 12종 × {×3, ×4} 비교 (정답 없음, 원본 128×64 기준)")
    d.p(f"작성 2026-11-07. 표본 `configs/stage1_sample.csv`(train 751명, 2,250장, seed 42). test·data/locked는 쓰지 않았다. 시간·처리량은 모두 {SERVER} 측정값이다.")
    d.p("**이 보고서는 순위·추천·선정을 하지 않는다.** 방법은 항상 같은 고정 순서(bicubic, lanczos, realesrgan, osdhuman, swinir, hat, mambairv2, pisa_sr λ0.0~2.0)로 나열하며, 값의 크기순 정렬은 하지 않았다. 선정은 사용자가 한다.")
    d.raw_html('<nav><a href="#gen">1단계 생성</a><a href="#repro">재현성</a><a href="#def">지표 정의</a><a href="#models">모델·라이선스</a><a href="#res">결과표</a>'
               '<a href="#figs">그림</a><a href="#ba">비포·애프터</a><a href="#lim">한계</a></nav>')

    d.h(2, "1. 1단계: SR 생성", "gen")
    d.ul(["출력 `runs/stage1_sample/<method>/x<k>/<원본파일명>.webp`, WebP 무손실 method 2, 저장 직후 픽셀 전수 비교. data/에는 쓰지 않았다.",
          "배치 정책 (가): realesrgan b4(배치 1과 비트 일치 확인됨), 나머지 b1. OMP/MKL 스레드 4. 이미지마다 파일명 기반 seed = sha256(\"42:<파일명>\") 앞 8자리 & 0x7fffffff.",
          "osdhuman, pisa_sr(λ 0.0/0.5/1.0/1.5/2.0), mambairv2, hat: 결정적 설정(cudnn.benchmark=False, cudnn.deterministic=True, SDPA math 백엔드만) + 예열 1회(첫 이미지를 한 번 돌려 버림) + 파일명 seed.",
          "swinir, realesrgan: 공식 설정(난수 미사용). swinir·hat·mambairv2 ×3은 공식 ×3 가중치.",
          "**realesrgan·osdhuman·pisa_sr ×3은 ×4 출력을 PIL BICUBIC(Image.resize, 축소 시 지지폭 확대 = 안티에일리어싱)으로 384×192로 축소했다. 세 방법 모두 같은 커널이다(`stage1/sr_gen.py --from-x4`).**",
          "bicubic·lanczos: 원본 PNG를 PIL BICUBIC / PIL LANCZOS로 ×k."])
    if man is not None:
        tot = int(man.present.sum())
        d.p(f"manifest: `results/stage1/stage1_sample_manifest.csv.gz`(파일별 SHA256·크기·픽셀 크기, 생성 로그 SHA256과 대조). 기대 {int(man.expected.sum()):,}장 중 {tot:,}장 존재, "
            f"누락 {int(man.missing.sum())}, 크기 불일치 {int(man.wrong_size.sum())}, 로그 SHA256 불일치 {int(man.log_sha_mismatch.sum())}, 임시 파일 {int(man.tmp_files.sum())}. "
            f"총 {man.total_bytes.sum() / 2**30:.2f} GiB.")
    d.table(gen_table(), "방법별 생성 기록(마지막 실행의 summary.json). 생성 중 CPU 작업(bicubic·lanczos·×3 축소)과 지표 계산이 같이 돌았으므로 시간은 참고값이다. 측정: " + SERVER)
    if fail:
        d.p("실패·재처리 내역: " + fail.get("text", ""))

    d.h(2, "2. 재현성", "repro")
    if rep is not None:
        rr = rep.copy()
        rr["method"] = rr.method.map(lambda m: LAB.get(m, m))
        d.table(rr.rename(columns={"method": "방법", "k": "배율", "n": "장수", "n_bit_identical": "비트 일치"}),
                "본 생성 설정 그대로 별도 프로세스에서 표본 앞 20장을 다시 생성해 본 출력과 배열 SHA1을 비교(`runs/stage1_step0/det20/`).")
    if s0rep is not None:
        t = s0rep[s0rep.status == "ok"][["method", "k", "config", "n", "n_bit_identical", "infer_s_median_excl_first"]].copy()
        t["config"] = t.config.map({"off": "공식", "det": "결정적", "offw": "공식+예열", "detw": "결정적+예열", "strictw": "결정적+deterministic_algorithms+예열"})
        d.table(t.rename(columns={"method": "방법", "k": "배율", "config": "설정", "n": "장수", "n_bit_identical": "정·역순 실행 비트 일치", "infer_s_median_excl_first": "s/장"}),
                f"0단계 측정(20장, 별도 프로세스 2회, 정방향 대 역순). s/장은 {SERVER}.")
    if pix is not None:
        t = pix.copy()
        t["method"] = t.method.map(lambda m: LAB.get(m, m))
        t["pair"] = t.pair.map({"det-offA": "결정적 − 공식A", "det-offB": "결정적 − 공식B", "offA-offB": "공식A − 공식B(공식 설정 실행 간)"})
        for c in ("mean_abs", "mean_frac_px_diff", "p99_abs"):
            t[c] = t[c].map(lambda v: f"{v:.4g}")
        d.table(t.rename(columns={"method": "방법", "k": "배율", "pair": "비교", "n": "장수", "n_identical": "완전 동일", "max_abs": "최대 픽셀 차(0~255)",
                                  "mean_abs": "평균 절대 차", "mean_frac_px_diff": "다른 픽셀 비율", "p99_abs": "99백분위 차"}),
                "결정적 설정(본 생성) 대 공식 기본 설정 출력 차이(표본 앞 20장). 공식 설정은 별도 프로세스로 두 번(A, B) 돌려, 공식 설정 자체의 실행 간 차이와 나란히 둔다. ×3(osdhuman·pisa_sr)은 각 설정의 ×4를 같은 PIL BICUBIC으로 축소한 것.")
    if met is not None:
        t = met[met.metric.isin(["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "grid_peak_db", "de2000"])].copy()
        t["method"] = t.method.map(lambda m: LAB.get(m, m))
        t["metric"] = t.metric.map(lambda c: NAMEKO.get(c, c))
        keep = ["method", "k", "metric", "det_mean", "offA_mean", "mean_det_minus_offA", "mean_abs_det_minus_offA", "max_abs_det_minus_offA", "mean_abs_offA_minus_offB"]
        t = t[keep]
        for c in keep[3:]:
            t[c] = t[c].map(lambda v: f"{v:.4g}")
        d.table(t.rename(columns={"method": "방법", "k": "배율", "metric": "지표", "det_mean": "결정적 평균", "offA_mean": "공식A 평균", "mean_det_minus_offA": "평균 차(결정적−공식A)",
                                  "mean_abs_det_minus_offA": "이미지별 |차| 평균", "max_abs_det_minus_offA": "이미지별 |차| 최대", "mean_abs_offA_minus_offB": "공식 실행 간 |차| 평균"}),
                "같은 20장에서 지표 값 차이(결정적 대 공식A, 그리고 공식A 대 공식B). 전체 지표는 `results/stage1/det_vs_official_metrics.csv`.")

    d.h(2, "3. 지표 정의와 전처리", "def")
    d.ul(["모든 지표는 원본 128×64와 실제 SR 사이에서만 계산한다(HR 정답 없음).",
          "**임베딩(DINOv3, PersonViT)**: 원본·SR·기준선 모두 PIL BICUBIC으로 256×128로 맞춘 뒤 모델별 정규화, 최종 norm 뒤 CLS 토큰, L2 정규화. 이미지별 코사인 = 원본 임베딩과 SR 임베딩의 내적.",
          "DINOv3: timm 변환본 `vit_large_patch16_dinov3.lvd1689m`(ViT-L/16, 303M), ImageNet mean/std, RoPE 주기를 bfloat16으로 잘라 원본 DINOv3와 맞춤(timm 모델 카드 안내). 보정 전후 코사인 차이 1e-6 수준(64장 확인).",
          "PersonViT: `lakeAGI/PersonViT` ViT-B LUPerson 자기지도 사전학습 체크포인트(checkpoint0260)의 teacher backbone, 입력 256×128, mean/std 0.5/0.5. Market 미세조정 가중치는 쓰지 않았다.",
          "**분포 거리**: 원본 2,250장 임베딩 집합과 방법별 SR 집합 사이 Fréchet distance(대칭 고윳값 분해로 행렬 제곱근, float64)와 KID(3차 다항 커널 (x·y/d+1)³, 부분집합 1,000장 × 100회, seed 42, 비편향 MMD²). L2 정규화 임베딩 기준. 집합 단위 점추정이며 신원 부트스트랩 CI는 없다(KID는 부분집합 간 표준편차를 같이 적음).",
          "**A-FINE**(CVPR 2025, pyiqa 사본 가중치): 시험 이미지 = SR(원 크기), 기준 이미지 = 원본을 SR 크기로 PIL BICUBIC 확대. 공식 계산 순서(QuickInference/afine.py), NLM 모듈만 배치용 원소별 구현(공식과 최대 차 1.2e-7). 보고: 총점 `afine_all_scale`(0~100), 충실도 F, 자연스러움 N(SR). **모든 A-FINE 값은 낮을수록 좋다(공식 README).** 원값 `afine_all`은 꼬리가 극단적이라 이미지 중앙값만 적는다. bicubic은 시험 이미지와 기준 이미지가 같아 비교 불가.",
          "**FFT(Y 채널, BT.601)**: Moisan periodic+smooth 분해로 경계 불연속 제거, 평균 제거, DFT/픽셀 수. 주파수 단위는 원본 픽셀당 주기(원본 나이퀴스트 0.5). SR 스펙트럼은 픽셀 중심 정렬 차이 δ=(k−1)/(2k) 원본 픽셀만큼 위상 보정.",
          "저주파 일치도 `lf_coh`: 원본 대역(|f|<0.5) 안의 원본·SR 복소 스펙트럼 정규화 상관(1이면 일치). `lf_relerr_db`: 같은 대역 상대 오차 dB.",
          "고주파 추가량 `hf_energy`: 원본 대역 밖 SR 스펙트럼 에너지 합. 표에는 같은 이미지 bicubic 대비 dB(`hf_db_vs_bicubic`)를 적는다. 많고 적음에 좋고 나쁨의 방향은 없다.",
          "스펙트럼 기울기 편차 `slope_dev_hi`: SR 고대역(f∈[0.55, 0.45k]) 방사형 파워 log-log 기울기 − 원본 기울기(f∈[0.05, 0.40]). 0에 가까우면 원본 경향 유지, 음수면 고주파 급감, 양수면 원본 경향보다 많음.",
          "격자 피크 `grid_peak_db`: 원본 표본화 주파수 정수배 격자점에서 로그 파워가 주변 7×7(가운데 3×3 제외) 중앙값보다 높은 정도의 최대값(dB). `peak_db`: 원본 대역 밖 전체에서 같은 국소 대비의 최대값.",
          "**×4 후 축소한 방법(realesrgan, pisa_sr, osdhuman)의 ×3 FFT 격자 점수에는 축소 과정(4→3 재표본화)의 영향이 포함된다.** 0단계 비교에서 ×4 후 축소가 직접 ×3보다 격자 피크가 4.7~7.5 dB 높았다.",
          "ΔE2000(참고값): SR을 원본 크기로 PIL BICUBIC 축소 후 픽셀별 CIEDE2000 평균(skimage, sRGB→Lab D65). 낮을수록 색 차이 작음.",
          "**해석 기준선**(임베딩 코사인용, 그림에만 겹쳐 그림): 같은 사진 + 가우시안 블러 σ=0.5(원본 픽셀, scipy, reflect), 같은 사진 + JPEG 품질 90(PIL, 4:2:0), 같은 사람 다른 카메라 원본(표본 안에서 seed 42로 하나, 2,250장 모두 표본 안에서 짝을 찾음, `results/stage1/baseline_othercam_pairs.csv`).",
          "**집계**: 이미지 → 신원 평균 → 751명 평균. 95% CI는 신원 단위 부트스트랩(10,000회, seed 42, 모든 방법·지표에 같은 재표본, 백분위 구간). 이미지별 원시 값 `results/stage1/per_image_metrics.csv`, `per_image_baselines.csv`, 요약 `summary.csv`."])
    d.p("각 지표가 재는 것과 못 재는 것: 임베딩 코사인은 해당 모델이 보는 의미·신원 정보의 보존을 재지만 사람 눈의 품질이나 ReID 성능을 직접 재지 않는다. A-FINE은 원본(확대본) 대비 충실도와 자연스러움을 함께 보는 화질 모델로, 기준 이미지가 저해상도 확대본이라 기준보다 나은 이미지를 낮은 값(좋음)으로 줄 수 있다. FFT 지표는 주파수 구성의 변화를 기술할 뿐, 추가된 고주파가 실제 디테일인지 환각인지 구분하지 못한다. ΔE는 축소 후 색만 본다.")

    d.h(2, "4. 모델·가중치·라이선스", "models")
    d.table(pd.DataFrame([
        {"모델": "DINOv3 ViT-L/16 (timm 변환본)", "가중치": "weights/stage1/dinov3/vit_large_patch16_dinov3.lvd1689m/model.safetensors (sha256 45172f20…)", "라이선스": "DINOv3 License(Meta). 연구 결과 발표 시 DINO Materials 사용을 명시할 의무, 군사·ITAR 용도 금지"},
        {"모델": "PersonViT ViT-B (LUPerson 자기지도)", "가중치": "weights/stage1/personvit/vitb…/checkpoint0260.pth (sha256 6aea7dfd…)", "라이선스": "코드 Apache-2.0, 가중치 저장소 표기 \"cc\"(세부 CC 종류 불명)"},
        {"모델": "A-FINE (pyiqa 사본) + CLIP ViT-B/32", "가중치": "weights/stage1/afine/pyiqa_mirror/afine.pth (sha256 0fd31f02…), ViT-B-32.pt", "라이선스": "코드 Apache-2.0, pyiqa 가중치 저장소 CC-BY-NC-SA-4.0. 공식 Google Drive가 쿼터 초과라 공식 README가 안내하는 pyiqa 사본 사용(키·strict 로드 일치)"},
    ]))
    d.p("**DINOv3 사용 표기(DINOv3 License 1.b.ii)**: 이 연구는 Meta의 DINOv3(DINO Materials, timm 변환 가중치 `vit_large_patch16_dinov3.lvd1689m`)를 사용했다. 이 결과를 발표할 때는 DINO Materials 사용을 논문에 명시해야 한다.")
    d.p("체크섬 전체: `weights/stage1/SHA256.csv`, SR 가중치 `results/stage1/step0_sr_weights.csv`.")

    d.h(2, "5. 결과표 (평균 [95% CI], 신원 751명)", "res")
    if sm is not None:
        for k in (3, 4):
            d.h(3, f"×{k}")
            d.table(summary_table(sm, k, ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "afine_all"]), f"×{k} 임베딩·A-FINE (A-FINE 원값은 이미지 중앙값만)")
            d.table(summary_table(sm, k, ["lf_coh", "lf_relerr_db", "hf_db_vs_bicubic", "slope_dev_hi", "grid_peak_db", "peak_db", "de2000"]), f"×{k} FFT·색")
        d.table(baseline_table(sm), "해석 기준선(배율 무관, 원본 128×64에서 만듦)")
    if dist is not None:
        t = dist.copy()
        t["method"] = t.method.map(lambda m: LAB.get(m, {"base_blur_s05": "기준선: 블러 σ0.5", "base_jpeg_q90": "기준선: JPEG q90", "base_othercam": "기준선: 다른 카메라"}.get(m, m)))
        t = t[["model", "method", "k", "n", "frechet", "kid", "kid_std_subsets"]]
        for c in ("frechet", "kid", "kid_std_subsets"):
            t[c] = t[c].map(lambda v: f"{v:.3g}")
        d.table(t.rename(columns={"model": "모델", "method": "집합", "k": "배율", "n": "장수", "frechet": "Fréchet (낮을수록 원본 분포와 가까움)", "kid": "KID (낮을수록 가까움)", "kid_std_subsets": "KID 부분집합 표준편차"}),
                "분포 거리(원본 집합 대비, L2 정규화 임베딩). 기준선 '다른 카메라'는 같은 신원들의 다른 사진 집합이다.")
    if pi is not None:
        nan = pi.drop(columns=["method", "k", "name", "pid", "cam"]).isna().sum()
        nan = nan[nan > 0]
        if len(nan):
            d.p("계산 불가(NaN) 값: " + ", ".join(f"{c} {int(v)}건" for c, v in nan.items()) + " (bicubic·lanczos 등에서 고대역 기울기 구간의 파워가 0에 가까운 경우 등, 집계에서 제외).")

    d.h(2, "6. 그림", "figs")
    d.h(3, "지표별 분포 (상자그림, 이미지 단위, 임베딩에는 기준선 3종을 회색으로 겹침)")
    for c in ["dinov3_large_cos", "personvit_base_cos", "afine_all_scale", "fidelity", "natural_dis", "lf_coh", "lf_relerr_db", "hf_energy", "hf_db_vs_bicubic", "slope_dev_hi", "grid_peak_db", "peak_db", "de2000"]:
        d.img(FIG / f"dist_{c}.png", f"{NAMEKO.get(c, c)} ({DIRKO.get(c, '')}) 분포")
    d.h(3, "요약 (평균과 신원 부트스트랩 95% CI)")
    for k in (3, 4):
        d.img(FIG / f"forest_x{k}.png", f"×{k} 방법 × 지표 평균·CI (A-FINE에서 bicubic 제외)")
    d.h(3, "지표 간 상관 (이미지 단위 Spearman)")
    for k in (3, 4):
        d.img(FIG / f"spearman_x{k}.png", f"×{k} Spearman: 왼쪽 11종 합침, 오른쪽 방법별 계산 후 평균")
    d.h(3, "t-SNE")
    cfg = json.loads((RES / "tsne_config.json").read_text()) if (RES / "tsne_config.json").exists() else {}
    d.p("설정: " + ", ".join(f"{k}={v}" for k, v in cfg.items()) + ". 좌표 `results/stage1/tsne_<model>_x<k>.csv.gz`. t-SNE의 거리·군집 크기는 해석에 주의가 필요하다(국소 구조만 보존).")
    for mk in ("dinov3_large", "personvit_base"):
        for k in (3, 4):
            d.img(FIG / f"tsne_{mk}_x{k}.png", f"t-SNE {mk} ×{k}: 원본 + 12종")
            d.img(FIG / f"tsne_{mk}_x{k}_facets.png", f"t-SNE {mk} ×{k}: 방법별 강조")
    d.h(3, "FFT")
    for k in (3, 4):
        d.img(FIG / f"fft_radial_x{k}.png", f"×{k} 방법별 평균 방사형 파워 스펙트럼(원본·bicubic 겹침)")
    for p in sorted(FIG.glob("fft2d_*_x*.png")):
        d.img(p, f"대표 이미지 2D 스펙트럼 비포·애프터 {p.stem}")

    d.h(2, "7. 비포·애프터 사진 비교", "ba")
    d.p("한 줄 = 한 이미지: 원본(최근접 확대) | bicubic | 나머지 11종. 칸마다 위는 전체(높이 256으로 축소), 아래는 얼굴·상의(이미지 위쪽 45%)를 SR 원 해상도로 자른 확대. 빨간 테두리는 그 줄의 대상 방법.")
    d.p("하위(나쁜 쪽) 정의: " + ", ".join(f"{c}={'낮을수록' if w == 'low' else ('높을수록' if w == 'high' else '0에서 멀수록')} 나쁨" for c, w in WORST.items()) +
        ". A-FINE은 bicubic 제외. 고주파 추가량은 방향이 없어 최악 목록에서 뺐다.")
    for k in (3, 4):
        d.img(FIG / f"random20_x{k}.png", f"d. ×{k} 무작위 20장(seed 42)")
    cw = read("common_worst.csv")
    for k in (3, 4):
        if cw is not None:
            c = cw[(cw.k == k) & (cw["count"] >= 2)]
            none = cw[(cw.k == k) & (cw["count"] == 0)].method.tolist()
            d.p(f"b. ×{k} 공통 최악: 같은 방법에서 2개 이상 지표의 하위 10장에 든 경우 {len(c)}건(그림은 등장 횟수 순 최대 30줄, 전체 `results/stage1/common_worst.csv`)."
                + (f" 공통 최악이 없는 방법: {', '.join(LAB[m] for m in none)} → (a) 지표별 최악 참조." if none else ""))
        d.img(FIG / f"common_worst_x{k}.png", f"b. ×{k} 여러 지표 공통 최악")
    ab = read("allbad_vs_specific.csv")
    d.p("c. 구분 규칙: 배율·지표마다 이미지의 방법 내 백분위(0 = 그 방법에서 가장 나쁨)를 구해, 비교 가능한 모든 방법에서 백분위 ≤ 10%면 '모든 방법에서 나쁨(원본 문제 후보)', "
        "한 방법의 하위 10장이면서 나머지 방법 백분위 중앙값 ≥ 50%면 '특정 방법만 나쁨'으로 표시했다(이 기준은 시각화용으로 정한 것). 목록 `results/stage1/allbad_vs_specific.csv`.")
    for k in (3, 4):
        if ab is not None:
            a = ab[(ab.k == k) & (ab.kind == "allbad")]
            s = ab[(ab.k == k) & (ab.kind == "specific")]
            d.p(f"×{k}: 모든 방법에서 나쁨 {a.name.nunique()}장({len(a)}건, 지표별), 특정 방법만 나쁨 {len(s)}건.")
        d.img(FIG / f"allbad_x{k}.png", f"c. ×{k} 모든 방법에서 나쁜 이미지")
        d.img(FIG / f"specific_x{k}.png", f"c. ×{k} 특정 방법에서만 나쁜 이미지")
    d.p("a. 방법·배율·지표별 최악 10장은 크기 때문에 별도 파일 `STAGE1_REPORT_worst_x3.html`, `STAGE1_REPORT_worst_x4.html`에 있다. 목록 `results/stage1/worst10.csv`, 원본 PNG `results/stage1/figures/worst/x<k>/`.")
    d.raw_html('<p><a href="STAGE1_REPORT_worst_x3.html">STAGE1_REPORT_worst_x3.html</a> · <a href="STAGE1_REPORT_worst_x4.html">STAGE1_REPORT_worst_x4.html</a></p>')

    d.h(2, "8. 한계", "lim")
    d.ul(["정답(HR)이 없으므로 모든 지표는 원본 128×64와의 관계만 본다. 원본보다 '좋아진' 정도는 직접 재지 못한다.",
          "PersonViT: 사전학습 때의 입력 정규화를 확인할 수 없다(사전학습 코드 미공개). 0.5/0.5는 공식 미세조정 config 값이다.",
          "PersonViT: 논문은 사전학습 체크포인트(epoch)를 ReID 정확도로 골랐다. 가중치에 라벨 학습은 없지만 체크포인트 선택에 벤치마크 정보가 간접적으로 들어갔다.",
          "DINOv3는 Meta 원본이 아니라 timm 변환본이다(RoPE 주기 bf16 보정으로 원본 출력과 맞춤, 모델 카드 기준).",
          "A-FINE: pyiqa 사본 가중치(공식 Google Drive 쿼터 초과). 기준 이미지가 원본의 bicubic 확대본이므로 bicubic 행은 비교 불가. 원값은 꼬리가 극단적이라 총점은 0~100 척도와 원값 중앙값으로만 본다.",
          "×4 후 축소한 방법(realesrgan, pisa_sr, osdhuman)의 ×3 값, 특히 FFT 격자 점수에는 축소 과정의 영향이 포함된다.",
          "결정적 설정은 공식 기본 설정과 커널 선택이 달라 출력이 조금 다르다(2절 표). 공식 설정 자체도 실행마다 달라진다.",
          "분포 거리(Fréchet, KID)는 집합 단위 점추정이며 신원 부트스트랩 CI가 없다. 2,250장, 768~1,024차원이므로 Fréchet의 공분산 추정에 표본 크기 영향이 있다.",
          "기준선 '다른 카메라'는 같은 표본 안의 짝이며 카메라 조합은 통제하지 않았다.",
          "c의 '모든 방법에서 나쁨/특정 방법만 나쁨' 기준(10%, 중앙값 50%)과 최악 정의는 시각화용으로 정한 것이다.",
          "생성 시간은 다른 CPU 작업과 동시에 돈 값이라 벤치마크가 아니다."])
    d.write(ROOT / "STAGE1_REPORT.md", ROOT / "STAGE1_REPORT.html", "Stage 1 보고서")

    for k in (3, 4):
        w = Doc()
        w.h(1, f"Stage 1 비포·애프터 a: 방법·지표별 최악 10장 (×{k})")
        w.p("한 줄 = 한 이미지: 원본(최근접 확대) | bicubic | 나머지 11종, 아래는 위쪽 45% 원 해상도 확대. 빨간 테두리 = 대상 방법. 칸 아래 숫자 = 그 지표 값. 순서는 고정 방법 순서이며 순위가 아니다.")
        w.p("하위 정의: " + ", ".join(f"{c}={'낮을수록' if v == 'low' else ('높을수록' if v == 'high' else '0에서 멀수록')} 나쁨" for c, v in WORST.items()) + ". A-FINE은 낮을수록 좋음(bicubic 제외).")
        for m in METHODS:
            w.h(2, LAB[m])
            for c in WORST:
                p = FIG / "worst" / f"x{k}" / f"{m}__{c}.png"
                if p.exists():
                    w.img(p, f"×{k} {LAB[m]} — {NAMEKO.get(c, c)} ({DIRKO.get(c, '')}) 최악 10장")
        w.write(None, ROOT / f"STAGE1_REPORT_worst_x{k}.html", f"Stage 1 최악 10장 x{k}")
    for p in ("STAGE1_REPORT.html", "STAGE1_REPORT_worst_x3.html", "STAGE1_REPORT_worst_x4.html", "STAGE1_REPORT.md"):
        print(p, f"{(ROOT / p).stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
