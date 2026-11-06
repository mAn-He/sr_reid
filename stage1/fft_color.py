"""Stage 1 지표 4(FFT 비포·애프터)와 5(ΔE2000). 정답(HR) 없이 원본 128×64와 SR(k×)만 쓴다. srreid-core env, CPU.

공통 전처리
- Y 채널: BT.601 luma, Y = 0.299R + 0.587G + 0.114B (0~255 실수).
- 경계 처리: FFT 전에 Moisan의 periodic-plus-smooth 분해로 주기 성분만 쓴다(이미지 가장자리 불연속이 만드는 십자 모양 가짜 고주파 제거, 창 함수 없음).
- 평균(DC)을 뺀 뒤 DFT를 픽셀 수로 나눠(F/(H·W)) 크기가 다른 원본과 SR을 같은 척도로 둔다.
- 주파수 단위: 원본 픽셀당 주기(cycles/original px). 원본 나이퀴스트 = 0.5, SR 나이퀴스트 = 0.5k.
- 위치 맞춤: SR 픽셀 j의 중심은 원본 좌표 (j+0.5)/k − 0.5에 있다(픽셀 중심 정렬, PIL·대부분 SR 모델과 같음).
  SR의 DFT는 j/k 위치를 가정하므로 δ = (k−1)/(2k) 원본 픽셀만큼 어긋난다. 저주파 일치도 계산 전에 SR 스펙트럼에 exp(+i2πfδ)를 곱해 보정한다(SR이 원본보다 δ만큼 앞에 표본화된 것으로 보이는 위상을 되돌림).

지표 (이미지별)
- lf_coh: 저주파 일치도. 원본 대역(|fy|<0.5, |fx|<0.5, 나이퀴스트 칸 제외) 안에서 원본 스펙트럼 F_O와 위치 보정한 SR 스펙트럼 F_S의
  복소 상관 Re Σ F_O·conj(F_S) / sqrt(Σ|F_O|²·Σ|F_S|²). 1이면 SR을 원본 대역으로 이상적 저역통과했을 때 원본과 (밝기 배율을 빼고) 같다. 위상까지 본다.
- lf_relerr_db: 같은 대역의 상대 오차 20·log10(‖F_S − F_O‖/‖F_O‖) dB. 낮을수록 일치.
- hf_energy: 고주파 추가량. 원본 대역 밖(|fy|≥0.5 또는 |fx|≥0.5) SR 스펙트럼 에너지의 합 Σ|F_S|² (픽셀당 평균 제곱 밝기 단위, Y²).
  hf_frac: 그 에너지가 전체 AC 에너지에서 차지하는 비율. 집계 단계에서 bicubic 대비 dB도 낸다.
- slope_o, slope_s_lo, slope_s_hi: 방사형(원형 평균) 파워 스펙트럼 log P – log f 직선 기울기.
  원본은 f∈[0.05, 0.40], SR 저대역은 같은 범위, SR 고대역은 f∈[0.55, 0.45k].
  slope_dev_hi = slope_s_hi − slope_o: 원본의 1/f^α 경향을 원본 나이퀴스트 너머로 이어 가는지(0에 가까울수록 원본 기울기 유지,
  음수로 크면 고주파가 빠르게 죽음(흐림), 양수면 고주파가 원본 경향보다 많음(잡음·과잉 질감)).
  slope_dev_nat = slope_s_hi − (−2): 자연 이미지 평균 기울기(파워 ∝ 1/f², 문헌값) 대비.
- grid_peak_db: 격자 점수. 원본 표본화 주파수의 정수배 격자점(fy, fx ∈ {0, ±1, ±2, …} cycles/original px, (0,0) 제외, SR 대역 안)에서
  로그 파워가 주변 7×7(가운데 3×3 제외) 중앙값보다 얼마나 높은지(dB)의 최대값. k배 확대 구조가 남긴 주기적 아티팩트를 본다.
- peak_db: 원본 대역 밖 전체에서 같은 국소 대비(7×7 중앙값 대비)의 최대값 dB. 격자점 밖의 주기적 무늬까지 본다.
- de2000: 원본과 SR을 원본 크기로 축소(PIL BICUBIC, 축소 시 PIL의 지지폭 확대 = 안티에일리어싱)한 것의 픽셀별 CIEDE2000 평균(skimage, sRGB→Lab D65).
"""
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

F_LO = (0.05, 0.40)


def to_y(rgb):
    a = np.asarray(rgb, dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def periodic_component(u):
    """Moisan (2011) periodic plus smooth decomposition. u = p + s, p를 돌려준다."""
    h, w = u.shape
    v = np.zeros_like(u)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] += u[0, :] - u[-1, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] += u[:, 0] - u[:, -1]
    q = np.arange(h)[:, None]
    r = np.arange(w)[None, :]
    den = 2 * np.cos(2 * np.pi * q / h) + 2 * np.cos(2 * np.pi * r / w) - 4
    den[0, 0] = 1.0
    s_hat = np.fft.fft2(v) / den
    s_hat[0, 0] = 0.0
    return u - np.real(np.fft.ifft2(s_hat))


def spectrum(y):
    p = periodic_component(y)
    p = p - p.mean()
    return np.fft.fft2(p) / p.size  # 정규화 DFT, 원점 [0,0]


def freqs(h, w, k):
    """원본 픽셀당 주기 단위 주파수 격자(fftfreq 순서)."""
    fy = np.fft.fftfreq(h) * k
    fx = np.fft.fftfreq(w) * k
    return fy[:, None] * np.ones((1, w)), np.ones((h, 1)) * fx[None, :]


def radial_slope(F, k, lo, hi):
    h, w = F.shape
    fy, fx = freqs(h, w, k)
    r = np.sqrt(fy ** 2 + fx ** 2)
    P = np.abs(F) ** 2
    df = 1.0 / 64  # 원본 짧은 변(64)의 주파수 해상도 폭으로 원형 구간을 나눔
    edges = np.arange(lo, hi + df, df)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= 3:
            xs.append(np.log10((a + b) / 2))
            ys.append(np.log10(P[m].mean() + 1e-20))
    if len(xs) < 3:
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def local_contrast_db(P_shift):
    """로그 파워와 주변 7×7(가운데 3×3 제외) 중앙값의 차(dB). 가운데를 빼려고 해당 칸을 nan 처리한 대신 큰 창 중앙값을 쓴다."""
    L = 10 * np.log10(P_shift + 1e-20)
    fp = np.ones((7, 7), bool)
    fp[2:5, 2:5] = False
    med = median_filter(L, footprint=fp, mode="wrap")
    return L - med


def fft_metrics(orig_rgb, sr_rgb, k):
    yo, ys = to_y(orig_rgb), to_y(sr_rgb)
    h, w = yo.shape
    H, W = ys.shape
    assert (H, W) == (h * k, w * k), (yo.shape, ys.shape, k)
    Fo, Fs = spectrum(yo), spectrum(ys)
    # 원본 대역: 원본 DFT 인덱스 (|i|<h/2, |j|<w/2), 나이퀴스트 칸 제외
    iy = np.r_[0:h // 2, -(h // 2) + 1:0]
    ix = np.r_[0:w // 2, -(w // 2) + 1:0]
    Fo_b = Fo[np.ix_(iy % h, ix % w)]
    Fs_b = Fs[np.ix_(iy % H, ix % W)]
    d = (k - 1) / (2 * k)
    fyo = (iy / h)[:, None]
    fxo = (ix / w)[None, :]
    Fs_b = Fs_b * np.exp(2j * np.pi * (fyo + fxo) * d)  # 원본 픽셀 단위 δ만큼 위치 보정(두 축 같음). 부호는 bicubic 위상 탐색으로 확인(최대 상관이 δ=−(k−1)/2k 이동에서 나옴)
    num = np.real(np.sum(Fo_b * np.conj(Fs_b)))
    den = np.sqrt(np.sum(np.abs(Fo_b) ** 2) * np.sum(np.abs(Fs_b) ** 2)) + 1e-20
    lf_coh = num / den
    lf_relerr_db = 20 * np.log10(np.linalg.norm(Fs_b - Fo_b) / (np.linalg.norm(Fo_b) + 1e-20) + 1e-20)
    fy, fx = freqs(H, W, k)
    out_band = (np.abs(fy) >= 0.5) | (np.abs(fx) >= 0.5)
    P = np.abs(Fs) ** 2
    hf_energy = float(P[out_band].sum())
    hf_frac = hf_energy / (float(P.sum()) + 1e-20)
    slope_o = radial_slope(Fo, 1, *F_LO)
    slope_s_lo = radial_slope(Fs, k, *F_LO)
    slope_s_hi = radial_slope(Fs, k, 0.55, 0.45 * k)
    C = local_contrast_db(np.fft.fftshift(P))
    Cu = np.fft.ifftshift(C)
    lattice = np.zeros_like(out_band)
    for my in range(-(k // 2), k // 2 + 1):
        for mx in range(-(k // 2), k // 2 + 1):
            if my == 0 and mx == 0:
                continue
            iy0, ix0 = int(round(my * H / k)) % H, int(round(mx * W / k)) % W
            if abs(my) <= k / 2 and abs(mx) <= k / 2:
                lattice[iy0, ix0] = True
    grid_peak_db = float(Cu[lattice].max()) if lattice.any() else np.nan
    peak_db = float(Cu[out_band].max())
    return {"lf_coh": float(lf_coh), "lf_relerr_db": float(lf_relerr_db), "hf_energy": hf_energy, "hf_frac": hf_frac,
            "slope_o": slope_o, "slope_s_lo": slope_s_lo, "slope_s_hi": slope_s_hi, "slope_dev_hi": slope_s_hi - slope_o,
            "slope_dev_nat": slope_s_hi + 2.0, "grid_peak_db": grid_peak_db, "peak_db": peak_db}


def de2000(orig_rgb, sr_rgb):
    from skimage.color import deltaE_ciede2000, rgb2lab
    o = np.asarray(orig_rgb)
    h, w = o.shape[:2]
    s = np.asarray(Image.fromarray(np.asarray(sr_rgb)).resize((w, h), Image.BICUBIC))
    return float(deltaE_ciede2000(rgb2lab(o), rgb2lab(s)).mean())


def radial_profile(rgb, k, nbins=None):
    """그림용 방사형 평균 파워 스펙트럼. f(원본 픽셀당 주기)와 P를 돌려준다."""
    F = spectrum(to_y(rgb))
    H, W = F.shape
    fy, fx = freqs(H, W, k)
    r = np.sqrt(fy ** 2 + fx ** 2)
    df = 1.0 / 64
    edges = np.arange(df, 0.5 * k + df, df)
    P = np.abs(F) ** 2
    f, p = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.any():
            f.append((a + b) / 2)
            p.append(P[m].mean())
    return np.array(f), np.array(p)
