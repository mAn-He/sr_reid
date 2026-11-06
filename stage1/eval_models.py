"""Stage 1 평가 모델 공용 적재기. srreid-quality env. 로직은 stage1/eval_model_check.py(적재 점검, 공식 경로와 등가 확인)와 같다.
- 임베딩: 원본·SR 모두 PIL BICUBIC으로 EMB_HW(256×128)로 맞춘 뒤 모델별 정규화. 최종 norm 뒤 CLS 토큰(PLIP는 AttentionPool2d 출력).
- A-FINE: dis = SR(원 크기), ref = 원본을 SR 크기로 PIL BICUBIC 확대. 모든 출력은 낮을수록 좋다.
third_party 소스는 고치지 않는다."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "weights" / "stage1"
TP = ROOT / "third_party"
EMB_HW = (256, 128)
DEV = torch.device("cuda")


def to_tensor(imgs, hw, mean, std):
    """imgs: PIL RGB 리스트. hw와 크기가 다르면 PIL BICUBIC 리사이즈."""
    arr = [np.asarray(im.resize((hw[1], hw[0]), Image.BICUBIC) if im.size != (hw[1], hw[0]) else im) for im in imgs]
    x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float().div(255.0)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


class Embedder:
    def __init__(self, name, model, fe, mean, std):
        self.name, self.model, self.fe, self.mean, self.std = name, model, fe, mean, std

    @torch.no_grad()
    def __call__(self, imgs, bs=64):
        out = []
        for i in range(0, len(imgs), bs):
            x = to_tensor(imgs[i:i + bs], EMB_HW, self.mean, self.std).to(DEV)
            out.append(F.normalize(self.fe(x).float(), dim=1).cpu())
        return torch.cat(out).numpy()


def dinov3(size="large"):
    import timm
    name = f"vit_{size}_patch16_dinov3.lvd1689m"
    m = timm.create_model(name, pretrained=True, pretrained_cfg_overlay=dict(file=str(W / f"dinov3/{name}/model.safetensors")), img_size=EMB_HW).eval().to(DEV)
    cfg = m.pretrained_cfg
    return Embedder(f"dinov3_{size}", m, lambda x: m.forward_features(x)[:, 0], cfg["mean"], cfg["std"])


def personvit(size="base", norm=(0.5, 0.5)):
    sys.path.insert(0, str(TP / "PersonViT/transreid_pytorch/model/backbones"))
    import vit_pytorch as vp
    ck = {"small": W / "personvit/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0220.pth",
          "base": W / "personvit/vitb.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0260.pth"}[size]
    fac = {"small": vp.vit_small_patch16_224_TransReID, "base": vp.vit_base_patch16_224_TransReID}[size]
    m = fac(img_size=EMB_HW, stride_size=16, drop_path_rate=0.0, camera=0, view=0, local_feature=False)
    raw = torch.load(ck, map_location="cpu")
    sd = {k[len("backbone."):]: v for k, v in raw["teacher"].items() if k.startswith("backbone.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"fc.weight", "fc.bias"}, (missing, unexpected)
    m = m.eval().to(DEV)
    mean, std = [norm[0]] * 3, [norm[1]] * 3
    sys.path.pop(0)
    return Embedder(f"personvit_{size}", m, lambda x: m(x), mean, std)


def plip():
    sys.path.insert(0, str(TP / "PLIP"))
    import visual_model as vm
    raw = torch.load(W / "plip/PLIP_RN50.pth.tar", map_location="cpu")
    m = vm.Image_encoder_ModifiedResNet([3, 4, 6, 3], 768, 8, input_resolution=[EMB_HW[1], EMB_HW[0]])
    m.load_state_dict(raw["ImgEncoder_state_dict"], strict=True)
    m = m.eval().to(DEV)
    sys.path.pop(0)
    return Embedder("plip_rn50", m, lambda x: m(x)[0], [0.357, 0.323, 0.328], [0.252, 0.242, 0.239])


class AFINE:
    """공식 QuickInference/afine.py 계산 순서. NLM 모듈만 배치용 원소별 torch.where 판(공식 batch 1과 최대 차 1.2e-7, eval_model_check.json)."""
    MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    def __init__(self):
        sys.path.insert(0, str(TP / "AFINE/QuickInference"))
        from utils.network_archs import AFINEQhead, AFINEDhead, AFINENLM_NR_Fit, AFINENLM_FR_Fit_with_limit, AFINELearnLambda
        from CLIP_ReturnFea import clip
        sd = torch.load(W / "afine/pyiqa_mirror/afine.pth", map_location="cpu")
        clip_model, _ = clip.load(str(W / "afine/ViT-B-32.pt"), device="cpu", jit=False)
        clip_model.load_state_dict(sd["finetuned_clip"])
        q = AFINEQhead(); q.load_state_dict(sd["natural"], strict=True)
        dh = AFINEDhead(); dh.load_state_dict(sd["fidelity"], strict=True)

        class BatchFR(AFINENLM_FR_Fit_with_limit):
            def forward(self, x):
                y3 = torch.clamp(self.yita3, self.yita3_lower, self.yita3_upper)
                y4 = torch.clamp(self.yita4, self.yita4_lower, self.yita4_upper)
                e = -1 * (x - y3) / (torch.abs(y4) + 1e-10)
                return torch.where(e >= 10, (self.yita1 - self.yita2) * torch.exp(-e) / (1 + torch.exp(-e)) + self.yita2,
                                   (self.yita1 - self.yita2) / (1 + torch.exp(e)) + self.yita2)

        class BatchNR(AFINENLM_NR_Fit):
            def forward(self, x):
                e = -1 * (x - self.yita3) / (torch.abs(self.yita4) + 1e-10)
                return torch.where(e >= 10, (self.yita1 - self.yita2) * torch.exp(-e) / (1 + torch.exp(-e)) + self.yita2,
                                   (self.yita1 - self.yita2) / (1 + torch.exp(e)) + self.yita2)

        sfr = BatchFR(yita1=2, yita2=-2, yita3=0.5, yita4=0.15, yita3_upper=0.95, yita3_lower=0.05, yita4_upper=0.70, yita4_lower=0.01)
        sfr.load_state_dict(sd["fidelity_scale"], strict=True)
        snr = BatchNR(yita1=2, yita2=-2, yita3=4.9592, yita4=21.5968); snr.load_state_dict(sd["natural_scale"], strict=True)
        ad = AFINELearnLambda(k=5); ad.load_state_dict(sd["adapter"], strict=True)
        self.mods = [clip_model, q, dh, sfr, snr, ad]
        for mm in self.mods:
            mm.eval().to(DEV)
        self.clip, self.q, self.dh, self.sfr, self.snr, self.ad = self.mods
        sys.path.pop(0)

    @staticmethod
    def scale_final(s, y1=100, y2=0, y3=-1.9710, y4=-2.3734):
        e = -1 * (s - y3) / (abs(y4) + 1e-10)
        return torch.where(e >= 10, (y1 - y2) * torch.exp(-e) / (1 + torch.exp(-e)) + y2, (y1 - y2) / (1 + torch.exp(e)) + y2)

    @torch.no_grad()
    def __call__(self, origs, srs, bs=32):
        """origs: 원본 PIL(128×64), srs: SR PIL(같은 크기끼리). 반환: 필드별 numpy 배열."""
        res = {k: [] for k in ("afine_all_scale", "afine_all", "fidelity", "natural_dis", "natural_ref")}
        size = srs[0].size
        hw = (size[1], size[0])
        for i in range(0, len(srs), bs):
            dis = to_tensor(srs[i:i + bs], hw, self.MEAN, self.STD).to(DEV)
            ref = to_tensor([o.resize(size, Image.BICUBIC) for o in origs[i:i + bs]], hw, self.MEAN, self.STD).to(DEV)
            h, w = hw
            ph, pw = (-h) % 32, (-w) % 32
            if ph or pw:
                dis = F.interpolate(dis, size=(h + ph, w + pw), mode="bicubic", align_corners=False)
                ref = F.interpolate(ref, size=(h + ph, w + pw), mode="bicubic", align_corners=False)
            _, fd = self.clip.encode_image(dis)
            _, fr = self.clip.encode_image(ref)
            nd, nr = self.snr(self.q(dis, fd)), self.snr(self.q(ref, fr))
            fid = self.sfr(self.dh(dis, ref, fd, fr))
            allv = self.ad(nd, nr, fid)
            for k, v in (("afine_all_scale", self.scale_final(allv)), ("afine_all", allv), ("fidelity", fid), ("natural_dis", nd), ("natural_ref", nr)):
                res[k].append(v.flatten().float().cpu())
        return {k: torch.cat(v).numpy() for k, v in res.items()}


def frechet(a, b):
    """두 임베딩 집합(행=표본)의 Fréchet distance ‖μ1−μ2‖² + tr(Σ1+Σ2−2(Σ1^½ Σ2 Σ1^½)^½).
    scipy sqrtm은 거의 특이한 공분산에서 수치가 깨지므로 대칭 고윳값 분해로 계산한다(음의 고윳값은 0으로 자름). float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    w, v = np.linalg.eigh(s1)
    r1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = r1 @ s2 @ r1
    tr_cs = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_cs)


def kid(a, b, n_subsets=100, subset=1000, seed=42):
    """KID: 3차 다항 커널 (x·y/d + 1)^3 의 비편향 MMD², 부분집합 평균(Bińkowski et al. 2018)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a.shape[1]
    m = min(subset, len(a), len(b))
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))
