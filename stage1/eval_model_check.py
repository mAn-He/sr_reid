"""Stage 1 평가 모델 적재 점검 (지표 계산 아님).

DINOv3(S/B/L, timm 사본), PersonViT(S/B, LUPerson 자기지도 사전학습), PLIP(RN50 영상 인코더), A-FINE을
로컬 가중치로 적재하고 표본 5장(원본 vs SR)에 돌려 파이프가 동작하는지만 본다.
- 원본: data/sr_input/bounding_box_train/<stem>.png (128x64)
- SR: runs/stage05/sr_smoke/bicubic/x4/... 에 이름이 있으면 그 WebP, 없으면 원본의 PIL BICUBIC x4 (메모리)
- 임베딩 모델: 양쪽 모두 PIL BICUBIC으로 256x128 리사이즈(축소 시 PIL 자체 안티에일리어싱) 후 각 모델 정규화.
- A-FINE: dis=SR, ref=원본을 SR 크기로 PIL BICUBIC 확대. 공식 QuickInference/afine.py 계산 순서를 그대로 따른다.
third_party 소스는 수정하지 않고 sys.path로 불러온다.
실행: HF_HUB_OFFLINE=1 flock /root/srreid_gpu.lock python stage1/eval_model_check.py
"""
import csv, hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/workspace/sr-reid")
W = ROOT / "weights/stage1"
TP = ROOT / "third_party"
OUT_JSON = ROOT / "results/stage1/eval_model_check.json"
OUT_CSV = ROOT / "results/stage1/eval_model_check.csv"
SERVER = "새 서버 RTX 4090"
EMB_HW = (256, 128)
DEV = torch.device("cuda")


def load_pairs():
    names = []
    with open(ROOT / "configs/stage1_sample.csv") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= 5:
                break
            names.append(r["name"])
    pairs = []
    for n in names:
        stem = Path(n).stem
        orig = Image.open(ROOT / f"data/sr_input/bounding_box_train/{stem}.png").convert("RGB")
        smoke = ROOT / f"runs/stage05/sr_smoke/bicubic/x4/bounding_box_train/{stem}.webp"
        if smoke.exists():
            sr, src = Image.open(smoke).convert("RGB"), "smoke_webp"
        else:
            sr, src = orig.resize((orig.width * 4, orig.height * 4), Image.BICUBIC), "pil_bicubic_x4_mem"
        pairs.append(dict(name=n, orig=orig, sr=sr, sr_src=src, orig_size=orig.size, sr_size=sr.size))
    return pairs


def to_tensor(imgs, hw, mean, std):
    arr = [np.asarray(im.resize((hw[1], hw[0]), Image.BICUBIC) if im.size != (hw[1], hw[0]) else im) for im in imgs]
    x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float().div(255.0)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def bench(fn, x5, bs=64, iters=20, warm=3):
    """batch 5 실측 VRAM, batch 64 실측 VRAM, batch 64 처리량(같은 5장 반복)."""
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        fn(x5)
    torch.cuda.synchronize()
    v5 = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    reps = math.ceil(bs / x5[0].shape[0]) if isinstance(x5, tuple) else math.ceil(bs / x5.shape[0])
    if isinstance(x5, tuple):
        x64 = tuple(t.repeat(reps, 1, 1, 1)[:bs] for t in x5)
    else:
        x64 = x5.repeat(reps, 1, 1, 1)[:bs]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warm):
            fn(x64)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x64)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    v64 = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    g = lambda b: round(b / 2**20, 1)
    return dict(weights_alloc_mib=g(base), peak_alloc_mib_bs5=g(v5[0]), peak_reserved_mib_bs5=g(v5[1]),
                peak_alloc_mib_bs64=g(v64[0]), peak_reserved_mib_bs64=g(v64[1]),
                img_per_s_bs64=round(bs * iters / dt, 1), precision="fp32", server=SERVER)


def cos_pairs(fe, xo, xs):
    with torch.no_grad():
        a, b = fe(xo).float(), fe(xs).float()
    return [round(float(v), 6) for v in F.cosine_similarity(a, b, dim=1).cpu()]


# ---------------- DINOv3 (timm) ----------------
def run_dinov3(pairs, size):
    import timm
    name = f"vit_{size}_patch16_dinov3.lvd1689m"
    f = W / f"dinov3/{name}/model.safetensors"
    m = timm.create_model(name, pretrained=True, pretrained_cfg_overlay=dict(file=str(f)), img_size=EMB_HW).eval().to(DEV)
    cfg = m.pretrained_cfg
    mean, std = cfg["mean"], cfg["std"]
    fe = lambda x: m.forward_features(x)[:, 0]  # 최종 norm 뒤 CLS 토큰 (HF Dinov3Model pooler_output과 같은 정의)
    xo = to_tensor([p["orig"] for p in pairs], EMB_HW, mean, std).to(DEV)
    xs = to_tensor([p["sr"] for p in pairs], EMB_HW, mean, std).to(DEV)
    with torch.no_grad():
        d = fe(xo).shape[1]
    r = dict(model=f"dinov3_{size}", timm_name=name, weight=str(f), params=n_params(m), embed_dim=int(d),
             num_prefix_tokens=int(getattr(m, "num_prefix_tokens", -1)), input_hw=list(EMB_HW),
             preprocess=f"PIL BICUBIC -> {EMB_HW}, mean={mean}, std={std}", pooling="CLS after final norm",
             cos_orig_sr=cos_pairs(fe, xo, xs))
    r.update(bench(fe, xo))
    del m; torch.cuda.empty_cache()
    return r


# ---------------- PersonViT ----------------
def run_personvit(pairs, size):
    sys.path.insert(0, str(TP / "PersonViT/transreid_pytorch/model/backbones"))
    import vit_pytorch as vp
    ck = {"small": W / "personvit/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0220.pth",
          "base": W / "personvit/vitb.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0260.pth"}[size]
    fac = {"small": vp.vit_small_patch16_224_TransReID, "base": vp.vit_base_patch16_224_TransReID}[size]
    m = fac(img_size=EMB_HW, stride_size=16, drop_path_rate=0.0, camera=0, view=0, local_feature=False)
    # 공식 load_param과 같은 변환(teacher.backbone.*, head 제외)을 직접 하고 누락 키를 기록한다.
    raw = torch.load(ck, map_location="cpu")
    sd = {k[len("backbone."):]: v for k, v in raw["teacher"].items() if k.startswith("backbone.")}
    head_keys = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in raw["teacher"] if not k.startswith("backbone.")})
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m = m.eval().to(DEV)
    mean = std = [0.5, 0.5, 0.5]  # 공식 ReID 미세조정 config(configs/market/vit_*.yml)의 PIXEL_MEAN/STD
    fe = lambda x: m(x)  # pool_method=None -> 최종 norm 뒤 CLS
    xo = to_tensor([p["orig"] for p in pairs], EMB_HW, mean, std).to(DEV)
    xs = to_tensor([p["sr"] for p in pairs], EMB_HW, mean, std).to(DEV)
    with torch.no_grad():
        d = fe(xo).shape[1]
    r = dict(model=f"personvit_{size}", weight=str(ck), ckpt_epoch=int(raw["epoch"]), ckpt_top_keys=list(raw.keys()),
             ckpt_args_subset={k: getattr(raw["args"], k) for k in ("arch", "image_size", "epochs", "out_dim", "use_masked_im_modeling", "cfs_idx_file", "pretrained_file", "load_from")},
             teacher_non_backbone_modules=head_keys,
             load_missing=list(missing), load_unexpected=list(unexpected), loaded_tensors=len(sd) - len(unexpected),
             params_excl_unused_fc=n_params(m) - n_params(m.fc),
             params=n_params(m), embed_dim=int(d), input_hw=list(EMB_HW),
             preprocess=f"PIL BICUBIC -> {EMB_HW}, mean={mean}, std={std}", pooling="CLS after final norm",
             cos_orig_sr=cos_pairs(fe, xo, xs))
    r.update(bench(fe, xo))
    del m, raw; torch.cuda.empty_cache()
    sys.path.pop(0)
    return r


# ---------------- PLIP ----------------
def run_plip(pairs):
    sys.path.insert(0, str(TP / "PLIP"))
    import visual_model as vm
    ck = W / "plip/PLIP_RN50.pth.tar"
    raw = torch.load(ck, map_location="cpu")
    # zs_infer.py와 같은 인자: layers [3,4,6,3], img_dim 768, heads 8, input_resolution=[width, height]
    m = vm.Image_encoder_ModifiedResNet([3, 4, 6, 3], 768, 8, input_resolution=[EMB_HW[1], EMB_HW[0]])
    res = m.load_state_dict(raw["ImgEncoder_state_dict"], strict=True)
    m = m.eval().to(DEV)
    mean, std = [0.357, 0.323, 0.328], [0.252, 0.242, 0.239]
    fe = lambda x: m(x)[0]
    xo = to_tensor([p["orig"] for p in pairs], EMB_HW, mean, std).to(DEV)
    xs = to_tensor([p["sr"] for p in pairs], EMB_HW, mean, std).to(DEV)
    with torch.no_grad():
        d = fe(xo).shape[1]
    r = dict(model="plip_rn50", weight=str(ck), ckpt_epoch=int(raw["epoch"]), ckpt_top_keys=list(raw.keys()),
             load_strict=str(res), params=n_params(m), embed_dim=int(d), input_hw=list(EMB_HW),
             preprocess=f"PIL BICUBIC -> {EMB_HW}, mean={mean}, std={std} (zs_infer.py)", pooling="AttentionPool2d 출력(global_image_out)",
             cos_orig_sr=cos_pairs(fe, xo, xs))
    r.update(bench(fe, xo))
    del m, raw; torch.cuda.empty_cache()
    sys.path.pop(0)
    return r


# ---------------- A-FINE ----------------
def run_afine(pairs):
    sys.path.insert(0, str(TP / "AFINE/QuickInference"))
    from utils.network_archs import AFINEQhead, AFINEDhead, AFINENLM_NR_Fit, AFINENLM_FR_Fit_with_limit, AFINELearnLambda
    from CLIP_ReturnFea import clip
    ap = W / "afine/pyiqa_mirror/afine.pth"
    sd = torch.load(ap, map_location="cpu")
    clip_model, _ = clip.load(str(W / "afine/ViT-B-32.pt"), device="cpu", jit=False)
    clip_model.load_state_dict(sd["finetuned_clip"])
    q = AFINEQhead(); q.load_state_dict(sd["natural"], strict=True)
    dh = AFINEDhead(); dh.load_state_dict(sd["fidelity"], strict=True)
    # 공식 NLM 모듈은 `if exp_pow >= 10` 때문에 batch 1만 된다. 같은 식을 원소별 torch.where로 쓴 서브클래스를 쓴다.
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
    sfr_o = AFINENLM_FR_Fit_with_limit(yita1=2, yita2=-2, yita3=0.5, yita4=0.15, yita3_upper=0.95, yita3_lower=0.05, yita4_upper=0.70, yita4_lower=0.01)
    sfr_o.load_state_dict(sd["fidelity_scale"], strict=True)
    snr_o = AFINENLM_NR_Fit(yita1=2, yita2=-2, yita3=4.9592, yita4=21.5968); snr_o.load_state_dict(sd["natural_scale"], strict=True)
    snr = BatchNR(yita1=2, yita2=-2, yita3=4.9592, yita4=21.5968); snr.load_state_dict(sd["natural_scale"], strict=True)
    ad = AFINELearnLambda(k=5); ad.load_state_dict(sd["adapter"], strict=True)
    mods = [clip_model, q, dh, sfr, snr, ad, sfr_o, snr_o]
    for mm in mods:
        mm.eval().to(DEV)
    mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    def scale_final(s, y1=100, y2=0, y3=-1.9710, y4=-2.3734):  # afine.py scale_finalscore의 원소별 버전
        e = -1 * (s - y3) / (abs(y4) + 1e-10)
        return torch.where(e >= 10, (y1 - y2) * torch.exp(-e) / (1 + torch.exp(-e)) + y2, (y1 - y2) / (1 + torch.exp(e)) + y2)

    def score(xx):
        dis, ref = xx
        _, _, h, w = dis.shape
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            dis = F.interpolate(dis, size=(h + ph, w + pw), mode="bicubic", align_corners=False)
            ref = F.interpolate(ref, size=(h + ph, w + pw), mode="bicubic", align_corners=False)
        _, fd = clip_model.encode_image(dis)
        _, fr = clip_model.encode_image(ref)
        nd, nr = snr(q(dis, fd)), snr(q(ref, fr))
        fid = sfr(dh(dis, ref, fd, fr))
        allv = ad(nd, nr, fid)
        return dict(afine_all_scale=scale_final(allv), afine_all=allv, fidelity=fid, natural_dis=nd, natural_ref=nr)

    sr_size = pairs[0]["sr"].size
    assert all(p["sr"].size == sr_size for p in pairs)
    hw = (sr_size[1], sr_size[0])
    dis = to_tensor([p["sr"] for p in pairs], hw, mean, std).to(DEV)
    ref = to_tensor([p["orig"].resize(sr_size, Image.BICUBIC) for p in pairs], hw, mean, std).to(DEV)
    with torch.no_grad():
        out = score((dis, ref))
        out_self = score((ref, ref))  # 참고: dis=ref 일 때
        # 참고: 표본 SR이 bicubic이라 dis==ref가 된다. 반응 확인용으로 dis=원본 PIL LANCZOS 확대본(메모리)도 계산한다.
        dis_lz = to_tensor([p["orig"].resize(sr_size, Image.LANCZOS) for p in pairs], hw, mean, std).to(DEV)
        out_lz = score((dis_lz, ref))
        # 등가성: 공식 모듈(batch 1)과 공식 scale_finalscore 식으로 한 장씩 계산해 batch 결과와 비교
        diffs = []
        for i in range(dis.shape[0]):
            d1, r1 = dis[i:i + 1], ref[i:i + 1]
            _, fd1 = clip_model.encode_image(d1); _, fr1 = clip_model.encode_image(r1)
            nd1, nr1 = snr_o(q(d1, fd1)), snr_o(q(r1, fr1))
            f1 = sfr_o(dh(d1, r1, fd1, fr1)); a1 = ad(nd1, nr1, f1)
            ep = -1 * (a1 - (-1.9710)) / (abs(-2.3734) + 1e-10)
            s1 = 100 * torch.exp(-ep) / (1 + torch.exp(-ep)) if ep >= 10 else 100 / (1 + torch.exp(ep))
            diffs.append(max(abs(float(s1) - float(out["afine_all_scale"].flatten()[i])), abs(float(a1) - float(out["afine_all"].flatten()[i])),
                             abs(float(f1) - float(out["fidelity"].flatten()[i])), abs(float(nd1) - float(out["natural_dis"].flatten()[i]))))
    per = {k: [round(float(v), 6) for v in t.flatten().cpu()] for k, t in out.items()}
    per_self = {k: [round(float(v), 6) for v in t.flatten().cpu()] for k, t in out_self.items()}
    r = dict(model="afine", weight=str(ap), clip_weight=str(W / "afine/ViT-B-32.pt"), afine_ckpt_keys=list(sd.keys()),
             params=sum(n_params(mm) for mm in mods[:6]), params_clip=n_params(clip_model), input_hw=list(hw),
             preprocess=f"dis=SR 원 크기, ref=원본 PIL BICUBIC -> {hw}; /255, CLIP mean={mean}, std={std}; 32배수 아니면 bicubic 보간",
             scores_dis_lanczos_ref_bicubic={k: [round(float(v), 6) for v in t.flatten().cpu()] for k, t in out_lz.items()},
             max_abs_dis_minus_ref=float((dis - ref).abs().max()),
             output_fields=list(per.keys()), batch_vs_official_bs1_maxabs=max(diffs), scores=per, scores_dis_eq_ref=per_self,
             note="모든 항목 낮을수록 좋음(공식 README). afine_all_scale in (0,100).")
    r.update(bench(lambda xx: score(xx), (dis, ref)))
    for mm in mods:
        mm.cpu()
    torch.cuda.empty_cache()
    sys.path.pop(0)
    return r


def main():
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE=1 필요"
    torch.backends.cudnn.benchmark = False
    pairs = load_pairs()
    import timm, torchvision
    info = dict(server=SERVER, gpu=torch.cuda.get_device_name(0), python=sys.version.split()[0], torch=torch.__version__,
                cuda=torch.version.cuda, timm=timm.__version__, torchvision=torchvision.__version__,
                env=sys.prefix, date=time.strftime("%Y-%m-%d %H:%M:%S"),
                samples=[dict(name=p["name"], sr_src=p["sr_src"], orig_size=p["orig_size"], sr_size=p["sr_size"]) for p in pairs])
    results = []
    jobs = [("dinov3_small", lambda: run_dinov3(pairs, "small")), ("dinov3_base", lambda: run_dinov3(pairs, "base")),
            ("dinov3_large", lambda: run_dinov3(pairs, "large")), ("personvit_small", lambda: run_personvit(pairs, "small")),
            ("personvit_base", lambda: run_personvit(pairs, "base")), ("plip_rn50", lambda: run_plip(pairs)),
            ("afine", lambda: run_afine(pairs))]
    for name, job in jobs:
        try:
            r = job(); r["ok"] = True
        except Exception as e:  # 실패도 기록
            import traceback
            r = dict(model=name, ok=False, error=repr(e), tb=traceback.format_exc())
        print(json.dumps({k: v for k, v in r.items() if k not in ("tb",)}, ensure_ascii=False, default=str), flush=True)
        results.append(r)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(info=info, results=results), ensure_ascii=False, indent=1, default=str))
    cols = ["model", "ok", "params", "embed_dim", "input_hw", "peak_alloc_mib_bs5", "peak_reserved_mib_bs5",
            "peak_alloc_mib_bs64", "peak_reserved_mib_bs64", "img_per_s_bs64", "precision", "server", "weight"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([json.dumps(r.get(c)) if isinstance(r.get(c), list) else r.get(c, "") for c in cols])


if __name__ == "__main__":
    main()
