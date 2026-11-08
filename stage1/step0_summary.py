"""Stage 1 0단계 SR 쪽 요약: 가중치 체크섬, 24조합 로드 확인표, 재현성표. srreid-core env.
출력: results/stage1/step0_sr_weights.csv, results/stage1/step0_sr_load.csv, results/stage1/step0_repro.csv"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S0 = ROOT / "runs" / "stage1_step0"
OUT = ROOT / "results" / "stage1"
W = ROOT / "weights"
W5 = W / "stage05" / "sr"
WEIGHTS = {
    "realesrgan": [W / "RealESRGAN_x4plus.pth"],
    "osdhuman": [W / "osdhuman" / "model.pkl", W / "osdhuman" / "model_embedding_change.pkl", W / "ram_swin_large_14m.pth",
                 W / "sd21_base" / "unet" / "diffusion_pytorch_model.safetensors", W / "sd21_base" / "vae" / "diffusion_pytorch_model.safetensors",
                 W / "sd21_base" / "text_encoder" / "model.safetensors"],
    "pisa_sr": [W / "pisa_sr" / "pisa_sr.pkl", W / "sd21_base" / "unet" / "diffusion_pytorch_model.safetensors",
                W / "sd21_base" / "vae" / "diffusion_pytorch_model.safetensors", W / "sd21_base" / "text_encoder" / "model.safetensors"],
    "swinir": [W5 / "001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth", W5 / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"],
    "hat": [W5 / "HAT_SRx3_ImageNet-pretrain.pth", W5 / "HAT_SRx4_ImageNet-pretrain.pth"],
    "mambairv2": [W5 / "mambairv2_classicSR_Base_x3.pth", W5 / "mambairv2_classicSR_Base_x4.pth"],
}
METHODS = ["bicubic", "lanczos", "realesrgan", "osdhuman", "swinir", "hat", "mambairv2"] + [f"pisa_sr_sem{s}" for s in ("0.0", "0.5", "1.0", "1.5", "2.0")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def weights():
    known = {}
    reg = W / "SHA256.csv"
    if reg.exists():
        known = {r["file"]: r["sha256"] for r in csv.DictReader(open(reg))}
    rows, seen = [], {}
    for m, ps in WEIGHTS.items():
        for p in ps:
            rp = p.resolve()
            if rp not in seen:
                seen[rp] = sha256(rp)
            rel = str(p.relative_to(W))
            rows.append({"method": m, "path": str(p), "resolved": str(rp), "bytes": rp.stat().st_size, "sha256": seen[rp],
                         "registry_sha256_match": ("" if rel not in known else str(known[rel] == seen[rp]))})
    return rows


def load_table():
    rows = []
    for m in METHODS:
        for k in (3, 4):
            f = S0 / "load" / "_logs" / f"{m}__x{k}__summary.json"
            if not f.exists():
                rows.append({"method": m, "k": k, "status": "missing"})
                continue
            d = json.loads(f.read_text())
            r = {"method": m, "k": k, "status": "ok" if d["n_done"] == d["n"] else "partial", "n": d["n_done"], "path": "derived_from_x4" if d["from_x4"] else ("cpu_pil" if m in ("bicubic", "lanczos") else "native"),
                 "batch": d.get("batch"), "load_s": d.get("load_seconds"), "infer_s_median": d.get("infer_s_per_img_median"),
                 "img_s_excl_first": (1 / d["infer_s_per_img_median_excl_first"]) if d.get("infer_s_per_img_median_excl_first") else None,
                 "peak_alloc_gib": d.get("peak_alloc_gib"), "peak_reserved_gib": d.get("peak_reserved_gib"), "wall_s": d["wall"], "server": d["server"]}
            if m in ("bicubic", "lanczos") or d["from_x4"]:
                r["img_s_excl_first"] = d["n_done"] / d["wall"] if d["wall"] else None
            rows.append(r)
    return rows


def sha_map(p):
    return {json.loads(l)["name"]: json.loads(l)["sha1"] for l in open(p)} if p.exists() else {}


def repro_table():
    rows = []
    L = S0 / "repro" / "_logs"
    for m, k in [("mambairv2", 3), ("mambairv2", 4), ("hat", 3), ("hat", 4), ("osdhuman", 4)] + [(f"pisa_sr_sem{x}", 4) for x in ("0.0", "0.5", "1.0", "1.5", "2.0")]:
        for cfg in ("off", "det", "offw", "detw", "strictw"):
            if not (L / f"{m}__x{k}__{cfg}__fwd.jsonl").exists():  # A100: detw만 측정
                continue
            a, b = sha_map(L / f"{m}__x{k}__{cfg}__fwd.jsonl"), sha_map(L / f"{m}__x{k}__{cfg}__rev.jsonl")
            sa = L / f"{m}__x{k}__{cfg}__fwd__summary.json"
            if not a or not b or not sa.exists():
                rows.append({"method": m, "k": k, "config": cfg, "status": "missing"})
                continue
            s = json.loads(sa.read_text())
            common = sorted(set(a) & set(b))
            rows.append({"method": m, "k": k, "config": cfg, "status": "ok", "n": len(common), "n_bit_identical": sum(a[n] == b[n] for n in common),
                         "infer_s_median_excl_first": s.get("infer_s_per_img_median_excl_first"), "peak_alloc_gib": s.get("peak_alloc_gib"),
                         "cudnn_benchmark": s.get("cudnn_benchmark"), "sdpa": json.dumps(s.get("sdpa")),
                         "diff_names": " ".join(n for n in common if a[n] != b[n]),
                         "warmup_first_vs_second": json.dumps(s.get("warmup")) if s.get("warmup") else ""})
    return rows


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [x for x in r if x not in keys]
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)}행")


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["weights", "load", "repro"]
    if "weights" in parts:
        write("step0_sr_weights.csv", weights())
    if "load" in parts:
        write("step0_sr_load.csv", load_table())
    if "repro" in parts:
        write("step0_repro.csv", repro_table())
