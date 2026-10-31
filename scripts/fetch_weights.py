"""가중치 확보와 sha256 기록. 실패는 failures.csv에 남기고 해당 방법만 건너뛴다.

- PiSA-SR, OSDHuman: 각 README의 공식 Google Drive 폴더
- ResNet-50: torchvision 공식 URL
- SD2.1-base: 로컬 사본 사용. HF 공식 저장소 메타데이터(LFS sha256)만 조회해 대조한다.
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.util import load_cfg, log_failure, sha256_file, write_csv  # noqa: E402

W = ROOT / "weights"
SHARED = Path("/workspace/weights")
PISA_DRIVE = "https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss"
OSD_DRIVE = "https://drive.google.com/drive/folders/1sNoSNcJC7cdf0RR5dK87PVTEupjOZ-3B"
RESNET = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
SD_API = "https://huggingface.co/api/models/stabilityai/stable-diffusion-2-1-base/tree/main?recursive=true"
SD_LFS = ["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors",
          "vae/diffusion_pytorch_model.safetensors"]


def link(name, target):
    dst = W / name
    if not dst.exists():
        dst.symlink_to(target)


def drive_folder(url, outdir, required):
    outdir.mkdir(parents=True, exist_ok=True)
    if not all(any(outdir.rglob(r)) for r in required):
        subprocess.call([sys.executable, "-m", "gdown", "--folder", url, "-O", str(outdir)])
    return [r for r in required if not any(outdir.rglob(r))]


def verify_sd21():
    out = {"repo": "stabilityai/stable-diffusion-2-1-base", "files": {}}
    try:
        with urllib.request.urlopen(SD_API, timeout=30) as r:
            tree = json.load(r)
        remote = {e["path"]: e["lfs"]["oid"] for e in tree if e.get("lfs")}
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        remote = None
        out["query_error"] = str(e)
    for f in SD_LFS:
        local = sha256_file(W / "sd21_base" / f)
        out["files"][f] = {"local_sha256": local, "official_sha256": remote.get(f) if remote else None}
    if remote is None:
        out["status"] = "미검증"
    elif all(v["local_sha256"] == v["official_sha256"] for v in out["files"].values()):
        out["status"] = "공식본과 동일"
    else:
        out["status"] = "공식본과 불일치"
    (W / "sd21_verification.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("SD2.1:", out["status"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_cfg(ap.parse_args().config)
    W.mkdir(exist_ok=True)
    link("sd21_base", SHARED / "sd21_base")
    link("ram_swin_large_14m.pth", SHARED / "osediff" / "ram_swin_large_14m.pth")
    link("RealESRGAN_x4plus.pth", SHARED / "realesrgan" / "RealESRGAN_x4plus.pth")
    link("bert-base-uncased", SHARED / "bert-base-uncased")
    verify_sd21()

    resnet = W / "resnet50-0676ba61.pth"
    if not resnet.exists():
        urllib.request.urlretrieve(RESNET, str(resnet) + ".part")
        Path(str(resnet) + ".part").rename(resnet)

    for name, url, out, req in (("pisa_sr", PISA_DRIVE, W / "pisa_sr", ["pisa_sr.pkl"]),
                                ("osdhuman", OSD_DRIVE, W / "osdhuman",
                                 ["model.pkl", "model_embedding_change.pkl"])):
        missing = drive_folder(url, out, req)
        if missing:
            log_failure(cfg, "weights", name, "", f"Google Drive 다운로드 실패, 없는 파일: {missing}")
            print(f"{name} 가중치 없음: {missing}", flush=True)

    # pyiqa LPIPS 가중치를 온라인일 때 미리 캐시한다. 이미지는 전송하지 않는다.
    import pyiqa
    pyiqa.create_metric("lpips", device="cpu")

    rows = [{"file": str(p.relative_to(W)), "sha256": sha256_file(p)}
            for p in sorted(W.rglob("*"))
            if p.is_file() and p.suffix in {".pth", ".pkl"}]
    rows += [{"file": f"sd21_base/{f}", "sha256": sha256_file(W / "sd21_base" / f)} for f in SD_LFS]
    write_csv(W / "SHA256.csv", rows, ["file", "sha256"])


if __name__ == "__main__":
    main()
