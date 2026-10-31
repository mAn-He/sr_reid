"""Stage 0 가중치 확보, sha256과 commit, 라이선스 기록. 끝난 파일은 건너뛴다. 실패는 stage0_failures.csv에 남긴다."""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.util import sha256_file, write_csv  # noqa: E402

W = ROOT / "weights" / "stage0"
CLIP_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
FILES = [
    ("ViT-B-16.pt", f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA}/ViT-B-16.pt", "CLIP-ReID model/clip/clip.py의 공식 URL"),
    ("SOLIDER/swin_base_patch4_window7_224_22k.pth",
     "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
     "Microsoft Swin 공식 릴리스. KPR GDrive에 없어 파일명으로 공식본을 받음"),
    ("SOLIDER/swin_base_reid.pth", "gdrive:1j9P2d_7ABjyItVaHwQHO87o_PlPtQ3_E", "KPR README 공식 GDrive"),
]
REPOS = {"CLIP-ReID": "MIT (third_party/CLIP-ReID/LICENSE)",
         "keypoint_promptable_reidentification": "Hippocratic License HL3-LAW-MEDIA-MIL-SOC-SV"}


def fetch(rel, url):
    dst = W / rel
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if url.startswith("gdrive:"):
        subprocess.check_call([sys.executable, "-m", "gdown", url[7:], "-O", str(tmp)])
    else:
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    tmp.replace(dst)


def main():
    rows, missing = [], []
    for rel, url, source in FILES:
        try:
            fetch(rel, url)
            rows.append({"file": rel, "sha256": sha256_file(W / rel), "source": source, "url": url})
        except Exception as e:  # 하나가 실패해도 나머지를 받고 실패로 기록한다.
            missing.append(f"{rel}: {e}")
            print("실패", rel, e, flush=True)
    write_csv(W / "SHA256.csv", rows, ["file", "sha256", "source", "url"])
    bad = [r["file"] for r in rows if r["file"] == "ViT-B-16.pt" and r["sha256"] != CLIP_SHA]
    missing += [f"{b}: URL의 sha256과 불일치" for b in bad]
    ann = ROOT / "data" / "kpr_annotations" / "kpr_market_annotations.zip"
    prov = {"commits": {n: subprocess.check_output(["git", "-C", str(ROOT / "third_party" / n), "rev-parse", "HEAD"], text=True).strip()
                        for n in REPOS},
            "licenses": REPOS, "solider_license": "tinyvision/SOLIDER Apache-2.0 (저장소 LICENSE 기준)",
            "kpr_annotation_zip": {"gdrive_id": "13ZPsAqKEHfSbL9luHZAZ6tALc_pFgjH7", "bytes": ann.stat().st_size},
            "missing": missing}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "stage0_provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(json.dumps(prov, ensure_ascii=False), flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
