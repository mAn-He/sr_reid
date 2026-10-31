"""Market-1501 확보: 지정 링크 다운로드 → 실패 시 incoming 대기(최대 7일) → sha256 기록 → 압축 해제 → test 잠금.

test 폴더를 data/locked로 옮기는 유일한 단계라 guard를 설치하지 않는다. 잠근 뒤로는 읽지 않는다.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.guard import LOCKED_NAMES  # noqa: E402
from common.util import EXIT_FATAL, sha256_file  # noqa: E402

INCOMING = ROOT / "data" / "incoming"
ZIP = INCOMING / "Market-1501-v15.09.15.zip"
URL = "https://drive.google.com/uc?id=0B8-rUzbwVRk0c054eEozWG9COHM"
URL_FUZZY = "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ"
WAIT_DAYS = 7
STABLE_S = 120


def valid_zip(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and any("bounding_box_train/" in n for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def download():
    for args in (["gdown", URL, "-O", str(ZIP)], ["gdown", "--fuzzy", URL_FUZZY, "-O", str(ZIP)]):
        print("다운로드 시도:", args[1:], flush=True)
        rc = subprocess.call([sys.executable, "-m", *args])
        if rc == 0 and ZIP.exists() and valid_zip(ZIP):
            return True
        print(f"실패 rc={rc}", flush=True)
        ZIP.unlink(missing_ok=True)
    return False


def wait_incoming():
    print(f"incoming 대기 모드: {INCOMING} 에 zip을 올리면 자동 시작 (최대 {WAIT_DAYS}일)", flush=True)
    t0 = time.time()
    while time.time() - t0 < WAIT_DAYS * 86400:
        for z in sorted(INCOMING.glob("*.zip")):
            s0 = z.stat().st_size
            time.sleep(STABLE_S)
            if z.stat().st_size == s0 and valid_zip(z):
                return z
        time.sleep(60)
    return None


def extract(zip_path):
    market, locked = ROOT / "data" / "market", ROOT / "data" / "locked"
    if market.exists() or locked.exists():  # 해제 도중 끊긴 경우에만 지운다. 완료되면 이 단계는 done 표식으로 건너뛴다.
        for d in (market, locked):
            if d.exists():
                subprocess.check_call(["chmod", "-R", "u+rwx", str(d)])
                shutil.rmtree(d)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or parts[0] == "__MACOSX" or len(parts) < 2:
                continue
            base = locked if parts[1] in LOCKED_NAMES else market
            dst = base.joinpath(*parts[1:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
    missing = [n for n in LOCKED_NAMES if not (locked / n).is_dir()]
    if missing or not (market / "bounding_box_train").is_dir():
        sys.exit(f"zip 구조가 예상과 다릅니다. 없는 폴더: {missing}")
    for dirpath, dirnames, filenames in os.walk(locked, topdown=False):
        for n in filenames + dirnames:
            os.chmod(os.path.join(dirpath, n), 0)
    os.chmod(locked, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    INCOMING.mkdir(parents=True, exist_ok=True)
    existing = [z for z in sorted(INCOMING.glob("*.zip")) if valid_zip(z)]
    zip_path = existing[0] if existing else (ZIP if download() else None)
    if zip_path is None:
        if args.download_only:
            sys.exit(EXIT_FATAL)
        zip_path = wait_incoming()
    if zip_path is None:
        print("7일 안에 데이터가 도착하지 않았습니다.", flush=True)
        sys.exit(EXIT_FATAL)
    digest = sha256_file(zip_path)
    (ROOT / "data" / "market_zip.sha256").write_text(f"{digest}  {zip_path.name}\n")
    print("zip sha256:", digest, flush=True)
    if not args.download_only:
        extract(zip_path)
        print("압축 해제 및 test 잠금 완료", flush=True)


if __name__ == "__main__":
    main()
