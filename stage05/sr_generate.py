"""Part F. SR 이미지 생성. 외부 SR 저장소는 수정하지 않고 sr/runner.py로 공식 스크립트를 호출한다.
입력: 원본 128×64를 무손실 PNG로 푼 data/sr_input/<split> (test는 data/locked/sr_input/<split>).
출력: data/<method>/x<k>/<split>/<name>.webp (WebP 무손실, 저장 직후 픽셀 전수 비교). ×3 공식 가중치가 없으면 ×4 출력을 PIL BICUBIC으로 축소한다.
--smoke N: train N장만 runs/stage05/sr_smoke 아래로 생성해 시간·정렬 측정에만 쓴다. 재실행하면 끝난 파일을 건너뛴다."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import guard  # noqa: E402

guard.install()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MF = Path("/workspace/miniforge3/envs")
W = ROOT / "weights"
W5 = W / "stage05" / "sr"
TP = ROOT / "third_party"
RAW = Path(os.environ.get("SRREID_RAW", "/root/srreid_raw"))
SEED = 42
WEBP = {"format": "WEBP", "lossless": True, "quality": 100, "method": 2}  # 사용자 결정: method 6 대신 2 (용량 차 최대 1.2%, 픽셀 일치)
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SPLITS = ("bounding_box_train", "query", "bounding_box_test")
# scales: 공식 가중치가 있는 배율. 없는 ×3은 ×4 출력에서 PIL BICUBIC 축소.
METHODS = {"bicubic": {"kind": "classic", "scales": (3, 4)}, "lanczos": {"kind": "classic", "scales": (3, 4)},
           "realesrgan": {"kind": "realesrgan", "scales": (4,), "env": "srreid-realesrgan", "repo": "Real-ESRGAN"},
           "osdhuman": {"kind": "osd", "scales": (4,), "env": "srreid-osdhuman", "repo": "OSDHuman"},
           "swinir": {"kind": "swinir", "scales": (3, 4), "env": "srreid-mambair", "repo": "SwinIR"},
           "hat": {"kind": "hat", "scales": (3, 4), "env": "srreid-mambair", "repo": "HAT"},
           "mambairv2": {"kind": "mamba", "scales": (3, 4), "env": "srreid-mambair", "repo": "MambaIR"}}
for s in LAMBDAS:
    METHODS[f"pisa_sr_sem{s}"] = {"kind": "pisa", "scales": (4,), "env": "srreid-pisasr", "repo": "PiSA-SR", "lambda_sem": s}


def input_dir(split):
    base = ROOT / "data" / ("sr_input" if split == "bounding_box_train" else "locked/sr_input")
    return base / split


def source_dir(split):
    return ROOT / "data" / "market" / split if split == "bounding_box_train" else ROOT / "data" / "locked" / split


def _unpack(args):
    src, dst = args
    if not dst.exists():
        with Image.open(src) as im:
            tmp = dst.with_name(dst.name + ".tmp.png")
            im.convert("RGB").save(tmp, "PNG")
        os.replace(tmp, dst)


def prepare_input(split):
    d = input_dir(split)
    d.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in source_dir(split).iterdir() if p.suffix == ".jpg")
    with ProcessPoolExecutor(12) as ex:
        list(ex.map(_unpack, [(p, d / (p.stem + ".png")) for p in srcs], chunksize=64))
    return d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_webp(arr, dst):
    """저장 직후 다시 읽어 픽셀 단위로 비교한다. 불일치면 실패로 돌려준다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(arr).save(tmp, **WEBP)
    with Image.open(tmp) as im:
        same = np.array_equal(np.asarray(im.convert("RGB")), arr)
    if not same:
        tmp.unlink()
        return None
    os.replace(tmp, dst)
    return sha(dst)


def finalize_one(task):
    """task: raw 경로, 입력 경로, [(k, 출력 경로, 축소 여부)]. 크기가 다르면 고치지 않고 실패로 기록한다."""
    raw, inp, targets, h0, w0 = task
    t0 = time.perf_counter()
    rows = []
    try:
        with Image.open(raw) as im:
            im = im.convert("RGB")
            for k, dst, derived in targets:
                if derived:
                    if im.size != (w0 * 4, h0 * 4):
                        rows.append({"name": dst.name, "k": k, "status": f"wrong_size {im.size}"})
                        continue
                    arr = np.asarray(im.resize((w0 * k, h0 * k), Image.BICUBIC))
                else:
                    if im.size != (w0 * k, h0 * k):
                        rows.append({"name": dst.name, "k": k, "status": f"wrong_size {im.size}"})
                        continue
                    arr = np.asarray(im)
                digest = write_webp(arr, dst)
                rows.append({"name": dst.name, "k": k, "status": "ok" if digest else "pixel_mismatch", "input_sha256": sha(inp), "output_sha256": digest or "",
                             "bytes": dst.stat().st_size if digest else 0, "derived_from_x4": derived})
    except Exception as e:  # 실패는 결과로 남긴다.
        rows.append({"name": Path(raw).name, "k": 0, "status": f"error {type(e).__name__}: {e}"})
    return rows, time.perf_counter() - t0


def out_dir(root, method, k, split):
    return Path(root) / method / f"x{k}" / split


NO_DERIVE = False  # 일괄 생성에서는 ×4만 만들고, ×3은 Stage 1 상위 방법에 한해 --derive-x3로 저장본에서 만든다.


def targets_for(method, k_run, name, root, split):
    """이 실행의 raw 하나로 만들 출력들. 스모크는 k_run=4인 가중치 전용 방법의 ×3도 같이 만들어 용량을 잰다."""
    m = METHODS[method]
    out = [(k_run, out_dir(root, method, k_run, split) / f"{name}.webp", False)]
    if k_run == 4 and 3 not in m["scales"] and not NO_DERIVE:
        out.append((3, out_dir(root, method, 3, split) / f"{name}.webp", True))
    return out


def external_cmd(method, k, in_dir, raw, work):
    m = METHODS[method]
    repo = TP / m["repo"]
    kind = m["kind"]
    if kind == "realesrgan":
        script, args = repo / "inference_realesrgan.py", ["-n", "RealESRGAN_x4plus", "-i", in_dir, "-o", raw, "-s", "4", "--suffix", "", "--ext", "png", "--model_path", W / "RealESRGAN_x4plus.pth"]
    elif kind == "pisa":
        script = repo / "test_pisasr.py"
        args = ["--pretrained_model_path", W / "sd21_base", "--pretrained_path", W / "pisa_sr" / "pisa_sr.pkl", "--process_size", 512, "--upscale", 4,
                "--input_image", in_dir, "--output_dir", raw, "--lambda_pix", 1.0, "--lambda_sem", m["lambda_sem"], "--seed", SEED]
    elif kind == "osd":
        script = repo / "run_inference.py"
        args = ["-i", in_dir, "-o", raw, "--osediff_path", W / "osdhuman" / "model.pkl", "--embedding_change_path", W / "osdhuman" / "model_embedding_change.pkl",
                "--pretrained_model_name_or_path", W / "sd21_base", "--ram_path", W / "ram_swin_large_14m.pth", "--ram_img_only", "--process_size", 512, "--upscale", 4, "--seed", SEED]
    elif kind == "swinir":
        # 공식 classical_sr 경로는 GT 폴더를 기준으로 돈다. GT는 쓰지 않으므로(지표 무시) LQ를 ×k bicubic으로 키운 임시본을 넣는다.
        gt, lq = work / "gt", work / "lq"
        gt.mkdir(parents=True, exist_ok=True)
        lq.mkdir(parents=True, exist_ok=True)
        for p in sorted(Path(in_dir).iterdir()):
            with Image.open(p) as im:
                im.resize((im.width * k, im.height * k), Image.BICUBIC).save(gt / p.name)
            (lq / f"{p.stem}x{k}{p.suffix}").symlink_to(p)
        script = repo / "main_test_swinir.py"
        args = ["--task", "classical_sr", "--scale", k, "--training_patch_size", 64, "--model_path", W5 / f"001_classicalSR_DF2K_s64w8_SwinIR-M_x{k}.pth", "--folder_lq", lq, "--folder_gt", gt]
    else:
        tag = "hat" if kind == "hat" else "mamba"
        opt = work / f"{tag}_x{k}.yml"
        name = f"{tag}_x{k}"
        if kind == "hat":
            base = repo / "options" / "test" / f"HAT_SRx{k}_ImageNet-pretrain.yml"
            weight = W5 / f"HAT_SRx{k}_ImageNet-pretrain.pth"
            # HAT는 basicsr 1.3.4.9를 전제로 rgb2ycbcr를 옛 위치에서 가져온다. 저장소를 고치지 않고 실행 스크립트에서 옛 이름만 잇는다. 추론 경로에서는 호출되지 않는다.
            imports = "import basicsr.utils.matlab_functions as mf, basicsr.utils.color_util as cu\nmf.rgb2ycbcr = cu.rgb2ycbcr\nimport hat.archs, hat.data, hat.models\nfrom basicsr.test import test_pipeline\n"
        else:
            base = repo / "options" / "test" / "mambairv2" / f"test_MambaIRv2_SR_x{k}.yml"
            weight = W5 / f"mambairv2_classicSR_Base_x{k}.pth"
            imports = "from basicsr.test import test_pipeline\n"
        import yaml
        o = yaml.safe_load(open(base))
        o["name"] = name
        o["num_gpu"] = 1
        o["manual_seed"] = SEED
        o["datasets"] = {"test_1": {"name": "custom", "type": "SingleImageDataset", "dataroot_lq": str(in_dir), "io_backend": {"type": "disk"}}}
        o["path"]["pretrain_network_g"] = str(weight)
        o["path"]["strict_load_g"] = True
        o["val"] = {"save_img": True, "suffix": "x"}
        opt.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(o, open(opt, "w"))
        script = work / f"run_{tag}.py"
        script.write_text(f"{imports}test_pipeline(r'{work}')\n")  # root_path만 작업 폴더로 바꾼다. 결과가 third_party 밖에 쓰인다.
        args = ["-opt", opt]
    return m["env"], repo, script, [str(x) for x in args]


def raw_output_name(method, path):
    """외부 스크립트 출력 파일명을 입력 stem으로 되돌린다."""
    stem = Path(path).stem
    for suf in ("_SwinIR", "_x"):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
    return stem


def raw_search_dirs(method, work, raw):
    kind = METHODS[method]["kind"]
    if kind == "swinir":
        return work / "results"
    if kind in ("hat", "mamba"):
        return work / "results"
    return raw


def generate(method, k, split, root, smoke, manifest_dir):
    """외부 스크립트가 쓰는 raw를 지켜보다 바로 WebP로 바꾼다. 인코딩이 GPU보다 느릴 수 있어 큐가 쌓여도 raw는 로컬 NVMe에만 둔다."""
    m = METHODS[method]
    inp = input_dir(split)
    names = sorted(p.stem for p in inp.iterdir() if p.suffix == ".png")
    if smoke:
        names = [names[i] for i in np.linspace(0, len(names) - 1, smoke).astype(int)]
    with Image.open(inp / f"{names[0]}.png") as im:
        w0, h0 = im.size
    pending = [n for n in names if not all(t[1].exists() for t in targets_for(method, k, n, root, split))]
    print(f"{method} x{k} {split}: 남은 {len(pending)}/{len(names)}", flush=True)
    stats = {"method": method, "k": k, "split": split, "n": len(names), "n_pending": len(pending), "sr_seconds": 0.0, "encode_cpu_seconds": 0.0, "encode_wall_extra": 0.0}
    if not pending:
        return stats
    work = RAW / f"{method}__x{k}__{split}"
    shutil.rmtree(work, ignore_errors=True)
    in_flat, raw = work / "in", work / "raw"
    in_flat.mkdir(parents=True)
    raw.mkdir()
    for n in pending:
        (in_flat / f"{n}.png").symlink_to(inp / f"{n}.png")
    manifest = Path(manifest_dir) / f"{method}__x{k}__{split}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "k", "status", "input_sha256", "output_sha256", "bytes", "derived_from_x4"]
    new = not manifest.exists()
    mf = open(manifest, "a", newline="")
    wr = csv.DictWriter(mf, fields)
    if new:
        wr.writeheader()
    done_raw, futures = set(), {}
    pool = ProcessPoolExecutor(int(os.environ.get("SRREID_ENCODERS", 12)))

    def submit_ready(paths):
        for p in paths:
            if p in done_raw:
                continue
            stem = raw_output_name(method, p)
            if stem not in pending_set:
                continue
            done_raw.add(p)
            futures[p] = pool.submit(finalize_one, (str(p), str(inp / f"{stem}.png"), targets_for(method, k, stem, root, split), h0, w0))

    def drain(block=False):
        for p in list(futures):
            f = futures[p]
            if block or f.done():
                rows, sec = f.result()
                stats["encode_cpu_seconds"] += sec
                for r in rows:
                    wr.writerow({a: r.get(a, "") for a in fields})
                mf.flush()
                del futures[p]
                Path(p).unlink(missing_ok=True)
    pending_set = set(pending)
    t_sr = time.perf_counter()
    if m["kind"] == "classic":
        flt = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
        for n in pending:
            with Image.open(in_flat / f"{n}.png") as im:
                im.resize((w0 * k, h0 * k), flt).save(raw / f"{n}.png")
        stats["sr_seconds"] = time.perf_counter() - t_sr
        submit_ready(list(raw.glob("*.png")))
    else:
        env, repo, script, args = external_cmd(method, k, in_flat, raw, work)
        timing = work / "timing.json"
        cmd = [str(MF / env / "bin" / "python"), str(ROOT / "sr" / "runner.py"), "--root", str(ROOT), "--repo", str(repo), "--script", str(script), "--out-dir", str(work),
               "--timing", str(timing), "--seed", str(SEED), "--fraction", "0.95", "--"] + args
        e = dict(os.environ, PYTHONPATH=str(repo), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1")
        log = open(work / "run.log", "w")
        proc = subprocess.Popen(cmd, cwd=str(work), stdout=log, stderr=subprocess.STDOUT, env=e, start_new_session=True)
        search = raw_search_dirs(method, work, raw)
        while proc.poll() is None:
            ready = [p for p in search.rglob("*.png") if time.time() - p.stat().st_mtime > 2]
            submit_ready(ready)
            drain()
            time.sleep(1)
        log.close()
        stats["sr_seconds"] = time.perf_counter() - t_sr
        stats["rc"] = proc.returncode
        submit_ready(list(search.rglob("*.png")))
        if timing.exists():
            tj = json.loads(timing.read_text())
            per = sorted(tj.get("per_image_seconds", []))
            stats.update(sec_per_image_median=float(np.median(per)) if per else None, n_timed=len(per), max_allocated_gib=tj["max_memory_allocated"] / 2**30,
                         first_save_seconds=tj.get("first_save_seconds"), foreign_mib=tj.get("max_foreign_mib"), error=tj.get("error"))
        else:
            stats["error"] = "timing.json 없음"
        shutil.copy(work / "run.log", Path(manifest_dir) / f"{method}__x{k}__{split}.log")
    t_enc = time.perf_counter()
    drain(block=True)
    pool.shutdown()
    stats["encode_wall_extra"] = time.perf_counter() - t_enc
    mf.close()
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    stats["n_ok"] = sum(r["status"] == "ok" for r in rows)
    stats["n_failed"] = sum(r["status"] != "ok" for r in rows)
    stats["webp_bytes_mean"] = {str(kk): float(np.mean([int(r["bytes"]) for r in rows if r["status"] == "ok" and r["k"] == str(kk)] or [0])) for kk in (3, 4)}
    shutil.rmtree(work, ignore_errors=True)
    return stats


def _derive_one(args):
    src, dst, h0, w0 = args
    with Image.open(src) as im:
        arr = np.asarray(im.convert("RGB").resize((w0 * 3, h0 * 3), Image.BICUBIC))
    digest = write_webp(arr, dst)
    return {"name": dst.name, "k": 3, "status": "ok" if digest else "pixel_mismatch", "input_sha256": sha(src), "output_sha256": digest or "", "bytes": dst.stat().st_size if digest else 0, "derived_from_x4": True}


def derive_x3(method, split, root, manifest_dir):
    """×3 가중치가 없는 방법의 ×3을 저장된 ×4 WebP에서 PIL BICUBIC 축소로 만든다. GPU를 쓰지 않는다. 있는 파일은 건너뛴다."""
    h0, w0 = 128, 64
    src = sorted(out_dir(root, method, 4, split).glob("*.webp"))
    jobs = [(p, out_dir(root, method, 3, split) / p.name, h0, w0) for p in src if not (out_dir(root, method, 3, split) / p.name).exists()]
    t = time.perf_counter()
    with ProcessPoolExecutor(int(os.environ.get("SRREID_ENCODERS", 12))) as ex:
        rows = list(ex.map(_derive_one, jobs, chunksize=32))
    manifest = Path(manifest_dir) / f"{method}__x3__{split}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "k", "status", "input_sha256", "output_sha256", "bytes", "derived_from_x4"]
    new = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        w = csv.DictWriter(f, fields)
        if new:
            w.writeheader()
        w.writerows(rows)
    return {"method": method, "k": 3, "split": split, "n": len(src), "n_pending": len(jobs), "n_ok": sum(r["status"] == "ok" for r in rows), "derive_seconds": time.perf_counter() - t}


def main():
    global NO_DERIVE
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--no-derive", action="store_true")
    ap.add_argument("--derive-x3", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["bounding_box_train"])
    ap.add_argument("--k", type=int, nargs="+", default=[4, 3])
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    NO_DERIVE = a.no_derive
    if not a.smoke:  # 볼륨이 부족하면 쓰기 실패로 죽기 전에 시작하지 않는다.
        subprocess.run(["bash", str(ROOT / "scripts" / "disk_guard.sh"), "20"], check=True)
    root = Path(a.out) if a.out else (ROOT / "runs" / "stage05" / "sr_smoke" if a.smoke else ROOT / "data")
    manifest_dir = root / "_manifests" if a.smoke else ROOT / "runs" / "stage05" / "sr_manifests"
    results = []
    if a.derive_x3:
        for method in a.methods:
            for split in a.splits:
                print(json.dumps(derive_x3(method, split, root, manifest_dir), ensure_ascii=False), flush=True)
        return
    for split in a.splits:
        prepare_input(split)
    for method in a.methods:
        for split in a.splits:
            for k in a.k:
                if k not in METHODS[method]["scales"]:
                    continue  # ×3 가중치가 없는 방법은 ×4 실행에서 함께 만든다.
                try:
                    st = generate(method, k, split, root, a.smoke, manifest_dir)
                except Exception as e:  # 실패도 결과로 기록하고 다음 방법으로 간다.
                    st = {"method": method, "k": k, "split": split, "error": f"{type(e).__name__}: {e}"}
                results.append(st)
                print(json.dumps(st, ensure_ascii=False, default=str), flush=True)
                out = manifest_dir / "generate_stats.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "a") as f:
                    f.write(json.dumps(st, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
