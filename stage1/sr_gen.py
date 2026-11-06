"""Stage 1 SR 생성기 (표본 전용). 방법 하나 × 배율 하나를 한 프로세스에서 돌린다(모델은 한 번만 불러옴).

- 추론 경로는 stage05/sr_batch_bench.py의 어댑터를 그대로 쓴다(배치 1 출력이 공식 스모크 출력과 같음을 검증한 경로). third_party 소스는 고치지 않는다.
- 배치 정책 (가): realesrgan b4(배치 1과 비트 일치), 나머지 b1.
- 이미지마다 파일명 기반 고정 seed를 다시 건다(--seed-mode per_image, 기본). seed = sha256("42:<파일명>") 앞 8자리 & 0x7fffffff.
  처리 순서·재시작과 무관하게 같은 이미지에는 같은 난수가 쓰인다.
- --det: 결정적 설정. cudnn.benchmark=False, cudnn.deterministic=True, SDPA는 math 백엔드만(flash·mem-efficient 끔).
- --from-x4: 이 배율(×3)을 같은 출력 루트의 ×4 결과에서 PIL BICUBIC 축소로 만든다(GPU 없음). ×4가 먼저 있어야 한다.
- --upscale-direct: 확산 방법(osdhuman, pisa_sr)을 공식 resize 경로 그대로 upscale=k로 돌린다(×3 직접 생성 비교용).
출력: <out>/<method>/x<k>/<원본파일명>.webp (WebP 무손실 method 2, 저장 직후 픽셀 전수 비교). 이미 있는 파일은 건너뛴다.
기록: <out>/_logs/<method>__x<k>.jsonl (이미지별 시간), <out>/_logs/<method>__x<k>__summary.json.
이미지는 열어 보지 않고 수치로만 다룬다."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
INP = ROOT / "data" / "sr_input" / "bounding_box_train"
SAMPLE = ROOT / "configs" / "stage1_sample.csv"
WEBP = {"format": "WEBP", "lossless": True, "quality": 100, "method": 2}
CLASSIC = ("bicubic", "lanczos")
REPO = {"realesrgan": "Real-ESRGAN", "osdhuman": "OSDHuman", "pisa_sr": "PiSA-SR", "swinir": "SwinIR", "hat": "HAT", "mambairv2": "MambaIR"}
SERVER = "새 서버(RTX 4090, EPYC 7282, cgroup v2)"


def image_seed(name):
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def sample_names(limit=None, offset=0):
    with open(SAMPLE) as f:
        names = [Path(r["name"]).stem for r in csv.DictReader(f)]
    names = names[offset:]
    return names[:limit] if limit else names


def write_webp(np, Image, arr, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(arr).save(tmp, **WEBP)
    with Image.open(tmp) as im:
        same = np.array_equal(np.asarray(im.convert("RGB")), arr)
    if not same:
        tmp.unlink()
        raise RuntimeError(f"WebP 픽셀 불일치 {dst}")
    os.replace(tmp, dst)
    return hashlib.sha256(dst.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(CLASSIC) + list(REPO))
    ap.add_argument("--lambda-sem", dest="lambda_sem", type=float, default=1.0)
    ap.add_argument("--k", type=int, required=True, choices=[3, 4])
    ap.add_argument("--out", default=str(ROOT / "runs" / "stage1_sample"))
    ap.add_argument("--tag", default="", help="출력 폴더 이름 뒤에 붙임(예: x3_direct)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed-mode", choices=["per_image", "global"], default="per_image")
    ap.add_argument("--det", action="store_true")
    ap.add_argument("--det-strict", action="store_true", help="--det + torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8")
    ap.add_argument("--from-x4", action="store_true")
    ap.add_argument("--upscale-direct", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="재현성 측정용: 저장 없이 sha1만 기록")
    ap.add_argument("--log-name", default=None)
    ap.add_argument("--warmup", action="store_true", help="본 처리 전에 첫 이미지를 한 번 돌려 버린다(프로세스 첫 추론의 커널 선택 차이 제거). 첫 추론과 본 추론의 차이를 기록")
    ap.add_argument("--reverse", action="store_true", help="재현성 측정용: 처리 순서를 뒤집는다")
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("srreid_guard", str(ROOT / "common" / "guard.py"))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guard.install()
    import numpy as np
    from PIL import Image

    mname = a.method if a.method != "pisa_sr" else f"pisa_sr_sem{a.lambda_sem}"
    sub = f"x{a.k}" + (f"_{a.tag}" if a.tag else "")
    a.out = str(Path(a.out).resolve())  # 뒤에서 os.chdir를 하므로 절대 경로로 고정
    out_dir = Path(a.out) / mname / sub
    log_dir = Path(a.out) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = a.log_name or f"{mname}__{sub}"
    names = sample_names(a.limit, a.offset)
    pending = names if a.no_write else [n for n in names if not (out_dir / f"{n}.webp").exists()]
    if a.reverse:
        pending = pending[::-1]
    print(f"{mname} {sub}: 남은 {len(pending)}/{len(names)}", flush=True)
    summary = {"server": SERVER, "method": mname, "k": a.k, "sub": sub, "n": len(names), "n_pending": len(pending), "seed_mode": a.seed_mode, "det": a.det,
               "from_x4": a.from_x4, "upscale_direct": a.upscale_direct, "batch": None, "omp": os.environ.get("OMP_NUM_THREADS"), "mkl": os.environ.get("MKL_NUM_THREADS")}
    jl = open(log_dir / f"{log_name}.jsonl", "a")
    t_start = time.perf_counter()

    if a.method in CLASSIC or a.from_x4:
        # CPU 경로. classic: 원본 PNG를 PIL로 k배. from_x4: 같은 루트의 ×4 WebP를 PIL BICUBIC으로 축소.
        resample = Image.LANCZOS if a.method == "lanczos" else Image.BICUBIC
        src4 = Path(a.out) / mname / "x4"
        n_ok = 0
        for n in pending:
            t0 = time.perf_counter()
            with Image.open(INP / f"{n}.png") as im:
                im = im.convert("RGB")
                w0, h0 = im.size
                if a.from_x4:
                    with Image.open(src4 / f"{n}.webp") as s:
                        s = s.convert("RGB")
                        assert s.size == (w0 * 4, h0 * 4), s.size
                        arr = np.asarray(s.resize((w0 * a.k, h0 * a.k), Image.BICUBIC))
                else:
                    arr = np.asarray(im.resize((w0 * a.k, h0 * a.k), resample))
            rec = {"name": n, "sec": time.perf_counter() - t0, "sha1": hashlib.sha1(arr.tobytes()).hexdigest()}
            if not a.no_write:
                rec["webp_sha256"] = write_webp(np, Image, arr, out_dir / f"{n}.webp")
            jl.write(json.dumps(rec) + "\n")
            n_ok += 1
        summary.update(n_done=n_ok, wall=time.perf_counter() - t_start, device="cpu")
    else:
        bench_path = ROOT / "stage05" / "sr_batch_bench.py"
        here = str(bench_path.parent)
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (here, str(ROOT / "stage1"))]
        sys.path.insert(0, str(ROOT / "third_party" / REPO[a.method]))
        spec = importlib.util.spec_from_file_location("sr_batch_bench", str(bench_path))
        bb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bb)
        bb.WORK.mkdir(parents=True, exist_ok=True)
        os.chdir(bb.WORK.parent)
        import torch
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.set_per_process_memory_fraction(0.95)
        if a.det_strict:
            a.det = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "CUBLAS_WORKSPACE_CONFIG=:4096:8 필요"
        if a.det:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        ns = argparse.Namespace(method=a.method, k=(4 if a.method in ("realesrgan", "osdhuman", "pisa_sr") else a.k), names=names or ["x"],
                                noise="global", lambda_sem=a.lambda_sem)
        if a.method in ("realesrgan", "osdhuman", "pisa_sr") and a.k == 3 and not a.upscale_direct:
            raise SystemExit("realesrgan·osdhuman·pisa_sr ×3은 --from-x4(×4 축소) 또는 --upscale-direct(확산만)로 만든다")
        if a.upscale_direct and a.method not in ("osdhuman", "pisa_sr"):
            raise SystemExit("--upscale-direct는 osdhuman·pisa_sr 전용")
        t_load = time.perf_counter()
        ad = bb.ADAPTERS[a.method](ns)
        if a.upscale_direct:
            ad.args.upscale = a.k  # 공식 스크립트에서 upscale은 입력 resize와 최종 resize에만 쓰인다(run_inference.py:216-265, test_pisasr.py:35-70)
        if a.det:  # BasicSR 어댑터가 init에서 benchmark=True로 켜므로 그 뒤에 덮어쓴다
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        t_load = time.perf_counter() - t_load
        bs = a.batch or (4 if a.method == "realesrgan" else 1)
        summary.update(det_strict=a.det_strict, batch=bs, load_seconds=t_load, torch=torch.__version__, cudnn_benchmark=torch.backends.cudnn.benchmark,
                       cudnn_deterministic=torch.backends.cudnn.deterministic,
                       sdpa={"flash": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                             "math": torch.backends.cuda.math_sdp_enabled()})
        torch.cuda.reset_peak_memory_stats()
        warm = None
        if a.warmup and pending:
            def seed_for(n):
                if a.seed_mode == "per_image":
                    s_ = image_seed(n)
                    random.seed(s_); np.random.seed(s_); torch.manual_seed(s_); torch.cuda.manual_seed_all(s_)
            n0 = pending[0]
            seed_for(n0)
            o1 = ad.run([ad.prep(n0)])[0]
            seed_for(n0)
            o2 = ad.run([ad.prep(n0)])[0]
            d = np.abs(o1.astype(np.int16) - o2.astype(np.int16))
            warm = {"name": n0, "first_vs_second_max_abs": int(d.max()), "first_vs_second_frac_px": float((d > 0).mean()), "first_vs_second_mean_abs": float(d.mean())}
            print("warmup", warm, flush=True)
        summary["warmup"] = warm
        times = []
        for i in range(0, len(pending), bs):
            chunk = pending[i:i + bs]
            if a.seed_mode == "per_image":
                s = image_seed(chunk[0])  # 배치 4(realesrgan)는 난수를 쓰지 않는다
                random.seed(s)
                np.random.seed(s)
                torch.manual_seed(s)
                torch.cuda.manual_seed_all(s)
            items = [ad.prep(n) for n in chunk]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = ad.run(items)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt / len(chunk))
            for n, o in zip(chunk, outs):
                assert o.shape[:2] == (128 * a.k, 64 * a.k), (n, o.shape)
                rec = {"name": n, "sec": dt / len(chunk), "sha1": hashlib.sha1(o.tobytes()).hexdigest(), "seed": image_seed(n) if a.seed_mode == "per_image" else None}
                if not a.no_write:
                    rec["webp_sha256"] = write_webp(np, Image, np.ascontiguousarray(o), out_dir / f"{n}.webp")
                jl.write(json.dumps(rec) + "\n")
            jl.flush()
        ts = sorted(times)
        summary.update(n_done=len(pending), wall=time.perf_counter() - t_start,
                       infer_s_per_img_mean=(sum(times) / len(times)) if times else None,
                       infer_s_per_img_median=ts[len(ts) // 2] if ts else None,
                       infer_s_per_img_median_excl_first=(sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else None),
                       peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    jl.close()
    (log_dir / f"{log_name}__summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
