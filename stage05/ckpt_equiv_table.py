"""Part H. logs/stage05/h5_*.out.json을 results/stage05_ckpt_equiv.csv로 묶는다. 판정: 결정론 모드에서 1스텝 차이 0, 200 iteration 차이가 반복 실행 간 차이 이내면 등가."""
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "stage05"
FIELDS = ["model", "input", "batch", "mode", "task", "status", "nondeterministic_ops", "loss_range", "grad_norm_mean", "grad_vec_noise_off", "grad_vec_noise_on", "grad_vec_diff_on_off",
          "grad_norm_range_off", "grad_norm_range_on", "grad_norm_mean_gap", "iters", "first_diff_on_off", "first_diff_off_off", "first_diff_on_on", "max_diff_on_off", "max_diff_off_off", "max_diff_on_on",
          "mean_diff_on_off", "mean_diff_off_off", "mean_diff_on_on", "within_noise"]


def load(p):
    return json.loads(Path(p).read_text()) if os.path.exists(p) else None


def diff(a, b):
    a, b = np.array(a), np.array(b)
    d = np.abs(a - b)
    nz = np.nonzero(d)[0]
    return (int(nz[0]) if len(nz) else "", float(d.max()), float(d.mean()))


rows = []
for p in sorted(glob.glob(str(LOG / "h[57]_step_*.out.json"))):
    m = re.match(r"h[57]_step_(.+)_x(\d)_b(\d+)_(\w+?)\.out\.json", os.path.basename(p))
    r = load(p)
    rows.append({"model": m[1], "input": f"x{m[2]}", "batch": m[3], "mode": m[4], "task": "step", "status": r.get("status"), "nondeterministic_ops": " ".join(r.get("nondeterministic_ops", [])),
                 **{k: r.get(k, "") for k in ("loss_range", "grad_norm_mean", "grad_vec_noise_off", "grad_vec_noise_on", "grad_vec_diff_on_off", "grad_norm_range_off", "grad_norm_range_on", "grad_norm_mean_gap")}})
seen = set()
for p in sorted(glob.glob(str(LOG / "h[57]_curve_*_off.out.json"))):
    stem = os.path.basename(p)[:-len("_off.out.json")]
    if stem in seen:
        continue
    seen.add(stem)
    m = re.match(r"h[57]_curve_(.+)_x(\d)_b(\d+)_(\w+)", stem)
    c = {t: load(LOG / f"{stem}_{t}.out.json") for t in ("off", "off2", "on", "on2")}
    if stem.startswith("h7") and c["on2"] is None:
        del c["on2"]
    if not all(v and v.get("status") == "ok" for v in c.values()):
        rows.append({"model": m[1], "input": f"x{m[2]}", "batch": m[3], "mode": m[4], "task": "curve", "status": "crashed"})
        continue
    L = {t: c[t]["losses"] for t in c}
    oo = diff(L["off"], L["off2"])
    nn = diff(L["on"], L["on2"]) if "on2" in L else ("", 0.0, 0.0)
    xs = [diff(L[x], L[y]) for x in ("off", "off2") for y in ("on", "on2") if y in L]
    fd = [x[0] for x in xs if x[0] != ""]
    mxx = max(x[1] for x in xs)
    rows.append({"model": m[1], "input": f"x{m[2]}", "batch": m[3], "mode": m[4], "task": "curve", "status": "ok", "nondeterministic_ops": " ".join(sorted(set(sum((c[t].get("nondeterministic_ops", []) for t in c), [])))),
                 "iters": len(L["off"]), "first_diff_on_off": min(fd) if fd else "", "first_diff_off_off": oo[0], "first_diff_on_on": nn[0], "max_diff_on_off": mxx, "max_diff_off_off": oo[1], "max_diff_on_on": nn[1],
                 "mean_diff_on_off": float(np.mean([x[2] for x in xs])), "mean_diff_off_off": oo[2], "mean_diff_on_on": nn[2], "within_noise": mxx <= max(oo[1], nn[1])})
with (ROOT / "results" / "stage05_ckpt_equiv.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["task"] == "step":
        print(r["model"], r["input"], r["batch"], r["mode"], r["status"], "loss_rng %.3g" % r["loss_range"] if r["loss_range"] != "" else "", "norm %.4g" % r["grad_norm_mean"], "vec noise %.3g/%.3g diff %.3g" % (r["grad_vec_noise_off"], r["grad_vec_noise_on"], r["grad_vec_diff_on_off"]),
              "normrange %.3g/%.3g gap %.3g" % (r["grad_norm_range_off"], r["grad_norm_range_on"], r["grad_norm_mean_gap"]), r["nondeterministic_ops"][:60])
    else:
        print(r["model"], r["input"], r["batch"], r["mode"], "CURVE", r["status"], "first", r.get("first_diff_on_off"), r.get("first_diff_off_off"), r.get("first_diff_on_on"), "max %.4g off %.4g on %.4g" % (r["max_diff_on_off"], r["max_diff_off_off"], r["max_diff_on_on"]),
              "mean %.4g %.4g %.4g" % (r["mean_diff_on_off"], r["mean_diff_off_off"], r["mean_diff_on_on"]), r["within_noise"])
