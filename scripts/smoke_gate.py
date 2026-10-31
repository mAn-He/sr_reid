"""스모크 통과 판정. 통과해야 run_pipeline.sh가 본 실행을 시작한다. 표준 라이브러리만 쓴다.
통과 조건: 모든 단계 성공, 리포트 산출물 존재, 실패 기록 없음.
Google Drive 등 가중치 확보 실패만 예외다. 코드 결함이 아니고 본 실행에서도 그 방법의 실패로 기록되기 때문이다."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "smoke_001"
DATA = ROOT / "runs" / "smoke_data"
OUTPUTS = ["REPORT.md", "results/pilot_summary.csv", "results/grid.png", "results/dose_response.png"]
REQUIRED = ["setup_envs", "freeze_config", "tests", "fetch_weights", "prepare_split", "make_lr", "lr_gate",
            "sr_exclusive", "sr_determinism", "train_hr", "train_mixed", "train_lr_only", "sr_rest", "eval_bicubic",
            "eval_reid", "eval_fidelity", "eval_bootstrap", "eval_correlate", "report_summary", "report_gonogo"]


def problems():
    found = []
    latest = {}
    tsv = ROOT / "logs" / f"stages_{RUN_ID}.tsv"
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            stage, status = line.split("\t")[:2]
            latest[stage] = status
    found += [f"단계 {s}: {latest.get(s, '실행 안 됨')}" for s in REQUIRED if latest.get(s) != "success"]
    found += [f"산출물 없음: {p}" for p in OUTPUTS if not (DATA / p).exists()]
    if (ROOT / "logs" / "GUARD_VIOLATION").exists():
        found.append("test 가드 위반")
    fail = DATA / "results" / "failures.csv"
    if fail.exists():
        with open(fail, newline="") as f:
            for r in csv.DictReader(f):
                if r["stage"] != "weights" and not r["reason"].startswith("가중치 없음"):
                    found.append(f"실패 기록: {r['stage']} {r['condition']} {r['image']} {r['reason'][:200]}")
    return found


def main():
    found = problems()
    if found:
        with open(ROOT / "logs" / "STATUS.md", "a") as f:
            f.write("\n## 스모크 미통과: 본 실행을 시작하지 않았습니다\n\n" + "".join(f"- {p}\n" for p in found))
        print("스모크 미통과", *found, sep="\n", flush=True)
        sys.exit(1)
    print("스모크 통과", flush=True)


if __name__ == "__main__":
    main()
