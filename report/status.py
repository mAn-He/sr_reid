"""logs/STATUS.md 작성: 단계별 성공/실패, 소요 시간, 다음에 확인할 것. 표준 라이브러리만 쓴다."""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    run_id, results_dir, final_note = sys.argv[1], ROOT / sys.argv[2], sys.argv[3]
    tsv = ROOT / "logs" / f"stages_{run_id}.tsv"
    latest = {}
    if tsv.exists():
        for line in tsv.read_text().splitlines():
            stage, status, seconds, when = line.split("\t")
            latest[stage] = (status, float(seconds), when)
    lines = [f"# STATUS ({run_id})", "", f"작성 시각: {time.strftime('%F %T %Z')}", "", f"**{final_note}**", "",
             "| 단계 | 상태 | 소요 | 완료 시각 |", "|---|---|---:|---|"]
    for stage, (status, sec, when) in latest.items():
        lines.append(f"| {stage} | {status} | {sec / 60:.1f}분 | {when} |")
    fail = results_dir / "failures.csv"
    n_fail = 0
    groups = {}
    if fail.exists():
        with open(fail, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["run_id"] == run_id]
        n_fail = len(rows)
        for r in rows:
            groups[(r["stage"], r["condition"])] = groups.get((r["stage"], r["condition"]), 0) + 1
    lines += ["", f"## 실패 기록: {n_fail}건 (`{fail.relative_to(ROOT)}`)", ""]
    lines += [f"- {s} / {c or '-'}: {n}건" for (s, c), n in sorted(groups.items())]
    guard = ROOT / "logs" / "GUARD_VIOLATION"
    if guard.exists():
        lines += ["", f"## test 가드 위반 발생: `{guard.relative_to(ROOT)}`를 먼저 확인하세요."]
    lines += ["", "## 다음에 확인할 것", "",
              "1. 이 파일의 실패 단계와 `failures.csv`",
              "2. `REPORT.md`의 go/no-go 판정",
              f"3. 육안 확인: `{ROOT}/{sys.argv[2]}/grid.png` (로컬 전용, 외부 업로드 금지)",
              f"4. `{ROOT}/{sys.argv[2]}/pilot_summary.csv`, `dose_response.png`",
              "5. `weights/sd21_verification.json`의 SD2.1 상태",
              f"6. 재실행: `cd {ROOT} && bash scripts/run_all.sh` (끝난 단계와 검증된 SR 출력은 건너뜁니다)"]
    (ROOT / "logs" / "STATUS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
