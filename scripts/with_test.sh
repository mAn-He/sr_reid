#!/usr/bin/env bash
# 허용된 test 접근 작업(공개 가중치 평가, RAM·등가성 측정, test SR 생성)만 이 래퍼로 실행한다. 끝나면 다시 잠근다.
#   bash scripts/with_test.sh "이유" 명령 ...
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REASON=$1; shift
mkdir -p "$ROOT/logs"
printf '%s\tstart\t%s\t%s\n' "$(date '+%F %T')" "$REASON" "$*" >> "$ROOT/logs/test_access.log"
chmod 700 "$ROOT/data/locked"
find "$ROOT/data/locked" -mindepth 1 -maxdepth 1 -exec chmod 700 {} +
# 원본 파일도 000이므로 처음 한 번만 파일 권한을 푼다. 평소 차단은 상위 디렉터리 000과 가드가 맡고, Stage 0.5 끝에 --relock-all로 파일까지 되돌린다.
for s in query bounding_box_test gt_query gt_bbox; do
  f=$(find "$ROOT/data/locked/$s" -maxdepth 1 -type f -print -quit 2>/dev/null)
  [ -n "$f" ] && [ "$(stat -c %a "$f")" = "0" ] && chmod -R u+rwX "$ROOT/data/locked/$s"
done
# 동시에 도는 다른 with_test 작업이 있으면 마지막 작업이 끝날 때만 잠근다.
exec 9>/tmp/srreid_with_test.lock
flock -s 9
relock() {
  exec 9>&-
  exec 8>/tmp/srreid_with_test.lock
  flock -xn 8 || return 0
  find "$ROOT/data/locked" -mindepth 1 -maxdepth 1 -exec chmod 000 {} +
  chmod 000 "$ROOT/data/locked"
}
trap relock EXIT
ALLOW_TEST=1 "$@" 9>&-
rc=$?
printf '%s\tend rc=%s\t%s\n' "$(date '+%F %T')" "$rc" "$REASON" >> "$ROOT/logs/test_access.log"
exit $rc
