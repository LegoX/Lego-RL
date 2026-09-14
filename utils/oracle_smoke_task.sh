#!/usr/bin/env bash
# Oracle / empty-patch smoke test of one Harbor task with plain docker (no harbor, no k8s).
#
#   scripts/tools/oracle_smoke_task.sh <task_dir> [oracle|empty] [grading_toolchain_dir]
#
# oracle: apply solution/solve.sh then run tests/test.sh  -> expect reward 1
# empty : run tests/test.sh on the untouched checkout      -> expect reward 0
#
# The grading toolchain (3rd arg, or $GRADING_TOOLCHAIN) is mounted read-only
# at /opt/grading exactly like HARBOR_HOSTPATH_MOUNTS does in the eval configs. Verifier logs
# land in $SMOKE_OUT/<dataset>/<instance>/<mode>/ (default /tmp/harbor-smoke, kept OUT of the
# dataset dir so the harbor task loader never sees them).
set -uo pipefail

TASK_DIR="$(cd "${1:?task dir}" && pwd)"
MODE="${2:-oracle}"
GRADING="${3:-${GRADING_TOOLCHAIN:?set GRADING_TOOLCHAIN=<grading toolchain dir> or pass it as arg 3}}"
IID="$(basename "$TASK_DIR")"
DS="$(basename "$(dirname "$TASK_DIR")")"
OUT="${SMOKE_OUT:-/tmp/harbor-smoke}/$DS/$IID/$MODE"
mkdir -p "$OUT"

IMAGE="$(sed -n 's/^docker_image = "\(.*\)"/\1/p' "$TASK_DIR/task.toml")"
[ -n "$IMAGE" ] || { echo "no docker_image in $TASK_DIR/task.toml" >&2; exit 2; }

echo "[smoke] $IID mode=$MODE image=$IMAGE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull -q "$IMAGE" >/dev/null || { echo "[smoke] pull failed" >&2; exit 2; }

SOLVE=""
[ "$MODE" = oracle ] && SOLVE="bash /solution/solve.sh > /logs/verifier/solve.log 2>&1 || { echo SOLVE_FAILED; tail -20 /logs/verifier/solve.log; }"

docker run --rm --entrypoint bash \
    -v "$TASK_DIR/tests:/tests:ro" \
    -v "$TASK_DIR/solution:/solution:ro" \
    -v "$GRADING:/opt/grading:ro" \
    -v "$OUT:/logs/verifier" \
    -e TMPDIR=/tmp \
    "$IMAGE" -c "
set -o pipefail
mkdir -p /logs/verifier
$SOLVE
bash /tests/test.sh > /logs/verifier/verifier.log 2>&1
echo \"test.sh exit=\$?\"
echo \"reward=\$(cat /logs/verifier/reward.txt 2>/dev/null)\"
" 2>&1 | tail -3
echo "[smoke] logs: $OUT"
