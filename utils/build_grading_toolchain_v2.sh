#!/usr/bin/env bash
# Build the offline SWE-bench grading toolchain: a self-contained grader that works
# inside ANY task image (no Python required in the image).
#
#   OUT=<dir> bash utils/build_grading_toolchain_v2.sh
#
# Needs egress from THIS host to github.com (python-build-standalone) and PyPI (swebench);
# the sandbox pods that consume it need none. ~150 MB, a couple of minutes.
#
#   <out>/python311/          python-build-standalone 3.11 (relocatable) with
#                             swebench==4.1.0 (+datasets, fastcore) installed
#   <out>/parser_offline_v2.py  offline grader for Verified + Multilingual records
#   <out>/parser_offline.py   v1 grader (verbatim copy, for the old verified test.sh)
#   <out>/pylibs/             v1 pylibs (verbatim copy, for the old verified test.sh)
#
# Mount it read-only at /opt/grading:
#   HARBOR_HOSTPATH_MOUNTS='[{"host_path":"<out>","mount_path":"/opt/grading","read_only":true}]'
#
# <out> must be readable from every sandbox NODE (hostPath is resolved node-side), so put it
# on shared storage or replicate it to each node.
#
# Idempotent: re-running only refreshes the two parser files unless FORCE=1.
set -euo pipefail

OUT="${OUT:?set OUT=<dir on shared storage, mounted at /opt/grading in the verifier pod>}"
V1="${V1:-}"   # optional: a v1 toolchain dir to copy parser_offline.py + pylibs/ from
PBS_TARBALL="${PBS_TARBALL:-}"   # optional local copy of the python-build-standalone tarball
PBS_URL="${PBS_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz}"
SWEBENCH_SPEC="${SWEBENCH_SPEC:-swebench==4.1.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUT"

if [ ! -x "$OUT/python311/bin/python3" ] || [ "${FORCE:-0}" = 1 ]; then
    # Download and extract FIRST, replace only on success: a 404/502 (curl without -f exits 0
    # on an HTML error body) must not leave /opt/grading with no interpreter at all.
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    if [ -n "$PBS_TARBALL" ]; then
        tarball="$PBS_TARBALL"
    else
        tarball="$tmp/pbs.tar.gz"
        echo "[toolchain] downloading $PBS_URL"
        curl -fsSL --retry 3 --retry-delay 5 -o "$tarball" "$PBS_URL"
    fi
    tar -xzf "$tarball" -C "$tmp"
    [ -x "$tmp/python/bin/python3" ] || { echo "[toolchain] extracted tree has no bin/python3" >&2; exit 1; }
    rm -rf "$OUT/python311"
    mv "$tmp/python" "$OUT/python311"
    rm -rf "$tmp"; trap - EXIT
    echo "[toolchain] installing $SWEBENCH_SPEC into $OUT/python311"
    "$OUT/python311/bin/python3" -m pip install -q --no-warn-script-location "$SWEBENCH_SPEC" 'datasets==2.16.1' 'fastcore<1.11'
    # relocatability check: the interpreter must not depend on the build path
    "$OUT/python311/bin/python3" -c 'import swebench, sys; print("swebench", swebench.__version__, "python", sys.version.split()[0])'
fi

install -m 0644 "$REPO_ROOT/utils/grading_toolchain_v2/parser_offline_v2.py" "$OUT/parser_offline_v2.py"

if [ -n "$V1" ] && [ -d "$V1" ]; then
    [ -f "$OUT/parser_offline.py" ] || install -m 0644 "$V1/parser_offline.py" "$OUT/parser_offline.py"
    if [ ! -d "$OUT/pylibs" ]; then
        echo "[toolchain] copying v1 pylibs (verified-compat) ..."
        cp -a "$V1/pylibs" "$OUT/pylibs"
    fi
fi

cat > "$OUT/README.md" <<EOF
# grading-toolchain-v2

Self-contained offline SWE-bench grading, mounted at /opt/grading inside verifier pods.
Built by utils/build_grading_toolchain_v2.sh (Lego-RL) on $(date +%F).

- python311/            python-build-standalone 3.11 + ${SWEBENCH_SPEC} (runs in any glibc image)
- parser_offline_v2.py  grader for SWE-bench Verified + Multilingual task records
- parser_offline.py     v1 grader (verified test.sh compat, uses /opt/miniconda3 python + pylibs)
- pylibs/               v1 site-packages (cpython-311 ABI, swebench 4.0.3)
EOF

echo "[toolchain] done -> $OUT"
du -sh "$OUT"
