#!/usr/bin/env bash
# Build the mounted Codex runtime image.
#
# Codex ships as a static-pie musl binary inside its npm package, so unlike the
# claude-code / opencode runtimes this needs no node, no dual-ABI build and no
# libc shims -- the final image is FROM scratch and carries only the vendor
# tree under the runtime root.
#
# Usage:
#   bash build-runtime-image.sh --image 127.0.0.1:5001/lego/c-codex-0.153.4:v0.1 --push
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_version"])' "$SCRIPT_DIR/manifest.json")"
IMAGE="c-codex-${VERSION}:v0.1"
PUSH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="${2:?}"; shift 2 ;;
    --push)  PUSH=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }

echo "[build] $IMAGE  (codex $VERSION)"
docker build --network=host --provenance=false --sbom=false \
  --build-arg "CODEX_VERSION=${VERSION}" \
  -t "$IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$SCRIPT_DIR"

# The image has no shell, so verify by extracting rather than running.
echo "[verify] extracting the runtime tree"
cid="$(docker create "$IMAGE" /codex-placeholder-never-run)"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
tmp="$(mktemp -d)"
docker cp "$cid:/opt/custom-agent-runtime/codex/bin/codex" "$tmp/codex"
chmod +x "$tmp/codex"
got="$("$tmp/codex" --version | awk '{print $NF}')"
[ "$got" = "$VERSION" ] || { echo "version mismatch: image has $got, manifest says $VERSION" >&2; exit 1; }
echo "[verify] codex $got  static=$(ldd "$tmp/codex" 2>&1 | grep -c 'statically linked')"
rm -rf "$tmp"

if [ "$PUSH" = 1 ]; then
  echo "[push] $IMAGE"
  docker push "$IMAGE"
fi
echo "[done] $IMAGE"
