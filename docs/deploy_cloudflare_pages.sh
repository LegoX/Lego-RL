#!/usr/bin/env bash
# Build the Lego-RL docs (fumadocs/Next.js static export) and deploy to a dedicated
# Cloudflare Pages project. Independent of the training dashboard project
# (lego-rl-dashboard); this one defaults to lego-rl.
#
# Reuses the same Cloudflare credentials as the dashboard sync:
#   CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID
# read from $ENV_FILE (default ~/.config/rl_dashboard_cloudflare.env), or
# already-exported env vars (e.g. via .env.cf).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_NAME="${PROJECT_NAME:-lego-rl}"
BRANCH_NAME="${BRANCH_NAME:-main}"
OUT_DIR="${OUT_DIR:-out}"
# Node 22 supports current wrangler; the dashboard pins v3 only for system Node 18.
WRANGLER_PKG="${WRANGLER_PKG:-wrangler@latest}"
ENV_FILE="${ENV_FILE:-$HOME/.config/rl_dashboard_cloudflare.env}"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }

# 1. Activate Node >= 20 via nvm (system Node on this host may be 18).
if [[ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]]; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"
  nvm use 22 >/dev/null 2>&1 || nvm use default >/dev/null 2>&1 || true
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
if (( NODE_MAJOR < 20 )); then
  log "ERROR: Node >= 20 required to build the docs (found $(node -v 2>/dev/null || echo none))."
  log "Install/activate it, e.g.: export NVM_DIR=\"\$HOME/.nvm\"; . \"\$NVM_DIR/nvm.sh\"; nvm install 22"
  exit 1
fi
log "Using node $(node -v), npm $(npm -v)"

# 2. Load Cloudflare credentials and the docs project config.
# The shared env file may set PROJECT_NAME/BRANCH_NAME for the DASHBOARD project,
# so capture the docs targets first, then prefer DOCS_PROJECT_NAME / DOCS_BRANCH_NAME
# from the file if present — otherwise the docs would deploy over the dashboard.
WANT_PROJECT="$PROJECT_NAME"
WANT_BRANCH="$BRANCH_NAME"
if [[ -f "$ENV_FILE" ]]; then
  log "Loading config from $ENV_FILE"
  # shellcheck disable=SC1090
  set -a; . "$ENV_FILE"; set +a
fi
PROJECT_NAME="${DOCS_PROJECT_NAME:-$WANT_PROJECT}"
BRANCH_NAME="${DOCS_BRANCH_NAME:-$WANT_BRANCH}"
if [[ -z "${CLOUDFLARE_API_TOKEN:-}" || -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]]; then
  log "ERROR: CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID must be set (env or $ENV_FILE)."
  exit 1
fi
export CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID

# 3. Install deps + static build. Prefer a deterministic, lockfile-based install
# (npm ci) whenever package-lock.json exists; fall back to npm install otherwise.
#
# npm ci always deletes node_modules and reinstalls from scratch. On a networked
# filesystem those few hundred MB of small files cost tens of minutes — it was
# the bulk of two ~45min deploys — and it is wasted whenever the lockfile has not
# moved. So stamp the lockfile's hash after a successful install and skip the
# reinstall while it still matches. FORCE_INSTALL=1 reinstalls regardless.
DEPS_STAMP="node_modules/.deps-stamp"
if [[ -f package-lock.json ]]; then
  WANT_DEPS="$(sha256sum package-lock.json | awk '{print $1}')"
  if [[ -z "${FORCE_INSTALL:-}" && -d node_modules && -f "$DEPS_STAMP" \
        && "$(cat "$DEPS_STAMP" 2>/dev/null)" == "$WANT_DEPS" ]]; then
    log "Dependencies unchanged since last install — skipping npm ci (FORCE_INSTALL=1 to override)"
  else
    log "Installing dependencies (npm ci)"
    npm ci
    printf '%s\n' "$WANT_DEPS" > "$DEPS_STAMP"
  fi
else
  log "Installing dependencies (npm install; no lockfile found)"
  npm install
fi
log "Building static export to $OUT_DIR/"
npm run build

if [[ ! -d "$OUT_DIR" ]]; then
  log "ERROR: build did not produce $OUT_DIR/ (check next.config.mjs output: 'export')."
  exit 1
fi

# 4. Ensure the Pages project exists (idempotent), then deploy.
if ! npx --yes "$WRANGLER_PKG" pages project list 2>/dev/null | grep -q "\b$PROJECT_NAME\b"; then
  log "Creating Cloudflare Pages project '$PROJECT_NAME' (production branch '$BRANCH_NAME')"
  npx --yes "$WRANGLER_PKG" pages project create "$PROJECT_NAME" \
    --production-branch "$BRANCH_NAME"
fi

log "Deploying $OUT_DIR/ to Cloudflare Pages project '$PROJECT_NAME'"
npx --yes "$WRANGLER_PKG" pages deploy "$OUT_DIR" \
  --project-name "$PROJECT_NAME" \
  --branch "$BRANCH_NAME" \
  --commit-dirty=true

log "Done. The public URL is printed above (https://$PROJECT_NAME.pages.dev)."
