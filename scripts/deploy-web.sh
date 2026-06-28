#!/usr/bin/env bash
# Frontend-only deployment: pull latest code and rebuild/restart the web container.
#
# Usage (from anywhere):
#   ./scripts/deploy-web.sh
#
# Does not rebuild or restart the API. Use deploy-full.sh for API or dependency changes.
# Requires: git, docker, docker compose.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
	echo "git is required but not installed." >&2
	exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
	echo "docker compose is required but not available." >&2
	exit 1
fi

if [[ -f .env ]]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

echo "==> Birdbrain web deploy (root: $ROOT)"
echo "==> Pulling latest from git..."
git pull --ff-only

echo "==> Building web image..."
docker compose build web

echo "==> Restarting web service (API unchanged)..."
docker compose up -d --no-deps web

echo "==> Done. App should be on port ${BIRDBRAIN_PORT:-3012}."
docker compose ps web
