#!/usr/bin/env bash
# Full deployment: pull latest code and rebuild/restart all Docker services.
#
# Usage (from anywhere):
#   ./scripts/deploy-full.sh
#   ./scripts/deploy-full.sh --no-cache   # ignore Docker layer cache (slow; first install)
#
# Requires: git, docker, docker compose, ./prod-models/ with production checkpoints.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NO_CACHE=0
for arg in "$@"; do
	case "$arg" in
		--no-cache) NO_CACHE=1 ;;
		-h | --help)
			echo "Usage: $0 [--no-cache]"
			exit 0
			;;
		*)
			echo "Unknown option: $arg" >&2
			exit 1
			;;
	esac
done

if ! command -v git >/dev/null 2>&1; then
	echo "git is required but not installed." >&2
	exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
	echo "docker compose is required but not available." >&2
	exit 1
fi

REQUIRED=(
	prod-models/birdbrain_v1-4.pt
	prod-models/birdbrain_resnet50_v1-4.pt
)
for file in "${REQUIRED[@]}"; do
	if [[ ! -f "$file" ]]; then
		echo "Missing production checkpoint: $file" >&2
		echo "Run ./scripts/sync-prod-models.sh after training, or copy checkpoints manually." >&2
		exit 1
	fi
done

if [[ -f .env ]]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

echo "==> Birdbrain full deploy (root: $ROOT)"
echo "==> Pulling latest from git..."
git pull --ff-only

BUILD_ARGS=()
if [[ "$NO_CACHE" -eq 1 ]]; then
	echo "==> Building all images (no cache)..."
	BUILD_ARGS+=(--no-cache)
else
	echo "==> Building all images..."
fi

docker compose build "${BUILD_ARGS[@]}"
echo "==> Starting services..."
docker compose up -d

echo "==> Done. App should be on port ${BIRDBRAIN_PORT:-3012}."
docker compose ps
