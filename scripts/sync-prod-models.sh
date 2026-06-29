#!/usr/bin/env bash
# Copy promoted production checkpoints from models/ to prod-models/.
#
# Usage: ./scripts/sync-prod-models.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC=("$ROOT/models/birdbrain_inat_v1-3.pt" "$ROOT/models/birdbrain_v1-4.pt" "$ROOT/models/birdbrain_resnet50_v1-4.pt")
DEST="$ROOT/prod-models"

mkdir -p "$DEST"

for file in "${SRC[@]}"; do
	if [[ ! -f "$file" ]]; then
		echo "Missing source checkpoint: $file" >&2
		exit 1
	fi
done

cp "${SRC[@]}" "$DEST/"
echo "Synced production checkpoints to $DEST:"
ls -lh "$DEST"/*.pt
