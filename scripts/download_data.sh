#!/usr/bin/env bash
# Download the main dataset into data/ (git-ignored), pinned to the commit
# every result so far was computed on.
#
#   scripts/download_data.sh
#
# Memora (Uddin et al., 2026): https://github.com/geniesinc/Memora
# LoCoMo and LongMemEval were inspected (E10, E11) but are not used (D13),
# so they are not downloaded here.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEMORA_URL=https://github.com/geniesinc/Memora.git
MEMORA_COMMIT=a6493188efc836d6511ed5e4163fe3ba87da30ff
DEST="$REPO/data/memora"

mkdir -p "$REPO/data"
if [ ! -d "$DEST/.git" ]; then
  git clone "$MEMORA_URL" "$DEST"
fi
if ! git -C "$DEST" cat-file -e "$MEMORA_COMMIT^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --quiet origin
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$MEMORA_COMMIT"
echo "Memora at $(git -C "$DEST" rev-parse --short HEAD) in data/memora"
