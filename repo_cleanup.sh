#!/usr/bin/env bash
set -euo pipefail

echo "[cleanup] Removing generated files from git tracking (kept locally on disk)..."

# Recursively untrack common junk and generated artifacts if they are currently tracked.
git rm -r --cached --ignore-unmatch   .ipynb_checkpoints   __pycache__   outputs   logs   results   checkpoints   notebook06_outputs   Dataset   dataset   data   cu_zr_mlip_project

# Untrack common binary/model artifacts if any are currently tracked.
git ls-files | grep -E '\.(pt|pth|model|ckpt|h5|hdf5|db)$' | while read -r f; do
  git rm --cached --ignore-unmatch "$f"
done || true

# Untrack macOS junk if present.
git ls-files | grep -E '(^|/)\.DS_Store$' | while read -r f; do
  git rm --cached --ignore-unmatch "$f"
done || true

echo "[cleanup] Current status:"
git status --short

cat <<'EOF'

Next recommended commands:
  git add .gitignore .dockerignore
  git commit -m "Clean repo: ignore generated artifacts and caches"

Review 'git status' before committing in case you want to keep any tracked artifact.

EOF
