#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CUZR_REPO_URL:-https://github.com/KirillTenikov/CuZr.git}"
REPO_BRANCH="${CUZR_REPO_BRANCH:-master}"
REPO_DIR="${CUZR_REPO_DIR:-/workspace/CuZr}"

mkdir -p /workspace /workspace/data /workspace/outputs /workspace/checkpoints /workspace/logs

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[on-start] Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "[on-start] Repo already exists at ${REPO_DIR}; updating"
  git -C "$REPO_DIR" fetch origin
  git -C "$REPO_DIR" checkout "$REPO_BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH" || true
fi

cat <<EOF

[on-start] CuZr training workspace is ready.

  Repo:      $REPO_DIR
  Data dir:  /workspace/data
  Outputs:   /workspace/outputs

Next steps after SSH login:
  cd $REPO_DIR
  python scripts/train_mace.py --config configs/mace_A.yaml --output-root /workspace/outputs/mace

EOF
