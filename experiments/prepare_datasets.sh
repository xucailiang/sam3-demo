#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"

CRACKFOREST_URL="${CRACKFOREST_URL:-https://github.com/cuilimeng/CrackForest-dataset.git}"
DEEPCRACK_URL="${DEEPCRACK_URL:-https://github.com/yhlleo/DeepCrack.git}"

echo "Preparing crack segmentation datasets..."
echo "Project:  $PROJECT_ROOT"
echo "Datasets: $DATASETS_DIR"

mkdir -p "$DATASETS_DIR"

clone_or_update() {
  local url="$1"
  local dir="$2"
  local name="$3"

  if [ -d "$dir/.git" ]; then
    echo "Updating $name..."
    git -C "$dir" pull --ff-only
  elif [ -e "$dir" ]; then
    echo "ERROR: $dir exists but is not a git repository." >&2
    echo "Move it aside or set ${name}_URL to a compatible source." >&2
    exit 1
  else
    echo "Cloning $name..."
    git clone "$url" "$dir"
  fi
}

clone_or_update "$CRACKFOREST_URL" "$DATASETS_DIR/CrackForest-dataset" "CrackForest"
clone_or_update "$DEEPCRACK_URL" "$DATASETS_DIR/DeepCrack" "DeepCrack"

DEEPCRACK_DATASET_DIR="$DATASETS_DIR/DeepCrack/dataset"
if [ -f "$DEEPCRACK_DATASET_DIR/DeepCrack.zip" ] && [ ! -d "$DEEPCRACK_DATASET_DIR/DeepCrack" ]; then
  echo "Extracting DeepCrack.zip..."
  (
    cd "$DEEPCRACK_DATASET_DIR"
    unzip -q DeepCrack.zip
  )
fi

if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  echo "Creating Python 3.12 uv environment..."
  uv venv --python 3.12 "$PROJECT_ROOT/.venv"
fi

echo "Installing/checking experiment dependencies..."
uv pip install -q -r "$PROJECT_ROOT/requirements-experiments.txt"

echo "Validating dataset layout..."
"$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/validate_datasets.py"

echo "Done."
