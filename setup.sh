#!/usr/bin/env bash
# Installs diffusion-pipe in its own venv, downloads the MiniMax H3 Image training files
# and creates a project folder with the example configs.
#
#   bash setup.sh --dry-run      show what it would do
#   bash setup.sh                install (safe to run again; it reuses what is there)
#
# Overrides (environment):
#   WORKDIR=/workspace           everything goes under here
#   PROJECT=mystyle              project folder: $WORKDIR/lora/$PROJECT
#   MODELS=$WORKDIR/models       where the model files go
#   COMFY_MODELS=$WORKDIR/ComfyUI/models   files already here are linked, not downloaded again
#   HF_TOKEN=...                 optional; the files are public
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
PROJECT="${PROJECT:-mystyle}"
LORA_DIR="$WORKDIR/lora"
PROJ="$LORA_DIR/$PROJECT"
MODELS="${MODELS:-$WORKDIR/models}"
COMFY_MODELS="${COMFY_MODELS:-$WORKDIR/ComfyUI/models}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
VENV="$LORA_DIR/venv"
DP="$LORA_DIR/diffusion-pipe"
DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# repo  path-in-repo  local path under $MODELS  size GB
FILES=(
  "Comfy-Org/MiniMax-H3-Image|diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors|diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors|21.0"
  "Comfy-Org/MiniMax-H3-Image|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|27.1"
  "Comfy-Org/MiniMax-H3-Image|vae/minimax_h3_image_vae_fp16.safetensors|vae/minimax_h3_image_vae_fp16.safetensors|1.2"
  "circlestone-labs/MiniMax-H3-Image-Undistill-Lora|minimax_h3_image_undistill_lora.safetensors|adapters/minimax_h3_image_undistill_lora.safetensors|0.6"
)

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
else
  GPU="none found"; (( DRY )) || die "nvidia-smi not found: run this on the GPU machine"
fi
(( DRY )) || mkdir -p "$WORKDIR" || die "cannot create $WORKDIR"
D="$WORKDIR"; while [[ ! -d "$D" ]]; do D="$(dirname "$D")"; done
FREE=$(df -BG --output=avail "$D" | tail -n1 | tr -dc '0-9')

say "GPU: $GPU · $FREE GB free in $WORKDIR"
NEED=0
for f in "${FILES[@]}"; do
  IFS='|' read -r repo src dst gb <<<"$f"
  if [[ -e "$MODELS/$dst" ]]; then state="present"
  elif [[ -e "$COMFY_MODELS/$src" ]]; then state="link from $COMFY_MODELS"
  elif [[ -e "$COMFY_MODELS/$dst" ]]; then state="link from $COMFY_MODELS"
  else state="download ${gb} GB"; NEED=$(python3 -c "print($NEED + $gb)"); fi
  printf '  %-58s %s\n' "$dst" "$state"
done
[[ -d "$VENV" ]] || NEED=$(python3 -c "print($NEED + 15)")
cat <<INFO
  diffusion-pipe   $DP
  venv             $VENV
  project          $PROJ
  disk needed      about ${NEED} GB (models + venv), plus room for cache and checkpoints
INFO
if python3 -c "import sys; sys.exit(0 if $FREE >= $NEED + 20 else 1)"; then :; else
  warn "only $FREE GB free; want at least $(python3 -c "print(round($NEED + 20))") GB"
  [[ "${FORCE:-0}" == 1 || $DRY == 1 ]] || die "set FORCE=1 to continue anyway"
fi
(( DRY )) && { say "dry run: nothing changed"; exit 0; }

say "diffusion-pipe and its venv"
mkdir -p "$LORA_DIR" "$MODELS"
[[ -d "$DP/.git" ]] || git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe "$DP"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q -U pip huggingface_hub

say "model files (in the background)"
mkdir -p "$LORA_DIR/logs"
pids=()
for f in "${FILES[@]}"; do
  IFS='|' read -r repo src dst gb <<<"$f"
  [[ -e "$MODELS/$dst" ]] && continue
  mkdir -p "$(dirname "$MODELS/$dst")"
  for c in "$COMFY_MODELS/$src" "$COMFY_MODELS/$dst"; do
    [[ -e "$c" ]] && { ln -s "$c" "$MODELS/$dst"; echo "  linked $dst"; continue 2; }
  done
  tmp="$MODELS/.hf/$(dirname "$dst")"; mkdir -p "$tmp"
  ( "$VENV/bin/hf" download "$repo" "$src" --local-dir "$tmp" >"$LORA_DIR/logs/dl_$(basename "$dst").log" 2>&1 \
      && mv "$tmp/$src" "$MODELS/$dst" ) & pids+=($!)
done

say "PyTorch (CUDA 13) and diffusion-pipe requirements"
"$PY" -m pip install -q torch torchvision --index-url "$TORCH_INDEX"
"$PY" -m pip install -q -r "$DP/requirements.txt"
"$PY" -m pip install -q tensorboard pillow

say "waiting for downloads"
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && die "a download failed: see $LORA_DIR/logs/dl_*.log (401/403: set HF_TOKEN)"
rm -rf "$MODELS/.hf"

say "project folder"
mkdir -p "$PROJ"/{dataset,train,output}
for c in dataset.toml config.toml; do
  if [[ -e "$PROJ/$c" ]]; then echo "  $PROJ/$c exists, left as is"; continue; fi
  sed -e "s|/workspace/lora/mystyle|$PROJ|g" -e "s|/workspace/models|$MODELS|g" "$ROOT/configs/$c" >"$PROJ/$c"
  echo "  wrote $PROJ/$c"
done

"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '·', torch.cuda.get_device_name(0))"
say "done"
cat <<INFO
  put the labeled images in   $PROJ/dataset
  check and stage them        $PY $ROOT/tools/check_captions.py $PROJ/dataset --trigger $PROJECT --stage $PROJ/train
  train                       see README.md, step 6
INFO
