#!/usr/bin/env bash
# Installs diffusion-pipe in its own venv, downloads the MiniMax H3 Image training files
# and creates a project folder with the example configs.
#
#   bash setup.sh --dry-run      show what it would do
#   bash setup.sh                install (safe to run again; it reuses what is there)
#   bash setup.sh --comfyui      also install ComfyUI, with the model files linked in, to test the LoRA
#
# Missing system packages (python3.X-venv, python3.X-dev, unzip) are installed with apt-get.
#
# Overrides (environment):
#   WORKDIR=/workspace           everything goes under here
#   PROJECT=mystyle              project folder: $WORKDIR/lora/$PROJECT
#   MODELS=$WORKDIR/models       where the model files go
#   COMFY=$WORKDIR/ComfyUI       ComfyUI folder (--comfyui installs it here)
#   COMFY_MODELS=$COMFY/models   files already here are linked, not downloaded again
#   HF_TOKEN=...                 optional; the files are public
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-/workspace}"
PROJECT="${PROJECT:-mystyle}"
LORA_DIR="$WORKDIR/lora"
PROJ="$LORA_DIR/$PROJECT"
MODELS="${MODELS:-$WORKDIR/models}"
COMFY="${COMFY:-$WORKDIR/ComfyUI}"
COMFY_MODELS="${COMFY_MODELS:-$COMFY/models}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
VENV="$LORA_DIR/venv"
DP="$LORA_DIR/diffusion-pipe"
DRY=0; WITH_COMFY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --comfyui) WITH_COMFY=1 ;;
    *) echo "unknown option: $arg (use --dry-run, --comfyui)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# A venv whose creation was interrupted (or failed for lack of python3.X-venv) has a python but no pip:
# start it again instead of reusing it.
make_venv() {
  if [[ -d "$1" ]] && ! "$1/bin/python" -m pip --version >/dev/null 2>&1; then
    warn "$1 is incomplete (an earlier run stopped while creating it): recreating it"
    rm -rf "$1"
  fi
  [[ -d "$1" ]] || python3 -m venv "$1" || { rm -rf "$1"; die "could not create the venv $1"; }
}

# repo  path-in-repo  local path under $MODELS  size GB
FILES=(
  "Comfy-Org/MiniMax-H3-Image|diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors|diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors|21.0"
  "Comfy-Org/MiniMax-H3-Image|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|27.1"
  "Comfy-Org/MiniMax-H3-Image|vae/minimax_h3_image_vae_fp16.safetensors|vae/minimax_h3_image_vae_fp16.safetensors|1.2"
  "circlestone-labs/MiniMax-H3-Image-Undistill-Lora|minimax_h3_image_undistill_lora.safetensors|adapters/minimax_h3_image_undistill_lora.safetensors|0.6"
)

# venv needs python3.X-venv; torch.compile builds Triton helpers against Python.h (python3.X-dev);
# unzip is for dataset archives. PyTorch pod images do not always ship them.
PYV="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
APT=()
T="$(mktemp -d)"
python3 -m venv "$T/v" >/dev/null 2>&1 && "$T/v/bin/python" -m pip --version >/dev/null 2>&1 || APT+=("python$PYV-venv")
rm -rf "$T"
[[ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')/Python.h" ]] || APT+=("python$PYV-dev")
command -v unzip >/dev/null 2>&1 || APT+=(unzip)

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"
else
  GPU="none found"; (( DRY )) || die "nvidia-smi not found: run this on the GPU machine"
fi
(( DRY )) || mkdir -p "$WORKDIR" || die "cannot create $WORKDIR"
D="$WORKDIR"; while [[ ! -d "$D" ]]; do D="$(dirname "$D")"; done
FREE=$(df -BG --output=avail "$D" | tail -n1 | tr -dc '0-9')
MNT=$(df --output=target "$D" | tail -n1)

say "GPU: $GPU · $FREE GB free in $WORKDIR (filesystem mounted at $MNT)"
[[ "$MNT" == / ]] && warn "$WORKDIR is on the root filesystem, not a separate volume: the free space above is the root disk's, and files outside a volume may not survive a pod restart"
if (( ${#APT[@]} )); then echo "  system packages to install: ${APT[*]}"; fi
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
(( WITH_COMFY )) && [[ ! -d "$COMFY/venv" ]] && NEED=$(python3 -c "print($NEED + 10)")
cat <<INFO
  diffusion-pipe   $DP
  venv             $VENV
  project          $PROJ
INFO
(( WITH_COMFY )) && echo "  ComfyUI          $COMFY"
cat <<INFO
  disk needed      about ${NEED} GB (models + venvs), plus room for cache and checkpoints
INFO
if python3 -c "import sys; sys.exit(0 if $FREE >= $NEED + 20 else 1)"; then :; else
  warn "only $FREE GB free; want at least $(python3 -c "print(round($NEED + 20))") GB"
  [[ "${FORCE:-0}" == 1 || $DRY == 1 ]] || die "set FORCE=1 to continue anyway"
fi
(( DRY )) && { say "dry run: nothing changed"; exit 0; }

if (( ${#APT[@]} )); then
  say "system packages: ${APT[*]}"
  SUDO=(); (( EUID == 0 )) || SUDO=(sudo)
  command -v apt-get >/dev/null 2>&1 || die "install ${APT[*]} with your package manager, then run setup.sh again"
  "${SUDO[@]}" apt-get update -qq && DEBIAN_FRONTEND=noninteractive "${SUDO[@]}" apt-get install -y -qq "${APT[@]}" >/dev/null \
    || die "could not install ${APT[*]}"
fi

say "diffusion-pipe and its venv"
mkdir -p "$LORA_DIR" "$MODELS"
# one submodule's LFS images fail to download, which aborts a normal recursive clone
export GIT_LFS_SKIP_SMUDGE=1
[[ -d "$DP/.git" ]] || git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe "$DP"
git -C "$DP" submodule update --init --recursive || warn "some diffusion-pipe submodules did not check out"
[[ -f "$DP/submodules/ComfyUI/comfy/sd.py" && -d "$DP/submodules/HunyuanVideo/hyvideo" ]] \
  || die "diffusion-pipe submodules are missing: delete $DP and run setup.sh again"
make_venv "$VENV"
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
"$PY" -m pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX"
"$PY" -m pip install -q -r "$DP/requirements.txt"
"$PY" -m pip install -q tensorboard pillow

say "waiting for downloads"
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && die "a download failed: see $LORA_DIR/logs/dl_*.log (401/403: set HF_TOKEN)"
rm -rf "$MODELS/.hf"

if (( WITH_COMFY )); then
  say "ComfyUI for testing"
  [[ -d "$COMFY/.git" ]] || git clone https://github.com/comfyanonymous/ComfyUI "$COMFY"
  make_venv "$COMFY/venv"
  CPY="$COMFY/venv/bin/python"
  "$CPY" -m pip install -q -U pip
  "$CPY" -m pip install -q torch torchvision torchaudio --index-url "$TORCH_INDEX"
  "$CPY" -m pip install -q -r "$COMFY/requirements.txt"
  for f in "${FILES[@]}"; do
    IFS='|' read -r repo src dst gb <<<"$f"
    [[ "$repo" == Comfy-Org/* ]] || continue   # the Undistill adapter is for training only
    [[ -e "$COMFY/models/$src" ]] && continue
    mkdir -p "$(dirname "$COMFY/models/$src")"
    ln -s "$MODELS/$dst" "$COMFY/models/$src"; echo "  linked $src"
  done
  mkdir -p "$COMFY/models/loras"
  "$CPY" "$COMFY/main.py" --listen 127.0.0.1 --port 8189 ${COMFY_ARGS:-} >"$LORA_DIR/logs/comfyui_check.log" 2>&1 & cpid=$!
  "$PY" "$ROOT/tools/render_epochs.py" --server http://127.0.0.1:8189 --check --wait 300 \
    || { kill "$cpid" 2>/dev/null; die "ComfyUI check failed: see $LORA_DIR/logs/comfyui_check.log"; }
  kill "$cpid"; wait "$cpid" 2>/dev/null || true
fi

say "project folder"
mkdir -p "$PROJ"/{dataset,train,output}
for c in dataset.toml config.toml; do
  if [[ -e "$PROJ/$c" ]]; then echo "  $PROJ/$c exists, left as is"; continue; fi
  sed -e "s|/workspace/lora/mystyle|$PROJ|g" -e "s|/workspace/models|$MODELS|g" "$ROOT/configs/$c" >"$PROJ/$c"
  echo "  wrote $PROJ/$c"
done

"$PY" -c "import torch, torchaudio; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '·', torch.cuda.get_device_name(0))"
say "done"
cat <<INFO
  put the labeled images in   $PROJ/dataset
  check and stage them        $PY $ROOT/tools/check_captions.py $PROJ/dataset --trigger $PROJECT --stage $PROJ/train
  train                       see README.md, step 5
INFO
(( WITH_COMFY )) && echo "  test in ComfyUI             see README.md, step 6"
true
