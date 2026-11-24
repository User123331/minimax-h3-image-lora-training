# MiniMax H3 Image LoRA training on one H100

Train a style LoRA for MiniMax H3 Image (example trigger word: `mystyle`) on a rented H100 80 GB with [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe), then test it in ComfyUI.

**Status: nothing here has been run on a GPU yet.** Commands and settings come from the docs of [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe), its [MiniMax H3 notes](https://github.com/tdrussell/diffusion-pipe/blob/main/docs/minimax_h3_notes.md), the [Undistill adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Undistill-Lora) and [Unsloth's H3 Image training page](https://unsloth.ai/docs/models/minimax-h3-image/fine-tune-minimax-h3-image), read on 2026-11-24. Times and costs are estimates.

## Repo

| Path | What it is |
|---|---|
| `setup.sh` | Installs diffusion-pipe in its own venv, downloads the training files, creates the project folder with the configs |
| `configs/config.toml`, `configs/dataset.toml` | Training and dataset configs (copied into the project folder by `setup.sh`) |
| `tools/check_captions.py` | Checks a labeled dataset and stages image + caption pairs for training |
| `docs/dataset-labeling-guide.md` | How to label the dataset (for a vision agent) |
| `docs/minimax-h3-image-lora-training-guide.md` | Research notes: trainers, dataset, settings, GPU prices |

## Overview

| Step | What | Time (estimate) |
|---|---|---|
| 1 | Rent the pod | 5 min |
| 2 | `bash setup.sh`: diffusion-pipe, the model files, the project folder | 20 to 40 min (downloads) |
| 3 | Upload, label and check the dataset | depends on the dataset; label before renting if you can |
| 4 | Look over the configs | 5 min |
| 5 | Cache, then train | caching a few minutes, training about an hour for 40 images |
| 6 | Test the saved epochs in ComfyUI, keep the best | 15 to 30 min |
| 7 | Download the LoRA, delete the pod | 5 min |

At RunPod's H100 prices ($1.99 to $2.89 per hour on 2026-11-22) one full pass is about 2 to 3 hours, so $4 to $9. Expect two or three training runs before the result is right; reuse the pod and the cache between them.

Why these choices:

- **diffusion-pipe** is config-file based (repeatable, no web UI to expose), trains directly on the ComfyUI `int8_convrot` files, and saves LoRAs in ComfyUI format.
- **int8 base, not bf16**: diffusion-pipe recommends training on `int8_convrot` (faster, less VRAM, and it says better quality). The LoRA it saves works on the bf16 and int8 checkpoints alike (AdaLN weights are not trained).
- **Undistill adapter**: H3 Image is distilled to 6 steps without CFG; plain training breaks that. The adapter is merged into the base only during training and is not part of your LoRA, so the LoRA runs in a normal 6-step workflow.

## 1. Rent the pod

- **GPU:** 1× H100 80 GB (SXM, PCIe or NVL). A100 80 GB works with the same steps, more slowly.
- **Template:** a PyTorch template with CUDA 13 and a `-devel` image (it has `nvcc`, which DeepSpeed may need).
- **Volume:** 150 GB on `/workspace`: models 50 GB, the venv about 15 GB, cache, checkpoints and outputs.
- **HTTP ports:** whatever you use to move files and test (for example ComfyUI on 8188). Adding a port later restarts the pod.

```bash
nvidia-smi                     # H100, driver 580 or newer
df -h /workspace
```

## 2. Install

```bash
cd /workspace
git clone https://github.com/User123331/minimax-h3-image-lora-training && cd minimax-h3-image-lora-training
bash setup.sh --dry-run        # GPU, files, disk
bash setup.sh
```

It creates:

| Path | What |
|---|---|
| `/workspace/lora/diffusion-pipe`, `/workspace/lora/venv` | the trainer, with PyTorch for CUDA 13 (its int8 kernels need it) |
| `/workspace/models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` | training base, 21.0 GB |
| `/workspace/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | text encoder, 27.1 GB |
| `/workspace/models/vae/minimax_h3_image_vae_fp16.safetensors` | VAE, 1.2 GB |
| `/workspace/models/adapters/minimax_h3_image_undistill_lora.safetensors` | Undistill adapter, 0.6 GB |
| `/workspace/lora/mystyle/` | project: `dataset/`, `train/`, `output/`, `config.toml`, `dataset.toml` |

Overrides are at the top of `setup.sh` (`PROJECT=` for another style name, `WORKDIR=`, `MODELS=`). If ComfyUI is already on the pod with these files in `/workspace/ComfyUI/models`, they are linked, not downloaded again; so on a fresh pod where you also want to test, install ComfyUI first.

## 3. Dataset

1. **Collect** 30 to 60 images of the one style (one artist or one source), at least 1024 px on the short side, varied in characters, framing and setting. Details in [the research notes](docs/minimax-h3-image-lora-training-guide.md), section 3.
2. **Upload** them to `/workspace/lora/mystyle/dataset/` (File Browser, JupyterLab, `rsync` or `scp`). Labeling does not need the GPU: do steps 2 and 3 on your own machine before renting if you can, and upload the finished folder.
3. **Label** with [the labeling guide](docs/dataset-labeling-guide.md): give it and the folder to a vision agent, using the prompt in its section 9. It writes one `<name>.txt` per kept image starting with `mystyle, `, moves rejects to `rejected/`, and writes `labels_report.csv`.
4. **Check and stage:**

```bash
cd /workspace/minimax-h3-image-lora-training
/workspace/lora/venv/bin/python tools/check_captions.py /workspace/lora/mystyle/dataset --trigger mystyle \
  --stage /workspace/lora/mystyle/train
```

It runs the checks from section 7 of the labeling guide (a caption per image, trigger once at the start, no style or quality words, 15 to 50 words, one line, no duplicates, short side 1024 px+, one report row per image) and exits with the list of failures. Only when everything passes does it replace the image + caption pairs in `train/` with the current ones, so panels rejected since the last run drop out. Train from `train/`, not `dataset/`: diffusion-pipe tries to load every file in its folder that is not a caption as an image, so `labels_report.csv` would stop it.

## 4. Configs

`setup.sh` copied [configs/config.toml](configs/config.toml) and [configs/dataset.toml](configs/dataset.toml) into the project folder with the paths filled in. Edit the copies there. How the numbers were picked:

| Setting | Value | Source and when to change it |
|---|---|---|
| `rank` | 32 | Unsloth's default for H3 Image; 16 for a lighter LoRA |
| `lr` | 5e-5 | Unsloth's default. diffusion-pipe's own H3 example uses 1e-4; try it if the style is still weak at the last epoch |
| `epochs`, `save_every_n_epochs` | 40, 2 | Unsloth's default run is 150 steps at batch 1 (150 image views) and warns that more can overcook the style. Here an epoch is 40 image views, so epochs 4 to 12 cover Unsloth's range and the later ones show where it overcooks. For another dataset size keep about 1,600 image views in total |
| `image_micro_batch_size_per_gpu`, `gradient_accumulation_steps` | 1, 1 | Batch 1, as in Unsloth's default run. diffusion-pipe trims each aspect-ratio bucket to a multiple of batch × accumulation, so with 9 buckets anything larger can leave small buckets out of training. Raise them only for a dataset of one aspect ratio |
| `resolutions` | 1024 | diffusion-pipe: the H3 VAE compresses 2x more than most, so train at 1024 or more |
| `timestep_sample_method`, `merge_adapters` | as in the file | diffusion-pipe's MiniMax-H3-Image section. Keep out any `cfg = ...` line: the adapter's README says not to combine it with CFG-augmented training |

diffusion-pipe makes no preview images during training; you judge the saved epochs in ComfyUI in step 6.

## 5. Cache, then train

Stop anything else using the GPU, such as ComfyUI.

diffusion-pipe's notes say the text encoder's RAM is not fully freed after caching and can run the machine out of memory, so cache and train in two runs:

```bash
cd /workspace/lora/diffusion-pipe && source ../venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --num_gpus=1 train.py --deepspeed --config /workspace/lora/mystyle/config.toml --cache_only
deepspeed --num_gpus=1 train.py --deepspeed --config /workspace/lora/mystyle/config.toml --trust_cache
```

- Run them in `tmux` or JupyterLab's terminal so a dropped connection does not stop the run.
- `--trust_cache` reuses the cache without rechecking the files. After you change, add or remove images or captions, run the caching command once with `--regenerate_cache`: the cache check looks at file names and captions, not pixels, so an image edited in place keeps its old latents otherwise.
- Watch the loss in the log, or in TensorBoard kept off the public proxy (it has no login): `tensorboard --logdir /workspace/lora/mystyle/output --host 127.0.0.1 --port 6006` on the pod, `ssh -L 6006:127.0.0.1:6006 <pod ssh address>` on your machine, then open `http://localhost:6006`.
- Resume an interrupted run with the training command plus `--resume_from_checkpoint`.
- Each run creates `output/<date>/`, with `epoch2/`, `epoch4/`, ... each holding `adapter_model.safetensors` in ComfyUI format.

## 6. Test the epochs in ComfyUI

You need a ComfyUI with MiniMax H3 Image support on this pod, at `/workspace/ComfyUI`, with an H3 Image workflow that has a LoRA loader.

Copy each saved LoRA under a name that says its epoch:

```bash
RUN=$(ls -td /workspace/lora/mystyle/output/*/ | head -n1)
for d in "$RUN"epoch*/; do e=$(basename "$d"); cp "$d/adapter_model.safetensors" "/workspace/ComfyUI/models/loras/mystyle_$e.safetensors"; done
```

Then in ComfyUI:

1. Switch off any other LoRA while judging, so only your style is in the image.
2. Load `mystyle_epochN.safetensors` at strength 1.0.
3. Fix the seed, size 1024 × 1024 (the training size; Unsloth found the style weaker at 2048), and prompt with subjects that are **not** in the dataset, starting with `mystyle, `. For example: `mystyle, an old fisherman in a yellow raincoat pulls a net onto a small boat at dawn, seen from the side, gulls overhead`.
4. Render the same seed and prompt for each epoch, earliest to latest.

Keep the earliest epoch where the style holds on new subjects. Signs of too far: every face looks like one character from the dataset, colours burn out, or prompts without `mystyle` change too. Then try strength 0.6 to 0.8, and other LoRAs back on if you want them together.

## 7. Keep the LoRA, delete the pod

- Download the chosen `mystyle_epochN.safetensors`, or put it on Hugging Face or Civitai.
- Copy the project's `config.toml`, `dataset.toml` and `train/` off the pod if you will train again.
- Delete the pod. A stopped pod keeps billing for its volume ($0.20/GB/month, about $30/month for 150 GB).

## Problems

| Symptom | Cause and fix |
|---|---|
| Killed, or out of memory after the caching phase | Text encoder RAM not freed: run the training command again; the cache is kept |
| CUDA out of memory in training | Something else on the GPU (step 5), or not enough room: add `blocks_to_swap = 16` |
| Error opening `labels_report.csv` or another non-image file | Training from `dataset/`: train from `train/` (step 3) |
| `int8` / `convrot` kernel errors | PyTorch is not a CUDA 13 build: `/workspace/lora/venv/bin/pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| DeepSpeed complains about NCCL | Prefix the command with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` |
| Images need more steps or CFG with the LoRA | Distillation broke: check `merge_adapters` points at the Undistill adapter and that there is no `cfg` line |
| Style too strong, same face everywhere | Earlier epoch, lower strength, or a more varied dataset |
| Style too weak at the last epoch | `lr = 1e-4`, or more epochs |

## Alternative: Unsloth Studio

[Unsloth Studio](https://unsloth.ai/docs/models/minimax-h3-image/fine-tune-minimax-h3-image) trains H3 Image LoRAs from a web UI with preview images during training: Images → Train, Model family `MiniMax-H3-Image`, upload the captioned images, defaults 150 steps, rank 32, 1024, LR 5e-5. It needs Unsloth 2026.11.5 or newer and about 95 GB of its own downloads (it does not reuse these files). Install with `curl -fsSL https://unsloth.ai/install.sh | sh` and start with `unsloth studio -H 0.0.0.0 -p <port>` on a port exposed on the pod (8888 is taken if JupyterLab runs). The `.safetensors` it exports loads in ComfyUI the same way as in step 6.
