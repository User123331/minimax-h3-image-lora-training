# Training a style LoRA for MiniMax H3 Image

Sources are the vendors' own docs and price pages, read on 2026-11-24. Nothing here was trained or run by me; time and cost figures marked "estimate" are my arithmetic from those docs.

## 1. How Civitai trains LoRAs on its site

Civitai's trainer is now **Training Studio** (the classic trainer was retired on 2026-11-16). It is a web front end over open-source trainers running on Civitai's rented GPUs:

- **AI Toolkit** (Ostris) is the default engine.
- **Kohya** (sd-scripts) is an option for SD 1.5, SDXL/Pony/Illustrious, Anima, Flux.1 and Chroma.

The flow is three steps: pick type (Character / Style / Concept) and base model, upload images and label them (free auto-label, tags or captions depending on the model), review steps and price, then start. You pay in Buzz, get sample images per epoch, and download the best epoch as a `.safetensors` file.

**H3 Image is supported.** The Civitai changelog of 2026-11-20 says: "Train MiniMax H3 Image LoRAs. Pick MiniMax H3 Image in Training Studio and accept its license terms." A third-party change tracker lists a default run at about 1,450 Buzz; I could not see the price page itself (it needs a login), so confirm the quote on the Review step.

Links: [Training Studio guide](https://education.civitai.com/civitais-guide-to-training-studio/), [Civitai changelog](https://civitai.com/changelog).

## 2. Can you do the same on RunPod / Hyperbolic / Vast?

Yes. Civitai adds nothing you cannot run yourself: rent a GPU, install one trainer, point it at your dataset. Trainers with H3 Image support:

| Trainer | VRAM stated by its docs | Notes |
|---|---|---|
| [Unsloth Studio](https://unsloth.ai/docs/models/minimax-h3-image/fine-tune-minimax-h3-image) | ~24 GB peak (int8 base), ~46 GB peak (bf16 base) | Web UI, simplest. Downloads ~95 GB. |
| [SimpleTuner](https://github.com/bghira/SimpleTuner/releases) v4.12.1+ | ~68 GB peak in its 80 GB example | Exports ComfyUI format. |
| [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) | not stated | Config-file based; uses the ComfyUI int8 files plus the [circlestone training adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter). |
| [Fizgig](https://github.com/shootthesound/Fizgig) | 12 GB | Experimental for H3 Image. |
| [AcademiaSD LoRAlab](https://github.com/AcademiaSD/AcademiaSD_LoRAlab-Trainer) | 8 GB (NF4 + block swap) | Experimental for H3 Image. |

## 3. Dataset for a style LoRA

A style LoRA needs **one consistent look across varied content**. If most images show the same character or the same framing, the LoRA learns that character or framing instead of the style.

- **Count:** 30 to 60 images. Quality and variety matter more than count.
- **Source:** images from the one artist or source whose look you want. Mixing artists blurs the style.
- **Slice pages:** if the source comes as full pages, cut them into single panels; do not feed whole pages.
- **Clean each panel:** remove or crop out speech bubbles, sound-effect lettering, panel borders, watermarks and scanlation credits. Whatever stays in gets learned.
- **Vary the content:** different characters, genders and ages; close-ups, half-body, full-body and wide shots; interiors, exteriors, day and night; a few backgrounds or objects with no people.
- **Resolution:** at least 1024 px on the short side. H3 Image's VAE compresses more than most, so low-resolution panels lose the linework. Use the highest-quality source you have rather than upscaling compressed ones.
- **Aspect ratios:** mixed ratios are fine; the trainers bucket them.

Only train on images you have the right to use; Civitai requires you to confirm that for H3 runs.

## 4. Labelling (captions)

H3 Image uses a Qwen3-VL text encoder, so it wants **natural-language captions, not booru tags**.

- One `.txt` file per image with the same base name: `panel_001.png` and `panel_001.txt`.
- Start every caption with a unique trigger word, e.g. `mystyle`.
- **Describe the content, leave the style out.** Say who and what is in the panel, the pose, the shot, the setting and the lighting. Do not write style words such as "comic", "cel shading" or "clean lineart": anything you leave unsaid is absorbed by the trigger word, which is what you want.
- One or two sentences is enough.

Example: `mystyle, a young man in a black coat stands on a rooftop at night, seen from below, city lights behind him, wind blowing his hair`

To do this fast: let a vision model draft captions (Civitai's and Unsloth's built-in auto-label, or JoyCaption / Qwen-VL locally), then strip out the style words and add the trigger by hand.

## 5. How to train, and do you need the model?

**Yes, you need the base model.** A LoRA is trained against the weights it will later run on, so the trainer loads H3 Image itself (it downloads the files for you). You train on H3 Image, not on Illustrious or any other model.

1. Rent a GPU pod with about 150 GB of disk (the model files alone are 40 to 95 GB depending on the trainer).
2. Install the trainer and upload the dataset folder.
3. Keep the trainer's **training adapter** on. H3 Image is distilled to 6 steps; training without the adapter breaks that distillation. The adapter is used only during training and is not part of your LoRA.
4. Starting settings: 1024 px, rank 16 to 32, learning rate 5e-5, 150 to 500 steps (Unsloth's default is 150; its docs warn that more can overcook the style). Set a sample prompt with the trigger word every 50 steps.
5. Stop when previews of subjects that are **not** in the dataset look right. Keep that checkpoint, not necessarily the last.
6. Export in ComfyUI format, copy the `.safetensors` to `models/loras` (or list its URL in `extra_loras.txt` in the workflow repo), and load it at strength 0.6 to 1.0.

## 6. Is an H100/H200 needed? Cost per option

An H100 is comfortable; an **H200 is not needed** for a LoRA. Small cards work with the pruned/quantized base, more slowly and with experimental trainers.

Measured speeds from Unsloth's docs: 1.5 steps/s on an RTX PRO 6000 (96 GB, bf16); 0.9 steps/s bf16 and 0.8 steps/s int8 on an A100 80 GB; about 8 minutes end to end for a 150-step run after the download. No published speed exists for the 8 to 24 GB paths.

| Option | Price per hour (2026-11-24) | Fits | One run, estimate |
|---|---|---|---|
| Civitai Training Studio | ~1,450 Buzz per default run (third-party figure) | no setup | one Buzz payment, no hourly bill |
| A100 80 GB, Vast | from $0.52 | bf16 base, any trainer | ~1 h, about $0.50 to $1 |
| H100 80 GB, Vast / RunPod community | from $1.73 / $1.99 | bf16 base, any trainer | ~1 h, about $2 |
| H100 80 GB, RunPod secure / Hyperbolic | $2.89 / $3.19 | same | ~1 h, about $3 |
| H200 141 GB, Vast / RunPod / Hyperbolic | $3.56 / $3.59 to $5.29 / $3.99 | more than needed | ~1 h, about $4 to $5 |
| RTX A6000 48 GB, RunPod / Vast | $0.33 to $0.69 / $0.40 | bf16 is tight (46 GB peak); int8 fits | 1 to 2 h, under $1.50 |
| RTX 5090 32 GB, Vast / RunPod | $0.47 / $0.69 to $1.39 | int8 base (24 GB peak) | 1 to 2 h, under $2 |
| RTX 4090 24 GB, RunPod / Vast | $0.34 to $0.89 / $0.40 | int8 has no headroom; use Fizgig or AcademiaSD | unknown speed; likely several hours |

The "one run" column assumes about 30 to 40 minutes of download and setup plus a 150 to 500 step run, and that you delete the pod afterwards. Most of the bill is download time, not training. Expect two or three runs before you are happy with the result.

**My pick:** try Civitai first if you want zero setup. For doing it yourself, an A100 80 GB on Vast or an H100 on RunPod community with Unsloth Studio is the least fiddly; it is also the same class of machine the workflow's `max` profile runs on, so you can test the LoRA on the same pod.

Price sources: [RunPod pricing](https://www.runpod.io/pricing), [Hyperbolic marketplace](https://www.hyperbolic.ai/marketplace), [Vast.ai rates as collected by WeHosts](https://wehosts.cn/en/gpu-prices). Marketplace prices move daily.
