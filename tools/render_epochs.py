#!/usr/bin/env python3
"""Queue the same seed and prompt in ComfyUI for every saved epoch of a LoRA, plus one render without it.

  python tools/render_epochs.py --project mystyle
  python tools/render_epochs.py --project mystyle --prompt "mystyle, ..." --seed 7 --strength 0.8
  python tools/render_epochs.py --check       only check that ComfyUI has the nodes and model files

The LoRAs must be in ComfyUI/models/loras as <project>_epochN.safetensors (README step 6).
Images land in ComfyUI/output/<project>_test/. Standard library only.
"""
import argparse, copy, json, re, sys, time, urllib.error, urllib.request
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / "workflows" / "h3_image_lora_test_api.json"
LOADERS = {"UNETLoader": ("unet_name", "diffusion_models"), "CLIPLoader": ("clip_name", "text_encoders"),
           "VAELoader": ("vae_name", "vae")}


def get(server, path):
    with urllib.request.urlopen(server + path, timeout=30) as r:
        return json.load(r)


def post_prompt(server, prompt):
    req = urllib.request.Request(server + "/prompt", data=json.dumps({"prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"ComfyUI refused the workflow: {e.read().decode(errors='replace')[:2000]}")


def wait_for(server, seconds):
    end = time.time() + seconds
    while True:
        try:
            return get(server, "/system_stats")
        except (urllib.error.URLError, OSError):
            if time.time() > end:
                sys.exit(f"no ComfyUI answering at {server}")
            time.sleep(2)


def check(server, wf):
    errors = []
    for node in sorted({n["class_type"] for n in wf.values()}):
        if node not in get(server, f"/object_info/{node}"):
            errors.append(f"node {node} missing: update ComfyUI")
    for n in wf.values():
        if n["class_type"] in LOADERS:
            key, folder = LOADERS[n["class_type"]]
            if n["inputs"][key] not in get(server, f"/models/{folder}"):
                errors.append(f"models/{folder}/{n['inputs'][key]} missing")
    for e in errors:
        print("FAIL", e)
    return not errors


def epoch_num(name):
    return int(re.search(r"_epoch(\d+)\.safetensors$", name).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--project", default="mystyle")
    ap.add_argument("--workflow", type=Path, default=WORKFLOW)
    ap.add_argument("--prompt", help="must start with '<project>, ' and name subjects not in the dataset")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for ComfyUI to start")
    a = ap.parse_args()
    server = a.server.rstrip("/")

    wf = json.loads(a.workflow.read_text(encoding="utf-8"))
    wait_for(server, a.wait)
    if not check(server, wf):
        sys.exit(1)
    if a.check:
        print(f"ComfyUI at {server} has the nodes and model files")
        return

    prompt_node = next(n for n in wf.values() if n["class_type"] == "MiniMaxH3ImageConditioning")
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    lora_id = next(k for k, n in wf.items() if n["class_type"] == "LoraLoaderModelOnly")
    save = next(n for n in wf.values() if n["class_type"] == "SaveImage")
    if a.prompt:
        prompt_node["inputs"]["prompt"] = a.prompt
    else:
        prompt_node["inputs"]["prompt"] = re.sub(r"^\w+, ", f"{a.project}, ", prompt_node["inputs"]["prompt"])
    if a.seed is not None:
        sampler["inputs"]["seed"] = a.seed

    pattern = re.compile(rf"^{re.escape(a.project)}_epoch\d+\.safetensors$")
    loras = sorted((n for n in get(server, "/models/loras") if pattern.match(n)), key=epoch_num)
    if not loras:
        sys.exit(f"no {a.project}_epochN.safetensors in ComfyUI/models/loras: copy them there first (README step 6)")

    base = copy.deepcopy(wf)
    base_model = base[lora_id]["inputs"]["model"]
    del base[lora_id]
    for n in base.values():
        if n["class_type"] == "KSampler":
            n["inputs"]["model"] = base_model
        if n["class_type"] == "SaveImage":
            n["inputs"]["filename_prefix"] = f"{a.project}_test/no_lora"
    print("queued no_lora", post_prompt(server, base))

    for name in loras:
        wf[lora_id]["inputs"].update(lora_name=name, strength_model=a.strength)
        save["inputs"]["filename_prefix"] = f"{a.project}_test/epoch{epoch_num(name)}"
        print("queued", name, post_prompt(server, wf))
    print(f"{len(loras) + 1} renders queued; images go to ComfyUI/output/{a.project}_test/")


if __name__ == "__main__":
    main()
