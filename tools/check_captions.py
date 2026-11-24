#!/usr/bin/env python3
"""Check a labeled LoRA dataset folder against the caption rules, and optionally stage it for training.

  python tools/check_captions.py DATASET --trigger mystyle
  python tools/check_captions.py DATASET --trigger mystyle --stage TRAIN_DIR

Exits 1 when any check fails. macOS metadata files (._* and .DS_Store) are ignored and never staged.
--stage copies only image + caption pairs into TRAIN_DIR
(TRAIN_DIR must be new, empty, or made by an earlier --stage, whose pairs are replaced; diffusion-pipe tries to open every non-caption file in its folder as an image, so labels_report.csv must stay out).
"""
import argparse, csv, re, shutil, sys
from pathlib import Path

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
# .json, so diffusion-pipe skips it
STAGE_MARKER = "staged_by_check_captions.json"
BANNED = [
    "illustration", "drawing", "digital art", "2d", "anime", "cartoon", "comic", "manga",
    "cel shading", "flat colors", "lineart", "line art", "clean lines", "screentone", "inked", "stylized",
    "art style", "in the style of", "panel", "masterpiece", "best quality", "high quality", "detailed", "4k",
    "beautiful", "stunning",
]


def is_macos_junk(p):
    return p.name.startswith("._") or p.name == ".DS_Store"


def files(folder):
    return [p for p in folder.iterdir() if p.is_file() and not is_macos_junk(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--min-words", type=int, default=15)
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument("--min-side", type=int, default=1024)
    ap.add_argument("--stage", type=Path, help="copy image + caption pairs here when every check passes")
    a = ap.parse_args()

    errors, seen = [], {}
    junk = [p for p in a.dataset.iterdir() if is_macos_junk(p)]
    images = sorted(p for p in files(a.dataset) if p.suffix.lower() in IMAGES)
    captions = sorted(p for p in files(a.dataset) if p.suffix == ".txt")
    stems = {p.stem for p in images}
    prefix = f"{a.trigger}, "

    for t in captions:
        if t.stem not in stems:
            errors.append(f"{t.name}: caption without an image")
    try:
        from PIL import Image
    except ImportError:
        Image = None
        print("Pillow not installed: image sizes not checked")

    for img in images:
        t = img.with_suffix(".txt")
        if not t.exists():
            errors.append(f"{img.name}: no caption")
            continue
        if Image is not None:
            with Image.open(img) as im:
                if min(im.size) < a.min_side:
                    errors.append(f"{img.name}: shorter side {min(im.size)} px < {a.min_side}")
        raw = t.read_text(encoding="utf-8")
        cap = raw.strip()
        if "\n" in raw.rstrip("\n") or raw != raw.rstrip("\n"):
            errors.append(f"{t.name}: must be a single line with no trailing line break")
        if not cap.startswith(prefix):
            errors.append(f"{t.name}: does not start with '{prefix}'")
        if len(re.findall(rf"\b{re.escape(a.trigger)}\b", cap, re.I)) != 1:
            errors.append(f"{t.name}: '{a.trigger}' must appear exactly once")
        low = cap.lower()
        hits = [w for w in BANNED if re.search(rf"\b{re.escape(w)}\b", low)]
        if hits:
            errors.append(f"{t.name}: banned words {hits}")
        n = len(cap[len(prefix):].split()) if cap.startswith(prefix) else len(cap.split())
        if not a.min_words <= n <= a.max_words:
            errors.append(f"{t.name}: {n} words after the trigger (want {a.min_words}-{a.max_words})")
        if cap in seen:
            errors.append(f"{t.name}: same caption as {seen[cap]}")
        seen.setdefault(cap, t.name)

    report = a.dataset / "labels_report.csv"
    if report.exists():
        with report.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rejected = a.dataset / "rejected"
        n_rej = sum(1 for p in files(rejected) if p.suffix.lower() in IMAGES) if rejected.is_dir() else 0
        if len(rows) != len(images) + n_rej:
            errors.append(f"labels_report.csv: {len(rows)} rows for {len(images) + n_rej} images")

    print(f"{len(images)} images, {len(captions)} captions")
    if junk:
        print(f"ignored {len(junk)} macOS metadata files (._* or .DS_Store)")
    if len(images) < 30:
        print(f"warning: only {len(images)} images; 30 to 60 is the target for a style LoRA")
    for e in errors:
        print("FAIL", e)
    if errors:
        sys.exit(1)
    print("all checks passed")

    if a.stage:
        marker = a.stage / STAGE_MARKER
        if a.stage.resolve() == a.dataset.resolve():
            sys.exit("--stage must be a different folder from the dataset")
        if a.stage.exists() and any(a.stage.iterdir()) and not marker.exists():
            sys.exit(f"{a.stage} is not empty and was not made by --stage; pick an empty or new folder")
        a.stage.mkdir(parents=True, exist_ok=True)
        for old in a.stage.iterdir():
            if old.is_file() and (is_macos_junk(old) or old.suffix.lower() in IMAGES or old.suffix == ".txt"):
                old.unlink()
        marker.write_text("{}\n", encoding="utf-8")
        for img in images:
            shutil.copy2(img, a.stage / img.name)
            shutil.copy2(img.with_suffix(".txt"), a.stage / img.with_suffix(".txt").name)
        print(f"staged {len(images)} pairs in {a.stage}")


if __name__ == "__main__":
    main()
