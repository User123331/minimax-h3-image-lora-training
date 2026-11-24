# Labeling guide for a style LoRA dataset

This guide is written for an AI agent with vision. Give it this file and the dataset folder. It assumes a **style** LoRA for MiniMax H3 Image with the trigger word `mystyle`; use your own trigger word in its place. Section 8 says what changes for a character LoRA.

Nothing in this guide was tested in a training run. The rules come from the trainers' documentation and common LoRA practice.

## 1. Goal

For every image in the dataset folder, write one caption file that:

- starts with the trigger word `mystyle`,
- describes what is in the picture in plain English,
- says nothing about the art style.

The trainer links everything the caption does not mention to the trigger word. The style is left out on purpose, so `mystyle` becomes the style.

## 2. Input and output

- **Input:** a folder of images (`.png`, `.jpg`, `.jpeg`, `.webp`).
- **Output per kept image:** a `.txt` file with the same base name, in the same folder. `panel_001.png` gets `panel_001.txt`.
- **Output per rejected image:** no `.txt`. Move the image to a `rejected/` subfolder. Never delete an image.
- **Report:** `labels_report.csv` in the dataset folder with the columns `filename,status,reason,caption`. `status` is `kept` or `rejected`.

Caption files are UTF-8, one line, no line break at the end, no quotes around the text.

## 3. Work one image at a time

1. List the image files and sort them by name.
2. Open one image and look at it. Never write a caption from the filename, from a neighbouring image, or from memory of the series.
3. Run the quality check in section 4.
4. If the image passes, write the caption with the rules in sections 5 and 6.
5. Save the `.txt`, add the row to `labels_report.csv`, then move to the next image.
6. If the folder already has a `.txt` for an image, read the image again and rewrite the caption to these rules. Do not trust the old one.

Do not caption several images in one step. Do not reuse a caption for a similar image. Each caption must come from looking at that image.

## 4. Quality check (reject or flag)

Reject the image, with the reason in the report, when any of these is true:

| Reason code | What you see |
|---|---|
| `text` | Speech bubbles, narration boxes, sound-effect lettering, watermarks or credits cover a visible part of the picture |
| `multi_panel` | More than one panel in the image |
| `low_res` | The shorter side is under 1024 px |
| `blurry` | Heavy compression artifacts, blur, or scan noise |
| `duplicate` | Same panel as an earlier image, or only a tiny crop difference |
| `unreadable` | You cannot tell what the picture shows |

Do not fix images yourself. Do not crop, upscale or inpaint. Only sort them.

If a small piece of text remains in a corner and the picture is otherwise good, keep it and add `small_text` in the `reason` column so a person can clean it later.

## 5. Caption rules

**Format:** `mystyle, <one or two sentences>`

- The caption always begins with exactly `mystyle, ` in lowercase. The trigger word appears once.
- Write full natural-language sentences. H3 Image reads sentences, so do not write comma-separated booru tags.
- Length is 15 to 50 words after the trigger word.
- Describe only what is visible. Do not guess a story, a name, or what happens next.

**Cover these points in this order, when they are visible:**

1. **Subject:** who or what is in the picture, and how many. Use generic words such as "a young man" or "an old woman".
2. **Appearance:** hair colour and length, clothing, notable items such as a sword or glasses.
3. **Action and expression:** pose, what the subject does, the facial expression.
4. **Framing:** close-up, half-body, full-body, wide shot, seen from above, seen from below, seen from behind.
5. **Setting:** place, objects in the background, indoors or outdoors.
6. **Light and time:** night, sunset, backlit, glowing light source, rain.

**Never write these (style words):**

illustration, drawing, digital art, 2D, anime, cartoon, comic, manga, cel shading, flat colors, lineart, line art, clean lines, screentone, inked, stylized, art style, in the style of, panel, by (artist name)

**Also never write:**

- Character names, series titles, studio or artist names. Describe the person instead.
- Quality words: masterpiece, best quality, high quality, detailed, 4k, beautiful, stunning.
- Things that are not in the picture ("no background", "without text").
- Opinions or feelings about the picture.

**Colours:** describe colours you see (black coat, blue glow). Colour of an object is content, not style.

**Empty or abstract panels:** describe them plainly, for example "a dark sky with a large glowing blue portal above a city skyline".

## 6. Examples

Good:

- `mystyle, a young man with short black hair and a long grey coat stands on a rooftop at night, seen from below. City lights glow behind him and wind moves his hair.`
- `mystyle, a close-up of a woman with long brown hair and green eyes looking to the side with a calm expression. A quiet library with tall shelves is behind her.`
- `mystyle, a wide shot of an empty train platform with cracked tiles and flickering lights. A stray dog sleeps on a bench at the far end.`

Bad, and why:

- `a man on a rooftop` (no trigger word, too short)
- `mystyle, comic style, clean lineart, a man on a rooftop` (style words)
- `mystyle, 1boy, black hair, coat, rooftop, night, from below` (tags, not sentences)
- `mystyle, <character name> stands on a rooftop` (character name)
- `mystyle, a stunning masterpiece of a hero about to win the final battle` (quality words and guessed story)

## 7. Final checks before you finish

Run these over the whole folder and fix every failure:

1. Every image outside `rejected/` has a `.txt` with the same base name.
2. No `.txt` exists without an image.
3. Every caption starts with `mystyle, ` and contains `mystyle` exactly once.
4. No caption contains a word from the two "never write" lists (check without case sensitivity).
5. No two captions are identical.
6. Every caption has 15 to 50 words after the trigger word and is a single line.
7. `labels_report.csv` has one row per original image.

Then write a short summary for the owner:

- number of images kept and rejected, with the count per reason code,
- variety of the kept set: how many close-ups, half-body, full-body, wide shots, and how many panels without people,
- a warning if fewer than 30 images were kept,
- a warning if one character appears in more than about a third of the kept images, because the LoRA then learns that character instead of the style.

## 8. If this is a character LoRA instead

Use the same procedure with two changes:

- Use `mystyle` as the name of the character: `mystyle, a young man standing on a rooftop at night ...`.
- Leave out the character's permanent features (face, hair, eye colour), so the trigger word absorbs them. Still describe clothing, pose, framing, setting and light, because those should stay changeable.

## 9. Prompt to give the agent

```
You are labeling an image dataset for LoRA training. Follow the attached guide
"Labeling guide for a style LoRA dataset" exactly.

Dataset folder: <PATH>

Go through the images one at a time in filename order. For each image: open it
and look at it, run the quality check, then either move it to rejected/ or write
its caption file. Every caption starts with "mystyle, " and describes only the
visible content in one or two plain sentences, with no style words, no names and
no quality words. Record every image in labels_report.csv.

When all images are done, run the final checks in section 7, fix any failures,
and give me the summary. Do not edit, crop or delete any image.
```
