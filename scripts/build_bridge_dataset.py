#!/usr/bin/env python3
"""Convert bridge_dataset (LabelMe JSON + YOLO) into LocateAnything JSONL.

Strategy:
- train/val split: use the 85 uploaded val images as val, remaining as train (Scheme A)
- whole-image samples: detection + per-class grounding
- small-object crop samples: for boxes whose model-input min side < threshold,
  save a crop around the box and write a grounding sample in the crop coordinate.
"""
import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from category_prompts import prompt_for

CLASS_NAMES = [
    "coating_rusting",
    "railing_rusting",
    "nut_rusting",
    "coating_peeling_off",
    "coating_dirty",
    "nut_missing",
    "nest",
]

PATCH_SIZE = 14
PAD_SIZE = 28          # merge_kernel_size[0] * patch_size
TOKEN_LIMIT = 25600


def clamp(v, lo=0, hi=1000):
    return int(round(max(lo, min(hi, v))))


def norm_x(x, width):
    return clamp(x / width * 1000)


def norm_y(y, height):
    return clamp(y / height * 1000)


def model_min_side(x1, y1, x2, y2, width, height):
    """Approx. box min side in model pixels after the official image processor."""
    npix = (width // PATCH_SIZE) * (height // PATCH_SIZE)
    scale = math.sqrt(TOKEN_LIMIT / npix) if npix > TOKEN_LIMIT else 1.0
    nw = int(width * scale)
    nh = int(height * scale)
    tw = math.ceil(nw / PAD_SIZE) * PAD_SIZE
    th = math.ceil(nh / PAD_SIZE) * PAD_SIZE
    fx = tw / width
    fy = th / height
    return min((x2 - x1) * fx, (y2 - y1) * fy)


def load_records(images_dir, jsons_dir):
    records = []
    for jf in sorted(jsons_dir.glob("*.json")):
        stem = jf.stem.lower()
        candidates = [c for c in images_dir.glob("*") if c.is_file() and c.stem.lower() == stem]
        if not candidates:
            print(f"[warn] no image for {jf.name}")
            continue
        img_path = sorted(candidates)[0]
        d = json.loads(jf.read_text())
        width = d.get("imageWidth", 4000)
        height = d.get("imageHeight", 3000)
        shapes = []
        for s in d.get("shapes", []):
            label = str(s.get("label", "")).strip()
            pts = s.get("points", [])
            if len(pts) < 2:
                continue
            x1, y1 = float(pts[0][0]), float(pts[0][1])
            x2, y2 = float(pts[1][0]), float(pts[1][1])
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            # clamp tiny numerical negatives
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(width, x2); y2 = min(height, y2)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            shapes.append({
                "label": label,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        records.append({
            "stem": stem,
            "image": img_path,
            "width": width,
            "height": height,
            "shapes": shapes,
        })
    return records


def box_tokens(label, x1, y1, x2, y2, width, height):
    return (f"<ref>{label}</ref>"
            f"<box><{norm_x(x1, width)}><{norm_y(y1, height)}>"
            f"<{norm_x(x2, width)}><{norm_y(y2, height)}></box>")


def make_detection_sample(rec):
    if not rec["shapes"]:
        return None
    labels = []
    seen = set()
    for s in rec["shapes"]:
        if s["label"] not in seen:
            labels.append(prompt_for(s["label"]))
            seen.add(s["label"])
    human = ("<image-1>\n"
             "Locate all the instances that matches the following description: "
             + "</c>".join(labels) + ".")
    gpt = "".join(box_tokens(s["label"], s["x1"], s["y1"], s["x2"], s["y2"],
                             rec["width"], rec["height"])
                  for s in rec["shapes"])
    return {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt", "value": gpt},
    ], "image": str(rec["image"])}


def make_grounding_sample(rec, label):
    boxes = [s for s in rec["shapes"] if s["label"] == label]
    if not boxes:
        return None
    human = (f"<image-1>\n"
             f"Locate all the instances that match the following description: {prompt_for(label)}.")
    gpt = "".join(box_tokens(s["label"], s["x1"], s["y1"], s["x2"], s["y2"],
                             rec["width"], rec["height"])
                  for s in boxes)
    return {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt", "value": gpt},
    ], "image": str(rec["image"])}


def make_crop_sample(rec, box, crop_path, margin=3.0, min_side=160):
    width, height = rec["width"], rec["height"]
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    side = max(bw, bh) * margin
    side = max(side, min_side)
    side = min(side, width, height)
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    left = min(max(0.0, cx - side / 2.0), width - side)
    top = min(max(0.0, cy - side / 2.0), height - side)
    left = int(round(left))
    top = int(round(top))
    right = left + int(round(side))
    bottom = top + int(round(side))
    img = Image.open(rec["image"]).convert("RGB")
    crop = img.crop((left, top, right, bottom))
    crop.save(crop_path, quality=95)

    nx1 = box["x1"] - left
    ny1 = box["y1"] - top
    nx2 = box["x2"] - left
    ny2 = box["y2"] - top
    cw = right - left
    ch = bottom - top
    human = (f"<image-1>\n"
             f'Locate all the instances that match the following description: {prompt_for(box["label"])}.')
    gpt = box_tokens(box["label"], nx1, ny1, nx2, ny2, cw, ch)
    return {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt", "value": gpt},
    ], "image": str(crop_path)}


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(samples)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data1/liyifan/BigModel/datasets/MyownData/bridge_dataset")
    ap.add_argument("--out", default="/data1/liyifan/BigModel/datasets/processed/bridge_dataset")
    ap.add_argument("--crop-threshold", type=float, default=28.0,
                    help="crop boxes whose model-input min side is below this (pixels)")
    ap.add_argument("--crop-margin", type=float, default=3.0)
    ap.add_argument("--min-crop-side", type=int, default=160)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    crops_dir = out / "crops"
    if out.exists():
        shutil.rmtree(out)
    crops_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(src / "images", src / "jsons")
    val_stems = {p.stem.lower() for p in (src / "val/images/val").glob("*")}
    train_records = [r for r in records if r["stem"] not in val_stems]
    val_records = [r for r in records if r["stem"] in val_stems]
    print(f"records={len(records)} train={len(train_records)} val={len(val_records)}")

    # Whole-image samples
    train_whole = []
    train_classes = Counter()
    for rec in train_records:
        det = make_detection_sample(rec)
        if det:
            # Use an absolute image path for simplicity. The recipe root will be "/".
            det["image"] = str(rec["image"])
            train_whole.append(det)
        for label in sorted({s["label"] for s in rec["shapes"]}):
            g = make_grounding_sample(rec, label)
            if g:
                g["image"] = str(rec["image"])
                train_whole.append(g)
                train_classes[label] += 1
    write_jsonl(out / "train_whole.jsonl", train_whole)

    # Crop samples for small boxes
    train_crops = []
    crop_classes = Counter()
    crop_count = 0
    for rec in train_records:
        for i, box in enumerate(rec["shapes"]):
            if model_min_side(box["x1"], box["y1"], box["x2"], box["y2"],
                              rec["width"], rec["height"]) >= args.crop_threshold:
                continue
            crop_name = f'{rec["stem"]}__{i}.jpg'
            crop_path = crops_dir / crop_name
            sample = make_crop_sample(rec, box, crop_path,
                                      margin=args.crop_margin,
                                      min_side=args.min_crop_side)
            if sample is None:
                continue
            sample["image"] = str(crop_path)
            train_crops.append(sample)
            crop_classes[box["label"]] += 1
            crop_count += 1
    write_jsonl(out / "train_crops.jsonl", train_crops)

    # Validation whole-image samples
    val_jsonl = []
    for rec in val_records:
        det = make_detection_sample(rec)
        if det:
            det["image"] = str(rec["image"])
            val_jsonl.append(det)
        # also per-class grounding for validation? keep detection only for simplicity
    write_jsonl(out / "val.jsonl", val_jsonl)

    # Recipe
    recipe = {
        "bridge_whole": {
            "annotation": str(out / "train_whole.jsonl"),
            "root": "/",
            "repeat_time": 1.0,
            "data_augment": True,
        },
        "bridge_crops": {
            "annotation": str(out / "train_crops.jsonl"),
            "root": "/",
            "repeat_time": 2.0,
            "data_augment": True,
        },
    }
    recipe_path = out / "bridge_recipe.json"
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {recipe_path}")

    print("train_whole class sample counts:", dict(sorted(train_classes.items())))
    print("train_crops class sample counts:", dict(sorted(crop_classes.items())))
    print("crop images:", crop_count)


if __name__ == "__main__":
    main()
