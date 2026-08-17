#!/usr/bin/env python3
"""Build tile + crop hybrid training data for LocateAnything.

Tile strategy:
- tile_size x tile_size sliding window with overlap
- only tiles containing at least one box (IoU >= min_visible_iou) are emitted
- boxes are clipped to tile and normalized to 0..1000
- crop samples from build_bridge_dataset.py are kept separately
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


def clamp(v, lo=0, hi=1000):
    return int(round(max(lo, min(hi, v))))


def norm_x(x, w):
    return clamp(x / w * 1000)


def norm_y(y, h):
    return clamp(y / h * 1000)


def load_records(images_dir, jsons_dir):
    records = []
    for jf in sorted(jsons_dir.glob("*.json")):
        stem = jf.stem.lower()
        candidates = [c for c in images_dir.glob("*") if c.is_file() and c.stem.lower() == stem]
        if not candidates:
            print(f"[warn] no image for {jf.name}")
            continue
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
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(width, x2); y2 = min(height, y2)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            shapes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        records.append({
            "stem": stem,
            "image": candidates[0],
            "width": width,
            "height": height,
            "shapes": shapes,
        })
    return records


def box_tokens(label, x1, y1, x2, y2, w, h):
    return (f"<ref>{label}</ref>"
            f"<box><{norm_x(x1, w)}><{norm_y(y1, h)}>"
            f"<{norm_x(x2, w)}><{norm_y(y2, h)}></box>")


def make_tile_sample(rec, tile_path, tile, boxes):
    x, y, tw, th = tile
    labels = []
    seen = set()
    for b in boxes:
        if b["label"] not in seen:
            labels.append(b["label"])
            seen.add(b["label"])
    human = ("<image-1>\n"
             "Locate all the instances that matches the following description: "
             + "</c>".join(labels) + ".")
    gpt = ""
    for b in boxes:
        nx1 = max(0.0, b["x1"] - x)
        ny1 = max(0.0, b["y1"] - y)
        nx2 = min(tw, b["x2"] - x)
        ny2 = min(th, b["y2"] - y)
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        gpt += box_tokens(b["label"], nx1, ny1, nx2, ny2, tw, th)
    if not gpt:
        return None
    return {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt", "value": gpt},
    ], "image": str(tile_path)}


def generate_tiles(records, tiles_dir, tile_size, overlap_ratio, min_visible_iou):
    tiles_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    tile_count = 0
    positive_count = 0
    assigned = Counter()
    stride_w = max(1, int(tile_size * (1 - overlap_ratio)))
    stride_h = max(1, int(tile_size * (1 - overlap_ratio)))
    for rec in records:
        W, H = rec["width"], rec["height"]
        xs = list(range(0, W - tile_size + 1, stride_w))
        if xs[-1] + tile_size < W:
            xs.append(W - tile_size)
        ys = list(range(0, H - tile_size + 1, stride_h))
        if ys[-1] + tile_size < H:
            ys.append(H - tile_size)
        img = Image.open(rec["image"]).convert("RGB")
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                tile_count += 1
                tw = min(tile_size, W - x)
                th = min(tile_size, H - y)
                hit = []
                for b in rec["shapes"]:
                    ix1 = max(b["x1"], x)
                    iy1 = max(b["y1"], y)
                    ix2 = min(b["x2"], x + tw)
                    iy2 = min(b["y2"], y + th)
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                    if area > 0 and inter / area >= min_visible_iou:
                        hit.append(b)
                if not hit:
                    continue
                tile_path = tiles_dir / f'{rec["stem"]}__{yi}_{xi}.jpg'
                crop = img.crop((x, y, x + tw, y + th))
                crop.save(tile_path, quality=95)
                sample = make_tile_sample(rec, tile_path, (x, y, tw, th), hit)
                if sample is None:
                    continue
                samples.append(sample)
                positive_count += 1
                for b in hit:
                    assigned[b["label"]] += 1
    return samples, tile_count, positive_count, assigned


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(samples)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data1/liyifan/BigModel/datasets/MyownData/bridge_dataset")
    ap.add_argument("--out", default="/data1/liyifan/BigModel/datasets/processed/bridge_dataset")
    ap.add_argument("--tile-size", type=int, default=1200)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-visible-iou", type=float, default=0.3)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    tiles_dir = out / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(src / "images", src / "jsons")
    val_stems = {p.stem.lower() for p in (src / "val/images/val").glob("*")}
    train_records = [r for r in records if r["stem"] not in val_stems]
    val_records = [r for r in records if r["stem"] in val_stems]
    print(f"records={len(records)} train={len(train_records)} val={len(val_records)}")

    train_samples, train_total, train_pos, train_assigned = generate_tiles(
        train_records, tiles_dir / "train", args.tile_size, args.overlap, args.min_visible_iou)
    write_jsonl(out / "train_tiles.jsonl", train_samples)

    val_samples, val_total, val_pos, val_assigned = generate_tiles(
        val_records, tiles_dir / "val", args.tile_size, args.overlap, args.min_visible_iou)
    write_jsonl(out / "val_tiles.jsonl", val_samples)

    print("train tiles total", train_total, "positive", train_pos, "assigned", dict(train_assigned))
    print("val tiles total", val_total, "positive", val_pos, "assigned", dict(val_assigned))


if __name__ == "__main__":
    main()
