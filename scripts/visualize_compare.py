#!/usr/bin/env python3
"""Create side-by-side comparison visualizations across checkpoints."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CHECKPOINTS = [
    ("600", "/data1/liyifan/BigModel/work_dirs/eval_full_images_merged_filtered_06.json"),
    ("1000", "/data1/liyifan/BigModel/work_dirs/eval_full_images_merged_filtered_1000.json"),
    ("1500", "/data1/liyifan/BigModel/work_dirs/eval_full_images_merged_filtered_1500.json"),
    ("2000", "/data1/liyifan/BigModel/work_dirs/eval_full_images_merged_filtered_2000.json"),
    ("2500", "/data1/liyifan/BigModel/work_dirs/eval_full_images_merged_filtered_2500.json"),
]

COLORS = {
    "600": "#FF0000",
    "1000": "#FF8C00",
    "1500": "#FFD700",
    "2000": "#0000FF",
    "2500": "#8A2BE2",
}
GT_COLOR = "#00CC00"


def load_data():
    data = {}
    for name, path in CHECKPOINTS:
        with open(path) as f:
            data[name] = json.load(f)
    return data


def draw_boxes(img, boxes, color, width=6):
    draw = ImageDraw.Draw(img)
    for label, bbox in boxes:
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1 + 4, max(0, y1 + 4)), label, fill=color)
    return img


def make_comparison(data, index, out_path, cell_size=(640, 480)):
    # Original image from first checkpoint (same for all)
    img_path = data["600"]["predictions"][index]["image"]
    orig = Image.open(img_path).convert("RGB")
    gts = data["600"]["gts"][index]["gts"]

    cells = []
    for name, _ in CHECKPOINTS:
        cell = orig.copy()
        # draw GT first (green)
        draw_boxes(cell, [(g["label"], tuple(g["bbox"])) for g in gts], GT_COLOR, width=6)
        preds = data[name]["predictions"][index]["predictions"]
        draw_boxes(cell, [(p["label"], tuple(p["bbox"])) for p in preds], COLORS[name], width=6)
        cell.thumbnail(cell_size, Image.LANCZOS)
        # add title bar
        title = Image.new("RGB", (cell.width, cell.height + 30), (255, 255, 255))
        title.paste(cell, (0, 30))
        d = ImageDraw.Draw(title)
        d.text((10, 5), f"checkpoint-{name}", fill="black")
        cells.append(title)

    # arrange horizontally
    total_w = sum(c.width for c in cells)
    total_h = max(c.height for c in cells)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    x = 0
    for c in cells:
        canvas.paste(c, (x, 0))
        x += c.width
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print("saved", out_path, canvas.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="1,5,6", help="comma separated eval json indices")
    ap.add_argument("--outdir", default="/data1/liyifan/BigModel/work_dirs/vis/compare")
    args = ap.parse_args()
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    data = load_data()
    for idx in indices:
        out = Path(args.outdir) / f"compare_idx{idx}.jpg"
        make_comparison(data, idx, str(out))


if __name__ == "__main__":
    main()
