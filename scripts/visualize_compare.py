#!/usr/bin/env python3
"""Generate per-checkpoint comparison images: each image shows GT + one checkpoint."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

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
LINE_WIDTH = 14
MAX_WIDTH = 1600


def load_data():
    data = {}
    for name, path in CHECKPOINTS:
        with open(path) as f:
            data[name] = json.load(f)
    return data


def draw_boxes(img, boxes, color, width=LINE_WIDTH):
    draw = ImageDraw.Draw(img)
    for label, bbox in boxes:
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1 + 8, max(0, y1 + 8)), label, fill=color)
    return img


def make_single(data, name, index, out_path):
    orig = Image.open(data[name]["predictions"][index]["image"]).convert("RGB")
    ow, oh = orig.size
    scale = MAX_WIDTH / ow
    new_size = (MAX_WIDTH, int(oh * scale))
    img = orig.resize(new_size, Image.LANCZOS)

    gts = data[name]["gts"][index]["gts"]
    preds = data[name]["predictions"][index]["predictions"]

    # scale boxes
    def scale_boxes(boxes):
        out = []
        for item in boxes:
            label, bbox = item["label"], item["bbox"]
            x1, y1, x2, y2 = bbox
            out.append((label, (x1 * scale, y1 * scale, x2 * scale, y2 * scale)))
        return out

    draw_boxes(img, scale_boxes(gts), GT_COLOR)
    draw_boxes(img, scale_boxes(preds), COLORS[name])

    # title bar
    title_h = 40
    canvas = Image.new("RGB", (img.width, img.height + title_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), f"checkpoint-{name}  idx{index}  (green=GT, {COLORS[name]}=pred)", fill="black")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print("saved", out_path, canvas.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="1,5,6", help="comma separated eval json indices")
    ap.add_argument("--outdir", default="/data1/liyifan/BigModel/work_dirs/vis/compare_individual")
    args = ap.parse_args()
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    data = load_data()
    for idx in indices:
        for name, _ in CHECKPOINTS:
            out = Path(args.outdir) / f"checkpoint-{name}" / f"compare_idx{idx}.jpg"
            make_single(data, name, idx, str(out))


if __name__ == "__main__":
    main()
