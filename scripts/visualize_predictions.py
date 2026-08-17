#!/usr/bin/env python3
"""Visualize LocateAnything predictions (single image or from eval JSON)."""
import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(text):
    out = []
    for m in BOX_RE.finditer(text):
        label = m.group(1)
        x1, y1, x2, y2 = map(int, m.groups()[1:])
        out.append((label, (x1, y1, x2, y2)))
    return out


def draw_boxes(img, boxes, color, width=4):
    draw = ImageDraw.Draw(img)
    for label, (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "from-json"], default="single")
    ap.add_argument("--model", default="/data1/liyifan/BigModel/work_dirs/full_lora/checkpoint-600")
    # single mode
    ap.add_argument("--image", default="")
    ap.add_argument("--phrase", default="")
    ap.add_argument("--categories", default="")
    # from-json mode
    ap.add_argument("--json", default="")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--output", default="/data1/liyifan/BigModel/work_dirs/vis/result.jpg")
    args = ap.parse_args()

    sys.path.insert(0, "/data1/liyifan/BigModel/grounding/LocateAnything")
    from locateanything_worker import LocateAnythingWorker

    worker = LocateAnythingWorker(args.model, device="cuda")

    if args.mode == "single":
        img = Image.open(args.image).convert("RGB")
        w, h = img.size
        if args.categories:
            cats = [c.strip() for c in args.categories.split(",") if c.strip()]
            result = worker.detect(img, cats, max_new_tokens=512, temperature=0.0)
        else:
            result = worker.ground_multi(img, args.phrase, max_new_tokens=512, temperature=0.0)
        answer = result["answer"] if isinstance(result, dict) else str(result)
        print("ANSWER:", answer)
        preds = [(lab, tuple(v / 1000 * (w if i % 2 == 0 else h) for i, v in enumerate(box)))
                 for lab, box in []]
        # simpler parse to pixel
        preds = []
        for lab, box in parse_boxes(answer):
            x1, y1, x2, y2 = box
            preds.append((lab, (x1/1000*w, y1/1000*h, x2/1000*w, y2/1000*h)))
        draw_boxes(img, preds, "red")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        img.save(args.output)
        print("saved", args.output)

    else:
        data = json.load(open(args.json))
        preds_data = data["predictions"][args.index]
        gts_data = data["gts"][args.index]
        img = Image.open(preds_data["image"]).convert("RGB")
        preds = [(p["label"], tuple(p["bbox"])) for p in preds_data["predictions"]]
        gts = [(g["label"], tuple(g["bbox"])) for g in gts_data["gts"]]
        draw_boxes(img, preds, "red")
        draw_boxes(img, gts, "green")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        img.save(args.output)
        print("saved", args.output)
        print("pred boxes", len(preds), "gt boxes", len(gts))


if __name__ == "__main__":
    main()
