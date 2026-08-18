#!/usr/bin/env python3
"""Run LocateAnything on all 800x800 tiles of val images, merge with NMS,
and evaluate whole-image detection vs original LabelMe JSON annotations."""
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from category_prompts import CATEGORY_PROMPTS

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
CLASS_NAMES = list(CATEGORY_PROMPTS.keys())


def normalize_label(label):
    label = label.strip()
    if label in CLASS_NAMES:
        return label
    low = label.lower()
    for k, v in CATEGORY_PROMPTS.items():
        if low == v.lower():
            return k
    for k in CLASS_NAMES:
        if k in low:
            return k
    return label


def parse_boxes(text):
    out = []
    for m in BOX_RE.finditer(text):
        label = normalize_label(m.group(1))
        x1, y1, x2, y2 = map(int, m.groups()[1:])
        out.append((label, (x1, y1, x2, y2)))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, nms_iou=0.5):
    """boxes: list of (label, (x1,y1,x2,y2)); class-specific NMS by area desc."""
    boxes = sorted(boxes, key=lambda b: (b[1][2]-b[1][0])*(b[1][3]-b[1][1]), reverse=True)
    kept = []
    for label, box in boxes:
        duplicate = False
        for k_label, k_box in kept:
            if k_label != label:
                continue
            if iou(box, k_box) >= nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append((label, box))
    return kept


def tile_grid(width, height, tile_size, overlap):
    stride = max(1, int(tile_size * (1 - overlap)))
    xs = list(range(0, width - tile_size + 1, stride))
    if xs[-1] + tile_size < width:
        xs.append(width - tile_size)
    ys = list(range(0, height - tile_size + 1, stride))
    if ys[-1] + tile_size < height:
        ys.append(height - tile_size)
    return xs, ys


def load_val_records(src):
    images_dir = Path(src) / "images"
    jsons_dir = Path(src) / "jsons"
    val_stems = {p.stem.lower() for p in (Path(src) / "val/images/val").glob("*")}
    records = []
    for jf in sorted(jsons_dir.glob("*.json")):
        stem = jf.stem.lower()
        if stem not in val_stems:
            continue
        img_candidates = [c for c in images_dir.glob("*") if c.is_file() and c.stem.lower() == stem]
        if not img_candidates:
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
            shapes.append({"label": label, "box": (x1, y1, x2, y2)})
        records.append({
            "stem": stem,
            "image": img_candidates[0],
            "width": width,
            "height": height,
            "shapes": shapes,
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/liyifan/BigModel/work_dirs/full_lora/checkpoint-600")
    ap.add_argument("--src", default="/data1/liyifan/BigModel/datasets/MyownData/bridge_dataset")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--tile-size", type=int, default=800)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--max-tile-cover", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="/data1/liyifan/BigModel/work_dirs/eval_full_images.json")
    args = ap.parse_args()

    sys.path.insert(0, "/data1/liyifan/BigModel/grounding/LocateAnything")
    from locateanything_worker import LocateAnythingWorker

    records = load_val_records(args.src)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"val images: {len(records)}", flush=True)

    worker = LocateAnythingWorker(args.model, device="cuda")

    if args.jsonl:
        import os
        samples = [json.loads(l) for l in open(args.jsonl)]
        # map stem -> record
        rec_by_stem = {r["stem"]: r for r in records}
        predictions = []
        all_gt = []
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0})
        # group samples by stem
        by_stem = defaultdict(list)
        for smp in samples:
            tile_path = smp["image"]
            fname = Path(tile_path).name
            # stem__yi_xi.jpg
            base = fname[:-4]
            # find last __ split
            if "__" not in base:
                print("skip bad tile filename", fname)
                continue
            stem, coord = base.split("__", 1)
            if "_" not in coord:
                print("skip bad tile filename", fname)
                continue
            yi_s, xi_s = coord.split("_", 1)
            by_stem[stem.lower()].append((smp, int(yi_s), int(xi_s)))

        for idx, (stem, tile_list) in enumerate(by_stem.items()):
            rec = rec_by_stem.get(stem)
            if rec is None:
                continue
            w, h = rec["width"], rec["height"]
            cats = sorted({s["label"] for s in rec["shapes"]})
            xs, ys = tile_grid(w, h, args.tile_size, args.overlap)
            pred_boxes = []
            for smp, yi, xi in tile_list:
                tile_img = Image.open(smp["image"]).convert("RGB")
                tw, th = tile_img.size
                x = xs[xi] if xi < len(xs) else 0
                y = ys[yi] if yi < len(ys) else 0
                result = worker.detect(tile_img, cats, max_new_tokens=512, temperature=0.0)
                answer = result["answer"] if isinstance(result, dict) else str(result)
                for label, norm_box in parse_boxes(answer):
                    bx1, by1, bx2, by2 = norm_box
                    px1 = bx1 / 1000 * tw + x
                    py1 = by1 / 1000 * th + y
                    px2 = bx2 / 1000 * tw + x
                    py2 = by2 / 1000 * th + y
                    area_ratio = (px2 - px1) * (py2 - py1) / (tw * th)
                    if area_ratio > args.max_tile_cover:
                        continue
                    pred_boxes.append((label, (px1, py1, px2, py2)))
            pred_boxes = nms(pred_boxes, args.nms_iou)
            gts = [(s["label"], s["box"]) for s in rec["shapes"]]
            used_pred = set()
            matches = []
            for glab, gbox in gts:
                best_iou = 0.0
                best_pi = -1
                for pi, (plab, pbox) in enumerate(pred_boxes):
                    if pi in used_pred or plab != glab:
                        continue
                    cur = iou(gbox, pbox)
                    if cur > best_iou:
                        best_iou = cur
                        best_pi = pi
                if best_pi >= 0 and best_iou >= args.match_iou:
                    used_pred.add(best_pi)
                    matches.append((glab, best_iou))
                else:
                    matches.append((glab, 0.0))
            for glab, _ in matches:
                class_stats[glab]["gt"] += 1
            for plab, _ in pred_boxes:
                class_stats[plab]["pred"] += 1
            for glab, iou_v in matches:
                if iou_v >= args.match_iou:
                    class_stats[glab]["tp"] += 1
                else:
                    class_stats[glab]["fn"] += 1
            for pi, (plab, pbox) in enumerate(pred_boxes):
                if pi not in used_pred:
                    class_stats[plab]["fp"] += 1
            predictions.append({"image": str(rec["image"]),
                                "predictions": [{"label": lab, "bbox": box} for lab, box in pred_boxes]})
            all_gt.append({"image": str(rec["image"]),
                           "gts": [{"label": lab, "bbox": box} for lab, box in gts]})
            if (idx + 1) % 10 == 0:
                print(f"  {idx+1}/{len(by_stem)} images done (jsonl mode)", flush=True)

        total_tp = sum(v["tp"] for v in class_stats.values())
        total_fp = sum(v["fp"] for v in class_stats.values())
        total_fn = sum(v["fn"] for v in class_stats.values())
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        summary = {"images": len(predictions), "mode": "jsonl_positive_tiles",
                   "match_iou": args.match_iou, "nms_iou": args.nms_iou,
                   "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
                   "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                   "per_class": {k: {kk: vv for kk, vv in v.items()} for k, v in class_stats.items()}}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "predictions": predictions, "gts": all_gt}, f, ensure_ascii=False, indent=2)
        print("saved", args.output)
        return

    predictions = []
    all_gt = []
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0})

    for idx, rec in enumerate(records):
        img = Image.open(rec["image"]).convert("RGB")
        w, h = rec["width"], rec["height"]
        cats = sorted({s["label"] for s in rec["shapes"]})
        xs, ys = tile_grid(w, h, args.tile_size, args.overlap)

        pred_boxes = []
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                tile = img.crop((x, y, x + args.tile_size, y + args.tile_size))
                tw, th = tile.size
                result = worker.detect(tile, cats, max_new_tokens=512, temperature=0.0)
                answer = result["answer"] if isinstance(result, dict) else str(result)
                for label, norm_box in parse_boxes(answer):
                    bx1, by1, bx2, by2 = norm_box
                    px1 = bx1 / 1000 * tw + x
                    py1 = by1 / 1000 * th + y
                    px2 = bx2 / 1000 * tw + x
                    py2 = by2 / 1000 * th + y
                    area_ratio = (px2 - px1) * (py2 - py1) / (tw * th)
                    if area_ratio > args.max_tile_cover:
                        continue
                    pred_boxes.append((label, (px1, py1, px2, py2)))

        pred_boxes = nms(pred_boxes, args.nms_iou)
        gts = [(s["label"], s["box"]) for s in rec["shapes"]]

        # matching
        used_pred = set()
        matches = []
        for glab, gbox in gts:
            best_iou = 0.0
            best_pi = -1
            for pi, (plab, pbox) in enumerate(pred_boxes):
                if pi in used_pred or plab != glab:
                    continue
                cur = iou(gbox, pbox)
                if cur > best_iou:
                    best_iou = cur
                    best_pi = pi
            if best_pi >= 0 and best_iou >= args.match_iou:
                used_pred.add(best_pi)
                matches.append((glab, best_iou))
            else:
                matches.append((glab, 0.0))

        for glab, _ in matches:
            class_stats[glab]["gt"] += 1
        for pi, (plab, pbox) in enumerate(pred_boxes):
            class_stats[plab]["pred"] += 1
        for glab, iou_v in matches:
            if iou_v >= args.match_iou:
                class_stats[glab]["tp"] += 1
            else:
                class_stats[glab]["fn"] += 1
        for pi, (plab, pbox) in enumerate(pred_boxes):
            if pi not in used_pred:
                class_stats[plab]["fp"] += 1

        predictions.append({
            "image": str(rec["image"]),
            "predictions": [{"label": lab, "bbox": box} for lab, box in pred_boxes],
        })
        all_gt.append({
            "image": str(rec["image"]),
            "gts": [{"label": lab, "bbox": box} for lab, box in gts],
        })

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(records)} images done", flush=True)

    # summary
    total_tp = sum(v["tp"] for v in class_stats.values())
    total_fp = sum(v["fp"] for v in class_stats.values())
    total_fn = sum(v["fn"] for v in class_stats.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary = {
        "images": len(records),
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_class": {k: {kk: vv for kk, vv in v.items()} for k, v in class_stats.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "predictions": predictions, "gts": all_gt}, f, ensure_ascii=False, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
