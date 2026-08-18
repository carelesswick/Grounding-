#!/usr/bin/env python3
"""Evaluate LocateAnything on val_tiles.jsonl (tile-level metrics)."""
import argparse
import json
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
        if k.lower() in low:
            return k
    return label


def parse_boxes(text):
    """Return list of (label, (x1,y1,x2,y2)) in normalized 0..1000 coords."""
    out = []
    for m in BOX_RE.finditer(text):
        label = normalize_label(m.group(1))
        x1, y1, x2, y2 = map(int, m.groups()[1:])
        out.append((label, (x1, y1, x2, y2)))
    return out


def to_pixel(box, w, h):
    x1, y1, x2, y2 = box
    return (x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h)


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


def evaluate_one(worker, sample, iou_threshold=0.5):
    img_path = sample["image"]
    human = sample["conversations"][0]["value"]
    gpt = sample["conversations"][1]["value"]

    m = re.search(r"description:\s*(.*?)\.", human)
    if not m:
        return None
    cats = [c.strip() for c in m.group(1).split("</c>") if c.strip()]
    if not cats:
        return None

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    result = worker.detect(img, cats, max_new_tokens=512, temperature=0.0)
    answer = result["answer"] if isinstance(result, dict) else str(result)

    preds = parse_boxes(answer)
    gts = parse_boxes(gpt)

    # Convert to pixel
    preds = [(lab, to_pixel(box, w, h)) for lab, box in preds]
    gts = [(lab, to_pixel(box, w, h)) for lab, box in gts]

    # Greedy matching per class
    used_pred = set()
    matches = []
    for gi, (glab, gbox) in enumerate(gts):
        best_iou = 0.0
        best_pi = -1
        for pi, (plab, pbox) in enumerate(preds):
            if pi in used_pred or plab != glab:
                continue
            cur = iou(gbox, pbox)
            if cur > best_iou:
                best_iou = cur
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_threshold:
            used_pred.add(best_pi)
            matches.append((glab, best_iou))
        else:
            matches.append((glab, 0.0))

    tp = sum(1 for _, v in matches if v >= iou_threshold)
    fp = len(preds) - len(used_pred)
    fn = len(gts) - tp

    return {
        "image": img_path,
        "categories": cats,
        "gt_count": len(gts),
        "pred_count": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/liyifan/BigModel/work_dirs/full_lora/checkpoint-600")
    ap.add_argument("--jsonl", default="/data1/liyifan/BigModel/datasets/processed/bridge_dataset/val_tiles.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--output", default="/data1/liyifan/BigModel/work_dirs/eval_val_tiles.json")
    args = ap.parse_args()

    sys.path.insert(0, "/data1/liyifan/BigModel/grounding/LocateAnything")
    from locateanything_worker import LocateAnythingWorker

    samples = [json.loads(l) for l in open(args.jsonl)]
    if args.limit > 0:
        samples = samples[:args.limit]
    # collect GT class names for dynamic normalization
    for smp in samples:
        for m in BOX_RE.finditer(smp["conversations"][1]["value"]):
            lab = m.group(1).strip()
            if lab not in CLASS_NAMES:
                CLASS_NAMES.append(lab)

    worker = LocateAnythingWorker(args.model, device="cuda")
    print(f"evaluating {len(samples)} samples ...", flush=True)

    results = []
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "ious": []})
    for i, s in enumerate(samples):
        r = evaluate_one(worker, s, args.iou)
        if r is None:
            continue
        results.append(r)
        for glab, iou_v in r["matches"]:
            st = class_stats[glab]
            st["gt"] += 1
            st["ious"].append(iou_v)
            if iou_v >= args.iou:
                st["tp"] += 1
            else:
                st["fn"] += 1
        if i % 20 == 0:
            print(f"  {i}/{len(samples)} done", flush=True)

    summary = {"samples": len(results), "iou_threshold": args.iou}
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    summary.update({"total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
                    "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
