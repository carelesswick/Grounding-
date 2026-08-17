"""LocateAnything-3B 推理小样例（在 3090 上自动使用 sdpa 回退）。"""
import argparse
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data1/liyifan/BigModel/models/LocateAnything-3B")
    p.add_argument("--image", default="/data1/liyifan/BigModel/datasets/raw/UAV_data/steel_bridge/006.jpg")
    p.add_argument("--phrase", default="rust")
    p.add_argument("--mode", default="multi", choices=["single", "multi", "detect"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    import sys
    sys.path.insert(0, "/data1/liyifan/BigModel/grounding/LocateAnything")
    from locateanything_worker import LocateAnythingWorker

    w = LocateAnythingWorker(args.model, device="cuda")
    img = Image.open(args.image).convert("RGB")
    if args.mode == "multi":
        r = w.ground_multi(img, args.phrase, max_new_tokens=args.max_new_tokens, temperature=0.0)
    elif args.mode == "single":
        r = w.ground_single(img, args.phrase, max_new_tokens=args.max_new_tokens, temperature=0.0)
    else:
        r = w.detect(img, [x.strip() for x in args.phrase.split(",")], max_new_tokens=args.max_new_tokens, temperature=0.0)
    print("IMAGE:", args.image)
    print("PHRASE:", args.phrase)
    print("ANSWER:", r["answer"].strip())
    print("BOXES:", LocateAnythingWorker.parse_boxes(r["answer"], img.width, img.height))

if __name__ == "__main__":
    main()
