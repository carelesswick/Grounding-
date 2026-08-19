#!/usr/bin/env python3
import json, sys, os
from pathlib import Path
from PIL import Image, ImageDraw

def draw_boxes(img, boxes, color, width=14):
    d=ImageDraw.Draw(img)
    for label, bbox in boxes:
        x1,y1,x2,y2=bbox
        x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
        d.rectangle([x1,y1,x2,y2], outline=color, width=width)
        d.text((x1+6, max(0,y1+6)), str(label), fill=color)

def main():
    jpath=sys.argv[1]; outdir=sys.argv[2]
    data=json.load(open(jpath)); os.makedirs(outdir, exist_ok=True)
    for i,(p,g) in enumerate(zip(data["predictions"], data["gts"])):
        img=Image.open(p["image"]).convert("RGB")
        scale=1.0
        if img.width>1600:
            scale=1600/img.width
            img=img.resize((1600,int(img.height*scale)), Image.LANCZOS)
        preds=[(x["label"], tuple(v*scale for v in x["bbox"])) for x in p["predictions"]]
        gts=[(x["label"], tuple(v*scale for v in x["bbox"])) for x in g["gts"]]
        draw_boxes(img,gts,"#00CC00")
        draw_boxes(img,preds,"#FF0000")
        out=os.path.join(outdir, f"img_{i:03d}.jpg")
        img.save(out, quality=92)
    print("done", len(data["predictions"]), "->", outdir)

if __name__=="__main__": main()
