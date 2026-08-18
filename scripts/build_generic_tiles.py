#!/usr/bin/env python3
"""Generic tile generation for LocateAnything."""
import argparse, json, shutil
from collections import Counter
from pathlib import Path
from PIL import Image

def norm(v,maxv): return int(round(max(0,min(maxv,v))/maxv*1000))
def box_tokens(label,x1,y1,x2,y2,w,h):
    return f"<ref>{label}</ref><box><{norm(x1,w)}><{norm(y1,h)}><{norm(x2,w)}><{norm(y2,h)}></box>"

def load_records(images_dir, jsons_dir):
    recs=[]
    for jf in sorted(jsons_dir.glob("*.json")):
        stem=jf.stem.lower(); cand=[c for c in images_dir.glob("*") if c.is_file() and c.stem.lower()==stem]
        if not cand: continue
        d=json.loads(jf.read_text()); W=d.get("imageWidth",4000); H=d.get("imageHeight",3000)
        shapes=[]
        for s in d.get("shapes",[]):
            lab=str(s.get("label","")).strip(); pts=s.get("points",[])
            if len(pts)<2: continue
            x1,y1=float(pts[0][0]),float(pts[0][1]); x2,y2=float(pts[1][0]),float(pts[1][1])
            x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
            x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
            if x2-x1<1 or y2-y1<1: continue
            shapes.append({"label":lab,"x1":x1,"y1":y1,"x2":x2,"y2":y2})
        recs.append({"stem":stem,"image":cand[0],"width":W,"height":H,"shapes":shapes})
    return recs

def generate(records,tiles_dir,tile_size,overlap,min_iou):
    tiles_dir.mkdir(parents=True,exist_ok=True); samples=[]; total=pos=0; assigned=Counter()
    stride=int(tile_size*(1-overlap))
    for rec in records:
        W,H=rec["width"],rec["height"]
        xs=list(range(0,W-tile_size+1,stride))
        if xs[-1]+tile_size<W: xs.append(W-tile_size)
        ys=list(range(0,H-tile_size+1,stride))
        if ys[-1]+tile_size<H: ys.append(H-tile_size)
        img=Image.open(rec["image"]).convert("RGB")
        for yi,y in enumerate(ys):
            for xi,x in enumerate(xs):
                total+=1; tw=min(tile_size,W-x); th=min(tile_size,H-y)
                hit=[]
                for b in rec["shapes"]:
                    ix1=max(b["x1"],x); iy1=max(b["y1"],y); ix2=min(b["x2"],x+tw); iy2=min(b["y2"],y+th)
                    if ix2<=ix1 or iy2<=iy1: continue
                    inter=(ix2-ix1)*(iy2-iy1); area=(b["x2"]-b["x1"])*(b["y2"]-b["y1"])
                    if area>0 and inter/area>=min_iou: hit.append(b)
                if not hit: continue
                tp=tiles_dir/f"{rec['stem']}__{yi}_{xi}.jpg"
                img.crop((x,y,x+tw,y+th)).save(tp,quality=95)
                labels=[]; seen=set()
                for b in hit:
                    if b["label"] not in seen: labels.append(b["label"]); seen.add(b["label"])
                human="<image-1>\nLocate all the instances that matches the following description: "+"</c>".join(labels)+"."
                gpt="".join(box_tokens(b["label"],max(0,b["x1"]-x),max(0,b["y1"]-y),min(tw,b["x2"]-x),min(th,b["y2"]-y),tw,th) for b in hit)
                samples.append({"conversations":[{"from":"human","value":human},{"from":"gpt","value":gpt}],"image":str(tp)})
                pos+=1
                for b in hit: assigned[b["label"]]+=1
    return samples,total,pos,assigned

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--tile-size",type=int,default=800); ap.add_argument("--overlap",type=float,default=0.2); ap.add_argument("--min-iou",type=float,default=0.3)
    args=ap.parse_args()
    src=Path(args.src); out=Path(args.out); tiles=out/"tiles"
    if tiles.exists(): shutil.rmtree(tiles)
    recs=load_records(src/"images",src/"jsons")
    val_stems={p.stem.lower() for p in (src/"val/images/val").glob("*") if p.is_file()}
    train=[r for r in recs if r["stem"] not in val_stems]; val=[r for r in recs if r["stem"] in val_stems]
    tr,total_t,pos_t,ass_t=generate(train,tiles/"train",args.tile_size,args.overlap,args.min_iou)
    va,total_v,pos_v,ass_v=generate(val,tiles/"val",args.tile_size,args.overlap,args.min_iou)
    def w(path,samples):
        with open(path,"w",encoding="utf-8") as f:
            for s in samples: f.write(json.dumps(s,ensure_ascii=False)+"\n")
        print(f"wrote {path} ({len(samples)})")
    w(out/"train_tiles.jsonl",tr); w(out/"val_tiles.jsonl",va)
    print("train tiles",total_t,"pos",pos_t,"assigned",dict(ass_t))
    print("val tiles",total_v,"pos",pos_v,"assigned",dict(ass_v))

if __name__=="__main__": main()
