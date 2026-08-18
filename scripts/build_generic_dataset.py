#!/usr/bin/env python3
"""Generic LabelMe JSON -> LocateAnything JSONL + crop samples."""
import argparse, json, math, shutil
from collections import Counter
from pathlib import Path
from PIL import Image

PATCH_SIZE=14; PAD_SIZE=28; TOKEN_LIMIT=25600

def model_min_side(x1,y1,x2,y2,w,h):
    npix=(w//PATCH_SIZE)*(h//PATCH_SIZE)
    scale=math.sqrt(TOKEN_LIMIT/npix) if npix>TOKEN_LIMIT else 1.0
    nw=int(w*scale); nh=int(h*scale)
    tw=math.ceil(nw/PAD_SIZE)*PAD_SIZE; th=math.ceil(nh/PAD_SIZE)*PAD_SIZE
    return min((x2-x1)*tw/w,(y2-y1)*th/h)

def norm(v,maxv): return int(round(max(0,min(maxv,v))/maxv*1000))
def box_tokens(label,x1,y1,x2,y2,w,h):
    return "<ref>"+label+"</ref><box><"+str(norm(x1,w))+"><"+str(norm(y1,h))+"><"+str(norm(x2,w))+"><"+str(norm(y2,h))+"></box>"

def load_records(images_dir, jsons_dir):
    recs=[]
    for jf in sorted(jsons_dir.glob("*.json")):
        stem=jf.stem.lower()
        cand=[c for c in images_dir.glob("*") if c.is_file() and c.stem.lower()==stem]
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

def make_det(rec):
    if not rec["shapes"]: return None
    labels=[]; seen=set()
    for s in rec["shapes"]:
        if s["label"] not in seen:
            labels.append(s["label"]); seen.add(s["label"])
    human="<image-1>\nLocate all the instances that matches the following description: "+"</c>".join(labels)+"."
    gpt="".join(box_tokens(s["label"],s["x1"],s["y1"],s["x2"],s["y2"],rec["width"],rec["height"]) for s in rec["shapes"])
    return {"conversations":[{"from":"human","value":human},{"from":"gpt","value":gpt}],"image":str(rec["image"])}

def make_ground(rec,label):
    boxes=[s for s in rec["shapes"] if s["label"]==label]
    if not boxes: return None
    human="<image-1>\nLocate all the instances that match the following description: "+label+"."
    gpt="".join(box_tokens(s["label"],s["x1"],s["y1"],s["x2"],s["y2"],rec["width"],rec["height"]) for s in boxes)
    return {"conversations":[{"from":"human","value":human},{"from":"gpt","value":gpt}],"image":str(rec["image"])}

def make_crop(rec,box,crop_path,margin=3.0,min_side=160):
    W,H=rec["width"],rec["height"]; bw=box["x2"]-box["x1"]; bh=box["y2"]-box["y1"]
    side=max(bw,bh)*margin; side=max(side,min_side); side=min(side,W,H)
    cx=(box["x1"]+box["x2"])/2; cy=(box["y1"]+box["y2"])/2
    left=int(round(min(max(0,cx-side/2),W-side))); top=int(round(min(max(0,cy-side/2),H-side)))
    right=left+int(round(side)); bottom=top+int(round(side))
    img=Image.open(rec["image"]).convert("RGB"); img.crop((left,top,right,bottom)).save(crop_path,quality=95)
    human="<image-1>\nLocate all the instances that match the following description: "+box["label"]+"."
    gpt=box_tokens(box["label"],box["x1"]-left,box["y1"]-top,box["x2"]-left,box["y2"]-top,right-left,bottom-top)
    return {"conversations":[{"from":"human","value":human},{"from":"gpt","value":gpt}],"image":str(crop_path)}

def write_jsonl(path,samples):
    with open(path,"w",encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s,ensure_ascii=False)+"\n")
    print(f"wrote {path} ({len(samples)})")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--crop-threshold",type=float,default=28.0)
    ap.add_argument("--crop-margin",type=float,default=3.0)
    ap.add_argument("--min-crop-side",type=int,default=160)
    args=ap.parse_args()
    src=Path(args.src); out=Path(args.out)
    if out.exists(): shutil.rmtree(out)
    (out/"crops").mkdir(parents=True,exist_ok=True)
    recs=load_records(src/"images",src/"jsons")
    val_stems={p.stem.lower() for p in (src/"val/images/val").glob("*") if p.is_file()}
    train=[r for r in recs if r["stem"] not in val_stems]; val=[r for r in recs if r["stem"] in val_stems]
    print(f"records={len(recs)} train={len(train)} val={len(val)}")
    whole=[]; crops=[]; crop_classes=Counter()
    for r in train:
        d=make_det(r)
        if d: whole.append(d)
        for lab in sorted({s["label"] for s in r["shapes"]}):
            g=make_ground(r,lab)
            if g: whole.append(g)
        for i,b in enumerate(r["shapes"]):
            if model_min_side(b["x1"],b["y1"],b["x2"],b["y2"],r["width"],r["height"]) < args.crop_threshold:
                cp=out/"crops"/f"{r['stem']}__{i}.jpg"
                s=make_crop(r,b,cp,args.crop_margin,args.min_crop_side)
                if s: crops.append(s); crop_classes[b["label"]]+=1
    write_jsonl(out/"train_whole.jsonl",whole); write_jsonl(out/"train_crops.jsonl",crops)
    val_jsonl=[]
    for r in val:
        d=make_det(r)
        if d: val_jsonl.append(d)
    write_jsonl(out/"val.jsonl",val_jsonl)
    filt=[s for s in crops if max(Image.open(s["image"]).size)<=1200]
    write_jsonl(out/"train_crops_filtered.jsonl",filt)
    recipe={"train_tiles":{"annotation":str(out/"train_tiles.jsonl"),"root":"/","repeat_time":1.0,"data_augment":False},
            "train_crops":{"annotation":str(out/"train_crops_filtered.jsonl"),"root":"/","repeat_time":2.0,"data_augment":False}}
    (out/"recipe.json").write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding="utf-8")
    print("crop_classes",dict(crop_classes))

if __name__=="__main__": main()
