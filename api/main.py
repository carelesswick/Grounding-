import base64
import io
import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

from api.schemas import (
    GroundingRequest,
    GroundingResponse,
    SegmentationRequest,
    SegmentationResponse,
    VisualizationRequest,
    VisualizationResponse,
)
from api.grounding_service import get_service, SCENE_MODEL_MAP
from api.sam_service import get_sam_service

app = FastAPI(
    title="LocateAnything Grounding & SAM Service",
    description="FastAPI wrapper for fine-tuned LocateAnything-3B and SAM segmentation",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_from_request(image: Optional[str], image_path: Optional[str]) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image:
        data = image
        if data.startswith("data:image"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("Either image or image_path is required")


@app.get("/health")
def health():
    return {"status": "ok", "scenes": list(SCENE_MODEL_MAP.keys())}


@app.post("/api/v1/grounding", response_model=GroundingResponse)
def grounding(req: GroundingRequest):
    service = get_service()
    return service.predict(req)


@app.post("/api/v1/grounding/upload", response_model=GroundingResponse)
async def grounding_upload(
    file: UploadFile = File(...),
    categories: Optional[str] = Form(None),
    phrase: Optional[str] = Form(None),
    scene: Optional[str] = Form(None),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.0),
):
    data = await file.read()
    image_b64 = base64.b64encode(data).decode()
    cats = [c.strip() for c in categories.split(",")] if categories else None
    req = GroundingRequest(
        image=image_b64,
        categories=cats,
        phrase=phrase,
        scene=scene,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    service = get_service()
    return service.predict(req)


@app.post("/api/v1/segment", response_model=SegmentationResponse)
def segment(req: SegmentationRequest):
    img = load_image_from_request(req.image, req.image_path)
    service = get_sam_service()
    masks = service.segment(img, req.boxes)
    return SegmentationResponse(image_size=[img.width, img.height], masks=masks)


@app.post("/api/v1/segment/upload", response_model=SegmentationResponse)
async def segment_upload(
    file: UploadFile = File(...),
    boxes: str = Form(...),
):
    data = await file.read()
    import json
    boxes_list = json.loads(boxes)
    image_b64 = base64.b64encode(data).decode()
    req = SegmentationRequest(image=image_b64, boxes=boxes_list)
    return segment(req)


@app.post("/api/v1/visualize", response_model=VisualizationResponse)
def visualize(req: VisualizationRequest):
    img = load_image_from_request(req.image, req.image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, box in enumerate(req.boxes):
        x1, y1, x2, y2 = box
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=8)
        label = req.labels[i] if i < len(req.labels) else f"box{i}"
        draw.text((x1 + 6, max(0, y1 + 6)), str(label), fill="red")
    # overlay masks if provided
    if req.masks_base64:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for mask_b64 in req.masks_base64:
            mask_data = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_data)).convert("L").resize(img.size)
            overlay_draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, 80))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return VisualizationResponse(image_base64=base64.b64encode(buf.getvalue()).decode())
