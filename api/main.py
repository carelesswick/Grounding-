import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import GroundingRequest, GroundingResponse
from api.grounding_service import get_service, SCENE_MODEL_MAP

app = FastAPI(
    title="LocateAnything Grounding Service",
    description="FastAPI wrapper for fine-tuned LocateAnything-3B models",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "scenes": list(SCENE_MODEL_MAP.keys())}


@app.post("/api/v1/grounding", response_model=GroundingResponse)
def grounding(req: GroundingRequest):
    service = get_service()
    result = service.predict(req)
    return result


@app.post("/api/v1/grounding/upload", response_model=GroundingResponse)
async def grounding_upload(
    file: UploadFile = File(...),
    categories: Optional[str] = Form(None),
    phrase: Optional[str] = Form(None),
    scene: Optional[str] = Form(None),
    max_new_tokens: int = Form(512),
    temperature: float = Form(0.0),
):
    import base64
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
