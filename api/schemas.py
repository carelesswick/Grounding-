from typing import List, Optional
from pydantic import BaseModel, Field


class GroundingRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 image (with or without data URL prefix)")
    image_path: Optional[str] = Field(None, description="Local image path on server")
    categories: Optional[List[str]] = Field(None, description="List of category names for detection")
    phrase: Optional[str] = Field(None, description="Single phrase for grounding")
    scene: Optional[str] = Field(None, description="Scene name for model routing: bridge/rail/tower/catenary/noisebarrier")
    max_new_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.0, ge=0.0, le=1.0)


class GroundingBox(BaseModel):
    label: str
    bbox: List[float] = Field(..., description="Pixel coordinates [x1, y1, x2, y2]")
    normalized_bbox: List[int] = Field(..., description="Normalized coordinates [0,1000]")
    confidence: float = 1.0


class GroundingResponse(BaseModel):
    scene: Optional[str]
    model_path: str
    image_size: List[int]
    boxes: List[GroundingBox]
    raw_answer: str


class SegmentationRequest(BaseModel):
    image: Optional[str] = None
    image_path: Optional[str] = None
    boxes: List[List[float]] = Field(..., description="List of pixel boxes [x1,y1,x2,y2]")


class SegmentationMask(BaseModel):
    bbox: List[float]
    score: float
    mask_base64: str


class SegmentationResponse(BaseModel):
    image_size: List[int]
    masks: List[SegmentationMask]


class VisualizationRequest(BaseModel):
    image: Optional[str] = None
    image_path: Optional[str] = None
    boxes: List[List[float]] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    masks_base64: List[str] = Field(default_factory=list)


class VisualizationResponse(BaseModel):
    image_base64: str
