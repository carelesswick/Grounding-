from typing import List, Optional
from pydantic import BaseModel, Field


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
EOF
scp -o ConnectTimeout=15 /tmp/sam_service.py /tmp/api_schemas2.py server:/data1/liyifan/BigModel/api/
