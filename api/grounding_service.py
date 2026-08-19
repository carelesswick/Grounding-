import base64
import io
import os
import re
import sys
from typing import List, Optional, Tuple

from PIL import Image

# Add official LocateAnything code path
CODE_PATH = "/data1/liyifan/BigModel/grounding/LocateAnything"
if CODE_PATH not in sys.path:
    sys.path.insert(0, CODE_PATH)

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

SCENE_MODEL_MAP = {
    "bridge": "/data1/liyifan/BigModel/work_dirs/full_lora/milestones/checkpoint-2000",
    "rail": "/data1/liyifan/BigModel/work_dirs/rail_lora_2000/checkpoint-2000",
    "tower": "/data1/liyifan/BigModel/work_dirs/tower_lora_2000/checkpoint-2000",
    "catenary": "/data1/liyifan/BigModel/work_dirs/catenary_lora_2000/checkpoint-2000",
    "noisebarrier": "/data1/liyifan/BigModel/work_dirs/noisebarrier_lora_2000/checkpoint-2000",
}

DEFAULT_SCENE = os.getenv("GROUNDING_DEFAULT_SCENE", "bridge")


def parse_boxes(text: str):
    out = []
    for m in BOX_RE.finditer(text):
        label = m.group(1)
        x1, y1, x2, y2 = map(int, m.groups()[1:])
        out.append((label, (x1, y1, x2, y2)))
    return out


class GroundingService:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._worker = None
        self._model_path = None

    def _load_model(self, model_path: str):
        from locateanything_worker import LocateAnythingWorker
        if self._worker is not None and self._model_path == model_path:
            return self._worker
        # release previous model to avoid OOM
        self._worker = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._worker = LocateAnythingWorker(model_path, device=self.device)
        self._model_path = model_path
        return self._worker

    def resolve_model_path(self, scene: Optional[str]) -> str:
        if scene and scene in SCENE_MODEL_MAP:
            return SCENE_MODEL_MAP[scene]
        if scene and scene not in SCENE_MODEL_MAP:
            raise ValueError(f"Unknown scene: {scene}. Available: {list(SCENE_MODEL_MAP.keys())}")
        return SCENE_MODEL_MAP.get(DEFAULT_SCENE)

    def load_image(self, request) -> Tuple[Image.Image, str]:
        if request.image_path:
            return Image.open(request.image_path).convert("RGB"), request.image_path
        if request.image:
            data = request.image
            if data.startswith("data:image"):
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)
            return Image.open(io.BytesIO(raw)).convert("RGB"), "base64"
        raise ValueError("Either image or image_path is required")

    def predict(self, request):
        model_path = self.resolve_model_path(request.scene)
        worker = self._load_model(model_path)
        img, source = self.load_image(request)
        w, h = img.size

        if request.categories:
            result = worker.detect(img, request.categories, max_new_tokens=request.max_new_tokens, temperature=request.temperature)
        elif request.phrase:
            result = worker.ground_multi(img, request.phrase, max_new_tokens=request.max_new_tokens, temperature=request.temperature)
        else:
            raise ValueError("Either categories or phrase is required")

        answer = result["answer"] if isinstance(result, dict) else str(result)
        boxes = []
        for label, norm_box in parse_boxes(answer):
            x1, y1, x2, y2 = norm_box
            px1, py1, px2, py2 = x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h
            boxes.append({
                "label": label,
                "bbox": [px1, py1, px2, py2],
                "normalized_bbox": [x1, y1, x2, y2],
                "confidence": 1.0,
            })
        return {
            "scene": request.scene,
            "model_path": model_path,
            "image_size": [w, h],
            "boxes": boxes,
            "raw_answer": answer,
        }


_service = None


def get_service():
    global _service
    if _service is None:
        device = os.getenv("GROUNDING_DEVICE", "cuda")
        _service = GroundingService(device=device)
    return _service
