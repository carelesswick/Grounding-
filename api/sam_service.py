import base64
import io
import os

import numpy as np
import torch
from PIL import Image


class SamService:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import SamModel, SamProcessor
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def segment(self, image: Image.Image, boxes):
        self._load()
        inputs = self.processor(image, input_boxes=[boxes], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pred_masks = outputs.pred_masks
        iou_scores = outputs.iou_scores
        masks_tensor = pred_masks[0]
        scores = iou_scores[0]
        results = []
        for i, box in enumerate(boxes):
            best = int(scores[i].argmax().item())
            mask = masks_tensor[i, best].cpu().numpy()
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            results.append({
                "bbox": box,
                "score": float(scores[i, best].item()),
                "mask_base64": base64.b64encode(buf.getvalue()).decode(),
            })
        return results


_service = None


def get_sam_service():
    global _service
    if _service is None:
        device = os.getenv("SAM_DEVICE", "cuda")
        _service = SamService(device=device)
    return _service
