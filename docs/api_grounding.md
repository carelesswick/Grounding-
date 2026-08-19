# Grounding Tool API 接口文档

## 服务信息

- 服务名：LocateAnything Grounding Service
- 启动方式：
  ```bash
  cd /data1/liyifan/BigModel
  conda activate bigmodel
  uvicorn api.main:app --host 0.0.0.0 --port 8001
  ```

## 健康检查

```bash
curl http://127.0.0.1:8001/health
```

返回：

```json
{"status":"ok","scenes":["bridge","rail","tower","catenary","noisebarrier"]}
```

## 场景路由

| scene | 模型路径 |
|---|---|
| bridge | work_dirs/full_lora/milestones/checkpoint-2000 |
| rail | work_dirs/rail_lora_2000/checkpoint-2000 |
| tower | work_dirs/tower_lora_2000/checkpoint-2000 |
| catenary | work_dirs/catenary_lora_2000/checkpoint-2000 |
| noisebarrier | work_dirs/noisebarrier_lora_2000/checkpoint-2000 |

## 推理接口

### POST /api/v1/grounding

请求体：

```json
{
  "image_path": "/data1/liyifan/BigModel/datasets/processed/rail_dataset/tiles/val/xxx.jpg",
  "categories": ["fastener"],
  "scene": "rail"
}
```

或使用 Base64 图片：

```json
{
  "image": "data:image/jpeg;base64,...",
  "phrase": "bird nest on steel structure",
  "scene": "tower"
}
```

### 响应格式

```json
{
  "scene": "rail",
  "model_path": "/data1/liyifan/BigModel/work_dirs/rail_lora_2000/checkpoint-2000",
  "image_size": [800, 800],
  "boxes": [
    {
      "label": "fastener",
      "bbox": [132.0, 84.8, 235.2, 144.0],
      "normalized_bbox": [165, 106, 294, 180],
      "confidence": 1.0
    }
  ],
  "raw_answer": "<ref>fastener</ref><box>...</box>"
}
```

## 文件上传接口

### POST /api/v1/grounding/upload

multipart/form-data：

- `file`: 图片文件
- `categories`: 逗号分隔类别，可选
- `phrase`: 文本描述，可选
- `scene`: 场景名，可选
- `max_new_tokens`: 默认 512
- `temperature`: 默认 0.0

## 说明

- `bbox` 为像素坐标 `[x1,y1,x2,y2]`
- `normalized_bbox` 为 LocateAnything 原始 0~1000 坐标
- `confidence` 当前固定为 1.0，后续可接入置信度输出

## SAM 分割接口

### POST /api/v1/segment

请求体：

```json
{
  "image_path": "/path/to/image.jpg",
  "boxes": [[132.0, 84.8, 235.2, 144.0]]
}
```

响应：

```json
{
  "image_size": [800, 800],
  "masks": [
    {"bbox": [132.0,84.8,235.2,144.0], "score": 0.95, "mask_base64": "..."}
  ]
}
```

## 可视化接口

### POST /api/v1/visualize

请求体：

```json
{
  "image_path": "/path/to/image.jpg",
  "boxes": [[132.0,84.8,235.2,144.0]],
  "labels": ["fastener"]
}
```

响应：

```json
{
  "image_base64": "/9j/..."
}
```

返回带检测框（和可选 mask 叠加）的 JPEG Base64 图片。
