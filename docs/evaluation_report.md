# BigModel Grounding 微调评估报告

## 1. 概述

本项目使用课题组桥梁病害数据集，对 `nvidia/LocateAnything-3B` 进行 LoRA 微调，目标是让模型能够根据文本描述定位病害目标。

当前模型：`checkpoint-600`

## 2. 数据说明

原始数据：

- 图片分辨率：4000×3000
- 训练图：339 张
- 验证图：85 张
- 类别：7 类

类别：

```text
coating_rusting
railing_rusting
nut_rusting
coating_peeling_off
coating_dirty
nut_missing
nest
```

## 3. 数据处理策略

由于整图直接训练在单张 RTX 3090 上显存不足，采用：

- Tile 切图：800×800，重叠率 20%
- 小目标 Crop：对模型输入尺寸小于阈值的框额外裁剪
- 训练数据：
  - train_tiles.jsonl：1056 条
  - train_crops_filtered.jsonl：100 条，repeat_time=2.0

## 4. 训练配置

- 模型：LocateAnything-3B
- 微调方式：LoRA rank=16
- 冻结：LLM + Backbone
- 训练：MLP + LoRA
- GPU：单张 RTX 3090
- Steps：600
- train_loss：0.7056

## 5. 评估结果

### 5.1 Tile 级别评估

评估数据：val_tiles.jsonl（239 条）
IoU 阈值：0.5

| 指标 | 数值 |
|---|---:|
| Precision | 0.5366 |
| Recall | 0.5019 |
| F1 | 0.5187 |

各类别 Recall：

| 类别 | TP | FN | Recall |
|---|---:|---:|---:|
| coating_rusting | 68 | 86 | 0.442 |
| railing_rusting | 28 | 8 | 0.778 |
| nut_rusting | 26 | 13 | 0.667 |
| coating_peeling_off | 7 | 13 | 0.350 |
| nut_missing | 2 | 4 | 0.333 |
| coating_dirty | 1 | 5 | 0.167 |
| nest | 0 | 2 | 0.000 |

### 5.2 整图合并评估

使用 tile 合并 + NMS，在原始 val 图上评估。

不同 `max-tile-cover` 过滤阈值对比：

| 阈值 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 不过滤 | 0.1657 | 0.3819 | 0.2311 |
| 0.8 | 0.2137 | 0.3681 | 0.2704 |
| **0.6** | **0.2146** | **0.3681** | **0.2711** |
| 0.5 | 0.2085 | 0.3403 | 0.2586 |

结论：默认使用 `--max-tile-cover 0.6`。

## 6. 当前问题

- 整图 Precision 仍偏低，模型容易输出大范围粗框
- 少样本类别（coating_dirty、nut_missing、nest）效果弱
- railing_rusting 整图 Recall 很低
- 需要继续训练或调整数据/后处理

## 7. 下一步

- 继续训练更多步数
- 对少样本类别做数据增强/上采样
- 优化 tile 合并与 NMS
- 开发 Grounding Tool FastAPI 接口
