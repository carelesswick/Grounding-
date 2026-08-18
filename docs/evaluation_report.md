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

---

## 8. 更新：1000 步训练后评估（checkpoint-1000）

### 8.1 Tile 级别对比

| 指标 | 600 步 | 1000 步 |
|---|---:|---:|
| Precision | 0.5366 | 0.5755 |
| Recall | 0.5019 | 0.5361 |
| F1 | 0.5187 | 0.5551 |

### 8.2 整图合并对比（max-tile-cover=0.6）

| 指标 | 600 步 | 1000 步 |
|---|---:|---:|
| Precision | 0.2146 | 0.2292 |
| Recall | 0.3681 | 0.4028 |
| F1 | 0.2711 | 0.2922 |

### 8.3 结论

- 从 600 步增加到 1000 步后，tile 级和整图级指标均提升
- 当前没有明显过拟合迹象
- 可以继续观察是否进一步训练到 1200～1500 步仍有收益

## 9. 更新：1500 步训练后评估（checkpoint-1500）

### 9.1 Tile 级别对比

| 指标 | 600 步 | 1000 步 | 1500 步 |
|---|---:|---:|---:|
| Precision | 0.5366 | 0.5755 | 0.6220 |
| Recall | 0.5019 | 0.5361 | 0.5817 |
| F1 | 0.5187 | 0.5551 | 0.6012 |

### 9.2 整图合并对比（max-tile-cover=0.6）

| 指标 | 600 步 | 1000 步 | 1500 步 |
|---|---:|---:|---:|
| Precision | 0.2146 | 0.2292 | 0.2446 |
| Recall | 0.3681 | 0.4028 | 0.4722 |
| F1 | 0.2711 | 0.2922 | 0.3223 |

### 9.3 结论

- 1500 步相比 1000 步仍在提升
- tile 级 F1 达到 0.6012
- 整图级 F1 达到 0.3223
- 目前仍未见明显过拟合，可以继续训练观察

## 10. 更新：2000 / 2500 步评估与收敛判断

### 10.1 Tile 级别

| 指标 | 1500 步 | 2000 步 | 2500 步 |
|---|---:|---:|---:|
| Precision | 0.6220 | 0.6667 | 0.6667 |
| Recall | 0.5817 | 0.6312 | 0.6236 |
| F1 | 0.6012 | 0.6484 | 0.6444 |

### 10.2 整图合并（max-tile-cover=0.6）

| 指标 | 1500 步 | 2000 步 | 2500 步 |
|---|---:|---:|---:|
| Precision | 0.2446 | 0.2482 | 0.2465 |
| Recall | 0.4722 | 0.4861 | 0.4861 |
| F1 | 0.3223 | 0.3286 | 0.3271 |

### 10.3 结论

- 2000 步达到当前最优：
  - tile 级 F1 = 0.6484
  - 整图级 F1 = 0.3286
- 2500 步相比 2000 步出现轻微下降或持平
- 判断已接近收敛，停止继续增加步数
- 推荐使用 checkpoint-2000 作为当前最终模型

## 11. 可视化对比

已生成不同 checkpoint 在同一验证图上的预测对比图。

图例：
- 绿色框：真实标注 GT
- 红色框：checkpoint-600
- 橙色框：checkpoint-1000
- 黄色框：checkpoint-1500
- 蓝色框：checkpoint-2000
- 紫色框：checkpoint-2500

图片路径（服务器本地）：

```text
/data1/liyifan/BigModel/work_dirs/vis/compare/compare_idx1.jpg
/data1/liyifan/BigModel/work_dirs/vis/compare/compare_idx5.jpg
/data1/liyifan/BigModel/work_dirs/vis/compare/compare_idx6.jpg
```

可通过 scp 下载到本地查看：

```bash
scp server:/data1/liyifan/BigModel/work_dirs/vis/compare/compare_idx1.jpg .
```

### 11.1 全部验证图对比

已生成全部 80 张验证图的对比图：

```text
/data1/liyifan/BigModel/work_dirs/vis/compare_all/
```

包含：
- `compare_idx0.jpg` ~ `compare_idx79.jpg`
- `index.html`：浏览器画廊，可直接打开查看全部图片

下载整个目录：

```bash
scp -r server:/data1/liyifan/BigModel/work_dirs/vis/compare_all .
```

或只下载 HTML 画廊后，把图片放在同目录用浏览器打开：

```bash
scp -r server:/data1/liyifan/BigModel/work_dirs/vis/compare_all ./compare_all
```

然后打开 `compare_all/index.html`。

### 11.2 单 checkpoint 单独对比图（推荐查看）

已生成每个 checkpoint 单独的对比图，每张图只包含：
- 绿色框：真实标注 GT
- 一种颜色框：对应 checkpoint 的预测框

目录：

```text
/data1/liyifan/BigModel/work_dirs/vis/compare_individual/
├── checkpoint-600/
├── checkpoint-1000/
├── checkpoint-1500/
├── checkpoint-2000/
├── checkpoint-2500/
└── index.html
```

每个 checkpoint 目录下有 80 张图：

```text
compare_idx0.jpg ~ compare_idx79.jpg
```

并配有 `index.html`，可用浏览器打开查看。

下载：

```bash
scp -r server:/data1/liyifan/BigModel/work_dirs/vis/compare_individual .
```

打开：

```text
compare_individual/index.html
```

## 12. 多词 Prompt 实验

### 12.1 实验内容

将困难类别 prompt 改为多词特征描述，例如：

- `nest` → `bird nest on steel structure`
- `nut_missing` → `missing nut on steel structure`
- `coating_dirty` → `dirty coating on bridge surface`

重新生成训练数据并从基座/中间模型继续训练。

### 12.2 结果

| 模型 | Tile F1 | 整图 F1 |
|---|---:|---:|
| 旧模型 2000 步（短标签） | 0.6484 | 0.3286 |
| 多词模型 2000 步 | 0.3633 | 0.1382 |

### 12.3 结论

- 多词 prompt 重新训练后指标明显下降
- 可能原因：
  - 训练流程/优化器重启导致未充分收敛
  - 多词 prompt 增加了输出不确定性，模型输出与短标签不匹配
  - 少样本类别本身仍难以通过 prompt 改变解决
- 结论：**当前继续使用旧模型 checkpoint-2000 作为最终模型**
- 后续可在推理阶段尝试多词 prompt，而不重新训练
