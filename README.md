# BigModel

课题组大模型项目工作目录（李以帆）。

## 当前环境

- 服务器：Hxps（Ubuntu 18.04, 4x RTX 3090 24GB）
- 项目路径：/data1/liyifan/BigModel
- Conda 环境：bigmodel（Python 3.10）
  - 激活：conda activate bigmodel
  - 环境路径：/home/liyf/anaconda3/envs/bigmodel
- CUDA Toolkit：/usr/local/cuda -> 12.1
- PyTorch：2.6.0+cu124（驱动 570 兼容）

## 官方模型与代码

- 模型：nvidia/LocateAnything-3B
  - 本地路径：models/LocateAnything-3B
  - 来源：HuggingFace 官方 repo（经 hf-mirror 下载）
  - SHA256：
    - model-00001-of-00002.safetensors = 923cfc10fed19808067da6df85a9a4220ddc1f9eb91ceee94c0fecd05d0f2d58
    - model-00002-of-00002.safetensors = 3459ba101f40594f3f62d3312014f1f8378b4ba3da3b1d562480045938fc7d47
- 代码：github.com/NVlabs/Eagle（官方最新，commit 783f656d127ee498137b5ff52603ce36c292d317）
  - 仓库目录：grounding/Eagle
  - Embodied 入口：grounding/LocateAnything -> grounding/Eagle/Embodied
  - 旧版 Tongmeng 目录代码备份：grounding/LocateAnything_tongmeng_backup（只作对比参考）

## 目录结构

- datasets/raw/UAV_data：无人机巡检图像
- datasets/raw/UAV_cot_train.json、UAV_cot_test.json：图像级思维链标注（仅作参考，Grounding 训练需另行构建检测框标注）
- datasets/annotations：检测框/文本 Grounding 标注
- grounding/Eagle：LocateAnything 官方仓库（sparse checkout Embodied）
- grounding/LocateAnything：官方 Embodied 目录入口
- models/LocateAnything-3B：官方模型权重
- api/：后续 Grounding / SAM / 可视化 Tool 接口
- tools/：后续 Tool 实现
- scripts/：训练、推理、数据脚本
- logs/：安装与训练日志

## 第一周任务

1. 病害检测框-文本推理数据集构建（适配 LocateAnything-3B）
2. Grounding 微调环境部署、小样例跑通
3. Grounding / SAM / 可视化 Tool 接口开发

## 关键命令

- 激活环境：conda activate bigmodel
- 检查环境：python scripts/check_env.py
- LocateAnything 推理示例：python scripts/run_locateanything.py
- 官方训练说明：grounding/LocateAnything/document/TRAINING.md
