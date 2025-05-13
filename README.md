# 📘 FLAN-Response-Evaluator

## 项目简介

本项目基于 **FLAN-T5-base 模型**，实现了一个阅读深度评估系统，用于评估学生阅读理解回答的质量。输入包括上下文（context）、问题（question）、答案（answer）、学生回答和文章摘要（abstract），模型生成对应的评价结果或重构输出。

项目支持完整的 **数据预处理 → 模型训练 → 推理部署** 流程，适用于阅读理解、教育技术、自动评分等任务场景。

------

## 🔧 项目结构

```plaintext
.gitignore               # Git 忽略文件配置
checkpoints/             # 模型检查点存储目录
data/                    # 数据目录
  narriative/            # 叙述型数据集
    test.json            # 测试集
    train.json           # 训练集
    valid.json           # 验证集
  squad/                 # SQuAD 数据集
    test.json            # 测试集
    train.json           # 训练集
    valid.json           # 验证集
  Vanilla/               # Vanilla 数据集
    test.json            # 测试集
    train.json           # 训练集
    valid.json           # 验证集
  数据说明.md            # 数据说明文档（中文）
  数据说明new.md         # 数据说明文档（更新版）
doc/                     # 文档目录（可放实验结果、说明）
model/                   # 模型相关代码
  flan_generator.py      # FLAN 模型封装与 LightningModule 实现
  __init__.py            # 模块初始化文件
preprocess/              # 数据预处理代码
  flan_dataset.py        # 数据集加载与预处理逻辑
  __init__.py            # 模块初始化文件
train_flan.py            # 模型训练主脚本
infer_flan.py            # 推理脚本
LICENSE                  # 许可证文件（如 MIT）
```

------

## 📦 环境依赖

建议使用 `conda` 或 `venv` 虚拟环境进行安装：

```bash
pip install -r requirements.txt
```

主要依赖如下：

- Python 3.8+
- `transformers`
- `datasets`
- `pytorch-lightning`
- `torch`
- `tqdm`
- `wandb`（可选，用于日志追踪）

------

## 🚀 使用说明

### 1️⃣ 数据预处理

确保你已将训练/验证/测试集放入 `data/` 目录对应子目录中。

### 2️⃣ 模型训练

```bash
python train_flan.py --data_dir data/Vanilla
```

### 3️⃣ 模型推理

```bash
python infer_flan.py --data_path data/Vanilla/test.json \
                     --model_path checkpoints/flan-vanilla/epoch=4-step=xxx.ckpt
```

------

## 📊 输出示例

```json
输入格式:
Q1: A1 [SEP] Q2: A2 [SEP] ... [SEP] 摘要内容 [SEP] 原文context

输出格式:
"fully/partially/black_response"
```

------

## 📝 License

本项目遵循 [MIT License](https://chatgpt.com/c/LICENSE)