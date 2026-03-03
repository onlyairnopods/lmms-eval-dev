# XChat 任务配置说明

## 概述

XChat 是一个多语言多模态基准测试，来源于论文 [https://arxiv.org/abs/2410.16153](https://arxiv.org/abs/2410.16153)，用于评估视觉语言模型在多种语言和任务上的表现。

## 数据集结构

### 支持的语言 (8种)
- English (英语)
- Chinese (中文)
- Hindi (印地语)
- Indonesian (印尼语)
- Japanese (日语)
- Kinyarwanda (卢旺达语)
- Korean (韩语)
- Spanish (西班牙语)

### 支持的任务类别 (10种)
1. `art_explanation` - 艺术作品解释
2. `bar_chart_interpretation` - 柱状图解读
3. `defeasible_reasoning` - 可撤销推理
4. `figurative_speech_explanation` - 修辞解释
5. `graph_interpretation` - 图表解读
6. `image_humor_understanding` - 图像幽默理解
7. `iq_test` - IQ测试
8. `ocr` - 光学字符识别
9. `science_figure_explanation` - 科学图表解释
10. `unusual_images` - 异常图像

## 文件结构

```
lmms_eval/tasks/xchat/
├── configs/                              # 80个任务配置文件
│   ├── xchat_English_art_explanation.yaml
│   └── ...
├── utils.py                              # 数据处理工具函数
├── judge_xchat.py                       # 批量评分脚本
├── xchat.yaml                           # 任务组配置（包含所有80个子任务）
├── xchat_English.yaml                    # 语言级配置（单语言单类别）
└── [语言目录]/
    ├── art_explanation/
    │   ├── data.json                    # 数据文件
    │   └── *.jpg                       # 图片文件
    └── ...
```

## 使用方法

### 1. 运行单个任务

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
  --tasks xchat_English_art_explanation \
  --batch_size 1 \
  --log_samples
```

### 2. 运行 xchat 组（所有 80 个任务）

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
  --tasks xchat \
  --batch_size 1 \
  --log_samples
```

### 3. 运行特定语言的所有任务

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
  --tasks xchat_English \
  --batch_size 1 \
  --log_samples
```

### 4. 测试时限制样本数

```bash
python -m lmms_eval \
  --model qwen3_vl \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct \
  --tasks xchat \
  --limit 2 \
  --batch_size 1
```

### 5. 批量评分（推理完成后）

推理完成后，使用 judge_xchat.py 进行批量评分：

```bash
# 设置 API Key
export OPENAI_API_KEY="your-openai-api-key"

# 运行评分
python -m lmms_eval.tasks.xchat.judge_xchat \
  -i eval_outputs/samples_xchat_English_art_explanation.jsonl \
  -o eval_outputs/scored_xchat_English_art_explanation.jsonl \
  --model gpt-4o \
  --concurrent 5
```

## 配置说明

### 配置文件类型

| 文件 | 说明 |
|------|------|
| `xchat.yaml` | 任务组配置，包含所有 80 个子任务 |
| `xchat_{语言}.yaml` | 语言级配置（如 xchat_English.yaml），只包含该语言的一个类别 |
| `configs/xchat_{语言}_{类别}.yaml` | 详细任务配置 |

### 运行逻辑

- **默认行为**：运行推理并保存结果，不会自动调用 judge
- **批量评分**：推理完成后，使用 `judge_xchat.py` 脚本手动进行批量评分

## LLM-as-Judge 评分

### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `JUDGE_MODEL` | 评分模型名称 | gpt-4o |
| `JUDGE_TEMPERATURE` | 评分温度 | 0.0 |
| `JUDGE_MAX_CONCURRENT` | 最大并发数 | 10 |

### 批量评分脚本使用

```bash
# 基本用法
python -m lmms_eval.tasks.xchat.judge_xchat \
  -i <输入文件> \
  -o <输出文件>

# 参数说明
# -i, --input:   输入文件 (模型输出 JSONL)
# -o, --output:  输出文件 (评分结果 JSONL)
# -m, --model:   评分模型 (默认: gpt-4o)
# -c, --concurrent: 并发数 (默认: 5)
# -n, --limit:   限制评分的样本数量
```

## 数据格式注意事项

### data.json 字段

每个数据条目包含以下字段：

| 字段 | 说明 |
|------|------|
| `capability` | 能力类型 (如 "vision") |
| `task` | 任务类别 |
| `instance_idx` | 实例索引 (用于匹配图片文件) |
| `system_prompt` | 系统提示 |
| `input` | 用户输入/问题 |
| `reference_answer` | 参考答案 |
| `score_rubric` | 评分标准 (1-5分) |
| `atomic_checklist` | 原子检查清单 |
| `background_knowledge` | 背景知识 (部分类别有) |
| `caption` | 图片描述 (部分类别有) |

### 图片文件命名

图片文件命名格式：`{instance_idx}.jpg` (如 `0.jpg`, `1.jpg`)

### 类别格式差异

不同类别的 `data.json` 字段略有不同：
- 有 `caption` 字段：defeasible_reasoning, figurative_speech_explanation, image_humor_understanding, ocr, unusual_images
- 无 `caption` 字段：art_explanation, bar_chart_interpretation, graph_interpretation, iq_test, science_figure_explanation

因此，每个类别需要单独的配置文件，不能合并加载。

## 常见问题

### Q: 为什么不能在一个任务中加载多个类别？
A: 不同类别的 JSON 数据字段不完全一致（如部分有 caption 字段），HuggingFace datasets 无法自动合并。

### Q: 如何为模型添加特定的 prompt 格式？
A: 可以在 `lmms_eval_specific_kwargs` 中添加模型特定的配置，参考其他任务的配置方式。

### Q: 如何查看模型的输出结果？
A: 使用 `--log_samples` 参数，结果会保存在 `eval_*_outputs/` 目录下的 JSONL 文件中。

### Q: 如何使用其他 Judge 模型？
A: 设置环境变量 `JUDGE_MODEL`，或使用脚本的 `--model` 参数。

### Q: 推理完成后如何进行评分？
A:
1. 运行推理时添加 `--log_samples` 参数保存结果
2. 使用 judge_xchat.py 脚本进行批量评分
