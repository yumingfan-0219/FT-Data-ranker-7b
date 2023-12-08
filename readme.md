# FT-Data-Ranker: 7B Model Track - Champion Solution

## 概览

本项目是在 "FT-Data-Ranker：大语言模型微调数据竞赛 - 7B模型赛道" 中，队伍 CTYUN-AI 的冠军方案。本方案描述了我们如何配置环境、处理数据、训练模型，并在比赛中取得第1名的成绩。

### 快速链接

- 比赛官网：[FT-Data-Ranker Competition](https://tianchi.aliyun.com/competition/entrance/532158)
- 环境配置指南：[Official Environment Setup](https://tianchi.aliyun.com/competition/entrance/532158/customize405)

## 步骤概述

1. **数据处理**：使用 `data-juicer` 工具处理英文和中文数据集。
2. **第一阶段采样**：生成 `train_bloom-c4.json` 文件。
3. **构建摘要数据**：生成 `train_bloom-c4-temp.json` 和 `train_bloom-c4-all-summ.jsonl` 文件。
4. **第二阶段采样**：生成 `train_bloom-c4-qwen14b-1w-summ.json` 文件。
5. **模型训练**：基于 DeepSpeed 训练脚本进行模型训练。
6. **模型测试**：在 80G A100 显卡上测试模型并提交结果。

## 详细步骤

代码执行：
1.使用data-juicer工具处理数据
1.1按照竞赛官方指引配置环境，具体参见https://tianchi.aliyun.com/competition/entrance/532158/customize405
1.2
对于英文数据集，我们先后使用bloom-oscar.yaml和redpajama-c4-refine.yaml进行采样
```
cd FT-Data-Ranker-7b-code-model\code-data\competition_kit\data-juicer
python tools/process_data.py --config /data/bloom-oscar.yaml
python tools/process_data.py --config /data/redpajama-c4-refine.yaml
```
对于中文数据集，我们使用alpaca-cot-zh-refine.yaml进行采样
```
python tools/process_data.py /data/alpaca-cot-zh-refine.yaml
```
注意上述路径需要更改为正确的路径

2.第一阶段采样
```
cd FT-Data-Ranker-7b-code-model/code-data/competition_kit/lm-training
python get_train_dataset_7b_1stage.py
```
得到train_bloom-c4.json文件，其中中英文配比为0.65

3.构建摘要数据
```
cd FT-Data-Ranker-7b-code-model\code-data\competition_kit
python construct_data.py
```
得到train_bloom-c4-temp.json文件，构造格式为instruction、input、output
```
qwen-infer.py
```
得到train_bloom-c4-all-summ.jsonl文件

4.第二阶段采样
```
cd FT-Data-Ranker-7b-code-model/code-data/competition_kit/lm-training
python get_train_dataset_7b_1stage.py
```
此时的数据从三个文件中获得：【1.2】的英文数据结果，【1.2】的中文数据结果，【3】的摘要构造数据结果
最终结果为train_bloom-c4-qwen14b-1w-summ.json

5.训练
基于原提交指南的train_scripts/deepspeed_train_7b_lora.sh对train_bloom-c4-qwen14b-1w-summ.json进行训练，得到最终的模型FFT-Data-Ranker-7b-code-model/model/bloom-c4-qwen14b-1w-summ
6.测试
在80G的A100显卡上基于原提交指南中的competition_kit/lm-evaluation-harness/examples/challenge-7B-stage1.sh进行测试并提交结果
