# ECG Time-Series Model Exploration

时间序列模型与 ECG 预测的独立研究工程。**这是实验起点，不是已完成的论文指标复现，也不是医疗诊断系统。**

## 当前范围

| 模型/任务 | 状态 |
|---|---|
| Persistence、1D-CNN、LSTM、Transformer | 统一预测 pipeline 已实现 |
| PatchTST | 从零训练的单变量结构改编，非官方完整复现 |
| Chronos | 可选零样本预测适配器，需下载预训练权重；未验证真实数据效果 |
| TimesNet、TimesFM | 后续计划，尚未实现 |
| 心律失常分类、少样本、异常检测 | 后续研究任务，尚无结果 |

## 1. 环境

建议 Python 3.11，CPU 可完成小规模验证，完整训练建议 GPU。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GPU 用户按 PyTorch 官网安装匹配 CUDA 的 torch，再安装 requirements。

## 2. 无需下载数据，先跑通

```bash
python -m ecgts.run --prepare --synthetic --data data/toy
python -m ecgts.run --data data/toy --model patchtst --epochs 2 --limit 64 --out runs/toy-patchtst
```

这里使用正弦波加噪声，**只能验证工程流程，不能用作 ECG 实验结论**。
输出目录必须尚不存在，避免覆盖实验。

## 3. 下载 MIT-BIH，做正式实验

```bash
python -m ecgts.run --prepare --data data/mitdb
for model in persistence cnn lstm transformer patchtst; do
  python -m ecgts.run --data data/mitdb --model "$model" --epochs 10 --out "runs/mitdb-${model}-42"
done
```

首次需要网络下载 PhysioNet WFDB 数据。读取 360 Hz 的 MLII 导联，按导联名选取，兼容 114 的导联顺序。仅缓存波形，本任务不使用节拍标签。默认输入 256 点（约 0.71 秒），预测 64 点（约 0.18 秒），步长 128。

训练/验证/测试按记录划分，具体记录见 `ecgts/data.py`。基于常见 DS1/DS2 协议改动：DS1 中 214 单独验证，排除与测试记录 202 同一受试者的 201；保留固定测试集。排除常见起搏记录 102、104、107、217。所有窗口均在单个记录内生成，均值/标准差只从训练集拟合。该协议不同于原始心搏分类协议，不应直接比较其论文数字。

`--limit 512` 可用于调试，会在每个 split 的窗口索引中均匀采样；正式实验不要添加。`--device cuda` 用于 GPU。基于验证 MSE 选最佳 checkpoint，最终只评估测试集；MSE/MAE 以训练集标准差归一化后的单位汇总，按预测点加权。患者之间的统计差异仍需后续按患者分析。

建议种子 42/43/44 各跑一次，汇报均值和标准差；当前没有自动搜索超参数，固定训练轮数并不代表各模型已充分调优。完整 token Transformer 的计算开销明显高于 PatchTST。

## 4. 可选 Chronos

```bash
pip install chronos-forecasting
python -m ecgts.run --data data/mitdb --model chronos --limit 128 --batch-size 8 --out runs/chronos-tiny
```

使用 `amazon/chronos-t5-tiny` 的原版 ChronosPipeline，20 条预测样本取中位数。此适配器尚需在目标机器验证依赖兼容性、权重下载和结果。零样本模型不训练，不能把它称作微调实验。ECG 采样频率、训练语料与自然时间序列存在差异，零样本效果需实测。

## 5. 结果和实验记录

每次输出 `metrics.json`（参数、版本、切分、训练统计量、验证曲线、测试指标）、`best.pt`（训练模型）、`predictions.npz`（首个测试 batch 的预测与真值）。数据和权重默认不入 Git。`results/` 仅保存人工审查过的小型指标文件。实际验证记录见 [docs/verification.md](docs/verification.md)。

检查失败案例：按患者拆解误差；比较 persistence；观察 QRS 峰附近误差；研究输入长度 256/512/1024、预测长度 64/128/256。长程依赖尚未通过默认短窗口验证。不能仅凭低预测误差推断诊断能力。

## 6. 上传 GitHub

先在 GitHub 建立空仓库 `ecg-tsfm-exploration`，然后在本项目目录运行：

```bash
git init
git add .
git commit -m "Add reproducible ECG forecasting research pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ecg-tsfm-exploration.git
git push -u origin main
```

不要上传 `data/`、`.venv/`、令牌或模型权重。研究描述应只写实际完成并留下实验记录的内容。

## 参考与后续

- MIT-BIH: https://physionet.org/content/mitdb/1.0.0/ （请遵守原始数据使用及引用要求）
- PatchTST: https://github.com/yuqinie98/PatchTST
- TimesNet: https://github.com/thuml/Time-Series-Library
- Chronos: https://github.com/amazon-science/chronos-forecasting
- TimesFM: https://github.com/google-research/timesfm

后续按官方实现与固定 commit 接入 TimesNet、TimesFM，再扩展患者级少样本训练、异常检测与带注释的心搏分类。需要重新定义任务、标签及评估协议，不能直接沿用预测分数。
