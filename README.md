# ECG Time-Series Forecasting

AI 辅助开发的 ECG 波形预测学习项目：在固定的 MIT-BIH 切分上比较简单基线、CNN 与 PatchTST-style 模型。实验由云端 AI 助手执行；它是可复核的起点，使用者仍需独立理解和复跑。

## 首次真实 MIT-BIH 实验

完整报告、实际指标、逐记录误差、训练日志与预测图见：

**[results/mitbih-seed42/REPORT.md](results/mitbih-seed42/REPORT.md)**

固定协议：43 条记录，20/1/22 条训练/验证/测试；MLII，360 Hz；输入 256 点、预测 64 点、窗口步长 512；seed=42，CNN 与 PatchTST-style 各训练 5 轮。无 `--limit`；窗口在完整记录上离散采样，不是预测所有采样点。

实测结果（训练标准差归一化后的单位）：

| 模型 | Test MSE | Test MAE |
|---|---:|---:|
| Persistence | 0.735008 | 0.422209 |
| Train mean | 0.807830 | 0.651700 |
| CNN | 0.391751 | 0.356620 |
| PatchTST-style | 0.351394 | 0.319932 |

本次 PatchTST-style 的测试 MSE 比 CNN 低约 10.3%，但两者在极端高幅波形窗口都存在明显低估。详见报告的逐记录结果和极端案例。

这是短窗口预测的单种子探索性实验，不是 PatchTST 官方完整复现，不是基础模型预训练或临床诊断评估。Chronos/TimesFM/TimesNet 尚未得到本项目真实效果验证。

## 环境和运行

建议 Python 3.12。CPU 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m unittest discover -s tests -v
bash scripts/run_mitbih.sh
pip install matplotlib
python scripts/summarize_mitbih.py
```

Windows 可用 `.venv\Scripts\activate` 激活，再逐条执行下列命令：

```bash
python -m ecgts.run --prepare --data data/mitdb
python -m ecgts.run --data data/mitdb --model persistence --stride 512 --epochs 5 --seed 42 --out runs/mitbih-persistence-seed42
python -m ecgts.run --data data/mitdb --model train_mean --stride 512 --epochs 5 --seed 42 --out runs/mitbih-train_mean-seed42
python -m ecgts.run --data data/mitdb --model cnn --stride 512 --epochs 5 --seed 42 --out runs/mitbih-cnn-seed42
python -m ecgts.run --data data/mitdb --model patchtst --stride 512 --epochs 5 --seed 42 --out runs/mitbih-patchtst-seed42
```

首次下载使用 PhysioNet 公开 S3 分发（MIT-BIH 1.0.0），按单段 ETag/长度核验新增文件，并对解码信号检查 WFDB checksum。输出目录必须尚不存在；每次改参数请使用新目录。完整实验的软件版本记录在 `results/environment.txt`，其中 torch 为 CPU 构建；严格复跑可先按该记录安装对应版本。

轻量流程检查（不用于 ECG 性能结论）：

```bash
python -m ecgts.run --prepare --synthetic --data data/toy
python -m ecgts.run --data data/toy --model patchtst --epochs 1 --limit 16 --out runs/toy
```

## 数据协议

- 使用 MIT-BIH Arrhythmia Database 1.0.0；按导联名读取 MLII，兼容记录 114 的导联顺序。无滤波与重采样。
- 参考常见 DS1/DS2 记录划分，训练侧的 214 留作验证；排除与测试记录 202 同一受试者的 201；排除常见起搏记录 102、104、107、217。具体清单在 `ecgts/data.py`。
- 先按记录隔离 train/val/test，再在记录内窗口化；训练集全局均值/标准差用于所有 split。
- 这个修改后的波形预测协议与论文中的心搏分类协议不同，不能直接比较其数字。
- 验证集只有一条记录，选模稳定性有限；未来应采用多受试者验证或交叉验证。

## 模型与证据范围

| 模型 | 状态 |
|---|---|
| Persistence / train mean | 真实 MIT-BIH 评价 |
| 1D-CNN | 固定 5 轮真实训练与测试 |
| PatchTST-style | 单变量结构改编、从零训练；固定 5 轮真实训练与测试 |
| LSTM / full-token Transformer | 实现及梯度测试通过；本次未开展真实训练 |
| Chronos | 可选适配器；真实运行和效果待验证 |
| TimesNet / TimesFM | 计划，尚未实现 |

PatchTST-style 包含 patch embedding、位置编码、两层 Transformer、输入窗口归一化和 flatten head。未实现官方全部配置与训练配方，不能声称达到论文效果。模型间归一化及结构不同，因此本实验也不是单独隔离 patching 因素的消融。

默认训练为 AdamW，lr=1e-3、weight_decay=1e-4、clip_norm=1。仅验证 MSE 用于选择 checkpoint；测试集在选模后评价。MSE/MAE 按预测点加权，另外保存逐测试记录的误差及记录宏平均 MSE。

## 产物

每次运行在 `runs/` 保存：

- `metrics.json`：参数、版本、数据切分、预处理统计量、数据校验摘要、代码版本、训练/验证历史、最佳轮次、总体与逐记录测试指标。
- `best.pt`：训练模型的最佳验证 checkpoint。
- `predictions.npz`：第一个测试 batch 的 context、target、prediction。

`results/` 保存审查后的小型指标、日志与图片；原始数据、虚拟环境和 checkpoint 不上传 GitHub。`docs/verification.md` 记录验证范围。

## 下一步

多种子评估、多个预测长度、更稳健的验证划分、峰值附近误差分析，以及官方模型与预训练基础模型的接入，均应独立运行并记录后再做结论。

## 参考

- [MIT-BIH 数据与引用要求](https://physionet.org/content/mitdb/1.0.0/)
- [PatchTST 官方代码](https://github.com/yuqinie98/PatchTST)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [TimesNet / Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [TimesFM](https://github.com/google-research/timesfm)
