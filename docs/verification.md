# 验证记录

日期：2026-10-15。

## 实际完成

- 安装 CPU PyTorch 2.14.1、NumPy 2.5.4、WFDB 4.3.1；完整依赖在 results/environment.txt。
- 4 项 unittest 通过：记录隔离、训练统计量/窗口边界、模型梯度、总体/逐记录指标汇总。
- 合成数据单轮 PatchTST-style 流程检查通过，未将其作为真实 ECG 效果。
- 真实 MIT-BIH 1.0.0 全部 43 条协议内记录已下载、解码；对原始数字信号检查 WFDB checksum。
- 主站下载较慢，改用 PhysioNet 公开 S3 分发；对同一 101.dat 做逐字节比对，一致。原始文件保留 SHA-256，新增下载另检查完整性与 S3 单段 ETag。
- 正式运行 persistence、train_mean、CNN、PatchTST-style，无 --limit；CNN 与 PatchTST-style 均 5 轮，seed=42。
- 全部 27,918 个测试窗口评价完成。模型仅按验证记录 214 选 checkpoint，不以测试误差选轮次。
- 原始 metrics、console log、数据 manifest、验证曲线、第一测试窗口及显式选出的极端误差窗口已保存。

## 结果入口

[完整报告](../results/mitbih-seed42/REPORT.md)

训练：25,380 窗口；验证：1,269；测试：27,918。context=256、horizon=64、stride=512。

| 模型 | Test MSE | Test MAE |
|---|---:|---:|
| Persistence | 0.735008 | 0.422209 |
| Train mean | 0.807830 | 0.651700 |
| CNN | 0.391751 | 0.356620 |
| PatchTST-style | 0.351394 | 0.319932 |

## 尚未完成

- 多种子、置信区间、多个预测长度、官方模型配方和基础模型效果评估。
- 长程时序、心搏分类、异常检测、临床或跨数据集验证。
- LSTM、full-token Transformer 的真实数据训练。
- 本次没有配置或运行 GitHub Actions；测试在当前云端环境执行。

本次代码补充、运行与报告由 AI 助手执行。指标是真实程序输出，但不能据此声称使用者已独立完成所有设计与分析。
