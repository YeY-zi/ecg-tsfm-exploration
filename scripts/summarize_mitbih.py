"""Build a report and figures from measured runs; never supplies metric constants."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['persistence', 'train_mean', 'cnn', 'patchtst']
output = Path('results/mitbih-seed42')
output.mkdir(parents=True, exist_ok=True)
results = {}
for model in models:
    run = Path(f'runs/mitbih-{model}-seed42')
    metrics = json.loads((run/'metrics.json').read_text())
    if metrics['synthetic'] or metrics['args']['limit'] is not None:
        raise ValueError('This report requires real-data runs without --limit')
    results[model] = metrics
    shutil.copy(run/'metrics.json', output/f'{model}.json')
    if (run/'train.log').exists(): shutil.copy(run/'train.log', output/f'{model}.log')
assert len({r['manifest_sha256'] for r in results.values()}) == 1
assert len({json.dumps(r['windows'],sort_keys=True) for r in results.values()}) == 1
fig,ax = plt.subplots(figsize=(7,4))
ax.bar(models, [results[m]['test']['mse'] for m in models])
ax.set(ylabel='Test MSE (normalized units)', title='MIT-BIH forecasting: single seed, 5-epoch budget')
fig.tight_layout(); fig.savefig(output/'test-mse.png',dpi=160); plt.close(fig)
fig,ax = plt.subplots(figsize=(7,4))
for m in ('cnn','patchtst'):
    history=results[m]['history']
    ax.plot([r['epoch'] for r in history],[r['mse'] for r in history],marker='o',label=m)
ax.legend(); ax.set(xlabel='Epoch',ylabel='Validation MSE',title='Checkpoint selection: record 214 only')
fig.tight_layout(); fig.savefig(output/'validation.png',dpi=160); plt.close(fig)
fig,ax = plt.subplots(figsize=(9,4))
for m in ('cnn','patchtst'):
    pred=np.load(f'runs/mitbih-{m}-seed42/predictions.npz')
    x,y,p=pred['context'][0],pred['target'][0],pred['prediction'][0]
    ax.plot(np.arange(len(x),len(x)+len(y))/360,p,label=m)
ax.plot(np.arange(len(x))/360,x,color='gray',label='context')
ax.plot(np.arange(len(x),len(x)+len(y))/360,y,color='black',label='target')
ax.axvline(len(x)/360,color='gray',linestyle='--'); ax.legend()
ax.set(xlabel='Seconds',ylabel='Normalized MLII',title='First test window: record 100, start sample 0 (not selected by error)')
fig.tight_layout(); fig.savefig(output/'first-window.png',dpi=160); plt.close(fig)
lines=['# MIT-BIH 首次真实数据实验','',
    '日期：2026-10-15。单次 seed=42 探索性实验，不是论文原始指标复现或基础模型评估。','',
    '## 固定实验协议','',
    '- 数据：MIT-BIH Arrhythmia Database 1.0.0；360 Hz，按名字选择 MLII。',
    '- 20 条训练记录、1 条验证记录（214）、22 条测试记录；排除与 202 同人的 201，另排除起搏记录。',
    '- 各记录完整读取后构窗；context=256（0.711 秒），horizon=64（0.178 秒），stride=512，无 --limit。',
    '- 步长意味着离散采样预测窗口，不是对连续波形全部采样点进行预测。目标区间互不重叠。',
    '- 仅训练集拟合全局均值/标准差；无滤波、无重采样、无注释标签。',
    '- CNN、PatchTST-style 均训练 5 轮；AdamW lr=0.001，weight_decay=0.0001，clip_norm=1，batch=64。',
    '- 用验证 MSE 选 checkpoint；完成训练后才评估测试集。未用测试集调参。',
    '- CPU，2 个 PyTorch 线程；完整软件版本见 environment.txt。',
    f"- 实际窗口数：{results['cnn']['windows']}。",'',
    '## 实测结果','',
    '| 模型 | 最佳轮次 | Val MSE | Test MSE | Test MAE | 记录宏平均 MSE | 秒数 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for m,r in results.items():
    lines.append(f"| {m} | {r['best_epoch'] or 'N/A'} | {r['validation']['mse']:.6f} | {r['test']['mse']:.6f} | {r['test']['mae']:.6f} | {r['test']['macro_record_mse']:.6f} | {r['elapsed_seconds']:.1f} |")
lines += ['', 'MSE/MAE 单位基于训练集标准差归一化；MSE 乘 std²、MAE 乘 std 可转换回原信号单位。训练均值基线始终输出归一化后的 0。','',
    '![Test error](test-mse.png)','![Validation](validation.png)','![First window](first-window.png)','',
    '## 逐记录结果与失败案例','',
    '以下按 PatchTST-style 测试 MSE 从大到小展示全部测试记录；属于事后误差分析，未据此重新训练。','',
    '| 测试记录 | Persistence MSE | Mean MSE | CNN MSE | PatchTST-style MSE |','|---|---:|---:|---:|---:|']
records=sorted(results['patchtst']['test']['per_record'],key=lambda k: -results['patchtst']['test']['per_record'][k]['mse'])
for rec in records:
    lines.append('| '+rec+' | '+' | '.join(f"{results[m]['test']['per_record'][rec]['mse']:.6f}" for m in models)+' |')
lines += ['', '## 解释范围','',
    '- 本次只有一个随机种子、一个窗口配置、5 轮训练；不能据此判断模型家族的普遍优劣。',
    '- 验证集仅一条记录，选模可能不稳定，未来应采用多受试者验证或交叉验证。',
    '- PatchTST 为本仓库单变量结构改编，无预训练、无官方完整配方；尚未评估 Chronos/TimesFM。',
    '- 任务是短窗口波形预测，不支持长程依赖、少样本、异常检测、诊断能力或临床效用结论。',
    '- 跨受试者测试仅说明此固定切分上的结果，不等于跨数据集泛化。',
    '- 峰值附近误差及原始记录质量需要进一步分析，不把高误差直接解释为心律失常。','',
    '## 复跑','', '```bash','python -m pip install -r requirements.txt','bash scripts/run_mitbih.sh',
    'python -m pip install matplotlib','python scripts/summarize_mitbih.py','```','',
    '原始数据与 checkpoint 不上传。每个模型 JSON 保留完整配置、训练历史、数据 manifest/校验摘要、代码版本与逐记录指标。',
    '初始代码与本次云端运行由 AI 助手执行；使用者应独立复跑、理解并分析结果后再作为自己的能力证明。']
(output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:30]))
