"""Build a report and figures from measured runs; never supplies metric constants."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from ecgts.models import build
from ecgts.data import Windows
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = ['persistence', 'train_mean', 'cnn', 'patchtst']
output = Path('results/mitbih-seed42')
output.mkdir(parents=True, exist_ok=True)
shutil.copy('results/environment.txt', output/'environment.txt')
results = {}
for model in models:
    run = Path(f'runs/mitbih-{model}-seed42')
    metrics = json.loads((run/'metrics.json').read_text())
    if metrics['synthetic'] or metrics['args']['limit'] is not None:
        raise ValueError('This report requires real-data runs without --limit')
    results[model] = metrics
    shutil.copy(run/'metrics.json', output/f'{model}.json')
    log = Path(f'runs/mitbih-{model}-seed42.log')
    if log.exists(): shutil.copy(log, output/f'{model}.log')
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
# Explicitly selected hard example: highest-error window in the worst record.
torch.set_num_threads(2)
worst_record = max(results['patchtst']['test']['per_record'],key=lambda r: results['patchtst']['test']['per_record'][r]['mse'])
reference = results['patchtst']['args']
ds = Windows(reference['data'],'test',reference['context'],reference['horizon'],reference['stride'])
selected = [i for i,(record,_) in enumerate(ds.index) if ds.records[record] == worst_record]
net = build('patchtst',reference['context'],reference['horizon'])
net.load_state_dict(torch.load('runs/mitbih-patchtst-seed42/best.pt',map_location='cpu',weights_only=True))
net.eval()
worst = (-1.0,None)
with torch.inference_mode():
    for offset in range(0,len(selected),64):
        ids = selected[offset:offset+64]
        x = torch.from_numpy(np.stack([ds[i][0] for i in ids]))
        y = torch.from_numpy(np.stack([ds[i][1] for i in ids]))
        error = (net(x)-y).square().mean(dim=1)
        value,loc = error.max(dim=0)
        if value.item() > worst[0]: worst = (value.item(),ids[loc.item()])
index = worst[1]
x,y = ds[index]
start = ds.index[index][1]
fig,ax = plt.subplots(figsize=(9,4))
ax.plot(np.arange(len(x))/360,x,color='gray',label='context')
future_t = np.arange(len(x),len(x)+len(y))/360
ax.plot(future_t,y,color='black',linewidth=2,label='target')
with torch.inference_mode():
    for name in ('cnn','patchtst'):
        net = build(name,reference['context'],reference['horizon'])
        net.load_state_dict(torch.load(f'runs/mitbih-{name}-seed42/best.pt',map_location='cpu',weights_only=True))
        net.eval()
        prediction = net(torch.from_numpy(x[None,:])).numpy()[0]
        ax.plot(future_t,prediction,label=name)
ax.plot(future_t,np.full(len(y),x[-1]),linestyle=':',label='persistence')
ax.axvline(len(x)/360,color='gray',linestyle='--')
ax.set(xlabel='Seconds relative to window start',ylabel='Normalized MLII',
       title=f'Selected extreme: record {worst_record}, start {start}, PatchTST MSE={worst[0]:.3f}')
ax.legend(); fig.tight_layout(); fig.savefig(output/'selected-hard-window.png',dpi=160); plt.close(fig)
hard_example = dict(record=worst_record,start_sample=start,start_seconds=start/360,patchtst_mse=worst[0],
                    selection='Highest PatchTST window MSE within the highest PatchTST record-MSE test record; post-hoc, not typical.')
(output/'selected-hard-window.json').write_text(json.dumps(hard_example,indent=2))
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
    '![Test error](test-mse.png)','![Validation](validation.png)','![First window](first-window.png)','',f"极端案例选择：记录 {worst_record}，起点 sample={start}（记录内 {start/360:.3f} 秒），窗口 PatchTST-style MSE={worst[0]:.6f}。先选该模型记录均误差最高的测试记录，再选其中窗口误差最高者；非典型样本，未用于调参。",'','![Selected hard window](selected-hard-window.png)','',
    '## 逐记录结果与失败案例','',
    '以下按 PatchTST-style 测试 MSE 从大到小展示全部测试记录；属于事后误差分析，未据此重新训练。','',
    '| 测试记录 | Persistence MSE | Mean MSE | CNN MSE | PatchTST-style MSE |','|---|---:|---:|---:|---:|']
records=sorted(results['patchtst']['test']['per_record'],key=lambda k: -results['patchtst']['test']['per_record'][k]['mse'])
for rec in records:
    lines.append('| '+rec+' | '+' | '.join(f"{results[m]['test']['per_record'][rec]['mse']:.6f}" for m in models)+' |')
baseline_mse = results['persistence']['test']['mse']
cnn_mse = results['cnn']['test']['mse']
patch_mse = results['patchtst']['test']['mse']
better_records = sum(results['patchtst']['test']['per_record'][r]['mse'] < results['cnn']['test']['per_record'][r]['mse'] for r in records)
worse_records = [r for r in records if results['patchtst']['test']['per_record'][r]['mse'] >= results['cnn']['test']['per_record'][r]['mse']]
lines += ['', '## 本次观察','',
    f'- PatchTST-style 测试 MSE 比 persistence 低 {1-patch_mse/baseline_mse:.2%}，比 CNN 低 {1-patch_mse/cnn_mse:.2%}。',
    f'- PatchTST-style 在 {better_records}/{len(records)} 条测试记录上 MSE 低于 CNN；未优于 CNN 的记录为 {worse_records}。这不是多种子显著性检验。',
    f'- 按 PatchTST-style 记录均误差选择的困难记录为 {worst_record}。所示极端窗口中，两种训练模型均明显低估目标高幅波形；总体误差改善不代表峰值预测已解决。',
    '- CNN 与 PatchTST-style 的最佳验证 checkpoint 都是第 5 轮，尚不能判断模型已收敛。',
    '- 后续可以预先定义峰值区域评价、延长训练预算和多种子协议；本次没有根据测试结果更改模型或重新训练。']
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
