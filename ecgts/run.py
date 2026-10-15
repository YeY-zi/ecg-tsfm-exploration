import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Windows, prepare
from .models import build


def evaluate(model, loader, device, by_record=False):
    if by_record and not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError('Record evaluation requires sequential sampling')
    model.eval()
    totals = np.zeros(3, dtype='float64')
    groups = {}
    offset = 0
    with torch.inference_mode():
        for x,y in loader:
            error = model(x.to(device)).cpu()-y
            if not torch.isfinite(error).all(): raise ValueError('Non-finite predictions')
            sq = error.square().sum(dim=1).numpy()
            ab = error.abs().sum(dim=1).numpy()
            totals += [sq.sum(), ab.sum(), y.numel()]
            if by_record:
                ds = loader.dataset
                for i in range(len(y)):
                    rec_idx, _ = ds.index[offset+i]
                    record = ds.records[rec_idx]
                    group = groups.setdefault(record,np.zeros(3,dtype='float64'))
                    group += [sq[i], ab[i], y.shape[1]]
            offset += len(y)
    def score(v): return dict(mse=float(v[0]/v[2]),mae=float(v[1]/v[2]))
    result = score(totals)
    if by_record:
        result['per_record'] = {k:dict(**score(v),windows=int(v[2]/loader.dataset.horizon)) for k,v in groups.items()}
        result['macro_record_mse'] = float(np.mean([v['mse'] for v in result['per_record'].values()]))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--data',default='data/mitdb')
    p.add_argument('--model',choices=['cnn','lstm','transformer','patchtst','persistence','train_mean','chronos'],default='patchtst')
    p.add_argument('--context',type=int,default=256)
    p.add_argument('--horizon',type=int,default=64)
    p.add_argument('--stride',type=int,default=128)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='runs/experiment')
    args = p.parse_args()
    if args.prepare:
        prepare(args.data,args.synthetic)
        print(f'Prepared {args.data}',flush=True)
        return
    if min(args.epochs,args.batch_size,args.threads) < 1: p.error('epochs, batch-size, threads must be positive')
    started = time.perf_counter()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest_path = Path(args.data)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    datasets = {s:Windows(args.data,s,args.context,args.horizon,args.stride,args.limit) for s in ['train','val','test']}
    loaders = {s:DataLoader(d,batch_size=args.batch_size,shuffle=(s=='train')) for s,d in datasets.items()}
    print('Windows:',{s:len(d) for s,d in datasets.items()},flush=True)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    history = []
    best_epoch = None
    if args.model in ('persistence','train_mean'):
        class Baseline(torch.nn.Module):
            def forward(self,x):
                return (x[:,-1:] if args.model=='persistence' else torch.zeros_like(x[:,-1:])).expand(-1,args.horizon)
        model = Baseline()
    elif args.model == 'chronos':
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained('amazon/chronos-t5-tiny',device_map=args.device,torch_dtype=torch.float32)
        class Chronos(torch.nn.Module):
            def forward(self,x):
                return pipeline.predict(x.cpu(),args.horizon,num_samples=20).median(dim=1).values.to(x.device)
        model = Chronos()
    else:
        model = build(args.model,args.context,args.horizon).to(args.device)
        optim = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best = float('inf')
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            model.train()
            train_sse = train_count = 0
            for x,y in loaders['train']:
                optim.zero_grad()
                error = model(x.to(args.device))-y.to(args.device)
                loss = error.square().mean()
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optim.step()
                train_sse += error.detach().square().sum().item()
                train_count += y.numel()
            val = evaluate(model,loaders['val'],args.device)
            history.append(dict(epoch=epoch+1,train_mse=train_sse/train_count,**val,seconds=time.perf_counter()-epoch_start))
            print(history[-1],flush=True)
            if val['mse'] < best:
                best = val['mse']
                best_epoch = epoch+1
                torch.save(model.state_dict(),out/'best.pt')
        model.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True))
    val = evaluate(model,loaders['val'],args.device)
    metrics = evaluate(model,loaders['test'],args.device,by_record=True)
    revision = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    result = dict(args=vars(args),synthetic=manifest['synthetic'],manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        code_revision=revision,utc_completed=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        parameter_count=sum(v.numel() for v in model.parameters()),
        windows={s:len(d) for s,d in datasets.items()},history=history,best_epoch=best_epoch,validation=val,test=metrics,
        elapsed_seconds=time.perf_counter()-started)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    x,y = next(iter(loaders['test']))
    model.eval()
    with torch.inference_mode(): pred = model(x.to(args.device)).cpu().numpy()
    np.savez(out/'predictions.npz',context=x.numpy(),target=y.numpy(),prediction=pred)
    print(json.dumps({k:v for k,v in metrics.items() if k!='per_record'}),flush=True)

if __name__ == '__main__': main()
