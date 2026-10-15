import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import Windows, prepare
from .models import build


def evaluate(model, loader, device):
    model.eval()
    squared = absolute = count = 0
    with torch.no_grad():
        for x,y in loader:
            error = model(x.to(device)).cpu()-y
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += y.numel()
    return dict(mse=squared/count,mae=absolute/count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--data',default='data/mitdb')
    p.add_argument('--model',choices=['cnn','lstm','transformer','patchtst','persistence','chronos'],default='patchtst')
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
        print(f'Prepared {args.data}')
        return
    if args.epochs < 1: p.error('epochs must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest_path = Path(args.data)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    datasets = {s:Windows(args.data,s,args.context,args.horizon,args.stride,args.limit) for s in ['train','val','test']}
    loaders = {s:DataLoader(d,batch_size=args.batch_size,shuffle=(s=='train')) for s,d in datasets.items()}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    history = []
    if args.model == 'persistence':
        class Persistence(torch.nn.Module):
            def forward(self,x): return x[:,-1:].expand(-1,args.horizon)
        model = Persistence()
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
            model.train()
            for x,y in loaders['train']:
                optim.zero_grad()
                loss = (model(x.to(args.device))-y.to(args.device)).square().mean()
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optim.step()
            val = evaluate(model,loaders['val'],args.device)
            history.append(dict(epoch=epoch+1,**val))
            print(history[-1],flush=True)
            if val['mse'] < best:
                best = val['mse']
                torch.save(model.state_dict(),out/'best.pt')
        model.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=True))
    metrics = evaluate(model,loaders['test'],args.device)
    result = dict(args=vars(args),synthetic=manifest['synthetic'],manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
        windows={s:len(d) for s,d in datasets.items()},history=history,test=metrics)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    x,y = next(iter(loaders['test']))
    model.eval()
    with torch.no_grad(): pred = model(x.to(args.device)).cpu().numpy()
    np.savez(out/'predictions.npz',context=x.numpy(),target=y.numpy(),prediction=pred)
    print(json.dumps(metrics))

if __name__ == '__main__': main()
