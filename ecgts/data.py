from pathlib import Path
import json
import hashlib
import numpy as np
from torch.utils.data import Dataset

# Common MIT-BIH DS1/DS2 partition; 201 and 202 share a subject.
# Exclude 201 to enforce subject separation from test record 202.
TRAIN = '101 106 108 109 112 114 115 116 118 119 122 124 203 205 207 208 209 215 220 223'.split()
VAL = '214'.split()
TEST = '100 103 105 111 113 117 121 123 200 202 210 212 213 219 221 222 228 230 231 232 233 234'.split()
SPLITS = dict(train=TRAIN, val=VAL, test=TEST)


def prepare(root, synthetic=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if synthetic:
        rng = np.random.default_rng(42)
        splits = dict(train=['toy_train'], val=['toy_val'], test=['toy_test'])
        for i, record in enumerate(sum(splits.values(), [])):
            t = np.arange(8192) / 360
            signal = np.sin(2*np.pi*(1+i*.1)*t) + .2*np.sin(2*np.pi*7*t)
            np.save(root / f'{record}.npy', (signal + rng.normal(0,.05,len(t))).astype('float32'))
    else:
        import wfdb
        splits = SPLITS
        raw = root / 'raw'
        raw.mkdir(exist_ok=True)
        records = sum(splits.values(), [])
        missing = [r for r in records if not (root / f'{r}.npy').exists()]
        if missing:
            wfdb.dl_database('mitdb', dl_dir=str(raw), records=missing, annotators=[])
        for record in records:
            if not (root / f'{record}.npy').exists():
                ecg = wfdb.rdrecord(str(raw / record))
                if 'MLII' not in ecg.sig_name or ecg.fs != 360:
                    raise ValueError(f'Unexpected lead/rate: {record} {ecg.sig_name} {ecg.fs}')
                np.save(root / f'{record}.npy', ecg.p_signal[:,ecg.sig_name.index('MLII')].astype('float32'))
    # Fit only to training records, before windowing.
    arrays = [np.load(root / f'{r}.npy') for r in splits['train']]
    count = sum(a.size for a in arrays)
    mean = sum(a.astype('float64').sum() for a in arrays) / count
    std = np.sqrt(sum(((a.astype('float64')-mean)**2).sum() for a in arrays)/count)
    if not np.isfinite(mean+std) or std <= 0:
        raise ValueError('Invalid training statistics')
    manifest = dict(synthetic=synthetic, fs=360, channel='MLII' if not synthetic else 'synthetic',
                    splits=splits, mean=float(mean), std=float(std))
    manifest['records'] = {}
    for record in sum(splits.values(), []):
        path = root / f'{record}.npy'
        a = np.load(path)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError(f'Invalid signal: {record}')
        manifest['records'][record] = dict(samples=int(a.size), npy_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest['source'] = 'synthetic' if synthetic else 'https://physionet.org/content/mitdb/1.0.0/'
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class Windows(Dataset):
    def __init__(self, root, split, context=256, horizon=64, stride=128, limit=None):
        if min(context, horizon, stride) <= 0:
            raise ValueError('Window dimensions must be positive')
        root = Path(root)
        meta = json.loads((root/'manifest.json').read_text())
        self.records = meta['splits'][split]
        self.arrays, self.index = [], []
        self.context, self.horizon = context, horizon
        for record in meta['splits'][split]:
            a = np.load(root/f'{record}.npy')
            a = ((a-meta['mean'])/meta['std']).astype('float32')
            self.arrays.append(a)
            self.index.extend((len(self.arrays)-1, start) for start in range(0,len(a)-context-horizon+1,stride))
        if limit is not None:
            if limit <= 0: raise ValueError('limit must be positive')
            # Evenly sample across records instead of selecting the first patient.
            ids = np.linspace(0,len(self.index)-1,min(limit,len(self.index)),dtype=int)
            self.index = [self.index[i] for i in ids] if self.index else []
        if not self.index: raise ValueError(f'No windows for {split}')

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        record, start = self.index[idx]
        a = self.arrays[record]
        end = start+self.context
        return a[start:end].copy(), a[end:end+self.horizon].copy()
