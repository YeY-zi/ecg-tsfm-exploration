import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from ecgts.data import prepare, Windows, SPLITS
from ecgts.models import build
from ecgts.run import evaluate
from torch.utils.data import DataLoader


class PipelineTests(unittest.TestCase):
    def test_no_subject_leakage(self):
        a,b,c = [set(SPLITS[k]) for k in ['train','val','test']]
        self.assertFalse(a & b or a & c or b & c)
        self.assertNotIn('201',a | b | c)  # Same individual as 202.

    def test_normalization_and_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            meta = prepare(root,True)
            train = np.load(Path(root)/'toy_train.npy')
            self.assertAlmostEqual(meta['mean'],float(train.mean()),places=5)
            self.assertAlmostEqual(meta['std'],float(train.std()),places=5)
            ds = Windows(root,'test',256,64,128)
            a = np.load(Path(root)/'toy_test.npy')
            x,y = ds[0]
            np.testing.assert_allclose(y,(a[256:320]-meta['mean'])/meta['std'])
            self.assertEqual(len(ds), (len(a)-320)//128+1)
            self.assertEqual(x.shape,(256,))

    def test_record_metrics_match_global(self):
        with tempfile.TemporaryDirectory() as root:
            prepare(root, True)
            ds = Windows(root, 'test', 32, 8, 16, 7)
            class Zero(torch.nn.Module):
                def forward(self, x): return torch.zeros(len(x), 8)
            metrics = evaluate(Zero(), DataLoader(ds, batch_size=3), 'cpu', by_record=True)
            expected = np.concatenate([ds[i][1] for i in range(len(ds))])
            self.assertAlmostEqual(metrics['mse'], float((expected**2).mean()), places=5)
            self.assertAlmostEqual(metrics['mae'], float(np.abs(expected).mean()), places=5)
            self.assertEqual(metrics['per_record']['toy_test']['windows'], 7)
            self.assertEqual(metrics['macro_record_mse'], metrics['mse'])

    def test_models_gradients(self):
        torch.set_num_threads(1)
        for name in ['cnn','lstm','transformer','patchtst']:
            model = build(name,32,8)
            output = model(torch.randn(2,32))
            self.assertEqual(output.shape,(2,8))
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__ == '__main__': unittest.main()
