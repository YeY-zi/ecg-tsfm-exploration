import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from ecgts.data import prepare, Windows, SPLITS
from ecgts.models import build


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

    def test_models_gradients(self):
        torch.set_num_threads(1)
        for name in ['cnn','lstm','transformer','patchtst']:
            model = build(name,32,8)
            output = model(torch.randn(2,32))
            self.assertEqual(output.shape,(2,8))
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__ == '__main__': unittest.main()
