"""CPU checks of the interfaces shared by training and data preparation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from s2_train_locator import Locator
from s3_train_reader import Reader, Windows
from t0_g1g2_probe import LocalPRC, N_CHIPS, prc_encode, prc_soft_decode


class TrainingComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_local_carrier_recovers_32_bits(self):
        for message in [0, 0xFFFFFFFF, 0x12345678]:
            key = b"component-check"
            carrier = LocalPRC(key, 32, 8, 8)
            chips = prc_encode(message, key, N_CHIPS, 32)
            soft = carrier.demodulate(carrier.make_delta(chips))
            bits, score = prc_soft_decode(soft, key, N_CHIPS, 32)
            expected = np.array([(message >> i) & 1 for i in range(31, -1, -1)])
            np.testing.assert_array_equal(bits, expected)
            self.assertGreater(score, 0)

    def test_locator_handles_class_and_null_queries(self):
        model = Locator(n_cls=33).eval()
        with torch.no_grad():
            z = torch.randn(2, 24, 32, 48, 84)
            out = model(z, torch.tensor([0, 32]))
        self.assertEqual(tuple(out.shape), (2, 32, 48, 84))
        self.assertTrue(torch.isfinite(out).all())

    def test_reader_output_grid(self):
        model = Reader(base=8).eval()
        with torch.no_grad():
            residual, evidence = model(torch.randn(1, 3, 9, 64, 64))
        self.assertEqual(tuple(residual.shape), (1, 32, 4, 4))
        self.assertEqual(residual.shape, evidence.shape)
        self.assertTrue(torch.isfinite(residual).all())

    def test_paired_dataset_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "example_00"
            source.mkdir()
            shape = (107, 128, 128, 3)
            np.save(source / "clean.npy", np.full(shape, 120, np.uint8))
            np.save(source / "w0.npy", np.full(shape, 121, np.uint8))
            np.save(source / "target0.npy", np.ones((32, 8, 8), np.float16))
            np.save(source / "mask.npy", np.ones((32, 8, 8), np.float16))
            (source / "meta.json").write_text(json.dumps({"draws": [{"k": 0}]}))
            dataset = Windows(tmp, win=8, length=1)
            x, target, mask = dataset[0]
            self.assertEqual(tuple(x.shape), (2, 107, 128, 128, 3))
            self.assertEqual(tuple(target.shape), (2, 32, 8, 8))
            self.assertTrue(torch.all(target[1] == 0))
            self.assertTrue(torch.all(mask[1] == 0))
            self.assertTrue(torch.all(x[0].int() - x[1].int() == 1))

    def test_custom_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["OBJECTLEVELWM_DATA"] = tmp
            env["PYTHONPATH"] = str(SCRIPTS)
            result = subprocess.run([sys.executable, "-c",
                "from project_paths import DATA_ROOT; print(DATA_ROOT)"],
                capture_output=True, text=True, check=True, env=env)
            self.assertEqual(Path(result.stdout.strip()), Path(tmp))


if __name__ == "__main__":
    unittest.main()
