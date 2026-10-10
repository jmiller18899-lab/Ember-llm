"""Offline proposal only; this probe is never imported by a training job."""
import importlib.util
from pathlib import Path
import unittest

PROBE = Path(__file__).resolve().parents[1] / 'reports/probes/wr8_loss_probe_20261010.py'


class LossProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PROBE.is_file(), 'No isolated first-divergence experiment exists')
        spec = importlib.util.spec_from_file_location('wr8_loss_probe', PROBE)
        self.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.probe)

    def test_shared_prefix_and_suffix_are_not_penalized(self):
        self.assertEqual(self.probe.first_divergence_mask([1, 2, 3, 9], [1, 2, 4, 9]), [False, False, True, False])

    def test_early_end_is_a_real_divergence(self):
        self.assertEqual(self.probe.first_divergence_mask([1, 2, 9], [1, 9]), [False, True])

    def test_identical_sequences_are_not_negative_examples(self):
        with self.assertRaises(ValueError):
            self.probe.first_divergence_mask([1, 9], [1, 9])


if __name__ == '__main__':
    unittest.main()
