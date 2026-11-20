import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from stage10_experiment import make_case,metrics,stacked,objective,assert_frozen,sha


class PilotAuditTests(unittest.TestCase):
    def test_exact_reference_has_zero_prediction_error_and_perfect_marking(self):
        item=make_case(51999,nc=4)
        report=metrics(item,item["target"])
        self.assertEqual(report["matrix_error_relative"],0.)
        self.assertEqual(report["defect_energy_error_mean"],0.)
        self.assertEqual(report["marker_mass_ratio"],1.)
        self.assertEqual(report["indicator_relative_l1"],0.)
        self.assertGreaterEqual(report["minimum_symmetric_eigenvalue"],-1e-15)

    def test_loss_uses_energy_and_correct_residual_pairing(self):
        batch=stacked([make_case(51998,nc=4)],"cpu")
        zero,e,g=objective(batch["target"],batch)
        self.assertLess(float(zero),1e-11)
        full,e,g=objective(torch.zeros_like(batch["target"]),batch)
        self.assertAlmostEqual(float(e),1.,places=6)
        self.assertGreater(float(g),0.)
        self.assertGreater(float(full),1.)

    def test_final_freeze_rejects_changed_protocol_source_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/"docs").mkdir()
            paths=[root/"code.py",root/"docs/stage10_protocol_zh.md",root/"weights.pt"]
            for p in paths:p.write_text("original")
            frozen=dict(source_sha256={"code.py":sha(paths[0])},protocol_sha256=sha(paths[1]))
            checkpoint=[dict(checkpoint="weights.pt",checkpoint_sha256=sha(paths[2]))]
            assert_frozen(frozen,checkpoint,root)
            for p in paths:
                p.write_text("changed")
                with self.assertRaises(RuntimeError):assert_frozen(frozen,checkpoint,root)
                p.write_text("original")


if __name__=="__main__":unittest.main()
