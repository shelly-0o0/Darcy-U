import unittest
from unittest.mock import patch
import numpy as np
import stage14_experiment as ex


class ConfirmationTests(unittest.TestCase):
    def test_all_baselines_solve_same_small_problem_and_charge_certification(self):
        outcomes=[ex.baseline_pipeline(55960,True,8,16,m) for m in ex.BASELINES]
        for out in outcomes:
            self.assertTrue(ex.valid_solver(out['solver']))
            self.assertGreater(out['full_seconds'],out['solver']['timings']['total_seconds'])
        np.testing.assert_allclose(outcomes[0]['qoi'],outcomes[1]['qoi'],rtol=1e-7,atol=1e-14)
        self.assertEqual(outcomes[2]['solve_dofs'],65)

    def test_coefficient_baseline_has_no_fine_matrix_or_amg(self):
        with patch.object(ex,'_full_cell_matrix',side_effect=AssertionError('Fine matrix forbidden')):
            result=ex.baseline_pipeline(55961,False,8,16,'coefficient_local')
        self.assertFalse(result['solver']['symmetric_diagonal_scaling'])
        self.assertEqual(ex.count_added_dofs(result['selected_cells'],8),16)

    def test_unified_acceptance_rejects_residual_backward_and_info_failures(self):
        valid=dict(converged=True,info=0,relative_residual=1e-10,componentwise_backward_error=1e-14)
        self.assertTrue(ex.valid_solver(valid))
        for key,bad in (('converged',False),('info',1),('relative_residual',2e-9),
                        ('componentwise_backward_error',2e-12)):
            self.assertFalse(ex.valid_solver({**valid,key:bad}))


if __name__=='__main__':unittest.main()
