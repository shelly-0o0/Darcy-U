import unittest
from unittest.mock import patch
import numpy as np
import stage14_mechanism as m
from stage10_experiment import coefficient,rhs_functions


class MechanismTests(unittest.TestCase):
    def test_seeded_budget_random_and_clustered_seeds(self):
        rng=np.random.default_rng(55019)
        for nc in (8,16,32):
            for _ in range(10):
                mask=np.zeros(nc*nc,dtype=bool)
                mask[rng.choice(nc*nc,max(1,nc//2),replace=False)]=True
                budget=nc*nc//4
                cells,meta=m.seeded_mark(rng.normal(size=nc*nc),nc,budget,mask)
                self.assertEqual(m.count_added_dofs(cells,nc),budget)
                self.assertTrue(set(np.flatnonzero(mask))<=set(cells))

    def test_explicit_infeasible_seed_and_capacity(self):
        with self.assertRaises(ValueError):m.seeded_mark(np.ones(16),4,1,np.ones(16,dtype=bool))
        with self.assertRaises(ValueError):m.seeded_mark(np.ones(4),2,4,np.zeros(4,dtype=bool))

    def test_fallback_keeps_seed_and_exact_budget(self):
        found=False
        for seed in range(500):
            rng=np.random.default_rng(seed);mask=np.zeros(16,dtype=bool);mask[0]=True
            cells,meta=m.seeded_mark(rng.normal(size=16),4,8,mask)
            if meta['fallback']:
                found=True;self.assertIn(0,cells);self.assertEqual(m.count_added_dofs(cells,4),8);break
        self.assertTrue(found)

    def test_bank_is_repeatable_reference_free_and_preserves_development_interfaces(self):
        for contrast in (False,True):
            geometry=55020 if not contrast else 55120
            k=coefficient(geometry,128,contrast);fs=rhs_functions(geometry)
            # Solver entry points must not be part of candidate generation.
            with patch('scipy.sparse.linalg.splu',side_effect=AssertionError('No solve in bank')):
                bank,mask=m.build_bank(k,fs,geometry,1024)
                repeated,_=m.build_bank(k,fs,geometry,1024)
            self.assertEqual(bank,repeated);self.assertEqual(len(bank),14)
            for item in bank:
                self.assertTrue(item['interface_covered'])
                self.assertTrue(item['ring_state_matched'])
                self.assertEqual(m.count_added_dofs(item['selected_cells'],64),1024)
            with self.assertRaises(ValueError):m.build_bank(k,fs+fs[1:],geometry,1024)

    def test_ring_lock_preserves_interface_centers_and_shared_edge_dofs(self):
        nc=32;g=55021;k=coefficient(g,2*nc);fs=rhs_functions(g)
        bank,mask=m.build_bank(k,fs,g,nc*nc//4)
        baseline=set(bank[0]['selected_cells'])
        for item in bank:
            selected=set(item['selected_cells'])
            for cell in np.flatnonzero(mask):
                self.assertEqual(cell in selected,cell in baseline)
                for other in m._neighbors(cell,nc):
                    self.assertEqual(cell in selected and other in selected,
                                     cell in baseline and other in baseline)

    def test_error_norm_does_not_cancel_opposite_goal_signs(self):
        value=m.error_metrics([0.,2.,1.],[1.,1.,1.],[2.,2.,2.])
        self.assertGreater(value['scaled_norm'],0.7)
        self.assertAlmostEqual(sum(value['signed_scaled_error']),0.)
        self.assertIsNone(m.error_metrics([1.,1.,1.],[1.,1.,1.],[1.,1.,1.])['E'])


if __name__=='__main__':unittest.main()
