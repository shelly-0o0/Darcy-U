"""Independent checks of action bookkeeping, reference isolation and exact budget."""
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse.linalg as sla

import stage12_experiment as ex
from stage12_schur import SchurActions,candidate_patches
from stage10_goal_numerics import assemble_nested,assemble_load


def small_problem(seed=53961):
    nc=4;k=ex.old.coefficient(seed,2*nc,True);fs=ex.old.rhs_functions(seed)
    case=assemble_nested(nc,k,fs[0]);loads=np.stack([case.ff]+[assemble_load(case,f)[0] for f in fs[1:]])
    s=SchurActions(case,loads)
    uf=sla.splu(case.Af.tocsc()).solve(loads[0]);uc=case.P@s.coarse_u[0]
    qc=loads[1:]@uc;target=loads[1:]@(uf-uc)
    scales=np.maximum(abs(qc),1e-30)
    return case,loads,s,target,scales


class ActionSearchTests(unittest.TestCase):
    def test_new_directions_match_actual_union_for_all_patch_types(self):
        case,loads,s,target,scales=small_problem()
        selected=np.zeros(16,dtype=bool);selected[[0,1,5,10]]=True
        yy,xx=np.mgrid[:4,:4];centers=((2*yy+1)*9+2*xx+1).ravel()
        choices,ids=ex.available_actions(s,candidate_patches(4),selected,ex.neighbor_table(4),centers,20)
        original=s.extra_indices(np.flatnonzero(selected));seen=set()
        for c,d in zip(choices,ids):
            union=np.union1d(np.flatnonzero(selected),c['new_cells'])
            expected=np.setdiff1d(s.extra_indices(union),original)
            np.testing.assert_array_equal(d,expected)
            self.assertNotIn(tuple(c['new_cells']),seen);seen.add(tuple(c['new_cells']))
        self.assertTrue({'single','horizontal_pair','vertical_pair','block_2x2'}<=set(c['kind'] for c in choices))

    def test_coupled_and_uncoupled_finish_exact_budget_with_reconstructible_trace(self):
        case,loads,s,target,scales=small_problem()
        for coupled in (True,False):
            result=ex.search_actions(s,target,scales,4,coupled)
            self.assertEqual(ex.count_added_dofs(result['selected_cells'],4),4)
            self.assertFalse(result['fallback'])
            change=np.zeros(3);cost=0
            for row in result['trace']:
                np.testing.assert_allclose(row['predicted_qoi_change_before'],change)
                change+=row['predicted_increment'];cost+=row['cost']
                self.assertEqual(row['spent'],cost)
                self.assertAlmostEqual(row['gain_per_dof'],(row['objective_before']-row['objective_after'])/row['cost'])
            np.testing.assert_allclose(change,result['predicted_qoi_change'])
            if coupled:self.assertLess(result['ignored_coupling_relative_change_error'],1e-9)

    def test_forced_fallback_preserves_partial_and_exact_budget(self):
        case,loads,s,target,scales=small_problem()
        with patch.object(ex,'available_actions',return_value=([],[])):
            for coupled in (True,False):
                result=ex.search_actions(s,target,scales,4,coupled)
                self.assertTrue(result['fallback'])
                self.assertEqual(result['partial_before_fallback']['added_dofs'],0)
                self.assertEqual(len(result['fallback_options']),2)
                self.assertEqual(ex.count_added_dofs(result['selected_cells'],4),4)

    def test_nonpositive_gains_do_not_silently_end_budget(self):
        case,loads,s,target,scales=small_problem()
        result=ex.search_actions(s,np.zeros(3),scales,4,True)
        self.assertEqual(result['added_dofs'],4)
        self.assertGreater(result['negative_predicted_gain_steps'],0)

    def test_amg_target_does_not_solve_fine_primal(self):
        original=sla.splu;dimensions=[]
        def tracked(a,*args,**kwargs):
            dimensions.append(a.shape[0]);return original(a,*args,**kwargs)
        with patch.object(ex.sla,'splu',side_effect=tracked):
            value=ex.action_pipeline(53962,4,True,'amg_coupled',4)
        self.assertNotIn(49,dimensions)
        self.assertIsNone(value['oracle_label_relative_residual'])
        self.assertNotIn('oracle_fine_primal_target_seconds',value['timings'])
        self.assertLess(value['schur_identity_relative_error'],1e-8)

    def test_oracle_target_and_actual_fem_values_are_consistent(self):
        ref=ex.previous.reference_case(53963,4,True)
        for method in ('oracle_coupled','oracle_uncoupled'):
            value=ex.action_pipeline(53963,4,True,method,4)
            np.testing.assert_allclose(value['search']['target_error'],ref['qf']-ref['qc'],rtol=1e-10,atol=1e-15)
            row=ex.previous.evaluate(value,ref)
            self.assertLess(row['schur_identity_relative_error'],1e-8)
            self.assertLessEqual(row['energy_error_ratio'],1+1e-8)
            self.assertEqual(row['added_dofs'],4)
            self.assertIn('oracle_fine_primal_target_seconds',value['timings'])
            total=sum(v for k,v in value['timings'].items() if k!='total_seconds')
            self.assertLessEqual(total,value['timings']['total_seconds'])

    def test_invalid_targets_and_budget_fail_before_search(self):
        case,loads,s,target,scales=small_problem()
        for bad in (np.array([np.nan,0.,0.]),np.zeros(2)):
            with self.assertRaises(ValueError):ex.search_actions(s,bad,scales,4,True)
        with self.assertRaises(ValueError):ex.search_actions(s,target,scales,9,True)


if __name__=='__main__':unittest.main()
