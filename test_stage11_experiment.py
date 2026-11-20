"""Reference separation, shared discretization and actual-output checks."""
import unittest
import numpy as np
import scipy.sparse.linalg as sla
import torch

import stage11_experiment as experiment
from stage10_goal_numerics import assemble_nested, assemble_load, _vertices
from stage10_error_model import ErrorOperator


class Stage11ExperimentTests(unittest.TestCase):
    def test_direct_assembly_matches_nested_for_all_loads(self):
        nc=4
        k=experiment.old.coefficient(52960,2*nc,True)
        funcs=experiment.old.rhs_functions(52960)
        a,loads=experiment.assemble_fine_only(k,funcs)
        case=assemble_nested(nc,k,funcs[0])
        np.testing.assert_allclose(a.toarray(),case.Af.toarray(),rtol=0,atol=0)
        for j,f in enumerate(funcs):
            np.testing.assert_allclose(loads[j],assemble_load(case,f)[0],rtol=0,atol=0)

    def test_quadratic_recovery_reproduces_boundary_zero_biquadratic(self):
        nc=5
        fine,_=_vertices(2*nc); coarse,_=_vertices(nc)
        f=lambda xy:xy[:,0]*(1-xy[:,0])*xy[:,1]*(1-xy[:,1])
        _,fi=_vertices(2*nc);_,ci=_vertices(nc)
        np.testing.assert_allclose(experiment.quadratic_recovery_matrix(nc)@f(coarse)[ci],
                                   f(fine)[fi],rtol=1e-13,atol=1e-15)

    def test_actual_adaptive_outcomes_and_direct_one_primal(self):
        nc=4;seed=52961
        ref=experiment.reference_case(seed,nc,True)
        for method in (*experiment.CONTROLS,"fine_direct"):
            value=experiment.online_pipeline(seed,nc,True,method,budget=4)
            row=experiment.evaluate(value,ref)
            self.assertLessEqual(row["energy_error_ratio"],1+1e-10)
            if method=="fine_direct":
                self.assertLess(row["error_reduction_ratio"],1e-10)
                self.assertNotIn("coarse_factor_solve_seconds",value["timings"])
                self.assertEqual(value["solve_dofs"],49)
            else:
                self.assertEqual(row["added_dofs"],4)
                self.assertEqual(row["solve_dofs"],13)
                self.assertEqual(len(row["qoi"]),3)
            components=sum(v for k,v in value["timings"].items() if k!="total_seconds")
            self.assertLessEqual(components,value["timings"]["total_seconds"])
        uniform=experiment.online_pipeline(seed,nc,True,"geometry_uniform",budget=4)
        self.assertNotIn("projection_seconds",uniform["timings"])

    def test_fixed_geometry_scores_do_not_depend_on_coefficients(self):
        a=experiment.online_pipeline(52962,4,False,"geometry_uniform",budget=4)
        b=experiment.online_pipeline(52963,4,True,"geometry_uniform",budget=4)
        self.assertEqual(a["selected_cells"],b["selected_cells"])

    def test_learned_pipeline_handles_three_goals_and_projects(self):
        torch.set_num_threads(1)
        torch.manual_seed(52964)
        model=ErrorOperator("symmetric").eval()
        result=experiment.online_pipeline(52964,4,False,"symmetric",model,budget=4)
        self.assertEqual(result["qoi"].shape,(3,))
        self.assertLess(result["projection_orthogonality"],1e-10)
        self.assertEqual(result["added_dofs"],4)

    def test_goal_metric_prevents_signed_cancellation_and_flags_zero(self):
        ref=experiment.reference_case(52965,4,False)
        out=experiment.online_pipeline(52965,4,False,"fine_direct")
        out["qoi"]=ref["qf"]+ref["scales"]*np.array([.1,-.1,0])
        row=experiment.evaluate(out,ref)
        self.assertAlmostEqual(row["scaled_error_l2"],np.sqrt(.02))
        ref["coarse_scaled_error"]=0.
        row=experiment.evaluate(out,ref)
        self.assertTrue(row["coarse_error_near_zero"])
        self.assertIsNone(row["error_reduction_ratio"])


if __name__=="__main__":unittest.main()
