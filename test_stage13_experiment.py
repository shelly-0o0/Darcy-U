"""Tests of online isolation, matched outputs, score semantics and failed solvers."""
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse.linalg as sla
import stage13_experiment as ex


class CostPipelineTests(unittest.TestCase):
    def test_coefficient_score_counts_each_valid_neighbor_once(self):
        logs=np.array([[0.,1.],[3.,2.]])
        k=np.exp(np.repeat(np.repeat(logs,2,axis=0),2,axis=1))
        np.testing.assert_allclose(ex.coefficient_scores(k),[4.,2.,4.,2.],atol=1e-14)
        np.testing.assert_array_equal(ex.coefficient_scores(np.ones((8,8))),np.zeros(16))

    def test_goal_score_matches_loop_over_actual_triangles(self):
        nc=4;k=ex.old.coefficient(54961,8,True);fs=ex.old.rhs_functions(54961)
        coarse=ex.assemble_local(nc,k,fs,[])
        sol=sla.splu(coarse.A).solve(coarse.loads.T)
        actual,qc=ex.coarse_goal_scores(k,coarse,sol)
        full=np.zeros(((2*nc+1)**2,4));full[coarse.independent_free_node_ids]=sol
        scale=np.maximum(abs(qc),max(abs(qc).max()*1e-12,1e-30));expected=[]
        for y in range(nc):
            for x in range(nc):
                ids=[2*y*(2*nc+1)+2*x,2*y*(2*nc+1)+2*x+2,
                     (2*y+2)*(2*nc+1)+2*x+2,(2*y+2)*(2*nc+1)+2*x]
                norms=[]
                for tri in ((0,1,2),(0,2,3)):
                    nodes=np.array(ids)[list(tri)];yy,xx=np.divmod(nodes,2*nc+1)
                    v=np.column_stack((np.ones(3),xx/(2*nc),yy/(2*nc)))
                    gradient=np.linalg.solve(v,full[nodes])[1:]
                    n=np.linalg.norm(gradient,axis=0);norms.append(n[0]*sum(n[1:]/scale))
                cell=k[2*y:2*y+2,2*x:2*x+2]
                expected.append(cell.mean()*(1+np.log(cell).std())*np.mean(norms)/nc**2)
        np.testing.assert_allclose(actual,expected,rtol=1e-12)

    def test_local_paths_never_assemble_or_solve_global_fine_system(self):
        original=ex.sla.splu;dimensions=[]
        def tracked(a,*args,**kwargs):
            dimensions.append(a.shape[0]);return original(a,*args,**kwargs)
        with patch.object(ex,'_full_cell_matrix',side_effect=AssertionError('fine matrix forbidden')),\
             patch.object(ex.prior,'assemble_fine_only',side_effect=AssertionError('fine assembly forbidden')),\
             patch.object(ex.sla,'splu',side_effect=tracked):
            for method in ('coefficient_local','coarse_goal_local'):
                value=ex.pipeline(54962,4,True,method,4)
                self.assertFalse(value['uses_global_fine_matrix'])
                self.assertEqual(value['solve_dofs'],13)
                self.assertEqual(ex.count_added_dofs(value['selected_cells'],4),4)
        self.assertNotIn(49,dimensions)

    def test_matched_backends_recompute_same_mesh_and_solution(self):
        for contrast in (False,True):
            values=[ex.pipeline(54963,8,contrast,m,16) for m in ('coarse_goal_local','coarse_goal_galerkin')]
            self.assertEqual(values[0]['selected_cells'],values[1]['selected_cells'])
            self.assertEqual(values[0]['independent_node_sha256'],values[1]['independent_node_sha256'])
            np.testing.assert_allclose(ex.lifted_solution(values[0]),ex.lifted_solution(values[1]),rtol=1e-10,atol=1e-14)
            np.testing.assert_allclose(values[0]['qoi'],values[1]['qoi'],rtol=1e-10,atol=1e-15)
            self.assertTrue(values[0]['local_assembly']['metadata']['reused_load_data'])

    def test_all_methods_have_valid_qois_and_charged_components(self):
        ref=ex.prior.reference_case(54964,4,True)
        for method in ex.METHODS:
            value=ex.pipeline(54964,4,True,method,4);row=ex.evaluate(value,ref)
            self.assertTrue(row['numerical_valid'],method)
            self.assertLessEqual(sum(v for k,v in value['timings'].items() if k!='total_seconds'),value['timings']['total_seconds'])
            self.assertLessEqual(row['energy_error_ratio'],1+1e-8)
            if method.startswith('fine_'):self.assertLess(row['error_reduction_ratio'],1e-5)

    def test_failed_pcg_is_retained_and_never_marked_valid(self):
        ref=ex.prior.reference_case(54965,4,False)
        with patch.object(ex,'solve_pcg',return_value=(np.zeros(49),dict(info=1000,converged=False,relative_residual=1.))):
            value=ex.pipeline(54965,4,False,'fine_amg_pcg',4)
        row=ex.evaluate(value,ref)
        self.assertFalse(row['solver_converged']);self.assertFalse(row['numerical_valid'])
        self.assertEqual(row['solver']['info'],1000)

    def test_unreliable_pcg_qoi_fails_even_with_small_claimed_residual(self):
        ref=ex.prior.reference_case(54966,4,False)
        value=ex.pipeline(54966,4,False,'fine_amg_pcg',4)
        value['qoi']=value['qoi']+ref['scales']*1e-3
        self.assertFalse(ex.evaluate(value,ref)['numerical_valid'])

    def test_local_quadrature_runs_once_for_coarse_and_adaptive(self):
        import stage13_local_fem as fem
        with patch.object(fem,'prepare_loads',wraps=fem.prepare_loads) as prepare:
            ex.pipeline(54967,4,False,'coarse_goal_local',4)
        self.assertEqual(prepare.call_count,1)


if __name__=='__main__':unittest.main()
