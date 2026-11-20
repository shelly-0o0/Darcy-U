"""Shared acceptance, actual residuals, scaled solves and correction failures."""
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import stage14_solver as solver


class CertifiedSolverTests(unittest.TestCase):
    def setUp(self):
        n=40
        self.a=sp.diags((-np.ones(n-1),2.2*np.ones(n),-np.ones(n-1)),(-1,0,1)).tocsr()
        self.b=np.column_stack((np.ones(n),np.linspace(-1,1,n),np.zeros(n)))

    def test_direct_and_amg_match_independent_solution_for_three_rhs(self):
        expected=sla.spsolve(self.a,self.b)
        for method in ('direct','amg'):
            u,m=solver.solve_system(self.a,self.b,method)
            np.testing.assert_allclose(u,expected,rtol=1e-8,atol=1e-10)
            self.assertTrue(m['converged']);self.assertEqual(m['rhs_count'],3)
            self.assertEqual(m['per_rhs'][2]['relative_residual'],0.)
            self.assertLessEqual(m['componentwise_backward_error'],1e-12)
            self.assertLessEqual(sum(v for k,v in m['timings'].items() if k!='total_seconds'),m['timings']['total_seconds'])

    def test_vector_shape_and_inputs_are_unchanged(self):
        a=self.a.copy(); b=self.b[:,0].copy()
        data=a.data.copy(); original=b.copy()
        u,m=solver.solve_system(a,b,'amg')
        self.assertEqual(u.shape,b.shape)
        np.testing.assert_array_equal(a.data,data);np.testing.assert_array_equal(b,original)
        self.assertTrue(m['symmetric_diagonal_scaling'])

    def test_componentwise_backward_error_rejects_bad_small_load_row(self):
        a=sp.eye(2);b=np.array([1.,1e-20]);u=np.array([1.,0.])
        m,r=solver.ResidualEvaluator(a).evaluate(u,b)
        self.assertLess(m['relative_residual'],1e-9)
        self.assertEqual(m['componentwise_backward_error'],1.)
        self.assertFalse(m['accepted'])

    def test_extended_residual_reveals_double_cancellation(self):
        a=sp.csr_matrix([[1e16,1.],[1.,1.]])
        b=np.array([1e16,2.]);u=np.ones(2)
        m,r=solver.ResidualEvaluator(a).evaluate(u,b)
        if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
            self.assertEqual(m['relative_residual_double'],0.)
            self.assertEqual(float(r[0,0]),-1.)
            self.assertGreater(m['relative_residual'],0.)

    def test_scaled_amg_handles_heterogeneous_diagonal_without_changing_problem(self):
        d=sp.diags(np.geomspace(.01,100.,self.a.shape[0]))
        a=(d@self.a@d).tocsr();expected=np.sin(np.arange(a.shape[0]));b=a@expected
        u,m=solver.solve_system(a,b,'amg')
        np.testing.assert_allclose(u,expected,rtol=1e-5,atol=1e-7)
        self.assertTrue(m['converged'])
        self.assertLessEqual(m['relative_residual'],solver.CONFIG['relative_residual_limit'])

    def test_false_cg_success_does_not_override_true_residual(self):
        def broken(a,b,**kwargs): return np.zeros_like(b),0
        with patch.object(solver.sla,'cg',side_effect=broken):
            u,m=solver.solve_system(self.a,self.b[:,0],'amg')
        self.assertFalse(m['converged']);self.assertEqual(m['relative_residual'],1.)
        self.assertEqual(m['total_refinements'],2)
        self.assertEqual(len(m['residual_history']),3)

    def test_cg_failure_is_retained_even_if_returned_solution_is_exact(self):
        def broken(a,b,**kwargs): return sla.spsolve(a,b),5
        with patch.object(solver.sla,'cg',side_effect=broken):
            _,m=solver.solve_system(self.a,self.b[:,0],'amg')
        self.assertTrue(m['accepted']);self.assertFalse(m['converged']);self.assertEqual(m['info'],5)

    def test_residual_correction_reuses_lu_and_is_charged(self):
        real=sla.splu;calls=[]
        class PerturbedFirstSolve:
            def __init__(self,a):self.factor=real(a);self.n=0
            def solve(self,b):
                self.n+=1;u=self.factor.solve(b)
                return u+.01 if self.n==1 else u
        def factor(a):calls.append(a.shape);return PerturbedFirstSolve(a)
        with patch.object(solver.sla,'splu',side_effect=factor):
            _,m=solver.solve_system(self.a,self.b[:,0],'direct')
        self.assertEqual(len(calls),1);self.assertEqual(m['total_refinements'],1)
        self.assertFalse(m['initial']['accepted']);self.assertTrue(m['converged'])
        self.assertGreater(m['timings']['certification_and_refinement_seconds'],0.)

    def test_invalid_shape_finiteness_symmetry_and_method_fail_explicitly(self):
        invalid=[(sp.eye(2),np.ones(3),'direct'),(sp.eye(2),np.array([1.,np.nan]),'direct'),
                 (sp.csr_matrix([[2.,1.],[0.,2.]]),np.ones(2),'amg'),
                 (sp.diags([1.,-1.]),np.ones(2),'amg'),(sp.eye(2),np.ones(2),'unknown')]
        for a,b,method in invalid:
            with self.assertRaises(ValueError):solver.solve_system(a,b,method)
        with self.assertRaises(ValueError):solver.solve_system(self.a,self.b,'direct',max_refinements=3)


if __name__=='__main__':unittest.main()
