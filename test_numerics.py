"""Small, meaningful checks for the finite-element and Schwarz machinery."""
import unittest
from unittest.mock import patch as mock_patch
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla

from numerics import (assemble_system,build_patches,b_orthonormalize,
                      patch_spectral_basis,certify_space,build_preconditioner,
                      solve_pcg,free_to_grid,energy_relative_error)


class NumericalTests(unittest.TestCase):
    def test_p1_manufactured_convergence(self):
        errors=[]
        for n in (8,16,32):
            f=lambda x,y:2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
            a,b,xy=assemble_system(np.ones((n,n)),f)
            x=sla.spsolve(a,b)
            exact=np.sin(np.pi*xy[:,0])*np.sin(np.pi*xy[:,1])
            errors.append(np.linalg.norm(x-exact)/n)
            self.assertTrue(np.allclose(a.toarray(),a.toarray().T))
        self.assertGreater(errors[0]/errors[1],3.5)
        self.assertGreater(errors[1]/errors[2],3.5)

    def test_load_quadrature_and_grid_mapping(self):
        a,b,xy=assemble_system(np.ones((8,8)))
        aa,bb,_=assemble_system(np.ones((8,8)),lambda x,y:np.ones_like(x))
        np.testing.assert_allclose(b,bb,rtol=1e-14)
        xx=np.arange(49.)
        grid=free_to_grid(xx)
        np.testing.assert_array_equal(grid[1:-1,1:-1].ravel(),xx)
        self.assertTrue(np.all(grid[0]==0))

    def test_partition_spectral_certificate_and_solve(self):
        n=24
        k=np.ones((n,n));k[7:11,:]=1e4;k[:,14:17]=1e4
        a,b,_=assemble_system(k)
        patches=build_patches(k,a,core_cells=8,overlap=2)
        count=np.zeros(a.shape[0])
        spaces=[]
        for p in patches:
            count[p.ids]+=p.weights
            la.cholesky(p.B)
            la.cholesky(p.A_D)
            z=patch_spectral_basis(p)
            np.testing.assert_allclose(z.T@p.B@z,np.eye(z.shape[1]),atol=1e-8)
            self.assertTrue(certify_space(p.A_N,p.B,z)["passed"])
            values=np.arange(len(p.ids),dtype=float)
            np.testing.assert_array_equal(p.from_grid(p.to_grid(values,32)),values)
            spaces.append(z)
        np.testing.assert_allclose(count,1.)
        m,stats=build_preconditioner(a,patches,spaces)
        rng=np.random.default_rng(1)
        u,v=rng.normal(size=(2,a.shape[0]))
        self.assertAlmostEqual(float(u@(m@v)),float(v@(m@u)),places=7)
        self.assertGreater(float(u@(m@u)),0.)
        x,metrics=solve_pcg(a,b,m,rtol=1e-9)
        ref=sla.spsolve(a,b)
        self.assertTrue(metrics["converged"])
        self.assertLess(energy_relative_error(a,x,ref),1e-8)
        self.assertLess(stats["coarse_dimension"],a.shape[0])
        # Cached local factors are reused on identical next-step matrices.
        fresh=build_patches(k,a,core_cells=8,overlap=2,previous_patches=patches)
        _,repeat_stats=build_preconditioner(a,fresh,spaces)
        self.assertEqual(repeat_stats["new_local_factors"],0)

    def test_full_certificate_rejects_complement_false_positive(self):
        # Z spans e1.  The complementary Rayleigh quotient is 1 > gamma,
        # but minimizing the energy over the Z component gives 0.19 < 0.5.
        a=np.array([[1.,.9],[.9,1.]])
        b=np.eye(2)
        z=np.array([[1.],[0.]])
        self.assertGreater(a[1,1],.5)
        result=certify_space(a,b,z,gamma=.5)
        self.assertFalse(result["passed"])
        self.assertLess(result["min_eig"],-.01)
        vals,vecs=la.eigh(a,b)
        self.assertTrue(certify_space(a,b,vecs[:,:1],gamma=.5)["passed"])

    def test_neumann_kernel_must_be_included(self):
        a=np.array([[1.,-1.],[-1.,1.]])
        b=np.eye(2)
        self.assertFalse(certify_space(a,b,np.empty((2,0)))["passed"])
        z=np.ones((2,1))
        self.assertTrue(certify_space(a,b,z)["passed"])

    def test_incremental_orthogonalization_removes_old_directions(self):
        rng=np.random.default_rng(4)
        s=rng.normal(size=(12,12));b=s.T@s+np.eye(12)
        old=b_orthonormalize(rng.normal(size=(12,3)),b)
        proposed=np.column_stack((old[:,0],rng.normal(size=12)))
        new=b_orthonormalize(proposed,b,existing=old)
        self.assertEqual(new.shape[1],1)
        np.testing.assert_allclose(old.T@b@new,0.,atol=1e-12)
        np.testing.assert_allclose(new.T@b@new,np.eye(1),atol=1e-12)

    def test_ldl_matches_full_eigen_checks(self):
        a=np.array([[1.,-1.],[-1.,1.]])
        for z in (np.empty((2,0)),np.ones((2,1)),np.eye(2)):
            for tol in (0.,5e-10):
                direct=certify_space(a,np.eye(2),z,tolerance=tol)
                inertia=certify_space(a,np.eye(2),z,tolerance=tol,method="ldl")
                self.assertEqual(direct["passed"],inertia["passed"])
        rng=np.random.default_rng(22)
        for _ in range(12):
            q,_=la.qr(rng.normal(size=(18,18)))
            a=(q*np.geomspace(1e-5,1,18)[None,:])@q.T
            z=rng.normal(size=(18,6))
            direct=certify_space(a,np.eye(18),z,gamma=.05)
            inertia=certify_space(a,np.eye(18),z,gamma=.05,method="ldl")
            self.assertEqual(direct["passed"],inertia["passed"])

    def test_residual_refresh_uses_true_residual_and_counts_iterations(self):
        a,b,_=assemble_system(np.ones((8,8)))
        real_cg=sla.cg
        calls=[]
        def cg_with_one_false_convergence(*args,**kwargs):
            calls.append(1)
            if len(calls)==1:
                x=np.zeros_like(args[1])
                kwargs["callback"](x)
                return x,0
            return real_cg(*args,**kwargs)
        with mock_patch("numerics.sla.cg",side_effect=cg_with_one_false_convergence):
            x,metrics=solve_pcg(a,b,rtol=1e-9,max_restarts=3)
        self.assertEqual(metrics["initial_info"],0)
        self.assertEqual(metrics["initial_relative_residual"],1.)
        self.assertEqual(metrics["restarts"],1)
        self.assertTrue(metrics["converged"])
        self.assertLessEqual(metrics["relative_residual"],1e-9)
        self.assertGreater(metrics["iterations"],1)


if __name__=="__main__":
    unittest.main(verbosity=2)
