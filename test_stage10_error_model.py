import unittest
import numpy as np
import torch
from numerics import assemble_system
from stage10_error_model import ErrorOperator, energy


class ModelStructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_energy_matches_sparse_nonconstant_operator(self):
        k=np.exp(np.random.default_rng(2).normal(size=(8,8)))
        a,_,_=assemble_system(k); m=7
        d=torch.tensor(a.diagonal().reshape(1,1,m,m))
        east=torch.tensor(np.pad(a.diagonal(1),(0,1)).reshape(1,1,m,m))[...,:-1]
        south=torch.tensor(a.diagonal(m).reshape(1,1,m-1,m))
        x=torch.randn(3,1,m,m,dtype=torch.float64)
        actual=energy(x,d,east,south).numpy()
        wanted=np.array([z.ravel()@(a@z.ravel()) for z in x.numpy()])
        np.testing.assert_allclose(actual,wanted,rtol=1e-12,atol=1e-12)

    def test_symmetric_operator_linearity_reciprocity_and_positivity(self):
        for n in (8,9,16):
            model=ErrorOperator(width=3).double()
            feat=torch.randn(1,3,n-1,n-1,dtype=torch.float64)
            d=torch.rand(1,1,n-1,n-1,dtype=torch.float64)+1
            x,y=torch.randn(2,1,n-1,n-1,dtype=torch.float64).split(1)
            hx=model(feat,x,d,n); hy=model(feat,y,d,n)
            mixed=model(feat,2*x-3*y,d,n)
            torch.testing.assert_close(mixed,2*hx-3*hy,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close((x*hy).sum(),(y*hx).sum(),rtol=1e-12,atol=1e-12)
            self.assertGreater(float((x*hx).sum()),0.)
            self.assertEqual(float(model(feat,torch.zeros_like(x),d,n).abs().max()),0.)

    def test_full_small_operator_is_spd_and_gradient_flows(self):
        n=5; q=(n-1)**2
        model=ErrorOperator(width=3).double()
        feat=torch.randn(1,3,n-1,n-1,dtype=torch.float64).expand(q,-1,-1,-1)
        diagonal=torch.ones(q,1,n-1,n-1,dtype=torch.float64)
        basis=torch.eye(q,dtype=torch.float64).reshape(q,1,n-1,n-1)
        matrix=model(feat,basis,diagonal,n).reshape(q,q)
        torch.testing.assert_close(matrix,matrix.T,rtol=1e-12,atol=1e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(matrix).min()),0.)
        matrix.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_untied_is_linear_but_not_forced_symmetric_after_update(self):
        model=ErrorOperator("untied",width=3).double()
        with torch.no_grad(): model.synthesis[0].add_(.2*torch.randn_like(model.synthesis[0]))
        feat=torch.randn(1,3,7,7,dtype=torch.float64); d=torch.ones(1,1,7,7,dtype=torch.float64)
        x,y=torch.randn(2,1,7,7,dtype=torch.float64).split(1)
        hx=model(feat,x,d,8);hy=model(feat,y,d,8)
        torch.testing.assert_close(model(feat,2*x-y,d,8),2*hx-hy,rtol=1e-12,atol=1e-12)
        self.assertGreater(float(((x*hy).sum()-(y*hx).sum()).abs()),1e-6)


if __name__ == "__main__": unittest.main()
