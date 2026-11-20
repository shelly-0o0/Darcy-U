"""Independent numerical checks for the Stage 15 G0 finite-element path.

Run with ``python -m unittest -v test_stage15_numerics`` from this directory.
These checks establish small-problem algebra and manufactured-data consistency;
they do not establish estimator reliability, adaptive convergence, or AI benefit.
The P1 reference assembler below uses affine triangle calculus, not skfem.asm.
"""

import unittest
from dataclasses import replace

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import spsolve

import stage15_cases as cases
import stage15_experiment as experiment
import stage15_fem as fem


def _constant(value):
    return lambda x, y: np.full(np.broadcast(x, y).shape, value, dtype=float)


def _independent_p1(mesh, coefficient=1.0, source=1.0):
    """Dense reference stiffness/load from physical affine shape functions."""
    matrix = np.zeros((mesh.p.shape[1], mesh.p.shape[1]))
    load = np.zeros(mesh.p.shape[1])
    for nodes in mesh.t.T:
        xy = mesh.p[:, nodes].T
        affine = np.column_stack((np.ones(3), xy))
        gradients = np.linalg.inv(affine)[1:, :].T
        area = abs(np.linalg.det(affine)) / 2.0
        matrix[np.ix_(nodes, nodes)] += coefficient * area * (gradients @ gradients.T)
        load[nodes] += source * area / 3.0
    return matrix, load


def _independent_layer_solution(x, y, values):
    """Integrate (c-t)/k(t) piecewise without calling case helpers."""
    edges = np.linspace(0.0, 1.0, 5)
    reciprocal_mass = np.sum(np.diff(edges) / values)
    first_moment = np.sum(np.diff(edges**2) / (2.0 * values))
    c = first_moment / reciprocal_mass
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    v = np.zeros_like(x)
    for left, right, value in zip(edges[:-1], edges[1:], values):
        upper = np.clip(x, left, right)
        v += (c * (upper - left) - (upper**2 - left**2) / 2.0) / value
    return v * np.sin(np.pi * y), c, v


def _quadrature_integral(function, xbreaks, ybreaks, order=32):
    """Independent tensor Gauss integration on known material/support pieces."""
    nodes, weights = leggauss(order)
    answer = 0.0
    for x0, x1 in zip(xbreaks[:-1], xbreaks[1:]):
        x = x0 + (nodes + 1.0) * (x1 - x0) / 2.0
        wx = weights * (x1 - x0) / 2.0
        for y0, y1 in zip(ybreaks[:-1], ybreaks[1:]):
            y = y0 + (nodes + 1.0) * (y1 - y0) / 2.0
            wy = weights * (y1 - y0) / 2.0
            answer += np.einsum('i,ij,j->', wx, function(x[:, None], y[None, :]), wy)
    return float(answer)


class ManufacturedCaseTests(unittest.TestCase):
    def test_bump_normalization_constant_and_support(self):
        # Integrate 1 - 4s^2 + 6s^4 - 4s^6 + s^8 exactly on [-1, 1].
        integral = 2.0 * (1.0 - 4.0/3.0 + 6.0/5.0 - 4.0/7.0 + 1.0/9.0)
        self.assertAlmostEqual(cases.BUMP_INTEGRAL, integral, delta=1e-15)
        values = cases.bump_1d(np.array([.01, .03, .25, .47, .49]), .25)
        self.assertGreater(float(values[2]), 0.0)
        np.testing.assert_allclose(values[[0, 1, 3, 4]], 0.0, atol=1e-50)
        self.assertTrue(np.all(values >= 0.0))

    def test_bump_rejects_nonpositive_or_nonfinite_radius_and_center(self):
        for radius in (0.0, -0.22, np.nan, np.inf, -np.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                cases.bump_1d(np.array([.2, .3]), .25, radius=radius)
        for center in (np.nan, np.inf, -np.inf):
            with self.subTest(center=center), self.assertRaises(ValueError):
                cases.bump_1d(np.array([.2, .3]), center)

    def test_layered_solution_matches_independent_piecewise_integral(self):
        x = np.array([0.0, 0.08, 0.25, 0.33, 0.5, 0.62, 0.75, 0.93, 1.0])[:, None]
        y = np.array([0.0, 0.17, 0.5, 0.87, 1.0])[None, :]
        for barrier in (False, True):
            with self.subTest(barrier=barrier):
                case = cases.manufactured_case('layered', contrast=100.0, barrier=barrier)
                high = 0.01 if barrier else 100.0
                expected, c, v = _independent_layer_solution(x, y, np.array([1., high, 1., high]))
                np.testing.assert_allclose(case.exact(x, y), expected, rtol=2e-13, atol=2e-14)
                coefficient = case.coefficient(x, y)
                expected_source = (1.0 + np.pi**2 * coefficient * v) * np.sin(np.pi * y)
                np.testing.assert_allclose(case.source(x, y), expected_source, rtol=2e-13, atol=2e-13)
                np.testing.assert_allclose(case.exact_grad(x, y)[0] * coefficient,
                                           (c - x) * np.sin(np.pi * y), rtol=2e-13, atol=2e-13)

    def test_manufactured_dirichlet_boundary_is_zero(self):
        t = np.linspace(0.0, 1.0, 31)
        x = np.concatenate((t, t, np.zeros_like(t), np.ones_like(t)))
        y = np.concatenate((np.zeros_like(t), np.ones_like(t), t, t))
        for kind in ('constant', 'smooth', 'layered'):
            case = cases.manufactured_case(kind)
            np.testing.assert_allclose(case.exact(x, y), 0.0, atol=2e-14)

    def test_layer_interface_solution_and_normal_flux_are_continuous(self):
        y = np.array([0.19, 0.51, 0.83])
        epsilon = 1e-10
        for barrier in (False, True):
            case = cases.manufactured_case('layered', contrast=1000.0, barrier=barrier)
            for interface in case.interfaces:
                left = np.full_like(y, interface - epsilon)
                right = np.full_like(y, interface + epsilon)
                flux_left = case.coefficient(left, y) * case.exact_grad(left, y)[0]
                flux_right = case.coefficient(right, y) * case.exact_grad(right, y)[0]
                np.testing.assert_allclose(flux_left, flux_right, rtol=0.0, atol=3e-10)
                # This O(epsilon/k_min) tolerance accounts for the real one-sided slope.
                np.testing.assert_allclose(case.exact(left, y), case.exact(right, y),
                                           rtol=0.0, atol=3e-7)

    def test_smooth_source_matches_independent_analytic_divergence(self):
        case = cases.manufactured_case('smooth')
        x = np.array([0.13, 0.27, 0.53, 0.79])
        y = np.array([0.21, 0.63, 0.37, 0.81])
        u = np.sin(np.pi*x) * np.sin(np.pi*y)
        ux = np.pi * np.cos(np.pi*x) * np.sin(np.pi*y)
        uy = np.pi * np.sin(np.pi*x) * np.cos(np.pi*y)
        coefficient = np.exp(0.5*x + 0.25*y)
        expected = coefficient * (2*np.pi**2*u - 0.5*ux - 0.25*uy)
        np.testing.assert_allclose(case.source(x, y), expected, rtol=2e-13)
        np.testing.assert_allclose(case.exact_grad(x, y), np.array([ux, uy]), rtol=2e-13)
        np.testing.assert_allclose(case.coefficient_grad(x, y),
                                   coefficient[None, :] * np.array([[0.5], [0.25]]), rtol=2e-13)

    def test_goals_are_normalized_and_reported_qoi_matches_independent_integral(self):
        case = cases.manufactured_case('layered', contrast=100.0, barrier=True)
        centers = ((0.25, 0.25), (0.75, 0.25), (0.5, 0.75))
        for index, (weight, (cx, cy)) in enumerate(zip(case.goals, centers)):
            xbreaks = sorted(set([cx - .22, cx + .22] +
                                [v for v in case.interfaces if cx-.22 < v < cx+.22]))
            ybreaks = [cy-.22, cy+.22]
            mass = _quadrature_integral(weight, xbreaks, ybreaks)
            self.assertAlmostEqual(mass, 1.0, delta=3e-12)
            integral = _quadrature_integral(lambda x, y: weight(x, y)*case.exact(x, y),
                                            xbreaks, ybreaks)
            self.assertAlmostEqual(integral, float(case.exact_qoi[index]), delta=3e-12)
        self.assertEqual(len(case.goals), 3)

    def test_case_rejects_invalid_parameters(self):
        for kwargs in ({'kind': 'unknown'}, {'contrast': 0.0}, {'contrast': -10.0},
                       {'contrast': np.nan}, {'contrast': np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cases.manufactured_case(**kwargs)


class AssemblyAndInterpolationTests(unittest.TestCase):
    def test_p1_stiffness_and_load_match_independent_affine_triangle_assembly(self):
        mesh = fem.make_mesh(4)
        problem = fem.assemble(mesh, _constant(3.5), (_constant(2.0),),
                               coefficient_alignment='element_aligned')
        expected_matrix, expected_load = _independent_p1(mesh, coefficient=3.5, source=2.0)
        np.testing.assert_allclose(problem.basis.doflocs, mesh.p, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(problem.A.toarray(), expected_matrix, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(problem.loads[:, 0], expected_load, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(problem.A @ np.ones(problem.basis.N), 0.0, atol=2e-13)

    def test_p2_reproduces_quadratic_patch_with_nonzero_boundary_data(self):
        mesh = fem.make_mesh(4)
        problem = fem.assemble(mesh, _constant(1.0), (_constant(-6.0),), degree=2,
                               coefficient_alignment='element_aligned')
        x, y = problem.basis.doflocs
        exact = x*x + x*y + 2*y*y + 3*x - 2*y + .4
        residual = problem.A @ exact - problem.loads[:, 0]
        np.testing.assert_allclose(residual[problem.free], 0.0, atol=8e-13)
        rhs = problem.loads[problem.free, 0] - problem.A[problem.free][:, problem.boundary] @ exact[problem.boundary]
        interior = spsolve(problem.A[problem.free][:, problem.free], rhs)
        np.testing.assert_allclose(interior, exact[problem.free], rtol=2e-13, atol=2e-13)

    def test_boundary_partition_and_homogeneous_solve(self):
        mesh = fem.make_mesh(4)
        problem = fem.assemble(mesh, _constant(1.0), (_constant(1.0),), degree=2,
                               coefficient_alignment='element_aligned')
        x, y = problem.basis.doflocs
        expected = np.flatnonzero((x == 0) | (x == 1) | (y == 0) | (y == 1))
        np.testing.assert_array_equal(np.sort(problem.boundary), expected)
        np.testing.assert_array_equal(np.sort(np.concatenate((problem.free, problem.boundary))),
                                      np.arange(problem.basis.N))
        solution, diagnostics = fem.solve(problem)
        np.testing.assert_array_equal(solution[problem.boundary], 0.0)
        np.testing.assert_allclose(problem.A[problem.free] @ solution,
                                   problem.loads[problem.free], atol=2e-13)
        self.assertTrue(diagnostics['converged'])

    def test_p1_p2_prolongation_preserves_affine_fields_energy_and_loads(self):
        case = cases.manufactured_case('layered', contrast=1000.0)
        mesh = fem.make_mesh(4)
        rhs = (case.source,) + case.goals
        p1 = fem.assemble(mesh, case.coefficient, rhs, degree=1,
                          coefficient_alignment='element_aligned')
        p2 = fem.assemble(mesh, case.coefficient, rhs, degree=2,
                          coefficient_alignment='element_aligned')
        prolongation = fem.prolongation_p1_p2(p1, p2)
        x1, y1 = p1.basis.doflocs
        x2, y2 = p2.basis.doflocs
        np.testing.assert_allclose(prolongation @ (0.7 + 2*x1 - 3*y1), 0.7 + 2*x2 - 3*y2,
                                   atol=3e-15)
        np.testing.assert_allclose((prolongation.T @ p2.A @ prolongation).toarray(),
                                   p1.A.toarray(), rtol=2e-13, atol=3e-11)
        np.testing.assert_allclose(prolongation.T @ p2.loads, p1.loads, rtol=2e-13, atol=2e-13)
        vector = np.sin(np.arange(p1.basis.N))
        self.assertAlmostEqual(float(vector @ (p1.A @ vector)),
                               float((prolongation @ vector) @ (p2.A @ (prolongation @ vector))),
                               delta=2e-10)

    def test_local_refinement_preserves_layer_alignment_and_area(self):
        mesh = fem.make_mesh(4)
        interfaces = cases.manufactured_case('layered').interfaces
        initial_elements = mesh.t.shape[1]
        for _ in range(3):
            centers = mesh.p[:, mesh.t].mean(axis=1)
            chosen = np.flatnonzero((centers[0] < .55) & (centers[1] < .55))
            mesh = mesh.refined(chosen)
            coordinates = mesh.p[:, mesh.t]
            low, high = coordinates[0].min(axis=0), coordinates[0].max(axis=0)
            for interface in interfaces:
                self.assertFalse(np.any((low < interface - 1e-14) & (high > interface + 1e-14)))
            a = coordinates[:, 1] - coordinates[:, 0]
            b = coordinates[:, 2] - coordinates[:, 0]
            area = abs(a[0]*b[1] - a[1]*b[0]) / 2.0
            self.assertGreater(float(area.min()), 0.0)
            self.assertAlmostEqual(float(area.sum()), 1.0, delta=2e-14)
        self.assertGreater(mesh.t.shape[1], initial_elements)

    def test_invalid_coefficient_and_degree_are_rejected(self):
        mesh = fem.make_mesh(4)
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fem.assemble(mesh, _constant(value), (_constant(1.0),))
        with self.assertRaises(ValueError):
            fem.assemble(mesh, _constant(1.0), (_constant(1.0),), degree=3)
        with self.assertRaises(ValueError):
            fem.assemble(mesh, _constant(1.0), (_constant(np.nan),))
        for invalid in (0, 1, -4, 4.5, True, np.nan):
            with self.subTest(n=invalid), self.assertRaises(ValueError):
                fem.make_mesh(invalid)

    def test_error_norms_are_exact_for_a_representable_polynomial(self):
        problem = fem.assemble(fem.make_mesh(4), _constant(7.0), (_constant(-6.0),),
                               degree=2, coefficient_alignment='element_aligned')
        exact = lambda x, y: x*x + x*y + 2*y*y + 3*x - 2*y + .4
        exact_grad = lambda x, y: np.array([2*x + y + 3, x + 4*y - 2])
        norms = fem.error_norms(problem, exact(*problem.basis.doflocs), exact, exact_grad)
        self.assertLess(norms['l2'], 3e-14)
        self.assertLess(norms['h1_seminorm'], 2e-13)
        self.assertLess(norms['energy'], 6e-13)


class DWRIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = cases.manufactured_case('layered', contrast=100.0)
        mesh = fem.make_mesh(4)
        rhs = (case.source,) + case.goals
        cls.p1 = fem.assemble(mesh, case.coefficient, rhs, degree=1,
                              coefficient_alignment='element_aligned')
        cls.p2 = fem.assemble(mesh, case.coefficient, rhs, degree=2,
                              coefficient_alignment='element_aligned')
        cls.solution1, _ = fem.solve(cls.p1)
        cls.solution2, _ = fem.solve(cls.p2)
        cls.prolongation = fem.prolongation_p1_p2(cls.p1, cls.p2)

    def _estimate(self, primal=None, adjoint=None, p2_primal=True):
        return fem.dwr(self.p1, self.p2,
                       self.solution1[:, 0] if primal is None else primal,
                       self.solution2[:, 1:] if adjoint is None else adjoint,
                       goal_indices=(1, 2, 3),
                       p2_primal=self.solution2[:, 0] if p2_primal else None)

    def test_same_space_adjoint_gives_zero_galerkin_residual(self):
        adjoint = self.prolongation @ self.solution1[:, 1:]
        result = self._estimate(adjoint=adjoint, p2_primal=False)
        np.testing.assert_allclose(result['global_estimates'], 0.0, atol=3e-13)
        np.testing.assert_allclose(result['algebraic_terms'], 0.0, atol=3e-13)
        residual = self.p1.loads[:, 0] - self.p1.A @ self.solution1[:, 0]
        # Boundary reactions are present: only admissible zero-trace adjoints are tested.
        self.assertGreater(float(np.linalg.norm(residual[self.p1.boundary])), 1e-5)
        np.testing.assert_allclose(residual @ self.solution1[:, 1:], 0.0, atol=3e-13)

    def test_enriched_dwr_equals_discrete_qoi_difference_and_local_sum(self):
        result = self._estimate()
        difference = self.p2.loads[:, 1:].T @ (self.solution2[:, 0] -
                                               self.prolongation @ self.solution1[:, 0])
        self.assertGreater(float(np.linalg.norm(difference)), 1e-6)
        np.testing.assert_allclose(result['total_estimates'], difference, rtol=2e-10, atol=3e-13)
        np.testing.assert_allclose(result['qoi_differences'], difference, rtol=2e-10, atol=3e-13)
        np.testing.assert_allclose(result['identity_errors'], 0.0, atol=3e-13)
        np.testing.assert_allclose(result['algebraic_terms'], 0.0, atol=3e-13)
        local = np.asarray(result['signed_element_contributions'])
        self.assertEqual(local.shape, (self.p1.mesh.t.shape[1], 3))
        np.testing.assert_allclose(local.sum(axis=0), result['global_estimates'], rtol=2e-10, atol=3e-13)

    def test_inexact_primal_needs_the_algebraic_term(self):
        perturbed = self.solution1[:, 0].copy()
        perturbed[self.p1.free] += 0.01 * (1.0 + np.sin(np.arange(len(self.p1.free))))
        result = self._estimate(primal=perturbed)
        true_difference = self.p2.loads[:, 1:].T @ (self.solution2[:, 0] - self.prolongation @ perturbed)
        np.testing.assert_allclose(result['total_estimates'], true_difference, rtol=2e-10, atol=3e-13)
        np.testing.assert_allclose(np.asarray(result['global_estimates']) + result['algebraic_terms'],
                                   true_difference, rtol=2e-10, atol=3e-13)
        self.assertGreater(float(np.linalg.norm(result['algebraic_terms'])), 1e-4)
        self.assertGreater(float(np.linalg.norm(np.asarray(result['global_estimates']) - true_difference)), 1e-4)

    def test_dwr_does_not_read_optional_primal_reference_to_form_estimate(self):
        without = self._estimate(p2_primal=False)
        with_reference = self._estimate()
        for key in ('global_estimates', 'algebraic_terms', 'total_estimates', 'signed_element_contributions'):
            np.testing.assert_allclose(without[key], with_reference[key], rtol=0.0, atol=0.0)

    def test_body_face_decomposition_matches_weak_residual_on_local_mesh(self):
        for kind in ('smooth', 'layered'):
            with self.subTest(kind=kind):
                case = cases.manufactured_case(kind, contrast=1000.0)
                mesh = fem.make_mesh(4)
                centers = mesh.p[:, mesh.t].mean(axis=1)
                mesh = mesh.refined(np.flatnonzero((centers[0] < .55) & (centers[1] < .55)))
                kwargs = dict(coefficient_alignment=case.metadata['coefficient_alignment'],
                              coefficient_grad=case.coefficient_grad)
                p1 = fem.assemble(mesh, case.coefficient, (case.source,) + case.goals, degree=1, **kwargs)
                p2 = fem.assemble(mesh, case.coefficient, (case.source,) + case.goals, degree=2, **kwargs)
                u1, _ = fem.solve(p1)
                u2, _ = fem.solve(p2)
                result = fem.dwr(p1, p2, u1[:, 0], u2[:, 1:],
                                 goal_indices=(1, 2, 3), p2_primal=u2[:, 0])
                self.assertTrue(result['strong']['available'])
                np.testing.assert_allclose(result['strong']['weak_global_difference'], 0.0,
                                           atol=3e-12)
                np.testing.assert_allclose(result['strong']['signed_element_contributions'].sum(axis=0),
                                           result['global_estimates'], rtol=2e-10, atol=3e-12)
                np.testing.assert_allclose(result['identity_errors'], 0.0, atol=3e-12)

    def test_dwr_rejects_incompatible_nested_problems_and_boundary_data(self):
        mismatched = fem.assemble(self.p1.mesh, self.p2.coefficient, self.p2.rhs_functions,
                                  degree=2, intorder=8, coefficient_alignment='element_aligned')
        with self.assertRaises(ValueError):
            fem.dwr(self.p1, mismatched, self.solution1[:, 0], self.solution2[:, 1:],
                    goal_indices=(1, 2, 3))
        changed = fem.assemble(self.p1.mesh, lambda x, y: 2*self.p2.coefficient(x, y),
                               self.p2.rhs_functions, degree=2, coefficient_alignment='element_aligned')
        with self.assertRaises(ValueError):
            fem.dwr(self.p1, changed, self.solution1[:, 0], self.solution2[:, 1:],
                    goal_indices=(1, 2, 3))
        bad_primal = self.solution1[:, 0].copy()
        bad_primal[self.p1.boundary[0]] = .01
        with self.assertRaises(ValueError):
            self._estimate(primal=bad_primal)
        bad_dual = self.solution2[:, 1:].copy()
        bad_dual[self.p2.boundary[0], 0] = .01
        with self.assertRaises(ValueError):
            self._estimate(adjoint=bad_dual)

    def test_dwr_rejects_wrong_shapes_and_unverified_material(self):
        with self.assertRaises(ValueError):
            self._estimate(primal=np.zeros(self.p1.basis.N + 1))
        with self.assertRaises(ValueError):
            self._estimate(adjoint=np.zeros((self.p2.basis.N - 1, 3)))
        unknown = fem.assemble(self.p1.mesh, _constant(1.0), (_constant(1.0), _constant(1.0)), degree=1)
        unknown2 = fem.assemble(self.p1.mesh, _constant(1.0), (_constant(1.0), _constant(1.0)), degree=2)
        with self.assertRaises(ValueError):
            fem.dwr(unknown, unknown2, np.zeros(unknown.basis.N), np.zeros((unknown2.basis.N, 1)),
                    goal_indices=(1,))


class AuditDriverTests(unittest.TestCase):
    def test_dorfler_marking_uses_absolute_goal_mass_and_deterministic_ties(self):
        local = np.array([[2., -2., 0.], [0., 0., 4.], [1., 0., 0.], [0., 0., 1.]])
        selected, record = experiment.mark(local, .5)
        # Equal first two scores retain element index ordering; signed cancellation
        # within an element must not erase its marking mass.
        np.testing.assert_array_equal(selected, [0, 1])
        self.assertAlmostEqual(record['captured_fraction'], .8)
        self.assertTrue(record['true_solution_not_used'])
        selected_sign_reversed, _ = experiment.mark(-local, .5)
        np.testing.assert_array_equal(selected_sign_reversed, selected)
        with self.assertRaises(ValueError):
            experiment.mark(np.zeros((4, 3)), .5)
        with self.assertRaises(ValueError):
            experiment.mark(np.array([[np.nan, 1., 1.]]), .5)

    def test_analytic_truth_changes_offline_errors_but_not_marking(self):
        case = cases.manufactured_case('constant')
        fake_truth = replace(case, exact=_constant(0.0),
                             exact_grad=lambda x, y: np.zeros((2,) + np.broadcast(x, y).shape),
                             exact_qoi=np.zeros(3))
        mesh = fem.make_mesh(4)
        real_row, real_local, _ = experiment.context(case, mesh, 'unit_test', 0)
        fake_row, fake_local, _ = experiment.context(fake_truth, mesh, 'unit_test', 0)
        np.testing.assert_array_equal(real_local, fake_local)
        real_selected, _ = experiment.mark(real_local, experiment.CONFIG['marking_theta'])
        fake_selected, _ = experiment.mark(fake_local, experiment.CONFIG['marking_theta'])
        np.testing.assert_array_equal(real_selected, fake_selected)
        self.assertNotEqual(real_row['p1']['q_ref'], fake_row['p1']['q_ref'])
        self.assertNotEqual(real_row['p1']['error_norms']['l2'], fake_row['p1']['error_norms']['l2'])


if __name__ == '__main__':
    unittest.main()
