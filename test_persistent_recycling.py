"""End-to-end wiring checks for frozen-space use during event solves.

Run each policy once on the same small 20-step synthetic sequence. Spies
wrap the real solvers and inspect actual operator identity, rather than
trusting a method name or a logged operator label. No benchmark artifacts
are written and no reference solutions are passed into the runner.
"""

import contextlib
import hashlib
import inspect
import io
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import persistent_recycling_experiment as experiment


def array_hash(value):
    value = np.asarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


class PersistentRecyclingTests(unittest.TestCase):
    EVENT_STEPS = {0, 3, 6, 9, 10, 12, 15, 18}
    RECYCLED = ("frozen_event_warm4", "frozen_all_balanced4", "frozen_all_warm4")

    @classmethod
    def setUpClass(cls):
        # 75001 is an already excluded integration-smoke seed, not a new
        # development/confirmation geometry used for selecting methods.
        cls.fields, cls.events = experiment.coefficient_sequence(
            32, 20, 75001, experiment.CONTRAST, "mixed")
        cls.results = {}
        real_matrix = experiment.AMGSequencePreconditioner.for_matrix
        real_balanced = experiment.balanced_preconditioner
        real_cached = experiment.solve_pcg_cached
        real_scipy = experiment.solve_pcg
        real_extract = experiment.cached_slow_ritz
        real_orthogonalize = experiment.a_orthonormalize
        with threadpool_limits(limits=1, user_api="blas"):
            cls.systems = [experiment.assemble_system(k)[:2] for k in cls.fields]
            for setting in experiment.specs():
                state = {"base_operators": [], "balanced_operators": {},
                         "solve_calls": [], "extractions": [], "orthogonalizations": []}

                def matrix_spy(instance, A, *args, **kwargs):
                    M, info = real_matrix(instance, A, *args, **kwargs)
                    state["base_operators"].append(M)
                    return M, info

                def balanced_spy(M, U, AU):
                    B = real_balanced(M, U, AU)
                    step = len(state["base_operators"]) - 1
                    if step in state["balanced_operators"]:
                        raise AssertionError("balanced operator built twice in one step")
                    state["balanced_operators"][step] = {"operator": B, "base": M}
                    return B

                def solver_spy(real, kind, *args, **kwargs):
                    bound = inspect.signature(real).bind(*args, **kwargs)
                    bound.apply_defaults()
                    state["solve_calls"].append({
                        "step": len(state["base_operators"]) - 1,
                        "kind": kind,
                        "operator": bound.arguments["M"],
                        "max_vectors": bound.arguments.get("max_vectors", 0),
                    })
                    return real(*args, **kwargs)

                def cached_spy(*args, **kwargs):
                    return solver_spy(real_cached, "cached", *args, **kwargs)

                def scipy_spy(*args, **kwargs):
                    return solver_spy(real_scipy, "scipy", *args, **kwargs)

                def extraction_spy(D, AD, MAD, *args, **kwargs):
                    U, AU, stats = real_extract(D, AD, MAD, *args, **kwargs)
                    state["extractions"].append({
                        "step": len(state["base_operators"]) - 1,
                        "input_hashes": tuple(array_hash(x) for x in (D, AD, MAD)),
                        "output_hashes": tuple(array_hash(x) for x in (U, AU)),
                        "output_rank": U.shape[1],
                        "status": stats["status"],
                    })
                    return U, AU, stats

                def orthogonalization_spy(A, directions, *args, **kwargs):
                    state["orthogonalizations"].append({
                        "step": len(state["base_operators"]) - 1,
                        "input_hash": array_hash(directions),
                    })
                    return real_orthogonalize(A, directions, *args, **kwargs)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(experiment.AMGSequencePreconditioner,
                                                      "for_matrix", autospec=True,
                                                      side_effect=matrix_spy))
                    stack.enter_context(patch.object(experiment, "balanced_preconditioner",
                                                      side_effect=balanced_spy))
                    stack.enter_context(patch.object(experiment, "solve_pcg_cached",
                                                      side_effect=cached_spy))
                    stack.enter_context(patch.object(experiment, "solve_pcg",
                                                      side_effect=scipy_spy))
                    stack.enter_context(patch.object(experiment, "cached_slow_ritz",
                                                      side_effect=extraction_spy))
                    stack.enter_context(patch.object(experiment, "a_orthonormalize",
                                                      side_effect=orthogonalization_spy))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    rows, solutions = experiment.run_sequence(cls.fields, cls.events, setting, 0)
                cls.results[setting["name"]] = dict(rows=rows, solutions=solutions, **state)

    def test_all_five_policies_complete_twenty_accurate_steps(self):
        self.assertEqual(set(self.results), {*self.RECYCLED, "baseline_scipy", "baseline_event_custom"})
        for name, result in self.results.items():
            with self.subTest(method=name):
                self.assertEqual(len(result["rows"]), 20)
                self.assertEqual(len(result["solutions"]), 20)
                self.assertEqual(len(result["base_operators"]), 20)
                self.assertEqual(len(result["solve_calls"]), 20)
                self.assertEqual({r["step"] for r in result["rows"] if r["refresh_event"]},
                                 self.EVENT_STEPS)
                self.assertTrue(all(r["converged"] and r["final_converged"] for r in result["rows"]))
                self.assertFalse(any(r["solver_fallback"] for r in result["rows"]))
                for (A, b), solution in zip(self.systems, result["solutions"]):
                    self.assertLessEqual(float(np.linalg.norm(b - A @ solution) / np.linalg.norm(b)),
                                         experiment.RTOL)

    def test_all_balanced_passes_the_actual_balanced_operator_on_events(self):
        result = self.results["frozen_all_balanced4"]
        self.assertEqual(set(result["balanced_operators"]), set(range(1, 20)))
        for call in result["solve_calls"]:
            step = call["step"]
            with self.subTest(step=step):
                if step == 0:
                    self.assertIs(call["operator"], result["base_operators"][step])
                else:
                    built = result["balanced_operators"][step]
                    self.assertIs(built["base"], result["base_operators"][step])
                    self.assertIs(call["operator"], built["operator"])
                    self.assertIsNot(call["operator"], result["base_operators"][step])
                self.assertEqual(call["kind"], "cached" if step in self.EVENT_STEPS else "scipy")

    def test_event_warm_and_all_warm_use_their_intended_actual_operators(self):
        for name in ("frozen_event_warm4", "frozen_all_warm4"):
            result = self.results[name]
            expected_balanced = set(range(20)) - self.EVENT_STEPS if name == "frozen_event_warm4" else set()
            self.assertEqual(set(result["balanced_operators"]), expected_balanced)
            for call in result["solve_calls"]:
                step = call["step"]
                with self.subTest(method=name, step=step):
                    expected = (result["balanced_operators"][step]["operator"]
                                if step in expected_balanced else result["base_operators"][step])
                    self.assertIs(call["operator"], expected)
                    self.assertEqual(call["kind"], "cached" if step in self.EVENT_STEPS else "scipy")

    def test_recycled_methods_capture_and_extract_only_the_initial_step(self):
        for name in self.RECYCLED:
            result = self.results[name]
            with self.subTest(method=name):
                self.assertEqual(len(result["extractions"]), 1)
                self.assertEqual(result["extractions"][0]["step"], 0)
                self.assertEqual(result["extractions"][0]["status"], "accepted")
                self.assertEqual(result["extractions"][0]["output_rank"], experiment.RANK)
                for call in result["solve_calls"]:
                    self.assertEqual(call["max_vectors"], experiment.LIMIT if call["step"] == 0 else 0)
                self.assertEqual(sum(r["extraction_requested"] for r in result["rows"]), 1)

    def test_same_initial_space_hash_and_no_later_replacement(self):
        reference = self.results[self.RECYCLED[0]]["extractions"][0]
        for name in self.RECYCLED:
            result = self.results[name]
            extraction = result["extractions"][0]
            with self.subTest(method=name):
                self.assertEqual(extraction["input_hashes"], reference["input_hashes"])
                self.assertEqual(extraction["output_hashes"], reference["output_hashes"])
                self.assertEqual([x["step"] for x in result["orthogonalizations"]], list(range(1, 20)))
                self.assertTrue(all(x["input_hash"] == extraction["output_hashes"][0]
                                    for x in result["orthogonalizations"]))

    def test_baselines_never_build_or_extract_recycling_spaces(self):
        for name in ("baseline_scipy", "baseline_event_custom"):
            result = self.results[name]
            with self.subTest(method=name):
                self.assertEqual(result["extractions"], [])
                self.assertEqual(result["orthogonalizations"], [])
                self.assertEqual(result["balanced_operators"], {})
                for call in result["solve_calls"]:
                    step = call["step"]
                    self.assertIs(call["operator"], result["base_operators"][step])
                    self.assertEqual(call["max_vectors"], 0)
                    expected = "cached" if name == "baseline_event_custom" and step in self.EVENT_STEPS else "scipy"
                    self.assertEqual(call["kind"], expected)


if __name__ == "__main__":
    unittest.main()
