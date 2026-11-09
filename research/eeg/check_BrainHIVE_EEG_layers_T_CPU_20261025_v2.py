"""Synthetic checks for the held-pool diagnostic; never opens experiment data."""
from __future__ import annotations

import ast
import copy
import io
import json
from pathlib import Path
import sys
import unittest

import analyze_BrainHIVE_EEG_layers_T_CPU_20261025_v2 as probe


def example_config():
    return {"schema": probe.SCHEMA, "seed": 1729, "ridge_lambdas": [0.1, 1.0],
            "limits": {"threads": 4, "maximum_seconds": 14400,
                       "maximum_extra_RSS_bytes": 16 << 30, "maximum_output_bytes": 4 << 30},
            "thresholds": {"minimum_relative_MSE_reduction": .01,
                           "minimum_absolute_cosine_margin_gain": .005,
                           "minimum_subjects_positive": 8}}


class ScopeChecks(unittest.TestCase):
    def setUp(self):
        self.rows = list(range(80))
        self.concepts = {r: f"c{r//10:02d}" for r in self.rows}
        self.held = self.rows[:40]
        self.base_fit = self.rows[40:]

    def test_split_covers_each_held_row_once_and_excludes_fit_concepts(self):
        a, b = probe.held_probe_split(self.held, self.concepts)
        for x, y in ((a, b), (b, a)):
            probe.validate_scopes(self.rows, self.base_fit, self.held,
                                  self.held, x, y, self.concepts)
        self.assertEqual(set(a) | set(b), set(self.held))
        self.assertEqual(len(a), len(b))
        self.assertEqual((a, b), probe.held_probe_split(self.held, self.concepts))

    def test_rejects_stitched_OOF_outer_split(self):
        a, b = probe.held_probe_split(self.rows, self.concepts)
        with self.assertRaises(AssertionError):
            probe.validate_scopes(self.rows, self.base_fit, self.held,
                                  self.rows, a, b, self.concepts)

    def test_rejects_stimulus_split_inside_same_concept(self):
        with self.assertRaises(AssertionError):
            probe.validate_scopes(self.rows, self.base_fit, self.held,
                                  self.held, self.held[::2], self.held[1::2], self.concepts)

    def test_negative_mapping_bijection_and_concept_exclusion(self):
        p = probe.different_concept_permutation(self.held, self.concepts, 1729, "synthetic")
        self.assertEqual(sorted(p), list(range(len(self.held))))
        self.assertTrue(all(self.concepts[self.held[i]] != self.concepts[self.held[j]]
                            for i, j in enumerate(p)))
        self.assertEqual(p, probe.different_concept_permutation(self.held, self.concepts, 1729, "synthetic"))

    def test_rejects_duplicate_rows(self):
        a, b = probe.held_probe_split(self.held, self.concepts)
        with self.assertRaises(AssertionError):
            probe.validate_scopes(self.rows, self.base_fit, self.held,
                                  self.held, a+[a[0]], b, self.concepts)


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        self.config = example_config()
        self.summary = {v: {k: {"relative_MSE_reduction": .012,
                                  "mean_cosine_margin_gain": .006,
                                  "subjects_joint_positive": 8}
                           for k in ("c_h0_minus_c_c", "c_h0_minus_c_h0_shuffled")}
                        for v in ("0.1", "1.0")}

    def decision(self):
        return probe.decide(self.summary, self.config["thresholds"], ["0.1", "1.0"])

    def test_joint_practical_rule_passes_without_authorizing_training(self):
        out = self.decision()
        self.assertEqual(out["status"], "SUPPORTED")
        self.assertFalse(out["automatic_new_candidate_authorized"])

    def test_every_lambda_and_both_controls_required(self):
        for strength in self.summary:
            for contrast in self.summary[strength]:
                with self.subTest(strength=strength, contrast=contrast):
                    row = self.summary[strength][contrast]
                    old = row["relative_MSE_reduction"]
                    row["relative_MSE_reduction"] = .00999
                    self.assertEqual(self.decision()["status"], "NOT_SUPPORTED")
                    row["relative_MSE_reduction"] = old

    def test_margin_and_subject_thresholds_are_conjunctive(self):
        row = self.summary["0.1"]["c_h0_minus_c_c"]
        row["mean_cosine_margin_gain"] = .00499
        self.assertEqual(self.decision()["status"], "NOT_SUPPORTED")
        row["mean_cosine_margin_gain"] = .006
        row["subjects_joint_positive"] = 7
        self.assertEqual(self.decision()["status"], "NOT_SUPPORTED")

    def test_statistics_branch_cannot_inherit_main_pass(self):
        for value in self.summary.values():
            value["c_h0_minus_c_hpre"] = {"relative_MSE_reduction": .02,
                                              "mean_cosine_margin_gain": .004,
                                              "subjects_joint_positive": 10}
        self.assertEqual(self.decision()["status"], "SUPPORTED")
        branch = probe.decide(self.summary, self.config["thresholds"], ["0.1", "1.0"],
                              ("c_h0_minus_c_hpre",))
        self.assertEqual(branch["status"], "NOT_SUPPORTED")

    def test_config_rejects_flexible_search_or_resource_expansion(self):
        probe.validate_config(self.config)
        for key, bad in (("ridge_lambdas", [.1]), ("seed", 1730)):
            config = copy.deepcopy(self.config)
            config[key] = bad
            with self.assertRaises(AssertionError):
                probe.validate_config(config)
        for key, value in (("threads", 8), ("maximum_seconds", 14401),
                           ("maximum_extra_RSS_bytes", (16 << 30)+1),
                           ("maximum_output_bytes", (4 << 30)+1)):
            config = copy.deepcopy(self.config)
            config["limits"][key] = value
            with self.assertRaises(AssertionError):
                probe.validate_config(config)


class NumericChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(4)
        cls.generator = torch.Generator(device="cpu").manual_seed(1729)
        if torch.cuda.is_initialized():
            raise AssertionError("Synthetic checks must start without CUDA context")

    def rand(self, *shape):
        return self.torch.randn(shape, generator=self.generator, dtype=self.torch.float64)

    def test_standardization_uses_only_fit_rows(self):
        torch = self.torch
        fit = self.rand(40, 5)
        evaluation = self.rand(20, 5)
        x, z, mean, scale = probe.standardize(torch, fit, evaluation, "c")
        x2, z2, mean2, scale2 = probe.standardize(torch, fit, evaluation+1000, "c")
        self.assertTrue(torch.equal(x, x2) and torch.equal(mean, mean2) and torch.equal(scale, scale2))
        self.assertTrue(torch.allclose(z2-z, 1000/scale))

    def test_same_width_duplicate_c_preserves_ridge_prediction(self):
        torch = self.torch
        fit, held, target = self.rand(80, 12), self.rand(30, 12), self.rand(80, 9)
        x, z, _, _ = probe.standardize(torch, fit, held, "c")
        xx, zz, _, _ = probe.standardize(torch, torch.cat((fit, fit), 1),
                                        torch.cat((held, held), 1), "c_c")
        self.assertTrue(torch.allclose(x@x.T, xx@xx.T, atol=1e-12, rtol=1e-12))
        for strength in (.1, 1.):
            a, _ = probe.ridge_predict(torch, x, target, z, strength)
            b, _ = probe.ridge_predict(torch, xx, target, zz, strength)
            self.assertLess(float((a-b).abs().max()), 1e-12)

    def test_constant_target_returns_probe_fit_target_mean(self):
        torch = self.torch
        fit, held = self.rand(40, 5), self.rand(20, 5)
        target = torch.full((40, 7), 2.5, dtype=torch.float64)
        x, z, _, _ = probe.standardize(torch, fit, held, "c")
        prediction, _ = probe.ridge_predict(torch, x, target, z, .1)
        self.assertTrue(torch.equal(prediction, target[:20]))

    def test_known_complement_recovers_held_signal_and_shuffled_control_does_not(self):
        torch = self.torch
        nfit, nheld, width = 256, 256, 6
        c, h0 = self.rand(nfit+nheld, width), self.rand(nfit+nheld, width)
        target = h0 @ self.rand(width, 8)
        taps = {"c": c, "h0": h0, "hpre": c}
        positions_fit = torch.arange(nfit)
        positions_held = torch.arange(nfit, nfit+nheld)
        pfit = torch.randperm(nfit, generator=self.generator)
        pheld = torch.randperm(nheld, generator=self.generator)
        errors = {}
        for method in ("c_c", "c_h0", "c_h0_shuffled"):
            a = probe.raw_feature(torch, taps, method, positions_fit, pfit)
            b = probe.raw_feature(torch, taps, method, positions_held, pheld)
            x, z, _, _ = probe.standardize(torch, a, b, method)
            pred, _ = probe.ridge_predict(torch, x, target[:nfit], z, .1)
            errors[method] = float((pred-target[nfit:]).square().mean())
        self.assertLess(errors["c_h0"], .1*errors["c_c"])
        self.assertLess(errors["c_h0"], .1*errors["c_h0_shuffled"])

    def test_multiplicative_LN_statistics_counterexample_cannot_pass_beyond_branch(self):
        torch = self.torch
        nfit, nheld, width = 512, 512, 8
        u = self.rand(nfit+nheld, width)
        u = u-u.mean(1, keepdim=True)
        u = u/u.std(1, unbiased=False, keepdim=True)
        scale = (.8*self.rand(nfit+nheld, 1)).exp()
        hpre = scale*u
        c = torch.nn.functional.layer_norm(hpre, (width,), eps=1e-6)
        # h0 has NO extra information beyond preLN state: it is exactly hpre.
        taps = {"c": c, "h0": hpre, "hpre": hpre}
        truth = hpre[nfit:]
        fit_positions, held_positions = torch.arange(nfit), torch.arange(nfit, nfit+nheld)
        wrong = truth[torch.randperm(nheld, generator=self.generator)]
        summary = {}
        for strength in (.1, 1.):
            values = {}
            for method in ("c_c", "c_prestats", "c_hpre", "c_h0"):
                fit = probe.raw_feature(torch, taps, method, fit_positions, list(range(nfit)))
                held = probe.raw_feature(torch, taps, method, held_positions, list(range(nheld)))
                x, z, _, _ = probe.standardize(torch, fit, held, method)
                prediction, _ = probe.ridge_predict(torch, x, hpre[:nfit], z, strength)
                values[method] = probe.prediction_metrics(torch, prediction, truth, wrong)
            self.assertTrue(torch.equal(values["c_h0"], values["c_hpre"]))
            # Merely appending mean/logsigma is not the multiplicative inverse.
            self.assertLess(float(values["c_h0"][:, 0].mean()),
                            .99*float(values["c_prestats"][:, 0].mean()))
            a, b = values["c_h0"], values["c_hpre"]
            summary[str(strength)] = {"c_h0_minus_c_hpre": {
                "relative_MSE_reduction": float((b[:, 0].mean()-a[:, 0].mean())/b[:, 0].mean()),
                "mean_cosine_margin_gain": float((a[:, 3]-b[:, 3]).mean()),
                "subjects_joint_positive": 0}}
        branch = probe.decide(summary, example_config()["thresholds"], ["0.1", "1.0"],
                              ("c_h0_minus_c_hpre",))
        self.assertEqual(branch["status"], "NOT_SUPPORTED")

    def test_mean_scale_control_has_equal_readout_width_and_retains_LN_statistics(self):
        torch = self.torch
        hpre = self.rand(24, 12) * self.rand(24, 1).abs().add(.5) + self.rand(24, 1)
        c = torch.nn.functional.layer_norm(hpre, (12,), eps=1e-6)
        taps = {"c": c, "h0": hpre, "hpre": hpre}
        a = probe.raw_feature(torch, taps, "c_cstats", torch.arange(24), list(range(24)))
        b = probe.raw_feature(torch, taps, "c_prestats", torch.arange(24), list(range(24)))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(tuple(a.shape), (24, 14))
        self.assertTrue(torch.equal(a[:, :12], b[:, :12]))
        self.assertGreater(float((a[:, -2:]-b[:, -2:]).abs().mean()), .1)

    def test_taps_match_reviewed_author_classes_on_synthetic_EEG(self):
        torch = self.torch
        # Execute class definitions only: no optional import, prior weights or data.
        root = Path(__file__).resolve().parent/"reproduction_sources/BrainHIVE_f06db69f/main"
        scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
        for file, names in (("layers.py", {"BaseMLP", "ResNet1DLayer", "ResNet1DBlock"}),
                            ("models_brain.py", {"BrainMLP"})):
            parsed = ast.parse((root/file).read_text(encoding="utf-8"))
            selected = [n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in names]
            self.assertEqual({n.name for n in selected}, names)
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(root/file), "exec"), scope)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            model = scope["BrainMLP"](4250, 1024).eval()
        with torch.no_grad():
            model.block.layers[0].norm.weight.copy_(torch.linspace(.7, 1.3, 1024))
            model.block.layers[0].norm.bias.copy_(torch.linspace(-.2, .2, 1024))
        eeg = self.rand(3, 17, 250).float()
        with torch.no_grad():
            h0, hpre, c = probe.native_taps(torch, eeg, model.state_dict())
            expected = model(eeg)
            self.assertTrue(torch.equal(expected, c))
            self.assertEqual(tuple(h0.shape), (3, 1024))
            self.assertEqual(tuple(hpre.shape), (3, 1024))
            reconstructed, identity = probe.reconstruct_hpre(torch, c, hpre,
                model.block.layers[0].norm.weight, model.block.layers[0].norm.bias)
            self.assertLess(float((reconstructed-hpre).abs().max()), 5e-6)
            self.assertTrue(identity["additive_statistics_are_not_full_reconstruction"])
        self.assertFalse(torch.cuda.is_initialized())


def run_checks():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for cls in (ScopeChecks, PolicyChecks, NumericChecks):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {"status": "PASS" if result.wasSuccessful() else "FAIL",
               "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "synthetic_only": True, "real_EEG_loaded": False, "D_V_test_numerical_rows_read": False,
               "GPU_used": False, "new_EEG_encoder_fit": False, "generated_images": 0}
    print(json.dumps(receipt, sort_keys=True))
    if not result.wasSuccessful():
        print(stream.getvalue(), file=sys.stderr)
        raise AssertionError("Synthetic held-pool checks failed")
    return receipt


if __name__ == "__main__":
    run_checks()
