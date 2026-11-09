"""Necessary deterministic CPU checks for the repeat-response math prototype.

No model, EEG file, image, diffusion package, or CUDA operation is used.  The
Monte Carlo item is a deliberately constructed failure example, not an EEG or
image result.  Its independent local Generator leaves global RNG untouched.
"""

import argparse
import hashlib
import itertools
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

import brainhive_repeat_response_bias_20261028_v1 as M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "audit/eeg_repeat_response_20261028_v1/local_math_checks.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.output}")
    torch.set_num_threads(2)
    global_rng_before = torch.random.get_rng_state().clone()
    python_rng_before = random.getstate()
    cuda_before = torch.cuda.is_initialized()
    assert not cuda_before, "CPU check process must start without CUDA initialization"
    gen = torch.Generator(device="cpu").manual_seed(1729)
    dtype = torch.float64
    x = torch.randn(19, 4, 3, generator=gen, dtype=dtype)
    means = M.subset_means(x)
    checks = []

    def passed(name: str, details: dict | None = None) -> None:
        checks.append({"name": name, "status": "PASS", **({"details": details} if details else {})})

    def close(actual, expected, atol=2e-12):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=atol)

    assert means.m3_subsets == ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    assert means.m2_subsets == ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    assert means.m3.shape == (4, 19, 3) and means.m2.shape == (6, 19, 3)
    close(means.m4, x.mean(dim=1))
    for n, subsets, values in ((3, means.m3_subsets, means.m3), (2, means.m2_subsets, means.m2)):
        for k, indices in enumerate(subsets):
            assert len(indices) == n
            close(values[k], x[:, indices].sum(dim=1) / n)
    passed("all_subset_shapes_and_lexicographic_order")

    close(means.m3.mean(dim=0), means.m4)
    close(means.m2.mean(dim=0), means.m4)
    passed("finite_population_subset_means_reconstruct_full_mean")

    weight = torch.tensor([[0.5, -0.25], [1.25, 0.75], [-1.5, 0.25]], dtype=dtype)
    bias = torch.tensor([0.125, -0.5], dtype=dtype)
    lin4, lin3 = means.m4 @ weight + bias, means.m3 @ weight + bias
    close(M.jackknife_response(lin4, lin3), lin4)
    passed("affine_response_counterterm_zero_to_float64_tolerance")

    h = torch.tensor([[2.0, 0.25, -0.5], [0.25, -1.0, 0.125], [-0.5, 0.125, 0.75]], dtype=dtype)

    def quadratic(z):
        return torch.einsum("...i,ij,...j->...", z, h, z) + z @ torch.tensor([0.25, -0.5, 0.75], dtype=dtype) + 0.125

    q4, q3, q2 = quadratic(means.m4), quadratic(means.m3), quadratic(means.m2)
    residual = x - means.m4[:, None]
    sample_cov = torch.einsum("bri,brj->bij", residual, residual) / 3
    trace_hs = torch.einsum("ij,bji->b", h, sample_cov)
    close(q3.mean(dim=0) - q4, trace_hs / 12)
    close(M.jackknife_response(q4, q3), q4 - trace_hs / 4)
    passed("quadratic_counterterm_exact_sample_covariance_contraction")

    # Exact finite distribution: all 16 equally likely iid Rademacher repeat sets.
    support = torch.tensor(list(itertools.product((-0.75, 0.75), repeat=4)), dtype=dtype)
    support = support[:, :, None] + 1.25
    exact = M.subset_means(support)
    exact_base, exact_loo = exact.m4.square(), exact.m3.square()
    corrected = M.jackknife_response(exact_base, exact_loo)
    close(exact_base.mean(), torch.tensor(1.25**2 + 0.75**2 / 4, dtype=dtype))
    close(corrected.mean(), torch.tensor(1.25**2, dtype=dtype))
    passed("quadratic_leading_expectation_bias_cancels_on_exact_iid_distribution")

    close(q2.mean(dim=0) - q4, 3 * (q3.mean(dim=0) - q4))
    passed("quadratic_R2_R3_R4_identity_is_finite_population_algebra_not_iid_evidence")

    expected = M.jackknife_response(q4, q3)
    for perm in itertools.permutations(range(4)):
        perm_means = M.subset_means(x[:, perm])
        close(M.jackknife_response(quadratic(perm_means.m4), quadratic(perm_means.m3)), expected)
    passed("all_24_repeat_permutations_invariant_within_float64_tolerance")

    identical = torch.tensor([[0.0, -0.0, 0.125, -7.25], [1e-150, -1e-150, 9.0, -0.0]], dtype=dtype)
    identical_trials = identical[:, None].expand(-1, 4, -1).clone()
    identical_means = M.subset_means(identical_trials)
    for value in (identical_means.m4, identical_means.m3, identical_means.m2):
        target = identical if value.ndim == 2 else identical[None].expand(value.shape[0], -1, -1)
        assert torch.equal(value.view(torch.int64), target.contiguous().view(torch.int64))
    same_response = identical[None].expand(4, -1, -1)
    assert M.jackknife_response(identical, same_response) is identical
    assert M.opposite_response(identical, same_response) is identical
    passed("identical_repeats_and_responses_exact_including_signed_zero")

    opposite = M.opposite_response(q4, q3)
    close(expected - q4, -(opposite - q4))
    close((expected + opposite) / 2, q4)
    passed("opposite_control_equal_magnitude_and_antipodal")

    zero = torch.zeros(2, 3, dtype=dtype)
    assert M.jackknife_response(zero, zero[None].expand(4, -1, -1)) is zero
    assert M.response_summary(zero, zero[None].expand(4, -1, -1))["correction_to_base_rms"] is None
    passed("zero_amplitude_is_valid_and_zero_response_does_not_divide_by_zero")

    rejected = 0

    def must_reject(fn):
        nonlocal rejected
        try:
            fn()
        except (TypeError, ValueError):
            rejected += 1
        else:
            raise AssertionError("Invalid input was accepted")

    for bad in (torch.zeros(2, 0, 3), torch.zeros(0, 4, 3), torch.zeros(2, 4, 0),
                torch.zeros(2, 3, 3), torch.zeros(4), torch.zeros(2, 4, 3, dtype=torch.int64),
                torch.full((2, 4, 3), float("nan")), torch.full((2, 4, 3), float("inf"))):
        must_reject(lambda bad=bad: M.subset_means(bad))
    for bad_base, bad_loo in (
        (torch.zeros(0), torch.zeros(4, 0)),
        (torch.zeros(2), torch.zeros(3, 2)),
        (torch.zeros(2), torch.zeros(4, 2, 1)),
        (torch.zeros(2, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float32)),
        (torch.zeros(2), torch.full((4, 2), float("nan"))),
        (torch.full((2,), float("nan")), torch.zeros(4, 2)),
    ):
        must_reject(lambda b=bad_base, l=bad_loo: M.jackknife_response(b, l))
    must_reject(lambda: M.jackknife_response(q4, q3, direction="invalid"))
    passed("invalid_zero_length_shape_dtype_NaN_Inf_and_direction_rejected", {"rejections": rejected})

    must_reject(lambda: M.jackknife_response(torch.tensor([1e308], dtype=dtype), torch.full((4, 1), -1e308, dtype=dtype)))
    passed("finite_inputs_that_overflow_counterterm_rejected")

    # Counterexample: T(x)=x1^2-x2^2 has zero leading bias under isotropic noise.
    # Jackknife adds estimation variance: exact MSE .25 -> 1/3 for n=4, sigma=1.
    samples = torch.randn(200000, 4, 2, generator=gen, dtype=dtype)
    mc_means = M.subset_means(samples)

    def trace_zero_quadratic(z):
        return z[..., 0].square() - z[..., 1].square()

    mc_base = trace_zero_quadratic(mc_means.m4)
    mc_corrected = M.jackknife_response(mc_base, trace_zero_quadratic(mc_means.m3))
    mse_base, mse_corrected = float(mc_base.square().mean()), float(mc_corrected.square().mean())
    mc_ratio = mse_corrected / mse_base
    assert abs(float(mc_base.mean())) < 0.008 and abs(float(mc_corrected.mean())) < 0.008
    assert abs(mse_base - 0.25) < 0.015 and abs(mse_corrected - 1 / 3) < 0.02
    assert 1.20 < mc_ratio < 1.48
    passed("Monte_Carlo_counterexample_bias_correction_can_increase_MSE", {
        "synthetic_only": True, "samples": 200000, "seed": 1729,
        "model": "iid N(0,I2), T(x)=x1^2-x2^2, target T(0)=0",
        "theoretical_base_MSE": 0.25, "theoretical_corrected_MSE": 1 / 3,
        "empirical_base_mean": float(mc_base.mean()),
        "empirical_corrected_mean": float(mc_corrected.mean()),
        "empirical_base_MSE": mse_base, "empirical_corrected_MSE": mse_corrected,
        "empirical_MSE_ratio": mc_ratio,
        "warning": "This is a mathematical failure example, not measured EEG or model efficacy.",
    })

    assert torch.equal(global_rng_before, torch.random.get_rng_state())
    assert random.getstate() == python_rng_before
    assert not torch.cuda.is_initialized()
    passed("global_torch_and_python_RNG_unchanged_and_CUDA_never_initialized")

    root = Path(__file__).resolve().parent
    files = [Path(__file__).resolve(), Path(M.__file__).resolve()]
    report = {
        "status": "CPU_MATH_ONLY_ALL_CHECKS_PASSED",
        "UTC": datetime.now(timezone.utc).isoformat(),
        "checks_count": len(checks), "checks": checks,
        "source_pins": [{"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        "GPU_jobs": 0, "parameter_updates": 0, "new_images": 0,
        "host_or_EEG_data_loaded": False, "global_RNG_unchanged": True,
        "interpretation": [
            "Classical jackknife mathematics; not a claim of a new statistical algorithm.",
            "Actual physical EEG repeats have not been established as independent or centered.",
            "Fixed-Q local response algebra is not unbiasedness of the full generator trajectory.",
            "The finite-population R2/R3/R4 identity cannot establish the noise assumptions.",
            "No EEG, reconstruction, novelty, or efficacy conclusion follows from these checks.",
            "Zero-length axes are rejected; zero-amplitude EEG/responses remain legitimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks_count": len(checks), "output": str(args.output), "synthetic_counterexample_MSE_ratio": mc_ratio}))


if __name__ == "__main__":
    main()
