"""Meaningful numerical checks on synthetic CPU inputs; no EEG/test/GPU reads."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from brainhive_brainkan_v1 import BrainKAN, fixed_electrode_time_groups


def reviewed_native_mlp():
    """Execute only reviewed, hash-pinned native MLP bodies; no optional runtime."""
    source = Path(__file__).resolve().parent / 'reproduction_sources/BrainHIVE_f06db69f'
    receipt = json.loads((source / '_fetch_receipt.json').read_text(encoding='utf-8'))
    expected = {item['path']: item['sha256'] for item in receipt['artifacts']}
    scope = dict(torch=torch, nn=torch.nn)
    for filename, names in [
        ('main/layers.py', ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock']),
        ('main/models_brain.py', ['BrainMLP']),
    ]:
        path = source / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[filename]
        nodes = [node for node in ast.parse(path.read_text(encoding='utf-8')).body
                 if isinstance(node, ast.ClassDef) and node.name in names]
        assert [node.name for node in nodes] == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    model = BrainKAN()
    assert sum(p.numel() for p in model.parameters()) == 6456512
    native = reviewed_native_mlp()
    assert sum(p.numel() for p in native.parameters()) == 6452224
    assert model.norm.eps == native.block.layers[0].norm.eps == 1e-6
    u = torch.linspace(-1, 1, 1001, dtype=torch.float64)
    basis = model.cubic_basis(u, model.knots.double())
    assert basis.shape == (1001, 8) and torch.isfinite(basis).all() and basis.min() >= -1e-14
    unity_error = float((basis.sum(-1) - 1).abs().max())
    assert unity_error < 1e-12
    assert basis[0, 0] == basis[-1, -1] == 1 and basis[0, 1:].count_nonzero() == 0
    probe = torch.tensor([-.823, -.317, .123, .823], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: model.cubic_basis(v, model.knots.double()), (probe,))
    fit = torch.randn(12, 17, 250)
    fit[:, 0, 0] = 3  # Verify a constant feature's safe scale.
    provenance = dict(split='T', rows=12, array_rows=list(range(12)),
                      source_sha256='a' * 64, split_manifest_sha256='b' * 64,
                      synthetic_engineering_only=True)
    for invalid in [dict(provenance, split='V'),
                    dict(provenance, array_rows=[0] * 12)]:
        try:
            model.fit_training_statistics(fit, invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid statistic-fit provenance was accepted')
        assert not bool(model.statistics_fitted)
    receipt = model.fit_training_statistics(fit, provenance)
    assert receipt['constant_or_small_scale_coordinates'] >= 1 and model.input_scale[0] == 1
    x = torch.randn(3, 17, 250)
    model.train()
    output = model(x)
    assert output.shape == (3, 1024) and torch.isfinite(output).all()
    # A nonconstant target avoids the zero-gradient sum of LayerNorm coordinates.
    loss = (output * torch.randn_like(output)).mean()
    loss.backward()
    grads = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()}
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert grads['spline_coefficients'] > 0 and grads['spline_out.weight'] > 0 and grads['proj_in.weight'] > 0
    model.eval()
    with torch.inference_mode():
        contributions = model.attribute_groups(x, fixed_electrode_time_groups())
        actual = model.pre_norm(x) - model.pre_norm(model.input_median.reshape(1, 17, 250))
        reconstruction_error = float((contributions.sum(1) - actual).abs().max())
        assert reconstruction_error < 5e-5
        reference = model(x)
        memory = io.BytesIO()
        torch.save(model.state_dict(), memory)
        memory.seek(0)
        restored = BrainKAN().eval()
        restored.load_state_dict(torch.load(memory, weights_only=True), strict=True)
        serialization_error = float((restored(x) - reference).abs().max())
        assert serialization_error == 0
        extreme = torch.full((2, 17, 250), 1e6)
        extreme[1].neg_()
        assert torch.isfinite(restored(extreme)).all()
    result = dict(status='CPU_NUMERICAL_CHECKS_PASSED', parameters=6456512,
                  original_BrainMLP_parameters=6452224, parameter_difference=4288,
                  native_source_hash_and_parameter_count_checked=True,
                  native_and_KAN_LayerNorm_eps=1e-6,
                  basis_partition_unity_max_error=unity_error, basis_double_gradcheck=True,
                  nonzero_finite_parameter_gradient_norms=grads,
                  exact_pre_LayerNorm_attribution_max_error=reconstruction_error,
                  strict_state_dict_replay_max_error=serialization_error,
                  T_only_statistic_contract_tested=True, synthetic_inputs_only=True,
                  actual_EEG_or_pretrained_weight_files_loaded=False,
                  GPU_work=False, actual_EEG_generation_or_gain_claimed=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
