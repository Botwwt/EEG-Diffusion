"""Classical additive RevNet before BrainMLP's irreversible first projection.

Pure PyTorch; no generator/runtime package imports. The author BrainMLP classes
are extracted from their SHA-pinned source AST and loaded with tensor weights.
This is an architectural hypothesis, not a claim that RevNets or EEG flows are
new. The final 4250-to1024 projection remains information losing.
"""
import ast
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


ARMS = ('native_continuation', 'serial_reversible', 'parallel_matched')
CORE_PARAMETERS = 6452224
FRONT_PARAMETERS = 4370024
TOTAL_PARAMETERS = CORE_PARAMETERS + FRONT_PARAMETERS


def _initialized_rng_devices():
    return list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []


def author_brain_class(models_source, layers_source):
    """Compile only four original classes; never execute their import graph."""
    wanted = ((Path(layers_source), ('BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock')),
              (Path(models_source), ('BrainMLP',)))
    scope = dict(torch=torch, nn=nn, F=F, Union=Union, List=List, Tuple=Tuple, Dict=Dict,
                 __name__='brainhive_author_ast_class_scope')
    for path, names in wanted:
        parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        found = {node.name: node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in names}
        if set(found) != set(names):
            raise ValueError('Missing author class: ' + str(path))
        module = ast.Module(body=[found[name] for name in names], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope['BrainMLP']


class CouplingFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2125, 128)
        self.fc2 = nn.Linear(128, 2125)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, value):
        return self.fc2(F.gelu(self.fc1(value)))


class CouplingStage(nn.Module):
    def __init__(self, serial=True):
        super().__init__()
        self.serial = bool(serial)
        self.f = CouplingFunction()
        self.g = CouplingFunction()

    def forward(self, left, right):
        new_left = left + self.f(right)
        new_right = right + self.g(new_left if self.serial else left)
        return new_left, new_right

    def inverse(self, new_left, new_right):
        if not self.serial:
            raise ValueError('Parallel control has no guaranteed analytic inverse')
        right = new_right - self.g(new_left)
        left = new_left - self.f(right)
        return left, right


class ReversibleFront(nn.Module):
    def __init__(self, serial=True):
        super().__init__()
        self.serial = bool(serial)
        self.stages = nn.ModuleList([CouplingStage(serial) for _ in range(4)])

    @staticmethod
    def split(value):
        shape = value.shape
        if not ((value.ndim == 2 and shape[1:] == (4250,)) or
                (value.ndim == 3 and shape[1:] == (17, 250))):
            raise ValueError('Expected [B,4250] or [B,17,250]')
        image = value.reshape(shape[0], 17, 250)
        return image[..., 0::2].reshape(shape[0], 2125), image[..., 1::2].reshape(shape[0], 2125)

    @staticmethod
    def merge(left, right):
        batch = left.shape[0]
        return torch.stack((left.reshape(batch, 17, 125), right.reshape(batch, 17, 125)), dim=-1).reshape(batch, 4250)

    def forward(self, value):
        left, right = self.split(value)
        for stage in self.stages:
            left, right = stage(left, right)
        return self.merge(left, right)

    def inverse(self, value):
        if not self.serial:
            raise ValueError('Parallel control has no guaranteed analytic inverse')
        left, right = self.split(value)
        for stage in reversed(self.stages):
            left, right = stage.inverse(left, right)
        return self.merge(left, right)


class EEGArchitecture(nn.Module):
    def __init__(self, core, arm, front_seed=1729):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.core = core
        # New parameter initialization must not advance the core/dropout stream.
        with torch.random.fork_rng(devices=_initialized_rng_devices()):
            torch.manual_seed(int(front_seed))
            self.front = nn.Identity() if arm == ARMS[0] else ReversibleFront(serial=arm == ARMS[1])
        self.enabled = True

    def forward(self, value):
        return self.core(self.front(value) if self.enabled else value)

    def affine(self):
        norm = self.core.block.layers[0].norm
        return norm.weight, norm.bias


def from_core_state(core_class, core_state, arm, front_seed=1729):
    """Construct on CPU; the loaded core and caller's RNG remain unchanged."""
    with torch.random.fork_rng(devices=_initialized_rng_devices()):
        torch.manual_seed(1729)
        core = core_class(4250, 1024, num_layers=1, dropout=.1)
        core.load_state_dict(core_state, strict=True)
        model = EEGArchitecture(core, arm, front_seed)
    expected = CORE_PARAMETERS if arm == ARMS[0] else TOTAL_PARAMETERS
    if sum(parameter.numel() for parameter in model.parameters()) != expected:
        raise ValueError('Unexpected architecture parameter count')
    return model


def symmetric_infonce(prediction, targets, logit_scale):
    prediction = F.normalize(prediction, dim=-1)
    targets = F.normalize(targets.to(prediction.dtype), dim=-1)
    logits = (prediction @ targets.T) * logit_scale.exp()
    labels = torch.arange(len(prediction), device=prediction.device)
    return .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def from_architecture_state(core_class, architecture_state, arm):
    """Load a saved full encoder without requiring the old checkpoint file."""
    core_state = {key[len('core.'):]: value for key, value in architecture_state.items() if key.startswith('core.')}
    model = from_core_state(core_class, core_state, arm)
    model.load_state_dict(architecture_state, strict=True)
    return model


def optimizer_groups(model, logit_scale):
    norm_ids = {id(parameter) for module in model.modules() if isinstance(module, nn.LayerNorm)
                for parameter in module.parameters()}
    decay = [parameter for name, parameter in model.named_parameters()
             if not name.endswith('bias') and id(parameter) not in norm_ids] + [logit_scale]
    no_decay = [parameter for name, parameter in model.named_parameters()
                if name.endswith('bias') or id(parameter) in norm_ids]
    return [dict(params=decay, weight_decay=.05), dict(params=no_decay, weight_decay=0.)]


def synthetic_checks(models_source, layers_source):
    """Independent identity/inverse/gradient/capacity checks, no real data."""
    core_class = author_brain_class(models_source, layers_source)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        core = core_class(4250, 1024, num_layers=1, dropout=.1).eval()
        state = copy.deepcopy(core.state_dict())
        values = torch.randn(4, 17, 250)
        targets = torch.randn(4, 1024)
        rng_before = torch.get_rng_state().clone()
        models = [from_core_state(core_class, state, arm).eval() for arm in ARMS]
        assert torch.equal(rng_before, torch.get_rng_state())
        expected = core(values)
        for model in models:
            assert torch.equal(model(values), expected)
            model.enabled = False
            assert torch.equal(model(values), expected)
            model.enabled = True
            assert all(torch.equal(model.core.state_dict()[key], value) for key, value in state.items())
        assert all(torch.equal(models[1].front.state_dict()[key], value)
                   for key, value in models[2].front.state_dict().items())
        dropout_outputs, dropout_states = [], []
        for model in models:
            model.train()
            torch.manual_seed(9182)
            dropout_outputs.append(model(values))
            dropout_states.append(torch.get_rng_state().clone())
            model.eval()
        assert all(torch.equal(dropout_outputs[0], value) for value in dropout_outputs[1:])
        assert all(torch.equal(dropout_states[0], value) for value in dropout_states[1:])
        serial = models[1]
        with torch.no_grad():
            for stage in serial.front.stages:
                stage.f.fc2.weight.normal_(0., .001)
                stage.g.fc2.weight.normal_(0., .001)
        transformed = serial.front(values)
        restored = serial.front.inverse(transformed)
        inverse_relative = float((restored - values.reshape(4, -1)).norm() / values.norm())
        assert inverse_relative < 2e-6 and not torch.equal(transformed, values.reshape(4, -1))
        assert not torch.equal(serial(values), expected)
        replay = from_architecture_state(core_class, serial.state_dict(), ARMS[1]).eval()
        assert torch.equal(replay(values), serial(values))
        serial.train()
        loss = symmetric_infonce(serial(values), targets, torch.tensor(2.6592))
        loss.backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in serial.parameters())
        assert sum(float(parameter.grad.square().sum()) for parameter in serial.front.parameters()) > 0
        assert sum(float(parameter.grad.square().sum()) for parameter in serial.core.parameters()) > 0
        return dict(status='PURE_TORCH_REVERSIBLE_EEG_SYNTHETIC_CHECKS_PASSED',
                    native_parameters=CORE_PARAMETERS, reversible_extra_parameters=FRONT_PARAMETERS,
                    reversible_total_parameters=TOTAL_PARAMETERS, parallel_parameters=TOTAL_PARAMETERS,
                    author_class_AST_only=True, exact_zero_head_and_disabled_recovery=True,
                    serial_parallel_initial_state_identical=True, initialization_preserves_CPU_RNG=True,
                    zero_head_paired_core_dropout_outputs_and_RNG_identical=True,
                    trained_nonzero_front_state_reload_exact=True,
                    nonzero_head_inverse_relative_error=inverse_relative, finite_core_and_front_gradients=True,
                    real_data_read=False, GPU_initialized=torch.cuda.is_initialized())


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--author-models-source', type=Path, required=True)
    parser.add_argument('--author-layers-source', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(synthetic_checks(args.author_models_source, args.author_layers_source)))
