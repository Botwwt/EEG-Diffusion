"""Discriminative functional alignment at the unchanged native IP interface.

Only detached, deterministic Q samples are retained during the native forward.
The auxiliary is built afterwards, outside UNet checkpoint regions. Native
processors, their return values, and the four-step inference path are unchanged.
"""
import contextlib
import random

import torch
import torch.nn.functional as F


def spread_positions(length, maximum=16, device=None):
    assert isinstance(length, int) and length > 0
    assert isinstance(maximum, int) and 1 <= maximum <= 16
    count = min(length, maximum)
    positions = torch.linspace(0, length - 1, count, device=device).round().long()
    assert len(positions.unique()) == count
    return positions


def concept_derangement(array_rows, concept_by_row, seed=1729):
    """Return input POSITIONS, never array rows, for one fixed T-only bijection."""
    assert len(array_rows) == len(set(array_rows)) and len(array_rows) > 1
    groups = {}
    for position, row in enumerate(array_rows):
        assert row in concept_by_row
        groups.setdefault(str(concept_by_row[row]), []).append(position)
    assert len(groups) > 1
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    ordered = []
    for key in keys:
        group = groups[key][:]
        rng.shuffle(group)
        ordered.extend(group)
    shift = max(map(len, groups.values()))
    assert shift * 2 <= len(array_rows), 'A concept derangement is impossible'
    result = [-1] * len(array_rows)
    for rank, position in enumerate(ordered):
        result[position] = ordered[(rank + shift) % len(ordered)]
    assert sorted(result) == list(range(len(array_rows)))
    assert all(concept_by_row[array_rows[i]] != concept_by_row[array_rows[j]]
               for i, j in enumerate(result))
    return torch.tensor(result, dtype=torch.long)


def ip_response(query, tokens, processor, heads):
    """Exactly the native IP SDP branch, before flattening/output projection."""
    assert query.ndim == 4 and query.shape[1] == heads
    batch, _, _, head_dim = query.shape
    assert tokens.ndim == 3 and tokens.shape[0] == batch
    key = processor.to_k_ip(tokens)
    value = processor.to_v_ip(tokens)
    assert key.shape == value.shape and key.shape[-1] == heads * head_dim
    key = key.view(batch, -1, heads, head_dim).transpose(1, 2)
    value = value.view(batch, -1, heads, head_dim).transpose(1, 2)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                         dropout_p=0.0, is_causal=False)


def response_rank_loss(student, positive, negative, epsilon=1e-6):
    """A detached per-query teacher scale; smaller means positive is closer."""
    assert student.shape == positive.shape == negative.shape
    assert student.ndim == 4 and epsilon == 1e-6
    positive = positive.detach().float()
    negative = negative.detach().float()
    dimensions = tuple(range(1, student.ndim))
    separation = (positive - negative).square().mean(dimensions).detach()
    d_positive = (student.float() - positive).square().mean(dimensions)
    d_negative = (student.float() - negative).square().mean(dimensions)
    margin = (d_positive - d_negative) / (separation + epsilon)
    loss = F.softplus(margin).mean()
    assert torch.isfinite(loss)
    return loss, dict(d_positive=d_positive.detach().mean(),
                      d_negative=d_negative.detach().mean(),
                      teacher_separation=separation.mean(),
                      near_zero_teacher_fraction=(separation <= epsilon).float().mean(),
                      normalized_margin=margin.detach().mean())


class NativeIPResponseCapture:
    """Observe native Q once; close capture before checkpoint recomputation.

    Installation adds only to_q observation hooks. No processor is replaced.
    Inference can simply omit this object. Its enabled=False context observes
    nothing and cannot change the original generator's result.
    """
    def __init__(self, unet, *, maximum_query_positions=16, expected_layers=70):
        assert maximum_query_positions == 16
        self.maximum = maximum_query_positions
        self.active = False
        self.queries = {}
        self.handles = []
        self.layers = {}
        for name, processor in unet.attn_processors.items():
            if type(processor).__name__ != 'IPAttnProcessor2_0':
                continue
            assert name.endswith('.processor')
            attention = unet.get_submodule(name[:-len('.processor')])
            assert attention.processor is processor
            assert processor.scale == 1.0
            assert all(not p.requires_grad for p in processor.parameters())
            self.layers[name] = (attention, processor)
            self.handles.append(attention.to_q.register_forward_hook(self._hook(name, attention)))
        assert len(self.layers) == expected_layers > 0

    def _hook(self, name, attention):
        def observe(module, inputs, output):
            if not self.active:
                return None
            assert name not in self.queries, 'Duplicate native Q in one capture'
            assert output.ndim == 3
            with torch.no_grad():
                batch, length, width = output.shape
                assert width % attention.heads == 0
                query = output.detach().view(batch, length, attention.heads,
                                             width // attention.heads).transpose(1, 2)
                if attention.norm_q is not None:
                    query = attention.norm_q(query)
                positions = spread_positions(length, self.maximum, query.device)
                # index_select owns only the sampled storage, unlike a view.
                query = query.index_select(2, positions).detach()
                assert not query.requires_grad
                self.queries[name] = query
            return None  # Returning None preserves the native Linear output.
        return observe

    @contextlib.contextmanager
    def capture(self, enabled=True):
        assert not self.active and not self.queries
        self.active = bool(enabled)
        try:
            yield self
        except BaseException:
            self.queries.clear()
            raise
        finally:
            self.active = False
        if enabled:
            assert set(self.queries) == set(self.layers), 'Native IP layer coverage changed'

    def auxiliary(self, student_tokens, positive_tokens, negative_tokens):
        assert not self.active and set(self.queries) == set(self.layers)
        assert student_tokens.requires_grad
        assert student_tokens.shape == positive_tokens.shape == negative_tokens.shape
        losses = []
        diagnostics = {}
        for name, (attention, processor) in self.layers.items():
            query = self.queries[name]
            assert query.shape[0] == student_tokens.shape[0]
            student = ip_response(query, student_tokens, processor, attention.heads)
            with torch.no_grad():
                positive = ip_response(query, positive_tokens.detach(), processor, attention.heads)
                negative = ip_response(query, negative_tokens.detach(), processor, attention.heads)
            loss, values = response_rank_loss(student, positive, negative)
            losses.append(loss)
            for key, value in values.items():
                diagnostics.setdefault(key, []).append(value)
        self.queries.clear()  # Backward holds only tensors required by its graph.
        result = torch.stack(losses).mean()
        return result, {key: torch.stack(values).mean() for key, values in diagnostics.items()}

    def close(self):
        assert not self.active
        self.queries.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
