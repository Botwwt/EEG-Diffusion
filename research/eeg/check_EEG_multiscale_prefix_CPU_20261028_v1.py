"""Real upstream reduced-size CPU math/causality tests, not trained results."""
import datetime, hashlib, importlib, json, sys, time, types
from pathlib import Path
import torch

L = Path(__file__).resolve().parent
S = L / 'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'
start = time.monotonic()
inventory = json.loads((S / 'source_inventory.json').read_bytes())
for entry in inventory['files']:
    assert hashlib.sha256((S / entry['path']).read_bytes()).hexdigest() == entry['sha256']
# Import only audited VAR/VQ modules. Do not invoke the author's models factory,
# training entry point, model hub downloads or W&B initialization.
package = types.ModuleType('models')
package.__path__ = [str(S / 'models')]
assert 'models' not in sys.modules
sys.modules['models'] = package
sys.path.insert(0, str(S))
import dist
dist.__dict__['__device'] = 'cpu'
from models.var_eeg import VAREEG
from models.vqvae import VQVAE
from eeg_multiscale_prefix_20261028_v1 import EEGPrefixDecoder, paired_scale_evidence

torch.set_num_threads(2)
torch.manual_seed(1729)
vae = VQVAE(vocab_size=32, z_channels=8, ch=32, test_mode=True, v_patch_nums=(1, 2, 3))
decoder = VAREEG(vae, depth=2, embed_dim=64, num_heads=4, cond_drop_rate=0., patch_nums=(1, 2, 3), flash_if_available=False, fused_if_available=False)
# A deliberate non-negligible random attention gate exercises memory influence.
# This engineering initialization is discarded; no pretrained model is changed.
decoder.init_weights(init_adaln_gamma=.1)
model = EEGPrefixDecoder(decoder).eval()
global_eeg = torch.randn(2, 1024)
patch = torch.randn(2, 5, 1024)
memory = model.centered_memory(global_eeg, patch)
previous = torch.randn(2, decoder.L-decoder.first_l, decoder.Cvae)
checks = {}
checks['centered_memory_preserves_global'] = torch.allclose(memory.mean(1), global_eeg, atol=5e-7, rtol=1e-6)
checks['no_new_learned_parameters'] = sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in decoder.parameters())
with torch.no_grad():
    original = decoder(previous, global_eeg)
    disabled = model(previous, global_eeg)
    checks['prefix_disabled_exact_author_logits'] = torch.equal(original, disabled)
    full = model(previous, global_eeg, eeg_memory=memory)
    pooled = model(previous, global_eeg, eeg_memory=global_eeg[:, None].expand_as(memory))
    checks['EEG_patch_variation_visible_with_same_global'] = (full-pooled).abs().max().item() > 1e-7
    changed = previous.clone()
    # The final-scale input features must not affect the first two scales.
    changed[:, 4:] += torch.randn_like(changed[:, 4:])*10
    altered = model(changed, global_eeg, eeg_memory=memory)
    checks['future_image_cannot_leak_via_EEG_prefix'] = torch.equal(full[:, :5], altered[:, :5])
    a = model.generate(global_eeg, seed=1729, top_k=5)
    b = decoder.autoregressive_infer_cfg(2, global_eeg, g_seed=1729, cfg=4., top_k=5)
    checks['disabled_free_generation_exact_author_pixels'] = torch.equal(a, b)
    c = model.generate(global_eeg, eeg_memory=memory, seed=1729, top_k=5)
    replay = model.generate(global_eeg, eeg_memory=memory, seed=1729, top_k=5)
    checks['EEG_only_free_generation_replay'] = torch.equal(c, replay)
    checks['cache_released_after_generation'] = all(not block.attn.caching and block.attn.cached_k is None for block in decoder.blocks)
    checks['finite_output_and_actual_VQ_decode'] = bool(torch.isfinite(c).all() and c.shape[0:2] == (2, 3))
    # Compare the cached free-running computation against the teacher-forced
    # computation of the SAME generated history; catches memory/cache masking
    # mismatches without using any real target image.
    cached_logits, generated_indices = [], []
    old_logits = decoder.get_logits
    old_embedding = decoder.vae_quant_proxy[0].embedding.forward
    def capture_logits(hidden, condition):
        value = old_logits(hidden, condition)
        cached_logits.append(value[:2].clone())
        return value
    def capture_indices(indices):
        generated_indices.append(indices.clone())
        return old_embedding(indices)
    decoder.get_logits = capture_logits
    decoder.vae_quant_proxy[0].embedding.forward = capture_indices
    try:
        model.generate(global_eeg, eeg_memory=memory, seed=1729, guidance=0., top_k=5)
    finally:
        decoder.get_logits = old_logits
        decoder.vae_quant_proxy[0].embedding.forward = old_embedding
    regenerated_prefix = decoder.vae_quant_proxy[0].idxBl_to_var_input(generated_indices)
    teacher_of_own_history = model(regenerated_prefix, global_eeg, eeg_memory=memory)
    cache_difference = (teacher_of_own_history-torch.cat(cached_logits, 1)).abs().max().item()
    checks['cached_and_full_mask_same_generated_history_logits'] = cache_difference < 2e-6
    try:
        with model._cached_attention():
            raise RuntimeError('injected cache cleanup check')
    except RuntimeError:
        pass
    checks['cache_released_after_exception'] = all(not block.attn.caching and block.attn.cached_k is None for block in decoder.blocks)
    targets = torch.randint(32, (2, decoder.L))
    checks['paired_evidence_same_condition_zero'] = bool((paired_scale_evidence(full, full, targets, (1,4,9)) == 0).all())
model.train()
model.zero_grad(set_to_none=True)
mem_grad = memory.detach().requires_grad_()
loss = model(previous, global_eeg, eeg_memory=mem_grad).square().mean()
loss.backward()
checks['gradient_reaches_patch_memory'] = bool(torch.isfinite(mem_grad.grad).all() and mem_grad.grad.norm() > 0)
checks['gradient_reaches_existing_attention'] = bool(decoder.blocks[0].attn.mat_qkv.weight.grad.norm() > 0)
record = dict(status='PASS' if all(checks.values()) else 'FAIL', UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), checks=checks, seconds=time.monotonic()-start, source_commit=inventory['commit'], reduced_random_weight_model=True, random_test_attention_gate_initialization=.1, pretrained_weights_loaded=False, real_EEG_or_research_images_used=False, GPU_jobs=0, trained_models=0, original_AVDE_parameters=sum(p.numel() for p in decoder.parameters()), synthesis_shape=list(c.shape), logit_max_difference_disabled=(original-disabled).abs().max().item(), generated_history_cached_logit_max_difference=cache_difference)
out = L / 'audit/multiscale_EEG_20261028_v1'
out.mkdir(exist_ok=True)
prior = out / 'CPU_contract_actual_v1.json'
if prior.exists():
    old = prior.read_bytes()
    if json.loads(old)['status'] == 'FAIL':
        (out / ('CPU_contract_failed_' + hashlib.sha256(old).hexdigest()[:12] + '.json')).write_bytes(old)
(out / 'CPU_contract_actual_v1.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8', newline='\n')
print(json.dumps(record))
assert all(checks.values()), checks
