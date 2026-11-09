"""Generate reviewable compatibility fixes without editing the pinned upstream."""
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'reproduction_sources/BrainHIVE_f06db69f'
DEST = BASE / 'reproduction_sources/BrainHIVE_f06db69f_parent_compat_v1'


def main():
    receipt = json.loads((SOURCE / '_fetch_receipt.json').read_text())
    assert receipt['all_source_blobs_verified']
    for item in receipt['artifacts']:
        assert hashlib.sha256((SOURCE / item['path']).read_bytes()).hexdigest() == item['sha256']
    assert not DEST.exists(), 'Keep previous prepared copies intact'
    shutil.copytree(SOURCE, DEST)
    edits = {
        'main/models_brain.py': [(
            'def forward(self, brain_signals: torch.Tensor, subject_ids: torch.Tensor):',
            'def forward(self, brain_signals: torch.Tensor, subject_ids: torch.Tensor = None):')],
        'train_clip.py': [(
            '    # 4. Load model\n',
            '    # Reproduction fix: seed before any randomly initialized brain model.\n'
            '    set_seed(int(training_args.seed))\n\n    # 4. Load model\n')],
        'scripts/build_reconstruction.sh': [(
            '"PO7","PO3","Oz","PO4","PO8","O1","Oz","O2"',
            '"PO7","PO3","POz","PO4","PO8","O1","Oz","O2"')],
        'build_reconstruction.py': [(
            'from main.models_adapter import IPProjectionModel, IPAttnAdapterModel',
            'from main.models_adapter import IPProjectionModel, IPAttnAdapterModel, FusionEncoderModel'), (
            '    proj.to(accelerator.device, weight_dtype)\n',
            '    # Reproduction fix: retain the trained Fusion Prior residual A.\n'
            '    # Brain alignment targets F(image), so inference must be P(A(brain)).\n'
            '    fusion_prior = FusionEncoderModel.from_pretrained(\n'
            '        args.prior_model_name_or_path, subfolder="fusion_encoder"\n'
            '    )\n'
            '    fusion_adapter = fusion_prior.adapter.eval().requires_grad_(False)\n'
            '    del fusion_prior\n'
            '    fusion_adapter.to(accelerator.device, weight_dtype)\n'
            '    proj.to(accelerator.device, weight_dtype)\n'), (
            '    clip_model.to(accelerator.device, weight_dtype)\n',
            '    clip_model.eval().requires_grad_(False)\n'
            '    clip_model.to(accelerator.device, weight_dtype)\n'), (
            '                ip_hidden_states = proj(brain_embeds)',
            '                ip_hidden_states = proj(fusion_adapter(brain_embeds))'), (
            '    del pipe, clip_model, proj',
            '    del pipe, clip_model, proj, fusion_adapter')],
    }
    patches, changed = [], []
    for filename, pairs in edits.items():
        original = (SOURCE / filename).read_text()
        text = original
        for old, new in pairs:
            assert text.count(old) == 1, (filename, old)
            text = text.replace(old, new)
        if filename.endswith('.py'):
            ast.parse(text)
        (DEST / filename).write_text(text, encoding='utf-8', newline='\n')
        patches.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                          fromfile='a/' + filename, tofile='b/' + filename))
        changed.append({'path': filename,
                        'upstream_sha256': hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest(),
                        'patched_sha256': hashlib.sha256((DEST / filename).read_bytes()).hexdigest()})
    patch = BASE / 'audit/BrainHIVE_parent_compat_20261020_v1.patch'
    patch.write_text(''.join(patches), encoding='utf-8', newline='\n')
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'commit': receipt['commit'],
              'status': 'COMPATIBILITY_COPY_PREPARED_NOT_A_REPRODUCTION_RESULT',
              'upstream_unchanged': True, 'files_changed': changed,
              'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
              'research_component_added': False, 'weights_loaded': False,
              'training_started': False, 'gpu_work_started': False,
              'remaining_preconditions': ['actual EEG and image_id join verification',
                  'channel map derived from actual EEG metadata',
                  'H14 prior payload full SHA/strict load, not metadata only',
                  'isolated compatible runtime; do not install latest dependencies into old environment',
                  'storage reservation, whole-stage profiling and authorized GPU time']}
    (BASE / 'audit/BrainHIVE_parent_compat_preparation_20261020_v1.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'files_changed': len(changed)}))


if __name__ == '__main__':
    main()
