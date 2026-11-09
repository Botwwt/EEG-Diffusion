"""Consume the real full1654 OOF bank; does not admit a component or benchmark.

No model library is imported before the complete20-fold certificate is checked.
Raw native conditions and each source encoder's own LN affine are preserved.
The old development D/V now belong to training; official200 remains exposed.
"""
import gc
import hashlib
import json
from pathlib import Path

SUBJECTS=tuple(f'sub-{i:02d}' for i in range(1,11))
STATUS='COMPLETE_ACTUAL_TRAIN_ONLY_FULL1654_20FOLD165400_OOF_ASSETS'


def read(p):return json.loads(Path(p).read_bytes())


def ref(p):
    p=Path(p)
    assert p.is_file() and not p.is_symlink()
    before=p.stat()
    with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    after=p.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(path=str(p),bytes=before.st_size,sha256=digest)


def checked(pin):
    assert set(pin)=={'path','bytes','sha256'} and ref(pin['path'])==pin
    return Path(pin['path'])


def validate_runtime_owner(runtime,package_file,package_version):
    # This manifest pins accelerate0.34.2, not the diffusers distribution.
    assert 'accelerate/__init__.py' in runtime['files'] and package_version=='0.34.2'
    assert Path(package_file).resolve()==(Path(runtime['path'])/'accelerate/__init__.py').resolve()


def validate_header(certificate,protocol_ref):
    assert certificate['status']==STATUS and certificate['protocol']==protocol_ref
    assert (certificate['subjects'],certificate['concepts'],certificate['images'],certificate['queries'])==(10,1654,16540,165400)
    assert certificate['raw_shape']==[165400,1024] and certificate['training_seed']==1729
    assert certificate['official200_previously_exposed'] is True and certificate['independent_H'] is False
    assert certificate['official_test_payloads_opened'] is False and certificate['final_component_or_benchmark_admitted'] is False
    assert certificate['all20_folds_and_all165400_join_values_verified'] is True
    assert certificate['query_order']=='subject sub-01..sub-10, then exact global array_row0..16539'
    keys=[f'{subject}/official_oof{fold}' for subject in SUBJECTS for fold in (0,1)]
    assert [item['task_id'] for item in certificate['folds']]==keys
    return keys


def expected_affine_ids(tasks,np):
    """Independent assembly map, with concept exclusion and exact global rows."""
    assert len(tasks)==20
    ids=np.full(165400,-1,dtype=np.int16)
    for number,key in enumerate(sorted(tasks)):
        t=tasks[key]
        assert key==t['subject']+'/official_oof'+str(t['held_fold']) and t['subject'] in SUBJECTS
        assert len(t['training_array_rows'])==len(t['exported_OOF_array_rows'])==8270
        fit=set(t['training_array_rows']);held=set(t['exported_OOF_array_rows'])
        assert len(fit)==len(held)==8270 and not fit&held and fit|held==set(range(16540))
        assert len(set(t['training_concepts']))==len(set(t['held_concepts']))==827
        assert not set(t['training_concepts'])&set(t['held_concepts'])
        where=SUBJECTS.index(t['subject'])*16540+np.asarray(t['exported_OOF_array_rows'],dtype=np.int64)
        assert (ids[where]==-1).all();ids[where]=number
    assert (ids>=0).all()
    return ids


def load_verified(certificate_ref,protocol_ref):
    """CPU read-only input materialization, after full lifecycle success only.

    Caller must separately review method, calibration, resources and execution.
    This function never creates that review and never opens official test data.
    """
    certificate=read(checked(certificate_ref));keys=validate_header(certificate,protocol_ref)
    protocol=read(checked(protocol_ref));root=Path(protocol['output_root'])
    assert Path(certificate_ref['path'])==root/'input_bank_completion.json'
    assert protocol['purpose']=='native_full1654_train_only_OOF_assets'
    assert protocol['official_test_payloads_allowed'] is False and protocol['old_Dfit_Dcheck_V_become_train'] is True
    # The source-bound asset contract checks real train image/concept identities.
    import sys
    sys.path.insert(0,str(Path(protocol_ref['path']).parent))
    import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G
    for source in protocol['source_pins']:G.checked(source)
    G.assert_input_aliases(protocol)
    tasks,images=G.load_scope(protocol)
    sys.path.insert(0,protocol['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    # The caller establishes the pinned process runtime before resource checks.
    # Reinitializing it would violate setup_runtime's cold-import invariant.
    # Re-attest its files and actual loaded package here instead of calling twice.
    runtime=C.bound(protocol['runtime_manifest']);folder=Path(runtime['path'])
    files={p.relative_to(folder).as_posix():C.sha(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    assert files==runtime['files'] and 'diffusers' in sys.modules and 'accelerate' in sys.modules
    import diffusers,accelerate
    validate_runtime_owner(runtime,accelerate.__file__,accelerate.__version__)
    from diffusers.utils import is_accelerate_available
    assert accelerate.__version__=='0.34.2' and is_accelerate_available()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    raw=np.load(checked(certificate['raw_conditions']),allow_pickle=False,mmap_mode='r')
    aid=np.load(checked(certificate['affine_id']),allow_pickle=False)
    weight=np.load(checked(certificate['affine_weight']),allow_pickle=False)
    bias=np.load(checked(certificate['affine_bias']),allow_pickle=False)
    assert raw.shape==(165400,1024) and raw.dtype==np.float32
    assert aid.shape==(165400,) and aid.dtype==np.int16 and np.array_equal(aid,expected_affine_ids(tasks,np))
    assert weight.shape==bias.shape==(20,1024) and weight.dtype==bias.dtype==np.float32
    assert np.isfinite(weight).all() and np.isfinite(bias).all() and np.min(np.abs(weight))>1e-8
    identity=read(checked(certificate['image_identity']))
    assert identity==[dict(array_row=i,stimulus_id=images[i]['stimulus_id'],concept_id=images[i]['concept_id']) for i in range(16540)]
    join_cells=0
    for number,item in enumerate(certificate['folds']):
        key=keys[number];audit=read(checked(item['audit']))
        assert audit['status']=='PASS_INDEPENDENT_FULL25_EPOCH_AND8270_HELD_NATIVE_OOF_AUDIT' and audit['protocol']==protocol_ref and audit['task_id']==key
        assert audit['all25_epoch_Adam_counters_moments_and_shuffle_checked'] and audit['all8270_manual_CPU_FP64_forwards_checked']
        assert audit['held_concepts_excluded_from_fit'] and audit['official_test_payloads_opened'] is False
        assert item['model']==audit['final_model'] and item['conditions']==audit['raw_conditions'] and item['rows']==audit['export_array_rows']
        rows=np.load(checked(item['rows']),allow_pickle=False)
        assert rows.tolist()==tasks[key]['exported_OOF_array_rows']
        source=np.load(checked(item['conditions']),allow_pickle=False,mmap_mode='r')
        assert source.shape==(8270,1024) and source.dtype==np.float32
        for lo in range(0,8270,256):
            part=rows[lo:lo+256];where=SUBJECTS.index(tasks[key]['subject'])*16540+part
            values=np.asarray(raw[where]);assert np.isfinite(values).all() and np.array_equal(values,source[lo:lo+len(part)])
            join_cells+=values.size
        model=torch.load(checked(item['model']),weights_only=True,map_location='cpu',mmap=True)
        assert (model['seed'],model['epochs'],model['updates'])==(1729,25,225)
        assert np.array_equal(weight[number],model['model']['block.layers.0.norm.weight'].numpy())
        assert np.array_equal(bias[number],model['model']['block.layers.0.norm.bias'].numpy())
        source._mmap.close();del source,model;gc.collect()
    assert join_cells==165400*1024
    assert certificate['target_source']==protocol['fused_target_for_future_inputs']
    target=torch.load(checked(certificate['target_source']),weights_only=True,map_location='cpu',mmap=True)
    assert target.shape==(16540,1024) and target.dtype==torch.float32 and torch.isfinite(target).all()
    assert not torch.cuda.is_initialized()
    return dict(condition=raw,affine_ids=aid,weight=weight,bias=bias,target=target,subjects=list(SUBJECTS),array_rows=list(range(16540)),
        certificate=certificate_ref,protocol=protocol_ref,join_cells_verified=join_cells,raw_conditions_L2_normalized=False,
        own_source_affines_verified=True,old_Dfit_Dcheck_V_are_training=True,official200_previously_exposed=True,
        independent_H=False,component_or_benchmark_admission=False)


def cpu_batch(inputs,indices):
    """Native raw c, source gamma/beta and stimulus F in exactly one row order."""
    import numpy as np
    import torch
    indices=np.asarray(indices)
    assert indices.ndim==1 and np.issubdtype(indices.dtype,np.integer) and len(indices)>0
    assert int(indices.min())>=0 and int(indices.max())<165400
    ids=inputs['affine_ids'][indices].astype(np.int64)
    pos=indices%16540
    return dict(condition=torch.from_numpy(np.asarray(inputs['condition'][indices]).copy()),
        weight=torch.from_numpy(inputs['weight'][ids].copy()),bias=torch.from_numpy(inputs['bias'][ids].copy()),
        target=inputs['target'][torch.from_numpy(pos.astype(np.int64))],
        subject_index=indices//16540,array_rows=pos,affine_ids=ids)
