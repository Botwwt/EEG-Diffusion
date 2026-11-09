"""Input-only scope, fixed full-count chunk transactions and original writer tests."""
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import brainhive_exposed_native_latent_contract_CPU_20261101_v1 as K


def run(s):
    K.validate_scope(s)
    tests=[]
    def reject(name,key,value):
        changed=copy.deepcopy(s);changed[key]=value
        try:K.validate_scope(changed)
        except AssertionError:tests.append(name)
        else:raise AssertionError('Expected refusal '+name)
    reject('truthful_exposure_cannot_be_false','official200_previously_exposed',False)
    reject('independent_H_cannot_be_true','independent_H',True)
    reject('no_test_payload_permission','official_test_image_payloads_allowed',True)
    reject('not_component_fit_admission','component_fit_or_benchmark_admitted',True)
    reject('no_NEW_GPU','host','new')
    reject('no_new_training_seed','new_training_seed',True)
    reject('full16540_shape_required','shape',[13230,4,64,64])
    reject('no_posterior_sample_substitution','mode','sample')
    reject('fixed2GiB_not_lower_estimate','maximum_output_bytes',1<<30)
    reject('fixed32_and33_chunk_segments','segment_chunk_ranges',[[0,33],[33,65]])
    reject('old_source_gates_not_modified','legacy_official_entrypoints_or_gates_modified',True)
    import build_BrainHIVE_official_train_latents_v1 as B
    from queue_HIVE_exposed_full16540_latents_CPU_20261101_v1 import exact_process
    if os.name=='posix':
        fields=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()
        tick=int(fields[19]);assert exact_process(os.getpid(),tick) and not exact_process(os.getpid(),tick+1)
        tests.append('actual_CPU_process_starttick_identity_not_just_PID')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)/'bank'
        registration=dict(shape=[16540,1,2,2],array_rows=list(range(16540)),dtype='float32',
            batch_size=4,chunk_size=256,actual_official_full_train_scope=False,
            source='CPU_TINY_DIMENSION_FULL_COUNT_FIXTURE_NOT_VAE')
        visited=[]
        def encode(ids):
            visited.extend(ids)
            return np.stack([np.full((1,2,2),i+.125,dtype=np.float32) for i in ids])
        class Boundary(Exception):pass
        def boundary(stage,number):
            if stage=='after_chunk_commit' and number==31:raise Boundary()
        try:B._build_bank(root,registration,encode,lambda:None,fault=boundary)
        except Boundary:pass
        else:raise AssertionError('Fixed first boundary not executed')
        assert visited==list(range(8192)) and len(list((root/'chunks').glob('chunk-*.json')))==32
        pins={p.name:B.sha(p) for p in (root/'chunks').glob('chunk-*.json')}
        assert not (root/'completion.json').exists()
        visited.clear()
        final=B._build_bank(root,registration,encode,lambda:None)
        assert visited==list(range(8192,16540)) and final['committed_chunks_reused']==32
        assert final['VAE_encode_calls_this_attempt']==2087 and len(final['committed_chunks'])==65
        assert all(B.sha(root/'chunks'/name)==digest for name,digest in pins.items())
        values=np.load(root/'native_VAE_train_modes.npy',allow_pickle=False)
        assert np.array_equal(values[:,0,0,0],np.arange(16540,dtype=np.float32)+.125)
        tests.append('real_transaction_full16540_count_two_segments_no_prefix_reencoding')
    # Retain the existing broad transaction/corruption/budget tests unchanged.
    checker=importlib.import_module('check_BrainHIVE_official_train_latents_CPU_v1')
    log=io.StringIO()
    outcome=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(checker.Checks))
    assert outcome.wasSuccessful(),log.getvalue()
    tests.append('original12_native_bank_transaction_and_binding_checks')
    return dict(status='PASS_EXPOSED_LATENT_SCOPE_AND_REAL_CPU_TRANSACTIONS_ONLY',
                checks=tests,original_checks=outcome.testsRun,
                original_check_log=log.getvalue(), actual_VAE_models_loaded=0,
                actual_new_latent_bank_built=False, new_GPU_jobs=0, new_parameter_updates=0,
                CPU_fixture_full_count=16540, CPU_fixture_latent_dimensions=[1,2,2])
