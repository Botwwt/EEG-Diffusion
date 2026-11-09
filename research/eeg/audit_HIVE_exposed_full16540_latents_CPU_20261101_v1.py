"""Full65 chunk/header/SHA coverage and all13230 old-T latent value replay."""
import argparse
import hashlib
from pathlib import Path
import time
import brainhive_exposed_native_latent_contract_CPU_20261101_v1 as K


def run(protocol):
    start = time.monotonic()
    s = K.setup(protocol)
    import numpy as np
    import torch
    import build_BrainHIVE_official_train_latents_v1 as B
    assert not torch.cuda.is_initialized()
    K.input_check(s)
    K.resources(s)
    root = Path(s['output_root'])
    first, second = K.read(root/'segment0/completion.json'), K.read(root/'segment1/completion.json')
    assert first['status'] == 'COMPLETE_PLANNED_FIRST32_CHUNKS_ONLY'
    assert second['status'] == 'COMPLETE_ALL16540_NATIVE_TRAIN_LATENTS'
    assert first['VAE_encode_calls_this_segment'] + second['VAE_encode_calls_this_segment'] == 4135
    registration = K.read(root/'bank/registration.json')
    assert registration['protocol'] == K.ref(protocol)
    assert registration['official200_previously_exposed'] is True and registration['independent_H'] is False
    done = B._verify_bank(root/'bank', registration)
    assert len(done['committed_chunks']) == 65 and done['array_rows'] == list(range(16540))
    assert done['shape'] == list(K.SHAPE) and done['parameters_fitted'] == 0
    bank = np.load(done['bank']['path'], allow_pickle=False, mmap_mode='r')
    assert bank.shape == K.SHAPE and bank.dtype == np.float32
    assert s['old_T']['available'], 'Original valid old-T bank required for this full subset audit'
    old = np.load(K.check(s['old_T']['bank']), allow_pickle=False, mmap_mode='r')
    assert old.shape == (13230,4,64,64) and old.dtype == np.float32
    ids = s['old_T']['array_rows']
    assert len(ids) == len(set(ids)) == 13230 and min(ids) >= 0 and max(ids) < 16540
    maximum = numerator = denominator = 0.0
    bit_exact = True
    for lo in range(0,13230,256):
        selection = ids[lo:lo+256]
        value = np.asarray(bank[selection])
        reference = np.asarray(old[lo:lo+len(selection)])
        assert np.allclose(value, reference, **s['numerical_guard'])
        bit_exact = bit_exact and np.array_equal(value,reference)
        difference = value.astype(np.float64)-reference.astype(np.float64)
        maximum = max(maximum,float(np.abs(difference).max()))
        numerator += float(np.square(difference).sum())
        denominator += float(np.square(reference.astype(np.float64)).sum())
        K.resources(s)
    K.check(s['old_T']['bank'])
    for pin in s['source_pins']:K.check(pin)
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS_ALL16540_LATENT_CHUNKS_AND_ALL13230_OLD_T_VALUE_REPLAY',
                  protocol=K.ref(protocol), bank=K.ref(done['bank']['path']),
                  bank_completion=K.ref(root/'bank/completion.json'),
                  full_native_VAE_images=16540, actual_chunks=65, actual_VAE_encode_calls=4135,
                  train_concepts=1654, old_T_images_checked=13230, old_T_element_values_checked=13230*4*64*64,
                  old_T_maximum_absolute_error=maximum, old_T_relative_L2_error=(numerator/denominator)**.5,
                  old_T_subset_bit_exact=bool(bit_exact), old_T_original_file_SHA_preserved=True,
                  guard=s['numerical_guard'], full16540_independent_VAE_reencoding_performed=False,
                  CUDA_initialized=False, parameters_fitted=0, official_test_image_payloads_opened=0,
                  seconds=time.monotonic()-start, independent_H=False, NEW_contacted=False)
    K.write(root/'independent_CPU_audit.json',result)
    print(__import__('json').dumps(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    run(parser.parse_args().protocol)
