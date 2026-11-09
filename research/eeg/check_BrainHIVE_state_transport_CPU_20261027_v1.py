"""D18 analytical ODE oracles, real native zero-flow identity and fullT coupling."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_state_transport_20261027_v1 as V
R=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    result=V.self_test();assert result['CUDA_initialized'] is False
    previous=U.read(R/'configs/crossbrain_supervision_20261027_v1/image_spec.json');d09=U.read(U.checked(previous['D09_completion']))
    inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));cache=torch.load(U.checked(inputs['native_cache']),weights_only=True,map_location='cpu',mmap=True)
    lookup={int(row):i for i,row in enumerate(cache['array_rows'])};raw=cache['condition'][[lookup[q['array_row']] for q in panel['Dcheck']]].clone()
    old=np.load(U.checked(d09['check_conditions']),allow_pickle=False);assert np.array_equal(raw.numpy(),old[:,0])
    selected=raw[[0,10,390,399]];model=V.StateFlow().eval().requires_grad_(False);mean=raw.mean(0);scale=(raw-mean).square().mean().sqrt()
    moved=V.transport(model,selected,mean,scale);assert moved.numpy().tobytes()==selected.numpy().tobytes()
    split=U.read(R/'configs/split_manifest.json');meta={int(x['array_row']):x for x in csv.DictReader((R/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    permutation=V.concept_endpoint_permutation(split['image_rows']['T'],meta);assert len(permutation)==13230
    # Even an analytically invertible contraction can compress distinctions.
    compressed=selected*.01;recovered=compressed/.01
    inverse_relative=float(((recovered-selected).norm(dim=-1)/selected.norm(dim=-1)).max())
    distance_ratio=float(torch.pdist(compressed).median()/torch.pdist(selected).median());assert inverse_relative<1e-6 and abs(distance_ratio-.01)<1e-6
    assert not torch.cuda.is_initialized();result=result|dict(status='D18_TEN_CPU_ODE_COORDINATE_NATIVE_IDENTITY_AND_ENDPOINT_COUPLING_CHECKS_PASSED',checks=10,
        all400_source_cache_vs_old_parent_bitwise=True,four_real_native_zero_field_outputs_bitwise=True,full13230_sameconcept_endpoint_derangement=True,
        compression_counterexample=dict(inverse_relative_error=inverse_relative,pairwise_distance_ratio=distance_ratio,does_not_validate_semantics=True),
        source=U.ref(__file__),new_parameter_updates=0,new_GPU_jobs=0,real_new_images=0)
    U.atomic(a.output,result);print(json.dumps(result))

if __name__=='__main__':main()
