"""Frozen30-model interface profile only; no target F, fitting or images."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_encoder_block_responses_20261027_v1 as V


def main():
    p=argparse.ArgumentParser()
    for n in ('input-completion','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(2)
    inputs=U.read(a.input_completion);assert inputs['status']=='D15_ALL30_STRICT_MODELS_ORIGINAL_R4_INPUTS_AND_OOF_ROLES_CPU_VERIFIED'
    original=torch.load(U.checked(inputs['CPU_reference']),weights_only=True,map_location='cpu')
    sys.path[:0]=['/root/atm_official_20261016','/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages','/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages']
    from models.atms import ATMS
    rows=[];arrays=[]
    for si,(name,subject) in enumerate(inputs['subjects'].items()):
        eeg=torch.load(U.checked(subject['input']),weights_only=True,map_location='cpu',mmap=True)
        for role,record in subject['models'].items():
            model=ATMS().eval().requires_grad_(False);state=torch.load(U.checked(record['checkpoint']),weights_only=True,map_location='cpu',mmap=True)
            model.load_state_dict(state,strict=True);ref=original[name+'/'+role];seen=[]
            handle=model.enc_eeg[0].register_forward_hook(lambda module,args,output:seen.append(output.detach().clone()))
            try:
                with torch.inference_mode():native=model(eeg[ref['array_rows']],torch.full((2,),si+1,dtype=torch.long))
            finally:handle.remove()
            assert len(seen)==1 and seen[0].shape==(2,36,40) and torch.equal(native,ref['prediction'])
            with torch.inference_mode():
                base,response=V.native_block_responses(model,seen[0])
                constant=seen[0].mean(1,keepdim=True).expand(-1,36,-1).contiguous()
                zero,zero_response=V.native_block_responses(model,constant)
            assert torch.equal(base,native) and zero_response.abs().max()<1e-5
            assert response.norm(dim=-1).min()>1e-6
            rows.append(dict(subject=name,role=role,checkpoint=record['checkpoint'],array_rows=ref['array_rows'],token_shape=[2,36,40],
                native_output_matches_original_CPU_reference_bitwise=True,head_replay_matches_native_bitwise=True,
                full_prediction_norm_mean=float(base.norm(dim=-1).mean()),response_norms_by_query_and_block=response.norm(dim=-1).tolist(),
                constant_token_zero_response_max=float(zero_response.abs().max()),normal_inference_query_target_access=False))
            arrays.append(response.numpy());del model,state
    responses=np.stack(arrays);assert responses.shape==(30,2,4,1024)
    U.save_array(a.output/'reference60_four_block_responses.npy',responses)
    result=dict(status='D16_ALL30_NATIVE_ENCODER_FOUR_BLOCK_RESPONSE_CPU_INTERFACE_PROFILE_COMPLETE',schema=V.SCHEMA,
        source=U.ref(__file__),module=U.ref(V.__file__),inputs=U.ref(a.input_completion),models=30,reference_queries=60,rows=rows,
        responses=U.ref(a.output/'reference60_four_block_responses.npy'),native_output_parity_all30_bitwise=True,
        learned_output_affine_preserved=True,no_hidden_feature_coordinate_assumption_between_models=True,
        perturbation='four9-position blocks in native36x40 internal feature tokens, replaced by same-query36-position feature mean',
        physical_EEG_time_causality_claim=False,neural_recoverability_or_image_gain_established=False,
        no_F_target_or_image_loaded=True,no_fitting=True,new_GPU_jobs=0,new_images=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized() and result['seconds']<60
    U.atomic(a.output/'completion.json',result);print(json.dumps({k:result[k] for k in ('status','models','reference_queries','seconds')}))


if __name__=='__main__':main()
