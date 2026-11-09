"""Frozen T features + all native IP K/V, synthetic Q: interface audit only."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import brainhive_paired_study_contract_20261025_v1 as C
import brainhive_posterior_response_math_20261027_v1 as M


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();started=time.monotonic();root=Path(__file__).resolve().parent
    C.setup_runtime(C.read(root/'configs/paired_effect_20261025_v1/study_review_v3.json'))
    from brainhive_native_models_v1 import prior_models,WORK
    import brainhive_gain_calibration_20261027_v1 as U
    torch.set_num_threads(2);torch.manual_seed(1729);assert not torch.cuda.is_initialized()
    split=U.read(root/'configs/split_manifest.json');contract=U.read(WORK/'data/component_input_contract_v1.json')
    target_ref=dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])
    target=torch.load(U.checked(target_ref),weights_only=True,map_location='cpu',mmap=True)
    ids=split['image_rows']['T'][:3];f=target[ids].float();assert f.shape==(3,1024)
    prior=prior_models('cpu',include_attention=True);adapter=prior['fusion_encoder'].adapter;projector=prior['proj']
    layers=prior['attn_adapter'].attn_processors;assert len(layers)==70
    w=torch.tensor([[.2,.3,.5]],dtype=torch.float32)
    with torch.no_grad():
        tokens=projector(adapter(f));mean_tokens=projector(adapter((w[0,:,None]*f).sum(0,keepdim=True)))
        assert tokens.shape==(3,4,2048);events=[]
        for name,layer in layers.items():
            dim=layer.to_k_ip.out_features;assert dim%64==0;heads=dim//64
            # Head width64 and shapes are checked; queries do NOT come from a
            # UNet or sampling trajectory, and this is not a teacher diagnosis.
            q=torch.randn(1,heads,8,64)
            k=layer.to_k_ip(tokens).reshape(3,4,heads,64).permute(0,2,1,3)[None]
            v=layer.to_v_ip(tokens).reshape(3,4,heads,64).permute(0,2,1,3)[None]
            mixed=M.marginal_attention(q,k,v,w)
            single=M.marginal_attention(q,k[:,:1],v[:,:1],torch.ones(1,1))
            exact=torch.nn.functional.scaled_dot_product_attention(q,k[:,0],v[:,0])
            assert torch.equal(single,exact) and torch.isfinite(mixed).all()
            km=layer.to_k_ip(mean_tokens).reshape(1,4,heads,64).transpose(1,2)
            vm=layer.to_v_ip(mean_tokens).reshape(1,4,heads,64).transpose(1,2)
            at_mean=torch.nn.functional.scaled_dot_product_attention(q,km,vm)
            events.append(dict(processor=name,heads=heads,hidden_dim=dim,
                               singleton_SDPA_exact=True,
                               conditional_response_noncommutation_relative_L2=float(torch.linalg.vector_norm(mixed-at_mean)/torch.linalg.vector_norm(mixed).clamp_min(1e-12))))
    assert not torch.cuda.is_initialized() and len(events)==70
    value=dict(status='ALL70_NATIVE_IP_AND_REAL_T_F_CPU_INTERFACE_PROBE_NOT_EEG_EFFECT',
               source=U.ref(__file__),math_source=U.ref(root/'brainhive_posterior_response_math_20261027_v1.py'),
               array_rows=ids,target_cache=target_ref,mixture_weights=w.tolist(),weights_have_no_EEG_posterior_claim=True,
               Q_is_synthetic=True,Q_from_actual_native_sampling=False,processors=events,
               mean_relative_noncommutation_gap=float(np.mean([x['conditional_response_noncommutation_relative_L2'] for x in events])),
               frozen_native_modules_strictly_loaded=True,CUDA_initialized=False,GPU_jobs=0,new_images=0,new_fit=0,
               native_pixel_parity_or_posterior_calibration_established=False,novelty_established=False,
               seconds=time.monotonic()-started)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='processors'}))


if __name__=='__main__':main()
