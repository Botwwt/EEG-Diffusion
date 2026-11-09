"""Meaningful fixed baseline/source checks before any new CUDA statistics."""
import argparse,json,time
from pathlib import Path
import brainhive_functional_mean_contract_20261103_v1 as G
def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();start=time.monotonic();s=G.setup(a.protocol)
 import numpy as np,torch
 from brainhive_bounded_component_v1 import BoundedConditionAdapter
 from brainhive_functional_mean_calibration_CPU_20261103_v1 import calibration_fields,output_bias
 torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 d09=G.read(G.checked(s['future_image_refs']['D09_completion']));inputs=G.read(G.checked(d09['inputs']))
 assert d09['status']=='COMPLETE_D09_STAGE400' and inputs['components']['ordinary']['checkpoint']==s['ordinary']
 panel=G.read(G.checked(inputs['panel']));assert len(panel['Dcheck'])==400 and len(panel['T_rows'])==13230
 bank=np.load(G.checked(d09['check_conditions']),allow_pickle=False);assert bank.shape==(400,3,1024) and bank.dtype==np.float32
 saved=torch.load(G.checked(s['ordinary']),weights_only=True,map_location='cpu');model=BoundedConditionAdapter().double().eval().requires_grad_(False);model.load_state_dict({k:v.double() for k,v in saved['model'].items()},strict=True)
 native=torch.load(G.checked(inputs['native_checkpoint']),weights_only=True,map_location='cpu')['model'];g=native['block.layers.0.norm.weight'].double().expand(400,-1);b=native['block.layers.0.norm.bias'].double().expand(400,-1)
 c=torch.from_numpy(bank[:,0].copy()).double()
 with torch.inference_mode():pred=model(c,g,b)[0];disabled=model(c,g,b,enabled=False)[0]
 assert torch.equal(c,disabled);expected=torch.from_numpy(bank[:,1].copy()).double()
 maximum=float((pred-expected).abs().max());assert torch.allclose(pred,expected,atol=1e-5,rtol=5e-6)
 rng=np.random.default_rng(1729);visual=rng.normal(size=(4,2048));eeg=rng.normal(size=(4,2048));fields=calibration_fields(visual,eeg)
 assert abs(fields['proof']['shared_token_norm']-fields['proof']['matched_token_norm'])<1e-10
 assert abs(fields['proof']['mean_centered_orthogonality'])<1e-10
 assert np.array_equal(calibration_fields(eeg,eeg)['shared_delta'],np.zeros(2048))
 assert not torch.cuda.is_initialized();resources=G.resources(s,start=True)
 done=dict(status='PASS_OLD_CPU_SOURCE_BASELINE400_AND_MEAN_ALGEBRA_PREFLIGHT_NO_CUDA',protocol=G.ref(a.protocol),ordinary=s['ordinary'],all400_original_O_CPU64_vs_cache_max_error=maximum,zero_disabled_exact=True,all_source_pins_checked=True,
   full_T_fit_input_contract_bound=s['input_pins']['component_contract'],T_mean_fit_not_started=True,new_GPU_jobs=0,new_images=0,resources=resources,seconds=time.monotonic()-start)
 G.write(Path(a.protocol).parent/'actual_CPU_preflight.json',done);print(json.dumps(done))
if __name__=='__main__':main()
