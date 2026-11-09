"""All2040 saved engineering conditionals, independent scalar normalizers."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def main():
    started=time.monotonic()
    root=Path('audit/VAR_multiscale_native_profile_complete_20261101_v3/result/profile')
    folder=Path('audit/VAR_likelihood_sampler_law_20261101_v1')
    source=folder/'actual_saved_engineering_paths.json'
    result=json.loads(source.read_bytes());assert result['token_conditionals_examined']==2040
    summaries={};checked=0;maxerror=0.
    for file,variants in [('fixed_class_native_and_fresh_trace.npz',('cached','fresh')),('switched_class_fresh_trace.npz',('switched',))]:
        with np.load(root/file,allow_pickle=False) as arrays:
            for variant in variants:
                prefix='' if variant=='switched' else variant+'_'
                total_full=total_top=0.;errors=[]
                for stage in range(10):
                    logits=arrays[f'{prefix}logits_{stage}'][0];codes=arrays[f'{prefix}codes_{stage}'][0]
                    masses=[];full=[];top=[];sizes=[]
                    for x,y in zip(logits,codes):
                        values=x.tolist();maximum=max(values)
                        weights=[math.exp(v-maximum) for v in values]
                        threshold=sorted(values,reverse=True)[599]
                        selected=[w for w,v in zip(weights,values) if v>=threshold]
                        z=math.fsum(weights);zt=math.fsum(selected)
                        assert values[int(y)]>=threshold
                        masses.append(zt/z);sizes.append(len(selected))
                        full.append(values[int(y)]-maximum-math.log(z))
                        top.append(values[int(y)]-maximum-math.log(zt));checked+=1
                    ref=result['paths'][variant]['scales'][stage]
                    observed=dict(retained_mass_min=min(masses),retained_mass_mean=math.fsum(masses)/len(masses),
                        retained_mass_max=max(masses),actual_threshold_support_count_min=min(sizes),actual_threshold_support_count_max=max(sizes),
                        saved_path_untruncated_log_probability=math.fsum(full),saved_path_top_k_log_probability=math.fsum(top),
                        saved_path_log_density_ratio=math.fsum(-math.log(m) for m in masses))
                    for key,value in observed.items():errors.append(abs(value-ref[key]))
                    total_full+=observed['saved_path_untruncated_log_probability'];total_top+=observed['saved_path_top_k_log_probability']
                errors.extend([abs(total_full-result['paths'][variant]['untruncated_log_probability']),
                    abs(total_top-result['paths'][variant]['top_k_log_probability'])])
                assert max(errors)<5e-11
                maxerror=max(maxerror,max(errors));summaries[variant]=dict(tokens=680,max_error=max(errors),log_density_ratio=total_top-total_full)
                assert time.monotonic()-started<120
    assert checked==2040
    value=dict(status='PASS_ALL2040_REAL_SAVED_LOGIT_CONDITIONALS_INDEPENDENT_SCALAR_SUM_AND_SORT',
        tokens=checked,unique_generated_paths=2,variants=summaries,maximum_error=maxerror,seconds=time.monotonic()-started,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_GPU_or_training_or_images=0,
        no_actual_T8_target_support_or_truncated_HMM_proof=True,no_EEG_gain=True)
    (folder/'independent_all2040_scalar_audit.json').open('xb').write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(value))


if __name__=='__main__':main()
