"""Post-result exact risk identity; descriptive scales are never a new method."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V

def main():
    p=argparse.ArgumentParser()
    for name in ('completion','CPU-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic()
    audit=U.read(a.CPU_audit);assert audit['all_actual_readout_and_bootstraps_recomputed'] and audit['all_sources_unchanged']
    done=U.read(a.completion);prepared=U.read(U.checked(done['prepared']));spec=U.read(U.checked(done['spec']))
    for ref in spec['source_pins']:U.checked(ref)
    target=np.load(U.checked(prepared['Dfit_projected']),allow_pickle=False)
    predictions=np.load(U.checked(done['predictions']),mmap_mode='r',allow_pickle=False)
    covariances=np.load(U.checked(done['posterior_covariances']),mmap_mode='r',allow_pickle=False)
    scores=np.load(U.checked(done['scores']),allow_pickle=False)
    metadata=U.read(U.checked(prepared['Dfit_metadata']));concepts=[r['concept_id'] for r in metadata['records']]
    changes=[];alignments=[];energies=[];expected=[];rows=[]
    for j,subject in enumerate(spec['subjects']):
        model=torch.load(U.checked(done['models'][subject+'_paired']),weights_only=True,map_location='cpu')
        mean=model['image_mean'].numpy();scale=model['image_std'].numpy();truth=(target-mean)/scale
        energy=[];alignment=[];declared=[]
        for ti,t in enumerate(V.TIMES):
            base=np.array(predictions[j,:,ti,0]);joint=np.array(predictions[j,:,ti,1])
            error=(truth-base)*scale;delta=(joint-base)*scale
            aa=np.square(delta).mean(1);bb=(error*delta).mean(1);change=aa-2*bb
            observed=scores[j,:,ti,1,1]-scores[j,:,ti,0,1]
            assert np.allclose(change,observed,atol=1e-9,rtol=1e-9)
            expected_base=float((np.diag(covariances[j,ti,0])*scale**2).mean())
            expected_joint=float((np.diag(covariances[j,ti,1])*scale**2).mean());assert expected_joint<=expected_base+1e-10
            rows.append(dict(subject=subject,timestep=t,observed_PC_MSE_change=float(change.mean()),update_energy_a=float(aa.mean()),
                             target_error_alignment_b=float(bb.mean()),exact_identity='MSE_joint-MSE_base = a-2b',
                             fraction_queries_with_negative_alignment=float((bb<0).mean()),
                             assumed_model_expected_base_MSE=expected_base,assumed_model_expected_joint_MSE=expected_joint,
                             posthoc_oracle_scale_not_a_method=float(np.clip(bb.mean()/max(aa.mean(),1e-15),0,1))))
            energy.append(aa);alignment.append(bb);declared.append([expected_base,expected_joint])
        energies.append(energy);alignments.append(alignment);expected.append(declared)
    energy=np.asarray(energies);alignment=np.asarray(alignments);assumed=np.asarray(expected)
    pertime=[]
    for ti,t in enumerate(V.TIMES):
        aa=energy[:,ti];bb=alignment[:,ti];delta=aa-2*bb
        pertime.append(dict(timestep=t,a=float(aa.mean()),b=float(bb.mean()),observed_MSE_change=V.bootstrap(delta,concepts,spec['subjects']),
                            posthoc_oracle_scale_not_a_method=float(np.clip(bb.mean()/aa.mean(),0,1)),
                            assumed_expected_MSE_change=float((assumed[:,ti,1]-assumed[:,ti,0]).mean())))
    result=dict(status='D21_EXACT_RISK_DECOMPOSITION_OF_ALL8200_QUERIES_AND4_TIMES_COMPLETE',source=U.ref(__file__),completion=U.ref(a.completion),CPU_audit=U.ref(a.CPU_audit),
                by_subject_time=rows,by_time=pertime,all_queries_and_registered_times_used=True,
                posthoc_scales_used_for_new_model_or_image_selection=False,new_fit=0,new_GPU_jobs=0,new_images=0,
                sole_cause_of_failure_established=False,scope='128D native VAE subspace, not full image utility',seconds=time.monotonic()-start)
    U.atomic(a.output/'risk_decomposition.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    times=[str(x) for x in V.TIMES];xx=np.arange(4)
    axes[0].bar(xx-.18,[p['observed_MSE_change']['mean'] for p in pertime],.36,label='Observed held MSE change')
    axes[0].bar(xx+.18,[p['assumed_expected_MSE_change'] for p in pertime],.36,label='Working-model expected change')
    axes[0].axhline(0,color='black',linewidth=.8);axes[0].set_xticks(xx,times);axes[0].set_xlabel('DDPM timestep');axes[0].set_ylabel('Joint minus c-only native PC MSE');axes[0].legend(fontsize=7)
    axes[1].plot(xx,[p['a'] for p in pertime],'o-',label='Update energy a')
    axes[1].plot(xx,[2*p['b'] for p in pertime],'s-',label='Twice error alignment 2b')
    axes[1].set_xticks(xx,times);axes[1].set_xlabel('DDPM timestep');axes[1].set_ylabel('Risk identity terms');axes[1].legend(fontsize=8)
    fig.suptitle('All10 subjects / 8200 exposed Dfit queries; diagnostic only',fontsize=10)
    image=a.output/'observed_vs_assumed_risk.png';fig.savefig(image,dpi=160);plt.close(fig)
    U.atomic(a.output/'completion.json',dict(status=result['status'],result=U.ref(a.output/'risk_decomposition.json'),plot=U.ref(image),seconds=time.monotonic()-start,new_GPU_jobs=0,new_fit=0,new_images=0))
    print(json.dumps(dict(status=result['status'],by_time=pertime,seconds=time.monotonic()-start)))

if __name__=='__main__':main()
