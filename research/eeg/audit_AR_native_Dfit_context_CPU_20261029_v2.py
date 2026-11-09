"""Independent complete-output audit; never imports the producer's math."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,csv,datetime,hashlib,json,time,traceback
from pathlib import Path
import numpy as np

M=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
ROOT=M/'AR_Dfit200_context_diagnostic_20261029_v1/native_fixed_final_result_v2'
OUT=M/'AR_Dfit200_context_diagnostic_20261029_v1/independent_full_CPU_audit_v2'
CONDITIONS=['own_EEG_own_GT_prefix','same_concept_donor_EEG_own_GT_prefix','zero_EEG_own_GT_prefix','own_EEG_same_concept_donor_prefix','same_concept_donor_EEG_same_concept_donor_prefix']
WIDTHS=np.square([1,2,3,4,5,6,8,10,13,16])
def ref(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def check(x):
    p=Path(x['path']);assert ref(p)==x;return p
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def run():
    started=time.monotonic();assert not OUT.exists();OUT.mkdir()
    completed=json.loads((ROOT/'completion.json').read_bytes())
    assert completed['status']=='COMPLETE_FIXED_NATIVE_FINAL_DFIT200_ALL_FIVE_CONTEXT_CONDITIONS_AND10SCALES'
    assert completed['queries']==200 and completed['concepts']==20 and completed['scales']==10 and completed['conditions']==CONDITIONS
    assert completed['new_optimizer_updates']==completed['new_generated_images']==0 and completed['frozen_EEG_VAR_VQ_unchanged'] is True
    assert completed['original_three_arm_training_unchanged'] is True and completed['primary_Dcheck_image_gate_changed'] is False
    specpath=check(completed['spec']);assert ref(specpath)['sha256']=='2fedba1537dcfa0cf9702946cc35d52447868f6036664ba04b6e209a66070980'
    spec=json.loads(specpath.read_bytes())
    for pin in spec['source_pins']:check(pin)
    binding=json.loads(check(completed['input_completion']).read_bytes())
    for pin in binding['files'].values():check(pin)
    data=json.loads(Path(binding['files']['queries.json']['path']).read_bytes());rows=data['rows'];donors=data['same_concept_donor']
    concepts=sorted(set(r['concept_id'] for r in rows));assert len(rows)==200 and len(concepts)==20 and len(donors)==200
    assert len(set((r['array_row'],r['stimulus_id']) for r in rows))==200
    assert sorted(donors)==list(range(200)) and all(i!=j and rows[i]['concept_id']==rows[j]['concept_id'] and rows[i]['partition']=='Dfit' for i,j in enumerate(donors))
    groups=[[i for i,r in enumerate(rows) if r['concept_id']==c] for c in concepts];assert all(len(g)==10 for g in groups)
    tokens=np.load(check(completed['all_token_NLL']),allow_pickle=False);scales=np.load(check(completed['all_scale_NLL']),allow_pickle=False)
    assert tokens.shape==(200,5,680) and tokens.dtype==np.float32 and np.isfinite(tokens).all() and tokens.min()>=-1e-6
    assert scales.shape==(200,5,10) and scales.dtype==np.float64 and np.isfinite(scales).all()
    calculated=np.empty_like(scales);offset=0
    for s,width in enumerate(WIDTHS):calculated[:,:,s]=tokens[:,:,offset:offset+width].mean(-1,dtype=np.float64);offset+=int(width)
    assert offset==680 and np.array_equal(calculated,scales)
    prefix_error=max(float(np.max(abs(tokens[:,0,0]-tokens[:,3,0]))),float(np.max(abs(tokens[:,1,0]-tokens[:,4,0]))))
    assert prefix_error==completed['first_scale_prefix_causality_max_NLL_error'] and prefix_error<=2e-5
    original=json.loads(check(completed['summary']).read_bytes());assert original['conditions']==CONDITIONS
    assert np.array_equal(np.array(original['mean_NLL_nats_per_token_by_condition_and_scale']),scales.mean(0))
    contrast_names=['own_minus_donor_EEG_at_GT_prefix','own_minus_zero_EEG_at_GT_prefix','own_minus_donor_EEG_at_donor_prefix','GT_minus_donor_prefix_at_own_EEG']
    contrasts=[scales[:,1]-scales[:,0],scales[:,2]-scales[:,0],scales[:,4]-scales[:,3],scales[:,3]-scales[:,0]]
    draws=np.random.default_rng(1729).integers(0,20,(1000,20));audit_contrasts={}
    for name,x in zip(contrast_names,contrasts):
        concept_means=np.stack([np.mean(x[g],axis=0) for g in groups]);boot=np.mean(concept_means[draws],axis=1);ci=np.percentile(boot,[2.5,97.5],axis=0).T
        mean=np.mean(x,axis=0);expected=original['contrasts'][name]
        assert np.array_equal(mean,np.array(expected['mean_log_likelihood_difference_nats_per_token']))
        assert np.array_equal(ci,np.array(expected['descriptive_concept_bootstrap95_per_scale']))
        weighted=x@WIDTHS/680;concept_weighted=concept_means@WIDTHS/680;wboot=np.mean(concept_weighted[draws],axis=1)
        audit_contrasts[name]=dict(mean_nats_per_token_by_scale=mean.tolist(),descriptive95_by_scale=ci.tolist(),
            all680_token_weighted_mean_nats_per_token=float(weighted.mean()),all680_token_weighted_descriptive95=np.percentile(wboot,[2.5,97.5]).tolist())
    fit=json.loads(check(completed['fit_completion']).read_bytes());assert fit['updates']==8270 and fit['epochs']==10 and fit['query_exposures']==132300 and fit['arm']=='native_global'
    assert fit['final_model']==completed['fixed_final'];check(fit['final_model'])
    receipt_path=R/'audit/AR_native_Dfit200_context_diagnostic_20261029_v2_job.json';receipt=json.loads(receipt_path.read_bytes())
    assert receipt['status']=='COMPLETED' and receipt['exit_code']==0 and receipt['termination_reason'] is None
    assert receipt['GPU_UUID']=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d' and receipt['maximum_seconds']==900
    table=OUT/'all200_queries_all5_conditions_all10_scales.csv'
    with table.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['query','stimulus_id','concept_id','array_row','same_concept_donor_query','condition','scale_side','tokens','NLL_nats_per_token'])
        for i,row in enumerate(rows):
            for k,name in enumerate(CONDITIONS):
                for s,side in enumerate((1,2,3,4,5,6,8,10,13,16)):w.writerow([i,row['stimulus_id'],row['concept_id'],row['array_row'],donors[i],name,side,int(WIDTHS[s]),repr(scales[i,k,s])])
    plot=None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        titles=['Own EEG vs same-concept donor; own prefix','Own EEG vs zero EEG; own prefix','Own EEG vs same-concept donor; donor prefix','Own prefix vs same-concept donor prefix; own EEG']
        for ax,name,title in zip(axes.flat,contrast_names,titles):
            x=audit_contrasts[name];y=np.array(x['mean_nats_per_token_by_scale']);ci=np.array(x['descriptive95_by_scale'])
            ax.plot(range(10),y,'o-',linewidth=1.4);ax.fill_between(range(10),ci[:,0],ci[:,1],alpha=.2);ax.axhline(0,color='black',linewidth=.6)
            ax.set_xticks(range(10),[1,2,3,4,5,6,8,10,13,16]);ax.set_xlabel('VQ scale side');ax.set_ylabel('Log-likelihood difference (nats/token)');ax.set_title(title,fontsize=10)
        fig.suptitle('Fixed-final native AR; all Dfit200; descriptive concept bootstrap\nTeacher-forced diagnostic, not free-generation improvement',fontsize=11)
        p=OUT/'all_scales_context_contrasts.png';fig.savefig(p,dpi=160);plt.close(fig);plot=ref(p)
    except ImportError:plot=dict(status='MATPLOTLIB_UNAVAILABLE_METRIC_AUDIT_RETAINED')
    value=dict(status='PASS_INDEPENDENT_ALL200_5CONDITIONS_680TOKENS_10SCALES_AND1000_BOOTSTRAPS_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        producer_completion=ref(ROOT/'completion.json'),wrapper_receipt=ref(receipt_path),queries=200,concepts=20,conditions=5,token_likelihood_values=680000,all10000_scale_values_recomputed_exact=True,
        prefix_first_scale_max_error=prefix_error,contrasts=audit_contrasts,all_query_table=ref(table),plot=plot,seconds=time.monotonic()-started,
        interpretation='Heldout developer Dfit teacher-forced conditional contrasts, no causal or conditional-MI identification, no free-generation/independent-confirmation/SOTA conclusion; positive numbers do not promote models or change the fixed three-arm image gate.',
        training_or_sampling_changed=False,new_GPU_jobs=0)
    write(OUT/'completion.json',value);print(json.dumps(value))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');q=parser.parse_args()
    if q.wait:
        deadline=time.monotonic()+21600
        while not (ROOT/'completion.json').exists():
            assert not (ROOT/'failure.json').exists(),'Actual GPU diagnostic failed; preserve failure, no CPU audit fabricated'
            assert not (M/'AR_native_Dfit_reserved_slot_20261029_v2/failure.json').exists(),'Actual scheduler failed; do not fabricate closure'
            assert time.monotonic()<deadline;time.sleep(20)
        # The producer writes completion just before its wrapper commits exit.
        deadline=time.monotonic()+30
        while True:
            p=R/'audit/AR_native_Dfit200_context_diagnostic_20261029_v2_job.json'
            if p.is_file() and json.loads(p.read_bytes()).get('status')!='RUNNING':break
            assert time.monotonic()<deadline;time.sleep(.2)
    try:run()
    except BaseException as error:
        if OUT.exists() and not (OUT/'failure.json').exists():write(OUT/'failure.json',dict(status='INDEPENDENT_NATIVE_DFIT_CPU_AUDIT_FAILURE_PRESERVED',error=repr(error),traceback=traceback.format_exc()))
        raise
