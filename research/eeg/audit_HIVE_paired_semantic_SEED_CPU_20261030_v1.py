"""Independent full2800 records, proxy formulas, low-level and bootstrap audit."""
import argparse,json,os,time
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,IMAGE_ARMS,atomic,checked,read,ref

def categories(row,threshold,np):
    scores=np.asarray(row['scores']);boxes=np.asarray(row['bboxes']);labels=np.asarray(row['labels']);assert np.isfinite(scores).all() and (scores[:-1]>=scores[1:]).all()
    if not len(scores):return set()
    keep=(scores>=threshold)&((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])>=50)
    return set(labels[keep].astype(int).tolist())
def manual_F1(left,right,np):
    recalls=[];precisions=[]
    for t in np.arange(.05,.95,.05):
        a=categories(left,t,np);b=categories(right,t,np);overlap=len(a&b)
        if b:recalls.append(overlap/len(b))
        if a:precisions.append(overlap/len(a))
        if not a and not b:break
    r=sum(recalls)/len(recalls) if recalls else 1.;p=sum(precisions)/len(precisions) if precisions else 1.
    return 2*r*p/(r+p) if r+p else 0.

def main(path):
    from run_HIVE_paired_semantic_SEED_followon_20261030_v1 import load,runtime,promote,load_features
    s,e,up,root,image_root,q,meta=load(path);upstream=promote(s,image_root);os.environ['CUDA_VISIBLE_DEVICES']='';runtime(s)
    import numpy as np,torch
    from PIL import Image
    from skimage.metrics import structural_similarity
    from brainhive_paired_semantic_SEED_statistics_CPU_20261030_v1 import METRICS,next_action
    torch.set_num_threads(2);start=time.monotonic();result=read(root/'secondary_readout.json');assert result['protocol']==ref(path) and result['all_queries']==2800 and result['upstream_audit']==ref(image_root/'independent_images400_CPU_audit.json')
    arrays=np.load(checked(result['arrays']),allow_pickle=False);targets=read(root/'targets.json');t={k:load_features(root,'target',k,torch) for k in ('EffNet','CapSim','ObjectF1')};maximum=np.zeros(6);pixel_sources=[]
    def pixels(p):
        with Image.open(p) as im:rgb=im.convert('RGB').resize((256,256),Image.Resampling.BICUBIC);return np.array(rgb,dtype=np.float64)/255,np.array(rgb.convert('L'),dtype=np.float64)/255
    for arm in IMAGE_ARMS:
        g={k:load_features(root,arm,k,torch) for k in t};values=arrays[arm];assert values.shape==(400,6) and np.isfinite(values).all();empty=0
        for i,query in enumerate(q):
            row=read(image_root/'records'/f'{i:03d}_{arm}.json');target=targets[i];assert all(row[k]==target[k]==v for k,v in query.items());assert not row['normal_inference_target_access'] and row['execution']==s['upstream_image_execution']
            assert ref(row['generated_path'])['sha256']==row['generated_sha256'] and ref(target['image_path'])['sha256']==meta[i]['image_sha256']==target['image_sha256'];pixel_sources.append(row['generated_sha256'])
            a=g['EffNet'][i].astype(float);b=t['EffNet'][i].astype(float);assert np.std(a)>0 and np.std(b)>0;eff=float(np.corrcoef(a,b)[0,1]);cap=float(np.dot(g['CapSim'][i].astype(float),t['CapSim'][i].astype(float)));f1=manual_F1(g['ObjectF1'][i],t['ObjectF1'][i],np);empty+=not categories(g['ObjectF1'][i],.05,np)
            rgb,y=pixels(row['generated_path']);trgb,ty=pixels(target['image_path']);pixel_corr=float(np.corrcoef(rgb.ravel(),trgb.ravel())[0,1]) if np.std(rgb)>0 and np.std(trgb)>0 else 0.
            ssim=structural_similarity(y,ty,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,data_range=1.,channel_axis=None)
            expected=np.array([eff,cap,f1,(eff+cap+f1)/3,pixel_corr,ssim]);error=np.abs(expected-values[i]);maximum=np.maximum(maximum,error);assert (error<=np.array([2e-12,5e-7,2e-12,2e-7,2e-12,1e-10])).all()
        assert empty==result['generated_counts'][arm]['generated_detector_empty']
    assert result['target_detector_empty']==sum(not categories(row,.05,np) for row in t['ObjectF1'])
    concepts=sorted({row['concept_id'] for row in q});groups=[[i for i,row in enumerate(q) if row['concept_id']==c] for c in concepts];draws=arrays['concept_bootstrap_draws'];assert np.array_equal(draws,np.random.default_rng(1729).integers(0,40,(1000,40)))
    contrasts={};max_stats=0.
    for a in ARMS:
        contrasts[a]={}
        for b in IMAGE_ARMS:
            if a==b:continue
            delta=arrays[a]-arrays[b];cluster=np.stack([delta[g].mean(0) for g in groups]);sample=np.array([sum(cluster[j] for j in d)/40 for d in draws]);ci=np.quantile(sample,[.025,.975],axis=0)
            contrasts[a][b]={}
            for j,key in enumerate(METRICS):
                value=result['contrasts'][a][b][key];err=max(abs(float(delta[:,j].mean())-value['difference']),float(np.max(np.abs(ci[:,j]-value['descriptive_concept_bootstrap95']))));max_stats=max(max_stats,err);assert err<2e-12
                contrasts[a][b][key]=dict(difference=float(delta[:,j].mean()),descriptive_concept_bootstrap95=ci[:,j].tolist())
    assert next_action(upstream['decision'],contrasts)==result['next_action'];checked(result['fixed_predeclared_qualitative']);assert not torch.cuda.is_initialized() and time.monotonic()-start<1800
    value=dict(status='COMPLETE_INDEPENDENT_ALL2800_SEED_LOWLEVEL_AND_30_COMPARISON_CPU_AUDIT',protocol=ref(path),readout=ref(root/'secondary_readout.json'),all2800_images_and_source_joins_passed=True,
        maximum_metric_formula_errors=dict(zip(METRICS,maximum.tolist())),maximum_bootstrap_error=max_stats,all_target_and_generated_detector_empty_counts_verified=True,public_empty_both_F1_one_NOT_object_recognition=True,
        next_action=result['next_action'],no_new_models_or_research_images=True,official_R80=False,CUDA_initialized=False,seconds=time.monotonic()-start)
    atomic(root/'independent_secondary_CPU_audit.json',value);print(json.dumps(value))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
