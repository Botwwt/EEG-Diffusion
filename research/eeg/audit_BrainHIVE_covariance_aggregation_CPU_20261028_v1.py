"""Reaggregate all T stimuli/concepts independently of the spectral probe."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();start=time.monotonic()
    manifest=json.loads((a.evidence_root/'actual_metadata_export_manifest.json').read_bytes())
    for item in manifest['files']:
        path=a.evidence_root/'actual'/item['path'];raw=path.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    base=a.evidence_root/'actual/audit/eeg_covariance_identifiability_20261028_v1/probe'
    summary=json.loads((base/'summary.json').read_bytes());assert summary['status']=='T_ONLY_REPEAT_DIFFERENCE_SPECTRAL_STRUCTURE_CPU_COMPLETE'
    values=np.load(base/'per_stimulus_all6_average.npy',allow_pickle=False);assert values.shape==(13230,2,2,3,3) and np.isfinite(values).all()
    concepts=[json.loads(r) for r in (base/'per_concept.jsonl').read_text('utf-8').splitlines()];assert len(concepts)==1323
    rows=[];means=[];largest=0.
    for item in concepts:
        ids=item['stimulus_rows'];assert len(ids)==10 and len(set(ids))==10;rows+=ids
        raw=np.array(item['metrics']);assert raw.shape==(2,2,3,2,3,3)
        actual=values[ids].mean(0);recorded=raw.mean(axis=(2,3));largest=max(largest,float(np.max(np.abs(actual-recorded))));assert np.allclose(actual,recorded,atol=1e-12,rtol=1e-12)
        means.append(actual)
    assert sorted(rows)==list(range(13230));means=np.array(means)
    indices=np.random.default_rng(1729).integers(0,1323,size=(1000,1323))
    draws=np.stack([means[index].mean(0) for index in indices]);low,high=np.quantile(draws,[.025,.975],axis=0)
    old=summary['average_all3_partitions_and2_directions']
    for key,value in [('mean',means.mean(0)),('descriptive_95_lower',low),('descriptive_95_upper',high)]:assert np.allclose(value,old[key],atol=1e-12,rtol=1e-12)
    primary=dict(Top1_mean=float(means[:,0,0,0,0].mean()),Top1_descriptive95=[float(low[0,0,0,0]),float(high[0,0,0,0])],AFC_mean=float(means[:,0,0,0,1].mean()),AFC_descriptive95=[float(low[0,0,0,1]),float(high[0,0,0,1])],correct_minus_cyclic_Top1=float(means[:,0,0,2,0].mean()),correct_minus_cyclic_Top1_descriptive95=[float(low[0,0,2,0]),float(high[0,0,2,0])])
    result=dict(status='ALL13230_STIMULI_AND1323_CONCEPTS_AGGREGATION_AND_BOOTSTRAP_INDEPENDENTLY_VERIFIED',source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),source_summary=dict(path=str(base/'summary.json'),sha256=hashlib.sha256((base/'summary.json').read_bytes()).hexdigest()),export_files_verified=len(manifest['files']),unique_stimuli=13230,concepts=1323,duplicate_or_missing_rows=0,maximum_stimulus_vs_concept_absolute_error=largest,all_four_views_means_and_intervals_verified=True,primary=primary,interpretation='Primary raw/full Top1 and2AFC intervals include chance; weak normalized secondary2AFC association preserved. Full-versus-diagonal increments not established. No architecture/model/image admission.',new_models=0,new_GPU_jobs=0,new_images=0,new_training_updates=0,ordinary_changed=False,seconds=time.monotonic()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
