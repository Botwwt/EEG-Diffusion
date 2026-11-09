"""More explicitly named old derived arrays; never raw EEG or learned models."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import ROOT,sha,write,torch
from rebalance_BrainHIVE_derived_storage_CPU_v1 import DATA,RAM


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    assert (ROOT/'audit/BrainHIVE_derived_storage_rebalance_v11.json').exists()
    paths=[]
    for folder in ['/root/mced_eeg_brainhive_teacher_confirmation_samples_v1','/root/mced_eeg_native_conditional_prior_samples_v1']:
        paths.extend(Path(folder).glob('*_samples.npz'))
    study=Path('/root/autodl-tmp/mced_research_bundle/studies')
    paths += [study/'brain_spatial_encoding_pilot_v1/spatial_features.pt',study/'repa_real_eeg_feature_probe_v1/features_and_labels.pt']
    paths.extend((study/'eeg_clip_visual_feature_cache_v2').glob('*token_mean.pt'))
    rime=Path('/root/rime_var_20261015/artifacts')
    paths += [rime/'balanced_repeat_calibration_v1/target_log_probabilities.pt',rime/'repeat_evidence_calibration_v1/target_log_probabilities.pt',rime/'trial_latents_v1/latents.pt']
    paths.extend((DATA/'archived_validation_samples_compact_v1/compact').rglob('*_validation_samples.pt'))
    paths=sorted(set(p for p in paths if p.is_file() and not p.is_symlink()))
    plan=ROOT/'audit/BrainHIVE_extra_derived_array_plan_v12.json'
    if not args.apply:
        assert not plan.exists()
        rows=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
        total=sum(r['bytes'] for r in rows)
        assert shutil.disk_usage('/dev/shm').free>total+(2<<30)
        write(plan,dict(source_sha256=sha(__file__),items=rows,bytes=total,raw_EEG_or_model_weights_in_scope=False))
        print('EXTRA_REGENERABLE_ARRAY_PLAN',len(rows),total,flush=True);return
    spec=json.loads(plan.read_text());assert spec['source_sha256']==sha(__file__)
    done=ROOT/'audit/BrainHIVE_extra_derived_arrays_v12.json';assert not done.exists()
    before={p:shutil.disk_usage(p)._asdict() for p in ['/root','/root/autodl-tmp','/dev/shm']}
    results=[]
    for row in spec['items']:
        p=Path(row['path']);assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256']
        if p.suffix=='.pt':
            data=torch.load(p,weights_only=True,map_location='cpu',mmap=True)
            if isinstance(data,dict):
                assert not any(str(k) in ['model','optimizer','state_dict'] or str(k).endswith(('.weight','.bias')) for k in data)
            del data
        destination=RAM/'extra_arrays'/p.relative_to('/root');destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();shutil.copyfile(p,destination);os.chmod(destination,0o600)
        assert sha(destination)==row['sha256']
        link=p.with_name(p.name+'.EXTRA_RAM_PENDING_v12');assert not link.exists();link.symlink_to(destination)
        os.replace(link,p);assert sha(p)==row['sha256']
        item=row|dict(RAM_target=str(destination),is_regenerable_cache=True,RAM_is_volatile=True)
        with (ROOT/'audit/BrainHIVE_extra_derived_array_progress_v12.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        results.append(item)
    after={p:shutil.disk_usage(p)._asdict() for p in before}
    write(done,dict(status='EXPLICIT_DERIVED_ARRAY_RAM_RELOCATION_COMPLETE',before=before,after=after,items=results,
        raw_EEG_removed=0,unique_images_removed=0,learned_models_removed=0,plan_sha256=sha(plan)))
    print(json.dumps(dict(status='EXTRA_ARRAYS_MOVED_WITH_BYTE_CHECKS',files=len(results),bytes=spec['bytes'],before=before,after=after)),flush=True)


if __name__=='__main__':main()
