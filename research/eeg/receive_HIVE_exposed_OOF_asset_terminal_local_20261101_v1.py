"""Accept bounded terminal metadata and independently total all20 actual fits."""
import argparse,base64,hashlib,json
from pathlib import Path,PurePosixPath


def main(receipt):
    root=Path(__file__).resolve().parent;r=json.loads(Path(receipt).read_bytes());assert r['exit_code']==0
    d=json.loads(r['stdout']);assert d['controller']['exact_live'] is False
    assert d['completed_fits']==d['completed_independent_audits']==20
    assert 'recovery1_controller_completion.json' in d['state'] and 'recovery1_controller_failure.json' not in d['state']
    out=root/'audit/HIVE_exposed_OOF_asset_complete_20261101_v1';assert not out.exists();out.mkdir()
    received=[];total=0
    for relative,item in d['terminal_files'].items():
        name=PurePosixPath(relative);assert not name.is_absolute() and '..' not in name.parts
        target=out/'actual'/Path(*name.parts);assert target.resolve().is_relative_to(out.resolve())
        raw=base64.b64decode(item['base64'],validate=True);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        total+=len(raw);assert total<16<<20;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        received.append(dict(relative_path=relative,source={k:item[k] for k in ('path','bytes','sha256')}))
    def read(p):return json.loads((out/'actual'/p).read_bytes())
    certificate=read('input_bank_completion.json');protocol=d['launch']['protocol']
    assert certificate['status']=='COMPLETE_ACTUAL_TRAIN_ONLY_FULL1654_20FOLD165400_OOF_ASSETS' and certificate['protocol']==protocol
    assert certificate['queries']==165400 and certificate['all20_folds_and_all165400_join_values_verified']
    assert certificate['official200_previously_exposed'] is True and certificate['independent_H'] is False
    assert certificate['final_component_or_benchmark_admitted'] is False and certificate['official_test_payloads_opened'] is False
    fits=[];max_error=0.
    for subject in range(1,11):
        for fold in range(2):
            task=f'sub-{subject:02d}/official_oof{fold}';c=read('tasks/'+task+'/completion.json');a=read('tasks/'+task+'/independent_CPU_audit.json')
            assert c['protocol']==a['protocol']==protocol and c['task_id']==a['task_id']==task
            assert (c['updates'],c['epochs'],c['held_queries'],c['temporary_updates'])==(225,25,8270,0)
            assert c['learned_weights_saved'] and not c['official_test_payloads_loaded']
            assert a['status']=='PASS_INDEPENDENT_FULL25_EPOCH_AND8270_HELD_NATIVE_OOF_AUDIT'
            assert [(e['epoch'],e['updates']) for e in a['epochs']]==[(e,e*9) for e in range(1,26)]
            assert all(e['all7_Adam_states_checked'] and e['shuffle_full_permutation_and_RNG_verified'] for e in a['epochs'])
            assert a['all8270_manual_CPU_FP64_forwards_checked'] and a['all25_epoch_Adam_counters_moments_and_shuffle_checked']
            assert c['final_model']==a['final_model'] and c['raw_conditions']==a['raw_conditions'] and c['export_array_rows']==a['export_array_rows']
            assert not a['official_test_payloads_opened'] and not a['CUDA_initialized']
            max_error=max(max_error,a['manual_CPU_GPU_max_absolute_error'])
            fits.append(dict(task_id=task,model=c['final_model'],updates=225,exposures=8270*25,exported_queries=8270,program_seconds=c['seconds'],CPU_audit_seconds=a['seconds'],manual_forward_max_error=a['manual_CPU_GPU_max_absolute_error']))
    caches=[]
    for p in sorted((out/'actual/cache_reviews').glob('*.json')):
        c=json.loads(p.read_bytes());assert c['status']=='COMPLETE_OWNED_CLEAN_OUTPUT_CACHE_ADVICE_ALL_BYTES_RETAINED'
        assert c['files_deleted_or_moved']==0 and not c['global_drop_caches_used']
        assert all(f['all_bytes_and_identity_unchanged'] for f in c['files'])
        caches.append(dict(label=c['label'],floor_met=c['original24GiB_floor_met_after'],headroom_after=c['after']['effective_headroom_bytes'],seconds=c['seconds']))
    assert len(caches)==18 and len({c['label'] for c in caches})==18
    summary=dict(status='PASS_LOCAL_ALL20_ACTUAL_FIT_AUDIT_CERTIFICATE_AND_ASSEMBLY_METADATA',actual_terminal_observation_UTC=d['UTC'],
        protocol=protocol,training_seed=1729,subjects=10,concepts=1654,images=16540,actual_fits=20,updates=sum(f['updates'] for f in fits),
        EEG_exposures=sum(f['exposures'] for f in fits),exported_queries=sum(f['exported_queries'] for f in fits),fit_details=fits,
        independent_CPU_forward_max_absolute_error=max_error,all500_epoch_audit_entries_locally_checked=True,
        raw165400_condition_values_not_downloaded_locally=True,not_every_gradient_transition_replayed=True,
        cache_reviews=caches,received_metadata_files=len(received),received_metadata_bytes=total,received=received,
        official200_previously_exposed=True,independent_H=False,final_component_or_benchmark_admitted=False,new_image_effect_result=False,
        second_method_confirmed=False,SOTA=False,CCF_A=False,goal_active=True)
    (out/'local_independent_scope_and_receiving.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    d.pop('terminal_files');d.pop('metadata_files',None)
    (out/'terminal_observation.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('fit_details','received','cache_reviews')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);a=p.parse_args();main(a.receipt)
