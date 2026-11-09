"""Reuse closed historical R4/full-V results to calibrate query sampling only.

Discovery reads and hashes seven existing score tables and optional DINO shards.
Replay performs no model/image fitting and no CUDA import. Existing DINO2AFC
values retain their ORIGINAL 1660-image gallery; sampling 80/400 queries does
not change that gallery. DINO feature discovery supports a separate later,
faithful 400-gallery CPU replay, not a claim that it has already occurred.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

SCHEMA = 'BrainHIVE_historical_fullV_screen_calibration_20261027_v1'
ROOT = Path(__file__).resolve().parent
SEED = 1729
PANEL_DRAWS = 200
BOOTSTRAP_DRAWS = 1000
GAIN = .01


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    temp.replace(path)


def verify(binding):
    path=Path(binding['path']);assert sha(path)==binding['sha256'],'Historical input changed: '+str(path)
    return path


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def discover(plan_path,data_root,output,max_seconds=600):
    started=time.monotonic();output=Path(output);assert not output.exists();output.mkdir(parents=True)
    plan=read(plan_path);data_root=Path(data_root)
    assert plan['schema']==SCHEMA and plan['fit_new_method'] is False and plan['select_new_query_ids_from_effects'] is False
    assert plan['simulation_seed']==SEED and plan['panel_draws']==PANEL_DRAWS and plan['bootstrap_draws']==BOOTSTRAP_DRAWS
    families=[];feature_roots={}
    for family in plan['families']:
        assert time.monotonic()-started<max_seconds
        item={k:family[k] for k in ('name','host','repeat','gallery_images','subjects','concepts','stimuli_per_concept','queries_per_arm','pairs')}
        assert item['repeat']=='R4' and item['gallery_images']==1660
        item.update(scores={},missing_scores=[],source=family)
        closure=data_root/family['closure']['path']
        if closure.is_file():
            assert sha(closure)==family['closure']['sha256'],'Closed historical summary changed'
            item['closure']=ref(closure)
        else:item['closure']=None
        for arm,spec in family['scores'].items():
            path=data_root/spec['path'];feature_roots[spec['features']]=family['queries_per_arm']
            if not path.is_file():item['missing_scores'].append(dict(arm=arm,path=str(path)));continue
            binding=ref(path)
            assert not spec['expected_sha256'] or binding['sha256']==spec['expected_sha256'],'Prior score SHA mismatch: '+arm
            table=records(path);assert len(table)==family['queries_per_arm']
            assert len({row['query_id'] for row in table})==len(table)
            mean=float(np.mean([float(row['DINO2AFC']) for row in table]))
            assert abs(mean-spec['expected_DINO_mean'])<1e-10,'Score mean differs from closed historical report'
            item['scores'][arm]=dict(binding=binding,fields=list(table[0]),rows=len(table),DINO2AFC_mean=mean,
                prior_per_query_SHA_available=spec['expected_sha256'] is not None,
                current_score_SHA_bound=True,closed_summary_mean_verified=True)
        item['available_pairs']=[pair for pair in family['pairs'] if all(arm in item['scores'] for arm in pair)]
        item['full_family_available']=not item['missing_scores'] and item['closure'] is not None
        families.append(item)
    feature_roots[plan['target_features']]=1660
    feature_inventory={}
    for folder,expected_rows in feature_roots.items():
        assert time.monotonic()-started<max_seconds
        root=data_root/folder;entry=dict(path=str(root),expected_rows=expected_rows,optional_later_replay_only=True)
        config=root/'config.json';completion=root/'completion.json'
        if not config.is_file() or not completion.is_file():
            entry.update(status='MISSING_FEATURE_METADATA',tensor_values_loaded=False);feature_inventory[folder]=entry;continue
        config_doc=read(config);complete=read(completion)
        assert len(config_doc['keys'])==expected_rows and len(set(config_doc['keys']))==expected_rows
        assert complete['status']=='COMPLETED' and complete['queries']==expected_rows
        assert complete['manifest_sha256']==config_doc['manifest_sha256']
        metadata=[ref(config),ref(completion)];shards=[];missing=[]
        for offset in range(0,expected_rows,256):
            assert time.monotonic()-started<max_seconds
            path=root/'DINO'/f'{offset:08d}.pt'
            if not path.is_file():missing.append(str(path));continue
            item=ref(path);relative=path.relative_to(data_root).as_posix()
            previous=plan['feature_prior_sha256'].get(relative)
            assert previous is None or previous==item['sha256'],'Pinned historical DINO shard changed'
            item['prior_closed_comparison_SHA_available']=previous is not None;shards.append(item)
        for path,binding in zip((config,completion),metadata):
            relative=path.relative_to(data_root).as_posix()
            previous=plan['feature_prior_sha256'].get(relative)
            assert previous is None or previous==binding['sha256'],'Pinned feature metadata changed'
        entry.update(status='COMPLETE_BOUND_DINO_SHARDS' if not missing else 'INCOMPLETE_DINO_SHARDS',
            metadata=metadata,DINO_shards=shards,missing_shards=missing,DINO_weight_sha256=config_doc['DINO_weight_sha256'],
            preprocessing=config_doc.get('dino_preprocessing'),key_order_SHA256=hashlib.sha256(json.dumps(config_doc['keys']).encode()).hexdigest(),
            tensor_values_loaded=False,untrusted_pickle_executed=False,GPU_calls=0)
        feature_inventory[folder]=entry
    for family in families:
        for info in family['scores'].values():verify(info['binding'])
    result=dict(schema=SCHEMA,status='HISTORICAL_FULLV_READ_ONLY_DISCOVERY_COMPLETE',plan=ref(plan_path),source=ref(__file__),
        data_root=str(data_root.resolve()),families=families,feature_inventory=feature_inventory,
        no_method_fit=True,new_images=0,new_GPU_calls=0,existing_scores_gallery=1660,
        score_subset_replay_is_not_400_gallery_recomputation=True,seconds=time.monotonic()-started)
    atomic(output/'discovery.json',result);return result


def category(value):return 'PRACTICAL_POSITIVE' if value>=GAIN else 'SMALL_POSITIVE' if value>0 else 'NONPOSITIVE'


def bootstrap_indices(subjects,concepts):
    rng=np.random.default_rng(SEED)
    # Preserve per-draw subject-then-concept random stream; use common draws
    # across all historical contrasts, without viewing any scores to select IDs.
    s=np.empty((BOOTSTRAP_DRAWS,subjects),np.int64);c=np.empty((BOOTSTRAP_DRAWS,concepts),np.int64)
    for i in range(BOOTSTRAP_DRAWS):
        s[i]=rng.integers(subjects,size=subjects);c[i]=rng.integers(concepts,size=concepts)
    return s,c


def interval(matrix,indices):
    subjects,concepts=indices
    values=matrix[subjects[:,:,None],concepts[:,None,:]].mean((1,2))
    return np.quantile(values,[.025,.975]).tolist()


def paired_matrix(candidate,baseline,family):
    left={r['query_id']:r for r in candidate};right={r['query_id']:r for r in baseline}
    assert len(left)==len(right)==family['queries_per_arm'] and set(left)==set(right)
    subjects=sorted({r['subject'] for r in left.values()});concepts=sorted({r['concept_id'] for r in left.values()})
    assert len(subjects)==family['subjects'] and len(concepts)==family['concepts']
    grouped={};stimuli={}
    for key,row in left.items():
        other=right[key]
        for field in ('subject','stimulus_id','concept_id','image_seed','score_target_path','EEG_source_stimulus_id'):
            assert row[field]==other[field],'Historical paired identity mismatch: '+field
        assert row['generated_path']!=other['generated_path'],'Historical generated path aliased'
        a,b=float(row['DINO2AFC']),float(other['DINO2AFC'])
        assert np.isfinite(a) and np.isfinite(b) and 0<=a<=1 and 0<=b<=1
        group=(row['subject'],row['concept_id']);grouped.setdefault(group,[]).append(a-b)
        stimuli.setdefault(group,set()).add(row['stimulus_id'])
    assert all(len(grouped[(s,c)])==family['stimuli_per_concept'] and len(stimuli[(s,c)])==family['stimuli_per_concept'] for s in subjects for c in concepts)
    for c in concepts:assert all(stimuli[(s,c)]==stimuli[(subjects[0],c)] for s in subjects)
    matrix=np.array([[np.mean(grouped[(s,c)]) for c in concepts] for s in subjects],np.float64)
    return matrix,subjects,concepts


def simulations(matrix,subjects,concepts):
    assert matrix.shape==(len(subjects),len(concepts)) and len(subjects)>=3 and len(concepts)>=40
    rng=np.random.default_rng(SEED);selections=[]
    for draw in range(PANEL_DRAWS):
        selections.append((rng.permutation(len(subjects))[:3],rng.permutation(len(concepts))[:40]))
    full=float(matrix.mean());full_category=category(full);result={};all_rows=[]
    for nconcepts in (8,40):
        for nsubjects in (1,3):
            indices=bootstrap_indices(nsubjects,nconcepts);rows=[]
            for draw,(selected_subjects,selected_concepts) in enumerate(selections):
                si=selected_subjects[:nsubjects];ci=selected_concepts[:nconcepts]
                selected=matrix[si[:,None],ci[None,:]];mean=float(selected.mean());low,high=interval(selected,indices)
                row=dict(draw=draw,concepts=nconcepts,subjects=nsubjects,queries=nsubjects*nconcepts*10,
                    subject_ids=[subjects[i] for i in si],concept_ids=[concepts[i] for i in ci],
                    DINO2AFC_difference=mean,descriptive95=[low,high],category=category(mean),
                    sign_matches_full=bool((mean>0)==(full>0)),category_matches_full=category(mean)==full_category,
                    CI_contains_full_point=low<=full<=high,CI_spans_zero=low<=0<=high,CI_spans_practical_gain=low<=GAIN<=high,
                    stage1_DINO_only_futility=high<GAIN,stage1_DINO_only_expand=high>=GAIN,
                    misleading_practical_mean=full<GAIN and mean>=GAIN,
                    missed_practical_full_effect=full>=GAIN and high<GAIN)
                rows.append(row);all_rows.append(row)
            result[f'{nconcepts}concept_{nsubjects}subject']=dict(panel_draws=PANEL_DRAWS,
                DINO_difference_quantiles=np.quantile([r['DINO2AFC_difference'] for r in rows],[.025,.5,.975]).tolist(),
                **{key:float(np.mean([r[key] for r in rows])) for key in ('sign_matches_full','category_matches_full','CI_contains_full_point',
                    'CI_spans_zero','CI_spans_practical_gain','stage1_DINO_only_futility','stage1_DINO_only_expand',
                    'misleading_practical_mean','missed_practical_full_effect')})
    return dict(full_query_mean=full,full_point_category=full_category,
                full_recomputed_descriptive95=interval(matrix,bootstrap_indices(len(subjects),len(concepts))),designs=result),all_rows


def replay(discovery_path,output,max_seconds=600):
    started=time.monotonic();output=Path(output);assert not output.exists();output.mkdir(parents=True)
    discovery=read(discovery_path);assert discovery['schema']==SCHEMA and discovery['status']=='HISTORICAL_FULLV_READ_ONLY_DISCOVERY_COMPLETE'
    verify(discovery['plan']);results=[];all_rows=[]
    for family in discovery['families']:
        assert time.monotonic()-started<max_seconds
        if family['closure'] is None:continue
        verify(family['closure']);tables={arm:records(verify(info['binding'])) for arm,info in family['scores'].items()}
        for candidate,baseline in family['available_pairs']:
            assert time.monotonic()-started<max_seconds
            matrix,subjects,concepts=paired_matrix(tables[candidate],tables[baseline],family)
            result,panels=simulations(matrix,subjects,concepts)
            result.update(family=family['name'],candidate=candidate,baseline=baseline,host=family['host'],repeat=family['repeat'],
                original_fixed_gallery=family['gallery_images'],full_queries=family['queries_per_arm'],
                within_concept_Top1_not_available_in_original_score_table=True,
                replay_is_DINO_futility_only_not_full_multimetric_rapid_gate=True)
            results.append(result)
            all_rows.extend(dict(family=family['name'],candidate=candidate,baseline=baseline,**row) for row in panels)
    assert results,'No verified complete historical comparison available'
    table=output/'simulated_panels.jsonl.gz'
    with gzip.open(table,'xt',encoding='utf-8') as stream:
        for row in all_rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    # Reverify bound score bytes after CPU replay; no source table is modified.
    for family in discovery['families']:
        for info in family['scores'].values():verify(info['binding'])
    summary=dict(schema=SCHEMA,status='COMPLETE_HISTORICAL_FIXED1660_SCORE_QUERY_SAMPLING_REPLAY',source=ref(__file__),
        discovery=ref(discovery_path),results=results,simulated_panel_rows=len(all_rows),per_panel=ref(table),
        seed=SEED,panel_draws=PANEL_DRAWS,bootstrap_draws=BOOTSTRAP_DRAWS,
        nested_selection='Each draw selects 3 subjects/40 concepts before inspecting effects; 1 subject/8 concepts are prefixes',
        sampling_without_replacement=True,bootstrap_with_replacement=True,
        all_ten_stimuli_retained_per_subject_concept=True,original_gallery_never_resampled=True,
        no_new_method_fit=True,new_GPU_calls=0,new_images=0,independent_confirmation=False,SOTA_established=False,
        original_data_previously_exposed=True,query_IDs_not_selected_for_future_method_evaluation=True,
        uncertainty_limits=['Full-V mean is a historical finite-sample reference, not the true population effect',
            'Historical gallery remains1660; this is not faithful400-gallery rescoring',
            'No within-concept Top1 is available in the original saved score CSVs',
            'Descriptive intervals and optional stopping have no nominal error control',
            'Do not train methods or choose favorable future query IDs from these V effects',
            'Families/hosts/repetition protocols are reported separately, never pooled'],seconds=time.monotonic()-started)
    atomic(output/'summary.json',summary);return summary


def self_test():
    # An analytically constant positive paired effect must survive all nested
    # panels, preserve all ten stimuli, and have a degenerate known interval.
    matrix=np.full((10,166),.02);subjects=[f's{i}' for i in range(10)];concepts=[f'c{i}' for i in range(166)]
    result,rows=simulations(matrix,subjects,concepts)
    assert len(rows)==PANEL_DRAWS*4 and result['full_point_category']=='PRACTICAL_POSITIVE'
    assert all(r['sign_matches_full'] and r['category_matches_full'] and not r['stage1_DINO_only_futility'] for r in rows)
    assert all(np.allclose(r['descriptive95'],[.02,.02],atol=1e-15) for r in rows)
    for draw in range(PANEL_DRAWS):
        selected=[r for r in rows if r['draw']==draw]
        small=next(r for r in selected if r['subjects']==1 and r['concepts']==8)
        large=next(r for r in selected if r['subjects']==3 and r['concepts']==40)
        assert small['concept_ids']==large['concept_ids'][:8] and small['subject_ids']==large['subject_ids'][:1]
    assert category(0)=='NONPOSITIVE' and category(.009999)=='SMALL_POSITIVE' and category(.01)=='PRACTICAL_POSITIVE'
    return dict(status='CPU_HISTORICAL_REPLAY_CONSTANT_EFFECT_NESTING_AND_BOUNDARY_CHECKS_PASSED',
                tests=4,synthetic_panels=len(rows),new_GPU_calls=0,real_historical_scores_read=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('discover','replay'));parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/historical_screen_calibration_20261027_v1/sources.json')
    parser.add_argument('--data-root',type=Path,default=Path('/root/brainhive_reproduction_20261020_v1'))
    parser.add_argument('--discovery',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--max-seconds',type=float,default=600)
    args=parser.parse_args(argv);assert 0<args.max_seconds<=600
    if args.self_test:print(json.dumps(self_test()));return 0
    assert args.output and args.mode
    if args.mode=='discover':result=discover(args.plan,args.data_root,args.output,args.max_seconds)
    else:assert args.discovery;result=replay(args.discovery,args.output,args.max_seconds)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],new_GPU_calls=0)));return 0


if __name__=='__main__':raise SystemExit(main())
