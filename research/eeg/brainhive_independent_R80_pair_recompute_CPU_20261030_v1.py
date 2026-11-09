"""Independent full-denominator paired arithmetic; no production imports/Torch."""
import csv,math
from pathlib import Path
import numpy as np

COMMON=('DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED')
PAPER=('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav')
IDENTITY=('subject','stimulus_id','concept_id','image_seed','score_target_path')


def load_table(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==2000 and len({row['query_id'] for row in rows})==2000
    return {row['query_id']:row for row in rows}


def make_count_matrices():
    rng=np.random.default_rng(1729);subject=[];concept=[]
    for unused in range(1000):
        subject.append(np.bincount(rng.integers(0,10,size=10),minlength=10))
        concept.append(np.bincount(rng.integers(0,200,size=200),minlength=200))
    return np.asarray(subject,dtype=np.float64),np.asarray(concept,dtype=np.float64)


def linear_quantile(values,probability):
    ordered=sorted(float(value) for value in values);position=(len(ordered)-1)*probability;lower=int(math.floor(position));upper=int(math.ceil(position))
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def check_pair_identity(left,right):
    assert set(left)==set(right) and len(left)==2000
    for key,row in left.items():
        assert all(row[field]==right[key][field] for field in IDENTITY),(key,'Wrongpairedidentity')
    subjects=sorted({row['subject'] for row in left.values()});concepts=sorted({row['concept_id'] for row in left.values()})
    assert len(subjects)==10 and len(concepts)==200
    lookup={(row['subject'],row['concept_id']):key for key,row in left.items()};assert len(lookup)==2000
    return subjects,concepts,lookup


def verify_SEED(table):
    maximum=0.;undefined=0
    for row in table.values():
        fields=[row[name] for name in ('ObjectF1','EffNet','CapSim','SEED')]
        if any(value in ('','None') for value in fields):undefined+=1;continue
        objectF1,effnet,capsim,seed=map(float,fields);error=abs(math.fsum([objectF1,effnet,capsim])/3-seed)
        assert math.isfinite(error) and error<1e-11;maximum=max(maximum,error)
    return dict(maximum_SEED_identity_error=maximum,undefined_rows_retained=undefined,total_rows=2000)


def recompute(left_parts,right_parts):
    left=left_parts['common'];right=right_parts['common'];subjects,concepts,lookup=check_pair_identity(left,right)
    for table in (left_parts['paper'],right_parts['paper']):check_pair_identity(left,table)
    seed_left=verify_SEED(left);seed_right=verify_SEED(right);subject_counts,concept_counts=make_count_matrices();records={}
    for kind,names in (('common',COMMON),('paper',PAPER)):
        for name in names:
            a_table=left_parts[kind];b_table=right_parts[kind]
            defined=[key for key in lookup.values() if a_table[key][name] not in ('','None') and b_table[key][name] not in ('','None')]
            if len(defined)!=2000:
                records[name]=dict(status='UNKNOWN_UNDEFINED_FULL_DENOMINATOR_RETAINED',defined=len(defined),expected=2000);continue
            a=np.asarray([[float(a_table[lookup[sub,con]][name]) for con in concepts] for sub in subjects]);b=np.asarray([[float(b_table[lookup[sub,con]][name]) for con in concepts] for sub in subjects]);difference=a-b
            assert np.isfinite(a).all() and np.isfinite(b).all()
            # Independent matrix-count weighting, not production's indexed
            # slices or einsum. Quantiles use explicit linear interpolation.
            draws=np.sum((subject_counts@difference)*concept_counts,axis=1)/2000
            records[name]=dict(status='COMPLETE',left_mean=math.fsum(a.ravel().tolist())/2000,right_mean=math.fsum(b.ravel().tolist())/2000,paired_left_minus_right=math.fsum(difference.ravel().tolist())/2000,descriptive_subject_concept_bootstrap95=[linear_quantile(draws,.025),linear_quantile(draws,.975)],per_subject_difference=[math.fsum(row.tolist())/200 for row in difference],defined=2000,direction='lower_is_better' if name in ('eval_effnet','eval_swav') else 'higher_is_better')
    return dict(queries=2000,subjects=10,concepts=200,metrics=records,SEED_left=seed_left,SEED_right=seed_right)


def compare_published(replayed,published):
    maximum=0.
    for name,actual in replayed['metrics'].items():
        expected=published[name];assert actual['status']==expected['status']
        if actual['status']!='COMPLETE':continue
        for field in ('left_mean','right_mean','paired_left_minus_right'):
            maximum=max(maximum,abs(actual[field]-expected[field]))
        maximum=max(maximum,max(abs(a-b) for a,b in zip(actual['descriptive_subject_concept_bootstrap95'],expected['descriptive_subject_concept_bootstrap95'])))
        if 'per_subject_difference' in expected:maximum=max(maximum,max(abs(a-b) for a,b in zip(actual['per_subject_difference'],expected['per_subject_difference'])))
        assert actual['defined']==expected['defined'] and actual['direction']==expected['direction']
    assert maximum<1e-11
    return maximum
