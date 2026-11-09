"""Exercise future128000-cell auditor using explicitly ALIASED old real tables.

Norm_only is an alias of bounded_F in this TEST ONLY. No new norm image score
is consumed, no efficacy result is produced, and image provenance is not tested.
"""
import csv,gzip,json,subprocess,sys
from pathlib import Path
import numpy as np
from brainhive_loss_control_R80_statistics_CPU_20261030_v1 import COMMON,PAPER,paired,fixed_draws
L=Path(__file__).resolve().parent;old=L/'audit/HIVE_Euclidean_F_postfit_complete_20261030_v3';out=L/'audit/HIVE_norm_only_future_auditor_OLD_ALIAS_check_20261030_v1'
assert not out.exists();(out/'actual/joint_comparison').mkdir(parents=True);data={}
for role in ('parent','ordinary','bounded_F','Euclidean_F','Norm_only'):
    source='bounded_F' if role=='Norm_only' else role;target=out/'tables'/role;target.mkdir(parents=True);data[role]={}
    for kind in ('common','paper'):
        raw=(old/'tables'/source/(kind+'.csv')).read_bytes();(target/(kind+'.csv')).write_bytes(raw)
        with (target/(kind+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        data[role][kind]={r['query_id']:r for r in rows}
reference=data['parent']['common'];subs=sorted({r['subject'] for r in reference.values()});cons=sorted({r['concept_id'] for r in reference.values()});cells={(r['subject'],r['concept_id']):key for key,r in reference.items()}
draws=fixed_draws();results={};rows=0
with gzip.open(out/'actual/joint_comparison/all4x16_full_paired_differences.csv.gz','xt',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['comparison','metric','query_id','subject','concept_id','difference']);writer.writeheader()
    for right in ('Euclidean_F','bounded_F','ordinary','parent'):
        key='Norm_only_minus_'+right;results[key]={}
        for kind,names in [('common',COMMON),('paper',PAPER)]:
            for name in names:
                a,b=[np.asarray([[float(data[r][kind][cells[sub,con]][name]) for con in cons] for sub in subs]) for r in ('Norm_only',right)]
                report=paired(a,b,*draws);report['status']='COMPLETE';results[key][name]=report
                for i,sub in enumerate(subs):
                    for j,con in enumerate(cons):writer.writerow(dict(comparison=key,metric=name,query_id=cells[sub,con],subject=sub,concept_id=con,difference=float(a[i,j]-b[i,j])));rows+=1
assert rows==128000
(out/'actual/joint_comparison/summary.json').write_bytes((json.dumps(dict(queries_per_method=2000,results=results,TEST_ONLY_OLD_TABLE_ALIAS=True,new_norm_only_scores_used=False),indent=2)+'\n').encode())
r=subprocess.run([sys.executable,'-B',str(L/'audit_HIVE_norm_only_received_local_20261030_v1.py'),'--root',str(out)],capture_output=True,text=True);assert r.returncode==0,r.stderr
report=json.loads((out/'independent_local_statistics.json').read_bytes());assert report['independently_verified_pairs']==128000
(out/'test_scope.json').write_bytes((json.dumps(dict(status='PASS_OLD_REAL_TABLE_ALIAS_FULL128000_AUDITOR_TEST_ONLY',paired_cells=128000,methods=5,comparisons=4,metrics=16,positive_zero_and_negative_contrasts_tested=True,new_norm_only_efficacy_assessed=False,new_norm_only_scores_used=False,image_identity_or_pixels_not_tested=True,maximum_statistic_error=report['maximum_statistic_error'],maximum_raw_error=report['maximum_raw_cell_error']),indent=2)+'\n').encode())
print((out/'test_scope.json').read_text())
