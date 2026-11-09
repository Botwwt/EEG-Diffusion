"""Engineering continuation of the exact v203 experiment after mmap allocation."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'code'))
import atm_continuous_margin_contract_20261106_v203 as B
ARMS,CONTRASTS,PAPER,R4_METRICS,R80_METRICS=B.ARMS,B.CONTRASTS,B.PAPER,B.R4_METRICS,B.R80_METRICS
resources,records,paired=B.resources,B.records,B.paired

def setup(path):
 q=json.loads(Path(path).read_bytes());assert q['status']=='LOCKED_CONTINUOUS_MARGIN_MMAP_ENGINEERING_CONTINUATION'
 c=B.setup(q['original_policy']['path']);K=c['K'];old=c['q'];K.checked(q['original_policy'])
 assert q['state_compatibility_policy']==q['original_policy']
 for k in ['seed','epochs','attempts_per_arm','exposures_per_arm','models','project_root','maximum_output_bytes','wrapper_reserve_bytes','resource_ledger','training_contract','storage_plan','runtime_sources','baseline_T','Qbank_completion','review_plan']:
  assert q[k]==old[k],k
 for k in ['core','checkpoint','schedule']:assert q['sources'][k]==old['sources'][k]
 for p in list(q['sources'].values())+q['preserved_failures']:K.checked(p)
 source=K.read(K.checked(q['original_policy']))
 assert q['original_deadline_epoch']==K.read(Path(source['code_root'])/'actual_launch.json')['absolute_deadline_epoch']
 probe=K.read(q['mapping_completion_path']);assert probe==q['mapping_expected_result'] and probe['attempt4_committed_and_CPU_verified'] and probe['no_mapping_after_release']
 K.checked(probe['copied_checkpoint']);assert q['resume_from_attempt']==4 and q['expected_stage_count']==443
 c.update(q=q,out=Path(q['output_root']),original_execution_policy=old)
 return c
