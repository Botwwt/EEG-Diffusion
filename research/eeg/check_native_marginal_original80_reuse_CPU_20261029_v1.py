"""Require exact original80 feature/record reuse before opening400 effects."""
import datetime,json
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parent;root=L/'audit/EEG_native_marginal_images_20261029_v2';checks={}
for arm in ('coupled','visual','marginal','donor'):
    host='old' if arm in ('coupled','donor') else 'new'
    old=root/'received_stage80_v1'/host;new=root/'received_stage400_v1'/host
    a=np.load(old/(arm+'_features.npy'),allow_pickle=False);b=np.load(new/(arm+'_features.npy'),allow_pickle=False)
    ra=json.loads((old/(arm+'_records.json')).read_bytes());rb=json.loads((new/(arm+'_records.json')).read_bytes())
    checks[arm+'_original80_features_bit_exact']=bool(np.array_equal(a,b[:80]))
    checks[arm+'_original80_records_and_images_bit_exact']=ra==rb[:80]
    checks[arm+'_complete400_order']=len(rb)==400 and [r['index'] for r in rb]==list(range(400))
assert all(checks.values()),checks
out=root/'original80_reuse_CPU_actual_v1.json';assert not out.exists();out.write_text(json.dumps(dict(status='PASS_ALL4_ORIGINAL80_REUSED_WITH_COMPLETE400_ORDER',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,effect_scores_computed=0),indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status='PASS_ALL4_ORIGINAL80_REUSED_WITH_COMPLETE400_ORDER',checks=len(checks))))
