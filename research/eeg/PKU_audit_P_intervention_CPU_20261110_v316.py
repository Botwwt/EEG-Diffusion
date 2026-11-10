"""Independently recompute every training-token shift and every intervention token."""
import hashlib,json,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');OUT=R/'runs/EEG_P_intervention_tokens_v316';start=time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

try:
    import numpy as np
    import torch
    torch.set_num_threads(4)
    done=json.loads((OUT/'completion.json').read_text());assert done['T_queries']==13230
    assert sha(OUT/'token_bank.pt')==done['token_bank_sha256'] and sha(OUT/'all_T_native_and_trained_tokens.npy')==done['T_tokens_sha256']
    bank=torch.load(OUT/'token_bank.pt',map_location='cpu',weights_only=True)
    table=np.load(OUT/'all_T_native_and_trained_tokens.npy',allow_pickle=False,mmap_mode='r');assert table.shape==(3,13230,4,2048)
    checks=[]
    for ci,case in enumerate(['P_positive','P_replay_pair']):
        total=np.zeros((4,2048),dtype=np.float64)
        for lo in range(0,13230,128):total+=(table[ci+1,lo:lo+128].astype(np.float64)-table[0,lo:lo+128].astype(np.float64)).sum(0)
        mean=(total/13230).astype(np.float32);assert np.array_equal(mean,bank['means'][ci].numpy())
        base=bank['full_D_tokens'][0].numpy().astype(np.float32);full=bank['full_D_tokens'][ci+1].numpy().astype(np.float32)
        common=(base+mean).astype(np.float16);specific=(full-mean).astype(np.float16)
        assert np.array_equal(common,bank['tokens'][ci*2].numpy()) and np.array_equal(specific,bank['tokens'][ci*2+1].numpy())
        checks.append(dict(case=case,all_T_tokens_replayed=13230*4*2048,all_D_tokens_replayed=2*830*4*2048,FP64_mean_then_FP32_exact=True,FP16_interventions_exact=True))
    old=torch.load(R/'runs/EEG_downstream_images_v303/condition_bank.pt',map_location='cpu',weights_only=True)
    assert bank['array_rows']==old['array_rows'] and bank['stimulus_ids']==old['stimulus_ids']
    for i in range(4):assert torch.equal(bank['condition'][i],old['condition'][0])
    (R/'runs/EEG_P_intervention_images_v316').mkdir(exist_ok=False)
    v=dict(status='PASS_ALL_T_MEANS_AND_ALL830_FOUR_TOKEN_INTERVENTIONS',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,checks=checks,old_parities=done['parities'],token_bank_sha256=done['token_bank_sha256'],condition_bank_sha256=done['token_bank_sha256'],policy_sha256=done['policy_sha256'],source_sha256=sha(__file__),new_images=0)
    p=OUT/'CPU_audit.json';p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
except BaseException as e:
    (OUT/'CPU_audit_failure.json').write_text(json.dumps(dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise
