"""Audit retained whole bank cells and coverage, not replay discarded raw logits."""
import argparse,datetime
import numpy as np
import VAR_full_T_native_bank_contract_CPU_20261102_v1 as C
def audit_part(q,index):
 p=C.OUT/f'part_{index:03d}';v=C.read(p/'completion.json');lo,hi=C.part_range(index)
 assert v['status']=='COMPLETE_FULL680_ALL1000_NATIVE_T_BANK_SEGMENT' and v['protocol']==C.ref(C.CODE/'protocol.json') and v['T_range']==[lo,hi]
 for pin in v['files']:C.checked(pin)
 a=np.load(p/'scale_log_likelihoods.npy',mmap_mode='r');z=np.load(p/'class_coverage.npy');t=np.load(p/'T_positions.npy')
 assert a.shape==(hi-lo,1000,10) and a.dtype==np.float64 and np.isfinite(a).all() and (a<=0).all()
 assert np.array_equal(z,np.ones((hi-lo,1000),np.uint8)) and np.array_equal(t,np.arange(lo,hi))
 records=v['streamed_FP32_token_hashes'];assert len(records)==(hi-lo)*16
 for j,r in enumerate(records):
  row=lo+j//16;a=j%16*64;b=min(a+64,1000);assert r['row']==row and r['classes']==[a,b] and r['terms']==(b-a)*680 and len(r['streamed_FP32_token_logprobs_SHA256'])==64 and r['all_token_GPU_vs_CPU_FP64_sums_exact']
 assert v['consumed_token_terms']==(hi-lo)*1000*680
 return dict(part=index,rows=hi-lo,retained_scale_cells=(hi-lo)*1000*10,streamed_token_terms=v['consumed_token_terms'],completion=C.ref(p/'completion.json'))
def run(protocol):
 q=C.setup(protocol);C.resources(q);parts=[audit_part(q,k) for k in range(207)]
 assert sum(v['rows'] for v in parts)==13230 and sum(v['retained_scale_cells'] for v in parts)==132300000
 C.write(C.OUT/'whole_bank_CPU_audit.json',dict(status='PASS_ALL13230_ROWS_132300000_RETAINED_SCALE_CELLS_AND_COMPLETE_CLASS_COVERAGE',protocol=C.ref(protocol),parts=parts,source_probability_kernel='DECLARED_FP16_NOT_FP32_BIT_IDENTITY',raw_token_arrays_persisted=False,producer_streaming_checks_not_independent_replay_of_discarded_tokens=True,new_EEG_fits=0,new_research_images=0,CUDA_initialized=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
