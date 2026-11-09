"""Complete T fixed validation-preprocessing VQ codes; no EEG fit or selection."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,datetime,hashlib,io,json,subprocess,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from EEG_generative_backend_loader_20261029_v1 import build_vq_only,sha,SCALES

def checked(item):
    p=Path(item['path']);assert sha(p)==item['sha256'],str(p)
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def tensor_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        v=value.detach().cpu().contiguous();h.update(key.encode());h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(spec_path,segment):
    started=time.monotonic();spec=json.loads(spec_path.read_bytes());root=Path(spec['token_output']);root.mkdir(exist_ok=True);out=root/f'segment_{segment:02d}';assert not out.exists();out.mkdir();phase='SOURCE_CONTRACT';base=peak()
    try:
        assert 0<=segment<7;lo=segment*2048;hi=min(13230,lo+2048)
        for pin in spec['source_pins']:checked(pin)
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-0a99d971-a46c-6e81-f157-c33243c6d57a'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        memory=json.loads(checked(spec['memory_completion']).read_bytes());identities=json.loads(checked(memory['files']['identities']).read_bytes());queries=identities['T'];assert len(queries)==13230 and len({q['concept_id'] for q in queries})==1323
        split=json.loads(checked(spec['split']).read_bytes());assert [q['array_row'] for q in queries]==split['image_rows']['T']
        assert not set(split['image_rows']['T'])&set(split['image_rows']['D']+split['image_rows']['V'])
        manifest={int(q['array_row']):q for q in csv.DictReader(checked(spec['image_manifest']).open()) if q['dataset_split']=='train'}
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        # The author validation path has no random augmentation: scalar short-
        # edge Resize256 bilinear, ToTensor, then RGB[-1,1]. Require the actual
        # released stimulus to be square instead of silently adding a crop.
        transform=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),transforms.ToTensor()])
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        vae,load_proof=build_vq_only(spec['author_source'],spec['vendor'],checked(spec['VQ_checkpoint']));vae.cuda();torch.cuda.reset_peak_memory_stats();before=tensor_digest(vae)
        codes=np.lib.format.open_memmap(out/'indices.npy',mode='w+',dtype=np.int16,shape=(hi-lo,680));joined=[];archives={};maximum_cpu=0;replay=None;engineering=[]
        def guard():assert time.monotonic()-started<840 and torch.cuda.max_memory_allocated()<20<<30 and peak()-base<10<<30
        def images(rows):
            tensors=[]
            for i in rows:
                q=queries[i];r=manifest[q['array_row']];assert r['stimulus_id']==q['stimulus_id'] and r['concept_id']==q['concept_id']
                if r['image_zip'] not in archives:archives[r['image_zip']]=zipfile.ZipFile(r['image_zip'])
                data=archives[r['image_zip']].read(r['image_member']);assert hashlib.sha256(data).hexdigest()==r['image_sha256'] and len(data)==int(r['image_bytes'])
                im=Image.open(io.BytesIO(data)).convert('RGB');assert im.width==im.height,('Author validation preprocessing requires square stimulus',q['stimulus_id'],im.size)
                t=transform(im).mul(2).sub(1);assert t.shape==(3,256,256);tensors.append(t)
                joined.append(dict(query_index=i,array_row=q['array_row'],stimulus_id=q['stimulus_id'],concept_id=q['concept_id'],image_sha256=r['image_sha256'],image_zip=r['image_zip'],image_member=r['image_member'],original_size=list(im.size)))
            return torch.stack(tensors).cuda()
        phase='FROZEN_VQ_COMPLETE_T_SEGMENT';position=lo
        with torch.inference_mode():
            while position<hi:
                guard();end=min(hi,position+(2 if position==0 else 8));x=images(range(position,end));indices=vae.img_to_idxBl(x)
                assert [z.shape[1] for z in indices]==[s*s for s in SCALES]
                result=torch.cat(indices,1);assert result.shape==(end-position,680) and result.min()>=0 and result.max()<4096
                codes[position-lo:end-lo]=result.cpu().numpy().astype(np.int16)
                if position==0:
                    again=torch.cat(vae.img_to_idxBl(x),1);assert torch.equal(result,again);replay=dict(fixed_query_indices=[0,1],identical_indices=True,extra_numeric_replay_images=2)
                    reconstruction=vae.idxBl_to_img(indices,same_shape=True,last_one=True)
                    for j in range(2):
                        for tag,value in [('source',x[j].add(1).mul(.5)),('VQ_reconstruction',reconstruction[j].add(1).mul(.5))]:
                            pixels=value.float().cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy();path=out/f'engineering_{j}_{tag}.png';Image.fromarray(pixels).save(path);engineering.append(dict(image=ref(path),kind=tag,query_index=j,engineering_only=True,EEG_reconstruction_benefit_claimed=False))
                position=end;codes.flush()
                progress=dict(status='OWNED_T_VQ_PREFIX_PRESERVED',segment=segment,query_start=lo,completed_until_query=position,planned_query_end=hi,indices_path=str(out/'indices.npy'),source_image_IDs=[q['stimulus_id'] for q in joined],seconds=time.monotonic()-started)
                pending=out/'progress.pending.json';pending.write_text(json.dumps(progress)+'\n',encoding='utf8');os.replace(pending,out/'progress.json')
        for archive in archives.values():archive.close()
        codes.flush();del codes;assert len(joined)==hi-lo and [x['query_index'] for x in joined]==list(range(lo,hi))
        assert before==tensor_digest(vae) and not any(p.requires_grad or p.grad is not None for p in vae.parameters())
        saved=np.load(out/'indices.npy',mmap_mode='r',allow_pickle=False);assert saved.dtype==np.int16 and saved.shape==(hi-lo,680) and saved.min()>=0 and saved.max()<4096
        write(out/'identity_rows.json',joined);guard()
        write(out/'completion.json',dict(status='FROZEN_STRICT_VQ_COMPLETE_T_SEGMENT_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),segment=segment,query_start=lo,query_end=hi,queries=hi-lo,indices=ref(out/'indices.npy'),identity_rows=ref(out/'identity_rows.json'),VQ_load=load_proof,VQ_weight_tensor_digest_before_after=before,all_VQ_parameters_and_buffers_frozen=True,preprocessing='Exact author validation Resize(shortedge256,BILINEAR), ToTensor, RGB[-1,1]; actual source square asserted; no crop, flip, augmentation or model/seed selection',physical_EEG_repeats=4,EEG_values_or_D_V_images_read=False,scales=list(SCALES),total_tokens_per_image=680,vocabulary=4096,indices_dtype='int16',numeric_replay=replay,engineering_images=engineering,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base,seconds=time.monotonic()-started,trained_models=0,new_research_images=0))
        print(json.dumps(dict(status='FROZEN_STRICT_VQ_COMPLETE_T_SEGMENT_CLOSED',segment=segment,queries=hi-lo,seconds=time.monotonic()-started)))
    except BaseException as error:
        write(out/'failure.json',dict(status='VQ_COMPLETE_T_SEGMENT_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,completed_until_query=locals().get('position'),saved_prefix_not_discarded=True));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--segment',type=int,required=True);a=p.parse_args();run(a.spec,a.segment)
