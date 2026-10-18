"""Fit TRAIN-only coordinates/operator; emit a separate EEG-only query packet."""
import argparse
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=Path(a.out);assert str(out)==cfg['prepared'];out.mkdir(exist_ok=False);begin=time.monotonic()
    report=dict(status='running',stage='contracts',official_test_read=False,GPU_test_performed=False,source_sha256=digest(__file__))
    write(out/'progress.json',report);write(out/'config.json',cfg)
    try:
        import numpy as np
        import torch
        assert not torch.cuda.is_available();torch.set_num_threads(2);d=cfg['latent_dimension'];assert d==64 and cfg['R']==4
        from precision_score_math import validate_math
        write(out/'synthetic_math_tests.json',validate_math())
        source=Path(cfg['source']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==cfg['source_commit']
        sys.path.insert(0,str(source/'Generation'));from diffusion_prior import DiffusionPriorUNet
        import hashlib
        source_class=hashlib.sha256(inspect.getsource(DiffusionPriorUNet).encode()).hexdigest()
        s1=Path(cfg['stage1']);split_path=s1/'split_manifest.json';split=json.loads(split_path.read_text());s1cfg=json.loads((s1/'config.json').read_text())
        ids=np.array(split['train_image_ids']);assert len(ids)==14540 and len(split['train_concepts'])==1454 and len(split['development_concepts'])==200
        assert not set(split['train_concepts'])&set(split['development_concepts']) and split['official_test_locked']
        old=Path(cfg['repeat_run']);manifest=json.loads((old/'training_cache_manifest.json').read_text())
        assert manifest['image_ids']==ids.tolist() and manifest['split_sha256']==digest(split_path)
        assert digest(manifest['cache']['path'])==manifest['cache']['sha256']
        zraw=np.load(manifest['cache']['path']);assert zraw.shape==(14540,4,1024) and zraw.dtype==np.float32
        cal_manifest=json.loads((old/'calibration_manifest.json').read_text());assert digest(old/'calibration.npz')==cal_manifest['calibration_sha256']
        cal=np.load(old/'calibration.npz');sessions=np.array(manifest['session_ids'])
        def debias(values,ss):
            values=values.astype(np.float64,copy=True)
            for session,bias in zip(cal['levels'],cal['bias']):values[ss==session]-=bias
            return values
        zcorrect=debias(zraw,sessions);zmean=zcorrect.mean(1)
        def pca(x):
            mean=x.mean(0);center=x-mean;cov=center.T@center/(len(x)-1);values,vectors=np.linalg.eigh(cov)
            eigen=values[-d:][::-1].copy();basis=vectors[:,-d:][:,::-1].copy();assert eigen.min()>1e-9
            scale=np.sqrt(eigen);projection=basis/scale
            return mean,basis,scale,projection,float(eigen.sum()/np.maximum(values,0).sum())
        zm,zb,zs,zp,zfraction=pca(zmean);train_z=(zmean-zm)@zp
        # The file includes the overall training pool, including reserved dev rows.
        # Only the declared14540 TRAIN rows below enter any fitted transform.
        feature_path=Path(s1cfg['features']);assert digest(feature_path)=='557fc952ba74a351fd17fee2ab11bf67d7b351d0d571fb32b25728fdad5e5831'
        all_features=torch.load(feature_path,map_location='cpu',weights_only=True)
        y=all_features['img_features'][ids].float();del all_features
        y=torch.nn.functional.normalize(y,dim=-1).numpy().astype(np.float64)
        ym,yb,ys,yp,yfraction=pca(y);train_y=(y-ym)@yp
        projected=(zcorrect-zm)@zp;within=[];between=[]
        for row,ss in zip(projected,sessions):
            groups=[row[ss==g] for g in np.unique(ss)];assert len(groups)==2 and all(len(g)==2 for g in groups)
            within.extend([g[0]-g[1] for g in groups]);between.append(groups[0].mean(0)-groups[1].mean(0))
        within=np.stack(within);between=np.stack(between)
        trial=within.T@within/(2*len(within))
        def psd(x,floor):
            values,vectors=np.linalg.eigh((x+x.T)/2);return (vectors*np.maximum(values,floor))@vectors.T
        session=psd(between.T@between/(2*len(between))-trial/2,0)
        repeat_cov=psd(session/2+trial/4,1e-5);chol=np.linalg.cholesky(repeat_cov)
        iso=np.eye(d)*np.trace(repeat_cov)/d;iso_chol=np.linalg.cholesky(iso)
        assert np.allclose(np.trace(iso),np.trace(repeat_cov),rtol=1e-12)
        # Every development input uses exactly the previously selected four trials.
        base_cfg=json.loads(Path('/root/avde_rime_20261015/native680_stage2_v1.json').read_text());query_path=Path(base_cfg['generation_query_packet'])
        packet=torch.load(query_path,map_location='cpu',weights_only=True)
        assert packet['split']=='development' and packet['eeg_uV'].shape==(200,4,63,200)
        assert len(torch.unique(packet['trial_ids']))==800 and torch.equal(packet['query_ids'],torch.arange(200))
        assert digest(old/'EEG_only_query_fusions.npz')==cal_manifest['query_fusions_sha256']
        qraw=np.load(old/'EEG_only_query_fusions.npz')['single_trials'];assert qraw.shape==(200,4,1024)
        qz=(debias(qraw,packet['session_ids'].numpy()).mean(1)-zm)@zp
        meta=np.load(Path(s1cfg['raw'])/'training_trial_metadata.npz');physical=np.stack([meta['session_ids'].ravel(),meta['onset_samples'].ravel()],1)
        train_physical=physical[(ids[:,None]*4+np.arange(4)).ravel()]
        dev_ids=np.array(split['development_query_image_ids']);query_physical=physical[(dev_ids[:,None]*4+np.arange(4)).ravel()]
        assert not set(map(tuple,train_physical))&set(map(tuple,query_physical))
        train=out/'training_only.pt';torch.save(dict(schema=cfg['schema'],z=torch.tensor(train_z,dtype=torch.float32),
            y=torch.tensor(train_y,dtype=torch.float32),image_ids=torch.tensor(ids),concept_ids=torch.tensor(ids//10),
            repeat_chol=torch.tensor(chol,dtype=torch.float32),isotropic_chol=torch.tensor(iso_chol,dtype=torch.float32),
            split_sha256=digest(split_path)),train)
        queries=out/'EEG_only_queries.pt';torch.save(dict(schema=cfg['schema'],z=torch.tensor(qz,dtype=torch.float32),
            query_ids=packet['query_ids'],trial_ids=packet['trial_ids'],session_ids=packet['session_ids'],R=4,
            source_query_sha256=digest(query_path),split_sha256=digest(split_path)),queries)
        transforms=out/'training_only_transforms.npz';np.savez(transforms,z_mean=zm,z_basis=zb,z_scale=zs,z_projection=zp,
            y_mean=ym,y_basis=yb,y_scale=ys,y_projection=yp,repeat_covariance=repeat_cov,isotropic_covariance=iso,
            trial_covariance=trial,session_covariance=session)
        manifest=dict(status='completed',schema=cfg['schema'],split='training_fit/internal_development_queries',dtype='float32 model packets, float64 coordinate estimates',
            training_packet=dict(path=str(train),bytes=train.stat().st_size,sha256=digest(train)),query_packet=dict(path=str(queries),bytes=queries.stat().st_size,sha256=digest(queries)),
            transforms=dict(path=str(transforms),bytes=transforms.stat().st_size,sha256=digest(transforms)),
            shapes=dict(training_z=[14540,d],training_y=[14540,d],query_z=[200,d],noise_covariance=[d,d]),
            preprocessing='Fixed trained FP32 LaBraM single-trial latents; session bias removed using training only; EEG PCA64/global-whitened; unit-normalized CLIP-H14 target PCA64/global-whitened; TRAIN-only fits',
            EEG_explained_variance_fraction=zfraction,visual_explained_variance_fraction=yfraction,
            repeat_noise_eigenvalues=np.linalg.eigvalsh(repeat_cov).tolist(),noise_trace_matched=True,
            split_manifest_sha256=digest(split_path),encoder=manifest['encoder'],feature_source_sha256=digest(feature_path),
            training_trial_cache_sha256=manifest['cache']['sha256'],source_class_sha256=source_class,source_commit=cfg['source_commit'],
            physical_training_trial_count=58160,physical_query_trial_count=800,physical_overlap=0,
            query_target_image_class_token_caption_gallery_absent=True,official_test_read=False,
            covariance_is_training_working_estimate=True,synthetic_EEG_noise_equivalent_to_real_lower_R_proven=False)
        write(out/'data_manifest.json',manifest)
        report.update(status='completed',stage='train_only_coordinates_and_repeat_operator_ready',manifest=manifest,
            environment=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,CUDA_initialized=torch.cuda.is_initialized()))
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(seconds=time.monotonic()-begin,utc=datetime.now(timezone.utc).isoformat());write(out/'progress.json',report);write(out/'results.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
