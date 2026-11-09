"""All eighty real T pairs and three coupled training-state distributions."""
import argparse,hashlib,sys,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from brainhive_overlap_gradient_contract_CPU_20261103_v1 import setup,resources,draw_axial,logcosh,DOMAINS,GEOMETRY

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);args=parser.parse_args();started=time.monotonic()
    spec,base,G=setup(args.protocol);before=resources(spec,base,G,start=True);root=Path(spec['output_root'])/'prepared';assert not root.exists();root.mkdir()
    prior=G.read(G.checked(spec['prior_prepared']));data=dict(np.load(G.checked(prior['arrays']),allow_pickle=False))
    trace=G.read(G.checked(spec['prior_trace']));noise=np.load(G.checked(spec['prior_noise']),mmap_mode='r',allow_pickle=False)
    old_full=dict(np.load(G.checked(spec['full_T_noise_diagnosis']),allow_pickle=False));alpha=old_full['alpha_cumprod']
    times=np.array(trace['timesteps'],np.int64);assert np.array_equal(times,np.array(trace['timesteps']))
    split=G.read(G.checked(base['input_pins']['split']));position={r:i for i,r in enumerate(split['image_rows']['T'])}
    latent=np.load(G.checked(spec['T_latent']),mmap_mode='r',allow_pickle=False)
    states=np.lib.format.open_memmap(root/'states.npy',mode='w+',dtype=np.float32,shape=(3,80,4,4,64,64))
    geometry=np.empty((3,80,4,len(GEOMETRY)),np.float64);axials=np.empty((80,4,3),np.float64);axial_seeds=np.empty((80,4),np.uint64)
    assert trace['rows']==spec['queries'] and len(spec['queries'])==80
    for index,row in enumerate(spec['queries']):
        zi=np.asarray(latent[position[row['array_row']]],np.float64);zj=np.asarray(latent[position[row['donor_array_row']]],np.float64)
        assert np.array_equal(zi.astype(np.float32),data['latent'][index]);delta=zi-zj;length=np.linalg.norm(delta);assert length>0;direction=delta/length
        for ti,timestep in enumerate(times):
            a=float(np.sqrt(alpha[timestep]));sig=float(np.sqrt(1-alpha[timestep]));eps=np.asarray(noise[index,ti],np.float64)
            separation=a/sig*length;axial=float(np.sum(eps*direction));perpendicular=eps-axial*direction
            assert abs(np.sum(perpendicular*direction))<1e-11
            seed=int(hashlib.sha256(f'1729|EEG_overlap_axial|{index}|{int(timestep)}'.encode()).hexdigest()[:15],16)
            accepted,attempts=draw_axial(np.random.default_rng(seed),separation)
            state0=a*zi+sig*eps;state1=.5*a*(zi+zj)+sig*eps;state2=.5*a*(zi+zj)+sig*(perpendicular+accepted*direction)
            axials[index,ti]=[axial,accepted,attempts];axial_seeds[index,ti]=seed
            for domain,state in enumerate((state0,state1,state2)):
                value=state.astype(np.float32);states[domain,index,ti]=value;exact=value.astype(np.float64)
                ei=(exact-a*zi)/sig;ej=(exact-a*zj)/sig;ell=.5*(np.sum(ej*ej)-np.sum(ei*ei))
                geometry[domain,index,ti]=[separation,ell,expit(ell),expit(-ell),-separation*separation/8,-np.log(4.)-logcosh(ell/2)]
            assert abs(geometry[2,index,ti,1]-separation*accepted)<1e-3
    states.flush();np.savez_compressed(root/'geometry_and_axials.npz',geometry=geometry,axials=axials,axial_seeds=axial_seeds,alpha_cumprod=alpha,timesteps=times)
    metadata=dict(status='COMPLETE_ALL80_T_THREE_DISTRIBUTION_CPU_STATES_NO_CUDA',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),
        protocol=G.ref(args.protocol),source=G.ref(__file__),states=G.ref(root/'states.npy'),geometry=G.ref(root/'geometry_and_axials.npz'),
        old_inputs=spec['prior_prepared'],rows=spec['queries'],domains=DOMAINS,geometry_keys=GEOMETRY,
        harmonic_sampling='StandardNormal axial proposal accepted bysech(d*s/2), independentGaussian perpendicular; normalized harmonic density. AfterFP32quantization scalarresponsibilities recomputed.',
        no_claim_all_midpoint_draws_ambiguous=True,new_models=0,new_GPU_jobs=0,new_images=0,NEW_contacted=False,seconds=time.monotonic()-started,
        resources_before=before,resources_after=resources(spec,base,G))
    G.write(root/'completion.json',metadata);print(G.json.dumps(metadata))
if __name__=='__main__':main()
