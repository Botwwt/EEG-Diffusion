"""Full T frozen target/gradient diagnosis; six temporary steps are restored."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
import copy, fcntl, hashlib, json, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import atm_relative_P_diagnostic_core_20261105_v183 as F
plan = json.loads(Path(sys.argv[1]).read_bytes())
root = Path(plan["parent_project"])
sys.path.insert(0, str(root/"code"))
import atm_deployment_training_contract_20261105_v179 as C
c = C.setup(root/"code/policy.json"); q, K, pq = c["q"], c["K"], c["physical"]
out = Path(plan["output_root"]); start = time.monotonic()
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__) == plan["sources"]["GPU"]
for pin in plan["sources"].values(): K.checked(pin)
before = C.resources(c, starting=True)
assert before["own_remaining_bytes"] >= plan["maximum_output_bytes"] + (4<<20)
repo = Path(pq["GPU_wrapper"]["path"]).parent
j = K.read(repo/"audit"/(plan["job_name"]+"_job.json"))
assert j["status"] == "RUNNING" and j["pid"] == os.getpid()
lock = (repo/"audit/TRIALSET_GPU_INTERLEAVE.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip() == pq["GPU_UUID"]
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
sys.path[:0] = pq["existing_runtime_paths"] + ["/root/atm_official_20261016"]
import torch
from models.atms import ATMS
torch.set_num_threads(2); torch.manual_seed(1729); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(pq["maximum_GPU_bytes"]/torch.cuda.get_device_properties(0).total_memory)

def guard():
    assert time.monotonic()-start < 850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0 < pq["maximum_extra_CPU_bytes"]
    assert torch.cuda.max_memory_allocated() < pq["maximum_GPU_bytes"]
    assert K.allocated(out) <= plan["maximum_output_bytes"]
    C.resources(c)

try:
    base = torch.load(K.checked(q["checkpoint"]), map_location="cpu", weights_only=True, mmap=True)
    mean = torch.load(K.checked(pq["EEG"]), map_location="cpu", weights_only=True, mmap=True)
    vis = torch.load(K.checked(pq["visual_target"]), map_location="cpu", weights_only=True, mmap=True)
    physical = np.load(K.checked(pq["physical_source"]), allow_pickle=True)
    physical = physical if isinstance(physical, dict) else physical.item()
    physical = physical["preprocessed_eeg_data"]
    z = np.load(K.checked(q["projector"]))
    weights = [torch.from_numpy(z[k].astype(np.float32)).cuda() for k in
               ("image_embeds.weight", "image_embeds.bias", "norm.weight", "norm.bias")]
    net = ATMS().cuda(); net.load_state_dict(base["model"], strict=True)
    params = tuple(net.parameters()); named = list(net.named_parameters())
    npar = sum(p.numel() for p in params); assert npar == 3200413
    scale_mask = torch.cat([torch.full((p.numel(),), name != "logit_scale", dtype=torch.bool, device="cuda") for name, p in named])
    saved_buffers = {n: b.clone() for n, b in net.named_buffers()}
    reset_buffers = lambda: [b.copy_(saved_buffers[n]) for n, b in net.named_buffers()]
    T = np.asarray(q["T_array_rows"], np.int64); assert len(T) == 13230
    assert hashlib.sha256(T.tobytes()).hexdigest() == q["T_rows_sha256"]
    order = np.random.default_rng(1739).permutation(1323)
    coverage = np.zeros(13230, np.uint8)
    nums = np.zeros((13230, 2, 3, 2, 4), np.float64)
    dens = np.zeros((13230, 2, 3), np.float64)
    global_g = torch.zeros((3, npar), device="cuda", dtype=torch.float64)
    grams, traces, fixtures, finite = [], [], {}, []
    coordinates = np.unique(np.linspace(0, npar-1, 4096, dtype=np.int64))
    for bi, lo in enumerate(range(0, 1323, 16)):
        guard()
        pos = (order[lo:lo+16, None]*10+np.arange(10)).ravel(); ids = T[pos]; coverage[pos] += 1
        x, y = mean[ids].cuda(), vis[ids].cuda()
        # All repeat readouts use original eval BN, not mixed training running states.
        net.eval()
        block = physical[ids].astype(np.float32)
        np.testing.assert_allclose(block.mean(1), mean[ids].numpy(), rtol=1e-5, atol=1e-5)
        eval_logits, eval_outputs = [], []
        with torch.no_grad():
            for ri in range(5):
                e = torch.from_numpy(block[:, ri]).cuda() if ri < 4 else x
                cv = net(e, torch.ones(len(x), device="cuda", dtype=torch.long))
                _, logits = F.losses(net, cv, y, weights)
                eval_logits.append(np.stack([v.double().cpu().numpy() for v in logits]))
                if bi == 0: eval_outputs.append(cv.cpu().numpy())
        saved = np.stack(eval_logits, axis=1)
        nu, de = F.repeat_mass(saved); nums[pos], dens[pos] = nu, de
        if bi == 0:
            fixtures.update(ids=ids, x=x.cpu().numpy(), y=y.cpu().numpy(), physical=block,
                            eval_logits=saved, eval_outputs=np.stack(eval_outputs), sample_indices=coordinates)
            cv = net(x, torch.ones(len(x), device="cuda", dtype=torch.long))
            losses, _ = F.losses(net, cv, y, weights)
            eval_g = []
            for k, loss in enumerate(losses):
                grads = torch.autograd.grad(loss, params, retain_graph=k<2, allow_unused=True)
                eval_g.append(F.gradient_flat(grads, params))
            gg = torch.stack(eval_g)
            fixtures["eval_gradient_samples"] = gg[:, coordinates].cpu().numpy()
            fixtures["eval_gradient_norms"] = gg.double().square().sum(1).cpu().numpy()
            del gg, eval_g, cv, losses, grads
        # One real native stochastic forward shared by every candidate objective.
        net.train(); torch.cuda.manual_seed(101729+bi)
        cv = net(x, torch.ones(len(x), device="cuda", dtype=torch.long))
        losses, _ = F.losses(net, cv, y, weights)
        gs = [F.gradient_flat(torch.autograd.grad(loss, params, retain_graph=k<2, allow_unused=True), params)
              for k, loss in enumerate(losses)]
        gg = torch.stack(gs).double()
        gram = gg[:, scale_mask] @ gg[:, scale_mask].T
        grams.append(gram.cpu().numpy()); global_g += gg*len(x)/13230
        traces.append(dict(batch=bi, queries=len(x), losses=[float(v.detach()) for v in losses],
                           mapper_cosines=F.cosine(grams[-1]).tolist(),
                           BN_updated_once=all(int(b)==int(saved_buffers[n])+1 for n,b in net.named_buffers() if n.endswith("num_batches_tracked"))))
        assert traces[-1]["BN_updated_once"]
        if bi == 0: fixtures["train_output"] = cv.detach().cpu().numpy()
        with torch.no_grad(): reset_buffers()
        del gg, gs, cv, losses
        # Fixed first two batches only; direct combined autograd, same saved Adam and mask.
        if bi < 2:
            for route in ("F_only", "F_plus_P_MSE", "F_plus_P_NCE"):
                net.load_state_dict(base["model"], strict=True)
                opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.01)
                opt.load_state_dict(copy.deepcopy(base["optimizer"]))
                net.train(); torch.cuda.manual_seed(101729+bi)
                cv = net(x, torch.ones(len(x), device="cuda", dtype=torch.long))
                ll, _ = F.losses(net, cv, y, weights)
                post_buffers = {n:b.detach().clone() for n,b in net.named_buffers()}
                with torch.no_grad():
                    net.eval()
                    pre = [float(v) for v in F.losses(net, net(x,torch.ones(len(x),device="cuda",dtype=torch.long)), y, weights)[0]]
                joint = ll[0] + (0 if route=="F_only" else ll[1 if route=="F_plus_P_MSE" else 2])
                opt.zero_grad(set_to_none=True); joint.backward()
                gradnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.))
                opt.step()
                with torch.no_grad():
                    net.eval()
                    post = [float(v) for v in F.losses(net,net(x,torch.ones(len(x),device="cuda",dtype=torch.long)),y,weights)[0]]
                    for n,b in net.named_buffers(): b.copy_(saved_buffers[n])
                    net.train(); torch.cuda.manual_seed(101729+bi)
                    train_post = [float(v) for v in F.losses(net,net(x,torch.ones(len(x),device="cuda",dtype=torch.long)),y,weights)[0]]
                finite.append(dict(batch=bi,route=route,train_before=[float(v.detach()) for v in ll],train_after=train_post,
                                   eval_BNmatched_before=pre,eval_after=post,gradient_norm=gradnorm,
                                   original_Adam_step=830,temporary_Adam_step=831,all_BN_baselines_matched=True))
                net.load_state_dict(base["model"],strict=True)
                del opt,cv,ll,joint
        assert all(torch.equal(b,saved_buffers[n]) for n,b in net.named_buffers())
        K.write(out/"progress.json",dict(batch=bi+1,total_batches=83,covered=int(coverage.sum()),updates_retained=0))
        del x,y
    assert np.all(coverage==1)
    for name,value in net.state_dict().items(): assert torch.equal(value.cpu(),base["model"][name]),name
    full_gram = (global_g[:,scale_mask] @ global_g[:,scale_mask].T).cpu().numpy()
    np.savez_compressed(out/"statistics.npz",T=T,coverage=coverage,numerators=nums,denominators=dens,
                        batch_grams=np.stack(grams),global_mapper_gram=full_gram)
    np.savez_compressed(out/"fixture.npz",**fixtures)
    K.write(out/"finite_steps.json",finite)
    K.write(out/"batch_trace.json",traces)
    guard()
    K.write(out/"GPU_completion.json",dict(status="COMPLETE_FULL_T_RELATIVE_P_DIAGNOSTIC",T_queries=13230,
          physical_repeats=4,batches=83,losses=list(F.LOSSES),temporary_updates=6,retained_updates=0,new_images=0,
          global_mapper_cosines=F.cosine(full_gram).tolist(),statistics=K.ref(out/"statistics.npz"),fixture=K.ref(out/"fixture.npz"),
          finite_steps=K.ref(out/"finite_steps.json"),trace=K.ref(out/"batch_trace.json"),
          checkpoint_unchanged=K.ref(K.checked(q["checkpoint"])),model_and_buffers_restored=True,
          seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
          extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
except BaseException as e:
    K.write(out/"GPU_failure.json",dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start))
    raise
