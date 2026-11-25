"""CPU regression tests for prospective render receipts, original batches, layout and EEG controls."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import paper_render_cache as cache


HERE = Path(__file__).resolve().parent


def pixels(folder, starts, done, n=19, bad=False):
    for first in starts:
        end = min(first + 8, n)
        for i in range(first, end):
            Image.new("RGB", (512, 512), (i, 2 if bad else 1, 3)).save(folder / f"{i:05d}.png")
        done(first, end)


def renderer(root):
    """Isolate training-only imports; exercise the actual renderer's CPU tensor operations."""
    modules = {}
    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        modules[name] = m
    unused = lambda *a, **kw: None
    module("common", ROOT=root, RUNS=root/"personal-01", PREP=root/"prep", PRIOR=root/"prior", TURBO=root/"turbo")
    module("exp1_swapdiff_3arm", ResidualIPCondLayer=unused, per_sample_cone=unused)
    module("exp6_draft_intra4", DRAFT_DIR=root, forward_branches_and_trunk=unused, rollout_4step_step_ips=unused)
    module("exp8_trial_precision_swapdiff", prec_erp_analytic=unused)
    module("main.models_brain", BrainEncoder=unused)
    module("o13_xbrain_shared_trunk", CrossSubjectHorizonEncoder=unused)
    module("paper_training", D1=root/"d1.pt", D2=root/"d2.pt", MATCHED=root/"matched.pt",
           REC=root/"rec.pt", SWAP=root/"swap.pt", generator=unused)
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("render_under_test", HERE/"paper_render.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "sub-01"
        self.request = {"render_cache_version": 2, "n": 19, "source": "one"}

    def run_images(self, generate=pixels, request=None):
        return cache.render_transaction(self.out, self.request if request is None else request, generate, lambda: None)

    def test_interrupt_resume_and_bit_exact_scored_repair(self):
        def interrupted(folder, starts, done):
            pixels(folder, starts[:1], done)
            raise InterruptedError("test interruption")
        with self.assertRaises(InterruptedError):
            self.run_images(interrupted)
        self.assertFalse(self.out.exists())
        seen = []
        def resume(folder, starts, done):
            seen.append(starts)
            pixels(folder, starts, done)
        self.assertTrue(self.run_images(resume))
        self.assertEqual(seen, [[8, 16]])
        final = self.out/"render.json"
        raw = final.read_bytes()
        manifest = json.loads(raw)
        self.assertFalse(self.run_images(lambda *args: self.fail("valid case regenerated")))
        cache.save_json(self.out/"scores.json", {"render_sha256": cache.sha(final), "ordered_image_hashes_sha256":
                        cache.hashlib.sha256("\n".join(manifest["pixels_sha256"]).encode()).hexdigest()})
        bad = self.out/"00010.png"
        bad.write_bytes(b"interrupted png")
        with self.assertRaisesRegex(ValueError, "differ from the committed"):
            self.run_images(lambda p,s,d: pixels(p,s,d,bad=True))
        self.assertEqual(bad.read_bytes(), b"interrupted png")
        self.assertTrue(self.run_images(resume))
        self.assertEqual(seen[-1], [8])
        self.assertEqual(cache.png_hash(bad), manifest["pixels_sha256"][10])
        final.write_bytes(b'{"interrupted"')
        self.assertTrue(self.run_images(lambda *args: self.fail("all actual images still valid")))
        self.assertEqual(final.read_bytes(), raw)
        with self.assertRaisesRegex(ValueError, "different inputs"):
            self.run_images(request=dict(self.request, source="two"))
        self.assertEqual(final.read_bytes(), raw)

    def test_historical_receipt_and_unbound_images_are_preserved(self):
        self.out.mkdir()
        (self.out/"render.json").write_bytes(b'{"draws":1}')
        with self.assertRaisesRegex(ValueError, "Historical"):
            self.run_images()
        self.assertFalse((self.root/".sub-01.render-v2").exists())
        (self.out/"render.json").unlink()
        (self.out/"00000.png").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "Unbound"):
            self.run_images()
        self.assertEqual((self.out/"00000.png").read_bytes(), b"old")

    def test_independent_writers_render_and_fit_once(self):
        worker = """
from pathlib import Path
import sys, time
from PIL import Image
import torch
import paper_render_cache as c
root = Path(sys.argv[1])
def fit():
    with (root/'fit-count').open('a') as f: f.write('1')
    time.sleep(.1)
    return {'weights': torch.arange(4)}
c.cached_layout(root/'layouts', {'sub':1}, fit, lambda pack: None, lambda: None)
def generate(stage, starts, done):
    with (root/'draw-count').open('a') as f: f.write('1')
    time.sleep(.1)
    Image.new('RGB', (512,512), (1,2,3)).save(stage/'00000.png')
    done(0, 1)
c.render_transaction(root/'images', {'render_cache_version':2,'n':1}, generate, lambda: None)
"""
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE":"1"}
        processes = [subprocess.Popen([sys.executable, "-B", "-c", worker, str(self.root)], cwd=HERE, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        for proc in processes:
            _, error = proc.communicate(timeout=60)
            self.assertEqual(proc.returncode, 0, error.decode(errors="replace"))
        self.assertEqual((self.root/"draw-count").read_text(), "1")
        self.assertEqual((self.root/"fit-count").read_text(), "1")

    def test_layout_uses_versioned_inputs_and_keeps_old_weights(self):
        base = self.root/"layouts"
        base.mkdir()
        (base/"ridge.pt").write_bytes(b"historical")
        fits = []
        def fit():
            fits.append(1)
            return {"w": torch.arange(4, dtype=torch.float32)}
        for req in [{"eeg":"A", "latent":"L"}, {"eeg":"B", "latent":"L"}, {"eeg":"B","latent":"M"}, {"eeg":"A","latent":"L"}]:
            _, meta = cache.cached_layout(base, req, fit, lambda p: self.assertEqual(tuple(p["w"].shape), (4,)), lambda: None)
        self.assertEqual(len(fits), 3)
        file = base/"v2"/meta["request_sha256"]/"ridge.pt"
        file.write_bytes(b"damaged")
        _, recovered = cache.cached_layout(base, meta["request"], fit, lambda p: None, lambda: None)
        self.assertEqual(meta, recovered)
        self.assertEqual((base/"ridge.pt").read_bytes(), b"historical")

    def test_checkpoint_hashes_track_subject_and_cross_object_donors(self):
        r = renderer(self.root)
        with self.assertRaisesRegex(ValueError, "Participant"):
            r.condition_files("d2", 2, cache.PinnedFiles())
        with self.assertRaisesRegex(ValueError, "Participant"):
            cache.track_for(self.root/"native-full-01",2)
        self.assertEqual(cache.track_for(self.root/"shared",8),"shared")
        parent = self.root/"backbone.pt"
        torch.save({"state_dict":{"w":torch.ones(1)}}, parent)
        torch.save({"backbone_ckpt":str(parent),"cond_layer":{}}, r.ARMS["d2"])
        _, _, first = r.condition_files("d2",1,cache.PinnedFiles())
        torch.save({"state_dict":{"w":torch.zeros(1)}}, parent)
        _, _, second = r.condition_files("d2-mean-erp",1,cache.PinnedFiles())
        self.assertEqual(first["condition_checkpoint"], second["condition_checkpoint"])
        self.assertNotEqual(first["nested_backbone"]["sha256"], second["nested_backbone"]["sha256"])
        self.assertEqual(r.FACTOR_CONDITION["d2-reference"], "d2")
        self.assertNotIn("d2-reference",r.NO_LAYOUT)
        self.assertEqual(r.ARMS["d2-positive-half"], self.root/"d2_positive_half_cond_st120_s2025.pt")
        selected = r.RUNS/"enc/object_tokens/sub-01/token-reliable.pt"
        selected.parent.mkdir(parents=True)
        torch.save({"net":torch.ones(1)},selected)
        cache.save_json(selected.with_suffix(".json"),{"mode":"token-reliable","subject":1,"checkpoint_sha256":cache.sha(selected)})
        wrong_source = r.FACTOR_CONDITION["token-reliable-wrong"]
        self.assertEqual(r.condition_files(wrong_source,1,cache.PinnedFiles())[2], r.condition_files("token-reliable",1,cache.PinnedFiles())[2])
        self.assertTrue(all(k in r.NO_LAYOUT for k in [*r.TOKEN_ARMS,"token-reliable-wrong"]))
        for panel, labels in [("official", np.arange(200)), ("fresh", np.repeat(np.arange(50),10)), ("dev", np.repeat(np.arange(20),10))]:
            d = r.donor_rows("token-reliable-wrong",panel,labels)
            self.assertTrue(np.all(labels != labels[d]))
            self.assertTrue(np.array_equal(d,np.roll(np.arange(len(labels)),len(labels)//2)))
            if panel != "official":
                self.assertTrue(np.array_equal(d%10, np.arange(len(labels))%10))
                np.testing.assert_array_equal(labels[r.donor_rows("wrong",panel,labels)],labels)

    def test_semantic_arithmetic_ablation_and_token_condition_exact(self):
        r = renderer(self.root)
        x = torch.arange(2*4*3*5).reshape(2,4,3,5).float()
        calls = []
        def branch(enc,tensor,sub,eff):
            calls.append((eff,tensor.clone()))
            flat=tensor.flatten(1)
            return flat*2,flat-3,None
        r.forward_branches_and_trunk=branch
        r.prec_erp_analytic=lambda b: b[:,0]*.7+b[:,3]*.3
        f,h=r.semantic_branch(None,x,1,"d2-mean-erp")
        self.assertEqual(len(calls),1)
        torch.testing.assert_close(calls[0][1],x.mean(1),rtol=0,atol=0)
        self.assertEqual(calls[0][0],4.)
        torch.testing.assert_close(f,x.mean(1).flatten(1)*2,rtol=0,atol=0)
        calls.clear()
        f,h=r.semantic_branch(None,x,1,"d2")
        self.assertEqual([c[0] for c in calls],[4.,1.])
        torch.testing.assert_close(f,.72*(r.prec_erp_analytic(x).flatten(1)*2)+.28*(x.flatten(2)*2).mean(1),rtol=0,atol=0)
        class Encoder(torch.nn.Module):
            def __init__(self,**kw):
                super().__init__()
            def backbone(self,b):
                return b.flatten(1)
        r.BrainEncoder=Encoder
        for mode in r.TOKEN_ARMS:
            p=r.RUNS/f"enc/object_tokens/sub-01/{mode}.pt"
            p.parent.mkdir(parents=True,exist_ok=True);torch.save({},p)
            cache.save_json(p.with_suffix(".json"),{"subject":1,"mode":mode,"checkpoint_sha256":cache.sha(p)})
            with patch.object(torch.Tensor,"cuda",lambda self,*a,**kw:self), patch.object(torch.nn.Module,"cuda",lambda self:self):
                tokens,digest=r.condition(mode,1,x.numpy(),[[0,1,2,3]],lambda z:z+2,lambda z:z*4)
            self.assertEqual(digest, cache.sha(p))
            torch.testing.assert_close(tokens,(x.mean(1).flatten(1).half()+2)*4,rtol=0,atol=0)

    def test_physical_eeg_array_and_row_order_are_bound(self):
        r = renderer(self.root); r.PREP.mkdir()
        ids={"ids":["one","two"]};cache.save_json(r.PREP/"eeg_train_ids.json",ids)
        cache.save_json(r.PREP/"train_ids.json",ids)
        path=r.PREP/"sub-01_train.npy"
        np.save(path,np.ones((2,4,17,250),dtype=np.float16))
        cache.save_json(path.with_suffix(".json"),{"output_sha256":cache.sha(path),"ids_sha256":cache.sha(r.PREP/"eeg_train_ids.json")})
        pinned=cache.PinnedFiles()
        _,proof=r.physical_eeg(1,"train",pinned)
        self.assertEqual(proof["array_sha256"],cache.sha(path))
        cache.save_json(r.PREP/"train_ids.json",{"ids":["two","one"]})
        with self.assertRaisesRegex(ValueError,"changed"):
            pinned.verify()
        with self.assertRaisesRegex(ValueError,"identities"):
            r.physical_eeg(1,"train",cache.PinnedFiles())

    def test_render_driver_reuses_one_layout_and_preserves_recipient_seeds(self):
        r=renderer(self.root);r.PREP.mkdir()
        rows=list(range(200))
        cache.save_json(r.PREP/"paper_split.json",{"official_rows":rows,"train_rows":rows})
        cache.save_json(r.PREP/"test_ids.json",{"paths":[f"test_images/{i:05d}_c{i}/img.jpg" for i in rows]})
        native=r.RUNS/"enc/object_tokens/sub-01/token-reliable.pt";native.parent.mkdir(parents=True)
        torch.save({},native);cache.save_json(native.with_suffix(".json"),{"subject":1,"mode":"token-reliable","checkpoint_sha256":cache.sha(native)})
        parent=self.root/"backbone.pt";torch.save({"state_dict":{}},parent)
        for m in ("d2","d2-positive-half"):
            torch.save({"backbone_ckpt":str(parent),"cond_layer":{}},r.ARMS[m])
        x=np.zeros((200,1,1,1),dtype=np.float32)
        r.physical_eeg=lambda sub,phase,files:(x,{"array_sha256":phase})
        r.fit_layout=lambda *a,**kw:("ridge-fixture",{"ridge_sha256":"same ridge"})
        r.generator_inputs=lambda f:{"generator":"fixture"}
        r.render_code=lambda f,s:{"renderer":"fixture"};r.runtime=lambda:{"device":"CPU fixture"}
        fake=types.ModuleType("diffusers")
        class Vae(torch.nn.Module):
            dtype=torch.float32
            config=SimpleNamespace(scaling_factor=.1)
            @classmethod
            def from_pretrained(cls,*a,**kw): return cls()
            def cuda(self): return self
            def enable_slicing(self): pass
        fake.AutoencoderKL=Vae
        pipe=SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.arange(4),sigmas=torch.arange(5),init_noise_sigma=1.),
                             unet=SimpleNamespace(dtype=torch.float16),vae=None)
        t=torch.ones(1,3)
        r.generator=lambda dtype:(pipe,None,None,t,t,t)
        calls=[]
        def layout(pack,x,g):
            calls.append(pack)
            return torch.arange(200)
        r.layout=layout
        def condition(*args,prepared,**kw): return torch.arange(200),prepared[2]["condition_checkpoint"]["sha256"]
        r.condition=condition
        seen=[]
        def draw(stage,pipe,vae,pr,ip,seeds,z0s,*,batch_starts,batch_complete):
            seen.append((ip.tolist(),list(seeds),None if z0s is None else z0s.tolist()))
            for start in batch_starts:
                end=min(start+8,len(seeds))
                for j in range(start,end):
                    Image.new("RGB",(512,512),(int(ip[j]),0,1)).save(stage/f"{j:05d}.png",compress_level=1)
                batch_complete(start,end)
        r.draw_images=draw
        arms=["token-reliable","token-reliable-wrong","d2-reference","d2-mean-erp","d2-positive-half"]
        a=SimpleNamespace(subject=1,budgets=["official-r4"],arms=arms)
        with patch.dict(sys.modules,{"diffusers":fake}):
            r.render(a)
            r.render(a)
            with self.assertRaisesRegex(ValueError,"Participant"):
                r.render(SimpleNamespace(subject=2,budgets=["official-r4"],arms=arms))
        self.assertEqual(len(seen),5);self.assertEqual(calls,["ridge-fixture"])
        self.assertEqual(seen[0][1],seen[1][1]);self.assertEqual(seen[1][0],np.roll(rows,100).tolist())
        self.assertIsNone(seen[0][2]);self.assertIsNone(seen[1][2])
        self.assertTrue(all(event[2]==rows and event[1]==seen[0][1] for event in seen[2:]))
        src=self.root/"images/personal"
        receipts=[json.loads((src/arm/"official-r4/sub-01/render.json").read_text()) for arm in arms]
        self.assertEqual(receipts[0]["condition_sources"],receipts[1]["condition_sources"])
        self.assertEqual(receipts[2]["checkpoint_sha256"],receipts[3]["checkpoint_sha256"])
        self.assertTrue(all(x["layout_source"]==receipts[2]["layout_source"] for x in receipts[2:]))

    def test_feature_entry_binds_nested_source_and_rejects_wrong_subject(self):
        r=renderer(self.root);r.PREP.mkdir()
        common=types.ModuleType("common");common.PREP=r.PREP;common.ROOT=r.ROOT;common.RUNS=r.RUNS
        with patch.dict(sys.modules,{"paper_render":r,"common":common}):
            spec=importlib.util.spec_from_file_location("feature_under_test",HERE/"paper_retrieval.py")
            feature=importlib.util.module_from_spec(spec);spec.loader.exec_module(feature)
        with self.assertRaisesRegex(ValueError,"Participant"):
            feature.measure(2,"d2",["official-r4"])
        native_rows=list(range(200));ids=[f"image{i}" for i in native_rows]
        cache.save_json(r.PREP/"paper_split.json",{"official_rows":native_rows})
        cache.save_json(r.PREP/"test_ids.json",{"ids":ids})
        f=r.PREP/"F_test.npy";x=np.random.default_rng(1).normal(size=(200,24)).astype(np.float32);np.save(f,x)
        cache.save_json(r.PREP/"paper_test_F_receipt.json",{"ordered_ids":ids,"F_test_sha256":cache.sha(f)})
        feature.physical_eeg=lambda *a:(x,{"array_sha256":"fixture"})
        feature.render_code=lambda *a:{"model":"fixture"}
        nested={"condition_checkpoint":{"sha256":"A"}, "nested_backbone":{"sha256":"B"}}
        feature.condition=lambda arm,sub,x,*a,**kw:(torch.from_numpy(x),"A",nested)
        feature.measure(1,"d2",["official-r4"])
        out=r.ROOT/"features/personal/d2/official-r4/sub-01"
        meta=json.loads((out/"features.json").read_text())
        self.assertEqual(meta["feature_source"]["condition_sources"],nested)
        self.assertEqual(meta["mean"]["feature_top1"],1.)
        self.assertEqual(meta["observations_sha256"],cache.sha(out/"features.npz"))
        nested["nested_backbone"]={"sha256":"changed"}
        with self.assertRaisesRegex(ValueError,"other inputs"):
            feature.measure(1,"d2",["official-r4"])


if __name__ == "__main__":
    unittest.main()
