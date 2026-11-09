"""Private, fixed-shape inference CUDA Graph for the actual SDXL UNet.

Uses the original operations/weights/dtypes. No compilation dependency, model
change, sampler change or training support. Every new input is copied; no query
condition may leak from the previous call. Unsupported arguments fail loudly.
"""
import torch


class StaticUNetGraph:
    def __init__(self,unet,warmups=3):
        assert not unet.training and not any(p.requires_grad for p in unet.parameters())
        self.original_forward=unet.forward;self.warmups=warmups
        self.graph=None;self.buffers=None;self.output=None
        self.counts=dict(logical_calls=0,logical_samples=0,warmup_calls=0,capture_calls=0,replays=0)

    @staticmethod
    def tensors(sample,timestep,encoder_hidden_states,added_cond_kwargs):
        assert isinstance(timestep,torch.Tensor) and timestep.ndim==0
        assert set(added_cond_kwargs)=={'text_embeds','time_ids','image_embeds'}
        images=added_cond_kwargs['image_embeds'];assert isinstance(images,list) and len(images)==1
        values=[sample,timestep,encoder_hidden_states,added_cond_kwargs['text_embeds'],
                added_cond_kwargs['time_ids'],images[0]]
        assert all(v.device.type=='cuda' and v.is_contiguous() for v in values)
        n=len(sample);assert sample.shape==(n,4,64,64) and images[0].shape==(n,1,1024)
        assert added_cond_kwargs['time_ids'].shape==(n,6)
        return values

    def original(self):
        x,t,context,pool,time_ids,image=self.buffers
        return self.original_forward(x,t,encoder_hidden_states=context,return_dict=False,
            added_cond_kwargs=dict(text_embeds=pool,time_ids=time_ids,image_embeds=[image]))[0]

    def __call__(self,sample,timestep,encoder_hidden_states,return_dict=False,added_cond_kwargs=None,**kwargs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert return_dict is False
        assert all(value is None for value in kwargs.values()),('Unsupported nonempty UNet argument',kwargs.keys())
        values=self.tensors(sample,timestep,encoder_hidden_states,added_cond_kwargs)
        self.counts['logical_calls']+=1;self.counts['logical_samples']+=len(sample)
        if self.graph is None:
            self.buffers=[v.detach().clone() for v in values]
            self.signature=[(tuple(v.shape),v.dtype,v.device,tuple(v.stride())) for v in values]
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.warmups):self.original();self.counts['warmup_calls']+=1
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self.original()
            self.counts['capture_calls']+=1
            self.graph.replay();self.counts['replays']+=1
        else:
            assert self.signature==[(tuple(v.shape),v.dtype,v.device,tuple(v.stride())) for v in values], 'Use a separately audited graph for another shape'
            for target,source in zip(self.buffers,values):target.copy_(source)
            self.graph.replay();self.counts['replays']+=1
        # Prevent a subsequent replay overwriting tensors a caller still holds.
        return (self.output.clone(),)


def install_static_graph(unet):
    graph=StaticUNetGraph(unet)
    unet.forward=graph
    return graph
