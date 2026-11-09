"""Explicit cache-native preprocessing and a single-forward EEG token interface.

This does not claim that existing whitened/cache coordinates are raw microvolts
or identical to the author's preprocessing. Statistics must come from T only.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class CacheNativeEEGTransform:
    channel_names: tuple
    center: np.ndarray
    scale: np.ndarray
    input_hz: int = 250
    output_hz: int = 200
    input_points: int = 250

    def __post_init__(self):
        n=len(self.channel_names)
        if len(set(x.upper() for x in self.channel_names))!=n:
            raise ValueError('Duplicate channel names')
        if self.center.shape!=(n,) or self.scale.shape!=(n,) or not np.isfinite(self.center).all() or not np.isfinite(self.scale).all() or (self.scale<=0).any():
            raise ValueError('Fixed T-only per-channel statistics are required')
        if (self.input_hz,self.output_hz,self.input_points)!=(250,200,250):
            raise ValueError('Only the explicitly defined 250-to-200Hz window is implemented')

    def __call__(self, eeg, *, channel_names):
        from scipy.signal import resample_poly
        values=np.asarray(eeg)
        if tuple(channel_names)!=self.channel_names:
            raise ValueError('Channel order must match the bound source metadata exactly')
        if values.ndim!=3 or values.shape[1:]!=(len(self.channel_names),250) or not np.isfinite(values).all():
            raise ValueError('Expected finite physical-R4 mean EEG BxCx250')
        standardized=(values.astype(np.float64)-self.center[None,:,None])/self.scale[None,:,None]
        result=resample_poly(standardized,4,5,axis=-1,padtype='constant')
        if result.shape[-1]!=200 or not np.isfinite(result).all():
            raise ValueError('Resampling contract failed')
        return result.astype(np.float32)[:,:,None,:]


def global_and_patch_conditions(encoder, eeg, *, input_chans):
    """One EEG-only author forward, retaining gradients and the exact pooled c.

The author's final pre-pooling normalization input is captured within that
same forward. No visual features are accepted or passed to the encoder.
"""
    if encoder.fc_norm is None or not isinstance(encoder.norm, torch.nn.Identity):
        raise ValueError('Only the audited author mean-pooling path is supported')
    if eeg.ndim!=4 or eeg.shape[-1]!=encoder.patch_size:
        raise ValueError('Input must be BxCxWindowsx200 for this EEG encoder')
    if len(input_chans)!=eeg.shape[1]+1 or len(set(input_chans))!=len(input_chans) or input_chans[0]!=0:
        raise ValueError('Real, unique electrode position indices including CLS are required')
    captured=[]
    handle=encoder.norm.register_forward_hook(lambda module,args,result:captured.append(result))
    try:
        global_eeg, image_loss, text_loss=encoder(eeg,input_chans=input_chans)
    finally:
        handle.remove()
    if len(captured)!=1 or image_loss is not None or text_loss is not None:
        raise RuntimeError('Unexpected author encoder path or visual side input')
    tokens=captured[0][:,1:]
    expected=eeg.shape[1]*eeg.shape[2]
    if tokens.shape[:2]!=(eeg.shape[0],expected):
        raise RuntimeError('Channel/time token coverage mismatch')
    patch_eeg=encoder.head(encoder.fc_norm(tokens))
    memory=global_eeg[:,None]+(patch_eeg-patch_eeg.mean(1,keepdim=True))
    return global_eeg,memory
