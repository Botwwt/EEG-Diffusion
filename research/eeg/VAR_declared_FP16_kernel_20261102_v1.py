"""Only decoder operations autocast. VQ and probability arithmetic stay FP32."""
import torch
from EEG_native_marginal_core_20261029_v2 import NativeMarginalDecoder
class DeclaredFP16Native(NativeMarginalDecoder):
 def forward(self,features,condition):
  with torch.autocast('cuda',dtype=torch.float16,enabled=True):value=super().forward(features,condition)
  return value.float()
 def marginal_logits(self,features,condition,tokens=5):
  with torch.autocast('cuda',dtype=torch.float16,enabled=True):value=super().marginal_logits(features,condition,tokens=tokens)
  return value.float()
