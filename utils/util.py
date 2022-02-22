import torch
import numpy as np
from hparams import hparams as hps

def mode(obj, model = False):
    if model and hps.is_cuda:
        obj = obj.cuda()
    elif hps.is_cuda:
        obj = obj.cuda(non_blocking = hps.pin_mem)
    return obj

def to_arr(var):
    return var.cpu().detach().numpy().astype(np.float32)

def get_mask_from_lengths(lengths, pad = False):
    max_len = torch.max(lengths).item()
    if pad and max_len%hps.n_frames_per_step != 0:
        max_len += hps.n_frames_per_step - max_len%hps.n_frames_per_step
        assert max_len%hps.n_frames_per_step == 0
    ids = torch.arange(0, max_len, out = torch.LongTensor(max_len))
    ids = mode(ids)
    mask = (ids < lengths.unsqueeze(1))
    return mask
