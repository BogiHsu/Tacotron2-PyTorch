import torch
import numpy as np
from hparams import hparams as hps

def mode(obj):
	if hps.is_cuda:
		obj = obj.cuda()
	return obj

def to_var(tensor):
	var = torch.autograd.Variable(tensor)
	return mode(var)

def to_arr(var):
	return var.cpu().detach().numpy().astype(np.float32)

def get_mask_from_lengths(lengths):
	max_len = torch.max(lengths).item()
	if hps.is_cuda:
		ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
	else:
		ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
	mask = (ids < lengths.unsqueeze(1)).byte()
	return mask
