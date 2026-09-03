import time
import numpy as np
import torch

def get_prior(N_SAMPLES_PER_CLASS, device):
    cls_probs = torch.tensor([cls_num / sum(N_SAMPLES_PER_CLASS) for cls_num in N_SAMPLES_PER_CLASS], device=device)
    prior_log = torch.log(cls_probs)
    prior_scaler = prior_log / prior_log.max() 
    return prior_log, prior_scaler