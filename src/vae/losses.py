import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, mus, log_sigmas):
        # bs x 2
        res = 0.5 * (mus.pow(2) + log_sigmas.exp() - log_sigmas - 1).sum(axis=1).mean()
        return res
    

class DoubleKLDivergence(nn.Module):
    def __init__(self):
      super().__init__()

    def forward(self, mus, log_sigmas):
        # bs x 2
        res = 0.5 * ((1 + 1/log_sigmas.exp()) * mus.pow(2) + log_sigmas.exp() + 1/log_sigmas.exp() - 2).sum(axis=1).mean()
        return res