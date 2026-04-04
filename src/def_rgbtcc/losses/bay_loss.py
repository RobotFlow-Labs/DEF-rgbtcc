"""Bayesian loss for crowd counting."""
import torch
from torch.nn import Module


class Bay_Loss(Module):
    def __init__(self, use_background: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = torch.tensor(0.0, device=self.device)
        for idx, prob in enumerate(prob_list):
            if prob is None:
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(
                    pre_density[idx].view((1, -1)) * prob, dim=1
                )
            loss += torch.sum(torch.abs(target - pre_count))
        return loss / len(prob_list)
