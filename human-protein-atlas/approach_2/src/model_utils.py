"""
Based on rebryk's script from here:
https://github.com/rebryk/kaggle/tree/master/human-protein/

contain methods useful for model finetuning

"""

import itertools
import random
from typing import Dict, List

import numpy as np
import torch


def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def freeze(model: torch.nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False

def freeze_till_last(model: torch.nn.Module):
    model.model.last_linear.weight.requires_grad=True
    model.model.last_linear.bias.requires_grad=True
    model.l0.weight.requires_grad=True
    model.l0.bias.requires_grad=True
    model.l1.weight.requires_grad=True
    model.l1.bias.requires_grad=True
    model.l2.weight.requires_grad=True
    model.l2.bias.requires_grad=True
    model.l3.weight.requires_grad=True
    model.l3.bias.requires_grad=True
    model.l4.weight.requires_grad=True
    model.l4.bias.requires_grad=True
    model.l5.weight.requires_grad=True
    model.l5.bias.requires_grad=True
    model.l6.weight.requires_grad=True
    model.l6.bias.requires_grad=True
    model.l7.weight.requires_grad=True
    model.l7.bias.requires_grad=True
    model.l8.weight.requires_grad=True
    model.l8.bias.requires_grad=True
    model.l9.weight.requires_grad=True
    model.l9.bias.requires_grad=True
    model.l10.weight.requires_grad=True
    model.l10.bias.requires_grad=True
    model.l11.weight.requires_grad=True
    model.l11.bias.requires_grad=True
    model.l12.weight.requires_grad=True
    model.l12.bias.requires_grad=True
    model.l13.weight.requires_grad=True
    model.l13.bias.requires_grad=True
    model.l14.weight.requires_grad=True
    model.l14.bias.requires_grad=True
    model.l15.weight.requires_grad=True
    model.l15.bias.requires_grad=True
    model.l16.weight.requires_grad=True
    model.l16.bias.requires_grad=True
    model.l17.weight.requires_grad=True
    model.l17.bias.requires_grad=True
    model.l18.weight.requires_grad=True
    model.l18.bias.requires_grad=True
    model.l19.weight.requires_grad=True
    model.l19.bias.requires_grad=True
    model.l20.weight.requires_grad=True
    model.l20.bias.requires_grad=True
    model.l21.weight.requires_grad=True
    model.l21.bias.requires_grad=True
    model.l22.weight.requires_grad=True
    model.l22.bias.requires_grad=True
    model.l23.weight.requires_grad=True
    model.l23.bias.requires_grad=True
    model.l24.weight.requires_grad=True
    model.l24.bias.requires_grad=True
    model.l25.weight.requires_grad=True
    model.l25.bias.requires_grad=True
    model.l26.weight.requires_grad=True
    model.l26.bias.requires_grad=True
    model.l27.weight.requires_grad=True
    model.l27.bias.requires_grad=True


def unfreeze(model: torch.nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)


def load_checkpoint(filename: str) -> Dict:
    return torch.load(filename)


def get_group_params(groups: List[List[torch.nn.Module]], lrs: np.ndarray) -> List[Dict]:
    """Create dicts defining parameter groups for an optimizer."""
    group_params = []

    for group, lr in zip(groups, lrs):
        params = {'params': list(itertools.chain(*[layer.parameters() for layer in group])), 'lr': lr}
        group_params.append(params)

    return group_params
