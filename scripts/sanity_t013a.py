"""Deterministic synthetic CPU gradient check, not a detection experiment."""
import argparse
import json
import platform
from pathlib import Path

import torch
from torch import nn

from taisp import DifferentiableISP
from taisp.models.parameter_predictor import ParameterPredictor
from taisp.training.initialization import adapt_from_initialization


class SyntheticLoss(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', torch.tensor(value))
        self.boxes = torch.ones(1, 4)

    def forward(self, original, enhanced):
        return (enhanced-self.value).square().mean()


def run_sanity():
    seed = 20260913
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x = torch.linspace(.1, .6, 3*9*11).reshape(1, 3, 9, 11)
    desired = x+.08
    predictor, isp = ParameterPredictor(), DifferentiableISP()
    detector, clip = SyntheticLoss(.9), SyntheticLoss(.4)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=.2)
    losses, head_gradient_norms = [], []
    for step in range(9):
        optimizer.zero_grad(set_to_none=True)
        result = adapt_from_initialization(x, isp, detector, clip, predictor(x))
        loss = (result.enhanced-desired).square().mean()
        losses.append(loss.item())
        if step < 8:
            loss.backward()
            head_gradient_norms.append(predictor.head.weight.grad.norm().item())
            optimizer.step()
    return {
        'purpose': 'Synthetic software/gradient sanity only; no detection evidence.',
        'seed': seed, 'device': 'cpu', 'threads': 1, 'python': platform.python_version(),
        'torch': torch.__version__, 'inner_steps': 3, 'inner_lr': .1, 'eps': 1e-12,
        'approximation': 'FOMAML-style detached update; identity state Jacobian',
        'outer_optimizer': 'SGD', 'outer_lr': .2, 'outer_steps': 8,
        'input': 'linspace(.1,.6,3*9*11).reshape(1,3,9,11)',
        'target': 'input + .08', 'outer_loss': 'mean squared pixel error',
        'losses': losses, 'before': losses[0], 'after': losses[-1],
        'head_gradient_norms': head_gradient_norms,
        'decreased': losses[-1] < losses[0],
        'isp_owned_state_unchanged': bool(torch.equal(isp.phi, torch.zeros(8))),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = run_sanity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    assert receipt['decreased'] and receipt['isp_owned_state_unchanged']
