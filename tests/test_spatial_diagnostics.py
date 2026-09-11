import torch
from torch import nn
from torch.nn import functional as F

from taisp import DifferentiableISP
from taisp.analysis.spatial_diagnostics import norm_match, partition_gradients
from taisp.losses.spatial_clip import PatchDirectionLoss


def test_direction_preserving_norm_match_and_zero():
    g, ref = torch.tensor([3.,4.]), torch.tensor([0.,10.])
    torch.testing.assert_close(norm_match(g,ref), torch.tensor([6.,8.]))
    assert norm_match(torch.zeros(2),ref).norm() == 0


def test_object_background_gradients_sum_to_full_patch_gradient():
    class Encoder(nn.Module):
        def patch_features(self,x):
            return F.adaptive_avg_pool2d(x,(7,7)).flatten(2).transpose(1,2)
    x=torch.linspace(.2,.6,3*14*14).reshape(1,3,14,14)
    isp=DifferentiableISP()
    guidance=PatchDirectionLoss(Encoder(),torch.tensor([1.,.5,-.2]))
    support=torch.arange(49)<15
    go,gb=partition_gradients(x,isp,guidance,support)
    whole=torch.autograd.grad(guidance(x,isp(x)),isp.phi)[0]
    torch.testing.assert_close(go+gb,whole)
    assert go.norm()>0 and gb.norm()>0
