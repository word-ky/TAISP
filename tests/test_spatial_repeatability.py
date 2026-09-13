import gzip
import pytest
import torch
from taisp import DifferentiableISP
from taisp.analysis.spatial_repeatability import CotangentRecorder,compare,pairs,save_tensors
from taisp.tta.spatial_dose import adapt_spatial_dose
from taisp.tta.trust_radius import adapt_clip_radius
from test_spatial_dose import UniformLoss


def test_pairwise_metrics_exact_and_undefined_cosine():
    a=torch.tensor([1.,2.]);b=torch.tensor([1.,3.]);z=torch.zeros(2)
    assert compare(a,a)['exact'] and compare(a,b)['max_absolute']==1
    assert compare(z,z)['cosine']==1 and compare(z,a)['cosine'] is None
    assert len(pairs([{'x':a},{'x':b},{'x':z}]))==3


@pytest.mark.parametrize('spatial',[False,True])
def test_hooks_capture_original_loss_cotangents_without_changing_runtime(spatial):
    x=torch.full((1,3,4,6),.4,dtype=torch.double);isp=DifferentiableISP().double()
    pseudo,clip=UniformLoss(),UniformLoss();clip.weight.data.fill_(.3)
    mask=x.new_zeros((1,1,4,6));mask[:,:,:,:3]=1
    def run():return adapt_spatial_dose(x,isp,pseudo,clip,mask) if spatial else adapt_clip_radius(x,isp,pseudo,clip)
    baseline=run();rec=CotangentRecorder(pseudo,clip)
    try:observed=run()
    finally:rec.close()
    assert torch.equal(baseline.phi,observed.phi) and torch.equal(baseline.enhanced,observed.enhanced)
    assert len(rec.records['pseudo'])==len(rec.records['clip'])==4
    for name,weight in [('pseudo',.2),('clip',.3)]:
        torch.testing.assert_close(rec.records[name][0],2*(isp(x,x.new_zeros(8))-weight)/x.numel(),rtol=1e-12,atol=1e-12)
    assert not pseudo._forward_hooks and not clip._forward_hooks


def test_lossless_raw_tensor_receipt(tmp_path):
    x=torch.arange(8.).reshape(2,4);path=tmp_path/'x.pt.gz';save_tensors(path,{'x':x})
    with gzip.open(path,'rb') as f:r=torch.load(f,weights_only=True)
    assert torch.equal(r['x'],x)
