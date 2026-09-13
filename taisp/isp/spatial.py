"""Pure deployment operations ported unchanged from T014 spatial_action."""
import math

import torch


def support_mask(image, boxes):
    height,width=image.shape[-2:]
    mask=image.new_zeros((1,1,height,width));rectangles=[]
    for x1,y1,x2,y2 in boxes.detach().cpu().tolist():
        left,top=math.floor(max(0,min(width,x1))),math.floor(max(0,min(height,y1)))
        right,bottom=math.ceil(max(0,min(width,x2))),math.ceil(max(0,min(height,y2)))
        mask[:,:,top:bottom,left:right]=1
        rectangles.append([left,top,right,bottom])
    return mask.detach(),rectangles


def compose(image, isp, mask, phi_obj, phi_bg):
    mask=mask.detach()
    return mask*isp(image,phi_obj)+(1-mask)*isp(image,phi_bg)


def regional_gradients(image, isp, mask, states, cotangents):
    """Exact ISP JVP chain rule, with A1 float64 masked reductions on-device."""
    x=image.detach()
    gradients={name:[] for name in cotangents}
    cs={name:c.detach().double() for name,c in cotangents.items()}
    for region,state in enumerate(states):
        m=(mask if region==0 else 1-mask).detach().double()
        values={name:[] for name in cs}
        for k in range(8):
            tangent=torch.zeros_like(state);tangent[k]=1
            _,column=torch.func.jvp(lambda p:isp(x,p),(state.detach(),),(tangent,))
            jd=column.detach().double()
            for name,c in cs.items():
                values[name].append((c*m*jd).sum())
        for name in cs:gradients[name].append(torch.stack(values[name]))
    return {name:torch.stack(v) for name,v in gradients.items()}
