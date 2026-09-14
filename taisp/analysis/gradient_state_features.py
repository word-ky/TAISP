"""The exact R043 pre-update 21-D feature definition; no models or labels."""
import math

EPS = 1e-12


def features(gp, gc, pseudo_loss, clip_loss):
    np = math.sqrt(math.fsum(v*v for v in gp))
    nc = math.sqrt(math.fsum(v*v for v in gc))
    up = [v/(np+EPS) for v in gp]
    uc = [v/(nc+EPS) for v in gc]
    return up + uc + [math.log(np+EPS), math.log(nc+EPS),
                     math.fsum(a*b for a,b in zip(up,uc)), pseudo_loss, clip_loss]
