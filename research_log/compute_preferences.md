
Recorded: 2026-09-14T20:06:12.4944744+08:00
## User compute preference
Prefer the A6000 server GPU for all detector/CLIP forward passes, backpropagation, ISP Jacobian-vector products and model-bearing experiments. Keep file I/O, hashing and small frozen float64 statistics on CPU. Preserve each task's declared numerical protocol and RNG settings; GPU preference does not authorize changing those settings.
