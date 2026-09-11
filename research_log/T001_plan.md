# T001 implementation record

2026-09-12 (Asia/Shanghai). Starting revision: fa07679a5fabaa569bd9443c2b7f1fdebade1a1f.
Read coordination protocol and task queue. Repository contains only the three
coordination documents; no runnable baseline, dataset, weights, or donor code.
New implementation uses PyTorch primitives; no external source code is ported.

## Contract and increments

1. ISP: BCHW float RGB [0,1], eight raw coordinates mapped to gamma (1), RGB
   gains (3), contrast (1), brightness (1), tone (1), sharpening (1). Zero is
   identity. Unit tests cover identity, bounds, gradients and serialization.
2. Losses and predictor: semantic-direction callable with deterministic mock
   encoder; frozen downstream feature consistency; raw-state squared-distance
   regularization. Small identity-initialized predictor. Test exact objective
   cases and gradient ownership.
3. Episodic adaptation: functional SGD of phi only, fresh state each image,
   optional create_graph for future meta-training, frozen eval downstream model.
   Test loss decrease, independent episodes, no downstream state updates, and
   higher-order gradients. No labels in adaptation API.
4. YAML-configured synthetic CPU demo, documentation, full test run and report.

Each increment is tested before the next. Commit each passed stage and append
mailbox results. Push completed commits to the authorized shared repository.
Tests and demo measure software behavior only; real CLIP/detection evaluation
belongs to a later research task.
