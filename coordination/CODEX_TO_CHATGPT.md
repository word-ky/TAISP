# Codex → ChatGPT

Codex: append implementation reports here using the template below. Do not overwrite prior reports unless correcting an error; preserve history so ChatGPT can reconstruct decisions.

## Report template

### TXXX — <task title>

**Status:** IN_PROGRESS | DONE | BLOCKED | NEEDS_REVIEW

**Commit(s):** <sha(s)>

**Files changed**
- ...

**Implementation decisions**
- ...

**Commands / tests / results**
```text
...
```

**Metrics / observations**
- ...

**Questions / blockers**
- ...

**Recommended next step**
- ...

---

No implementation report yet.

### T001 — Bootstrap started (2026-09-12)

**Status:** IN_PROGRESS

Read PROTOCOL.md and T001 at fa07679a. No baseline source exists to reuse.
Implementing bounded eight-coordinate differentiable ISP, pluggable mock
semantic guidance, frozen downstream consistency, functional episodic SGD,
tests and synthetic demo. CLIP weights and real detection metrics are out of
this bootstrap's scope. Detailed implementation sequence: research_log/T001_plan.md.

Stage 1 passed: python -m pytest tests/test_isp.py -q -> 4 passed in 15.55s. Eight bounded raw coordinates; zero identity; finite-difference gradient verification and serialization passed.

Stage 2 passed: python -m pytest tests/test_losses.py -q -> 4 passed in 19.11s. Mock semantic direction is negative projection of feature displacement (not a CLIP result); consistency detaches its reference; regularization is squared L2; predictor starts at identity.

Stage 3 passed: python -m pytest tests/test_adapt.py -q -> 5 passed in 12.94s. Functional SGD, episode reset, frozen downstream parameters and buffers, final enhanced-image prediction, consistency-only gradient, and two-step unroll finite-difference verification passed. User authorized A6000 execution; remote workspace created under wjq/TAISP.
