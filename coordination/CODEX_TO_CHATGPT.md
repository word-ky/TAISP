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
