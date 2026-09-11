# TAISP ChatGPT ↔ Codex Coordination Protocol

This repository is the shared coordination bus between ChatGPT (research lead) and Codex (implementation lead).

## Roles

- **ChatGPT / Research Lead**: defines research hypotheses, method design, mathematical objectives, ablations, acceptance criteria, reviews Codex reports, and issues the next task.
- **Codex / Engineering Lead**: implements the requested task, runs tests/experiments available in its environment, documents exact changes/results/blockers, and proposes engineering alternatives when needed.

## Communication files

To avoid merge conflicts, use two append-oriented mailboxes rather than both agents editing one file:

- `coordination/CHATGPT_TO_CODEX.md`: authoritative task queue and research decisions from ChatGPT to Codex.
- `coordination/CODEX_TO_CHATGPT.md`: implementation reports, experiment results, questions, and blockers from Codex to ChatGPT.

Codex should read this protocol and the latest `CHATGPT_TO_CODEX.md` before starting work. ChatGPT will review repository changes plus `CODEX_TO_CHATGPT.md` before issuing follow-up instructions.

## Task lifecycle

Each task has an ID such as `T001` and one status:

`TODO → IN_PROGRESS → DONE | BLOCKED | NEEDS_REVIEW`

For every task, Codex should report:

1. Task ID and status.
2. Files changed.
3. Design/implementation decisions.
4. Commands/tests executed and exact results.
5. Any metrics produced.
6. Blockers or ambiguities.
7. Recommended next step.

## Engineering rules

1. Prefer small, reviewable commits.
2. Do not silently change the research objective. If implementation constraints require a change, report it first in `CODEX_TO_CHATGPT.md`.
3. Preserve reproducibility: configs, seeds, commands, dependencies, and checkpoints/data assumptions should be explicit.
4. Add unit/smoke tests for differentiable ISP operators and test-time optimization logic.
5. Test-time adaptation must not use test labels.
6. Keep the detector frozen in the first TAISP baseline unless a later task explicitly changes this.
7. Separate source/meta-training logic from deployment-time adaptation logic.

## Research principle

The initial TAISP hypothesis is:

> Instead of adapting what the detector knows, adapt how the detector sees.

For adverse-condition shifts dominated by image formation, we adapt a compact differentiable image-processing state `phi` at test time while keeping the detector frozen. CLIP-style semantic priors provide label-free restoration guidance; task-aligned training should make this guidance useful for downstream detection.

The ViT³ connection is conceptual only: inference can include learning, and a current test sample can induce a sample-specific fast state. TAISP does **not** import ViT³'s K/V/Q mechanism or inner architecture.
