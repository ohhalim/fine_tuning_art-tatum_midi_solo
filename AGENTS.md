# AGENTS.md

## Project Focus

This repository is currently focused on Stage B symbolic MIDI generation probes.

Primary goal:

- Build a reliable local MIDI generation and validation pipeline.
- Prove model behavior with small, reproducible experiments before expanding scope.

Current active branch scope:

- Issue #20: Stage B grammar-constrained tiny-overfit.

Do not expand into Spring Boot, realtime DAW/plugin work, SaaS, UI, or deployment unless the user explicitly asks for that new scope.

## Autonomy Rules

The agent may continue working within the active issue scope without asking for every small step.

Allowed without additional permission:

- inspect files and git status
- edit repo files
- run local tests and local scripts
- create local commits after a coherent validated change
- update docs that describe the current implementation or experiment result

Must ask first:

- any `git push`, including same-branch upstream push
- pull request creation or PR metadata updates
- GitHub issue creation or issue metadata updates
- deployment
- external uploads
- destructive cleanup of generated files, checkpoints, datasets, or user-created outputs

If the user says "PR 올려", "이슈 만들어", "배포해", or equivalent, that counts as explicit permission for that requested remote action only.

## Local Commit Policy

Local commits are allowed when all are true:

- the change belongs to the active issue or explicitly requested task
- the working tree does not include unrelated user changes
- relevant validation commands pass, or failures are documented in the final response
- the commit message is specific and conventional

Preferred commit prefixes:

- `docs:`
- `test:`
- `fix:`
- `feat:`
- `chore:`

Do not squash or rewrite previous commits unless the user explicitly asks.

## Required Harness

Before local commits, run the smallest relevant harness mode:

```bash
bash scripts/agent_harness.sh quick
```

For changes that touch inference behavior, metrics, generation, or model loading, also run:

```bash
bash scripts/agent_harness.sh demo
```

For Stage A training-mode changes, run:

```bash
bash scripts/agent_harness.sh tiny-compare
```

For Stage B window dataset/model-vocab changes, run:

```bash
bash scripts/agent_harness.sh stage-b-window-prepare
```

For Stage B decode/generation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generation-probe
```

For Stage B constrained note-grammar changes, run:

```bash
bash scripts/agent_harness.sh stage-b-constrained-probe
```

If a harness mode is too slow or fails for an environment reason, record the reason clearly in the final answer.

## Quality Gate

Never treat a MIDI file as successful just because a `.mid` file exists.

At minimum, check:

- non-zero note count
- enough unique pitches
- phrase coverage for requested bars
- max note duration ratio
- max simultaneous notes
- dead-air ratio where applicable
- fallback usage

One-note, two-note, repeated single-pitch, long sustain block, or chord-block outputs are invalid model outputs for Stage A review.

## Documentation Policy

Docs must stay honest about model quality.

Use wording like:

- "Stage A symbolic MIDI generation prototype"
- "model-serving and validation pipeline"
- "tiny-overfit smoke"
- "full-checkpoint/from-scratch training path"

Avoid claiming:

- reliable personalized jazz model
- production-ready improviser
- LoRA fine-tuned jazz model, unless a pretrained base and adapter training are actually validated

## Branch Hygiene

Keep one branch focused on one issue when possible.

If a task grows beyond the active issue, stop and propose a new issue/branch boundary before implementing unrelated work.

Generated artifacts under `outputs/`, `samples/`, `checkpoints/`, and raw MIDI datasets are not committed unless the user explicitly asks.
