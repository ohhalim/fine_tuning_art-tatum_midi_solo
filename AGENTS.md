# AGENTS.md

## Project Focus

This repository is currently focused on a Stage B MIDI-to-solo generation MVP.

Primary goal:

- Build a reliable local pipeline that accepts an input MIDI file and exports ranked jazz solo MIDI candidates.
- Keep the implementation hybrid: model-conditioned generation, constrained decoding, candidate ranking, and retrieval fallback.
- Prove behavior with small, reproducible experiments before broad training or Brad style adaptation claims.

Current handoff scope:

- Latest functional issue completed: Issue #982, Stage B MIDI-to-solo MVP current evidence consolidation source-context refresh.
- Latest audit issue completed: Issue #986, Stage B MIDI-to-solo MVP completion audit source-context refresh.
- Latest quality gap decision completed: Issue #988, Stage B MIDI-to-solo quality gap decision source-context refresh.
- Latest listening review quality gap completed: Issue #990, Stage B MIDI-to-solo listening review quality gap source-context refresh.
- Latest MVP delivery package completed: Issue #992, Stage B MIDI-to-solo MVP delivery package source-context refresh.
- Latest README final evidence refresh completed: Issue #994, Stage B MIDI-to-solo README final evidence source-context refresh.
- Latest final status audit completed: Issue #996, Stage B MIDI-to-solo final status audit source-context refresh.
- Latest post-MVP quality iteration plan completed: Issue #998, Stage B MIDI-to-solo post-MVP quality iteration plan source-context refresh.
- Latest quality rubric baseline completed: Issue #1000, Stage B MIDI-to-solo quality rubric baseline source-context refresh.
- Latest candidate failure labeling completed: Issue #1002, Stage B MIDI-to-solo candidate failure labeling source-context refresh.
- Latest targeted quality repair sweep completed: Issue #1004, Stage B MIDI-to-solo targeted quality repair sweep source-context refresh.
- Latest targeted quality repair audio package completed: Issue #1006, Stage B MIDI-to-solo targeted quality repair audio package source-context refresh.
- Latest targeted quality repair listening review package completed: Issue #1008, Stage B MIDI-to-solo targeted quality repair listening review package source-context refresh.
- Latest targeted quality repair listening review input guard completed: Issue #1010, Stage B MIDI-to-solo targeted quality repair listening review input guard source-context refresh.
- Latest targeted quality repair objective-only next decision completed: Issue #1012, Stage B MIDI-to-solo targeted quality repair objective-only next decision source-context refresh.
- Latest targeted quality repair follow-up decision completed: Issue #1014, Stage B MIDI-to-solo targeted quality repair follow-up decision source-context refresh.
- Latest songlike melody contour repair sweep completed: Issue #1016, Stage B MIDI-to-solo songlike melody contour repair sweep source-context refresh.
- Latest songlike melody contour repair audio package completed: Issue #1018, Stage B MIDI-to-solo songlike melody contour repair audio package source-context refresh.
- Latest songlike melody contour repair listening review package completed: Issue #1020, Stage B MIDI-to-solo songlike melody contour repair listening review package source-context refresh.
- Latest songlike melody contour repair listening review input guard completed: Issue #1022, Stage B MIDI-to-solo songlike melody contour repair listening review input guard source-context refresh.
- Latest songlike melody contour repair objective-only next decision completed: Issue #1024, Stage B MIDI-to-solo songlike melody contour repair objective-only next decision source-context refresh.
- Latest songlike melody contour repair follow-up decision completed: Issue #1026, Stage B MIDI-to-solo songlike melody contour repair follow-up decision source-context refresh.
- Latest songlike melody contour phrase/rhythm repair sweep completed: Issue #1028, Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair sweep source-context refresh.
- Latest songlike melody contour phrase/rhythm repair audio package completed: Issue #946, Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair audio package source-context refresh.
- Latest songlike melody contour phrase/rhythm repair listening review package completed: Issue #948, Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair listening review package source-context refresh.
- Latest songlike melody contour phrase/rhythm repair listening review input guard completed: Issue #950, Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair listening review input guard source-context refresh.
- Latest songlike melody contour phrase/rhythm repair objective-only next decision completed: Issue #952, Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair objective-only next decision source-context refresh.
- Latest songlike melody contour phrase/rhythm repair follow-up decision completed: Issue #954, Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair follow-up decision source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-context pitch-role bridge completed: Issue #956, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role bridge source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-context pitch-role objective decision completed: Issue #958, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role objective decision source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing repair sweep completed: Issue #960, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair sweep source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing repair audio package completed: Issue #962, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair audio package source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing repair listening review package completed: Issue #964, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair listening review package source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing repair listening review input guard completed: Issue #966, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair listening review input guard source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing repair objective-only next decision completed: Issue #968, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair objective-only next decision source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing repair follow-up decision completed: Issue #970, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair follow-up decision source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair sweep completed: Issue #972, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair sweep source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair audio package completed: Issue #974, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair audio package source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair listening review package completed: Issue #976, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair listening review package source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair listening review input guard completed: Issue #978, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair listening review input guard source-context refresh.
- Latest songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair objective-only next decision completed: Issue #980, Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair objective-only next decision source-context refresh.
- Latest documentation issue completed: Issue #984, Stage B MIDI-to-solo README evidence source-context refresh.
- Current branch should be `main` before starting new work.
- Open issue queue after post-MVP quality iteration plan source-context refresh merge: `0`.
- Recommended next issue: Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair audio package source-context refresh.

Do not expand into Spring Boot, realtime DAW/plugin work, SaaS, UI, or deployment unless the user explicitly asks for that new scope.

## Autonomy Rules

The agent may continue working within the active issue scope without asking for every small step.

Allowed without additional permission:

- inspect files and git status
- edit repo files
- run local tests and local scripts
- create local commits after a coherent validated change
- update docs that describe the current implementation or experiment result
- create GitHub issues that follow `docs/CORE_PLAN.md`
- create focused issue branches from latest `main`
- push agent-created issue branches
- create PRs for agent-created issue branches
- merge agent-created PRs when the Conditional Auto-Merge Policy is satisfied

Must ask first:

- merging pull requests that were not created by the agent in the current task flow
- deployment
- external uploads
- destructive cleanup of generated files, checkpoints, datasets, or user-created outputs
- raw dataset, checkpoint, or generated artifact upload
- cloud/GPU spend
- secrets, credentials, or repository settings changes
- merge conflict resolution that rewrites or discards user work

If the user says "자동으로 진행해", "플랜대로 계속해", or equivalent, that counts as permission to repeat the issue -> branch -> commit -> push -> PR -> safe auto-merge loop within `docs/CORE_PLAN.md` until a critical blocker appears.

If the host UI still asks for tool approval, that is a runtime permission gate outside this repo policy. Within this repository policy, plan-scoped GitHub issue/branch/PR/merge work is already allowed without re-asking the user.

## Public Naming Rules

Keep public GitHub artifacts project-focused and tool-neutral.

- Do not include implementation tool names in branch names, issue titles, issue bodies, PR titles, PR bodies, or commit messages unless the user explicitly asks.
- Do not use PR title prefixes that identify the implementation tool.
- Use neutral branch names such as `issue-123-short-topic`, `docs/short-topic`, `feat/short-topic`, or `fix/short-topic`.
- Use issue and PR titles that describe the project change directly, for example `Stage B phrase duration repair` or `포트폴리오용 README 정리`.
- Keep merge reports and final summaries focused on issue number, PR number, merge commit, changed files, and validation commands.

## Public Record Style

Write public records like a production engineering handoff.

- Use structured sections when the artifact is more than a trivial change: `Summary`, `Context`, `Scope`, `Validation`, `Risk`, and `Follow-up`.
- State the user-visible or project-visible behavior first, then implementation details.
- Record the exact validation commands that were run, including environment/tooling failures when they happen.
- Keep decisions evidence-based: include measured metrics, reviewed artifacts, issue links, and remaining tradeoffs.
- Avoid vague labels such as `update`, `misc`, or `cleanup` when a specific project outcome can be named.
- Keep commit subjects conventional and specific, with Korean bodies for context, changes, and validation when the change is non-trivial.
- Do not claim final musical quality from proxy or objective checks alone; document it as a review boundary and name the next review step.

## Conditional Auto-Merge Policy

The agent may merge a pull request without asking again only when all of the following are true:

- the user explicitly asked the agent to continue through PR completion for the current issue, or explicitly allowed automatic merge behavior
- the pull request was created by the agent for the current issue branch
- the pull request targets `main`
- the PR is marked mergeable by GitHub
- the working branch contains only changes scoped to the current issue
- all relevant local validation commands passed, or the only failures are documented environment/tooling issues that do not affect the change
- the PR has no unresolved requested changes or review blockers known to the agent
- the merge method is the repository default merge method unless the user specified another method

The agent must not auto-merge when any of the following are true:

- the user says "내가 머지할게", "머지 전에는 불러", "merge는 내가 할게", or equivalent
- the PR includes deployment, credentials, destructive cleanup, raw dataset/checkpoint uploads, or external publishing
- GitHub reports merge conflicts or an unknown/unmergeable state after a reasonable refresh
- validation failed for reasons related to the changed code
- the diff includes unrelated user changes
- the PR was opened by someone else or predates the current agent task flow

After an automatic merge, the agent should report:

- issue number
- PR number and URL
- merge commit SHA when available
- validation commands run
- next recommended issue or branch boundary

## Local Commit Policy

Local commits are allowed when all are true:

- the change belongs to the active issue or explicitly requested task
- the working tree does not include unrelated user changes
- relevant validation commands pass, or failures are documented in the final response
- the commit message is specific and conventional

Commit cadence:

- Prefer frequent commits at small validated boundaries instead of one large end-of-issue commit.
- Commit after each coherent unit such as tests added, implementation wired, docs updated, or harness result recorded.
- Do not create noisy checkpoint commits for broken or unvalidated work unless the user explicitly asks for a work-in-progress snapshot.
- When a PR is already open, push additional focused commits to the same issue branch rather than amending or squashing history.

Commit messages:

- Write commit messages in Korean unless the user asks otherwise.
- Keep the prefix conventional, but make the subject specific enough to explain the change.
- Prefer messages like `feat: Stage B strict gate 결과를 sweep summary에 연결` over vague messages like `update` or `fix`.
- For larger changes, use a multi-line commit body that records why the change was needed and which validation was run.

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

For Stage B MIDI-to-solo post-MVP quality iteration plan changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-post-mvp-quality-iteration-plan
```

This harness selects the quality rubric baseline after technical MVP completion,
preserves source/current outside-soloing repair context, and avoids musical
quality or preference claims.

For Stage B MIDI-to-solo quality rubric baseline changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-quality-rubric-baseline
```

This harness prepares candidate failure labeling, preserves source/current
outside-soloing repair context, and avoids musical quality or preference claims.

For Stage B MIDI-to-solo candidate failure labeling changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-candidate-failure-labeling
```

This harness labels current MIDI candidates against the quality rubric, preserves
source/current outside-soloing repair context, and avoids musical quality or
preference claims.

For Stage B MIDI-to-solo targeted quality repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-targeted-quality-repair-sweep
```

This harness runs the targeted repair sweep, preserves source/current
outside-soloing repair context, and avoids musical quality or preference claims.

For Stage B MIDI-to-solo targeted quality repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-targeted-quality-repair-audio-package
```

This harness renders targeted repair MIDI candidates to WAV, preserves
source/current outside-soloing repair context, and avoids audio quality or
preference claims.

For Stage B MIDI-to-solo targeted quality repair listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-targeted-quality-repair-listening-review-package
```

This harness packages targeted repair WAV/MIDI candidates for listening review,
preserves source/current outside-soloing repair context, and avoids preference
or musical quality claims.

For Stage B MIDI-to-solo targeted quality repair listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-targeted-quality-repair-listening-review-input-guard
```

This harness blocks preference fill while validated listening review input is
pending, preserves source/current outside-soloing repair context, and avoids
preference or musical quality claims.

For Stage B MIDI-to-solo targeted quality repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-targeted-quality-repair-objective-only-next-decision
```

This harness routes pending listening review input to the next objective-only
follow-up boundary, preserves source/current outside-soloing repair context,
and avoids preference or musical quality claims.

For Stage B MIDI-to-solo targeted quality repair follow-up decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-targeted-quality-repair-followup-decision
```

This harness selects the dominant remaining repair target, preserves
source/current outside-soloing repair context, and avoids preference or musical
quality claims.

For Stage B MIDI-to-solo songlike melody contour repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-repair-sweep
```

This harness reduces songlike melody labels, preserves source/current
outside-soloing repair context, and avoids preference or musical quality
claims.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-repair-sweep
```

This harness checks whether phrase/rhythm failure labels are reduced without claiming listening or musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-repair-audio-package
```

This harness renders phrase/rhythm repair MIDI candidates to WAV files and validates technical metadata without claiming listening or musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-repair-listening-review-package
```

This harness packages phrase/rhythm repair WAV/MIDI candidates for listening review without claiming preference or musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-repair-listening-review-input-guard
```

This harness blocks preference fill while listening review input is pending and keeps quality claims false.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-repair-objective-only-next-decision
```

This harness routes pending listening review input to the next objective-only follow-up boundary without claiming quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair follow-up decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-repair-followup-decision
```

This harness selects the chord-context pitch-role bridge when phrase/rhythm repair leaves outside-soloing and chord-tone landing labels not evaluable.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role bridge changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-context-pitch-role-bridge
```

This harness attaches fallback chord context to phrase/rhythm repair candidates and records pitch-role metrics without claiming musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role objective decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-context-pitch-role-objective-decision
```

This harness selects the next repair target from chord-context pitch-role objective evidence without claiming musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-repair-sweep
```

This harness repairs final landing and strong-beat chord-tone roles for phrase/rhythm candidates without claiming musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-repair-audio-package
```

This harness renders chord-tone landing repair MIDI candidates to WAV files and validates technical metadata without claiming listening or musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-repair-listening-review-package
```

This harness packages chord-tone landing repair WAV/MIDI candidates for listening review while keeping preference and quality claims false.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-repair-listening-review-input-guard
```

This harness blocks chord-tone landing repair preference fill while listening review input is pending and routes to objective-only next decision.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-repair-objective-only-next-decision
```

This harness routes pending review input and residual outside-soloing pitch-role risk to the next follow-up decision without claiming musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair follow-up decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-repair-followup-decision
```

This harness selects residual outside-soloing pitch-role repair after chord-tone landing risk is resolved without claiming musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-outside-soloing-repair-sweep
```

This harness reduces residual outside-soloing pitch-role risk after chord-tone landing repair and records source/current risk context without claiming musical quality.

For Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-songlike-melody-contour-phrase-rhythm-chord-tone-landing-outside-soloing-repair-audio-package
```

This harness renders outside-soloing repair MIDI candidates to WAV files and validates technical metadata without claiming listening or musical quality.

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

For Stage B overlap/dedup gate changes, run:

```bash
bash scripts/agent_harness.sh stage-b-overlap-gate
```

For Stage B multi-sample review-gate changes, run:

```bash
bash scripts/agent_harness.sh stage-b-stronger-probe
```

For Stage B 8-bar approach/passing-note phrase changes, run:

```bash
bash scripts/agent_harness.sh stage-b-8bar-approach-phrase
```

For Stage B swing/motif rhythm phrase changes, run:

```bash
bash scripts/agent_harness.sh stage-b-swing-motif-phrase
```

For Stage B real phrase reference statistics changes, run:

```bash
bash scripts/agent_harness.sh stage-b-reference-stats
```

For Stage B reference pitch-role landing statistics changes, run:

```bash
bash scripts/agent_harness.sh stage-b-reference-pitch-roles
```

For chord progression annotation coverage audit changes, run:

```bash
bash scripts/agent_harness.sh chord-coverage-audit
```

For Stage B chord-labeled evaluation subset changes, run:

```bash
bash scripts/agent_harness.sh stage-b-chord-labeled-eval
```

For Stage B generated candidate chord-labeled eval bridge changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generated-chord-eval
```

For Stage B data-guide hybrid generated chord eval changes, run:

```bash
bash scripts/agent_harness.sh stage-b-data-guide-generated-chord-eval
```

For Stage B review markdown chord eval summary changes, run:

```bash
bash scripts/agent_harness.sh stage-b-review-markdown-chord-eval
```

For Stage B listening review notes schema changes, run:

```bash
bash scripts/agent_harness.sh stage-b-listening-review-notes
```

For Stage B full review manifest listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-full-review-notes
```

For Stage B objective MIDI note review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-objective-midi-review
```

For Stage B objective flags review flow changes, run:

```bash
bash scripts/agent_harness.sh stage-b-objective-flags-review-flow
```

For Stage B overlap-free solo-line review export changes, run:

```bash
bash scripts/agent_harness.sh stage-b-overlap-free-review-export
```

For Stage B duration variation review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-variation-review
```

For Stage B phrase/cadence review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-phrase-cadence-review
```

For Stage B phrase naturalness objective metric changes, run:

```bash
bash scripts/agent_harness.sh stage-b-objective-midi-review
bash scripts/agent_harness.sh stage-b-phrase-cadence-review
```

For Stage B phrase recovery review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-phrase-recovery-review
```

For Stage B data motif phrase recovery review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-phrase-recovery-review
```

For Stage B objective-clean review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-clean-review-package
```

For Stage B proxy-keep focused review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-proxy-keep-focused-package
```

For Stage B clean context diagnostics changes, run:

```bash
bash scripts/agent_harness.sh stage-b-clean-context-diagnostics
```

For Stage B filled listening review aggregate changes, run:

```bash
bash scripts/agent_harness.sh stage-b-listening-review-aggregate
```

For Stage B data-derived motif template extraction changes, run:

```bash
bash scripts/agent_harness.sh stage-b-motif-templates
```

For Stage B data-derived motif baseline generation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-compare
```

For Stage B data motif review export changes, run:

```bash
bash scripts/agent_harness.sh stage-b-data-motif-review-export
```

For Stage B review MIDI chord-context/straight-grid changes, run:

```bash
bash scripts/agent_harness.sh stage-b-review-context-grid
```

For Stage B straight-grid guide-tone/cadence candidate changes, run:

```bash
bash scripts/agent_harness.sh stage-b-guide-tone-cadence
```

For Stage B data-motif rhythm plus guide-tone/cadence pitch hybrid changes, run:

```bash
bash scripts/agent_harness.sh stage-b-data-guide-hybrid
```

For Stage B collapse/sampling-sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-collapse-sweep
```

For Stage B strict collapse-aware review-gate changes, use the same sweep harness and verify
`passed_strict_sweep_gate` in the generated report.

For Stage B 2-file Brad generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-2file-brad-probe
```

This probe records basic/strict pass-rate. A musical quality failure is a report outcome, not
automatically a harness failure, unless the script itself crashes or produces no grammar-valid samples.

For Stage B coverage-aware constrained generation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-coverage-aware-probe
```

This probe tests whether constrained `POSITION` selection can improve temporal coverage without
claiming unconstrained model quality.

For Stage B coverage-aware A/B sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-coverage-ab-sweep
```

This probe compares plain constrained generation against coverage-aware constrained generation
across note-group density settings.

For Stage B candidate ranking changes, run:

```bash
bash scripts/agent_harness.sh stage-b-candidate-ranking
```

This harness generates an A/B sweep and ranks generated MIDI candidates for listening/review priority.

For Stage B chord-aware pitch constrained generation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-chord-aware-probe
```

This harness compares plain, coverage-aware, and coverage+chord-aware constrained generation, then ranks the generated MIDI candidates.

For Stage B longer phrase-generation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```

This harness tests a 4-bar coverage+chord-aware constrained phrase and exports the top review MIDI candidates. It exists because short 2-bar candidates can be valid but still feel unfinished.

For Stage B margin-recovered timing/repetition repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-repair
```

This harness tests whether the selected pitch-vocabulary candidate can keep focused unique pitch coverage while reducing dead-air and adjacent pitch repetition.

For Stage B margin-recovered timing/repetition focused context changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-focused-context
```

This harness packages the selected timing/repetition repair candidate with chord/bass context and verifies focused context decision readiness.

For Stage B margin-recovered timing/repetition focused listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-focused-listening-notes
```

This harness writes the focused listening review notes template for the selected timing/repetition context keep candidate.

For Stage B margin-recovered timing/repetition focused listening fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-focused-listening-fill
```

This harness fills the focused listening review notes from MIDI/context evidence and records whether the candidate remains follow-up work.

For Stage B margin-recovered phrase/vocabulary repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-repair
```

This harness tests whether the timing/repetition candidate can keep dead-air and pitch coverage while reducing adjacent repeats and wide intervals.

For Stage B margin-recovered phrase/vocabulary focused context changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-focused-context
```

This harness packages the selected phrase/vocabulary repair candidate with chord/bass context and verifies focused context decision readiness.

For Stage B margin-recovered phrase/vocabulary focused listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-focused-listening-notes
```

This harness writes the focused listening review notes template for the selected phrase/vocabulary context keep candidate.

For Stage B margin-recovered phrase/vocabulary focused listening fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-focused-listening-fill
```

This harness fills the focused listening review notes from MIDI/context evidence and records the keep/follow-up boundary.

For Stage B margin-recovered phrase/vocabulary keep stability changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-keep-stability
```

This harness compares the filled keep candidate against qualified phrase/vocabulary sweep peers and records the stability boundary.

For Stage B margin-recovered phrase/vocabulary qualified peer focused context changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-peer-focused-context
```

This harness packages the qualified peer candidate with chord/bass context and verifies focused context decision readiness.

For Stage B margin-recovered phrase/vocabulary qualified peer focused listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-notes
```

This harness writes the focused listening review notes template for the qualified peer context keep candidate.

For Stage B margin-recovered phrase/vocabulary qualified peer focused listening fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-fill
```

This harness fills the qualified peer focused listening review notes from MIDI/context evidence and records the fallback keep boundary.

For Stage B margin-recovered phrase/vocabulary two-candidate keep consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-two-candidate-keep
```

This harness joins selected and peer filled keep notes with the stability summary and records the two-candidate evidence boundary.

For Stage B margin-recovered phrase/vocabulary human listening comparison boundary changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-human-listening-comparison
```

This harness prepares pending human listening fields and verifies whether selected and peer keep candidates are distinct enough for A/B listening comparison.

For Stage B margin-recovered phrase/vocabulary duplicate-candidate source divergence changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duplicate-source-divergence
```

This harness audits whether selected and peer keep candidates are distinct outputs or duplicate MIDI content from a shared sample seed.

For Stage B margin-recovered phrase/vocabulary sample-seed diversity repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-sample-seed-diversity
```

This harness demotes duplicate sample-seed peers from distinct-output support and records the repaired claim boundary.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-sweep
```

This harness runs a focused checkpoint-based sweep outside the duplicate sample seed range and reports whether a distinct sample-seed qualified candidate exists.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed focused context changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-context
```

This harness packages the selected distinct sample-seed candidate with chord/bass context and verifies focused context decision readiness.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed focused listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-notes
```

This harness writes the focused listening review notes template for the distinct sample-seed context keep candidate.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed focused listening fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-fill
```

This harness fills the distinct sample-seed focused listening review notes from MIDI/context evidence and records the keep/follow-up boundary.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed remaining blocker changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker
```

This harness summarizes the focused listening fill blockers and records the next repair target.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed remaining blocker repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker-repair-sweep
```

This harness runs an additional checkpoint-based sampling sweep against the remaining blocker target and records whether a target-qualified candidate exists.

For Stage B margin-recovered phrase/vocabulary distinct sample-seed dead-air adjacent repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-dead-air-adjacent-repair
```

This harness runs a lower-temperature checkpoint-based sampling sweep against the dead-air and adjacent-repeat target.

For Stage B margin-recovered phrase/vocabulary coverage-aware adjacent constrained repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-coverage-aware-adjacent-constrained-repair
```

This harness runs coverage-aware constrained decoding to reduce adjacent pitch repeats and records the remaining dead-air boundary.

For Stage B margin-recovered phrase/vocabulary duration coverage fill repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-repair
```

This harness builds duration/coverage fill variants for the constrained partial candidate and verifies the objective guardrails.

For Stage B margin-recovered phrase/vocabulary duration coverage fill focused context changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-context
```

This harness packages the selected duration/coverage fill candidate with chord/bass context and verifies focused context decision readiness.

For Stage B margin-recovered phrase/vocabulary duration coverage fill focused listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-notes
```

This harness writes the focused listening review notes template for the selected duration/coverage fill candidate.

For Stage B margin-recovered phrase/vocabulary duration coverage fill focused listening fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-fill
```

This harness fills the selected duration/coverage fill focused listening review notes from MIDI/context evidence.

For Stage B margin-recovered phrase/vocabulary duration coverage fill keep consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-keep-consolidation
```

This harness consolidates the selected duration/coverage fill keep candidate and records the single postprocess-candidate claim boundary.

For Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio boundary changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-boundary
```

This harness prepares the source-vs-fill human/audio review boundary and keeps preference fields pending.

For Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-review-input-guard
```

This harness verifies that human/audio preference remains pending without validated review input.

For Stage B margin-recovered phrase/vocabulary duration coverage fill audio review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-audio-review-package
```

This harness builds the source/fill MIDI review package and input template without claiming a preference.

For Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-review
```

This harness reviews source vs fill from MIDI evidence only and does not claim human/audio preference.

For Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-consolidation
```

This harness consolidates MIDI evidence preference and keeps human/audio preference unclaimed.

For Stage B margin-recovered phrase/vocabulary duration coverage fill external human/audio boundary changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-external-human-audio-boundary
```

This harness records the external review input requirement before any human/audio preference claim.

For Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-local-audio-render-package
```

This harness packages local audio render readiness without claiming rendered audio quality.

For Stage B local audio render tooling setup changes, run:

```bash
bash scripts/agent_harness.sh stage-b-local-audio-render-tooling
```

This harness checks renderer/soundfont readiness without installing packages or rendering audio.

For Stage B renderer path decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-renderer-path-decision
```

This harness records whether renderer path or install approval is required before local audio render attempt.

For Stage B duration coverage fill local audio render attempt changes, run:

```bash
bash scripts/agent_harness.sh stage-b-local-audio-render-attempt
```

This harness renders source/fill MIDI to local WAV files and validates technical WAV metadata without claiming listening preference.

For Stage B duration coverage fill user listening review fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-user-listening-review-fill
```

This harness applies a validated user listening review from rendered WAV files and keeps broad model quality unclaimed.

For Stage B duration coverage fill user listening review consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-user-listening-review-consolidation
```

This harness consolidates MIDI evidence, technical WAV validation, and single-user listening preference.

For Stage B duration coverage fill next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-next-decision
```

This harness selects the next auto-progress boundary after consolidated fill evidence.

For Stage B duration coverage fill broader repeatability sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-broader-repeatability-sweep
```

This harness applies duration/coverage fill gate checks to distinct source candidates and records repeatability boundaries.

For Stage B duration coverage fill dead-air gain repeatability repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-dead-air-gain-repeatability-repair
```

This harness requires selected distinct source candidates to keep the MIDI gate while reducing dead-air.

For Stage B duration coverage fill repeatability consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-repeatability-consolidation
```

This harness consolidates current keep listening support and distinct-source MIDI/dead-air gain evidence.

For Stage B duration coverage fill repeatability audio review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-repeatability-audio-review-package
```

This harness renders repeatability source candidates to local WAV files and validates technical WAV metadata without claiming listening preference.

For Stage B duration coverage fill repeatability user listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-repeatability-user-listening-review
```

This harness records user listening feedback for repeatability WAV files and keeps human/audio keep claims false when candidates need follow-up.

For Stage B duration coverage fill outside-soloing repair decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-decision
```

This harness converts repeatability user listening follow-up into pitch-role and phrase-clarity repair targets.

For Stage B duration coverage fill outside-soloing repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-sweep
```

This harness builds pitch-role repair candidates for repeatability sources while preserving dead-air gain and monophonic gates.

For Stage B duration coverage fill outside-soloing repair audio review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-audio-review-package
```

This harness renders selected outside-soloing repair candidates to WAV files and validates technical audio metadata without claiming preference.

For Stage B duration coverage fill outside-soloing repair user listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-user-listening-review
```

This harness records missing listening review input as pending while allowing objective-only follow-up without claiming human preference.

For Stage B duration coverage fill outside-soloing repair objective evidence changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-objective-evidence
```

This harness consolidates repaired candidate objective evidence while keeping human/audio preference claims false.

For Stage B duration coverage fill outside-soloing repair next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-next-decision
```

This harness converts objective evidence support into the next repeatability sweep boundary.

For Stage B duration coverage fill outside-soloing repair broader repeatability changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-broader-repeatability
```

This harness aggregates policy-level objective repeatability across selected outside-soloing repair sources.

For Stage B duration coverage fill outside-soloing repair repeatability consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-repeatability-consolidation
```

This harness consolidates selected-source objective support, policy repeatability support, and the pending review boundary without claiming human/audio preference.

For Stage B duration coverage fill outside-soloing repair final decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-final-decision
```

This harness records the objective-only final boundary and routes the next automatic task to model-core evidence README refresh without claiming human/audio preference.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-objective-next
```

This harness closes the controlled checkpoint temperature-guard objective path and routes the next boundary to controlled training scale expansion without claiming human/audio preference or musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale expansion decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-expansion-decision
```

This harness selects the next bounded local training scale after the controlled checkpoint objective path without running full training or claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-smoke
```

This harness executes the selected `2048/512` local training smoke and validates checkpoint readiness without claiming generation or musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-generation-probe
```

This harness probes generation from the selected-scale checkpoint and routes the next boundary by objective strict-gate outcome without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale repair decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-repair-decision
```

This harness selects the next repair target after selected-scale checkpoint generation failure without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-density-grammar-collapse-repair-probe
```

This harness tests the selected repair target against the selected-scale checkpoint and records repeatability as the next boundary without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repeatability probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-density-grammar-collapse-repeatability-probe
```

This harness repeats the selected repair condition across seeds and routes remaining dead-air separately without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air remaining blocker decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-remaining-blocker-decision
```

This harness selects the selected-scale dead-air repair target after density/grammar/collapse repeatability support without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repair-probe
```

This harness tests whether the selected-scale checkpoint can remove the dead-air blocker under the selected repair condition without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair repeatability probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repair-repeatability-probe
```

This harness repeats the selected-scale dead-air repair condition across seeds and routes unstable repeatability to a temperature guard decision without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repeatability-temperature-guard-decision
```

This harness selects the lower-temperature repeatability guard target after selected-scale dead-air repair repeatability failure without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repeatability-temperature-guard-repair-probe
```

This harness tests the selected lower-temperature guard config and routes remaining instability to a follow-up decision without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard follow-up decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repeatability-temperature-guard-followup-decision
```

This harness selects the postprocess-removal dead-air repair target after partial temperature guard repair without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-probe
```

This harness tests the reused-position guard against postprocess removal and dead-air failures without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-consolidation
```

This harness consolidates objective MIDI support and routes qualified candidates to an audio review package without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair audio review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-audio-review-package
```

This harness renders selected objective-supported MIDI candidates to WAV and records technical audio validation without claiming listening preference.

For Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-listening-review
```

This harness writes the pending listening review template and blocks preference fill without validated review input.

For Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-objective-next
```

This harness closes the objective MIDI path and routes to MVP evidence consolidation without claiming listening quality.

For Stage B MIDI-to-solo MVP current evidence consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-current-evidence-consolidation
```

This harness consolidates the input contract, context extraction, ranked MIDI candidates, technical WAV render path, selected-scale objective repair boundary, CLI technical path, model-conditioned pitch-contour objective path, and changed-ratio repair objective path without claiming musical quality.

For Stage B MIDI-to-solo README evidence refresh changes, run:

```bash
git diff --check
bash scripts/agent_harness.sh quick
```

This validation keeps the README evidence boundary aligned with current reports without adding generation or quality claims.

For Stage B MIDI-to-solo MVP completion audit changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-completion-audit
```

This harness audits technical model-core MVP completion, including the CLI technical path, model-conditioned pitch-contour objective path, and changed-ratio repair objective path, while keeping musical quality, human preference, broad model quality, and product readiness claims excluded.

For Stage B MIDI-to-solo quality gap decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-quality-gap-decision
```

This harness selects the next quality-gap target after technical MVP completion, preserves source/current outside-soloing repair context, and avoids musical quality or immediate human-review claims.

For Stage B MIDI-to-solo listening review quality gap changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-listening-review-quality-gap
```

This harness routes the listening review quality gap to MVP delivery packaging, preserves source/current outside-soloing repair context, and avoids musical quality or human/audio preference claims.

For Stage B MIDI-to-solo MVP delivery package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-delivery-package
```

This harness records the runnable CLI and local artifact evidence for MVP delivery, preserves source/current outside-soloing repair context, and avoids raw artifact upload or quality claims.

For Stage B MIDI-to-solo final status audit changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-final-status-audit
```

This harness audits technical MVP completion and README evidence reflection, preserves source/current outside-soloing repair context, and avoids musical quality or human/audio preference claims.

For Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio review decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-review-decision
```

This harness routes pitch-contour changed-ratio review evidence to the next repair probe without claiming musical quality or human/audio preference.

For Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-probe
```

This harness repairs pitch-contour candidates with a lower pitch-change ratio objective and routes passed repaired MIDI candidates to audio render packaging without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-audio-package
```

This harness renders changed-ratio repaired model-conditioned MIDI candidates to WAV and verifies technical audio metadata without claiming musical quality or human/audio preference.

For Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-listening-review-package
```

This harness packages changed-ratio repaired WAV/MIDI candidates for pending listening review without claiming musical quality or human/audio preference.

For Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-listening-review-input-guard
```

This harness blocks changed-ratio repair preference fill while listening review input is pending.

For Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-objective-next
```

This harness routes changed-ratio repair objective evidence to current evidence consolidation without claiming musical quality or human/audio preference.

For Stage B MIDI-to-solo model-conditioned input path quality alignment changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-quality-alignment
```

This harness records the fallback replacement requirements for the model-conditioned input path and routes to the next probe without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-probe
```

This harness checks model-conditioned MIDI/WAV evidence against the current fallback input-to-WAV contract and routes to candidate export without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path candidate export changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-candidate-export
```

This harness exports model-conditioned strict MIDI candidates through the ranked input-path contract and routes to ranked audio render without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-audio-render-package
```

This harness renders model-conditioned ranked MIDI exports to WAV and records technical replacement evidence without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path replacement consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-replacement-consolidation
```

This harness consolidates model-conditioned ranked MIDI/WAV technical replacement evidence and routes to listening review packaging without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-listening-review-package
```

This harness packages ranked WAV/MIDI review items and keeps human/audio preference pending without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-listening-review-input-guard
```

This harness blocks model-conditioned input-path preference fill while listening review input is pending.

For Stage B MIDI-to-solo model-conditioned input path objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-objective-next
```

This harness selects the next model-conditioned input-path boundary from objective MIDI/WAV evidence only.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-decision
```

This harness defines the dead-air/timing repair target and guardrails after objective-only evidence.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-probe
```

This harness repairs ranked model-conditioned MIDI candidate timing gaps and verifies objective dead-air guardrails without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-audio-package
```

This harness renders repaired model-conditioned dead-air/timing MIDI candidates to WAV and verifies technical audio metadata without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair objective next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-objective-next
```

This harness selects the next objective repair boundary after repaired dead-air/timing audio evidence without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-decision
```

This harness defines the pitch-contour repair target after repaired dead-air/timing objective evidence without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-probe
```

This harness repairs wide pitch intervals in dead-air/timing repaired model-conditioned MIDI candidates without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-audio-package
```

This harness renders pitch-contour repaired model-conditioned MIDI candidates to WAV and verifies technical audio metadata without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-listening-review-package
```

This harness packages pitch-contour repaired WAV/MIDI candidates for pending listening review without claiming musical quality.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-listening-review-input-guard
```

This harness blocks pitch-contour preference fill while listening review input is pending.

For Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-objective-next
```

This harness selects the next boundary after pitch-contour objective evidence without claiming musical quality.

For Stage B MIDI-to-solo phrase-bank retrieval baseline changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-retrieval-baseline
```

This harness extracts data-derived phrase/motif templates, exports ranked MIDI candidates, and keeps musical quality claims excluded.

For Stage B MIDI-to-solo phrase-bank audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-audio-render-package
```

This harness renders phrase-bank ranked MIDI exports to WAV and records technical audio metadata without claiming musical quality.

For Stage B MIDI-to-solo phrase-bank listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-listening-review-package
```

This harness packages phrase-bank MIDI/WAV review items and keeps preference pending without claiming musical quality.

For Stage B MIDI-to-solo phrase-bank listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-listening-review-input-guard
```

This harness blocks preference fill while phrase-bank listening review input is pending and routes to the objective-only next decision.

For Stage B MIDI-to-solo phrase-bank objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-objective-only-next-decision
```

This harness selects the next phrase-bank boundary from objective MIDI/WAV evidence only, without claiming human/audio preference or musical quality.

For Stage B MIDI-to-solo phrase-bank dead-air density repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-probe
```

This harness repairs phrase-bank candidates for dead-air and density variation, then routes objective-supported repaired MIDI candidates to audio packaging without claiming musical quality.

For Stage B MIDI-to-solo phrase-bank dead-air density repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-audio-package
```

This harness renders dead-air/density repaired MIDI candidates to local WAV files and records technical audio metadata without claiming listening preference.

For Stage B MIDI-to-solo phrase-bank dead-air density repair listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-listening-review-package
```

This harness packages dead-air/density repaired MIDI/WAV candidates for pending listening review without claiming human/audio preference.

For Stage B MIDI-to-solo phrase-bank dead-air density repair listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-listening-review-input-guard
```

This harness blocks preference fill while repaired phrase-bank listening review input is pending and routes to the objective-only next decision.

For Stage B MIDI-to-solo phrase-bank dead-air density repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-objective-only-next-decision
```

This harness selects the next repaired phrase-bank boundary from objective MIDI/WAV evidence only and routes CLI MVP packaging without claiming listening preference.

For Stage B MIDI-to-solo phrase-bank CLI MVP package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-cli-mvp-package
```

This harness builds a runnable input-MIDI to repaired ranked MIDI package manifest without claiming listening preference or musical quality.

For Stage B MIDI-to-solo phrase-bank CLI user-input smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-cli-user-input-smoke
```

This harness validates the phrase-bank CLI package with an explicit input MIDI path and keeps listening preference unclaimed.

For Stage B MIDI-to-solo phrase-bank CLI audio render smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-cli-audio-render-smoke
```

This harness renders explicit-input phrase-bank CLI MIDI candidates to WAV and records technical metadata without claiming audio quality.

For Stage B MIDI-to-solo phrase-bank CLI listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-cli-listening-review-package
```

This harness packages explicit-input phrase-bank CLI WAV/MIDI candidates for pending listening review without claiming preference.

For Stage B MIDI-to-solo phrase-bank CLI listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-cli-listening-review-input-guard
```

This harness blocks CLI phrase-bank preference fill while listening review input is pending.

For Stage B MIDI-to-solo phrase-bank CLI objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-phrase-bank-cli-objective-only-next-decision
```

This harness routes CLI technical evidence to current evidence consolidation without quality claims.

For Stage B MIDI-to-solo model-direct user listening review fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-user-listening-review-fill
```

This harness records single-user listening input for the rendered model-direct WAV candidates without claiming human/audio keep preference or broad MIDI-to-solo musical quality.

For Stage B MIDI-to-solo model-direct songlike melody rejection analysis changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-songlike-rejection-analysis
```

This harness analyzes rejected model-direct MIDI candidates for fixed density, repeated rhythm templates, and interval-cap compression before routing to jazz phrase vocabulary repair planning.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-decision
```

This harness converts songlike rejection evidence into the next repair probe targets without claiming musical quality improvement.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-probe
```

This harness generates repaired MIDI candidates and verifies fixed-density, repeated-rhythm, interval-cap, and overlap guardrails without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-audio-package
```

This harness renders repaired MIDI candidates to local WAV files and validates technical WAV metadata without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-listening-review
```

This harness prepares pending listening review input and blocks preference fill while review input is missing.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-objective-next
```

This harness routes pending listening review evidence to the next objective repair target without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repair
```

This harness generates contour/phrase-shape repaired MIDI candidates and verifies stepwise contour reduction without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-audio-package
```

This harness renders contour/phrase-shape repaired MIDI candidates to WAV files and validates technical WAV metadata without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-listening-review
```

This harness prepares pending contour/phrase-shape listening review input and blocks preference fill while review input is missing.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-objective-next
```

This harness records the objective-clean contour phrase-shape boundary and routes to repeatability checking without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-clean repeatability sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-sweep
```

This harness generates distinct contour phrase-shape MIDI variants and verifies objective-clean repeatability without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-clean repeatability consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-consolidation
```

This harness consolidates objective-clean repeatability support and routes the generated MIDI variants to audio review packaging without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability audio package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-audio-package
```

This harness renders repeatability MIDI variants to WAV files and validates technical WAV metadata without claiming listening quality.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-listening-review
```

This harness prepares pending repeatability listening review input and blocks preference fill while review input is missing.

For Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability objective-only next decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-objective-next
```

This harness closes the repeatability objective path and routes to evidence refresh without claiming listening quality.

For Stage B MIDI-to-solo training scale expansion decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-training-scale-expansion-decision
```

This harness selects the next bounded training scale smoke without running broad training or claiming model quality.

For Stage B MIDI-to-solo controlled training scale smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-training-scale-smoke
```

This harness runs the bounded `512/128`, `max_sequence=160` local training smoke and summarizes checkpoint readiness without claiming model quality.

For Stage B MIDI-to-solo controlled scale checkpoint generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-generation-probe
```

This harness probes generation/decode from the controlled MIDI-to-solo scale checkpoint and records the strict gate boundary without claiming model quality.

For Stage B MIDI-to-solo controlled scale checkpoint repair decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-repair-decision
```

This harness converts controlled checkpoint generation gate failure into a density/collapse/postprocess repair target without claiming model quality.

For Stage B MIDI-to-solo controlled scale checkpoint density/collapse repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-density-collapse-repair-probe
```

This harness runs constrained density/collapse repair on the controlled checkpoint and routes any remaining strict-gate blocker without claiming model quality.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air remaining blocker decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-remaining-blocker-decision
```

This harness selects the dead-air repair target after density/collapse support while keeping musical quality and listening preference unclaimed.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-probe
```

This harness runs the selected constrained dead-air repair probe and routes single-seed support to repeatability without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repair repeatability probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-repeatability-probe
```

This harness runs the configured seed repeatability probe and records partial repeatability as a follow-up boundary without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-decision
```

This harness selects the lower-temperature repeatability guard after partial seed failure while keeping musical quality and listening preference unclaimed.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-repair-probe
```

This harness runs the selected lower-temperature guard sweep and routes qualified results to consolidation without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-repair-consolidation
```

This harness consolidates objective MIDI support and routes candidates to an audio review package without claiming musical quality.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard audio review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-audio-review-package
```

This harness renders selected objective-supported MIDI candidates to WAV and records technical audio validation without claiming listening preference.

For Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-listening-review
```

This harness writes the pending listening review template and blocks preference fill without validated review input.

For Stage B generic jazz base readiness audit changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-readiness-audit
```

This harness verifies that dataset pool evidence and Stage B objective-path evidence support Phase 4 preparation without claiming broad trained-model quality or Brad style adaptation.

For Stage B generic jazz base manifest contract changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-manifest-contract
```

This harness verifies generic/Brad manifest split counts and leakage guards before Stage B duration-explicit window preparation, without claiming broad trained-model quality or Brad style adaptation.

For Stage B generic split duration-explicit window preparation smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-manifest-window-smoke
```

This harness prepares a small generic manifest prefix as `stage_b_v1` duration-explicit windows and verifies train/val token records plus vocab fit without running broad training.

For Stage B generic base tiny training smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-tiny-training-smoke
```

This harness copies a small generic Stage B window token subset into the training path and verifies the training command succeeds without claiming broad model quality.

For Stage B generic model-core training data plan changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-model-core-training-data-plan
```

This harness consolidates the repair-loop stop decision, manifest contract, window smoke, and tiny training smoke into the next full-window preparation plan without claiming broad trained-model quality.

For Stage B generic full manifest window preparation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-full-manifest-window-preparation
```

This harness prepares the full generic train/val manifests as Stage B window records and validates token/vocab guards without running training.

For Stage B generic base training scale smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-training-scale-smoke
```

This harness runs a larger-than-tiny local training smoke from the full generic Stage B window records and validates checkpoint/loss evidence without claiming broad trained-model quality.

For Stage B generic base scale checkpoint generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-generation-probe
```

This harness loads the generic-base scale checkpoint into the Stage B generation/decode path and records raw gate results without claiming broad trained-model quality or Brad style adaptation.

For Stage B generic base scale checkpoint grammar/representation decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-grammar-representation-decision
```

This harness classifies the scale-checkpoint raw generation failure and selects the next density/coverage repair target without claiming root cause or musical quality.

For Stage B generic base scale checkpoint density/coverage repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-density-coverage-repair-probe
```

This harness runs a coverage-aware constrained repair probe from the scale checkpoint and compares note-count failure plus coverage deltas against the raw baseline without claiming broad model quality.

For Stage B generic base scale checkpoint density/coverage remaining blocker decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-density-coverage-remaining-blocker-decision
```

This harness selects the duration/long-note repair target after density/coverage qualification while keeping musical quality and listening preference unclaimed.

For Stage B generic base scale checkpoint duration/long-note repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-duration-long-note-repair-probe
```

This harness runs a duration-token constrained repair probe from the scale checkpoint and compares long-note failure count plus coverage deltas against the density/coverage repair baseline without claiming broad model quality.

For Stage B generic base scale checkpoint duration/long-note remaining blocker decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-duration-long-note-remaining-blocker-decision
```

This harness selects the sustained coverage/dead-air repair target after duration/long-note qualification while keeping musical quality and listening preference unclaimed.

For Stage B generic base scale checkpoint sustained coverage/dead-air repair probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-sustained-coverage-dead-air-repair-probe
```

This harness increases constrained note-group density while preserving duration-token guardrails, then verifies dead-air failure removal, sustained coverage recovery, and no broad model quality claim.

For Stage B generic base scale checkpoint objective gate consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-objective-gate-consolidation
```

This harness consolidates current seed-set objective gate support and routes the next boundary to repeatability without claiming musical quality or human/audio preference.

For Stage B generic base scale checkpoint objective gate repeatability sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-objective-gate-repeatability-sweep
```

This harness repeats the objective gate probe across multiple seed starts and records repeatability while keeping musical quality and human/audio preference unclaimed.

For Stage B generic base scale checkpoint repeatability consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-base-scale-checkpoint-repeatability-consolidation
```

This harness consolidates objective MIDI gate repeatability evidence and routes the next boundary to README evidence refresh without claiming musical quality or human/audio preference.

For Stage B MIDI-to-solo MVP input contract changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-contract
```

This harness defines the input MIDI to ranked jazz solo MIDI candidate contract, including constrained decoding, candidate ranking, and retrieval fallback boundaries.

For Stage B MIDI-to-solo context extraction changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-context-extraction
```

This harness extracts bar/position/chord/bass context rows from an input MIDI fixture without claiming harmony-analysis quality or completed MIDI-to-solo generation.

For Stage B MIDI-to-solo training resource probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-training-resource-probe
```

This harness checks context extraction, full Stage B window records, and scale-smoke checkpoint resources before conditioned generation without claiming final MIDI-to-solo output quality.

For Stage B MIDI-to-solo conditioned generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-conditioned-generation-probe
```

This harness exports ranked context-conditioned MIDI candidates and verifies objective gates without claiming completed MVP quality or human/audio preference.

For Stage B MIDI-to-solo candidate audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-candidate-audio-render-package
```

This harness renders exported MIDI-to-solo candidates to local WAV files and validates technical WAV metadata without claiming audio quality or human preference.

For Stage B MIDI-to-solo MVP execution consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-mvp-execution-consolidation
```

This harness consolidates the input-to-context-to-MIDI-to-WAV technical path while keeping musical quality, model-direct generation quality, and human preference unclaimed.

For Stage B MIDI-to-solo model-direct generation repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-generation-repair
```

This harness compares the current scale-smoke checkpoint sequence budget with the 8-bar / 24-note MIDI-to-solo contract and defines the sequence-budget repair boundary without claiming direct model quality.

For Stage B MIDI-to-solo model-direct sequence budget repair smoke changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-sequence-budget-repair-smoke
```

This harness runs a `max_sequence=160` smoke checkpoint and verifies that its direct Stage B sequence budget can enter the 8-bar generation probe without claiming model quality.

For Stage B MIDI-to-solo model-direct 8-bar generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-8bar-generation-probe
```

This harness generates fallback-free 8-bar MIDI from the repaired checkpoint, records grammar/review gate evidence, and routes overlap-heavy failures to the monophonic overlap repair boundary without claiming model quality.

For Stage B MIDI-to-solo model-direct monophonic overlap repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-monophonic-overlap-repair
```

This harness caps generated durations to the next planned position, compares the repaired outputs against the prior 8-bar probe, and records review-gate evidence without claiming model quality or human preference.

For Stage B MIDI-to-solo model-direct audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-audio-render-package
```

This harness renders repaired model-direct MIDI candidates to WAV files and validates technical WAV metadata without claiming audio quality, model quality, or human preference.

For Stage B MIDI-to-solo model-direct audio evidence consolidation changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-audio-evidence-consolidation
```

This harness consolidates model-direct objective MIDI evidence and WAV render evidence without claiming model quality, musical quality, or human preference.

For Stage B MIDI-to-solo model-direct phrase quality diagnostics changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-phrase-quality-diagnostics
```

This harness diagnoses note-level phrase risks from model-direct MIDI candidates and routes the next repair boundary without claiming musical quality or human preference.

For Stage B MIDI-to-solo model-direct pitch contour repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-pitch-contour-repair
```

This harness applies model-direct pitch range and adjacent interval guards, compares note-level diagnostics before/after, and keeps musical quality and human preference unclaimed.

For Stage B MIDI-to-solo model-direct timing phrase repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-timing-phrase-repair
```

This harness applies compact timing positions and duration fill, compares note-level dead-air diagnostics before/after, and keeps musical quality and human preference unclaimed.

For Stage B MIDI-to-solo model-direct listening review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-listening-review-package
```

This harness packages timing-repaired MIDI candidates as MIDI/WAV review files, writes the pending review input template, and keeps listening preference unclaimed.

For Stage B MIDI-to-solo model-direct user listening review input guard changes, run:

```bash
bash scripts/agent_harness.sh stage-b-midi-to-solo-model-direct-user-listening-review-input-guard
```

This harness verifies that pending review input blocks preference fill and keeps listening preference unclaimed.

For Stage B generic tiny checkpoint generation probe changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-generation-probe
```

This harness loads the generic tiny checkpoint into the Stage B generation/decode path and records gate results without claiming broad model quality.

For Stage B generic tiny checkpoint grammar repair changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-grammar-repair
```

This harness compares raw generation with constrained + jazz-duration generation from the same tiny checkpoint and records repair gates without claiming broad model quality.

For Stage B generic tiny checkpoint repair repeatability changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-repeatability
```

This harness runs a seed-expanded constrained + jazz-duration repair probe and records repeatability gates without claiming broad model quality.

For Stage B generic tiny checkpoint repair review package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-review-package
```

This harness packages strict-valid repair candidates for review and records MIDI paths without claiming musical quality.

For Stage B generic tiny checkpoint repair listening notes changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-listening-notes
```

This harness builds pending human-review listening notes for packaged repair candidates without filling or claiming musical quality.

For Stage B generic tiny checkpoint repair listening fill changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-listening-fill
```

This harness guards listening-note fill when human review input is absent and keeps musical quality claims blocked.

For Stage B generic tiny checkpoint repair audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-audio-render-package
```

This harness packages the pending repair candidates for local WAV rendering without attempting render or claiming audio quality.

For Stage B generic tiny checkpoint repair local audio render attempt changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-local-audio-render-attempt
```

This harness renders the packaged repair candidates to local WAV files and validates WAV metadata without claiming listening quality.

For Stage B generic tiny checkpoint repair user listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-user-listening-review
```

This harness records the single-user reject-all listening result and routes the plunk-and-stop failure to the next repair decision.

For Stage B generic tiny checkpoint repair phrase continuation decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-decision
```

This harness converts the plunk-and-stop listening rejection into phrase-continuation repair targets without claiming quality.

For Stage B generic tiny checkpoint repair phrase continuation sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-sweep
```

This harness runs a chord-aware phrase-continuation repair sweep and requires at least one objective-qualified candidate without claiming quality.

For Stage B generic tiny checkpoint repair phrase continuation audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-audio-render-package
```

This harness packages the selected phrase-continuation repair candidate for local WAV rendering without claiming audio quality.

For Stage B generic tiny checkpoint repair phrase continuation local audio render attempt changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-local-audio-render-attempt
```

This harness renders the selected phrase-continuation repair candidate to local WAV and validates technical metadata without claiming listening quality.

For Stage B generic tiny checkpoint repair phrase continuation MIDI note failure review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-midi-note-failure-review
```

This harness records the user rejection with MIDI note-sequence evidence and routes the next repair target to range/interval guard work.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-decision
```

This harness converts the MIDI note failure evidence into range/interval guard targets without claiming repaired candidate quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sweep
```

This harness runs constrained interval-cap sweeps and verifies actual MIDI note range/interval guard candidates without claiming listening quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-audio-render-package
```

This harness packages range/interval guard candidates for local WAV rendering without attempting render or claiming listening quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard local audio render attempt changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-local-audio-render-attempt
```

This harness renders range/interval guard candidates to local WAV files and validates technical metadata without claiming listening quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard user listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-user-listening-review
```

This harness records user listening rejection for range/interval guard WAV files without claiming audio quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard rejection analysis changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-rejection-analysis
```

This harness analyzes rejected range/interval guard MIDI candidates and records evidence flags without claiming a musical-quality root cause.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-repair-decision
```

This harness converts sparse phrase evidence into the next repair sweep boundary without claiming musical quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-repair-sweep
```

This harness runs coverage-aware sparse phrase repair generation and verifies objective gap reduction without claiming listening quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-audio-render-package
```

This harness packages sparse phrase repair MIDI candidates for local WAV rendering without claiming audio quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-local-audio-render-attempt
```

This harness renders sparse phrase repair MIDI candidates to local WAV files and validates technical metadata without claiming listening quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase user listening review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-user-listening-review
```

This harness records user listening rejection for sparse phrase repair WAV files without claiming audio quality.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase rejection analysis changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-rejection-analysis
```

This harness analyzes rejected sparse phrase MIDI candidates and records whether objective proxy evidence is insufficient before routing to model-core review.

For Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase model core review changes, run:

```bash
bash scripts/agent_harness.sh stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-model-core-review
```

This harness records the stop decision for the constraint/postprocess repair loop and routes the next work to model-core training/data planning.

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
