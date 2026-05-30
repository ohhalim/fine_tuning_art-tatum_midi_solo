# AGENTS.md

## Project Focus

This repository is currently focused on Stage B symbolic MIDI generation probes.

Primary goal:

- Build a reliable local MIDI generation and validation pipeline.
- Prove model behavior with small, reproducible experiments before expanding scope.

Current handoff scope:

- Latest functional issue completed: Issue #399, Stage B generic tiny checkpoint repair review package.
- Current branch should be `main` before starting new work.
- Recommended next issue: Stage B generic tiny checkpoint repair listening notes.

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

- Do not include assistant/tool names in branch names, issue titles, issue bodies, PR titles, PR bodies, or commit messages unless the user explicitly asks.
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
