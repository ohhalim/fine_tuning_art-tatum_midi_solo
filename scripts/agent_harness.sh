#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

MODE="${1:-quick}"

usage() {
  cat <<'EOF'
Agent harness for local validation.

Usage:
  bash scripts/agent_harness.sh <mode>

Modes:
  status        Show branch and working tree status.
  quick         Run unit tests, compile checks, and git whitespace check.
  demo          Run quick checks plus the MVP demo.
  tiny-prepare  Verify tiny-overfit dataset preparation only.
  tiny-compare  Compare full-model tiny training with random-base LoRA-only.
  control-tiny  Run control_v1 full-model tiny-overfit smoke.
  stage-b-window-prepare
                Prepare Stage B phrase windows and verify token vocab fit.
  stage-b-generation-probe
                Run a Stage B raw decode/generation gate probe.
  stage-b-raw-generation-repeatability
                Run raw Stage B generation across multiple seeds and source files.
  stage-b-dead-air-diagnostics
                Diagnose dead-air outliers from a Stage B generation probe report.
  stage-b-seed-strict-margin-diagnostics
                Diagnose per-seed strict margin risk from a Stage B repeatability summary.
  stage-b-margin-recovered-review-export
                Export review metrics from a margin-recovered Stage B repeatability summary.
  stage-b-margin-recovered-listening-notes
                Build pending listening notes for margin-recovered Stage B candidates.
  stage-b-margin-recovered-proxy-review-fill
                Fill margin-recovered listening notes with MIDI-metric proxy review.
  stage-b-margin-recovered-proxy-keep-focused-package
                Build a focused solo/context package for the margin-recovered proxy keep.
  stage-b-margin-recovered-focused-context-decision
                Review the margin-recovered focused package against solo/context MIDI metrics.
  stage-b-margin-recovered-focused-fallback-comparison
                Compare all margin-recovered candidates with focused context decisions.
  stage-b-margin-recovered-pitch-dead-air-repair
                Run expanded top-k sampling and select a pitch/dead-air repair candidate.
  stage-b-margin-recovered-pitch-vocab-sweep
                Run seed/top-k sweep and select a pitch-vocabulary qualified candidate.
  stage-b-margin-recovered-pitch-vocab-focused-context
                Package and review the selected pitch-vocabulary sweep candidate in context.
  stage-b-margin-recovered-pitch-vocab-focused-listening-notes
                Build focused listening notes for the selected pitch-vocabulary candidate.
  stage-b-margin-recovered-pitch-vocab-focused-listening-fill
                Fill the selected pitch-vocabulary focused listening review notes.
  stage-b-margin-recovered-timing-repetition-repair
                Run top-k/temperature repair sweep and select a timing/repetition improved candidate.
  stage-b-margin-recovered-timing-repetition-focused-context
                Package and review the selected timing/repetition repair candidate in context.
  stage-b-margin-recovered-timing-repetition-focused-listening-notes
                Build focused listening notes for the selected timing/repetition candidate.
  stage-b-margin-recovered-timing-repetition-focused-listening-fill
                Fill the selected timing/repetition focused listening review notes.
  stage-b-margin-recovered-phrase-vocabulary-repair
                Run phrase/vocabulary repair sweep and select a reduced-repeat/interval candidate.
  stage-b-margin-recovered-phrase-vocabulary-focused-context
                Package and review the selected phrase/vocabulary repair candidate in context.
  stage-b-margin-recovered-phrase-vocabulary-focused-listening-notes
                Build focused listening notes for the selected phrase/vocabulary candidate.
  stage-b-margin-recovered-phrase-vocabulary-focused-listening-fill
                Fill the selected phrase/vocabulary focused listening review notes.
  stage-b-margin-recovered-phrase-vocabulary-keep-stability
                Compare the filled keep candidate against phrase/vocabulary sweep peers.
  stage-b-margin-recovered-phrase-vocabulary-peer-focused-context
                Package and review the qualified phrase/vocabulary peer candidate in context.
  stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-notes
                Build focused listening notes for the qualified phrase/vocabulary peer candidate.
  stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-fill
                Fill the qualified phrase/vocabulary peer focused listening review notes.
  stage-b-margin-recovered-phrase-vocabulary-two-candidate-keep
                Consolidate selected and peer phrase/vocabulary keep candidates.
  stage-b-margin-recovered-phrase-vocabulary-human-listening-comparison
                Build pending human-listening comparison boundary for selected and peer keep candidates.
  stage-b-margin-recovered-phrase-vocabulary-duplicate-source-divergence
                Audit whether selected and peer keep candidates are duplicate outputs from shared sample seed.
  stage-b-margin-recovered-phrase-vocabulary-sample-seed-diversity
                Repair distinct-output claim boundary for duplicate sample-seed candidates.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-sweep
                Run a focused repair sweep with sample seed ranges outside the duplicate seed.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-context
                Package and review the selected distinct sample-seed phrase/vocabulary candidate in context.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-notes
                Build focused listening notes for the selected distinct sample-seed candidate.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-fill
                Fill the selected distinct sample-seed focused listening review notes.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker
                Summarize remaining blockers for the distinct sample-seed needs-followup candidate.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker-repair-sweep
                Run a target sweep for the distinct sample-seed remaining blockers.
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-dead-air-adjacent-repair
                Run a targeted sweep for distinct sample-seed dead-air and adjacent-repeat blockers.
  stage-b-margin-recovered-phrase-vocabulary-coverage-aware-adjacent-constrained-repair
                Run coverage-aware constrained decoding for adjacent-repeat repair.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-repair
                Build duration/coverage fill repair variants for the constrained partial candidate.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-context
                Package and review the duration/coverage fill candidate in context.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-notes
                Build focused listening notes for the duration/coverage fill candidate.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-fill
                Fill focused listening notes for the duration/coverage fill candidate.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-keep-consolidation
                Consolidate the duration/coverage fill keep candidate boundary.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-boundary
                Build pending human/audio review boundary for source vs duration fill.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-review-input-guard
                Guard human/audio review fill against missing review input.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-audio-review-package
                Build source/fill MIDI review package and input template.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-review
                Review source vs duration fill from MIDI evidence only.
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-consolidation
                Consolidate MIDI evidence review claim boundaries.
  stage-b-duration-coverage-broader-repeatability-sweep
                Run broader duration/coverage fill repeatability over distinct source candidates.
  stage-b-duration-coverage-dead-air-gain-repeatability-repair
                Repair duration/coverage fill repeatability selection to require dead-air gain.
  stage-b-duration-coverage-repeatability-consolidation
                Consolidate current keep and distinct-source repeatability evidence.
  stage-b-duration-coverage-repeatability-audio-review-package
                Render repeatability source candidates for user listening review.
  stage-b-duration-coverage-repeatability-user-listening-review
                Fill user listening review for repeatability source WAV files.
  stage-b-duration-coverage-outside-soloing-repair-decision
                Decide next repair boundary after outside-soloing user review.
  stage-b-duration-coverage-outside-soloing-repair-sweep
                Build pitch-role repair candidates for outside-soloing repeatability sources.
  stage-b-duration-coverage-outside-soloing-repair-audio-review-package
                Render outside-soloing repair candidates for user listening review.
  stage-b-duration-coverage-outside-soloing-repair-user-listening-review
                Guard outside-soloing repair listening review when user input is absent.
  stage-b-duration-coverage-outside-soloing-repair-objective-evidence
                Consolidate objective evidence for outside-soloing repair candidates.
  stage-b-duration-coverage-outside-soloing-repair-next-decision
                Decide next objective-only step after outside-soloing repair evidence.
  stage-b-duration-coverage-outside-soloing-repair-broader-repeatability
                Aggregate outside-soloing repair policy repeatability across sources.
  stage-b-duration-coverage-outside-soloing-repair-repeatability-consolidation
                Consolidate outside-soloing repair objective repeatability boundaries.
  stage-b-duration-coverage-outside-soloing-repair-final-decision
                Decide final objective-only boundary after outside-soloing repair repeatability.
  stage-b-generic-base-readiness-audit
                Assess Stage B readiness before generic jazz base preparation.
  stage-b-generic-base-manifest-contract
                Build and validate generic/Brad split manifest contract before Stage B window preparation.
  stage-b-generic-manifest-window-smoke
                Prepare Stage B duration-explicit windows from generic split manifest prefix.
  stage-b-generic-base-tiny-training-smoke
                Run a tiny training smoke from generic Stage B window records.
  stage-b-generic-model-core-training-data-plan
                Build the generic model-core training data plan after repair-loop stop.
  stage-b-generic-full-manifest-window-preparation
                Prepare full generic train/val manifests as Stage B window records.
  stage-b-generic-base-training-scale-smoke
                Run a larger-than-tiny training smoke from full generic Stage B window records.
  stage-b-generic-base-scale-checkpoint-generation-probe
                Probe generation/decode from the generic-base scale checkpoint.
  stage-b-generic-base-scale-checkpoint-grammar-representation-decision
                Decide the next repair target after scale-checkpoint raw generation failure.
  stage-b-generic-base-scale-checkpoint-density-coverage-repair-probe
                Run density/coverage repair probe for the generic-base scale checkpoint.
  stage-b-generic-base-scale-checkpoint-density-coverage-remaining-blocker-decision
                Decide the remaining blocker after density/coverage target qualification.
  stage-b-generic-base-scale-checkpoint-duration-long-note-repair-probe
                Run duration/long-note repair probe for the generic-base scale checkpoint.
  stage-b-generic-base-scale-checkpoint-duration-long-note-remaining-blocker-decision
                Decide the remaining blocker after duration/long-note repair qualification.
  stage-b-generic-base-scale-checkpoint-sustained-coverage-dead-air-repair-probe
                Run sustained coverage/dead-air repair probe for the generic-base scale checkpoint.
  stage-b-generic-base-scale-checkpoint-objective-gate-consolidation
                Consolidate current seed-set objective gate support before repeatability.
  stage-b-generic-base-scale-checkpoint-objective-gate-repeatability-sweep
                Run objective gate repeatability sweep for the generic-base scale checkpoint.
  stage-b-generic-base-scale-checkpoint-repeatability-consolidation
                Consolidate objective gate repeatability evidence for the generic-base scale checkpoint.
  stage-b-midi-to-solo-mvp-contract
                Define the MIDI-to-solo MVP input/output contract and run plan.
  stage-b-midi-to-solo-context-extraction
                Extract MIDI-to-solo context rows from an input MIDI fixture.
  stage-b-midi-to-solo-training-resource-probe
                Check MIDI-to-solo context, Stage B windows, and scale-smoke checkpoint resources.
  stage-b-midi-to-solo-conditioned-generation-probe
                Export ranked context-conditioned MIDI-to-solo candidates.
  stage-b-midi-to-solo-phrase-bank-retrieval-baseline
                Export ranked MIDI-to-solo candidates from input context and data-derived phrase-bank templates.
  stage-b-midi-to-solo-phrase-bank-audio-render-package
                Render phrase-bank MIDI-to-solo candidates to local WAV files.
  stage-b-midi-to-solo-phrase-bank-listening-review-package
                Package phrase-bank MIDI/WAV candidates for pending listening review.
  stage-b-midi-to-solo-phrase-bank-listening-review-input-guard
                Guard phrase-bank preference fill while listening review input is pending.
  stage-b-midi-to-solo-phrase-bank-objective-only-next-decision
                Select the next phrase-bank boundary from objective MIDI/WAV evidence only.
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-probe
                Repair phrase-bank candidates for dead-air and density variation.
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-audio-package
                Render dead-air/density repaired phrase-bank MIDI candidates to WAV files.
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-listening-review-package
                Package dead-air/density repaired MIDI/WAV candidates for pending listening review.
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-listening-review-input-guard
                Guard repaired phrase-bank preference fill while listening review input is pending.
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-objective-only-next-decision
                Select the next repaired phrase-bank boundary from objective MIDI/WAV evidence only.
  stage-b-midi-to-solo-phrase-bank-cli-mvp-package
                Build a runnable input-MIDI to repaired ranked MIDI package manifest.
  stage-b-midi-to-solo-phrase-bank-cli-user-input-smoke
                Validate the phrase-bank CLI package with an explicit input MIDI path.
  stage-b-midi-to-solo-phrase-bank-cli-audio-render-smoke
                Render explicit-input phrase-bank CLI MIDI candidates to WAV.
  stage-b-midi-to-solo-phrase-bank-cli-listening-review-package
                Package explicit-input phrase-bank CLI WAV/MIDI candidates for pending listening review.
  stage-b-midi-to-solo-phrase-bank-cli-listening-review-input-guard
                Block CLI phrase-bank preference fill while listening review input is pending.
  stage-b-midi-to-solo-phrase-bank-cli-objective-only-next-decision
                Route CLI technical evidence to current evidence consolidation without quality claims.
  stage-b-midi-to-solo-candidate-audio-render-package
                Render exported MIDI-to-solo candidates to local WAV files.
  stage-b-midi-to-solo-mvp-execution-consolidation
                Consolidate input-to-MIDI-to-WAV technical execution evidence.
  stage-b-midi-to-solo-mvp-current-evidence-consolidation
                Consolidate current MVP evidence from contract, generated MIDI, WAV, and objective repair.
  stage-b-midi-to-solo-mvp-completion-audit
                Audit technical model-core MVP completion and separate remaining quality claims.
  stage-b-midi-to-solo-quality-gap-decision
                Decide the next quality-gap repair target after technical MVP completion.
  stage-b-midi-to-solo-listening-review-quality-gap
                Separate remaining listening-review quality gap after objective repair evidence.
  stage-b-midi-to-solo-mvp-delivery-package
                Build the current technical MVP delivery manifest and claim boundary.
  stage-b-midi-to-solo-final-status-audit
                Audit final technical MVP status against README and delivery package evidence.
  stage-b-midi-to-solo-post-mvp-quality-iteration-plan
                Plan the first post-MVP musical quality iteration boundary.
  stage-b-midi-to-solo-quality-rubric-baseline
                Build the post-MVP MIDI evidence quality rubric baseline.
  stage-b-midi-to-solo-candidate-failure-labeling
                Label current MIDI-to-solo candidates against the quality rubric.
  stage-b-midi-to-solo-targeted-quality-repair-sweep
                Run targeted MIDI-to-solo quality repair from candidate failure labels.
  stage-b-midi-to-solo-targeted-quality-repair-audio-package
                Render targeted quality repair MIDI candidates to WAV.
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-review-decision
                Decide the next pitch-contour changed-ratio repair boundary.
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-probe
                Repair pitch-contour candidates with a lower pitch-change ratio objective.
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-audio-package
                Render changed-ratio repaired model-conditioned MIDI candidates to WAV.
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-listening-review-package
                Package changed-ratio repaired WAV/MIDI candidates for listening review.
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-listening-review-input-guard
                Block changed-ratio repair preference fill while listening review input is pending.
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-objective-next
                Decide the objective-only next boundary after changed-ratio repair input guard.
  stage-b-midi-to-solo-model-conditioned-input-path-quality-alignment
                Decide model-conditioned input-path alignment requirements and next probe target.
  stage-b-midi-to-solo-model-conditioned-input-path-listening-review-input-guard
                Block model-conditioned input-path preference fill while listening review input is pending.
  stage-b-midi-to-solo-model-conditioned-input-path-objective-next
                Select the next model-conditioned input-path boundary from objective MIDI/WAV evidence only.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-decision
                Decide the dead-air/timing repair target after model-conditioned objective evidence.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-probe
                Repair dead-air/timing gaps in ranked model-conditioned MIDI candidates.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-audio-package
                Render repaired model-conditioned dead-air/timing MIDI candidates to WAV.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-objective-next
                Select the next objective boundary after repaired dead-air/timing audio evidence.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-decision
                Define the pitch-contour repair target after repaired dead-air/timing objective evidence.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-probe
                Repair wide pitch intervals in dead-air/timing repaired model-conditioned MIDI candidates.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-audio-package
                Render pitch-contour repaired model-conditioned MIDI candidates to WAV.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-listening-review-package
                Package pitch-contour repaired WAV/MIDI candidates for pending listening review.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-listening-review-input-guard
                Block pitch-contour preference fill while listening review input is pending.
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-objective-next
                Select the next boundary after pitch-contour objective evidence.
  stage-b-midi-to-solo-model-direct-generation-repair
                Define the model-direct generation repair boundary from sequence budget evidence.
  stage-b-midi-to-solo-model-direct-sequence-budget-repair-smoke
                Run a max_sequence 160 smoke and verify direct 8-bar sequence budget readiness.
  stage-b-midi-to-solo-training-scale-expansion-decision
                Decide the next bounded MIDI-to-solo training scale smoke after objective path support.
  stage-b-midi-to-solo-controlled-training-scale-smoke
                Run the bounded 512/128 max_sequence 160 MIDI-to-solo training scale smoke.
  stage-b-midi-to-solo-controlled-scale-checkpoint-generation-probe
                Probe generation/decode from the controlled MIDI-to-solo scale checkpoint.
  stage-b-midi-to-solo-controlled-scale-checkpoint-repair-decision
                Decide the next repair target after controlled checkpoint generation gate failure.
  stage-b-midi-to-solo-controlled-scale-checkpoint-density-collapse-repair-probe
                Run constrained density/collapse repair probe for the controlled checkpoint.
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-remaining-blocker-decision
                Decide the remaining dead-air blocker after controlled density/collapse repair.
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-probe
                Run constrained dead-air repair probe for the controlled checkpoint.
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-repeatability-probe
                Run configured seed repeatability probe for the controlled dead-air repair.
  stage-b-midi-to-solo-model-direct-8bar-generation-probe
                Run fallback-free model-direct 8-bar MIDI generation and record review gate evidence.
  stage-b-midi-to-solo-model-direct-monophonic-overlap-repair
                Repair model-direct overlap by capping duration to the next planned position.
  stage-b-midi-to-solo-model-direct-audio-render-package
                Render repaired model-direct MIDI candidates to WAV and validate technical metadata.
  stage-b-midi-to-solo-model-direct-audio-evidence-consolidation
                Consolidate model-direct objective gate and WAV render evidence.
  stage-b-midi-to-solo-model-direct-phrase-quality-diagnostics
                Diagnose model-direct MIDI phrase risks from note-level evidence.
  stage-b-midi-to-solo-model-direct-pitch-contour-repair
                Repair model-direct pitch/register contour with range and interval guards.
  stage-b-midi-to-solo-model-direct-timing-phrase-repair
                Repair model-direct timing phrase gaps with compact onset and duration fill.
  stage-b-midi-to-solo-model-direct-listening-review-package
                Package timing-repaired model-direct MIDI as WAV files and pending listening review input.
  stage-b-midi-to-solo-model-direct-user-listening-review-input-guard
                Guard user listening review fill when package review input is still pending.
  stage-b-generic-tiny-checkpoint-generation-probe
                Probe generation/decode from the generic tiny checkpoint.
  stage-b-generic-tiny-checkpoint-grammar-repair
                Compare raw and grammar-repaired generation from the generic tiny checkpoint.
  stage-b-generic-tiny-checkpoint-repair-repeatability
                Run a seed-expanded repeatability probe for the generic tiny checkpoint repair.
  stage-b-generic-tiny-checkpoint-repair-review-package
                Package strict-valid generic tiny checkpoint repair candidates for review.
  stage-b-generic-tiny-checkpoint-repair-listening-notes
                Build pending listening notes for generic tiny checkpoint repair candidates.
  stage-b-generic-tiny-checkpoint-repair-listening-fill
                Guard or fill generic tiny checkpoint repair listening notes.
  stage-b-generic-tiny-checkpoint-repair-audio-render-package
                Package generic tiny checkpoint repair candidates for local audio rendering.
  stage-b-generic-tiny-checkpoint-repair-local-audio-render-attempt
                Render generic tiny checkpoint repair candidates to local WAV files.
  stage-b-generic-tiny-checkpoint-repair-user-listening-review
                Record user listening rejection for generic tiny checkpoint repair WAV files.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-decision
                Route plunk-and-stop rejection to phrase-continuation repair targets.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-sweep
                Run phrase-continuation repair sweep from the generic tiny checkpoint.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-audio-render-package
                Package the phrase-continuation repair candidate for local audio rendering.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-local-audio-render-attempt
                Render the phrase-continuation repair candidate to a local WAV file.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-midi-note-failure-review
                Record phrase-continuation rejection with MIDI note sequence failure evidence.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-decision
                Convert MIDI note failure evidence into range/interval guard targets.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sweep
                Run constrained range/interval cap sweep and audit actual MIDI notes.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-audio-render-package
                Package range/interval guard candidates for local audio rendering.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-local-audio-render-attempt
                Render range/interval guard candidates to local WAV files.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-user-listening-review
                Record user listening rejection for range/interval guard WAV files.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-user-listening-review
                Record user listening rejection for sparse phrase repair WAV files.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-rejection-analysis
                Analyze sparse phrase rejection evidence and record objective proxy gap.
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-model-core-review
                Decide model-core transition after sparse phrase objective proxy gap.
  stage-b-constrained-probe
                Run a constrained Stage B note-group smoke.
  stage-b-overlap-gate
                Run constrained Stage B generation with overlap postprocess gate.
  stage-b-stronger-probe
                Run a multi-sample constrained Stage B review-gate probe.
  stage-b-collapse-sweep
                Run Stage B top-k sampling sweep with collapse diagnostics.
  stage-b-2file-brad-probe
                Run a two-file Brad Stage B generation probe with strict reporting.
  stage-b-coverage-aware-probe
                Run a two-file Brad Stage B coverage-aware generation probe.
  stage-b-coverage-ab-sweep
                Run plain vs coverage-aware Stage B constrained generation sweep.
  stage-b-candidate-ranking
                Run coverage A/B sweep and rank generated MIDI candidates.
  stage-b-chord-aware-probe
                Run coverage/chord-aware Stage B sweep and rank candidates.
  stage-b-longer-phrase-probe
                Run a 4-bar coverage/chord-aware Stage B probe and export review MIDI.
  stage-b-pitch-mode-compare
                Compare 4-bar tones vs tones_tensions Stage B pitch modes.
  stage-b-8bar-approach-phrase
                Compare 8-bar tones/tensions/approach Stage B phrase candidates.
  stage-b-swing-motif-phrase
                Compare 8-bar baseline approach with swing/motif phrase grammar.
  stage-b-reference-stats
                Build real Stage B phrase-window reference statistics.
  stage-b-reference-pitch-roles
                Build reference pitch-role landing stats and compare latest hybrid candidates.
  stage-b-motif-templates
                Extract data-derived Stage B motif/rhythm templates.
  stage-b-data-motif-compare
                Compare hand-written swing against data-derived motif baseline.
  stage-b-data-motif-review-export
                Export named MIDI review candidates for data-derived motif compare.
  stage-b-review-context-grid
                Export review MIDI with chord context and straight-grid reference.
  stage-b-guide-tone-cadence
                Export straight-grid guide-tone/cadence candidates with chord context.
  stage-b-data-guide-hybrid
                Export data-motif rhythm plus guide-tone/cadence pitch candidates.
  stage-b-overlap-free-review-export
                Export overlap-free solo-line review MIDI variants and objective priority.
  stage-b-duration-variation-review
                Export varied-duration review candidates and objective priority.
  stage-b-phrase-cadence-review
                Export phrase/cadence review candidates and objective priority.
  stage-b-phrase-recovery-review
                Compare phrase/cadence against leap-recovery phrase candidates.
  stage-b-data-motif-phrase-recovery-review
                Compare data-motif guide tones against data-motif phrase recovery.
  stage-b-contour-landing-repair
                Compare data-motif phrase recovery against contour/cadence landing repair.
  stage-b-rhythm-phrase-variation
                Compare contour landing repair against rhythm/phrase vocabulary variation.
  stage-b-clean-review-package
                Extract objective-clean data-motif phrase recovery candidates for listening review.
  stage-b-proxy-keep-focused-package
                Extract proxy-keep reviewed candidates into a focused context listening package.
  stage-b-clean-context-diagnostics
                Diagnose objective-clean context MIDI candidates before subjective review.
  stage-b-clean-listening-review-notes
                Build clean listening review notes template for 3 context candidates.
  stage-b-focused-listening-review-notes
                Build focused listening review notes template for a proxy-keep package.
  manifest-dry-run
                Run audit -> manifest -> prepare_role_dataset smoke.
  chord-coverage-audit
                Audit chord annotation coverage in role metadata, sidecars, and MIDI text events.
  stage-b-chord-labeled-eval
                Evaluate the tiny chord-labeled subset manifest and pitch-role summary contract.
  stage-b-generated-chord-eval
                Evaluate generated candidate report bridge against known chord metadata.
  stage-b-data-guide-generated-chord-eval
                Generate data-guide hybrid review package and evaluate its known chord metadata.
  stage-b-review-markdown-chord-eval
                Generate data-guide hybrid review package and write a combined chord-eval review markdown.
  stage-b-listening-review-notes
                Generate pending listening review notes from the combined chord-eval review flow.
  stage-b-full-review-notes
                Generate pending listening review notes from the full data-guide review manifest.
  stage-b-objective-midi-review
                Run objective note-level MIDI review on the full data-guide review manifest.
  stage-b-objective-flags-review-flow
                Attach objective MIDI flags to review notes and aggregate review priority.
  stage-b-listening-review-aggregate
                Aggregate pending or filled listening review notes into next-step signals.
  all           Run demo and tiny-compare.

Environment:
  PYTHON_BIN    Optional Python interpreter override.
  RUN_ID        Optional run id for tiny modes.
EOF
}

print_header() {
  printf '\n== %s ==\n' "$1"
}

run_status() {
  print_header "Git status"
  git branch --show-current
  git status -sb
}

run_quick() {
  print_header "Unit tests"
  "$PYTHON_BIN" -m unittest discover tests

  print_header "Compile checks"
  "$PYTHON_BIN" -m compileall \
    scripts/generate.py \
    scripts/control_tokens.py \
    scripts/checkpoint_utils.py \
    scripts/train_qlora.py \
    scripts/run_stage_a_tiny_overfit.py \
    scripts/compare_stage_a_tiny_modes.py \
    scripts/run_control_v1_tiny_overfit.py \
    scripts/stage_b_tokens.py \
    scripts/prepare_role_dataset.py \
    scripts/run_stage_b_window_tiny_overfit.py \
    scripts/run_stage_b_generation_probe.py \
    scripts/run_stage_b_raw_generation_repeatability_sweep.py \
    scripts/diagnose_stage_b_dead_air_outliers.py \
    scripts/diagnose_stage_b_seed_strict_margins.py \
    scripts/build_stage_b_margin_recovered_review_export.py \
    scripts/build_stage_b_margin_recovered_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_proxy_review.py \
    scripts/build_stage_b_margin_recovered_focused_package.py \
    scripts/review_stage_b_margin_recovered_focused_context.py \
    scripts/select_stage_b_margin_recovered_repair_candidate.py \
    scripts/summarize_stage_b_margin_recovered_pitch_vocab_sweep.py \
    scripts/build_stage_b_margin_recovered_pitch_vocab_focused_package.py \
    scripts/build_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    scripts/summarize_stage_b_margin_recovered_timing_repetition_repair.py \
    scripts/build_stage_b_margin_recovered_timing_repetition_focused_package.py \
    scripts/build_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py \
    scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_package.py \
    scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    scripts/fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_keep_stability.py \
    scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep.py \
    scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_sweep.py \
    scripts/render_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.py \
    scripts/fill_stage_b_duration_coverage_outside_soloing_repair_user_listening_review.py \
    scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.py \
    scripts/decide_stage_b_duration_coverage_outside_soloing_repair_next_step.py \
    scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.py \
    scripts/assess_stage_b_generic_base_readiness.py \
    scripts/check_stage_b_generic_base_manifest_contract.py \
    scripts/run_stage_b_generic_manifest_window_smoke.py \
    scripts/run_stage_b_generic_base_tiny_training_smoke.py \
    scripts/build_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison.py \
    scripts/audit_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence.py \
    scripts/repair_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity.py \
    scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep.py \
    scripts/run_stage_b_sampling_sweep.py \
    scripts/run_stage_b_coverage_ab_sweep.py \
    scripts/run_stage_b_pitch_mode_compare.py \
    scripts/run_stage_b_phrase_grammar_compare.py \
    scripts/run_stage_b_reference_stats.py \
    scripts/run_stage_b_motif_template_extraction.py \
    scripts/run_stage_b_data_motif_generation_compare.py \
    scripts/rank_stage_b_candidates.py \
    scripts/audit_brad_mehldau_dataset.py \
    scripts/audit_jazz_piano_dataset.py \
    scripts/build_jazz_training_manifests.py \
    scripts/run_manifest_prepare_smoke.py \
    scripts/audit_chord_progression_coverage.py \
    scripts/evaluate_chord_labeled_subset.py \
    scripts/evaluate_generated_candidate_chords.py \
    scripts/build_listening_review_notes.py \
    scripts/build_focused_listening_review_notes.py \
    scripts/summarize_listening_review_notes.py \
    scripts/build_focused_review_package.py \
    scripts/review_midi_note_objectives.py \
    scripts/train_stage_a_full.py \
    scripts/train_stage_a_adapter.py \
    inference/app \
    tests

  print_header "Diff whitespace check"
  git diff --check
}

run_demo() {
  run_quick
  print_header "MVP demo"
  bash scripts/run_mvp_demo.sh
}

run_tiny_prepare() {
  local run_id="${RUN_ID:-harness_prepare}"
  print_header "Tiny-overfit prepare"
  "$PYTHON_BIN" scripts/run_stage_a_tiny_overfit.py \
    --prepare_only \
    --sample_count 2 \
    --output_root outputs/stage_a_tiny_harness \
    --run_id "$run_id"
}

run_tiny_compare() {
  local run_id="${RUN_ID:-harness_compare}"
  print_header "Stage A tiny mode comparison"
  "$PYTHON_BIN" scripts/compare_stage_a_tiny_modes.py \
    --sample_count 3 \
    --epochs 200 \
    --lr 0.001 \
    --max_sequence 128 \
    --primer_max_tokens 24 \
    --num_samples 3 \
    --output_root outputs/stage_a_tiny_harness \
    --run_id "$run_id"
}

run_control_tiny() {
  local run_id="${RUN_ID:-harness_control_v1}"
  print_header "Stage A control_v1 tiny overfit"
  "$PYTHON_BIN" scripts/run_control_v1_tiny_overfit.py \
    --sample_count 1 \
    --epochs 200 \
    --lr 0.001 \
    --max_sequence 192 \
    --primer_max_tokens 96 \
    --num_samples 3 \
    --output_root outputs/stage_a_tiny_harness \
    --run_id "$run_id"
}

run_stage_b_window_prepare() {
  local run_id="${RUN_ID:-harness_stage_b_window_prepare}"
  print_header "Stage B window prepare"
  "$PYTHON_BIN" scripts/run_stage_b_window_tiny_overfit.py \
    --run_id "$run_id" \
    --max_files 1 \
    --prepare_only
}

run_stage_b_generation_probe() {
  local run_id="${RUN_ID:-harness_stage_b_generation_probe}"
  print_header "Stage B generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --max_files 1 \
    --epochs 50 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 5 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8 \
    --top_k 4 \
    --postprocess_overlap \
    --require_note_groups \
    --require_valid_sample \
    --require_strict_valid_sample
}

run_stage_b_raw_generation_repeatability() {
  local run_id="${RUN_ID:-harness_stage_b_raw_generation_repeatability}"
  local issue_number="${ISSUE_NUMBER:-0}"
  local max_files="${MAX_FILES:-2}"
  local min_source_files="${MIN_SOURCE_FILES:-2}"
  local seeds="${SEEDS:-17,23,31}"
  local epochs="${EPOCHS:-50}"
  local batch_size="${BATCH_SIZE:-8}"
  local max_sequence="${MAX_SEQUENCE:-96}"
  local num_samples="${NUM_SAMPLES:-3}"
  local top_k="${TOP_K:-4}"
  local temperature="${TEMPERATURE:-0.9}"
  local max_dead_air_outlier_rate="${MAX_DEAD_AIR_OUTLIER_RATE:-0.25}"
  local warning_min_strict_samples_per_seed="${WARNING_MIN_STRICT_SAMPLES_PER_SEED:-2}"
  print_header "Stage B raw generation repeatability sweep"
  "$PYTHON_BIN" scripts/run_stage_b_raw_generation_repeatability_sweep.py \
    --run_id "$run_id" \
    --issue_number "$issue_number" \
    --max_files "$max_files" \
    --seeds "$seeds" \
    --epochs "$epochs" \
    --batch_size "$batch_size" \
    --max_sequence "$max_sequence" \
    --num_samples "$num_samples" \
    --top_k "$top_k" \
    --temperature "$temperature" \
    --min_seed_count 3 \
    --min_source_files "$min_source_files" \
    --min_strict_samples_per_seed 1 \
    --min_overall_strict_rate 0.67 \
    --warning_min_strict_samples_per_seed "$warning_min_strict_samples_per_seed" \
    --dead_air_gate 0.8 \
    --max_dead_air_outlier_rate "$max_dead_air_outlier_rate" \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_dead_air_diagnostics() {
  local run_id="${RUN_ID:-harness_stage_b_dead_air_diagnostics}"
  local report_path="${REPORT_PATH:-outputs/stage_b_generation_probe/issue_224_stage_b_raw_generation_repeatability_final2_seed31_files2/report.json}"
  print_header "Stage B dead-air diagnostics"
  "$PYTHON_BIN" scripts/diagnose_stage_b_dead_air_outliers.py \
    --run_id "$run_id" \
    --report_path "$report_path" \
    --expected_outliers 1
}

run_stage_b_seed_strict_margin_diagnostics() {
  local run_id="${RUN_ID:-harness_stage_b_seed_strict_margin_diagnostics}"
  local summary_path="${SUMMARY_PATH:-outputs/stage_b_raw_generation_repeatability/issue_232_stage_b_larger_source_risk_boundary_files6/repeatability_summary.json}"
  print_header "Stage B seed strict margin diagnostics"
  "$PYTHON_BIN" scripts/diagnose_stage_b_seed_strict_margins.py \
    --run_id "$run_id" \
    --summary_path "$summary_path" \
    --warning_min_strict_samples_per_seed 2 \
    --expected_margin_warning_seeds 17
}

run_stage_b_margin_recovered_review_export() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_review_export}"
  local summary_path="${SUMMARY_PATH:-outputs/stage_b_raw_generation_repeatability/issue_238_stage_b_candidate_count_margin_recovery/repeatability_summary.json}"
  print_header "Stage B margin-recovered candidate review export"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_review_export.py \
    --run_id "$run_id" \
    --summary_path "$summary_path" \
    --expected_candidate_count 3
}

run_stage_b_margin_recovered_listening_notes() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_listening_notes}"
  local review_export="${REVIEW_EXPORT:-outputs/stage_b_margin_recovered_review_export/harness_stage_b_margin_recovered_review_export/candidate_review_export.json}"
  print_header "Stage B margin-recovered listening review notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_listening_notes.py \
    --run_id "$run_id" \
    --review_export "$review_export" \
    --expected_candidate_count 3
}

run_stage_b_margin_recovered_proxy_review_fill() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_proxy_review}"
  local review_notes="${REVIEW_NOTES:-outputs/stage_b_margin_recovered_listening_notes/harness_stage_b_margin_recovered_listening_notes/listening_review_notes_template.json}"
  print_header "Stage B margin-recovered MIDI proxy review fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_proxy_review.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_keep_candidate_id margin_recovered_rank_2_seed_31_sample_5
}

run_stage_b_margin_recovered_proxy_keep_focused_package() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_margin_recovered_proxy_review}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_proxy_keep_focused_package}"
  local review_notes="${REVIEW_NOTES:-outputs/stage_b_margin_recovered_proxy_review/${source_run_id}/listening_review_notes_proxy_filled.json}"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered MIDI proxy review fill"
    RUN_ID="$source_run_id" run_stage_b_margin_recovered_proxy_review_fill
  fi
  print_header "Stage B margin-recovered proxy keep focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_focused_package.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id margin_recovered_rank_2_seed_31_sample_5 \
    --min_candidates 1
}

run_stage_b_margin_recovered_focused_context_decision() {
  local focused_run_id="${FOCUSED_RUN_ID:-harness_stage_b_margin_recovered_proxy_keep_focused_package_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_focused_context_decision}"
  local focused_package="${FOCUSED_PACKAGE:-outputs/stage_b_margin_recovered_focused_package/${focused_run_id}/focused_review_package.json}"
  if [[ ! -f "$focused_package" ]]; then
    RUN_ID="$focused_run_id" run_stage_b_margin_recovered_proxy_keep_focused_package
  fi
  print_header "Stage B margin-recovered focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$run_id" \
    --focused_package "$focused_package" \
    --expected_candidate_id margin_recovered_rank_2_seed_31_sample_5 \
    --expected_decision needs_followup
}

run_stage_b_margin_recovered_focused_fallback_comparison() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_focused_fallback_package}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_focused_fallback_decision}"
  local review_notes="${REVIEW_NOTES:-outputs/stage_b_margin_recovered_proxy_review/harness_stage_b_margin_recovered_proxy_review/listening_review_notes_proxy_filled.json}"
  print_header "Stage B margin-recovered all-candidate focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_focused_package.py \
    --run_id "$package_run_id" \
    --review_notes "$review_notes" \
    --decision all \
    --min_candidates 3
  print_header "Stage B margin-recovered focused fallback comparison"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$run_id" \
    --focused_package "outputs/stage_b_margin_recovered_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_count 3
}

run_stage_b_margin_recovered_pitch_dead_air_repair() {
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_margin_recovered_pitch_dead_air_repair_generation}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_dead_air_repair_selection}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered pitch/dead-air repair generation"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$generation_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 31 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 12 \
    --temperature 0.9 \
    --top_k 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 254

  print_header "Stage B margin-recovered pitch/dead-air repair selection"
  "$PYTHON_BIN" scripts/select_stage_b_margin_recovered_repair_candidate.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${generation_run_id}/report.json" \
    --baseline_candidate_id margin_recovered_rank_2_seed_31_sample_5 \
    --baseline_dead_air 0.4444444444444444 \
    --baseline_unique_pitch_count 4 \
    --expected_sample_index 8 \
    --require_partial_repair
}

run_stage_b_margin_recovered_pitch_vocab_sweep() {
  local seed17_run_id="${SEED17_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_seed17_topk5_temp090_n24}"
  local seed31_run_id="${SEED31_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_seed31_topk5_temp090_n24}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_sweep}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered pitch vocabulary sweep seed 17"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed17_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 17 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 24 \
    --temperature 0.9 \
    --top_k 5 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 256

  print_header "Stage B margin-recovered pitch vocabulary sweep seed 31"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed31_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 31 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 24 \
    --temperature 0.9 \
    --top_k 5 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 256

  print_header "Stage B margin-recovered pitch vocabulary sweep summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_pitch_vocab_sweep.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed17_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed31_run_id}/report.json" \
    --require_qualified \
    --expected_source_run_id "$seed17_run_id" \
    --expected_sample_index 4
}

run_stage_b_margin_recovered_pitch_vocab_focused_context() {
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_sweep}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_context_decision}"
  local candidate_id="margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4"
  local sweep_summary="outputs/stage_b_margin_recovered_pitch_vocab_sweep/${sweep_run_id}/pitch_vocab_sweep_summary.json"
  if [[ ! -f "$sweep_summary" ]]; then
    print_header "Stage B margin-recovered pitch vocabulary sweep"
    RUN_ID="$sweep_run_id" run_stage_b_margin_recovered_pitch_vocab_sweep
  fi
  print_header "Stage B margin-recovered pitch vocabulary focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_pitch_vocab_focused_package.py \
    --run_id "$package_run_id" \
    --sweep_summary "$sweep_summary" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered pitch vocabulary focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_pitch_vocab_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_pitch_vocab_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_listening_notes}"
  local candidate_id="margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4"
  local package_path="outputs/stage_b_margin_recovered_pitch_vocab_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered pitch vocabulary focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_pitch_vocab_focused_context
  fi
  print_header "Stage B margin-recovered pitch vocabulary focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_pitch_vocab_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_pitch_vocab_focused_listening_fill}"
  local candidate_id="margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4"
  local review_notes="outputs/stage_b_margin_recovered_pitch_vocab_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered pitch vocabulary focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_pitch_vocab_focused_listening_notes
  fi
  print_header "Stage B margin-recovered pitch vocabulary focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_pitch_vocab_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision needs_followup
}

run_stage_b_margin_recovered_timing_repetition_repair() {
  local seed37_run_id="${SEED37_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_seed37_topk7_temp086_n48}"
  local seed41_run_id="${SEED41_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_seed41_topk7_temp086_n48}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_repair}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered timing/repetition repair seed 37"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed37_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 37 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.86 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 264

  print_header "Stage B margin-recovered timing/repetition repair seed 41"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed41_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 41 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.86 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 264

  print_header "Stage B margin-recovered timing/repetition repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_timing_repetition_repair.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed37_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed41_run_id}/report.json" \
    --require_qualified \
    --require_timing_repetition_improvement \
    --expected_source_run_id "$seed37_run_id" \
    --expected_sample_index 39
}

run_stage_b_margin_recovered_timing_repetition_focused_context() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_repair}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_context_decision}"
  local candidate_id="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39"
  local repair_summary="outputs/stage_b_margin_recovered_timing_repetition_repair/${repair_run_id}/timing_repetition_repair_summary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered timing/repetition repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_timing_repetition_repair
  fi
  print_header "Stage B margin-recovered timing/repetition focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_timing_repetition_focused_package.py \
    --run_id "$package_run_id" \
    --repair_summary "$repair_summary" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered timing/repetition focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_timing_repetition_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_timing_repetition_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_listening_notes}"
  local candidate_id="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39"
  local package_path="outputs/stage_b_margin_recovered_timing_repetition_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered timing/repetition focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_timing_repetition_focused_context
  fi
  print_header "Stage B margin-recovered timing/repetition focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_timing_repetition_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_timing_repetition_focused_listening_fill}"
  local candidate_id="margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39"
  local review_notes="outputs/stage_b_margin_recovered_timing_repetition_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered timing/repetition focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_timing_repetition_focused_listening_notes
  fi
  print_header "Stage B margin-recovered timing/repetition focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_timing_repetition_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision needs_followup
}

run_stage_b_margin_recovered_phrase_vocabulary_repair() {
  local seed43_run_id="${SEED43_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed43_topk7_temp082_n48}"
  local seed61_run_id="${SEED61_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed61_topk7_temp082_n48}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_repair}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  print_header "Stage B margin-recovered phrase/vocabulary repair seed 43"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed43_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 43 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.82 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 272

  print_header "Stage B margin-recovered phrase/vocabulary repair seed 61"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed61_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 61 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.82 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 272

  print_header "Stage B margin-recovered phrase/vocabulary repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py \
    --run_id "$run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed43_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed61_run_id}/report.json" \
    --require_qualified \
    --require_phrase_vocabulary_improvement \
    --expected_source_run_id "$seed43_run_id" \
    --expected_sample_index 43
}

run_stage_b_margin_recovered_phrase_vocabulary_focused_context() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_repair}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_context_decision}"
  local candidate_id="margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_repair
  fi
  print_header "Stage B margin-recovered phrase/vocabulary focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_package.py \
    --run_id "$package_run_id" \
    --repair_summary "$repair_summary" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered phrase/vocabulary focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes}"
  local candidate_id="margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43"
  local package_path="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_phrase_vocabulary_focused_context
  fi
  print_header "Stage B margin-recovered phrase/vocabulary focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill}"
  local candidate_id="margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43"
  local review_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes
  fi
  print_header "Stage B margin-recovered phrase/vocabulary focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep
}

run_stage_b_margin_recovered_phrase_vocabulary_keep_stability() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_repair}"
  local fill_run_id="${FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_keep_stability}"
  local candidate_id="margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json"
  local filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${fill_run_id}/focused_listening_review_notes_filled.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_repair
  fi
  if [[ ! -f "$filled_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary focused listening fill"
    RUN_ID="$fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill
  fi
  print_header "Stage B margin-recovered phrase/vocabulary keep stability"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_keep_stability.py \
    --run_id "$run_id" \
    --repair_summary "$repair_summary" \
    --filled_notes "$filled_notes" \
    --expected_selected_candidate_id "$candidate_id" \
    --min_qualified_candidates 2 \
    --require_qualified_peer
}

run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_context() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_repair}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_context_decision}"
  local candidate_id="margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_repair
  fi
  print_header "Stage B margin-recovered phrase/vocabulary peer focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_package.py \
    --run_id "$package_run_id" \
    --repair_summary "$repair_summary" \
    --candidate_id "$candidate_id" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered phrase/vocabulary peer focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_notes}"
  local candidate_id="margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25"
  local package_path="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary peer focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_context
  fi
  print_header "Stage B margin-recovered phrase/vocabulary peer focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill}"
  local candidate_id="margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25"
  local review_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary peer focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_notes
  fi
  print_header "Stage B margin-recovered phrase/vocabulary peer focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep
}

run_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep() {
  local stability_run_id="${STABILITY_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_keep_stability}"
  local selected_fill_run_id="${SELECTED_FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill}"
  local peer_fill_run_id="${PEER_FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep}"
  local stability_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_keep_stability/${stability_run_id}/keep_stability_summary.json"
  local selected_filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${selected_fill_run_id}/focused_listening_review_notes_filled.json"
  local peer_filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${peer_fill_run_id}/focused_listening_review_notes_filled.json"
  if [[ ! -f "$stability_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary keep stability"
    RUN_ID="$stability_run_id" run_stage_b_margin_recovered_phrase_vocabulary_keep_stability
  fi
  if [[ ! -f "$selected_filled_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary focused listening fill"
    RUN_ID="$selected_fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill
  fi
  if [[ ! -f "$peer_filled_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary peer focused listening fill"
    RUN_ID="$peer_fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill
  fi
  print_header "Stage B margin-recovered phrase/vocabulary two-candidate keep"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep.py \
    --run_id "$run_id" \
    --stability_summary "$stability_summary" \
    --selected_filled_notes "$selected_filled_notes" \
    --peer_filled_notes "$peer_filled_notes" \
    --min_keep_candidates 2 \
    --min_qualified_sources 2 \
    --max_qualified_rate 0.05 \
    --require_not_human_audio_review
}

run_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison() {
  local two_candidate_run_id="${TWO_CANDIDATE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep}"
  local selected_fill_run_id="${SELECTED_FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill}"
  local peer_fill_run_id="${PEER_FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison}"
  local two_candidate_keep="outputs/stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep/${two_candidate_run_id}/two_candidate_keep_summary.json"
  local selected_filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${selected_fill_run_id}/focused_listening_review_notes_filled.json"
  local peer_filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${peer_fill_run_id}/focused_listening_review_notes_filled.json"
  if [[ ! -f "$two_candidate_keep" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary two-candidate keep"
    RUN_ID="$two_candidate_run_id" run_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep
  fi
  if [[ ! -f "$selected_filled_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary focused listening fill"
    RUN_ID="$selected_fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill
  fi
  if [[ ! -f "$peer_filled_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary peer focused listening fill"
    RUN_ID="$peer_fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill
  fi
  print_header "Stage B margin-recovered phrase/vocabulary human listening comparison boundary"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison.py \
    --run_id "$run_id" \
    --two_candidate_keep "$two_candidate_keep" \
    --selected_filled_notes "$selected_filled_notes" \
    --peer_filled_notes "$peer_filled_notes" \
    --min_candidates 2 \
    --require_pending \
    --require_no_preference \
    --expect_note_sequence_match
}

run_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_repair}"
  local human_run_id="${HUMAN_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence}"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json"
  local human_comparison="outputs/stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison/${human_run_id}/human_listening_comparison_boundary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_repair
  fi
  if [[ ! -f "$human_comparison" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary human listening comparison boundary"
    RUN_ID="$human_run_id" run_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison
  fi
  print_header "Stage B margin-recovered phrase/vocabulary duplicate source divergence"
  "$PYTHON_BIN" scripts/audit_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence.py \
    --run_id "$run_id" \
    --repair_summary "$repair_summary" \
    --human_comparison "$human_comparison" \
    --require_shared_sample_seed \
    --require_duplicate_output \
    --expected_boundary shared_sample_seed_duplicate_output
}

run_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_repair}"
  local duplicate_run_id="${DUPLICATE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity}"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json"
  local duplicate_audit="outputs/stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence/${duplicate_run_id}/duplicate_source_divergence_audit.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_repair
  fi
  if [[ ! -f "$duplicate_audit" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary duplicate source divergence"
    RUN_ID="$duplicate_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence
  fi
  print_header "Stage B margin-recovered phrase/vocabulary sample-seed diversity repair"
  "$PYTHON_BIN" scripts/repair_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity.py \
    --run_id "$run_id" \
    --repair_summary "$repair_summary" \
    --duplicate_audit "$duplicate_audit" \
    --expected_boundary single_distinct_sample_seed_keep_support \
    --require_duplicate_demoted
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep() {
  local seed109_run_id="${SEED109_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed109_topk7_temp082_n48}"
  local seed157_run_id="${SEED157_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed157_topk7_temp082_n48}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_repair}"
  local sample_seed_repair_run_id="${SAMPLE_SEED_REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  local sample_seed_repair="outputs/stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity_repair/${sample_seed_repair_run_id}/sample_seed_diversity_repair.json"
  if [[ ! -f "$sample_seed_repair" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary sample-seed diversity repair"
    RUN_ID="$sample_seed_repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity
  fi
  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed repair seed 109"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed109_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 109 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.82 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 298

  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed repair seed 157"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$seed157_run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --skip_prepare \
    --skip_train \
    --seed 157 \
    --max_files 6 \
    --max_sequence 96 \
    --num_samples 48 \
    --temperature 0.82 \
    --top_k 7 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --require_valid_sample \
    --require_strict_valid_sample \
    --require_note_groups \
    --issue_number 298

  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py \
    --run_id "$repair_run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed109_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed157_run_id}/report.json"

  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed sweep"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep.py \
    --run_id "$run_id" \
    --repair_summary "outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json" \
    --sample_seed_repair "$sample_seed_repair" \
    --min_candidates 96 \
    --expected_blocked_seed 85
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_context() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_repair}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_context_decision}"
  local candidate_id="margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_repair/${repair_run_id}/phrase_vocabulary_repair_summary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed sweep"
    REPAIR_RUN_ID="$repair_run_id" RUN_ID="$sweep_run_id" run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep
  fi
  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_package.py \
    --run_id "$package_run_id" \
    --repair_summary "$repair_summary" \
    --candidate_id "$candidate_id" \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_notes}"
  local candidate_id="margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47"
  local package_path="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_context
  fi
  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_fill() {
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_fill}"
  local candidate_id="margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47"
  local review_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused listening notes"
    RUN_ID="$notes_run_id" run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_notes
  fi
  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision needs_followup
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker() {
  local fill_run_id="${FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_fill}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker}"
  local filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${fill_run_id}/focused_listening_review_notes_filled.json"
  if [[ ! -f "$filled_notes" ]]; then
    print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed focused listening fill"
    RUN_ID="$fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_fill
  fi
  print_header "Stage B margin-recovered phrase/vocabulary distinct sample-seed remaining blocker"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker.py \
    --run_id "$run_id" \
    --filled_notes "$filled_notes" \
    --expected_decision needs_followup \
    --require_remaining_blockers
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker_repair_sweep() {
  local seed181_run_id="${SEED181_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed181_topk8_temp090_n48}"
  local seed223_run_id="${SEED223_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed223_topk8_temp086_n48}"
  local repair_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker_repair}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  if [[ ! -f "outputs/stage_b_generation_probe/${seed181_run_id}/report.json" ]]; then
    print_header "Stage B distinct sample-seed remaining blocker repair seed 181"
    "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
      --run_id "$seed181_run_id" \
      --checkpoint_dir "$checkpoint_dir" \
      --skip_prepare \
      --skip_train \
      --seed 181 \
      --max_files 6 \
      --max_sequence 96 \
      --num_samples 48 \
      --temperature 0.90 \
      --top_k 8 \
      --postprocess_overlap \
      --max_simultaneous_notes 2 \
      --require_valid_sample \
      --require_strict_valid_sample \
      --require_note_groups \
      --issue_number 308
  fi
  if [[ ! -f "outputs/stage_b_generation_probe/${seed223_run_id}/report.json" ]]; then
    print_header "Stage B distinct sample-seed remaining blocker repair seed 223"
    "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
      --run_id "$seed223_run_id" \
      --checkpoint_dir "$checkpoint_dir" \
      --skip_prepare \
      --skip_train \
      --seed 223 \
      --max_files 6 \
      --max_sequence 96 \
      --num_samples 48 \
      --temperature 0.86 \
      --top_k 8 \
      --postprocess_overlap \
      --max_simultaneous_notes 2 \
      --require_valid_sample \
      --require_strict_valid_sample \
      --require_note_groups \
      --issue_number 308
  fi
  print_header "Stage B distinct sample-seed remaining blocker repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py \
    --run_id "$repair_run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed181_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed223_run_id}/report.json" \
    --previous_candidate_id margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47 \
    --previous_dead_air 0.375 \
    --previous_unique_pitch_count 6 \
    --previous_note_count 13 \
    --previous_adjacent_pitch_repeats 1 \
    --previous_max_interval 3 \
    --min_unique_pitch_count 7 \
    --max_dead_air_ratio_exclusive 0.376 \
    --min_note_count 12 \
    --max_simultaneous_notes 1 \
    --max_duplicated_3_note_chunks 0 \
    --max_adjacent_pitch_repeats_exclusive 1 \
    --max_interval_exclusive 12
}

run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_dead_air_adjacent_repair() {
  local seed269_run_id="${SEED269_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed269_topk7_temp080_n48}"
  local seed311_run_id="${SEED311_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_seed311_topk7_temp078_n48}"
  local repair_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_dead_air_adjacent_repair}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  if [[ ! -f "outputs/stage_b_generation_probe/${seed269_run_id}/report.json" ]]; then
    print_header "Stage B distinct sample-seed dead-air/adjacent repair seed 269"
    "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
      --run_id "$seed269_run_id" \
      --checkpoint_dir "$checkpoint_dir" \
      --skip_prepare \
      --skip_train \
      --seed 269 \
      --max_files 6 \
      --max_sequence 96 \
      --num_samples 48 \
      --temperature 0.80 \
      --top_k 7 \
      --postprocess_overlap \
      --max_simultaneous_notes 2 \
      --require_valid_sample \
      --require_strict_valid_sample \
      --require_note_groups \
      --issue_number 310
  fi
  if [[ ! -f "outputs/stage_b_generation_probe/${seed311_run_id}/report.json" ]]; then
    print_header "Stage B distinct sample-seed dead-air/adjacent repair seed 311"
    "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
      --run_id "$seed311_run_id" \
      --checkpoint_dir "$checkpoint_dir" \
      --skip_prepare \
      --skip_train \
      --seed 311 \
      --max_files 6 \
      --max_sequence 96 \
      --num_samples 48 \
      --temperature 0.78 \
      --top_k 7 \
      --postprocess_overlap \
      --max_simultaneous_notes 2 \
      --require_valid_sample \
      --require_strict_valid_sample \
      --require_note_groups \
      --issue_number 310
  fi
  print_header "Stage B distinct sample-seed dead-air/adjacent repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py \
    --run_id "$repair_run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed269_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed311_run_id}/report.json" \
    --previous_candidate_id margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47 \
    --previous_dead_air 0.375 \
    --previous_unique_pitch_count 6 \
    --previous_note_count 13 \
    --previous_adjacent_pitch_repeats 1 \
    --previous_max_interval 3 \
    --min_unique_pitch_count 7 \
    --max_dead_air_ratio_exclusive 0.376 \
    --min_note_count 12 \
    --max_simultaneous_notes 1 \
    --max_duplicated_3_note_chunks 0 \
    --max_adjacent_pitch_repeats_exclusive 1 \
    --max_interval_exclusive 12
}

run_stage_b_margin_recovered_phrase_vocabulary_coverage_aware_adjacent_constrained_repair() {
  local seed353_run_id="${SEED353_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_coverage_adjacent_seed353_groups8}"
  local seed397_run_id="${SEED397_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocab_coverage_adjacent_seed397_groups10_duration}"
  local repair_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_coverage_aware_adjacent_constrained_repair}"
  local checkpoint_dir="${CHECKPOINT_DIR:-outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints}"
  if [[ ! -f "outputs/stage_b_generation_probe/${seed353_run_id}/report.json" ]]; then
    print_header "Stage B coverage-aware adjacent constrained seed 353"
    "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
      --run_id "$seed353_run_id" \
      --checkpoint_dir "$checkpoint_dir" \
      --skip_prepare \
      --skip_train \
      --seed 353 \
      --max_files 6 \
      --max_sequence 96 \
      --num_samples 24 \
      --temperature 0.82 \
      --top_k 7 \
      --generation_mode constrained \
      --coverage_aware_positions \
      --coverage_position_window 1 \
      --chord_aware_pitches \
      --chord_pitch_mode tones_tensions \
      --chord_pitch_repeat_window 4 \
      --constrained_note_groups_per_bar 8 \
      --postprocess_overlap \
      --max_simultaneous_notes 2 \
      --require_valid_sample \
      --require_strict_valid_sample \
      --require_note_groups \
      --issue_number 312
  fi
  if [[ ! -f "outputs/stage_b_generation_probe/${seed397_run_id}/report.json" ]]; then
    print_header "Stage B coverage-aware adjacent constrained seed 397"
    "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
      --run_id "$seed397_run_id" \
      --checkpoint_dir "$checkpoint_dir" \
      --skip_prepare \
      --skip_train \
      --seed 397 \
      --max_files 6 \
      --max_sequence 96 \
      --num_samples 24 \
      --temperature 0.82 \
      --top_k 7 \
      --generation_mode constrained \
      --coverage_aware_positions \
      --coverage_position_window 1 \
      --chord_aware_pitches \
      --chord_pitch_mode tones_tensions \
      --chord_pitch_repeat_window 4 \
      --jazz_duration_tokens \
      --constrained_note_groups_per_bar 10 \
      --postprocess_overlap \
      --max_simultaneous_notes 2 \
      --require_valid_sample \
      --require_strict_valid_sample \
      --require_note_groups \
      --issue_number 312
  fi
  print_header "Stage B coverage-aware adjacent constrained repair summary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py \
    --run_id "$repair_run_id" \
    --report_path "outputs/stage_b_generation_probe/${seed353_run_id}/report.json" \
    --report_path "outputs/stage_b_generation_probe/${seed397_run_id}/report.json" \
    --previous_candidate_id margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47 \
    --previous_dead_air 0.375 \
    --previous_unique_pitch_count 6 \
    --previous_note_count 13 \
    --previous_adjacent_pitch_repeats 1 \
    --previous_max_interval 3 \
    --min_unique_pitch_count 7 \
    --max_dead_air_ratio_exclusive 0.376 \
    --min_note_count 12 \
    --max_simultaneous_notes 1 \
    --max_duplicated_3_note_chunks 0 \
    --max_adjacent_pitch_repeats_exclusive 1 \
    --max_interval_exclusive 12
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair() {
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair}"
  print_header "Stage B duration/coverage fill repair"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair.py \
    --run_id "$run_id" \
    --summary_path "outputs/stage_b_margin_recovered_phrase_vocabulary_repair/harness_stage_b_margin_recovered_phrase_vocabulary_coverage_aware_adjacent_constrained_repair/phrase_vocabulary_repair_summary.json" \
    --fill_max_additions 4 \
    --fill_max_additions 6 \
    --fill_max_additions 8 \
    --fill_max_additions 10 \
    --dead_air_threshold_sec 0.18 \
    --simultaneous_limit 1 \
    --min_unique_pitch_count 7 \
    --max_dead_air_ratio_exclusive 0.376 \
    --min_note_count 12 \
    --max_simultaneous_notes 1 \
    --max_duplicated_3_note_chunks 0 \
    --max_adjacent_pitch_repeats_exclusive 1 \
    --max_interval_exclusive 12 \
    --require_qualified \
    --require_dead_air_improvement \
    --expected_fill_addition_count 6
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_context() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_package}"
  local decision_run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_context_decision}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local repair_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair/${repair_run_id}/duration_coverage_fill_repair_summary.json"
  if [[ ! -f "$repair_summary" ]]; then
    print_header "Stage B duration/coverage fill repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair
  fi
  print_header "Stage B duration/coverage fill focused package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_package.py \
    --run_id "$package_run_id" \
    --repair_summary "$repair_summary" \
    --decision phrase_vocabulary_duration_coverage_fill_qualified \
    --expected_candidate_id "$candidate_id"

  print_header "Stage B duration/coverage fill focused context decision"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_focused_context.py \
    --run_id "$decision_run_id" \
    --focused_package "outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_notes() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_context_decision}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_notes}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local package_path="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/${package_run_id}/focused_review_package.json"
  local decision_path="outputs/stage_b_margin_recovered_focused_context_decision/${decision_run_id}/focused_context_decision.json"
  if [[ ! -f "$package_path" || ! -f "$decision_path" ]]; then
    print_header "Stage B duration/coverage fill focused context"
    PACKAGE_RUN_ID="$package_run_id" RUN_ID="$decision_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_context
  fi
  print_header "Stage B duration/coverage fill focused listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --focused_package "$package_path" \
    --focused_context_decision "$decision_path" \
    --expected_candidate_id "$candidate_id" \
    --expected_prior_decision keep_for_focused_listening
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_fill() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_package}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_context_decision_with_midi_coverage}"
  local notes_run_id="${NOTES_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_notes_with_midi_coverage}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_fill}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local review_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/${notes_run_id}/focused_listening_review_notes_template.json"
  if [[ ! -f "$review_notes" ]]; then
    print_header "Stage B duration/coverage fill focused listening notes"
    PACKAGE_RUN_ID="$package_run_id" DECISION_RUN_ID="$decision_run_id" RUN_ID="$notes_run_id" \
      run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_notes
  fi
  print_header "Stage B duration/coverage fill focused listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_decision keep
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair}"
  local fill_run_id="${FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_fill}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local duration_fill_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair/${repair_run_id}/duration_coverage_fill_repair_summary.json"
  local filled_notes="outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/${fill_run_id}/focused_listening_review_notes_filled.json"
  if [[ ! -f "$duration_fill_summary" ]]; then
    print_header "Stage B duration/coverage fill repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair
  fi
  if [[ ! -f "$filled_notes" ]]; then
    print_header "Stage B duration/coverage fill focused listening fill"
    RUN_ID="$fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_fill
  fi
  print_header "Stage B duration/coverage fill keep consolidation"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation.py \
    --run_id "$run_id" \
    --duration_fill_summary "$duration_fill_summary" \
    --filled_notes "$filled_notes" \
    --expected_candidate_id "$candidate_id" \
    --expected_boundary single_postprocess_candidate_keep_support \
    --require_not_human_audio_review \
    --require_postprocess_claim_boundary postprocess_duration_coverage_fill_candidate
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary() {
  local keep_run_id="${KEEP_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local keep_consolidation="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation/${keep_run_id}/duration_coverage_fill_keep_consolidation.json"
  local duration_fill_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair/${repair_run_id}/duration_coverage_fill_repair_summary.json"
  if [[ ! -f "$keep_consolidation" ]]; then
    print_header "Stage B duration/coverage fill keep consolidation"
    RUN_ID="$keep_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation
  fi
  if [[ ! -f "$duration_fill_summary" ]]; then
    print_header "Stage B duration/coverage fill repair"
    RUN_ID="$repair_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair
  fi
  print_header "Stage B duration/coverage fill human/audio boundary"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary.py \
    --run_id "$run_id" \
    --keep_consolidation "$keep_consolidation" \
    --duration_fill_summary "$duration_fill_summary" \
    --expected_candidate_id "$candidate_id" \
    --require_pending \
    --require_no_preference \
    --expect_distinct_midi_content
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_guard() {
  local boundary_run_id="${BOUNDARY_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_guard}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local human_audio_boundary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary/${boundary_run_id}/duration_coverage_fill_human_audio_boundary.json"
  if [[ ! -f "$human_audio_boundary" ]]; then
    print_header "Stage B duration/coverage fill human/audio boundary"
    RUN_ID="$boundary_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary
  fi
  print_header "Stage B duration/coverage fill human/audio review input guard"
  "$PYTHON_BIN" scripts/fill_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review.py \
    --run_id "$run_id" \
    --human_audio_boundary "$human_audio_boundary" \
    --expected_candidate_id "$candidate_id" \
    --require_pending_without_input \
    --require_no_preference_without_input
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package() {
  local boundary_run_id="${BOUNDARY_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary}"
  local guard_run_id="${GUARD_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_guard}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local human_audio_boundary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary/${boundary_run_id}/duration_coverage_fill_human_audio_boundary.json"
  local review_fill_guard="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_fill/${guard_run_id}/duration_coverage_fill_human_audio_review_fill.json"
  if [[ ! -f "$human_audio_boundary" ]]; then
    print_header "Stage B duration/coverage fill human/audio boundary"
    RUN_ID="$boundary_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary
  fi
  if [[ ! -f "$review_fill_guard" ]]; then
    print_header "Stage B duration/coverage fill human/audio review input guard"
    RUN_ID="$guard_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_guard
  fi
  print_header "Stage B duration/coverage fill audio review package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package.py \
    --run_id "$run_id" \
    --human_audio_boundary "$human_audio_boundary" \
    --review_fill_guard "$review_fill_guard" \
    --expected_candidate_id "$candidate_id" \
    --require_files_exist \
    --require_no_preference
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local audio_review_package="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package/${package_run_id}/duration_coverage_fill_audio_review_package.json"
  if [[ ! -f "$audio_review_package" ]]; then
    print_header "Stage B duration/coverage fill audio review package"
    RUN_ID="$package_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package
  fi
  print_header "Stage B duration/coverage fill MIDI evidence review"
  "$PYTHON_BIN" scripts/review_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence.py \
    --run_id "$run_id" \
    --audio_review_package "$audio_review_package" \
    --expected_candidate_id "$candidate_id" \
    --expected_preference duration_coverage_fill_keep \
    --require_no_human_audio_preference \
    --require_audio_not_rendered
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation() {
  local review_run_id="${REVIEW_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local midi_evidence_review="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review/${review_run_id}/duration_coverage_fill_midi_evidence_review.json"
  if [[ ! -f "$midi_evidence_review" ]]; then
    print_header "Stage B duration/coverage fill MIDI evidence review"
    RUN_ID="$review_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review
  fi
  print_header "Stage B duration/coverage fill MIDI evidence consolidation"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation.py \
    --run_id "$run_id" \
    --midi_evidence_review "$midi_evidence_review" \
    --expected_candidate_id "$candidate_id" \
    --expected_boundary midi_evidence_preference_support \
    --require_no_human_audio_preference
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary() {
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local midi_evidence_consolidation="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation/${consolidation_run_id}/duration_coverage_fill_midi_evidence_consolidation.json"
  if [[ ! -f "$midi_evidence_consolidation" ]]; then
    print_header "Stage B duration/coverage fill MIDI evidence consolidation"
    RUN_ID="$consolidation_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation
  fi
  print_header "Stage B duration/coverage fill external human/audio boundary"
  "$PYTHON_BIN" scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary.py \
    --run_id "$run_id" \
    --midi_evidence_consolidation "$midi_evidence_consolidation" \
    --expected_candidate_id "$candidate_id" \
    --expected_boundary external_human_audio_review_required_for_human_preference_claim \
    --require_no_human_audio_preference \
    --require_pending_external_review
}

run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package() {
  local external_boundary_run_id="${EXTERNAL_BOUNDARY_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package}"
  local candidate_id="margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6"
  local external_boundary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary/${external_boundary_run_id}/duration_coverage_fill_external_human_audio_boundary.json"
  local audio_review_package="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package/${audio_package_run_id}/duration_coverage_fill_audio_review_package.json"
  if [[ ! -f "$external_boundary" ]]; then
    print_header "Stage B duration/coverage fill external human/audio boundary"
    RUN_ID="$external_boundary_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary
  fi
  if [[ ! -f "$audio_review_package" ]]; then
    print_header "Stage B duration/coverage fill audio review package"
    RUN_ID="$audio_package_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package
  fi
  print_header "Stage B duration/coverage fill local audio render package"
  "$PYTHON_BIN" scripts/build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package.py \
    --run_id "$run_id" \
    --external_human_audio_boundary "$external_boundary" \
    --audio_review_package "$audio_review_package" \
    --expected_candidate_id "$candidate_id" \
    --require_required_midi_exists \
    --require_no_audio_claim
}

run_stage_b_local_audio_render_tooling() {
  local run_id="${RUN_ID:-harness_stage_b_local_audio_render_tooling}"
  print_header "Stage B local audio render tooling"
  "$PYTHON_BIN" scripts/check_stage_b_local_audio_render_tooling.py \
    --run_id "$run_id" \
    --require_no_system_modification
}

run_stage_b_renderer_path_decision() {
  local tooling_run_id="${TOOLING_RUN_ID:-harness_stage_b_local_audio_render_tooling}"
  local run_id="${RUN_ID:-harness_stage_b_renderer_path_decision}"
  local tooling_report="outputs/stage_b_local_audio_render_tooling/${tooling_run_id}/stage_b_local_audio_render_tooling.json"
  if [[ ! -f "$tooling_report" ]]; then
    print_header "Stage B local audio render tooling"
    RUN_ID="$tooling_run_id" run_stage_b_local_audio_render_tooling
  fi
  print_header "Stage B renderer path decision"
  "$PYTHON_BIN" scripts/decide_stage_b_renderer_path.py \
    --run_id "$run_id" \
    --tooling_report "$tooling_report" \
    --expected_decision renderer_path_or_install_approval_required \
    --require_no_execution
}

run_stage_b_local_audio_render_attempt() {
  local local_package_run_id="${LOCAL_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_local_audio_render_attempt}"
  local local_audio_render_package="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package/${local_package_run_id}/duration_coverage_fill_local_audio_render_package.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$local_audio_render_package" ]]; then
    print_header "Stage B duration/coverage fill local audio render package"
    RUN_ID="$local_package_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package
  fi
  print_header "Stage B duration/coverage fill local audio render attempt"
  "$PYTHON_BIN" scripts/render_stage_b_duration_coverage_fill_audio.py \
    --run_id "$run_id" \
    --local_audio_render_package "$local_audio_render_package" \
    --soundfont "$soundfont" \
    --expected_file_count 2 \
    --require_no_quality_claim
}

run_stage_b_user_listening_review_fill() {
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_duration_coverage_fill_local_audio_render_attempt}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_user_listening_review_fill}"
  local audio_render_report="outputs/stage_b_duration_coverage_fill_local_audio_render_attempt/${audio_render_run_id}/stage_b_duration_coverage_fill_local_audio_render_attempt.json"
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B duration/coverage fill local audio render attempt"
    RUN_ID="$audio_render_run_id" run_stage_b_local_audio_render_attempt
  fi
  print_header "Stage B duration/coverage fill user listening review fill"
  "$PYTHON_BIN" scripts/fill_stage_b_duration_coverage_user_listening_review.py \
    --run_id "$run_id" \
    --audio_render_report "$audio_render_report" \
    --reviewer "user" \
    --preference duration_coverage_fill_keep \
    --timing duration_coverage_fill_keep \
    --phrase duration_coverage_fill_keep \
    --vocabulary duration_coverage_fill_keep \
    --source_assessment "source sounds like random notes and is hard to understand" \
    --fill_assessment "fill sounds much more jazz-like as soloing" \
    --notes "user listened to rendered WAV files and preferred the duration coverage fill candidate" \
    --expected_preference duration_coverage_fill_keep \
    --require_human_audio_preference \
    --require_no_broad_quality_claim
}

run_stage_b_user_listening_review_consolidation() {
  local midi_evidence_run_id="${MIDI_EVIDENCE_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_duration_coverage_fill_local_audio_render_attempt}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_duration_coverage_fill_user_listening_review_fill}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_user_listening_review_consolidation}"
  local midi_evidence="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation/${midi_evidence_run_id}/duration_coverage_fill_midi_evidence_consolidation.json"
  local audio_render_report="outputs/stage_b_duration_coverage_fill_local_audio_render_attempt/${audio_render_run_id}/stage_b_duration_coverage_fill_local_audio_render_attempt.json"
  local user_review="outputs/stage_b_duration_coverage_fill_user_listening_review_fill/${user_review_run_id}/stage_b_duration_coverage_fill_user_listening_review_fill.json"
  if [[ ! -f "$midi_evidence" ]]; then
    print_header "Stage B duration/coverage fill MIDI evidence consolidation"
    RUN_ID="$midi_evidence_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation
  fi
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B duration/coverage fill local audio render attempt"
    RUN_ID="$audio_render_run_id" run_stage_b_local_audio_render_attempt
  fi
  if [[ ! -f "$user_review" ]]; then
    print_header "Stage B duration/coverage fill user listening review fill"
    RUN_ID="$user_review_run_id" run_stage_b_user_listening_review_fill
  fi
  print_header "Stage B duration/coverage fill user listening review consolidation"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_user_listening_consolidation.py \
    --run_id "$run_id" \
    --midi_evidence_consolidation "$midi_evidence" \
    --audio_render_attempt "$audio_render_report" \
    --user_listening_review_fill "$user_review" \
    --expected_boundary midi_evidence_and_single_user_listening_support_duration_coverage_fill_keep \
    --expected_preferred_candidate duration_coverage_fill_keep \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_next_decision() {
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_duration_coverage_fill_user_listening_review_consolidation}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_next_decision}"
  local consolidation="outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation/${consolidation_run_id}/stage_b_duration_coverage_fill_user_listening_review_consolidation.json"
  if [[ ! -f "$consolidation" ]]; then
    print_header "Stage B duration/coverage fill user listening review consolidation"
    RUN_ID="$consolidation_run_id" run_stage_b_user_listening_review_consolidation
  fi
  print_header "Stage B duration/coverage fill next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_duration_coverage_next_step.py \
    --run_id "$run_id" \
    --consolidation "$consolidation" \
    --expected_next_boundary broader_repeatability_sweep \
    --require_auto_progress_allowed \
    --require_no_critical_user_input
}

run_stage_b_duration_coverage_broader_repeatability_sweep() {
  local next_decision_run_id="${NEXT_DECISION_RUN_ID:-harness_stage_b_duration_coverage_fill_next_decision}"
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_duration_coverage_fill_user_listening_review_consolidation}"
  local duration_fill_run_id="${DURATION_FILL_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair}"
  local distinct_sweep_run_id="${DISTINCT_SWEEP_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_broader_repeatability_sweep}"
  local next_decision="outputs/stage_b_duration_coverage_fill_next_decision/${next_decision_run_id}/stage_b_duration_coverage_fill_next_decision.json"
  local consolidation="outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation/${consolidation_run_id}/stage_b_duration_coverage_fill_user_listening_review_consolidation.json"
  local duration_fill_summary="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair/${duration_fill_run_id}/duration_coverage_fill_repair_summary.json"
  local distinct_sweep="outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep/${distinct_sweep_run_id}/distinct_sample_seed_sweep_summary.json"
  if [[ ! -f "$next_decision" ]]; then
    print_header "Stage B duration/coverage fill next decision"
    RUN_ID="$next_decision_run_id" run_stage_b_duration_coverage_next_decision
  fi
  if [[ ! -f "$consolidation" ]]; then
    print_header "Stage B duration/coverage fill user listening review consolidation"
    RUN_ID="$consolidation_run_id" run_stage_b_user_listening_review_consolidation
  fi
  if [[ ! -f "$duration_fill_summary" ]]; then
    print_header "Stage B duration/coverage fill repair"
    RUN_ID="$duration_fill_run_id" run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair
  fi
  if [[ ! -f "$distinct_sweep" ]]; then
    print_header "Stage B distinct sample-seed sweep"
    RUN_ID="$distinct_sweep_run_id" run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep
  fi
  print_header "Stage B duration/coverage fill broader repeatability sweep"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_broader_repeatability_sweep.py \
    --run_id "$run_id" \
    --next_decision "$next_decision" \
    --user_listening_consolidation "$consolidation" \
    --duration_fill_summary "$duration_fill_summary" \
    --distinct_sample_seed_sweep "$distinct_sweep" \
    --expected_boundary qualified_gate_repeatability_with_partial_dead_air_gain \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_dead_air_gain_repeatability_repair() {
  local broader_repeatability_run_id="${BROADER_REPEATABILITY_RUN_ID:-harness_stage_b_duration_coverage_fill_broader_repeatability_sweep}"
  local distinct_sweep_run_id="${DISTINCT_SWEEP_RUN_ID:-harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair}"
  local broader_repeatability="outputs/stage_b_duration_coverage_fill_broader_repeatability_sweep/${broader_repeatability_run_id}/stage_b_duration_coverage_fill_broader_repeatability_sweep.json"
  local distinct_sweep="outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep/${distinct_sweep_run_id}/distinct_sample_seed_sweep_summary.json"
  if [[ ! -f "$broader_repeatability" ]]; then
    print_header "Stage B duration/coverage fill broader repeatability sweep"
    RUN_ID="$broader_repeatability_run_id" run_stage_b_duration_coverage_broader_repeatability_sweep
  fi
  if [[ ! -f "$distinct_sweep" ]]; then
    print_header "Stage B distinct sample-seed sweep"
    RUN_ID="$distinct_sweep_run_id" run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep
  fi
  print_header "Stage B duration/coverage fill dead-air gain repeatability repair"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.py \
    --run_id "$run_id" \
    --broader_repeatability_sweep "$broader_repeatability" \
    --distinct_sample_seed_sweep "$distinct_sweep" \
    --expected_boundary qualified_gate_repeatability_with_dead_air_gain \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_repeatability_consolidation() {
  local user_consolidation_run_id="${USER_CONSOLIDATION_RUN_ID:-harness_stage_b_duration_coverage_fill_user_listening_review_consolidation}"
  local dead_air_repair_run_id="${DEAD_AIR_REPAIR_RUN_ID:-harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_repeatability_consolidation}"
  local user_consolidation="outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation/${user_consolidation_run_id}/stage_b_duration_coverage_fill_user_listening_review_consolidation.json"
  local dead_air_repair="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/${dead_air_repair_run_id}/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json"
  if [[ ! -f "$user_consolidation" ]]; then
    print_header "Stage B duration/coverage fill user listening review consolidation"
    RUN_ID="$user_consolidation_run_id" run_stage_b_user_listening_review_consolidation
  fi
  if [[ ! -f "$dead_air_repair" ]]; then
    print_header "Stage B duration/coverage fill dead-air gain repeatability repair"
    RUN_ID="$dead_air_repair_run_id" run_stage_b_duration_coverage_dead_air_gain_repeatability_repair
  fi
  print_header "Stage B duration/coverage fill repeatability consolidation"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_repeatability_consolidation.py \
    --run_id "$run_id" \
    --user_listening_consolidation "$user_consolidation" \
    --dead_air_gain_repair "$dead_air_repair" \
    --expected_boundary current_keep_and_distinct_source_dead_air_gain_midi_support \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_repeatability_audio_review_package() {
  local repeatability_consolidation_run_id="${REPEATABILITY_CONSOLIDATION_RUN_ID:-harness_stage_b_duration_coverage_fill_repeatability_consolidation}"
  local dead_air_repair_run_id="${DEAD_AIR_REPAIR_RUN_ID:-harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_repeatability_audio_review_package}"
  local repeatability_consolidation="outputs/stage_b_duration_coverage_fill_repeatability_consolidation/${repeatability_consolidation_run_id}/stage_b_duration_coverage_fill_repeatability_consolidation.json"
  local dead_air_repair="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/${dead_air_repair_run_id}/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$repeatability_consolidation" ]]; then
    print_header "Stage B duration/coverage fill repeatability consolidation"
    RUN_ID="$repeatability_consolidation_run_id" run_stage_b_duration_coverage_repeatability_consolidation
  fi
  if [[ ! -f "$dead_air_repair" ]]; then
    print_header "Stage B duration/coverage fill dead-air gain repeatability repair"
    RUN_ID="$dead_air_repair_run_id" run_stage_b_duration_coverage_dead_air_gain_repeatability_repair
  fi
  print_header "Stage B duration/coverage fill repeatability audio review package"
  "$PYTHON_BIN" scripts/render_stage_b_duration_coverage_fill_repeatability_audio_review_package.py \
    --run_id "$run_id" \
    --repeatability_consolidation "$repeatability_consolidation" \
    --dead_air_gain_repair "$dead_air_repair" \
    --soundfont "$soundfont" \
    --expected_file_count 2 \
    --require_no_quality_claim
}

run_stage_b_duration_coverage_repeatability_user_listening_review() {
  local audio_review_run_id="${AUDIO_REVIEW_RUN_ID:-harness_stage_b_duration_coverage_fill_repeatability_audio_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_repeatability_user_listening_review_fill}"
  local audio_review_package="outputs/stage_b_duration_coverage_fill_repeatability_audio_review_package/${audio_review_run_id}/stage_b_duration_coverage_fill_repeatability_audio_review_package.json"
  if [[ ! -f "$audio_review_package" ]]; then
    print_header "Stage B duration/coverage fill repeatability audio review package"
    RUN_ID="$audio_review_run_id" run_stage_b_duration_coverage_repeatability_audio_review_package
  fi
  print_header "Stage B duration/coverage fill repeatability user listening review"
  "$PYTHON_BIN" scripts/fill_stage_b_duration_coverage_repeatability_user_listening_review.py \
    --run_id "$run_id" \
    --audio_review_package "$audio_review_package" \
    --reviewer "user" \
    --overall_decision reject_all \
    --candidate_decision needs_followup \
    --timing outside_or_unclear \
    --phrase outside_or_unclear \
    --vocabulary outside_or_unclear \
    --assessment "both candidates sound difficult and outside-soloing-like" \
    --notes "user listened to both repeatability WAV files; both felt difficult and outside-soloing-like" \
    --expected_boundary repeatability_audio_review_needs_followup \
    --expected_overall_decision reject_all \
    --require_no_keep_claim \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_decision() {
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_duration_coverage_fill_repeatability_user_listening_review_fill}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_decision}"
  local user_review="outputs/stage_b_duration_coverage_fill_repeatability_user_listening_review_fill/${user_review_run_id}/stage_b_duration_coverage_fill_repeatability_user_listening_review_fill.json"
  if [[ ! -f "$user_review" ]]; then
    print_header "Stage B duration/coverage fill repeatability user listening review"
    RUN_ID="$user_review_run_id" run_stage_b_duration_coverage_repeatability_user_listening_review
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair decision"
  "$PYTHON_BIN" scripts/decide_stage_b_duration_coverage_outside_soloing_repair.py \
    --run_id "$run_id" \
    --user_listening_review "$user_review" \
    --expected_next_boundary outside_soloing_pitch_role_phrase_clarity_repair \
    --require_auto_progress_allowed \
    --require_no_critical_user_input \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_sweep() {
  local outside_decision_run_id="${OUTSIDE_DECISION_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_decision}"
  local dead_air_repair_run_id="${DEAD_AIR_REPAIR_RUN_ID:-harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep}"
  local outside_decision="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_decision/${outside_decision_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_decision.json"
  local dead_air_repair="outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/${dead_air_repair_run_id}/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json"
  if [[ ! -f "$outside_decision" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair decision"
    RUN_ID="$outside_decision_run_id" run_stage_b_duration_coverage_outside_soloing_repair_decision
  fi
  if [[ ! -f "$dead_air_repair" ]]; then
    print_header "Stage B duration/coverage fill dead-air gain repeatability repair"
    RUN_ID="$dead_air_repair_run_id" run_stage_b_duration_coverage_dead_air_gain_repeatability_repair
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair sweep"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_sweep.py \
    --run_id "$run_id" \
    --outside_soloing_decision "$outside_decision" \
    --dead_air_gain_repair "$dead_air_repair" \
    --expected_boundary outside_soloing_pitch_role_repair_candidates \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_audio_review_package() {
  local repair_sweep_run_id="${REPAIR_SWEEP_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package}"
  local repair_sweep="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/${repair_sweep_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$repair_sweep" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair sweep"
    RUN_ID="$repair_sweep_run_id" run_stage_b_duration_coverage_outside_soloing_repair_sweep
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair audio review package"
  "$PYTHON_BIN" scripts/render_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.py \
    --run_id "$run_id" \
    --outside_soloing_repair_sweep "$repair_sweep" \
    --soundfont "$soundfont" \
    --expected_file_count 2 \
    --require_no_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_user_listening_review() {
  local audio_review_run_id="${AUDIO_REVIEW_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill}"
  local audio_review_package="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/${audio_review_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.json"
  if [[ ! -f "$audio_review_package" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair audio review package"
    RUN_ID="$audio_review_run_id" run_stage_b_duration_coverage_outside_soloing_repair_audio_review_package
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair user listening review"
  "$PYTHON_BIN" scripts/fill_stage_b_duration_coverage_outside_soloing_repair_user_listening_review.py \
    --run_id "$run_id" \
    --audio_review_package "$audio_review_package" \
    --expected_boundary outside_soloing_repair_audio_review_pending \
    --require_pending_without_input \
    --require_no_preference_without_input \
    --require_objective_auto_progress_allowed
}

run_stage_b_duration_coverage_outside_soloing_repair_objective_evidence() {
  local repair_sweep_run_id="${REPAIR_SWEEP_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep}"
  local review_fill_run_id="${REVIEW_FILL_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation}"
  local repair_sweep="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/${repair_sweep_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json"
  local review_fill="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/${review_fill_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.json"
  if [[ ! -f "$repair_sweep" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair sweep"
    RUN_ID="$repair_sweep_run_id" run_stage_b_duration_coverage_outside_soloing_repair_sweep
  fi
  if [[ ! -f "$review_fill" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair user listening review"
    RUN_ID="$review_fill_run_id" run_stage_b_duration_coverage_outside_soloing_repair_user_listening_review
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair objective evidence"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.py \
    --run_id "$run_id" \
    --repair_sweep "$repair_sweep" \
    --user_review_fill "$review_fill" \
    --expected_boundary outside_soloing_repair_objective_evidence_support \
    --require_no_preference_claim \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_next_decision() {
  local objective_run_id="${OBJECTIVE_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_next_decision}"
  local objective_evidence="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/${objective_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.json"
  if [[ ! -f "$objective_evidence" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair objective evidence"
    RUN_ID="$objective_run_id" run_stage_b_duration_coverage_outside_soloing_repair_objective_evidence
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_duration_coverage_outside_soloing_repair_next_step.py \
    --run_id "$run_id" \
    --objective_evidence "$objective_evidence" \
    --expected_next_boundary outside_soloing_repair_broader_repeatability_sweep \
    --require_auto_progress_allowed \
    --require_no_critical_user_input \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_broader_repeatability() {
  local next_decision_run_id="${NEXT_DECISION_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_next_decision}"
  local repair_sweep_run_id="${REPAIR_SWEEP_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep}"
  local next_decision="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision/${next_decision_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision.json"
  local repair_sweep="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/${repair_sweep_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json"
  if [[ ! -f "$next_decision" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair next decision"
    RUN_ID="$next_decision_run_id" run_stage_b_duration_coverage_outside_soloing_repair_next_decision
  fi
  if [[ ! -f "$repair_sweep" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair sweep"
    RUN_ID="$repair_sweep_run_id" run_stage_b_duration_coverage_outside_soloing_repair_sweep
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair broader repeatability"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.py \
    --run_id "$run_id" \
    --next_decision "$next_decision" \
    --repair_sweep "$repair_sweep" \
    --expected_boundary outside_soloing_repair_policy_repeatability_support \
    --require_no_preference_claim \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_repeatability_consolidation() {
  local objective_run_id="${OBJECTIVE_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation}"
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep}"
  local review_fill_run_id="${REVIEW_FILL_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation}"
  local objective_evidence="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/${objective_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.json"
  local broader_repeatability="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/${repeatability_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.json"
  local review_fill="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/${review_fill_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.json"
  if [[ ! -f "$objective_evidence" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair objective evidence"
    RUN_ID="$objective_run_id" run_stage_b_duration_coverage_outside_soloing_repair_objective_evidence
  fi
  if [[ ! -f "$broader_repeatability" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair broader repeatability"
    RUN_ID="$repeatability_run_id" run_stage_b_duration_coverage_outside_soloing_repair_broader_repeatability
  fi
  if [[ ! -f "$review_fill" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair user listening review"
    RUN_ID="$review_fill_run_id" run_stage_b_duration_coverage_outside_soloing_repair_user_listening_review
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair repeatability consolidation"
  "$PYTHON_BIN" scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation.py \
    --run_id "$run_id" \
    --objective_evidence "$objective_evidence" \
    --broader_repeatability "$broader_repeatability" \
    --user_review_fill "$review_fill" \
    --expected_boundary outside_soloing_repair_objective_repeatability_support \
    --require_pending_review_guard \
    --require_no_preference_claim \
    --require_no_broad_quality_claim
}

run_stage_b_duration_coverage_outside_soloing_repair_final_decision() {
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation}"
  local run_id="${RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_final_decision}"
  local repeatability_consolidation="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation/${repeatability_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation.json"
  if [[ ! -f "$repeatability_consolidation" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair repeatability consolidation"
    RUN_ID="$repeatability_run_id" run_stage_b_duration_coverage_outside_soloing_repair_repeatability_consolidation
  fi
  print_header "Stage B duration/coverage fill outside-soloing repair final decision"
  "$PYTHON_BIN" scripts/decide_stage_b_duration_coverage_outside_soloing_repair_final_decision.py \
    --run_id "$run_id" \
    --repeatability_consolidation "$repeatability_consolidation" \
    --expected_final_boundary outside_soloing_repair_objective_path_complete \
    --expected_next_boundary stage_b_model_core_evidence_readme_refresh \
    --require_auto_progress_allowed \
    --require_no_critical_user_input \
    --require_no_preference_claim \
    --require_no_broad_quality_claim
}

run_stage_b_generic_base_readiness_audit() {
  local final_decision_run_id="${FINAL_DECISION_RUN_ID:-harness_stage_b_duration_coverage_fill_outside_soloing_repair_final_decision}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_readiness_audit}"
  local final_decision="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_final_decision/${final_decision_run_id}/stage_b_duration_coverage_fill_outside_soloing_repair_final_decision.json"
  if [[ ! -f "$final_decision" ]]; then
    print_header "Stage B duration/coverage fill outside-soloing repair final decision"
    RUN_ID="$final_decision_run_id" run_stage_b_duration_coverage_outside_soloing_repair_final_decision
  fi
  print_header "Stage B generic base readiness audit"
  "$PYTHON_BIN" scripts/assess_stage_b_generic_base_readiness.py \
    --run_id "$run_id" \
    --final_decision "$final_decision" \
    --expected_boundary stage_b_generic_base_readiness_audit \
    --expected_next_boundary stage_b_generic_base_manifest_contract \
    --require_phase4_prep_ready \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_base_manifest_contract() {
  local readiness_run_id="${READINESS_RUN_ID:-harness_stage_b_generic_base_readiness_audit}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_manifest_contract}"
  local readiness_audit="outputs/stage_b_generic_base_readiness_audit/${readiness_run_id}/stage_b_generic_base_readiness_audit.json"
  if [[ ! -f "$readiness_audit" ]]; then
    print_header "Stage B generic base readiness audit"
    RUN_ID="$readiness_run_id" run_stage_b_generic_base_readiness_audit
  fi
  print_header "Stage B generic base manifest contract"
  "$PYTHON_BIN" scripts/check_stage_b_generic_base_manifest_contract.py \
    --run_id "$run_id" \
    --readiness_audit "$readiness_audit" \
    --expected_boundary stage_b_generic_base_manifest_contract \
    --expected_next_boundary stage_b_generic_stage_b_window_prepare_smoke \
    --require_contract_ready \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_manifest_window_smoke() {
  local manifest_contract_run_id="${MANIFEST_CONTRACT_RUN_ID:-harness_stage_b_generic_base_manifest_contract}"
  local run_id="${RUN_ID:-harness_stage_b_generic_manifest_window_smoke}"
  local manifests_dir="outputs/stage_b_generic_base_manifest_contract/${manifest_contract_run_id}/manifests"
  if [[ ! -f "${manifests_dir}/generic_jazz_train.txt" || ! -f "${manifests_dir}/generic_jazz_val.txt" ]]; then
    print_header "Stage B generic base manifest contract"
    RUN_ID="$manifest_contract_run_id" run_stage_b_generic_base_manifest_contract
  fi
  print_header "Stage B generic manifest window smoke"
  "$PYTHON_BIN" scripts/run_stage_b_generic_manifest_window_smoke.py \
    --run_id "$run_id" \
    --manifests_dir "$manifests_dir" \
    --expected_boundary stage_b_generic_stage_b_window_prepare_smoke \
    --expected_next_boundary stage_b_generic_base_tiny_training_smoke \
    --require_smoke_ready \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_base_tiny_training_smoke() {
  local window_smoke_run_id="${WINDOW_SMOKE_RUN_ID:-harness_stage_b_generic_manifest_window_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local source_tokenized_dir="outputs/stage_b_generic_manifest_window_smoke/${window_smoke_run_id}/roles/lead/tokenized"
  if [[ ! -d "${source_tokenized_dir}/train" || ! -d "${source_tokenized_dir}/val" ]]; then
    print_header "Stage B generic manifest window smoke"
    RUN_ID="$window_smoke_run_id" run_stage_b_generic_manifest_window_smoke
  fi
  print_header "Stage B generic base tiny training smoke"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_tiny_training_smoke.py \
    --run_id "$run_id" \
    --source_tokenized_dir "$source_tokenized_dir" \
    --expected_boundary stage_b_generic_base_tiny_training_smoke \
    --expected_next_boundary stage_b_generic_tiny_checkpoint_generation_probe \
    --require_training_smoke_passed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_generation_probe() {
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_generation_probe}"
  local checkpoint_dir="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/checkpoints"
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  print_header "Stage B generic tiny checkpoint generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generic_tiny_checkpoint_generation_probe.py \
    --run_id "$run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --expected_boundary stage_b_generic_tiny_checkpoint_generation_probe \
    --require_probe_completed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_grammar_repair() {
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_grammar_repair}"
  local checkpoint_dir="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/checkpoints"
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  print_header "Stage B generic tiny checkpoint grammar repair"
  "$PYTHON_BIN" scripts/run_stage_b_generic_tiny_checkpoint_grammar_repair.py \
    --run_id "$run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --expected_boundary stage_b_generic_tiny_checkpoint_grammar_repair \
    --require_repair_passed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_repair_repeatability() {
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_repeatability}"
  local checkpoint_dir="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/checkpoints"
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  print_header "Stage B generic tiny checkpoint repair repeatability"
  "$PYTHON_BIN" scripts/run_stage_b_generic_tiny_checkpoint_repair_repeatability.py \
    --run_id "$run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_repeatability_probe \
    --require_repeatability_passed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_repair_review_package() {
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_repeatability}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_review_package}"
  local repeatability_report="outputs/stage_b_generic_tiny_checkpoint_repair_repeatability/${repeatability_run_id}/stage_b_generic_tiny_checkpoint_repair_repeatability.json"
  if [[ ! -f "$repeatability_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair repeatability"
    RUN_ID="$repeatability_run_id" run_stage_b_generic_tiny_checkpoint_repair_repeatability
  fi
  print_header "Stage B generic tiny checkpoint repair review package"
  "$PYTHON_BIN" scripts/build_stage_b_generic_tiny_checkpoint_repair_review_package.py \
    --run_id "$run_id" \
    --repeatability_report "$repeatability_report" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_review_package \
    --require_review_package_ready \
    --require_no_musical_quality_claim \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_repair_listening_notes() {
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local review_package_report="outputs/stage_b_generic_tiny_checkpoint_repair_review_package/${review_package_run_id}/stage_b_generic_tiny_checkpoint_repair_review_package.json"
  if [[ ! -f "$review_package_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair review package"
    RUN_ID="$review_package_run_id" run_stage_b_generic_tiny_checkpoint_repair_review_package
  fi
  print_header "Stage B generic tiny checkpoint repair listening notes"
  "$PYTHON_BIN" scripts/build_stage_b_generic_tiny_checkpoint_repair_listening_notes.py \
    --run_id "$run_id" \
    --review_package_report "$review_package_report" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_listening_notes \
    --require_listening_notes_ready \
    --require_pending_human_review \
    --require_no_musical_quality_claim \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_repair_listening_fill() {
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_report="outputs/stage_b_generic_tiny_checkpoint_repair_listening_notes/${listening_notes_run_id}/stage_b_generic_tiny_checkpoint_repair_listening_notes.json"
  if [[ ! -f "$listening_notes_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair listening notes"
    RUN_ID="$listening_notes_run_id" run_stage_b_generic_tiny_checkpoint_repair_listening_notes
  fi
  print_header "Stage B generic tiny checkpoint repair listening fill"
  "$PYTHON_BIN" scripts/fill_stage_b_generic_tiny_checkpoint_repair_listening_notes.py \
    --run_id "$run_id" \
    --listening_notes_report "$listening_notes_report" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_listening_fill \
    --require_pending_without_input \
    --require_no_quality_without_input \
    --require_objective_auto_progress_allowed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_tiny_checkpoint_repair_audio_render_package() {
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_report="outputs/stage_b_generic_tiny_checkpoint_repair_listening_fill/${listening_fill_run_id}/stage_b_generic_tiny_checkpoint_repair_listening_fill.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$listening_fill_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair listening fill"
    RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" run_stage_b_generic_tiny_checkpoint_repair_listening_fill
  fi
  print_header "Stage B generic tiny checkpoint repair audio render package"
  "$PYTHON_BIN" scripts/build_stage_b_generic_tiny_checkpoint_repair_audio_render_package.py \
    --run_id "$run_id" \
    --listening_fill_report "$listening_fill_report" \
    --soundfont "$soundfont" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_audio_render_package \
    --min_planned_outputs 5 \
    --require_required_midi_exists \
    --require_no_audio_claim
}

run_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt() {
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local local_audio_render_package="outputs/stage_b_generic_tiny_checkpoint_repair_audio_render_package/${audio_package_run_id}/stage_b_generic_tiny_checkpoint_repair_audio_render_package.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$local_audio_render_package" ]]; then
    print_header "Stage B generic tiny checkpoint repair audio render package"
    RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" run_stage_b_generic_tiny_checkpoint_repair_audio_render_package
  fi
  print_header "Stage B generic tiny checkpoint repair local audio render attempt"
  "$PYTHON_BIN" scripts/render_stage_b_generic_tiny_checkpoint_repair_audio.py \
    --run_id "$run_id" \
    --local_audio_render_package "$local_audio_render_package" \
    --soundfont "$soundfont" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt \
    --expected_file_count 5 \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_user_listening_review() {
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_report="outputs/stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt/${audio_render_run_id}/stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt.json"
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair local audio render attempt"
    RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" run_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt
  fi
  print_header "Stage B generic tiny checkpoint repair user listening review"
  "$PYTHON_BIN" scripts/fill_stage_b_generic_tiny_checkpoint_repair_user_listening_review.py \
    --run_id "$run_id" \
    --audio_render_report "$audio_render_report" \
    --reviewer "user" \
    --overall_decision reject_all \
    --candidate_decision reject \
    --primary_failure plunk_and_stop \
    --timing too_short_or_stiff \
    --phrase fragmented \
    --vocabulary not_musical \
    --assessment "all candidates only plunk briefly and end" \
    --notes "user listened to all five WAV files; all candidates felt like short plunk-and-stop fragments" \
    --expected_boundary generic_tiny_checkpoint_repair_audio_review_reject_all \
    --expected_overall_decision reject_all \
    --expected_primary_failure plunk_and_stop \
    --expected_file_count 5 \
    --require_no_keep_claim \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision() {
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review="outputs/stage_b_generic_tiny_checkpoint_repair_user_listening_review/${user_review_run_id}/stage_b_generic_tiny_checkpoint_repair_user_listening_review.json"
  if [[ ! -f "$user_review" ]]; then
    print_header "Stage B generic tiny checkpoint repair user listening review"
    RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" run_stage_b_generic_tiny_checkpoint_repair_user_listening_review
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation decision"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation.py \
    --run_id "$run_id" \
    --user_listening_review "$user_review" \
    --expected_next_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep \
    --require_auto_progress_allowed \
    --require_no_critical_user_input \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local checkpoint_dir="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/checkpoints"
  local decision_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision/${decision_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision.json"
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation decision"
    RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation sweep"
  "$PYTHON_BIN" scripts/run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep.py \
    --run_id "$run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --decision_report "$decision_report" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep \
    --min_target_qualified 1 \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package() {
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep/${sweep_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$sweep_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation sweep"
    RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation audio render package"
  "$PYTHON_BIN" scripts/build_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package.py \
    --run_id "$run_id" \
    --sweep_report "$sweep_report" \
    --soundfont "$soundfont" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package \
    --min_planned_outputs 1 \
    --require_target_qualified \
    --require_required_midi_exists \
    --require_no_audio_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt() {
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package/${phrase_audio_package_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$phrase_audio_package" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation audio render package"
    RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation local audio render attempt"
  "$PYTHON_BIN" scripts/render_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio.py \
    --run_id "$run_id" \
    --local_audio_render_package "$phrase_audio_package" \
    --soundfont "$soundfont" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt \
    --expected_file_count 1 \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review() {
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local audio_render_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt/${local_audio_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt.json"
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation local audio render attempt"
    RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation MIDI note failure review"
  "$PYTHON_BIN" scripts/fill_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review.py \
    --run_id "$run_id" \
    --audio_render_report "$audio_render_report" \
    --reviewer "user" \
    --assessment "candidate is not musical; MIDI notes show random large register jumps" \
    --notes "user rejected the rendered WAV as non-musical and requested MIDI note-level review" \
    --expected_boundary generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all \
    --expected_primary_failure midi_note_random_large_leaps \
    --expected_file_count 1 \
    --min_max_interval 24 \
    --require_no_keep_claim \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision() {
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review/${failure_review_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review.json"
  if [[ ! -f "$failure_review_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation MIDI note failure review"
    RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard decision"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard.py \
    --run_id "$run_id" \
    --failure_review_report "$failure_review_report" \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision \
    --expected_next_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep() {
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local checkpoint_dir="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/checkpoints"
  local range_decision_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision/${range_decision_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision.json"
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  if [[ ! -f "$range_decision_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard decision"
    RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sweep"
  "$PYTHON_BIN" scripts/run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep.py \
    --run_id "$run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --decision_report "$range_decision_report" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SWEEP_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep \
    --min_target_qualified 1 \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package() {
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep/${range_sweep_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$range_sweep_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sweep"
    RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard audio render package"
  "$PYTHON_BIN" scripts/build_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package.py \
    --run_id "$run_id" \
    --sweep_report "$range_sweep_report" \
    --soundfont "$soundfont" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_AUDIO_RENDER_PACKAGE_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package \
    --min_planned_outputs 1 \
    --require_target_qualified \
    --require_required_midi_exists \
    --require_no_audio_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt() {
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package/${range_audio_package_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$range_audio_package" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard audio render package"
    RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard local audio render attempt"
  "$PYTHON_BIN" scripts/render_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio.py \
    --run_id "$run_id" \
    --local_audio_render_package "$range_audio_package" \
    --soundfont "$soundfont" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review() {
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_audio_render_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt/${range_local_audio_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt.json"
  if [[ ! -f "$range_audio_render_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard local audio render attempt"
    RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard user listening review"
  "$PYTHON_BIN" scripts/fill_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review.py \
    --run_id "$run_id" \
    --audio_render_report "$range_audio_render_report" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_USER_LISTENING_REVIEW_2026-05-30.md \
    --reviewer "user" \
    --overall_decision reject_all \
    --candidate_decision reject \
    --primary_failure subjective_not_musical \
    --timing outside_or_unclear \
    --phrase not_musical \
    --vocabulary not_musical \
    --assessment "all range/interval guard candidates rejected by single-user listening review" \
    --notes "objective range/interval guard passed, but listening review did not accept any rendered candidate" \
    --expected_boundary generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_review_reject_all \
    --expected_overall_decision reject_all \
    --expected_primary_failure subjective_not_musical \
    --expected_file_count 3 \
    --require_no_keep_claim \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis() {
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review/${range_user_review_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review.json"
  if [[ ! -f "$range_user_review_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard user listening review"
    RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard rejection analysis"
  "$PYTHON_BIN" scripts/analyze_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection.py \
    --run_id "$run_id" \
    --user_listening_review_report "$range_user_review_report" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_REJECTION_ANALYSIS_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis \
    --expected_candidate_count 3 \
    --require_reject_all_source \
    --require_no_quality_claim \
    --min_common_evidence_flags 1
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision() {
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis/${rejection_analysis_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis.json"
  if [[ ! -f "$rejection_analysis" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard rejection analysis"
    RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair.py \
    --run_id "$run_id" \
    --rejection_analysis "$rejection_analysis" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_REPAIR_DECISION_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision \
    --expected_next_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep \
    --require_primary_target sparse_phrase_continuity_after_range_interval_guard \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep() {
  local sparse_decision_run_id="${SPARSE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep}"
  local checkpoint_dir="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/checkpoints"
  local sparse_decision="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision/${sparse_decision_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision.json"
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  if [[ ! -f "$sparse_decision" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision"
    RUN_ID="$sparse_decision_run_id" REJECTION_ANALYSIS_RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep"
  "$PYTHON_BIN" scripts/run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep.py \
    --run_id "$run_id" \
    --checkpoint_dir "$checkpoint_dir" \
    --decision_report "$sparse_decision" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_REPAIR_SWEEP_2026-05-30.md \
    --coverage_aware_positions \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep \
    --require_gap_reduction \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package() {
  local sparse_sweep_run_id="${SPARSE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep}"
  local sparse_decision_run_id="${SPARSE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package}"
  local sparse_sweep="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep/${sparse_sweep_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$sparse_sweep" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep"
    RUN_ID="$sparse_sweep_run_id" SPARSE_DECISION_RUN_ID="$sparse_decision_run_id" REJECTION_ANALYSIS_RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package"
  "$PYTHON_BIN" scripts/build_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package.py \
    --run_id "$run_id" \
    --sweep_report "$sparse_sweep" \
    --soundfont "$soundfont" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_AUDIO_RENDER_PACKAGE_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package \
    --expected_status ready_for_local_render \
    --min_planned_outputs 3 \
    --require_target_qualified \
    --require_required_midi_exists \
    --require_no_audio_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt() {
  local sparse_audio_package_run_id="${SPARSE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package}"
  local sparse_sweep_run_id="${SPARSE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep}"
  local sparse_decision_run_id="${SPARSE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt}"
  local sparse_audio_package="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package/${sparse_audio_package_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package.json"
  local soundfont="${SOUNDFONT_PATH:-$HOME/.local/share/soundfonts/generaluser-gs/v1.471.sf2}"
  if [[ ! -f "$sparse_audio_package" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package"
    RUN_ID="$sparse_audio_package_run_id" SPARSE_SWEEP_RUN_ID="$sparse_sweep_run_id" SPARSE_DECISION_RUN_ID="$sparse_decision_run_id" REJECTION_ANALYSIS_RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt"
  "$PYTHON_BIN" scripts/render_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio.py \
    --run_id "$run_id" \
    --local_audio_render_package "$sparse_audio_package" \
    --soundfont "$soundfont" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_LOCAL_AUDIO_RENDER_ATTEMPT_2026-05-30.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review() {
  local sparse_local_audio_run_id="${SPARSE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt}"
  local sparse_audio_package_run_id="${SPARSE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package}"
  local sparse_sweep_run_id="${SPARSE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep}"
  local sparse_decision_run_id="${SPARSE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review}"
  local sparse_audio_render_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt/${sparse_local_audio_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt.json"
  if [[ ! -f "$sparse_audio_render_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase local audio render attempt"
    RUN_ID="$sparse_local_audio_run_id" SPARSE_AUDIO_PACKAGE_RUN_ID="$sparse_audio_package_run_id" SPARSE_SWEEP_RUN_ID="$sparse_sweep_run_id" SPARSE_DECISION_RUN_ID="$sparse_decision_run_id" REJECTION_ANALYSIS_RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase user listening review"
  "$PYTHON_BIN" scripts/fill_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review.py \
    --run_id "$run_id" \
    --audio_render_report "$sparse_audio_render_report" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_USER_LISTENING_REVIEW_2026-06-01.md \
    --reviewer "user" \
    --overall_decision reject_all \
    --candidate_decision reject \
    --primary_failure subjective_not_musical \
    --timing outside_or_unclear \
    --phrase not_musical \
    --vocabulary not_musical \
    --assessment "all sparse phrase repair candidates rejected by single-user listening review" \
    --notes "sparse phrase objective gate passed, but listening review did not accept any rendered candidate" \
    --expected_boundary generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_review_reject_all \
    --expected_overall_decision reject_all \
    --expected_primary_failure subjective_not_musical \
    --expected_file_count 3 \
    --require_no_keep_claim \
    --require_no_quality_claim
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis() {
  local sparse_user_review_run_id="${SPARSE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review}"
  local sparse_local_audio_run_id="${SPARSE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt}"
  local sparse_audio_package_run_id="${SPARSE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package}"
  local sparse_sweep_run_id="${SPARSE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep}"
  local sparse_decision_run_id="${SPARSE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis}"
  local sparse_user_review_report="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review/${sparse_user_review_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review.json"
  if [[ ! -f "$sparse_user_review_report" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase user listening review"
    RUN_ID="$sparse_user_review_run_id" SPARSE_LOCAL_AUDIO_RUN_ID="$sparse_local_audio_run_id" SPARSE_AUDIO_PACKAGE_RUN_ID="$sparse_audio_package_run_id" SPARSE_SWEEP_RUN_ID="$sparse_sweep_run_id" SPARSE_DECISION_RUN_ID="$sparse_decision_run_id" REJECTION_ANALYSIS_RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase rejection analysis"
  "$PYTHON_BIN" scripts/analyze_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection.py \
    --run_id "$run_id" \
    --user_listening_review_report "$sparse_user_review_report" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_REJECTION_ANALYSIS_2026-06-01.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis \
    --expected_candidate_count 3 \
    --require_reject_all_source \
    --require_no_quality_claim \
    --require_proxy_gap
}

run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review() {
  local sparse_rejection_run_id="${SPARSE_REJECTION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis}"
  local sparse_user_review_run_id="${SPARSE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review}"
  local sparse_local_audio_run_id="${SPARSE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt}"
  local sparse_audio_package_run_id="${SPARSE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package}"
  local sparse_sweep_run_id="${SPARSE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep}"
  local sparse_decision_run_id="${SPARSE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision}"
  local rejection_analysis_run_id="${REJECTION_ANALYSIS_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis}"
  local range_user_review_run_id="${RANGE_USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review}"
  local range_local_audio_run_id="${RANGE_LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt}"
  local range_audio_package_run_id="${RANGE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package}"
  local range_sweep_run_id="${RANGE_SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep}"
  local range_decision_run_id="${RANGE_DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision}"
  local failure_review_run_id="${FAILURE_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review}"
  local local_audio_run_id="${LOCAL_AUDIO_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt}"
  local phrase_audio_package_run_id="${PHRASE_AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package}"
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision}"
  local user_review_run_id="${USER_REVIEW_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_audio_render_package}"
  local listening_fill_run_id="${LISTENING_FILL_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_fill}"
  local listening_notes_run_id="${LISTENING_NOTES_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_listening_notes}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision}"
  local sparse_rejection_analysis="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis/${sparse_rejection_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis.json"
  if [[ ! -f "$sparse_rejection_analysis" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase rejection analysis"
    RUN_ID="$sparse_rejection_run_id" SPARSE_USER_REVIEW_RUN_ID="$sparse_user_review_run_id" SPARSE_LOCAL_AUDIO_RUN_ID="$sparse_local_audio_run_id" SPARSE_AUDIO_PACKAGE_RUN_ID="$sparse_audio_package_run_id" SPARSE_SWEEP_RUN_ID="$sparse_sweep_run_id" SPARSE_DECISION_RUN_ID="$sparse_decision_run_id" REJECTION_ANALYSIS_RUN_ID="$rejection_analysis_run_id" RANGE_USER_REVIEW_RUN_ID="$range_user_review_run_id" RANGE_LOCAL_AUDIO_RUN_ID="$range_local_audio_run_id" RANGE_AUDIO_PACKAGE_RUN_ID="$range_audio_package_run_id" RANGE_SWEEP_RUN_ID="$range_sweep_run_id" RANGE_DECISION_RUN_ID="$range_decision_run_id" FAILURE_REVIEW_RUN_ID="$failure_review_run_id" LOCAL_AUDIO_RUN_ID="$local_audio_run_id" PHRASE_AUDIO_PACKAGE_RUN_ID="$phrase_audio_package_run_id" SWEEP_RUN_ID="$sweep_run_id" DECISION_RUN_ID="$decision_run_id" USER_REVIEW_RUN_ID="$user_review_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" LISTENING_FILL_RUN_ID="$listening_fill_run_id" LISTENING_NOTES_RUN_ID="$listening_notes_run_id" TRAINING_SMOKE_RUN_ID="$training_smoke_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis
  fi
  print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase model core review"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review.py \
    --run_id "$run_id" \
    --sparse_rejection_analysis "$sparse_rejection_analysis" \
    --doc_path docs/STAGE_B_GENERIC_TINY_CHECKPOINT_REPAIR_PHRASE_CONTINUATION_RANGE_INTERVAL_GUARD_SPARSE_PHRASE_MODEL_CORE_REVIEW_DECISION_2026-06-01.md \
    --expected_boundary stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision \
    --expected_next_boundary stage_b_generic_model_core_training_data_plan \
    --require_stop_repair_loop \
    --require_diagnostic_only \
    --require_no_quality_claim
}

run_stage_b_generic_model_core_training_data_plan() {
  local model_core_run_id="${MODEL_CORE_RUN_ID:-harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision}"
  local manifest_contract_run_id="${MANIFEST_CONTRACT_RUN_ID:-harness_stage_b_generic_base_manifest_contract}"
  local window_smoke_run_id="${WINDOW_SMOKE_RUN_ID:-harness_stage_b_generic_manifest_window_smoke}"
  local training_smoke_run_id="${TRAINING_SMOKE_RUN_ID:-harness_stage_b_generic_base_tiny_training_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_model_core_training_data_plan}"
  local model_core_decision="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision/${model_core_run_id}/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review_decision.json"
  local manifest_contract="outputs/stage_b_generic_base_manifest_contract/${manifest_contract_run_id}/stage_b_generic_base_manifest_contract.json"
  local window_smoke="outputs/stage_b_generic_manifest_window_smoke/${window_smoke_run_id}/stage_b_generic_manifest_window_smoke.json"
  local tiny_training_smoke="outputs/stage_b_generic_base_tiny_training_smoke/${training_smoke_run_id}/stage_b_generic_base_tiny_training_smoke.json"
  if [[ ! -f "$model_core_decision" ]]; then
    print_header "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase model core review"
    RUN_ID="$model_core_run_id" run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review
  fi
  if [[ ! -f "$manifest_contract" ]]; then
    print_header "Stage B generic base manifest contract"
    RUN_ID="$manifest_contract_run_id" run_stage_b_generic_base_manifest_contract
  fi
  if [[ ! -f "$window_smoke" ]]; then
    print_header "Stage B generic manifest window smoke"
    RUN_ID="$window_smoke_run_id" MANIFEST_CONTRACT_RUN_ID="$manifest_contract_run_id" run_stage_b_generic_manifest_window_smoke
  fi
  if [[ ! -f "$tiny_training_smoke" ]]; then
    print_header "Stage B generic base tiny training smoke"
    RUN_ID="$training_smoke_run_id" WINDOW_SMOKE_RUN_ID="$window_smoke_run_id" run_stage_b_generic_base_tiny_training_smoke
  fi
  print_header "Stage B generic model-core training data plan"
  "$PYTHON_BIN" scripts/plan_stage_b_generic_model_core_training_data.py \
    --run_id "$run_id" \
    --model_core_decision "$model_core_decision" \
    --manifest_contract "$manifest_contract" \
    --window_smoke "$window_smoke" \
    --tiny_training_smoke "$tiny_training_smoke" \
    --doc_path docs/STAGE_B_GENERIC_MODEL_CORE_TRAINING_DATA_PLAN_2026-06-01.md \
    --expected_boundary stage_b_generic_model_core_training_data_plan \
    --expected_next_boundary stage_b_generic_full_manifest_window_preparation \
    --min_generic_train_files 2000 \
    --min_generic_val_files 200 \
    --require_stop_repair_loop \
    --require_no_quality_claim
}

run_stage_b_generic_full_manifest_window_preparation() {
  local plan_run_id="${PLAN_RUN_ID:-harness_stage_b_generic_model_core_training_data_plan}"
  local manifest_contract_run_id="${MANIFEST_CONTRACT_RUN_ID:-harness_stage_b_generic_base_manifest_contract}"
  local run_id="${RUN_ID:-harness_stage_b_generic_full_manifest_window_preparation}"
  local training_data_plan="outputs/stage_b_generic_model_core_training_data_plan/${plan_run_id}/stage_b_generic_model_core_training_data_plan.json"
  local manifests_dir="outputs/stage_b_generic_base_manifest_contract/${manifest_contract_run_id}/manifests"
  if [[ ! -f "$training_data_plan" ]]; then
    print_header "Stage B generic model-core training data plan"
    RUN_ID="$plan_run_id" MANIFEST_CONTRACT_RUN_ID="$manifest_contract_run_id" run_stage_b_generic_model_core_training_data_plan
  fi
  if [[ ! -f "${manifests_dir}/generic_jazz_train.txt" || ! -f "${manifests_dir}/generic_jazz_val.txt" ]]; then
    print_header "Stage B generic base manifest contract"
    RUN_ID="$manifest_contract_run_id" run_stage_b_generic_base_manifest_contract
  fi
  print_header "Stage B generic full manifest window preparation"
  "$PYTHON_BIN" scripts/run_stage_b_generic_full_manifest_window_preparation.py \
    --run_id "$run_id" \
    --training_data_plan "$training_data_plan" \
    --manifests_dir "$manifests_dir" \
    --doc_path docs/STAGE_B_GENERIC_FULL_MANIFEST_WINDOW_PREPARATION_2026-06-01.md \
    --expected_boundary stage_b_generic_full_manifest_window_preparation \
    --expected_next_boundary stage_b_generic_base_training_scale_smoke \
    --require_ready \
    --require_no_training_claim \
    --require_no_quality_claim \
    --min_tokenized_train_files 1 \
    --min_tokenized_val_files 1
}

run_stage_b_generic_base_training_scale_smoke() {
  local full_window_run_id="${FULL_WINDOW_RUN_ID:-harness_stage_b_generic_full_manifest_window_preparation}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_training_scale_smoke}"
  local full_window_preparation="outputs/stage_b_generic_full_manifest_window_preparation/${full_window_run_id}/stage_b_generic_full_manifest_window_preparation.json"
  if [[ ! -f "$full_window_preparation" ]]; then
    print_header "Stage B generic full manifest window preparation"
    RUN_ID="$full_window_run_id" run_stage_b_generic_full_manifest_window_preparation
  fi
  print_header "Stage B generic base training scale smoke"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_training_scale_smoke.py \
    --run_id "$run_id" \
    --full_window_preparation "$full_window_preparation" \
    --doc_path docs/STAGE_B_GENERIC_BASE_TRAINING_SCALE_SMOKE_2026-06-01.md \
    --expected_boundary stage_b_generic_base_training_scale_smoke \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
    --require_training_scale_smoke_passed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_base_scale_checkpoint_generation_probe() {
  local scale_training_run_id="${SCALE_TRAINING_RUN_ID:-harness_stage_b_generic_base_training_scale_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_generation_probe}"
  local training_scale_smoke="outputs/stage_b_generic_base_training_scale_smoke/${scale_training_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$training_scale_smoke" ]]; then
    print_header "Stage B generic base training scale smoke"
    RUN_ID="$scale_training_run_id" run_stage_b_generic_base_training_scale_smoke
  fi
  print_header "Stage B generic base scale checkpoint generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_generation_probe.py \
    --run_id "$run_id" \
    --training_scale_smoke "$training_scale_smoke" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_GENERATION_PROBE_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
    --require_probe_completed \
    --require_no_broad_quality_claim \
    --require_no_brad_style_claim
}

run_stage_b_generic_base_scale_checkpoint_grammar_representation_decision() {
  local generation_probe_run_id="${GENERATION_PROBE_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_generation_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_grammar_representation_decision}"
  local generation_probe="outputs/stage_b_generic_base_scale_checkpoint_generation_probe/${generation_probe_run_id}/stage_b_generic_base_scale_checkpoint_generation_probe.json"
  if [[ ! -f "$generation_probe" ]]; then
    print_header "Stage B generic base scale checkpoint generation probe"
    RUN_ID="$generation_probe_run_id" run_stage_b_generic_base_scale_checkpoint_generation_probe
  fi
  print_header "Stage B generic base scale checkpoint grammar representation decision"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_base_scale_checkpoint_grammar_representation.py \
    --run_id "$run_id" \
    --generation_probe "$generation_probe" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_GRAMMAR_REPRESENTATION_DECISION_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_grammar_representation_decision \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe \
    --require_density_coverage_target \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_grammar_representation_decision}"
  local generation_probe_run_id="${GENERATION_PROBE_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_generation_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe}"
  local decision_report="outputs/stage_b_generic_base_scale_checkpoint_grammar_representation_decision/${decision_run_id}/stage_b_generic_base_scale_checkpoint_grammar_representation_decision.json"
  local baseline_generation_probe="outputs/stage_b_generic_base_scale_checkpoint_generation_probe/${generation_probe_run_id}/stage_b_generic_base_scale_checkpoint_generation_probe.json"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B generic base scale checkpoint grammar representation decision"
    RUN_ID="$decision_run_id" GENERATION_PROBE_RUN_ID="$generation_probe_run_id" run_stage_b_generic_base_scale_checkpoint_grammar_representation_decision
  fi
  if [[ ! -f "$baseline_generation_probe" ]]; then
    print_header "Stage B generic base scale checkpoint generation probe"
    RUN_ID="$generation_probe_run_id" run_stage_b_generic_base_scale_checkpoint_generation_probe
  fi
  print_header "Stage B generic base scale checkpoint density coverage repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --baseline_generation_probe "$baseline_generation_probe" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DENSITY_COVERAGE_REPAIR_PROBE_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision \
    --require_target_qualified \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision}"
  local repair_report="outputs/stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe/${repair_run_id}/stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B generic base scale checkpoint density coverage repair probe"
    RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe
  fi
  print_header "Stage B generic base scale checkpoint density coverage remaining blocker decision"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker.py \
    --run_id "$run_id" \
    --density_coverage_repair "$repair_report" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DENSITY_COVERAGE_REMAINING_BLOCKER_DECISION_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe \
    --require_duration_target \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe}"
  local remaining_blocker_decision="outputs/stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision/${decision_run_id}/stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision.json"
  local density_coverage_repair="outputs/stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe/${repair_run_id}/stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe.json"
  if [[ ! -f "$density_coverage_repair" ]]; then
    print_header "Stage B generic base scale checkpoint density coverage repair probe"
    RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe
  fi
  if [[ ! -f "$remaining_blocker_decision" ]]; then
    print_header "Stage B generic base scale checkpoint density coverage remaining blocker decision"
    RUN_ID="$decision_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision
  fi
  print_header "Stage B generic base scale checkpoint duration long-note repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe.py \
    --run_id "$run_id" \
    --remaining_blocker_decision "$remaining_blocker_decision" \
    --density_coverage_repair "$density_coverage_repair" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DURATION_LONG_NOTE_REPAIR_PROBE_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision \
    --require_target_qualified \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision}"
  local repair_report="outputs/stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe/${repair_run_id}/stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B generic base scale checkpoint duration long-note repair probe"
    RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe
  fi
  print_header "Stage B generic base scale checkpoint duration long-note remaining blocker decision"
  "$PYTHON_BIN" scripts/decide_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker.py \
    --run_id "$run_id" \
    --duration_long_note_repair "$repair_report" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_DURATION_LONG_NOTE_REMAINING_BLOCKER_DECISION_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe \
    --require_dead_air_target \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe}"
  local remaining_blocker_decision="outputs/stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision/${decision_run_id}/stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision.json"
  local duration_long_note_repair="outputs/stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe/${repair_run_id}/stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe.json"
  if [[ ! -f "$duration_long_note_repair" ]]; then
    print_header "Stage B generic base scale checkpoint duration long-note repair probe"
    RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe
  fi
  if [[ ! -f "$remaining_blocker_decision" ]]; then
    print_header "Stage B generic base scale checkpoint duration long-note remaining blocker decision"
    RUN_ID="$decision_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision
  fi
  print_header "Stage B generic base scale checkpoint sustained coverage dead-air repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe.py \
    --run_id "$run_id" \
    --remaining_blocker_decision "$remaining_blocker_decision" \
    --duration_long_note_repair "$duration_long_note_repair" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_SUSTAINED_COVERAGE_DEAD_AIR_REPAIR_PROBE_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_objective_gate_consolidation \
    --require_target_qualified \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_objective_gate_consolidation() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_objective_gate_consolidation}"
  local repair_probe="outputs/stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe/${repair_run_id}/stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe.json"
  if [[ ! -f "$repair_probe" ]]; then
    print_header "Stage B generic base scale checkpoint sustained coverage dead-air repair probe"
    RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe
  fi
  print_header "Stage B generic base scale checkpoint objective gate consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_generic_base_scale_checkpoint_objective_gate.py \
    --run_id "$run_id" \
    --repair_probe "$repair_probe" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_OBJECTIVE_GATE_CONSOLIDATION_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_objective_gate_consolidation \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep \
    --require_repeatability_target \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep() {
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_objective_gate_consolidation}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep}"
  local consolidation="outputs/stage_b_generic_base_scale_checkpoint_objective_gate_consolidation/${consolidation_run_id}/stage_b_generic_base_scale_checkpoint_objective_gate_consolidation.json"
  local repair_probe="outputs/stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe/${repair_run_id}/stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe.json"
  if [[ ! -f "$repair_probe" ]]; then
    print_header "Stage B generic base scale checkpoint sustained coverage dead-air repair probe"
    RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe
  fi
  if [[ ! -f "$consolidation" ]]; then
    print_header "Stage B generic base scale checkpoint objective gate consolidation"
    RUN_ID="$consolidation_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_generic_base_scale_checkpoint_objective_gate_consolidation
  fi
  print_header "Stage B generic base scale checkpoint objective gate repeatability sweep"
  "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep.py \
    --run_id "$run_id" \
    --consolidation "$consolidation" \
    --repair_probe "$repair_probe" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_OBJECTIVE_GATE_REPEATABILITY_SWEEP_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep \
    --expected_next_boundary stage_b_generic_base_scale_checkpoint_repeatability_consolidation \
    --require_target_qualified \
    --require_no_quality_claim
}

run_stage_b_generic_base_scale_checkpoint_repeatability_consolidation() {
  local sweep_run_id="${SWEEP_RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep}"
  local run_id="${RUN_ID:-harness_stage_b_generic_base_scale_checkpoint_repeatability_consolidation}"
  local repeatability_sweep="outputs/stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep/${sweep_run_id}/stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep.json"
  if [[ ! -f "$repeatability_sweep" ]]; then
    print_header "Stage B generic base scale checkpoint objective gate repeatability sweep"
    RUN_ID="$sweep_run_id" run_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep
  fi
  print_header "Stage B generic base scale checkpoint repeatability consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_generic_base_scale_checkpoint_repeatability.py \
    --run_id "$run_id" \
    --repeatability_sweep "$repeatability_sweep" \
    --doc_path docs/STAGE_B_GENERIC_BASE_SCALE_CHECKPOINT_REPEATABILITY_CONSOLIDATION_2026-06-01.md \
    --expected_boundary stage_b_generic_base_scale_checkpoint_repeatability_consolidation \
    --expected_next_boundary stage_b_model_core_evidence_readme_refresh \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_mvp_contract() {
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_mvp_contract}"
  print_header "Stage B MIDI-to-solo MVP input contract"
  "$PYTHON_BIN" scripts/define_stage_b_midi_to_solo_mvp_contract.py \
    --run_id "$run_id" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MVP_INPUT_CONTRACT_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_mvp_input_contract \
    --expected_next_boundary stage_b_midi_to_solo_context_extraction_mvp \
    --require_fallback \
    --require_no_final_claim
}

run_stage_b_midi_to_solo_context_extraction() {
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  print_header "Stage B MIDI-to-solo context extraction MVP"
  "$PYTHON_BIN" scripts/extract_stage_b_midi_to_solo_context.py \
    --run_id "$run_id" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTEXT_EXTRACTION_MVP_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_context_extraction_mvp \
    --expected_next_boundary stage_b_midi_to_solo_training_resource_probe \
    --min_context_bars 4 \
    --require_no_final_claim
}

run_stage_b_midi_to_solo_training_resource_probe() {
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local full_window_run_id="${FULL_WINDOW_RUN_ID:-harness_stage_b_generic_full_manifest_window_preparation}"
  local scale_training_run_id="${SCALE_TRAINING_RUN_ID:-harness_stage_b_generic_base_training_scale_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_training_resource_probe}"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local full_window_preparation="outputs/stage_b_generic_full_manifest_window_preparation/${full_window_run_id}/stage_b_generic_full_manifest_window_preparation.json"
  local training_scale_smoke="outputs/stage_b_generic_base_training_scale_smoke/${scale_training_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  if [[ ! -f "$full_window_preparation" ]]; then
    print_header "Stage B generic full manifest window preparation"
    RUN_ID="$full_window_run_id" run_stage_b_generic_full_manifest_window_preparation
  fi
  if [[ ! -f "$training_scale_smoke" ]]; then
    print_header "Stage B generic base training scale smoke"
    RUN_ID="$scale_training_run_id" FULL_WINDOW_RUN_ID="$full_window_run_id" run_stage_b_generic_base_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo training resource probe"
  "$PYTHON_BIN" scripts/check_stage_b_midi_to_solo_training_resource_probe.py \
    --run_id "$run_id" \
    --context_report "$context_report" \
    --full_window_preparation "$full_window_preparation" \
    --training_scale_smoke "$training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_TRAINING_RESOURCE_PROBE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_training_resource_probe \
    --expected_next_boundary stage_b_midi_to_solo_conditioned_generation_probe \
    --require_ready \
    --require_no_final_claim
}

run_stage_b_midi_to_solo_conditioned_generation_probe() {
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local resource_run_id="${RESOURCE_RUN_ID:-harness_stage_b_midi_to_solo_training_resource_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_conditioned_generation_probe}"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local resource_probe="outputs/stage_b_midi_to_solo_training_resource_probe/${resource_run_id}/stage_b_midi_to_solo_training_resource_probe.json"
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  if [[ ! -f "$resource_probe" ]]; then
    print_header "Stage B MIDI-to-solo training resource probe"
    RUN_ID="$resource_run_id" CONTEXT_RUN_ID="$context_run_id" run_stage_b_midi_to_solo_training_resource_probe
  fi
  print_header "Stage B MIDI-to-solo conditioned generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_conditioned_generation_probe.py \
    --run_id "$run_id" \
    --context_report "$context_report" \
    --resource_probe "$resource_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONDITIONED_GENERATION_PROBE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_conditioned_generation_probe \
    --expected_next_boundary stage_b_midi_to_solo_candidate_audio_render_package \
    --require_exported_candidates \
    --require_no_final_claim
}

run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline() {
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_retrieval_baseline}"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  print_header "Stage B MIDI-to-solo phrase-bank retrieval baseline"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline.py \
    --run_id "$run_id" \
    --context_report "$context_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_RETRIEVAL_BASELINE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_retrieval_baseline \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_audio_render_package \
    --require_exported_candidates \
    --require_no_final_claim
}

run_stage_b_midi_to_solo_phrase_bank_audio_render_package() {
  local phrase_bank_run_id="${PHRASE_BANK_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_retrieval_baseline}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_audio_render_package}"
  local phrase_bank_report="outputs/stage_b_midi_to_solo_phrase_bank_retrieval_baseline/${phrase_bank_run_id}/stage_b_midi_to_solo_phrase_bank_retrieval_baseline.json"
  if [[ ! -f "$phrase_bank_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank retrieval baseline"
    RUN_ID="$phrase_bank_run_id" run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline
  fi
  print_header "Stage B MIDI-to-solo phrase-bank audio render package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_phrase_bank_audio.py \
    --run_id "$run_id" \
    --phrase_bank_report "$phrase_bank_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_AUDIO_RENDER_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_audio_render_package \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_listening_review_package \
    --require_phrase_bank_audio_path \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_listening_review_package() {
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_listening_review_package}"
  local audio_render_report="outputs/stage_b_midi_to_solo_phrase_bank_audio_render_package/${audio_run_id}/stage_b_midi_to_solo_phrase_bank_audio_render_package.json"
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank audio render package"
    RUN_ID="$audio_run_id" run_stage_b_midi_to_solo_phrase_bank_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo phrase-bank listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_phrase_bank_listening_review_package.py \
    --run_id "$run_id" \
    --audio_render_report "$audio_render_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_LISTENING_REVIEW_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_listening_review_package \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_listening_review_input_guard \
    --require_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_listening_review_input_guard() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_listening_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_listening_review_input_guard}"
  local source_package="outputs/stage_b_midi_to_solo_phrase_bank_listening_review_package/${package_run_id}/stage_b_midi_to_solo_phrase_bank_listening_review_package.json"
  if [[ ! -f "$source_package" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank listening review package"
    RUN_ID="$package_run_id" run_stage_b_midi_to_solo_phrase_bank_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo phrase-bank listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_phrase_bank_listening_review_input.py \
    --run_id "$run_id" \
    --source_package "$source_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_LISTENING_REVIEW_INPUT_GUARD_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_objective_only_next_decision \
    --require_guard_completed \
    --require_pending_input \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_objective_only_next_decision() {
  local phrase_bank_run_id="${PHRASE_BANK_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_retrieval_baseline}"
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_audio_render_package}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_listening_review_input_guard}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_objective_only_next_decision}"
  local phrase_bank_report="outputs/stage_b_midi_to_solo_phrase_bank_retrieval_baseline/${phrase_bank_run_id}/stage_b_midi_to_solo_phrase_bank_retrieval_baseline.json"
  local audio_render_report="outputs/stage_b_midi_to_solo_phrase_bank_audio_render_package/${audio_run_id}/stage_b_midi_to_solo_phrase_bank_audio_render_package.json"
  local input_guard_report="outputs/stage_b_midi_to_solo_phrase_bank_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_phrase_bank_listening_review_input_guard.json"
  if [[ ! -f "$phrase_bank_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank retrieval baseline"
    RUN_ID="$phrase_bank_run_id" run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline
  fi
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank audio render package"
    RUN_ID="$audio_run_id" PHRASE_BANK_RUN_ID="$phrase_bank_run_id" run_stage_b_midi_to_solo_phrase_bank_audio_render_package
  fi
  if [[ ! -f "$input_guard_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank listening review input guard"
    RUN_ID="$input_guard_run_id" AUDIO_RUN_ID="$audio_run_id" PHRASE_BANK_RUN_ID="$phrase_bank_run_id" run_stage_b_midi_to_solo_phrase_bank_listening_review_input_guard
  fi
  print_header "Stage B MIDI-to-solo phrase-bank objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_phrase_bank_objective_next.py \
    --run_id "$run_id" \
    --input_guard_report "$input_guard_report" \
    --phrase_bank_report "$phrase_bank_report" \
    --audio_render_report "$audio_render_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe \
    --require_objective_decision \
    --require_repair_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe() {
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_objective_only_next_decision}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe}"
  local objective_next_report="outputs/stage_b_midi_to_solo_phrase_bank_objective_only_next_decision/${objective_next_run_id}/stage_b_midi_to_solo_phrase_bank_objective_only_next_decision.json"
  if [[ ! -f "$objective_next_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank objective-only next decision"
    RUN_ID="$objective_next_run_id" run_stage_b_midi_to_solo_phrase_bank_objective_only_next_decision
  fi
  print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe.py \
    --run_id "$run_id" \
    --objective_next_report "$objective_next_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_DEAD_AIR_DENSITY_REPAIR_PROBE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package \
    --require_repair_probe_completed \
    --require_target_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package}"
  local repair_report="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/${repair_run_id}/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair probe"
    RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe
  fi
  print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair audio package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio.py \
    --run_id "$run_id" \
    --repair_report "$repair_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_DEAD_AIR_DENSITY_REPAIR_AUDIO_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package \
    --expected_file_count 3 \
    --require_repaired_audio_path \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package() {
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package}"
  local audio_render_report="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package/${audio_run_id}/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package.json"
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair audio package"
    RUN_ID="$audio_run_id" run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package
  fi
  print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package.py \
    --run_id "$run_id" \
    --audio_render_report "$audio_render_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_DEAD_AIR_DENSITY_REPAIR_LISTENING_REVIEW_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard \
    --expected_review_item_count 3 \
    --require_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard}"
  local source_package="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package/${package_run_id}/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package.json"
  if [[ ! -f "$source_package" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair listening review package"
    RUN_ID="$package_run_id" run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input.py \
    --run_id "$run_id" \
    --source_package "$source_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_DEAD_AIR_DENSITY_REPAIR_LISTENING_REVIEW_INPUT_GUARD_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision \
    --require_guard_completed \
    --require_pending_input \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision() {
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe}"
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision}"
  local input_guard_report="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard.json"
  local repair_probe_report="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe/${repair_run_id}/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe.json"
  local audio_package_report="outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package/${audio_run_id}/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package.json"
  if [[ ! -f "$repair_probe_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair probe"
    RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe
  fi
  if [[ ! -f "$audio_package_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair audio package"
    RUN_ID="$audio_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package
  fi
  if [[ ! -f "$input_guard_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair listening review input guard"
    RUN_ID="$input_guard_run_id" AUDIO_RUN_ID="$audio_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard
  fi
  print_header "Stage B MIDI-to-solo phrase-bank dead-air density repair objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_next.py \
    --run_id "$run_id" \
    --input_guard_report "$input_guard_report" \
    --repair_probe_report "$repair_probe_report" \
    --audio_package_report "$audio_package_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_DEAD_AIR_DENSITY_REPAIR_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_mvp_package \
    --require_objective_decision \
    --require_cli_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package() {
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package}"
  print_header "Stage B MIDI-to-solo phrase-bank CLI MVP package"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py \
    --run_id "$run_id" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_CLI_MVP_PACKAGE_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_mvp_package \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke \
    --require_cli_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke() {
  local input_midi="${INPUT_MIDI:-midi_dataset/midi/studio/Geri Allen/Home Grown/Alone Together.midi}"
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke}"
  local package_report="outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/${package_run_id}/stage_b_midi_to_solo_phrase_bank_cli_mvp_package.json"
  if [[ ! -f "$package_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI MVP package explicit input"
    "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package.py \
      --input_midi "$input_midi" \
      --run_id "$package_run_id" \
      --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_mvp_package \
      --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke \
      --require_cli_ready \
      --require_no_quality_claim
  fi
  print_header "Stage B MIDI-to-solo phrase-bank CLI user-input smoke"
  "$PYTHON_BIN" scripts/check_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke.py \
    --run_id "$run_id" \
    --cli_package_report "$package_report" \
    --expected_input_midi "$input_midi" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_CLI_USER_INPUT_SMOKE_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke \
    --require_explicit_input \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke() {
  local smoke_run_id="${SMOKE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke}"
  local smoke_report="outputs/stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke/${smoke_run_id}/stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke.json"
  if [[ ! -f "$smoke_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI user-input smoke"
    RUN_ID="$smoke_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke
  fi
  print_header "Stage B MIDI-to-solo phrase-bank CLI audio render smoke"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_phrase_bank_cli_audio_smoke.py \
    --run_id "$run_id" \
    --user_input_smoke_report "$smoke_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_CLI_AUDIO_RENDER_SMOKE_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_listening_review_package \
    --expected_file_count 3 \
    --sample_rate 44100 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package() {
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package}"
  local audio_render_report="outputs/stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke/${audio_run_id}/stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke.json"
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI audio render smoke"
    RUN_ID="$audio_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke
  fi
  print_header "Stage B MIDI-to-solo phrase-bank CLI listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package.py \
    --run_id "$run_id" \
    --audio_render_report "$audio_render_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_CLI_LISTENING_REVIEW_PACKAGE_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_listening_review_package \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard \
    --expected_review_item_count 3 \
    --require_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard}"
  local source_package="outputs/stage_b_midi_to_solo_phrase_bank_cli_listening_review_package/${package_run_id}/stage_b_midi_to_solo_phrase_bank_cli_listening_review_package.json"
  if [[ ! -f "$source_package" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI listening review package"
    RUN_ID="$package_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo phrase-bank CLI listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input.py \
    --run_id "$run_id" \
    --source_package "$source_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_CLI_LISTENING_REVIEW_INPUT_GUARD_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision \
    --require_guard_completed \
    --require_pending_input \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision() {
  local smoke_run_id="${SMOKE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke}"
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision}"
  local user_input_smoke_report="outputs/stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke/${smoke_run_id}/stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke.json"
  local audio_render_report="outputs/stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke/${audio_run_id}/stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke.json"
  local input_guard_report="outputs/stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard.json"
  if [[ ! -f "$user_input_smoke_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI user-input smoke"
    RUN_ID="$smoke_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke
  fi
  if [[ ! -f "$audio_render_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI audio render smoke"
    RUN_ID="$audio_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke
  fi
  if [[ ! -f "$input_guard_report" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI listening review input guard"
    RUN_ID="$input_guard_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard
  fi
  print_header "Stage B MIDI-to-solo phrase-bank CLI objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_phrase_bank_cli_objective_next.py \
    --run_id "$run_id" \
    --input_guard_report "$input_guard_report" \
    --user_input_smoke_report "$user_input_smoke_report" \
    --audio_render_report "$audio_render_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_PHRASE_BANK_CLI_OBJECTIVE_ONLY_NEXT_DECISION_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_mvp_current_evidence_consolidation \
    --require_objective_decision \
    --require_current_evidence_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_candidate_audio_render_package() {
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_conditioned_generation_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_candidate_audio_render_package}"
  local generation_report="outputs/stage_b_midi_to_solo_conditioned_generation_probe/${generation_run_id}/stage_b_midi_to_solo_conditioned_generation_probe.json"
  if [[ ! -f "$generation_report" ]]; then
    print_header "Stage B MIDI-to-solo conditioned generation probe"
    RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_conditioned_generation_probe
  fi
  print_header "Stage B MIDI-to-solo candidate audio render package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_candidate_audio.py \
    --run_id "$run_id" \
    --conditioned_generation_report "$generation_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CANDIDATE_AUDIO_RENDER_PACKAGE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_candidate_audio_render_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_mvp_execution_consolidation() {
  local contract_run_id="${CONTRACT_RUN_ID:-harness_stage_b_midi_to_solo_mvp_contract}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local resource_run_id="${RESOURCE_RUN_ID:-harness_stage_b_midi_to_solo_training_resource_probe}"
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_conditioned_generation_probe}"
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_candidate_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_mvp_execution_consolidation}"
  local contract_report="outputs/stage_b_midi_to_solo_mvp_contract/${contract_run_id}/stage_b_midi_to_solo_mvp_contract.json"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local resource_probe="outputs/stage_b_midi_to_solo_training_resource_probe/${resource_run_id}/stage_b_midi_to_solo_training_resource_probe.json"
  local generation_probe="outputs/stage_b_midi_to_solo_conditioned_generation_probe/${generation_run_id}/stage_b_midi_to_solo_conditioned_generation_probe.json"
  local audio_render="outputs/stage_b_midi_to_solo_candidate_audio_render_package/${audio_run_id}/stage_b_midi_to_solo_candidate_audio_render_package.json"
  if [[ ! -f "$contract_report" ]]; then
    print_header "Stage B MIDI-to-solo MVP input contract"
    RUN_ID="$contract_run_id" run_stage_b_midi_to_solo_mvp_contract
  fi
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  if [[ ! -f "$resource_probe" ]]; then
    print_header "Stage B MIDI-to-solo training resource probe"
    RUN_ID="$resource_run_id" CONTEXT_RUN_ID="$context_run_id" run_stage_b_midi_to_solo_training_resource_probe
  fi
  if [[ ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo conditioned generation probe"
    RUN_ID="$generation_run_id" CONTEXT_RUN_ID="$context_run_id" RESOURCE_RUN_ID="$resource_run_id" run_stage_b_midi_to_solo_conditioned_generation_probe
  fi
  if [[ ! -f "$audio_render" ]]; then
    print_header "Stage B MIDI-to-solo candidate audio render package"
    RUN_ID="$audio_run_id" GENERATION_RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_candidate_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo MVP execution consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_mvp_execution.py \
    --run_id "$run_id" \
    --contract_report "$contract_report" \
    --context_report "$context_report" \
    --resource_probe "$resource_probe" \
    --generation_probe "$generation_probe" \
    --audio_render "$audio_render" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MVP_EXECUTION_CONSOLIDATION_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_mvp_execution_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_generation_repair \
    --require_technical_mvp \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_generation_repair() {
  local mvp_run_id="${MVP_RUN_ID:-harness_stage_b_midi_to_solo_mvp_execution_consolidation}"
  local scale_smoke_run_id="${SCALE_SMOKE_RUN_ID:-harness_stage_b_generic_base_training_scale_smoke}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_generation_repair}"
  local mvp_execution="outputs/stage_b_midi_to_solo_mvp_execution_consolidation/${mvp_run_id}/stage_b_midi_to_solo_mvp_execution_consolidation.json"
  local training_scale_smoke="outputs/stage_b_generic_base_training_scale_smoke/${scale_smoke_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$mvp_execution" ]]; then
    print_header "Stage B MIDI-to-solo MVP execution consolidation"
    RUN_ID="$mvp_run_id" run_stage_b_midi_to_solo_mvp_execution_consolidation
  fi
  if [[ ! -f "$training_scale_smoke" ]]; then
    print_header "Stage B generic base training scale smoke"
    RUN_ID="$scale_smoke_run_id" run_stage_b_generic_base_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo model-direct generation repair"
  "$PYTHON_BIN" scripts/check_stage_b_midi_to_solo_model_direct_generation_repair.py \
    --run_id "$run_id" \
    --mvp_execution "$mvp_execution" \
    --training_scale_smoke "$training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_GENERATION_REPAIR_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_generation_repair \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke \
    --require_technical_mvp \
    --require_sequence_budget_gap \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke() {
  local previous_repair_run_id="${PREVIOUS_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_generation_repair}"
  local full_window_run_id="${FULL_WINDOW_RUN_ID:-harness_stage_b_generic_full_manifest_window_preparation}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local previous_repair="outputs/stage_b_midi_to_solo_model_direct_generation_repair/${previous_repair_run_id}/stage_b_midi_to_solo_model_direct_generation_repair.json"
  local full_window_preparation="outputs/stage_b_generic_full_manifest_window_preparation/${full_window_run_id}/stage_b_generic_full_manifest_window_preparation.json"
  local scale_output_root="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${run_id}/scale_smoke"
  local repaired_training_scale_smoke="${scale_output_root}/${scale_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$previous_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct generation repair"
    RUN_ID="$previous_repair_run_id" run_stage_b_midi_to_solo_model_direct_generation_repair
  fi
  if [[ ! -f "$full_window_preparation" ]]; then
    print_header "Stage B generic full manifest window preparation"
    RUN_ID="$full_window_run_id" run_stage_b_generic_full_manifest_window_preparation
  fi
  if [[ ! -f "$repaired_training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo model-direct max_sequence 160 scale smoke"
    "$PYTHON_BIN" scripts/run_stage_b_generic_base_training_scale_smoke.py \
      --run_id "$scale_run_id" \
      --output_root "$scale_output_root" \
      --full_window_preparation "$full_window_preparation" \
      --max_sequence 160 \
      --expected_boundary stage_b_generic_base_training_scale_smoke \
      --expected_next_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
      --require_training_scale_smoke_passed \
      --require_no_broad_quality_claim \
      --require_no_brad_style_claim
  fi
  print_header "Stage B MIDI-to-solo model-direct sequence budget repair smoke"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.py \
    --run_id "$run_id" \
    --previous_repair "$previous_repair" \
    --repaired_training_scale_smoke "$repaired_training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_SEQUENCE_BUDGET_REPAIR_SMOKE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_8bar_generation_probe \
    --require_sequence_budget_sufficient \
    --require_probe_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_8bar_generation_probe() {
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_repair="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local repaired_training_scale_smoke="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/scale_smoke/${scale_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$sequence_budget_repair" || ! -f "$repaired_training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo model-direct sequence budget repair smoke"
    RUN_ID="$sequence_budget_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke
  fi
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  print_header "Stage B MIDI-to-solo model-direct 8-bar generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_8bar_generation_probe.py \
    --run_id "$run_id" \
    --sequence_budget_repair "$sequence_budget_repair" \
    --context_report "$context_report" \
    --repaired_training_scale_smoke "$repaired_training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_8BAR_GENERATION_PROBE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_8bar_generation_probe \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_monophonic_overlap_repair \
    --require_probe_completed \
    --require_generated_midi \
    --require_grammar_gate \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair() {
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_probe="outputs/stage_b_midi_to_solo_model_direct_8bar_generation_probe/${previous_direct_run_id}/stage_b_midi_to_solo_model_direct_8bar_generation_probe.json"
  local sequence_budget_repair="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local repaired_training_scale_smoke="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/scale_smoke/${scale_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$previous_direct_probe" ]]; then
    print_header "Stage B MIDI-to-solo model-direct 8-bar generation probe"
    RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_8bar_generation_probe
  fi
  if [[ ! -f "$sequence_budget_repair" || ! -f "$repaired_training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo model-direct sequence budget repair smoke"
    RUN_ID="$sequence_budget_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke
  fi
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  print_header "Stage B MIDI-to-solo model-direct monophonic overlap repair"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.py \
    --run_id "$run_id" \
    --previous_direct_probe "$previous_direct_probe" \
    --sequence_budget_repair "$sequence_budget_repair" \
    --context_report "$context_report" \
    --repaired_training_scale_smoke "$repaired_training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_MONOPHONIC_OVERLAP_REPAIR_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_monophonic_overlap_repair \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_audio_render_package \
    --require_repair_completed \
    --require_review_gate_repaired \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_audio_render_package() {
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair="outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/${overlap_repair_run_id}/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.json"
  if [[ ! -f "$overlap_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct monophonic overlap repair"
    RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair
  fi
  print_header "Stage B MIDI-to-solo model-direct audio render package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_model_direct_audio.py \
    --run_id "$run_id" \
    --model_direct_overlap_repair "$overlap_repair" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_AUDIO_RENDER_PACKAGE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_audio_render_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation() {
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local overlap_repair="outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/${overlap_repair_run_id}/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.json"
  local audio_render="outputs/stage_b_midi_to_solo_model_direct_audio_render_package/${audio_render_run_id}/stage_b_midi_to_solo_model_direct_audio_render_package.json"
  if [[ ! -f "$audio_render" ]]; then
    print_header "Stage B MIDI-to-solo model-direct audio render package"
    RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo model-direct audio evidence consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_model_direct_audio_evidence.py \
    --run_id "$run_id" \
    --objective_report "$overlap_repair" \
    --audio_render_report "$audio_render" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_AUDIO_EVIDENCE_CONSOLIDATION_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_audio_evidence_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics \
    --min_midi_count 3 \
    --min_wav_count 3 \
    --require_technical_path \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics() {
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_report="outputs/stage_b_midi_to_solo_model_direct_audio_evidence_consolidation/${evidence_run_id}/stage_b_midi_to_solo_model_direct_audio_evidence_consolidation.json"
  if [[ ! -f "$evidence_report" ]]; then
    print_header "Stage B MIDI-to-solo model-direct audio evidence consolidation"
    RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation
  fi
  print_header "Stage B MIDI-to-solo model-direct phrase quality diagnostics"
  "$PYTHON_BIN" scripts/diagnose_stage_b_midi_to_solo_model_direct_phrase_quality.py \
    --run_id "$run_id" \
    --audio_evidence_report "$evidence_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_PHRASE_QUALITY_DIAGNOSTICS_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics \
    --min_candidate_count 3 \
    --require_diagnostics_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_pitch_contour_repair() {
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_report="outputs/stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics/${diagnostics_run_id}/stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics.json"
  local sequence_budget_repair="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local repaired_training_scale_smoke="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/scale_smoke/${scale_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$diagnostics_report" ]]; then
    print_header "Stage B MIDI-to-solo model-direct phrase quality diagnostics"
    RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics
  fi
  if [[ ! -f "$sequence_budget_repair" || ! -f "$repaired_training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo model-direct sequence budget repair smoke"
    RUN_ID="$sequence_budget_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke
  fi
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  print_header "Stage B MIDI-to-solo model-direct pitch contour repair"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_pitch_contour_repair.py \
    --run_id "$run_id" \
    --source_diagnostics "$diagnostics_report" \
    --sequence_budget_repair "$sequence_budget_repair" \
    --context_report "$context_report" \
    --repaired_training_scale_smoke "$repaired_training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_PITCH_CONTOUR_REPETITION_REPAIR_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_pitch_contour_repetition_repair \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_timing_phrase_repair \
    --require_repair_completed \
    --require_pitch_repair_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_timing_phrase_repair() {
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair="outputs/stage_b_midi_to_solo_model_direct_pitch_contour_repetition_repair/${pitch_repair_run_id}/stage_b_midi_to_solo_model_direct_pitch_contour_repetition_repair.json"
  local sequence_budget_repair="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local repaired_training_scale_smoke="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/scale_smoke/${scale_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$pitch_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct pitch contour repair"
    RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_pitch_contour_repair
  fi
  if [[ ! -f "$sequence_budget_repair" || ! -f "$repaired_training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo model-direct sequence budget repair smoke"
    RUN_ID="$sequence_budget_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke
  fi
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  print_header "Stage B MIDI-to-solo model-direct timing phrase repair"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_timing_phrase_repair.py \
    --run_id "$run_id" \
    --source_pitch_repair "$pitch_repair" \
    --sequence_budget_repair "$sequence_budget_repair" \
    --context_report "$context_report" \
    --repaired_training_scale_smoke "$repaired_training_scale_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_TIMING_PHRASE_REPAIR_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_timing_phrase_repair \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_listening_review_package \
    --require_repair_completed \
    --require_timing_repair_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_listening_review_package() {
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local timing_repair="outputs/stage_b_midi_to_solo_model_direct_timing_phrase_repair/${timing_repair_run_id}/stage_b_midi_to_solo_model_direct_timing_phrase_repair.json"
  if [[ ! -f "$timing_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct timing phrase repair"
    RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_timing_phrase_repair
  fi
  print_header "Stage B MIDI-to-solo model-direct listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_listening_review_package.py \
    --run_id "$run_id" \
    --timing_phrase_repair "$timing_repair" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_LISTENING_REVIEW_PACKAGE_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_listening_review_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard() {
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard}"
  local review_package="outputs/stage_b_midi_to_solo_model_direct_listening_review_package/${review_package_run_id}/stage_b_midi_to_solo_model_direct_listening_review_package.json"
  if [[ ! -f "$review_package" ]]; then
    print_header "Stage B MIDI-to-solo model-direct listening review package"
    RUN_ID="$review_package_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo model-direct user listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_model_direct_user_listening_review_input.py \
    --run_id "$run_id" \
    --listening_review_package "$review_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_USER_LISTENING_REVIEW_INPUT_GUARD_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_user_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_objective_only_next_decision \
    --require_guard_completed \
    --require_pending_input \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_user_listening_review_fill() {
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard}"
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_fill}"
  local review_package="outputs/stage_b_midi_to_solo_model_direct_listening_review_package/${review_package_run_id}/stage_b_midi_to_solo_model_direct_listening_review_package.json"
  local input_guard="outputs/stage_b_midi_to_solo_model_direct_user_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_model_direct_user_listening_review_input_guard.json"
  if [[ ! -f "$review_package" ]]; then
    print_header "Stage B MIDI-to-solo model-direct listening review package"
    RUN_ID="$review_package_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_listening_review_package
  fi
  if [[ ! -f "$input_guard" ]]; then
    print_header "Stage B MIDI-to-solo model-direct user listening review input guard"
    RUN_ID="$input_guard_run_id" REVIEW_PACKAGE_RUN_ID="$review_package_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard
  fi
  print_header "Stage B MIDI-to-solo model-direct user listening review fill"
  "$PYTHON_BIN" scripts/fill_stage_b_midi_to_solo_model_direct_user_listening_review.py \
    --run_id "$run_id" \
    --listening_review_package "$review_package" \
    --input_guard_report "$input_guard" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_USER_LISTENING_REVIEW_FILL_2026-06-03.md \
    --reviewer user \
    --preferred_rank 3 \
    --overall_decision reject_all \
    --candidate_decision relative_best_needs_followup \
    --timing songlike_not_soloing \
    --phrase songlike_not_soloing \
    --vocabulary songlike_not_soloing \
    --primary_failure songlike_melody_not_soloing \
    --assessment "rank 3 is relatively best, but all candidates sound like simple song melody rather than jazz soloing" \
    --notes "single user listening review of three rendered WAV files" \
    --expected_boundary stage_b_midi_to_solo_model_direct_user_listening_review_fill \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis \
    --expected_overall_decision reject_all \
    --expected_preferred_rank 3 \
    --require_review_completed \
    --require_no_keep_claim \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_songlike_rejection_analysis() {
  local review_fill_run_id="${REVIEW_FILL_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_fill}"
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard}"
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis}"
  local review_fill="outputs/stage_b_midi_to_solo_model_direct_user_listening_review_fill/${review_fill_run_id}/stage_b_midi_to_solo_model_direct_user_listening_review_fill.json"
  if [[ ! -f "$review_fill" ]]; then
    print_header "Stage B MIDI-to-solo model-direct user listening review fill"
    RUN_ID="$review_fill_run_id" REVIEW_PACKAGE_RUN_ID="$review_package_run_id" INPUT_GUARD_RUN_ID="$input_guard_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_user_listening_review_fill
  fi
  print_header "Stage B MIDI-to-solo model-direct songlike melody rejection analysis"
  "$PYTHON_BIN" scripts/analyze_stage_b_midi_to_solo_model_direct_songlike_rejection.py \
    --run_id "$run_id" \
    --user_listening_review_fill "$review_fill" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_SONGLIKE_MELODY_REJECTION_ANALYSIS_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision \
    --require_analysis_completed \
    --require_rejection_signals \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision() {
  local songlike_analysis_run_id="${SONGLIKE_ANALYSIS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis}"
  local review_fill_run_id="${REVIEW_FILL_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_fill}"
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard}"
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision}"
  local songlike_analysis="outputs/stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis/${songlike_analysis_run_id}/stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis.json"
  if [[ ! -f "$songlike_analysis" ]]; then
    print_header "Stage B MIDI-to-solo model-direct songlike melody rejection analysis"
    RUN_ID="$songlike_analysis_run_id" REVIEW_FILL_RUN_ID="$review_fill_run_id" REVIEW_PACKAGE_RUN_ID="$review_package_run_id" INPUT_GUARD_RUN_ID="$input_guard_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_songlike_rejection_analysis
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair.py \
    --run_id "$run_id" \
    --songlike_rejection_analysis "$songlike_analysis" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_REPAIR_DECISION_2026-06-03.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe \
    --min_repair_target_count 6 \
    --require_auto_progress_allowed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe() {
  local repair_decision_run_id="${REPAIR_DECISION_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision}"
  local songlike_analysis_run_id="${SONGLIKE_ANALYSIS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis}"
  local review_fill_run_id="${REVIEW_FILL_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_fill}"
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard}"
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local repair_decision="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision/${repair_decision_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision.json"
  if [[ ! -f "$repair_decision" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair decision"
    RUN_ID="$repair_decision_run_id" SONGLIKE_ANALYSIS_RUN_ID="$songlike_analysis_run_id" REVIEW_FILL_RUN_ID="$review_fill_run_id" REVIEW_PACKAGE_RUN_ID="$review_package_run_id" INPUT_GUARD_RUN_ID="$input_guard_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe.py \
    --run_id "$run_id" \
    --repair_decision "$repair_decision" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package \
    --require_probe_completed \
    --require_target_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package() {
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local repair_decision_run_id="${REPAIR_DECISION_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision}"
  local songlike_analysis_run_id="${SONGLIKE_ANALYSIS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis}"
  local review_fill_run_id="${REVIEW_FILL_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_fill}"
  local review_package_run_id="${REVIEW_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_listening_review_package}"
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard}"
  local timing_repair_run_id="${TIMING_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_timing_phrase_repair}"
  local pitch_repair_run_id="${PITCH_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_pitch_contour_repair}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics}"
  local evidence_run_id="${EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local overlap_repair_run_id="${OVERLAP_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local previous_direct_run_id="${PREVIOUS_DIRECT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_8bar_generation_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local scale_run_id="${SCALE_RUN_ID:-max_sequence_160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe/${repair_probe_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe.json"
  if [[ ! -f "$repair_probe" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair probe"
    RUN_ID="$repair_probe_run_id" REPAIR_DECISION_RUN_ID="$repair_decision_run_id" SONGLIKE_ANALYSIS_RUN_ID="$songlike_analysis_run_id" REVIEW_FILL_RUN_ID="$review_fill_run_id" REVIEW_PACKAGE_RUN_ID="$review_package_run_id" INPUT_GUARD_RUN_ID="$input_guard_run_id" TIMING_REPAIR_RUN_ID="$timing_repair_run_id" PITCH_REPAIR_RUN_ID="$pitch_repair_run_id" DIAGNOSTICS_RUN_ID="$diagnostics_run_id" EVIDENCE_RUN_ID="$evidence_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" OVERLAP_REPAIR_RUN_ID="$overlap_repair_run_id" PREVIOUS_DIRECT_RUN_ID="$previous_direct_run_id" SEQUENCE_BUDGET_RUN_ID="$sequence_budget_run_id" CONTEXT_RUN_ID="$context_run_id" SCALE_RUN_ID="$scale_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair audio package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package.py \
    --run_id "$run_id" \
    --repair_probe "$repair_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_REPAIR_AUDIO_PACKAGE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review() {
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package/${audio_package_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package.json"
  if [[ ! -f "$audio_package" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair audio package"
    RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair listening review"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review.py \
    --run_id "$run_id" \
    --audio_package "$audio_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_REPAIR_LISTENING_REVIEW_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_only_next_decision \
    --expected_file_count 3 \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next() {
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review/${listening_review_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review.json"
  if [[ ! -f "$listening_review" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair listening review"
    RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next.py \
    --run_id "$run_id" \
    --listening_review "$listening_review" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_REPAIR_OBJECTIVE_NEXT_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair \
    --min_repair_target_count 6 \
    --require_stepwise_target \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair() {
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next/${objective_next_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next.json"
  if [[ ! -f "$objective_next" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary repair objective-only next decision"
    RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repair"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair.py \
    --run_id "$run_id" \
    --objective_next "$objective_next" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_REPAIR_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package \
    --require_repair_completed \
    --require_target_passed \
    --require_stepwise_reduced \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package() {
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair/${contour_repair_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair.json"
  if [[ ! -f "$contour_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repair"
    RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape audio package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package.py \
    --run_id "$run_id" \
    --contour_repair "$contour_repair" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_AUDIO_PACKAGE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review() {
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package/${contour_audio_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package.json"
  if [[ ! -f "$contour_audio" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape audio package"
    RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape listening review"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review.py \
    --run_id "$run_id" \
    --audio_package "$contour_audio" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_LISTENING_REVIEW_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_only_next_decision \
    --expected_file_count 3 \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next() {
  local contour_listening_run_id="${CONTOUR_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next}"
  local contour_listening="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review/${contour_listening_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review.json"
  local contour_repair="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair/${contour_repair_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair.json"
  if [[ ! -f "$contour_listening" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape listening review"
    RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review
  fi
  if [[ ! -f "$contour_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repair"
    RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next.py \
    --run_id "$run_id" \
    --listening_review "$contour_listening" \
    --contour_repair "$contour_repair" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_OBJECTIVE_NEXT_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_sweep \
    --require_objective_clean \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep() {
  local contour_objective_run_id="${CONTOUR_OBJECTIVE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next}"
  local contour_listening_run_id="${CONTOUR_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep}"
  local contour_objective="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next/${contour_objective_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next.json"
  if [[ ! -f "$contour_objective" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-only next decision"
    RUN_ID="$contour_objective_run_id" CONTOUR_LISTENING_RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-clean repeatability sweep"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep.py \
    --run_id "$run_id" \
    --objective_next "$contour_objective" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_REPEATABILITY_SWEEP_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_sweep \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_consolidation \
    --sample_count 6 \
    --max_interval 12 \
    --min_sample_count 6 \
    --min_qualified_count 6 \
    --require_repeatability_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation() {
  local repeatability_sweep_run_id="${REPEATABILITY_SWEEP_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep}"
  local contour_objective_run_id="${CONTOUR_OBJECTIVE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next}"
  local contour_listening_run_id="${CONTOUR_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation}"
  local repeatability_sweep="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep/${repeatability_sweep_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep.json"
  if [[ ! -f "$repeatability_sweep" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape objective-clean repeatability sweep"
    RUN_ID="$repeatability_sweep_run_id" CONTOUR_OBJECTIVE_RUN_ID="$contour_objective_run_id" CONTOUR_LISTENING_RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability.py \
    --run_id "$run_id" \
    --repeatability_sweep "$repeatability_sweep" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_REPEATABILITY_CONSOLIDATION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_clean_repeatability_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_review_package \
    --min_sample_count 6 \
    --min_qualified_count 6 \
    --require_objective_support \
    --require_audio_review_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package() {
  local repeatability_consolidation_run_id="${REPEATABILITY_CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation}"
  local repeatability_sweep_run_id="${REPEATABILITY_SWEEP_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep}"
  local contour_objective_run_id="${CONTOUR_OBJECTIVE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next}"
  local contour_listening_run_id="${CONTOUR_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package}"
  local repeatability_consolidation="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation/${repeatability_consolidation_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation.json"
  if [[ ! -f "$repeatability_consolidation" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability consolidation"
    RUN_ID="$repeatability_consolidation_run_id" REPEATABILITY_SWEEP_RUN_ID="$repeatability_sweep_run_id" CONTOUR_OBJECTIVE_RUN_ID="$contour_objective_run_id" CONTOUR_LISTENING_RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability audio package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package.py \
    --run_id "$run_id" \
    --repeatability_consolidation "$repeatability_consolidation" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_REPEATABILITY_AUDIO_PACKAGE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_review_package \
    --expected_file_count 6 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review() {
  local repeatability_audio_run_id="${REPEATABILITY_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package}"
  local repeatability_consolidation_run_id="${REPEATABILITY_CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation}"
  local repeatability_sweep_run_id="${REPEATABILITY_SWEEP_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep}"
  local contour_objective_run_id="${CONTOUR_OBJECTIVE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next}"
  local contour_listening_run_id="${CONTOUR_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review}"
  local repeatability_audio="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package/${repeatability_audio_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package.json"
  if [[ ! -f "$repeatability_audio" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability audio package"
    RUN_ID="$repeatability_audio_run_id" REPEATABILITY_CONSOLIDATION_RUN_ID="$repeatability_consolidation_run_id" REPEATABILITY_SWEEP_RUN_ID="$repeatability_sweep_run_id" CONTOUR_OBJECTIVE_RUN_ID="$contour_objective_run_id" CONTOUR_LISTENING_RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability listening review"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review.py \
    --run_id "$run_id" \
    --audio_package "$repeatability_audio" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_REPEATABILITY_LISTENING_REVIEW_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review \
    --expected_next_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_only_next_decision \
    --expected_file_count 6 \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next() {
  local repeatability_listening_run_id="${REPEATABILITY_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review}"
  local repeatability_audio_run_id="${REPEATABILITY_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package}"
  local repeatability_consolidation_run_id="${REPEATABILITY_CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation}"
  local repeatability_sweep_run_id="${REPEATABILITY_SWEEP_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep}"
  local contour_objective_run_id="${CONTOUR_OBJECTIVE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next}"
  local contour_listening_run_id="${CONTOUR_LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review}"
  local contour_audio_run_id="${CONTOUR_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package}"
  local contour_repair_run_id="${CONTOUR_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next}"
  local listening_review_run_id="${LISTENING_REVIEW_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review}"
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next}"
  local repeatability_listening="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review/${repeatability_listening_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review.json"
  local repeatability_consolidation="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation/${repeatability_consolidation_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation.json"
  if [[ ! -f "$repeatability_listening" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability listening review"
    RUN_ID="$repeatability_listening_run_id" REPEATABILITY_AUDIO_RUN_ID="$repeatability_audio_run_id" REPEATABILITY_CONSOLIDATION_RUN_ID="$repeatability_consolidation_run_id" REPEATABILITY_SWEEP_RUN_ID="$repeatability_sweep_run_id" CONTOUR_OBJECTIVE_RUN_ID="$contour_objective_run_id" CONTOUR_LISTENING_RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review
  fi
  if [[ ! -f "$repeatability_consolidation" ]]; then
    print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability consolidation"
    RUN_ID="$repeatability_consolidation_run_id" REPEATABILITY_SWEEP_RUN_ID="$repeatability_sweep_run_id" CONTOUR_OBJECTIVE_RUN_ID="$contour_objective_run_id" CONTOUR_LISTENING_RUN_ID="$contour_listening_run_id" CONTOUR_AUDIO_RUN_ID="$contour_audio_run_id" CONTOUR_REPAIR_RUN_ID="$contour_repair_run_id" OBJECTIVE_NEXT_RUN_ID="$objective_next_run_id" LISTENING_REVIEW_RUN_ID="$listening_review_run_id" AUDIO_PACKAGE_RUN_ID="$audio_package_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation
  fi
  print_header "Stage B MIDI-to-solo model-direct jazz phrase vocabulary contour phrase-shape repeatability objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next.py \
    --run_id "$run_id" \
    --listening_review "$repeatability_listening" \
    --repeatability_consolidation "$repeatability_consolidation" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_DIRECT_JAZZ_PHRASE_VOCABULARY_CONTOUR_PHRASE_SHAPE_REPEATABILITY_OBJECTIVE_NEXT_DECISION_2026-06-04.md \
    --expected_final_boundary stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_path_complete \
    --expected_next_boundary stage_b_model_core_evidence_readme_refresh \
    --min_sample_count 6 \
    --min_qualified_count 6 \
    --require_objective_support \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_training_scale_expansion_decision() {
  local training_resource_run_id="${TRAINING_RESOURCE_RUN_ID:-harness_stage_b_midi_to_solo_training_resource_probe}"
  local sequence_budget_run_id="${SEQUENCE_BUDGET_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke}"
  local objective_path_run_id="${OBJECTIVE_PATH_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_training_scale_expansion_decision}"
  local training_resource="outputs/stage_b_midi_to_solo_training_resource_probe/${training_resource_run_id}/stage_b_midi_to_solo_training_resource_probe.json"
  local sequence_budget="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/${sequence_budget_run_id}/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json"
  local objective_path="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next/${objective_path_run_id}/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next.json"
  if [[ ! -f "$training_resource" ]]; then
    print_header "Stage B MIDI-to-solo training resource probe"
    RUN_ID="$training_resource_run_id" run_stage_b_midi_to_solo_training_resource_probe
  fi
  if [[ ! -f "$sequence_budget" ]]; then
    print_header "Stage B MIDI-to-solo model-direct sequence budget repair smoke"
    RUN_ID="$sequence_budget_run_id" run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke
  fi
  if [[ ! -f "$objective_path" ]]; then
    print_header "Stage B MIDI-to-solo repeatability objective-only next decision"
    RUN_ID="$objective_path_run_id" run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next
  fi
  print_header "Stage B MIDI-to-solo training scale expansion decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_training_scale_expansion.py \
    --run_id "$run_id" \
    --training_resource "$training_resource" \
    --sequence_budget "$sequence_budget" \
    --objective_path "$objective_path" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_TRAINING_SCALE_EXPANSION_DECISION_2026-06-04.md \
    --target_train_records 512 \
    --target_val_records 128 \
    --expected_boundary stage_b_midi_to_solo_training_scale_expansion_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_training_scale_smoke \
    --min_selected_train_records 512 \
    --min_selected_val_records 128 \
    --require_scale_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_training_scale_smoke() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_training_scale_expansion_decision}"
  local full_window_run_id="${FULL_WINDOW_RUN_ID:-harness_stage_b_generic_full_manifest_window_preparation}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_512_128_maxseq160}"
  local decision_report="outputs/stage_b_midi_to_solo_training_scale_expansion_decision/${decision_run_id}/stage_b_midi_to_solo_training_scale_expansion_decision.json"
  local full_window_preparation="outputs/stage_b_generic_full_manifest_window_preparation/${full_window_run_id}/stage_b_generic_full_manifest_window_preparation.json"
  local output_root="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke"
  local training_output_root="${output_root}/${run_id}/training_smoke"
  local training_smoke="${training_output_root}/${training_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo training scale expansion decision"
    RUN_ID="$decision_run_id" run_stage_b_midi_to_solo_training_scale_expansion_decision
  fi
  if [[ ! -f "$full_window_preparation" ]]; then
    print_header "Stage B generic full manifest window preparation"
    RUN_ID="$full_window_run_id" run_stage_b_generic_full_manifest_window_preparation
  fi
  if [[ ! -f "$training_smoke" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke raw run"
    "$PYTHON_BIN" scripts/run_stage_b_generic_base_training_scale_smoke.py \
      --run_id "$training_run_id" \
      --output_root "$training_output_root" \
      --full_window_preparation "$full_window_preparation" \
      --train_records 512 \
      --val_records 128 \
      --min_train_records 512 \
      --min_val_records 128 \
      --max_sequence 160 \
      --expected_boundary stage_b_generic_base_training_scale_smoke \
      --expected_next_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
      --require_training_scale_smoke_passed \
      --require_no_broad_quality_claim \
      --require_no_brad_style_claim
  fi
  print_header "Stage B MIDI-to-solo controlled training scale smoke"
  "$PYTHON_BIN" scripts/summarize_stage_b_midi_to_solo_controlled_training_scale_smoke.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --training_smoke "$training_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_TRAINING_SCALE_SMOKE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_training_scale_smoke \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe \
    --min_train_records 512 \
    --min_val_records 128 \
    --require_checkpoint \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe() {
  local controlled_training_run_id="${CONTROLLED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_512_128_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe}"
  local generation_probe_run_id="${GENERATION_PROBE_RUN_ID:-controlled_scale_checkpoint}"
  local output_root="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe"
  local controlled_training="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${controlled_training_run_id}/stage_b_midi_to_solo_controlled_training_scale_smoke.json"
  local training_scale_smoke="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${controlled_training_run_id}/training_smoke/${training_run_id}/stage_b_generic_base_training_scale_smoke.json"
  local generic_output_root="${output_root}/${run_id}/generic_generation_probe"
  local generation_probe="${generic_output_root}/${generation_probe_run_id}/stage_b_generic_base_scale_checkpoint_generation_probe.json"
  if [[ ! -f "$controlled_training" || ! -f "$training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke"
    RUN_ID="$controlled_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_training_scale_smoke
  fi
  if [[ "${FORCE_GENERATION_PROBE:-0}" == "1" || ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint raw generation probe"
    "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_generation_probe.py \
      --run_id "$generation_probe_run_id" \
      --output_root "$generic_output_root" \
      --training_scale_smoke "$training_scale_smoke" \
      --issue_number 554 \
      --max_sequence 160 \
      --num_samples 3 \
      --seed 43 \
      --max_simultaneous_notes 1 \
      --expected_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
      --require_probe_completed \
      --require_no_broad_quality_claim \
      --require_no_brad_style_claim
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint generation probe"
  "$PYTHON_BIN" scripts/summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe.py \
    --run_id "$run_id" \
    --controlled_training "$controlled_training" \
    --generation_probe "$generation_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_GENERATION_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe \
    --min_sample_count 1 \
    --require_generation_executable \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision() {
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision}"
  local generation_probe="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe/${generation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe.json"
  if [[ ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint generation probe"
    RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint repair decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_repair.py \
    --run_id "$run_id" \
    --generation_probe "$generation_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_REPAIR_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe \
    --require_repair_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision}"
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe}"
  local controlled_training_run_id="${CONTROLLED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_512_128_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision.json"
  local generation_probe="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe/${generation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${controlled_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint repair decision"
    RUN_ID="$decision_run_id" GENERATION_RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision
  fi
  if [[ ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint generation probe"
    RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke"
    RUN_ID="$controlled_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint density/collapse repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --baseline_generation_probe "$generation_probe" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DENSITY_COLLAPSE_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision \
    --require_target_supported \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint density/collapse repair probe"
    RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air remaining blocker decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker.py \
    --run_id "$run_id" \
    --density_collapse_repair "$repair_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REMAINING_BLOCKER_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe \
    --require_dead_air_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local controlled_training_run_id="${CONTROLLED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_512_128_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision.json"
  local baseline_repair="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${controlled_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$baseline_repair" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint density/collapse repair probe"
    RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe
  fi
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air remaining blocker decision"
    RUN_ID="$decision_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke"
    RUN_ID="$controlled_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --baseline_repair "$baseline_repair" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe \
    --require_target_qualified \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local controlled_training_run_id="${CONTROLLED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_512_128_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${controlled_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repair probe"
    RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" REPAIR_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke"
    RUN_ID="$controlled_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repair repeatability probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe.py \
    --run_id "$run_id" \
    --repair_report "$repair_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPAIR_REPEATABILITY_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision \
    --require_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision() {
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe/${repeatability_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe.json"
  if [[ ! -f "$repeatability_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repair repeatability probe"
    RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard.py \
    --run_id "$run_id" \
    --repeatability_report "$repeatability_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe \
    --require_guard_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe() {
  local guard_run_id="${GUARD_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local controlled_training_run_id="${CONTROLLED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_512_128_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe}"
  local guard_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision/${guard_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${controlled_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$guard_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard decision"
    RUN_ID="$guard_run_id" REPEATABILITY_RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke"
    RUN_ID="$controlled_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe.py \
    --run_id "$run_id" \
    --guard_decision_report "$guard_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation \
    --require_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation() {
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe}"
  local guard_run_id="${GUARD_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe/${repair_probe_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair probe"
    RUN_ID="$repair_probe_run_id" GUARD_RUN_ID="$guard_run_id" REPEATABILITY_RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair.py \
    --run_id "$run_id" \
    --repair_report "$repair_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_REPAIR_CONSOLIDATION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package \
    --min_sample_count 9 \
    --require_objective_support \
    --require_audio_review_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package() {
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe}"
  local guard_run_id="${GUARD_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package}"
  local consolidation_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation/${consolidation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation.json"
  if [[ ! -f "$consolidation_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair consolidation"
    RUN_ID="$consolidation_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" GUARD_RUN_ID="$guard_run_id" REPEATABILITY_RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard audio review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_package.py \
    --run_id "$run_id" \
    --consolidation_report "$consolidation_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_AUDIO_REVIEW_PACKAGE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review() {
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package}"
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe}"
  local guard_run_id="${GUARD_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review}"
  local audio_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package/${audio_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package.json"
  if [[ ! -f "$audio_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard audio review package"
    RUN_ID="$audio_run_id" CONSOLIDATION_RUN_ID="$consolidation_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" GUARD_RUN_ID="$guard_run_id" REPEATABILITY_RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard listening review"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review.py \
    --run_id "$run_id" \
    --audio_package "$audio_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_LISTENING_REVIEW_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_only_next_decision \
    --expected_file_count 3 \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next() {
  local listening_run_id="${LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review}"
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package}"
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation}"
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe}"
  local guard_run_id="${GUARD_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe}"
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision}"
  local baseline_run_id="${BASELINE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next}"
  local listening_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review/${listening_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review.json"
  local consolidation_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation/${consolidation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation.json"
  if [[ ! -f "$listening_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard listening review"
    RUN_ID="$listening_run_id" AUDIO_RUN_ID="$audio_run_id" CONSOLIDATION_RUN_ID="$consolidation_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" GUARD_RUN_ID="$guard_run_id" REPEATABILITY_RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review
  fi
  if [[ ! -f "$consolidation_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard repair consolidation"
    RUN_ID="$consolidation_run_id" REPAIR_PROBE_RUN_ID="$repair_probe_run_id" GUARD_RUN_ID="$guard_run_id" REPEATABILITY_RUN_ID="$repeatability_run_id" REPAIR_RUN_ID="$repair_run_id" DECISION_RUN_ID="$decision_run_id" BASELINE_RUN_ID="$baseline_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint dead-air repeatability temperature guard objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next.py \
    --run_id "$run_id" \
    --listening_review "$listening_report" \
    --consolidation_report "$consolidation_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_OBJECTIVE_NEXT_DECISION_2026-06-04.md \
    --expected_final_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_path_complete \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision \
    --min_sample_count 9 \
    --min_candidate_count 3 \
    --require_objective_support \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision() {
  local training_resource_run_id="${TRAINING_RESOURCE_RUN_ID:-harness_stage_b_midi_to_solo_training_resource_probe}"
  local current_training_run_id="${CURRENT_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_training_scale_smoke}"
  local objective_run_id="${OBJECTIVE_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision}"
  local training_resource="outputs/stage_b_midi_to_solo_training_resource_probe/${training_resource_run_id}/stage_b_midi_to_solo_training_resource_probe.json"
  local current_training="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/${current_training_run_id}/stage_b_midi_to_solo_controlled_training_scale_smoke.json"
  local objective_path="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next/${objective_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next.json"
  if [[ ! -f "$training_resource" ]]; then
    print_header "Stage B MIDI-to-solo training resource probe"
    RUN_ID="$training_resource_run_id" run_stage_b_midi_to_solo_training_resource_probe
  fi
  if [[ ! -f "$current_training" ]]; then
    print_header "Stage B MIDI-to-solo controlled training scale smoke"
    RUN_ID="$current_training_run_id" run_stage_b_midi_to_solo_controlled_training_scale_smoke
  fi
  if [[ ! -f "$objective_path" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint temperature guard objective-only next decision"
    RUN_ID="$objective_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale expansion decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion.py \
    --run_id "$run_id" \
    --training_resource "$training_resource" \
    --current_controlled_training "$current_training" \
    --objective_path "$objective_path" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_EXPANSION_DECISION_2026-06-04.md \
    --target_train_records 2048 \
    --target_val_records 512 \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke \
    --min_selected_train_records 2048 \
    --min_selected_val_records 512 \
    --require_scale_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision}"
  local full_window_run_id="${FULL_WINDOW_RUN_ID:-harness_stage_b_generic_full_manifest_window_preparation}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision.json"
  local full_window_preparation="outputs/stage_b_generic_full_manifest_window_preparation/${full_window_run_id}/stage_b_generic_full_manifest_window_preparation.json"
  local output_root="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke"
  local training_output_root="${output_root}/${run_id}/training_smoke"
  local training_smoke="${training_output_root}/${training_run_id}/stage_b_generic_base_training_scale_smoke.json"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale expansion decision"
    RUN_ID="$decision_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision
  fi
  if [[ ! -f "$full_window_preparation" ]]; then
    print_header "Stage B generic full manifest window preparation"
    RUN_ID="$full_window_run_id" run_stage_b_generic_full_manifest_window_preparation
  fi
  if [[ ! -f "$training_smoke" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale raw run"
    "$PYTHON_BIN" scripts/run_stage_b_generic_base_training_scale_smoke.py \
      --run_id "$training_run_id" \
      --output_root "$training_output_root" \
      --full_window_preparation "$full_window_preparation" \
      --train_records 2048 \
      --val_records 512 \
      --min_train_records 2048 \
      --min_val_records 512 \
      --max_sequence 160 \
      --seed 47 \
      --expected_boundary stage_b_generic_base_training_scale_smoke \
      --expected_next_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
      --require_training_scale_smoke_passed \
      --require_no_broad_quality_claim \
      --require_no_brad_style_claim
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
  "$PYTHON_BIN" scripts/summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --training_smoke "$training_smoke" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_SMOKE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe \
    --min_train_records 2048 \
    --min_val_records 512 \
    --require_checkpoint \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe() {
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe}"
  local generation_probe_run_id="${GENERATION_PROBE_RUN_ID:-controlled_training_scale_checkpoint}"
  local output_root="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe"
  local selected_training="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke.json"
  local training_scale_smoke="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/stage_b_generic_base_training_scale_smoke.json"
  local generic_output_root="${output_root}/${run_id}/generic_generation_probe"
  local generation_probe="${generic_output_root}/${generation_probe_run_id}/stage_b_generic_base_scale_checkpoint_generation_probe.json"
  if [[ ! -f "$selected_training" || ! -f "$training_scale_smoke" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  if [[ "${FORCE_GENERATION_PROBE:-0}" == "1" || ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale raw generation probe"
    "$PYTHON_BIN" scripts/run_stage_b_generic_base_scale_checkpoint_generation_probe.py \
      --run_id "$generation_probe_run_id" \
      --output_root "$generic_output_root" \
      --training_scale_smoke "$training_scale_smoke" \
      --issue_number 582 \
      --max_sequence 160 \
      --num_samples 3 \
      --seed 47 \
      --max_simultaneous_notes 1 \
      --expected_boundary stage_b_generic_base_scale_checkpoint_generation_probe \
      --require_probe_completed \
      --require_no_broad_quality_claim \
      --require_no_brad_style_claim
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale generation probe"
  "$PYTHON_BIN" scripts/summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe.py \
    --run_id "$run_id" \
    --selected_training "$selected_training" \
    --generation_probe "$generation_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_GENERATION_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe \
    --min_sample_count 3 \
    --require_generation_executable \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision() {
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision}"
  local generation_probe="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe/${generation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe.json"
  if [[ ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale generation probe"
    RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale repair decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair.py \
    --run_id "$run_id" \
    --generation_probe "$generation_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_REPAIR_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe \
    --require_repair_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision}"
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe}"
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision.json"
  local baseline_generation_probe="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe/${generation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale repair decision"
    RUN_ID="$decision_run_id" GENERATION_RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision
  fi
  if [[ ! -f "$baseline_generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale generation probe"
    RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --baseline_generation_probe "$baseline_generation_probe" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DENSITY_GRAMMAR_COLLAPSE_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe \
    --require_target_supported \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe}"
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repair probe"
    RUN_ID="$repair_run_id" SELECTED_TRAINING_RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repeatability probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe.py \
    --run_id "$run_id" \
    --repair_report "$repair_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DENSITY_GRAMMAR_COLLAPSE_REPEATABILITY_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_dead_air_remaining_blocker_decision \
    --require_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision() {
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision}"
  local repeatability_probe="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe/${repeatability_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe.json"
  if [[ ! -f "$repeatability_probe" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale density/grammar/collapse repeatability probe"
    RUN_ID="$repeatability_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air remaining blocker decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker.py \
    --run_id "$run_id" \
    --repeatability_probe "$repeatability_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DEAD_AIR_REMAINING_BLOCKER_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_dead_air_remaining_blocker_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe \
    --require_dead_air_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision}"
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air remaining blocker decision"
    RUN_ID="$decision_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe.py \
    --run_id "$run_id" \
    --decision_report "$decision_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DEAD_AIR_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe \
    --require_target_qualified \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe}"
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair probe"
    RUN_ID="$repair_run_id" SELECTED_TRAINING_RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair repeatability probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe.py \
    --run_id "$run_id" \
    --repair_report "$repair_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DEAD_AIR_REPAIR_REPEATABILITY_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision \
    --require_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision() {
  local repeatability_run_id="${REPEATABILITY_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision}"
  local repeatability_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe/${repeatability_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe.json"
  if [[ ! -f "$repeatability_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair repeatability probe"
    RUN_ID="$repeatability_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard.py \
    --run_id "$run_id" \
    --repeatability_report "$repeatability_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_DECISION_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe \
    --require_guard_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision}"
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard decision"
    RUN_ID="$decision_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe.py \
    --run_id "$run_id" \
    --guard_decision_report "$decision_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_REPAIR_PROBE_2026-06-04.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision \
    --require_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard repair probe"
    RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard follow-up decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup.py \
    --run_id "$run_id" \
    --temperature_guard_repair_report "$repair_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_DEAD_AIR_REPEATABILITY_TEMPERATURE_GUARD_FOLLOWUP_DECISION_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe \
    --require_repair_target \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision}"
  local selected_training_run_id="${SELECTED_TRAINING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke}"
  local training_run_id="${TRAINING_RUN_ID:-controlled_2048_512_maxseq160}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision/${decision_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision.json"
  local checkpoint_dir="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/${selected_training_run_id}/training_smoke/${training_run_id}/checkpoints"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repeatability temperature guard follow-up decision"
    RUN_ID="$decision_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision
  fi
  if [[ ! -f "${checkpoint_dir}/checkpoint_epoch1.pt" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
    RUN_ID="$selected_training_run_id" TRAINING_RUN_ID="$training_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe.py \
    --run_id "$run_id" \
    --followup_decision_report "$decision_report" \
    --checkpoint_dir "$checkpoint_dir" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_POSTPROCESS_REMOVAL_DEAD_AIR_REPAIR_PROBE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation \
    --require_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation() {
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation}"
  local repair_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe/${repair_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe.json"
  if [[ ! -f "$repair_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair probe"
    RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair.py \
    --run_id "$run_id" \
    --repair_probe_report "$repair_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_POSTPROCESS_REMOVAL_DEAD_AIR_REPAIR_CONSOLIDATION_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package \
    --min_sample_count 9 \
    --require_objective_support \
    --require_audio_review_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package() {
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package}"
  local consolidation_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation/${consolidation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation.json"
  if [[ ! -f "$consolidation_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation"
    RUN_ID="$consolidation_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair audio review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_package.py \
    --run_id "$run_id" \
    --consolidation_report "$consolidation_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_POSTPROCESS_REMOVAL_DEAD_AIR_REPAIR_AUDIO_REVIEW_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package \
    --expected_file_count 3 \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review() {
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package}"
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review}"
  local audio_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package/${audio_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package.json"
  if [[ ! -f "$audio_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair audio review package"
    RUN_ID="$audio_run_id" CONSOLIDATION_RUN_ID="$consolidation_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair listening review"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review.py \
    --run_id "$run_id" \
    --audio_package "$audio_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_POSTPROCESS_REMOVAL_DEAD_AIR_REPAIR_LISTENING_REVIEW_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review \
    --expected_next_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_only_next_decision \
    --expected_file_count 3 \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next() {
  local listening_run_id="${LISTENING_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review}"
  local audio_run_id="${AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package}"
  local consolidation_run_id="${CONSOLIDATION_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation}"
  local repair_run_id="${REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next}"
  local listening_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review/${listening_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review.json"
  local consolidation_report="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation/${consolidation_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation.json"
  if [[ ! -f "$listening_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair listening review"
    RUN_ID="$listening_run_id" AUDIO_RUN_ID="$audio_run_id" CONSOLIDATION_RUN_ID="$consolidation_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review
  fi
  if [[ ! -f "$consolidation_report" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation"
    RUN_ID="$consolidation_run_id" REPAIR_RUN_ID="$repair_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation
  fi
  print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next.py \
    --run_id "$run_id" \
    --listening_review "$listening_report" \
    --consolidation_report "$consolidation_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CONTROLLED_SCALE_CHECKPOINT_TRAINING_SCALE_POSTPROCESS_REMOVAL_DEAD_AIR_REPAIR_OBJECTIVE_NEXT_2026-06-05.md \
    --expected_final_boundary stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_path_complete \
    --expected_next_boundary stage_b_midi_to_solo_mvp_current_evidence_consolidation \
    --min_sample_count 9 \
    --min_candidate_count 3 \
    --require_objective_support \
    --require_pending_review \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_mvp_current_evidence_consolidation() {
  local contract_run_id="${CONTRACT_RUN_ID:-harness_stage_b_midi_to_solo_mvp_contract}"
  local context_run_id="${CONTEXT_RUN_ID:-harness_stage_b_midi_to_solo_context_extraction}"
  local resource_run_id="${RESOURCE_RUN_ID:-harness_stage_b_midi_to_solo_training_resource_probe}"
  local generation_run_id="${GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_conditioned_generation_probe}"
  local candidate_audio_run_id="${CANDIDATE_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_candidate_audio_render_package}"
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next}"
  local cli_objective_next_run_id="${CLI_OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision}"
  local model_conditioned_pitch_contour_objective_next_run_id="${MODEL_CONDITIONED_PITCH_CONTOUR_OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_next}"
  local model_conditioned_pitch_contour_changed_ratio_repair_objective_next_run_id="${MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REPAIR_OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_mvp_current_evidence_consolidation}"
  local contract_report="outputs/stage_b_midi_to_solo_mvp_contract/${contract_run_id}/stage_b_midi_to_solo_mvp_contract.json"
  local context_report="outputs/stage_b_midi_to_solo_context_extraction/${context_run_id}/stage_b_midi_to_solo_context_extraction.json"
  local resource_probe="outputs/stage_b_midi_to_solo_training_resource_probe/${resource_run_id}/stage_b_midi_to_solo_training_resource_probe.json"
  local generation_probe="outputs/stage_b_midi_to_solo_conditioned_generation_probe/${generation_run_id}/stage_b_midi_to_solo_conditioned_generation_probe.json"
  local candidate_audio="outputs/stage_b_midi_to_solo_candidate_audio_render_package/${candidate_audio_run_id}/stage_b_midi_to_solo_candidate_audio_render_package.json"
  local objective_next="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next/${objective_next_run_id}/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next.json"
  local cli_objective_next="outputs/stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision/${cli_objective_next_run_id}/stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision.json"
  local model_conditioned_pitch_contour_objective_next="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_only_next_decision/${model_conditioned_pitch_contour_objective_next_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_only_next_decision.json"
  local model_conditioned_pitch_contour_changed_ratio_repair_objective_next="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_only_next_decision/${model_conditioned_pitch_contour_changed_ratio_repair_objective_next_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_only_next_decision.json"
  if [[ ! -f "$contract_report" ]]; then
    print_header "Stage B MIDI-to-solo MVP input contract"
    RUN_ID="$contract_run_id" run_stage_b_midi_to_solo_mvp_contract
  fi
  if [[ ! -f "$context_report" ]]; then
    print_header "Stage B MIDI-to-solo context extraction MVP"
    RUN_ID="$context_run_id" run_stage_b_midi_to_solo_context_extraction
  fi
  if [[ ! -f "$resource_probe" ]]; then
    print_header "Stage B MIDI-to-solo training resource probe"
    RUN_ID="$resource_run_id" CONTEXT_RUN_ID="$context_run_id" run_stage_b_midi_to_solo_training_resource_probe
  fi
  if [[ ! -f "$generation_probe" ]]; then
    print_header "Stage B MIDI-to-solo conditioned generation probe"
    RUN_ID="$generation_run_id" CONTEXT_RUN_ID="$context_run_id" RESOURCE_RUN_ID="$resource_run_id" run_stage_b_midi_to_solo_conditioned_generation_probe
  fi
  if [[ ! -f "$candidate_audio" ]]; then
    print_header "Stage B MIDI-to-solo candidate audio render package"
    RUN_ID="$candidate_audio_run_id" GENERATION_RUN_ID="$generation_run_id" run_stage_b_midi_to_solo_candidate_audio_render_package
  fi
  if [[ ! -f "$objective_next" ]]; then
    print_header "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair objective-only next decision"
    RUN_ID="$objective_next_run_id" run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next
  fi
  if [[ ! -f "$cli_objective_next" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI objective-only next decision"
    RUN_ID="$cli_objective_next_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision
  fi
  if [[ ! -f "$model_conditioned_pitch_contour_objective_next" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour objective-only next decision"
    RUN_ID="$model_conditioned_pitch_contour_objective_next_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_next
  fi
  if [[ ! -f "$model_conditioned_pitch_contour_changed_ratio_repair_objective_next" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair objective-only next decision"
    RUN_ID="$model_conditioned_pitch_contour_changed_ratio_repair_objective_next_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next
  fi
  print_header "Stage B MIDI-to-solo MVP current evidence consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_mvp_current_evidence.py \
    --run_id "$run_id" \
    --contract_report "$contract_report" \
    --context_report "$context_report" \
    --resource_probe "$resource_probe" \
    --generation_probe "$generation_probe" \
    --audio_render "$candidate_audio" \
    --objective_next "$objective_next" \
    --cli_objective_next "$cli_objective_next" \
    --model_conditioned_pitch_contour_objective_next "$model_conditioned_pitch_contour_objective_next" \
    --model_conditioned_pitch_contour_changed_ratio_repair_objective_next "$model_conditioned_pitch_contour_changed_ratio_repair_objective_next" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MVP_CURRENT_EVIDENCE_CONSOLIDATION_2026-06-09.md \
    --issue_number 728 \
    --expected_boundary stage_b_midi_to_solo_mvp_current_evidence_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_readme_evidence_refresh \
    --min_exported_candidates 3 \
    --min_rendered_wav_files 3 \
    --min_objective_sample_count 9 \
    --require_current_evidence_support \
    --require_model_conditioned_pitch_contour_objective \
    --require_model_conditioned_pitch_contour_changed_ratio_repair_objective \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_mvp_completion_audit() {
  local current_evidence_run_id="${CURRENT_EVIDENCE_RUN_ID:-harness_stage_b_midi_to_solo_mvp_current_evidence_consolidation}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_mvp_completion_audit}"
  local current_evidence="outputs/stage_b_midi_to_solo_mvp_current_evidence_consolidation/${current_evidence_run_id}/stage_b_midi_to_solo_mvp_current_evidence_consolidation.json"
  if [[ ! -f "$current_evidence" ]]; then
    print_header "Stage B MIDI-to-solo MVP current evidence consolidation"
    RUN_ID="$current_evidence_run_id" run_stage_b_midi_to_solo_mvp_current_evidence_consolidation
  fi
  print_header "Stage B MIDI-to-solo MVP completion audit"
  "$PYTHON_BIN" scripts/audit_stage_b_midi_to_solo_mvp_completion.py \
    --run_id "$run_id" \
    --current_evidence "$current_evidence" \
    --readme_path README.md \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MVP_COMPLETION_AUDIT_2026-06-09.md \
    --issue_number 732 \
    --expected_boundary stage_b_midi_to_solo_mvp_completion_audit \
    --expected_next_boundary stage_b_midi_to_solo_quality_gap_decision \
    --require_technical_mvp_completion \
    --require_model_conditioned_pitch_contour_objective \
    --require_model_conditioned_pitch_contour_changed_ratio_repair_objective \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_quality_gap_decision() {
  local completion_audit_run_id="${COMPLETION_AUDIT_RUN_ID:-harness_stage_b_midi_to_solo_mvp_completion_audit}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_quality_gap_decision}"
  local completion_audit="outputs/stage_b_midi_to_solo_mvp_completion_audit/${completion_audit_run_id}/stage_b_midi_to_solo_mvp_completion_audit.json"
  if [[ ! -f "$completion_audit" ]]; then
    print_header "Stage B MIDI-to-solo MVP completion audit"
    RUN_ID="$completion_audit_run_id" run_stage_b_midi_to_solo_mvp_completion_audit
  fi
  print_header "Stage B MIDI-to-solo quality gap decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_quality_gap.py \
    --run_id "$run_id" \
    --mvp_completion_audit "$completion_audit" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_QUALITY_GAP_DECISION_2026-06-09.md \
    --issue_number 734 \
    --expected_boundary stage_b_midi_to_solo_quality_gap_decision \
    --expected_next_boundary stage_b_midi_to_solo_listening_review_quality_gap \
    --expected_target listening_review_quality_gap \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_listening_review_quality_gap() {
  local quality_gap_run_id="${QUALITY_GAP_RUN_ID:-harness_stage_b_midi_to_solo_quality_gap_decision}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_listening_review_quality_gap}"
  local quality_gap="outputs/stage_b_midi_to_solo_quality_gap_decision/${quality_gap_run_id}/stage_b_midi_to_solo_quality_gap_decision.json"
  if [[ ! -f "$quality_gap" ]]; then
    print_header "Stage B MIDI-to-solo quality gap decision"
    RUN_ID="$quality_gap_run_id" run_stage_b_midi_to_solo_quality_gap_decision
  fi
  print_header "Stage B MIDI-to-solo listening review quality gap"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_listening_review_quality_gap.py \
    --run_id "$run_id" \
    --quality_gap_decision "$quality_gap" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_LISTENING_REVIEW_QUALITY_GAP_2026-06-09.md \
    --issue_number 736 \
    --expected_boundary stage_b_midi_to_solo_listening_review_quality_gap \
    --expected_next_boundary stage_b_midi_to_solo_mvp_delivery_package \
    --expected_target mvp_delivery_package \
    --require_delivery_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_mvp_delivery_package() {
  local listening_gap_run_id="${LISTENING_GAP_RUN_ID:-harness_stage_b_midi_to_solo_listening_review_quality_gap}"
  local cli_package_run_id="${CLI_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_phrase_bank_cli_mvp_package}"
  local changed_ratio_audio_run_id="${CHANGED_RATIO_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_mvp_delivery_package}"
  local listening_gap="outputs/stage_b_midi_to_solo_listening_review_quality_gap/${listening_gap_run_id}/stage_b_midi_to_solo_listening_review_quality_gap.json"
  local cli_package="outputs/stage_b_midi_to_solo_phrase_bank_cli_mvp_package/${cli_package_run_id}/stage_b_midi_to_solo_phrase_bank_cli_mvp_package.json"
  local changed_ratio_audio="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package/${changed_ratio_audio_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package.json"
  if [[ ! -f "$listening_gap" ]]; then
    print_header "Stage B MIDI-to-solo listening review quality gap"
    RUN_ID="$listening_gap_run_id" run_stage_b_midi_to_solo_listening_review_quality_gap
  fi
  if [[ ! -f "$cli_package" ]]; then
    print_header "Stage B MIDI-to-solo phrase-bank CLI MVP package"
    RUN_ID="$cli_package_run_id" run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package
  fi
  if [[ ! -f "$changed_ratio_audio" ]]; then
    print_header "Stage B MIDI-to-solo changed-ratio repair audio package"
    RUN_ID="$changed_ratio_audio_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package
  fi
  print_header "Stage B MIDI-to-solo MVP delivery package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_mvp_delivery_package.py \
    --run_id "$run_id" \
    --listening_review_quality_gap "$listening_gap" \
    --cli_mvp_package "$cli_package" \
    --changed_ratio_audio_package "$changed_ratio_audio" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MVP_DELIVERY_PACKAGE_2026-06-09.md \
    --issue_number 738 \
    --expected_boundary stage_b_midi_to_solo_mvp_delivery_package \
    --expected_next_boundary stage_b_midi_to_solo_readme_final_evidence_refresh \
    --require_delivery_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_final_status_audit() {
  local delivery_package_run_id="${DELIVERY_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_mvp_delivery_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_final_status_audit}"
  local delivery_package="outputs/stage_b_midi_to_solo_mvp_delivery_package/${delivery_package_run_id}/stage_b_midi_to_solo_mvp_delivery_package.json"
  if [[ ! -f "$delivery_package" ]]; then
    print_header "Stage B MIDI-to-solo MVP delivery package"
    RUN_ID="$delivery_package_run_id" run_stage_b_midi_to_solo_mvp_delivery_package
  fi
  print_header "Stage B MIDI-to-solo final status audit"
  "$PYTHON_BIN" scripts/audit_stage_b_midi_to_solo_final_status.py \
    --run_id "$run_id" \
    --delivery_package "$delivery_package" \
    --readme_path README.md \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_FINAL_STATUS_AUDIT_2026-06-09.md \
    --issue_number 742 \
    --expected_boundary stage_b_midi_to_solo_final_status_audit \
    --expected_next_boundary stage_b_midi_to_solo_post_mvp_quality_iteration_plan \
    --require_technical_mvp_complete \
    --require_readme_reflected \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_post_mvp_quality_iteration_plan() {
  local final_status_run_id="${FINAL_STATUS_RUN_ID:-harness_stage_b_midi_to_solo_final_status_audit}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_post_mvp_quality_iteration_plan}"
  local final_status="outputs/stage_b_midi_to_solo_final_status_audit/${final_status_run_id}/stage_b_midi_to_solo_final_status_audit.json"
  if [[ ! -f "$final_status" ]]; then
    print_header "Stage B MIDI-to-solo final status audit"
    RUN_ID="$final_status_run_id" run_stage_b_midi_to_solo_final_status_audit
  fi
  print_header "Stage B MIDI-to-solo post-MVP quality iteration plan"
  "$PYTHON_BIN" scripts/plan_stage_b_midi_to_solo_post_mvp_quality_iteration.py \
    --run_id "$run_id" \
    --final_status_audit "$final_status" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_POST_MVP_QUALITY_ITERATION_PLAN_2026-06-09.md \
    --issue_number 744 \
    --expected_boundary stage_b_midi_to_solo_post_mvp_quality_iteration_plan \
    --expected_next_boundary stage_b_midi_to_solo_quality_rubric_baseline \
    --expected_target quality_rubric_baseline \
    --require_quality_rubric \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_quality_rubric_baseline() {
  local post_mvp_plan_run_id="${POST_MVP_PLAN_RUN_ID:-harness_stage_b_midi_to_solo_post_mvp_quality_iteration_plan}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_quality_rubric_baseline}"
  local post_mvp_plan="outputs/stage_b_midi_to_solo_post_mvp_quality_iteration_plan/${post_mvp_plan_run_id}/stage_b_midi_to_solo_post_mvp_quality_iteration_plan.json"
  if [[ ! -f "$post_mvp_plan" ]]; then
    print_header "Stage B MIDI-to-solo post-MVP quality iteration plan"
    RUN_ID="$post_mvp_plan_run_id" run_stage_b_midi_to_solo_post_mvp_quality_iteration_plan
  fi
  print_header "Stage B MIDI-to-solo quality rubric baseline"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_quality_rubric_baseline.py \
    --run_id "$run_id" \
    --post_mvp_quality_plan "$post_mvp_plan" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_QUALITY_RUBRIC_BASELINE_2026-06-09.md \
    --issue_number 746 \
    --expected_boundary stage_b_midi_to_solo_quality_rubric_baseline \
    --expected_next_boundary stage_b_midi_to_solo_candidate_failure_labeling \
    --expected_target candidate_failure_labeling \
    --min_rubric_item_count 8 \
    --require_candidate_labeling_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_candidate_failure_labeling() {
  local rubric_run_id="${RUBRIC_RUN_ID:-harness_stage_b_midi_to_solo_quality_rubric_baseline}"
  local delivery_run_id="${DELIVERY_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_mvp_delivery_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_candidate_failure_labeling}"
  local rubric="outputs/stage_b_midi_to_solo_quality_rubric_baseline/${rubric_run_id}/stage_b_midi_to_solo_quality_rubric_baseline.json"
  local delivery="outputs/stage_b_midi_to_solo_mvp_delivery_package/${delivery_run_id}/stage_b_midi_to_solo_mvp_delivery_package.json"
  if [[ ! -f "$rubric" ]]; then
    print_header "Stage B MIDI-to-solo quality rubric baseline"
    RUN_ID="$rubric_run_id" run_stage_b_midi_to_solo_quality_rubric_baseline
  fi
  if [[ ! -f "$delivery" ]]; then
    print_header "Stage B MIDI-to-solo MVP delivery package"
    RUN_ID="$delivery_run_id" run_stage_b_midi_to_solo_mvp_delivery_package
  fi
  print_header "Stage B MIDI-to-solo candidate failure labeling"
  "$PYTHON_BIN" scripts/label_stage_b_midi_to_solo_candidate_failures.py \
    --run_id "$run_id" \
    --rubric_baseline "$rubric" \
    --mvp_delivery_package "$delivery" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_CANDIDATE_FAILURE_LABELING_2026-06-09.md \
    --issue_number 748 \
    --expected_boundary stage_b_midi_to_solo_candidate_failure_labeling \
    --min_candidate_count 6 \
    --require_labeling_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_targeted_quality_repair_sweep() {
  local labeling_run_id="${LABELING_RUN_ID:-harness_stage_b_midi_to_solo_candidate_failure_labeling}"
  local rubric_run_id="${RUBRIC_RUN_ID:-harness_stage_b_midi_to_solo_quality_rubric_baseline}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_targeted_quality_repair_sweep}"
  local labeling="outputs/stage_b_midi_to_solo_candidate_failure_labeling/${labeling_run_id}/stage_b_midi_to_solo_candidate_failure_labeling.json"
  local rubric="outputs/stage_b_midi_to_solo_quality_rubric_baseline/${rubric_run_id}/stage_b_midi_to_solo_quality_rubric_baseline.json"
  if [[ ! -f "$labeling" ]]; then
    print_header "Stage B MIDI-to-solo candidate failure labeling"
    RUN_ID="$labeling_run_id" run_stage_b_midi_to_solo_candidate_failure_labeling
  fi
  if [[ ! -f "$rubric" ]]; then
    print_header "Stage B MIDI-to-solo quality rubric baseline"
    RUN_ID="$rubric_run_id" run_stage_b_midi_to_solo_quality_rubric_baseline
  fi
  print_header "Stage B MIDI-to-solo targeted quality repair sweep"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_targeted_quality_repair_sweep.py \
    --run_id "$run_id" \
    --candidate_failure_labeling "$labeling" \
    --rubric_baseline "$rubric" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_TARGETED_QUALITY_REPAIR_SWEEP_2026-06-09.md \
    --issue_number 750 \
    --expected_boundary stage_b_midi_to_solo_targeted_quality_repair_sweep \
    --min_candidate_count 6 \
    --require_sweep_completed \
    --require_failure_delta \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_targeted_quality_repair_audio_package() {
  local repair_sweep_run_id="${REPAIR_SWEEP_RUN_ID:-harness_stage_b_midi_to_solo_targeted_quality_repair_sweep}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_targeted_quality_repair_audio_package}"
  local repair_sweep_report="outputs/stage_b_midi_to_solo_targeted_quality_repair_sweep/${repair_sweep_run_id}/stage_b_midi_to_solo_targeted_quality_repair_sweep.json"
  if [[ ! -f "$repair_sweep_report" ]]; then
    print_header "Stage B MIDI-to-solo targeted quality repair sweep"
    RUN_ID="$repair_sweep_run_id" run_stage_b_midi_to_solo_targeted_quality_repair_sweep
  fi
  print_header "Stage B MIDI-to-solo targeted quality repair audio package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_targeted_quality_repair_audio.py \
    --run_id "$run_id" \
    --targeted_quality_repair_sweep_report "$repair_sweep_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_TARGETED_QUALITY_REPAIR_AUDIO_PACKAGE_2026-06-09.md \
    --issue_number 752 \
    --expected_boundary stage_b_midi_to_solo_targeted_quality_repair_audio_package \
    --expected_next_boundary stage_b_midi_to_solo_targeted_quality_repair_listening_review_package \
    --expected_file_count 6 \
    --sample_rate 44100 \
    --require_audio_package_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision() {
  local quality_gap_run_id="${QUALITY_GAP_RUN_ID:-harness_stage_b_midi_to_solo_quality_gap_decision}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision}"
  local quality_gap="outputs/stage_b_midi_to_solo_quality_gap_decision/${quality_gap_run_id}/stage_b_midi_to_solo_quality_gap_decision.json"
  if [[ ! -f "$quality_gap" ]]; then
    print_header "Stage B MIDI-to-solo quality gap decision"
    RUN_ID="$quality_gap_run_id" run_stage_b_midi_to_solo_quality_gap_decision
  fi
  print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio review decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review.py \
    --run_id "$run_id" \
    --quality_gap_decision "$quality_gap" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REVIEW_DECISION_2026-06-09.md \
    --issue_number 716 \
    --changed_ratio_review_threshold 0.5 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe \
    --expected_target lower_pitch_change_ratio_repair_probe \
    --require_repair_probe \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe() {
  local changed_ratio_decision_run_id="${CHANGED_RATIO_DECISION_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision}"
  local pitch_contour_probe_run_id="${PITCH_CONTOUR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe}"
  local changed_ratio_decision="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision/${changed_ratio_decision_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision.json"
  local pitch_contour_probe="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/${pitch_contour_probe_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe.json"
  if [[ ! -f "$changed_ratio_decision" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio review decision"
    RUN_ID="$changed_ratio_decision_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision
  fi
  if [[ ! -f "$pitch_contour_probe" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour probe"
    RUN_ID="$pitch_contour_probe_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe
  fi
  print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe.py \
    --run_id "$run_id" \
    --changed_ratio_decision "$changed_ratio_decision" \
    --pitch_contour_probe "$pitch_contour_probe" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REPAIR_PROBE_2026-06-09.md \
    --issue_number 718 \
    --min_repaired_candidates 3 \
    --dead_air_threshold_seconds 0.5 \
    --preferred_pitch_min 48 \
    --preferred_pitch_max 88 \
    --max_adjacent_interval 12 \
    --max_pitch_changed_ratio 0.5 \
    --min_unique_pitch_count 20 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package \
    --require_repair_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package() {
  local changed_ratio_repair_run_id="${CHANGED_RATIO_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package}"
  local changed_ratio_repair_probe_report="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe/${changed_ratio_repair_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe.json"
  if [[ ! -f "$changed_ratio_repair_probe_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair probe"
    RUN_ID="$changed_ratio_repair_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe
  fi
  print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair audio package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio.py \
    --run_id "$run_id" \
    --changed_ratio_repair_probe_report "$changed_ratio_repair_probe_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REPAIR_AUDIO_PACKAGE_2026-06-09.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package \
    --expected_file_count 3 \
    --sample_rate 44100 \
    --require_audio_package_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package() {
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package}"
  local package_run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package}"
  local audio_package_report="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package/${audio_package_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package.json"
  if [[ ! -f "$audio_package_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair audio package"
    RUN_ID="$audio_package_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package.py \
    --run_id "$package_run_id" \
    --audio_package_report "$audio_package_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REPAIR_LISTENING_REVIEW_PACKAGE_2026-06-09.md \
    --issue_number 722 \
    --expected_review_item_count 3 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard \
    --require_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard}"
  local source_package="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package/${package_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package.json"
  if [[ ! -f "$source_package" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair listening review package"
    RUN_ID="$package_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input.py \
    --run_id "$run_id" \
    --source_package "$source_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REPAIR_LISTENING_REVIEW_INPUT_GUARD_2026-06-09.md \
    --issue_number 724 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_only_next_decision \
    --require_guard_completed \
    --require_preference_blocked \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next() {
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next}"
  local input_guard_report="outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard.json"
  if [[ ! -f "$input_guard_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair listening review input guard"
    RUN_ID="$input_guard_run_id" run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard
  fi
  print_header "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next.py \
    --run_id "$run_id" \
    --input_guard_report "$input_guard_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_PITCH_CONTOUR_CHANGED_RATIO_REPAIR_OBJECTIVE_NEXT_DECISION_2026-06-09.md \
    --issue_number 726 \
    --max_interval_threshold 12 \
    --max_pitch_changed_ratio 0.5 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_mvp_current_evidence_consolidation \
    --require_objective_support \
    --require_current_evidence_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment() {
  local quality_gap_run_id="${QUALITY_GAP_RUN_ID:-harness_stage_b_midi_to_solo_quality_gap_decision}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment}"
  local quality_gap="outputs/stage_b_midi_to_solo_quality_gap_decision/${quality_gap_run_id}/stage_b_midi_to_solo_quality_gap_decision.json"
  if [[ ! -f "$quality_gap" ]]; then
    print_header "Stage B MIDI-to-solo quality gap decision"
    RUN_ID="$quality_gap_run_id" run_stage_b_midi_to_solo_quality_gap_decision
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path quality alignment"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment.py \
    --run_id "$run_id" \
    --quality_gap_decision "$quality_gap" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_QUALITY_ALIGNMENT_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_probe \
    --expected_probe_target replace_fallback_with_model_conditioned_input_path_probe \
    --require_probe_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_probe() {
  local alignment_run_id="${ALIGNMENT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment}"
  local quality_gap_run_id="${QUALITY_GAP_RUN_ID:-harness_stage_b_midi_to_solo_quality_gap_decision}"
  local fallback_generation_run_id="${FALLBACK_GENERATION_RUN_ID:-harness_stage_b_midi_to_solo_conditioned_generation_probe}"
  local fallback_audio_run_id="${FALLBACK_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_candidate_audio_render_package}"
  local model_direct_repair_run_id="${MODEL_DIRECT_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local model_direct_audio_run_id="${MODEL_DIRECT_AUDIO_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_probe}"
  local alignment="outputs/stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment/${alignment_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment.json"
  local fallback_generation="outputs/stage_b_midi_to_solo_conditioned_generation_probe/${fallback_generation_run_id}/stage_b_midi_to_solo_conditioned_generation_probe.json"
  local fallback_audio="outputs/stage_b_midi_to_solo_candidate_audio_render_package/${fallback_audio_run_id}/stage_b_midi_to_solo_candidate_audio_render_package.json"
  local model_direct_repair="outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/${model_direct_repair_run_id}/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.json"
  local model_direct_audio="outputs/stage_b_midi_to_solo_model_direct_audio_render_package/${model_direct_audio_run_id}/stage_b_midi_to_solo_model_direct_audio_render_package.json"
  if [[ ! -f "$alignment" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path quality alignment"
    RUN_ID="$alignment_run_id" QUALITY_GAP_RUN_ID="$quality_gap_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment
  fi
  if [[ ! -f "$fallback_generation" ]]; then
    print_header "Stage B MIDI-to-solo conditioned generation probe"
    RUN_ID="$fallback_generation_run_id" run_stage_b_midi_to_solo_conditioned_generation_probe
  fi
  if [[ ! -f "$fallback_audio" ]]; then
    print_header "Stage B MIDI-to-solo candidate audio render package"
    RUN_ID="$fallback_audio_run_id" GENERATION_RUN_ID="$fallback_generation_run_id" run_stage_b_midi_to_solo_candidate_audio_render_package
  fi
  if [[ ! -f "$model_direct_repair" ]]; then
    print_header "Stage B MIDI-to-solo model-direct monophonic overlap repair"
    RUN_ID="$model_direct_repair_run_id" run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair
  fi
  if [[ ! -f "$model_direct_audio" ]]; then
    print_header "Stage B MIDI-to-solo model-direct audio render package"
    RUN_ID="$model_direct_audio_run_id" OVERLAP_REPAIR_RUN_ID="$model_direct_repair_run_id" run_stage_b_midi_to_solo_model_direct_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path probe"
  "$PYTHON_BIN" scripts/probe_stage_b_midi_to_solo_model_conditioned_input_path.py \
    --run_id "$run_id" \
    --alignment_report "$alignment" \
    --fallback_generation_report "$fallback_generation" \
    --fallback_audio_report "$fallback_audio" \
    --model_direct_repair_report "$model_direct_repair" \
    --model_direct_audio_report "$model_direct_audio" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_PROBE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_probe \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_candidate_export \
    --require_model_conditioned_evidence \
    --require_candidate_export \
    --require_replacement_not_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export() {
  local probe_run_id="${PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_probe}"
  local model_direct_repair_run_id="${MODEL_DIRECT_REPAIR_RUN_ID:-harness_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export}"
  local probe_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_probe/${probe_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_probe.json"
  local model_direct_repair="outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/${model_direct_repair_run_id}/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair.json"
  local model_direct_generation="outputs/stage_b_midi_to_solo_model_direct_monophonic_overlap_repair/${model_direct_repair_run_id}/generation_probe/monophonic_overlap_repair/report.json"
  if [[ ! -f "$probe_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path probe"
    RUN_ID="$probe_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_probe
  fi
  if [[ ! -f "$model_direct_repair" || ! -f "$model_direct_generation" ]]; then
    print_header "Stage B MIDI-to-solo model-direct monophonic overlap repair"
    RUN_ID="$model_direct_repair_run_id" run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path candidate export"
  "$PYTHON_BIN" scripts/export_stage_b_midi_to_solo_model_conditioned_input_path_candidates.py \
    --run_id "$run_id" \
    --probe_report "$probe_report" \
    --model_direct_repair_report "$model_direct_repair" \
    --model_direct_generation_report "$model_direct_generation" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_CANDIDATE_EXPORT_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_candidate_export \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package \
    --require_ranked_export_contract \
    --require_audio_render_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package() {
  local candidate_export_run_id="${CANDIDATE_EXPORT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package}"
  local candidate_export="outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/${candidate_export_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export.json"
  if [[ ! -f "$candidate_export" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path candidate export"
    RUN_ID="$candidate_export_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path audio render package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_model_conditioned_input_path_audio.py \
    --run_id "$run_id" \
    --candidate_export_report "$candidate_export" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_AUDIO_RENDER_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation \
    --expected_file_count 3 \
    --require_replacement_technical_path \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation() {
  local candidate_export_run_id="${CANDIDATE_EXPORT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation}"
  local candidate_export="outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/${candidate_export_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export.json"
  local audio_render="outputs/stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package/${audio_render_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package.json"
  if [[ ! -f "$candidate_export" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path candidate export"
    RUN_ID="$candidate_export_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export
  fi
  if [[ ! -f "$audio_render" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path audio render package"
    RUN_ID="$audio_render_run_id" CANDIDATE_EXPORT_RUN_ID="$candidate_export_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path replacement consolidation"
  "$PYTHON_BIN" scripts/consolidate_stage_b_midi_to_solo_model_conditioned_input_path_replacement.py \
    --run_id "$run_id" \
    --candidate_export_report "$candidate_export" \
    --audio_render_report "$audio_render" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_REPLACEMENT_CONSOLIDATION_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package \
    --require_technical_replacement \
    --require_listening_review_package \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package() {
  local replacement_run_id="${REPLACEMENT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package}"
  local replacement="outputs/stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation/${replacement_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation.json"
  local audio_render="outputs/stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package/${audio_render_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package.json"
  if [[ ! -f "$replacement" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path replacement consolidation"
    RUN_ID="$replacement_run_id" AUDIO_RENDER_RUN_ID="$audio_render_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation
  fi
  if [[ ! -f "$audio_render" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path audio render package"
    RUN_ID="$audio_render_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package.py \
    --run_id "$run_id" \
    --replacement_report "$replacement" \
    --audio_render_report "$audio_render" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_LISTENING_REVIEW_PACKAGE_2026-06-05.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard \
    --expected_review_item_count 3 \
    --require_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard}"
  local source_package="outputs/stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package/${package_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package.json"
  if [[ ! -f "$source_package" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path listening review package"
    RUN_ID="$package_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input.py \
    --run_id "$run_id" \
    --source_package "$source_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_LISTENING_REVIEW_INPUT_GUARD_2026-06-08.md \
    --issue_number 684 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision \
    --require_guard_completed \
    --require_pending_input \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_objective_next() {
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard}"
  local candidate_export_run_id="${CANDIDATE_EXPORT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export}"
  local audio_render_run_id="${AUDIO_RENDER_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision}"
  local input_guard="outputs/stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard.json"
  local candidate_export="outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/${candidate_export_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export.json"
  local audio_render="outputs/stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package/${audio_render_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package.json"
  if [[ ! -f "$input_guard" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path listening review input guard"
    RUN_ID="$input_guard_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard
  fi
  if [[ ! -f "$candidate_export" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path candidate export"
    RUN_ID="$candidate_export_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export
  fi
  if [[ ! -f "$audio_render" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path audio render package"
    RUN_ID="$audio_render_run_id" CANDIDATE_EXPORT_RUN_ID="$candidate_export_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_input_path_objective_next.py \
    --run_id "$run_id" \
    --input_guard_report "$input_guard" \
    --candidate_export_report "$candidate_export" \
    --audio_render_report "$audio_render" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_OBJECTIVE_NEXT_DECISION_2026-06-08.md \
    --issue_number 686 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision \
    --require_objective_decision \
    --require_repair_required \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision() {
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision}"
  local objective_next="outputs/stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision/${objective_next_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_objective_only_next_decision.json"
  if [[ ! -f "$objective_next" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path objective-only next decision"
    RUN_ID="$objective_next_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_objective_next
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision.py \
    --run_id "$run_id" \
    --objective_next_report "$objective_next" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_DECISION_2026-06-08.md \
    --issue_number 688 \
    --target_dead_air_max 0.35 \
    --max_postprocess_removal_ratio 0.25 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe \
    --require_decision_completed \
    --require_repair_probe \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe() {
  local decision_run_id="${DECISION_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision}"
  local candidate_export_run_id="${CANDIDATE_EXPORT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe}"
  local decision_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision/${decision_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision.json"
  local candidate_export="outputs/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export/${candidate_export_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_candidate_export.json"
  if [[ ! -f "$decision_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair decision"
    RUN_ID="$decision_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision
  fi
  if [[ ! -f "$candidate_export" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path candidate export"
    RUN_ID="$candidate_export_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe.py \
    --run_id "$run_id" \
    --repair_decision_report "$decision_report" \
    --candidate_export_report "$candidate_export" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PROBE_2026-06-08.md \
    --issue_number 690 \
    --min_repaired_candidates 3 \
    --dead_air_threshold_seconds 0.5 \
    --max_start_gap_seconds 0.49 \
    --fill_note_duration_seconds 0.18 \
    --min_note_duration_seconds 0.04 \
    --preferred_pitch_min 48 \
    --preferred_pitch_max 88 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package \
    --require_repair_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package() {
  local repair_probe_run_id="${REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package}"
  local repair_probe_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/${repair_probe_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe.json"
  if [[ ! -f "$repair_probe_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair probe"
    RUN_ID="$repair_probe_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair audio package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio.py \
    --run_id "$run_id" \
    --repair_probe_report "$repair_probe_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_AUDIO_PACKAGE_2026-06-08.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision \
    --expected_file_count 3 \
    --sample_rate 44100 \
    --require_audio_package_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next() {
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision}"
  local audio_package_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package/${audio_package_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package.json"
  if [[ ! -f "$audio_package_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair audio package"
    RUN_ID="$audio_package_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair objective next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next.py \
    --run_id "$run_id" \
    --audio_package_report "$audio_package_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_OBJECTIVE_NEXT_DECISION_2026-06-08.md \
    --issue_number 694 \
    --expected_count 3 \
    --max_interval_threshold 12 \
    --max_added_note_ratio_review_threshold 0.75 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision \
    --require_objective_decision \
    --require_wide_interval_followup \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision() {
  local objective_next_run_id="${OBJECTIVE_NEXT_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision}"
  local objective_next_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision/${objective_next_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision.json"
  if [[ ! -f "$objective_next_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair objective next decision"
    RUN_ID="$objective_next_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour.py \
    --run_id "$run_id" \
    --objective_next_report "$objective_next_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PITCH_CONTOUR_DECISION_2026-06-09.md \
    --issue_number 696 \
    --target_max_interval 12 \
    --target_dead_air_max 0.35 \
    --min_repaired_candidate_count 3 \
    --max_simultaneous_notes 1 \
    --max_added_note_ratio_review_threshold 0.75 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe \
    --require_pitch_contour_decision \
    --require_repair_probe \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe() {
  local pitch_contour_decision_run_id="${PITCH_CONTOUR_DECISION_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision}"
  local dead_air_repair_probe_run_id="${DEAD_AIR_REPAIR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe}"
  local pitch_contour_decision_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision/${pitch_contour_decision_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision.json"
  local dead_air_repair_probe_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe/${dead_air_repair_probe_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe.json"
  if [[ ! -f "$pitch_contour_decision_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour decision"
    RUN_ID="$pitch_contour_decision_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision
  fi
  if [[ ! -f "$dead_air_repair_probe_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair probe"
    RUN_ID="$dead_air_repair_probe_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour probe"
  "$PYTHON_BIN" scripts/run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe.py \
    --run_id "$run_id" \
    --pitch_contour_decision_report "$pitch_contour_decision_report" \
    --dead_air_repair_probe_report "$dead_air_repair_probe_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PITCH_CONTOUR_PROBE_2026-06-09.md \
    --issue_number 698 \
    --min_repaired_candidates 3 \
    --dead_air_threshold_seconds 0.5 \
    --preferred_pitch_min 48 \
    --preferred_pitch_max 88 \
    --max_adjacent_interval 12 \
    --min_unique_pitch_count 8 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package \
    --require_repair_passed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package() {
  local pitch_contour_probe_run_id="${PITCH_CONTOUR_PROBE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package}"
  local pitch_contour_probe_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe/${pitch_contour_probe_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe.json"
  if [[ ! -f "$pitch_contour_probe_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour probe"
    RUN_ID="$pitch_contour_probe_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour audio package"
  "$PYTHON_BIN" scripts/render_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio.py \
    --run_id "$run_id" \
    --pitch_contour_probe_report "$pitch_contour_probe_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PITCH_CONTOUR_AUDIO_PACKAGE_2026-06-09.md \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package \
    --expected_file_count 3 \
    --sample_rate 44100 \
    --require_audio_package_completed \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package() {
  local audio_package_run_id="${AUDIO_PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package}"
  local audio_package_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package/${audio_package_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package.json"
  if [[ ! -f "$audio_package_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour audio package"
    RUN_ID="$audio_package_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review package"
  "$PYTHON_BIN" scripts/build_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package.py \
    --run_id "$run_id" \
    --audio_package_report "$audio_package_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PITCH_CONTOUR_LISTENING_REVIEW_PACKAGE_2026-06-09.md \
    --issue_number 702 \
    --expected_review_item_count 3 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard \
    --require_package_ready \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard() {
  local package_run_id="${PACKAGE_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard}"
  local source_package="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package/${package_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package.json"
  if [[ ! -f "$source_package" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review package"
    RUN_ID="$package_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review input guard"
  "$PYTHON_BIN" scripts/guard_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input.py \
    --run_id "$run_id" \
    --source_package "$source_package" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PITCH_CONTOUR_LISTENING_REVIEW_INPUT_GUARD_2026-06-09.md \
    --issue_number 704 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard \
    --expected_next_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_only_next_decision \
    --require_guard_completed \
    --require_preference_blocked \
    --require_no_quality_claim
}

run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_next() {
  local input_guard_run_id="${INPUT_GUARD_RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard}"
  local run_id="${RUN_ID:-harness_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_next}"
  local input_guard_report="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard/${input_guard_run_id}/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard.json"
  if [[ ! -f "$input_guard_report" ]]; then
    print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour listening review input guard"
    RUN_ID="$input_guard_run_id" run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard
  fi
  print_header "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour objective-only next decision"
  "$PYTHON_BIN" scripts/decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_next.py \
    --run_id "$run_id" \
    --input_guard_report "$input_guard_report" \
    --doc_path docs/STAGE_B_MIDI_TO_SOLO_MODEL_CONDITIONED_INPUT_PATH_DEAD_AIR_TIMING_REPAIR_PITCH_CONTOUR_OBJECTIVE_NEXT_DECISION_2026-06-09.md \
    --issue_number 706 \
    --max_interval_threshold 12 \
    --pitch_changed_ratio_review_threshold 0.5 \
    --expected_boundary stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_only_next_decision \
    --expected_next_boundary stage_b_midi_to_solo_mvp_current_evidence_consolidation \
    --require_objective_decision \
    --require_current_evidence_ready \
    --require_no_quality_claim
}

run_stage_b_constrained_probe() {
  local run_id="${RUN_ID:-harness_stage_b_constrained_probe}"
  print_header "Stage B constrained probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --max_files 1 \
    --epochs 1 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 1 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --top_k 1 \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_overlap_gate() {
  local run_id="${RUN_ID:-harness_stage_b_overlap_gate}"
  print_header "Stage B overlap gate"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --max_files 1 \
    --epochs 1 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 1 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --top_k 1 \
    --require_note_groups \
    --require_valid_sample \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_stronger_probe() {
  local run_id="${RUN_ID:-harness_stage_b_stronger_probe}"
  print_header "Stage B stronger multi-sample probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 24 \
    --max_files 1 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --top_k 2 \
    --require_note_groups \
    --require_all_grammar_samples \
    --require_valid_sample \
    --min_valid_samples 1 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_collapse_sweep() {
  local run_id="${RUN_ID:-harness_stage_b_collapse_sweep}"
  print_header "Stage B collapse sampling sweep"
  "$PYTHON_BIN" scripts/run_stage_b_sampling_sweep.py \
    --run_id "$run_id" \
    --issue_number 31 \
    --top_ks 1,2 \
    --temperatures 0.9 \
    --train_top_k 2 \
    --max_files 1 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --constrained_note_groups_per_bar 4 \
    --max_simultaneous_notes 2 \
    --require_all_grammar_samples \
    --min_best_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_2file_brad_probe() {
  local run_id="${RUN_ID:-harness_stage_b_2file_brad_probe}"
  print_header "Stage B 2-file Brad generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 33 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --generation_mode constrained \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_coverage_aware_probe() {
  local run_id="${RUN_ID:-harness_stage_b_coverage_aware_probe}"
  print_header "Stage B coverage-aware generation probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 37 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --generation_mode constrained \
    --coverage_aware_positions \
    --coverage_position_window 0 \
    --constrained_note_groups_per_bar 4 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_coverage_ab_sweep() {
  local run_id="${RUN_ID:-harness_stage_b_coverage_ab_sweep}"
  print_header "Stage B coverage-aware A/B sweep"
  "$PYTHON_BIN" scripts/run_stage_b_coverage_ab_sweep.py \
    --run_id "$run_id" \
    --issue_number 39 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --modes plain,coverage \
    --note_groups_per_bar_values 4,6,8 \
    --coverage_position_window 0 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_candidate_ranking() {
  local run_id="${RUN_ID:-harness_stage_b_candidate_ranking}"
  local sweep_run_id="${run_id}_ab_sweep"
  print_header "Stage B candidate ranking"
  "$PYTHON_BIN" scripts/run_stage_b_coverage_ab_sweep.py \
    --run_id "$sweep_run_id" \
    --issue_number 43 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --modes plain,coverage \
    --note_groups_per_bar_values 4,6,8 \
    --coverage_position_window 0 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
  "$PYTHON_BIN" scripts/rank_stage_b_candidates.py \
    --run_id "$run_id" \
    --issue_number 43 \
    --ab_sweep_report "outputs/stage_b_coverage_ab_sweep/${sweep_run_id}/ab_sweep_report.json" \
    --top_n 12 \
    --min_top_strict_candidates 1
}

run_stage_b_chord_aware_probe() {
  local run_id="${RUN_ID:-harness_stage_b_chord_aware_probe}"
  local sweep_run_id="${run_id}_ab_sweep"
  print_header "Stage B chord-aware pitch probe"
  "$PYTHON_BIN" scripts/run_stage_b_coverage_ab_sweep.py \
    --run_id "$sweep_run_id" \
    --issue_number 45 \
    --max_files 2 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 96 \
    --num_samples 3 \
    --modes plain,coverage,coverage_chord \
    --note_groups_per_bar_values 4,6,8 \
    --coverage_position_window 0 \
    --chord_pitch_mode tones \
    --chord_pitch_repeat_window 2 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
  "$PYTHON_BIN" scripts/rank_stage_b_candidates.py \
    --run_id "$run_id" \
    --issue_number 45 \
    --ab_sweep_report "outputs/stage_b_coverage_ab_sweep/${sweep_run_id}/ab_sweep_report.json" \
    --top_n 12 \
    --min_top_strict_candidates 1
}

run_stage_b_longer_phrase_probe() {
  local run_id="${RUN_ID:-harness_stage_b_longer_phrase_probe}"
  print_header "Stage B longer coverage/chord-aware phrase probe"
  "$PYTHON_BIN" scripts/run_stage_b_generation_probe.py \
    --run_id "$run_id" \
    --issue_number 49 \
    --max_files 2 \
    --window_bars 4 \
    --window_stride_bars 2 \
    --min_window_target_notes 8 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 192 \
    --num_samples 3 \
    --bars 4 \
    --generation_mode constrained \
    --coverage_aware_positions \
    --coverage_position_window 0 \
    --chord_aware_pitches \
    --chord_pitch_mode tones \
    --chord_pitch_repeat_window 2 \
    --constrained_note_groups_per_bar 8 \
    --postprocess_overlap \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --require_note_groups \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
  "$PYTHON_BIN" scripts/export_stage_b_review_candidates.py \
    --source_report "outputs/stage_b_generation_probe/${run_id}/report.json" \
    --run_id "$run_id" \
    --top_n 3 \
    --mode coverage_chord \
    --copy_midi
}

run_stage_b_pitch_mode_compare() {
  local run_id="${RUN_ID:-harness_stage_b_pitch_mode_compare}"
  print_header "Stage B tones vs tones_tensions pitch-mode comparison"
  "$PYTHON_BIN" scripts/run_stage_b_pitch_mode_compare.py \
    --run_id "$run_id" \
    --issue_number 55 \
    --max_files 2 \
    --window_bars 4 \
    --window_stride_bars 2 \
    --min_window_target_notes 8 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 192 \
    --num_samples 3 \
    --bars 4 \
    --pitch_modes tones,tones_tensions \
    --coverage_position_window 0 \
    --chord_pitch_repeat_window 2 \
    --note_groups_per_bar 8 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --copy_review_midi \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_8bar_approach_phrase() {
  local run_id="${RUN_ID:-harness_stage_b_8bar_approach_phrase}"
  print_header "Stage B 8-bar approach phrase comparison"
  "$PYTHON_BIN" scripts/run_stage_b_pitch_mode_compare.py \
    --run_id "$run_id" \
    --issue_number 57 \
    --max_files 2 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 384 \
    --num_samples 3 \
    --bars 8 \
    --pitch_modes tones,tones_tensions,approach_tensions \
    --coverage_position_window 0 \
    --chord_pitch_repeat_window 2 \
    --note_groups_per_bar 8 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --copy_review_midi \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_swing_motif_phrase() {
  local run_id="${RUN_ID:-harness_stage_b_swing_motif_phrase}"
  print_header "Stage B swing/motif phrase grammar comparison"
  "$PYTHON_BIN" scripts/run_stage_b_phrase_grammar_compare.py \
    --run_id "$run_id" \
    --issue_number 59 \
    --max_files 2 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --epochs 3 \
    --batch_size 8 \
    --max_sequence 384 \
    --num_samples 3 \
    --bars 8 \
    --grammar_modes approach_baseline,swing_motif_approach \
    --coverage_position_window 0 \
    --chord_pitch_repeat_window 2 \
    --note_groups_per_bar 8 \
    --max_simultaneous_notes 2 \
    --temperature 0.9 \
    --top_k 2 \
    --min_valid_samples 1 \
    --min_strict_valid_samples 1 \
    --min_best_strict_valid_samples 1 \
    --max_collapse_warning_sample_rate 0.34 \
    --require_all_grammar_samples \
    --copy_review_midi \
    --n_layers 1 \
    --num_heads 4 \
    --d_model 64 \
    --dim_feedforward 128 \
    --lora_r 4 \
    --lora_alpha 8
}

run_stage_b_reference_stats() {
  local run_id="${RUN_ID:-harness_stage_b_reference_stats}"
  print_header "Stage B reference phrase statistics"
  "$PYTHON_BIN" scripts/run_stage_b_reference_stats.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --max_records 64 \
    --generated_report outputs/stage_b_phrase_grammar_compare/harness_stage_b_swing_motif_phrase/phrase_grammar_compare_report.json
}

run_stage_b_reference_pitch_roles() {
  local run_id="${RUN_ID:-harness_stage_b_reference_pitch_roles}"
  local generated_run_id="${run_id}_generated"
  print_header "Stage B generated hybrid candidate report"
  RUN_ID="$generated_run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B reference pitch-role landing statistics"
  "$PYTHON_BIN" scripts/run_stage_b_reference_stats.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --max_records 64 \
    --generated_report "outputs/stage_b_data_motif_compare/${generated_run_id}/data_motif_compare_report.json"
}

run_stage_b_motif_templates() {
  local run_id="${RUN_ID:-harness_stage_b_motif_templates}"
  print_header "Stage B motif template extraction"
  "$PYTHON_BIN" scripts/run_stage_b_motif_template_extraction.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --top_n 20
}

run_stage_b_data_motif_compare() {
  local run_id="${RUN_ID:-harness_stage_b_data_motif_compare}"
  print_header "Stage B data-derived motif generation compare"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8
}

run_stage_b_data_motif_review_export() {
  local run_id="${RUN_ID:-harness_stage_b_data_motif_review_export}"
  print_header "Stage B data motif review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi
}

run_stage_b_review_context_grid() {
  local run_id="${RUN_ID:-harness_stage_b_review_context_grid}"
  print_header "Stage B review context and straight-grid export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,hand_written_swing,data_motif \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi
}

run_stage_b_guide_tone_cadence() {
  local run_id="${RUN_ID:-harness_stage_b_guide_tone_cadence}"
  print_header "Stage B straight-grid guide-tone/cadence review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,straight_guide_tones,hand_written_swing,data_motif \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi
}

run_stage_b_data_guide_hybrid() {
  local run_id="${RUN_ID:-harness_stage_b_data_guide_hybrid}"
  print_header "Stage B data-motif rhythm plus guide-tone pitch review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,straight_guide_tones,hand_written_swing,data_motif,data_motif_guide_tones \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi
}

run_stage_b_overlap_free_review_export() {
  local run_id="${RUN_ID:-harness_stage_b_overlap_free_review_export}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B overlap-free solo-line review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 97 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes straight_grid,straight_guide_tones,hand_written_swing,data_motif,data_motif_guide_tones \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_duration_variation_review() {
  local run_id="${RUN_ID:-harness_stage_b_duration_variation_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B duration-variation review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 99 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes varied_grid,varied_guide_tones,hand_written_swing,data_motif,data_motif_guide_tones \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_phrase_cadence_review() {
  local run_id="${RUN_ID:-harness_stage_b_phrase_cadence_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B phrase/cadence review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 101 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes phrase_cadence,varied_guide_tones,data_motif,data_motif_guide_tones \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_phrase_recovery_review() {
  local run_id="${RUN_ID:-harness_stage_b_phrase_recovery_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B phrase recovery review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 105 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes phrase_cadence,phrase_recovery \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_data_motif_phrase_recovery_review() {
  local run_id="${RUN_ID:-harness_stage_b_data_motif_phrase_recovery_review}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B data-motif phrase recovery review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 107 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes data_motif_guide_tones,data_motif_phrase_recovery,phrase_recovery \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_contour_landing_repair() {
  local run_id="${RUN_ID:-harness_stage_b_contour_landing_repair}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B contour/cadence landing repair review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 115 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes data_motif_phrase_recovery,data_motif_contour_landing_repair \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_rhythm_phrase_variation() {
  local run_id="${RUN_ID:-harness_stage_b_rhythm_phrase_variation}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B rhythm/phrase vocabulary variation review export"
  "$PYTHON_BIN" scripts/run_stage_b_data_motif_generation_compare.py \
    --run_id "$run_id" \
    --issue_number 118 \
    --input_dir ./midi_dataset/midi/studio \
    --baseline_modes data_motif_contour_landing_repair,data_motif_rhythm_phrase_variation \
    --max_files 4 \
    --window_bars 8 \
    --window_stride_bars 4 \
    --min_window_target_notes 16 \
    --motif_length 4 \
    --max_bar_span 2 \
    --max_records 64 \
    --template_top_n 32 \
    --num_samples 3 \
    --bars 8 \
    --note_groups_per_bar 8 \
    --review_top_n 3 \
    --copy_review_midi \
    --overlap_free_review_midi
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_clean_review_package() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_data_motif_phrase_recovery_review}"
  local run_id="${RUN_ID:-harness_stage_b_clean_review_package}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${source_run_id}/review_manifest.json"
  local objective_report_path="outputs/stage_b_objective_midi_review/${source_run_id}/objective_midi_note_review.json"
  if [[ ! -f "$review_manifest_path" || ! -f "$objective_report_path" ]]; then
    print_header "Stage B data-motif phrase recovery review export"
    SOURCE_RUN_ID="$source_run_id" RUN_ID="$source_run_id" run_stage_b_data_motif_phrase_recovery_review
  fi
  print_header "Stage B objective-clean review package"
  "$PYTHON_BIN" scripts/build_clean_review_package.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --objective_report "$objective_report_path" \
    --allowed_modes data_motif_phrase_recovery \
    --copy_files \
    --min_candidates 3
}

run_stage_b_proxy_keep_focused_package() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_phrase_shape_tension_proxy}"
  local objective_run_id="${OBJECTIVE_RUN_ID:-harness_stage_b_rhythm_phrase_variation}"
  local run_id="${RUN_ID:-harness_stage_b_proxy_keep_focused_package}"
  local review_notes_file="${REVIEW_NOTES_FILE:-phrase_shape_tension_repaired_review_notes_midi_proxy.json}"
  local review_notes_path="${REVIEW_NOTES_PATH:-outputs/stage_b_listening_review_notes/${source_run_id}/${review_notes_file}}"
  local objective_report_path="outputs/stage_b_objective_midi_review/${objective_run_id}/objective_midi_note_review.json"
  if [[ ! -f "$review_notes_path" ]]; then
    printf 'Missing proxy-filled review notes: %s\n' "$review_notes_path" >&2
    return 2
  fi
  if [[ ! -f "$objective_report_path" ]]; then
    printf 'Missing objective MIDI review report: %s\n' "$objective_report_path" >&2
    return 2
  fi
  print_header "Stage B proxy-keep focused review package"
  "$PYTHON_BIN" scripts/build_focused_review_package.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path" \
    --objective_report "$objective_report_path" \
    --copy_files \
    --min_candidates 1
}

run_stage_b_clean_context_diagnostics() {
  local source_run_id="${SOURCE_RUN_ID:-harness_stage_b_clean_review_package}"
  local run_id="${RUN_ID:-harness_stage_b_clean_context_diagnostics}"
  local clean_package_path="outputs/stage_b_clean_review_package/${source_run_id}/clean_review_package.json"
  if [[ ! -f "$clean_package_path" ]]; then
    print_header "Stage B objective-clean review package"
    RUN_ID="$source_run_id" run_stage_b_clean_review_package
  fi
  print_header "Stage B clean context diagnostics"
  "$PYTHON_BIN" scripts/build_clean_context_diagnostics.py \
    --run_id "$run_id" \
    --clean_package "$clean_package_path" \
    --min_candidates 3
}


run_stage_b_clean_listening_review_notes() {
  local clean_run_id="${CLEAN_RUN_ID:-harness_stage_b_clean_review_package}"
  local diagnostics_run_id="${DIAGNOSTICS_RUN_ID:-harness_stage_b_clean_context_diagnostics}"
  local run_id="${RUN_ID:-harness_stage_b_clean_listening_review_notes}"
  local clean_package_path="outputs/stage_b_clean_review_package/${clean_run_id}/clean_review_package.json"
  local diagnostics_path="outputs/stage_b_clean_context_diagnostics/${diagnostics_run_id}/clean_context_diagnostics.json"
  if [[ ! -f "$clean_package_path" ]]; then
    RUN_ID="$clean_run_id" run_stage_b_clean_review_package
  fi
  if [[ ! -f "$diagnostics_path" ]]; then
    SOURCE_RUN_ID="$clean_run_id" RUN_ID="$diagnostics_run_id" run_stage_b_clean_context_diagnostics
  fi
  print_header "Stage B clean listening review notes"
  "$PYTHON_BIN" scripts/build_clean_listening_review_notes.py     --run_id "$run_id"     --clean_package "$clean_package_path"     --clean_context_diagnostics "$diagnostics_path"
}

run_stage_b_focused_listening_review_notes() {
  local focused_run_id="${FOCUSED_RUN_ID:-harness_stage_b_register_safe_proxy_keep_focused_package}"
  local run_id="${RUN_ID:-harness_stage_b_focused_listening_review_notes}"
  local focused_package_path="${FOCUSED_PACKAGE_PATH:-outputs/stage_b_focused_review_package/${focused_run_id}/focused_review_package.json}"
  if [[ ! -f "$focused_package_path" ]]; then
    printf 'Missing focused review package: %s\n' "$focused_package_path" >&2
    return 2
  fi
  print_header "Stage B focused listening review notes"
  "$PYTHON_BIN" scripts/build_focused_listening_review_notes.py \
    --run_id "$run_id" \
    --focused_package "$focused_package_path"
}

run_manifest_dry_run() {
  local run_id="${RUN_ID:-harness_manifest_prepare}"
  print_header "Manifest prepare dry-run"
  "$PYTHON_BIN" scripts/run_manifest_prepare_smoke.py \
    --run_id "$run_id" \
    --audit_max_files 100 \
    --train_files 4 \
    --val_files 2 \
    --overwrite
}

run_chord_coverage_audit() {
  local run_id="${RUN_ID:-harness_chord_coverage_audit}"
  print_header "Chord progression coverage audit"
  "$PYTHON_BIN" scripts/audit_chord_progression_coverage.py \
    --run_id "$run_id" \
    --input_dir ./midi_dataset \
    --role_meta_roots ./data,./outputs \
    --max_midi_files 120
}

run_stage_b_chord_labeled_eval() {
  local run_id="${RUN_ID:-harness_stage_b_chord_labeled_eval}"
  print_header "Stage B chord-labeled evaluation subset"
  "$PYTHON_BIN" scripts/evaluate_chord_labeled_subset.py \
    --run_id "$run_id" \
    --manifest data/eval/stage_b_chord_labeled_tiny/manifest.json \
    --min_samples 2 \
    --min_notes 16
}

run_stage_b_generated_chord_eval() {
  local run_id="${RUN_ID:-harness_stage_b_generated_chord_eval}"
  print_header "Stage B generated candidate chord-labeled eval bridge"
  "$PYTHON_BIN" scripts/evaluate_generated_candidate_chords.py \
    --run_id "$run_id" \
    --write_tiny_fixture \
    --candidate_limit 1
}

run_stage_b_data_guide_generated_chord_eval() {
  local run_id="${RUN_ID:-harness_stage_b_data_guide_generated_chord_eval}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B data-guide hybrid generated chord eval"
  "$PYTHON_BIN" scripts/evaluate_generated_candidate_chords.py \
    --run_id "$run_id" \
    --candidate_report "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json" \
    --candidate_limit 6
}

run_stage_b_review_markdown_chord_eval() {
  local run_id="${RUN_ID:-harness_stage_b_review_markdown_chord_eval}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B combined review markdown with chord eval"
  "$PYTHON_BIN" scripts/evaluate_generated_candidate_chords.py \
    --run_id "$run_id" \
    --candidate_report "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json" \
    --review_markdown "outputs/stage_b_data_motif_review/${run_id}/review_candidates.md" \
    --candidate_limit 6
}

run_stage_b_listening_review_notes() {
  local run_id="${RUN_ID:-harness_stage_b_listening_review_notes}"
  print_header "Stage B combined review markdown with chord eval"
  RUN_ID="$run_id" run_stage_b_review_markdown_chord_eval
  print_header "Stage B listening review notes template"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --generated_chord_eval_report "outputs/stage_b_generated_chord_eval/${run_id}/generated_chord_eval_report.json" \
    --source_review_markdown "outputs/stage_b_generated_chord_eval/${run_id}/review_candidates_with_chord_eval.md"
}

run_stage_b_full_review_notes() {
  local run_id="${RUN_ID:-harness_stage_b_full_review_notes}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B full review manifest listening notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json" \
    --source_review_markdown "outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
}

run_stage_b_objective_midi_review() {
  local run_id="${RUN_ID:-harness_stage_b_objective_midi_review}"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
}

run_stage_b_objective_flags_review_flow() {
  local run_id="${RUN_ID:-harness_stage_b_objective_flags_review_flow}"
  local review_manifest_path="outputs/stage_b_data_motif_review/${run_id}/review_manifest.json"
  local review_markdown_path="outputs/stage_b_data_motif_review/${run_id}/review_candidates.md"
  local objective_report_path="outputs/stage_b_objective_midi_review/${run_id}/objective_midi_note_review.json"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  print_header "Stage B data-guide hybrid review package"
  RUN_ID="$run_id" run_stage_b_data_guide_hybrid
  print_header "Stage B objective MIDI note review"
  "$PYTHON_BIN" scripts/review_midi_note_objectives.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path"
  print_header "Stage B objective-aware listening review notes"
  "$PYTHON_BIN" scripts/build_listening_review_notes.py \
    --run_id "$run_id" \
    --review_manifest "$review_manifest_path" \
    --source_review_markdown "$review_markdown_path" \
    --objective_midi_review_report "$objective_report_path"
  print_header "Stage B objective-aware listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

run_stage_b_listening_review_aggregate() {
  local run_id="${RUN_ID:-harness_stage_b_listening_review_aggregate}"
  local review_notes_path="outputs/stage_b_listening_review_notes/${run_id}/review_notes_template.json"
  if [[ ! -f "$review_notes_path" || "${FORCE_REGEN:-0}" == "1" ]]; then
    print_header "Stage B listening review notes template"
    RUN_ID="$run_id" run_stage_b_listening_review_notes
  else
    print_header "Stage B listening review notes template"
    printf 'Using existing review notes: %s\n' "$review_notes_path"
  fi
  print_header "Stage B listening review aggregate"
  "$PYTHON_BIN" scripts/summarize_listening_review_notes.py \
    --run_id "$run_id" \
    --review_notes "$review_notes_path"
}

case "$MODE" in
  status)
    run_status
    ;;
  quick)
    run_quick
    ;;
  demo)
    run_demo
    ;;
  tiny-prepare)
    run_tiny_prepare
    ;;
  tiny-compare)
    run_tiny_compare
    ;;
  control-tiny)
    run_control_tiny
    ;;
  stage-b-window-prepare)
    run_stage_b_window_prepare
    ;;
  stage-b-generation-probe)
    run_stage_b_generation_probe
    ;;
  stage-b-raw-generation-repeatability)
    run_stage_b_raw_generation_repeatability
    ;;
  stage-b-dead-air-diagnostics)
    run_stage_b_dead_air_diagnostics
    ;;
  stage-b-seed-strict-margin-diagnostics)
    run_stage_b_seed_strict_margin_diagnostics
    ;;
  stage-b-margin-recovered-review-export)
    run_stage_b_margin_recovered_review_export
    ;;
  stage-b-margin-recovered-listening-notes)
    run_stage_b_margin_recovered_listening_notes
    ;;
  stage-b-margin-recovered-proxy-review-fill)
    run_stage_b_margin_recovered_proxy_review_fill
    ;;
  stage-b-margin-recovered-proxy-keep-focused-package)
    run_stage_b_margin_recovered_proxy_keep_focused_package
    ;;
  stage-b-margin-recovered-focused-context-decision)
    run_stage_b_margin_recovered_focused_context_decision
    ;;
  stage-b-margin-recovered-focused-fallback-comparison)
    run_stage_b_margin_recovered_focused_fallback_comparison
    ;;
  stage-b-margin-recovered-pitch-dead-air-repair)
    run_stage_b_margin_recovered_pitch_dead_air_repair
    ;;
  stage-b-margin-recovered-pitch-vocab-sweep)
    run_stage_b_margin_recovered_pitch_vocab_sweep
    ;;
  stage-b-margin-recovered-pitch-vocab-focused-context)
    run_stage_b_margin_recovered_pitch_vocab_focused_context
    ;;
  stage-b-margin-recovered-pitch-vocab-focused-listening-notes)
    run_stage_b_margin_recovered_pitch_vocab_focused_listening_notes
    ;;
  stage-b-margin-recovered-pitch-vocab-focused-listening-fill)
    run_stage_b_margin_recovered_pitch_vocab_focused_listening_fill
    ;;
  stage-b-margin-recovered-timing-repetition-repair)
    run_stage_b_margin_recovered_timing_repetition_repair
    ;;
  stage-b-margin-recovered-timing-repetition-focused-context)
    run_stage_b_margin_recovered_timing_repetition_focused_context
    ;;
  stage-b-margin-recovered-timing-repetition-focused-listening-notes)
    run_stage_b_margin_recovered_timing_repetition_focused_listening_notes
    ;;
  stage-b-margin-recovered-timing-repetition-focused-listening-fill)
    run_stage_b_margin_recovered_timing_repetition_focused_listening_fill
    ;;
  stage-b-margin-recovered-phrase-vocabulary-repair)
    run_stage_b_margin_recovered_phrase_vocabulary_repair
    ;;
  stage-b-margin-recovered-phrase-vocabulary-focused-context)
    run_stage_b_margin_recovered_phrase_vocabulary_focused_context
    ;;
  stage-b-margin-recovered-phrase-vocabulary-focused-listening-notes)
    run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes
    ;;
  stage-b-margin-recovered-phrase-vocabulary-focused-listening-fill)
    run_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill
    ;;
  stage-b-margin-recovered-phrase-vocabulary-keep-stability)
    run_stage_b_margin_recovered_phrase_vocabulary_keep_stability
    ;;
  stage-b-margin-recovered-phrase-vocabulary-peer-focused-context)
    run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_context
    ;;
  stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-notes)
    run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_notes
    ;;
  stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-fill)
    run_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_fill
    ;;
  stage-b-margin-recovered-phrase-vocabulary-two-candidate-keep)
    run_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep
    ;;
  stage-b-margin-recovered-phrase-vocabulary-human-listening-comparison)
    run_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duplicate-source-divergence)
    run_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence
    ;;
  stage-b-margin-recovered-phrase-vocabulary-sample-seed-diversity)
    run_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-sweep)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-context)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_context
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-notes)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_notes
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-fill)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_fill
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker-repair-sweep)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker_repair_sweep
    ;;
  stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-dead-air-adjacent-repair)
    run_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_dead_air_adjacent_repair
    ;;
  stage-b-margin-recovered-phrase-vocabulary-coverage-aware-adjacent-constrained-repair)
    run_stage_b_margin_recovered_phrase_vocabulary_coverage_aware_adjacent_constrained_repair
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-repair)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-context)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_context
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-notes)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_notes
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-listening-fill)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_focused_listening_fill
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-keep-consolidation)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-boundary)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-review-input-guard)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_guard
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-audio-review-package)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-review)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-consolidation)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-external-human-audio-boundary)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary
    ;;
  stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-local-audio-render-package)
    run_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_local_audio_render_package
    ;;
  stage-b-local-audio-render-tooling)
    run_stage_b_local_audio_render_tooling
    ;;
  stage-b-renderer-path-decision)
    run_stage_b_renderer_path_decision
    ;;
  stage-b-local-audio-render-attempt)
    run_stage_b_local_audio_render_attempt
    ;;
  stage-b-user-listening-review-fill)
    run_stage_b_user_listening_review_fill
    ;;
  stage-b-user-listening-review-consolidation)
    run_stage_b_user_listening_review_consolidation
    ;;
  stage-b-duration-coverage-next-decision)
    run_stage_b_duration_coverage_next_decision
    ;;
  stage-b-duration-coverage-broader-repeatability-sweep)
    run_stage_b_duration_coverage_broader_repeatability_sweep
    ;;
  stage-b-duration-coverage-dead-air-gain-repeatability-repair)
    run_stage_b_duration_coverage_dead_air_gain_repeatability_repair
    ;;
  stage-b-duration-coverage-repeatability-consolidation)
    run_stage_b_duration_coverage_repeatability_consolidation
    ;;
  stage-b-duration-coverage-repeatability-audio-review-package)
    run_stage_b_duration_coverage_repeatability_audio_review_package
    ;;
  stage-b-duration-coverage-repeatability-user-listening-review)
    run_stage_b_duration_coverage_repeatability_user_listening_review
    ;;
  stage-b-duration-coverage-outside-soloing-repair-decision)
    run_stage_b_duration_coverage_outside_soloing_repair_decision
    ;;
  stage-b-duration-coverage-outside-soloing-repair-sweep)
    run_stage_b_duration_coverage_outside_soloing_repair_sweep
    ;;
  stage-b-duration-coverage-outside-soloing-repair-audio-review-package)
    run_stage_b_duration_coverage_outside_soloing_repair_audio_review_package
    ;;
  stage-b-duration-coverage-outside-soloing-repair-user-listening-review)
    run_stage_b_duration_coverage_outside_soloing_repair_user_listening_review
    ;;
  stage-b-duration-coverage-outside-soloing-repair-objective-evidence)
    run_stage_b_duration_coverage_outside_soloing_repair_objective_evidence
    ;;
  stage-b-duration-coverage-outside-soloing-repair-next-decision)
    run_stage_b_duration_coverage_outside_soloing_repair_next_decision
    ;;
  stage-b-duration-coverage-outside-soloing-repair-broader-repeatability)
    run_stage_b_duration_coverage_outside_soloing_repair_broader_repeatability
    ;;
  stage-b-duration-coverage-outside-soloing-repair-repeatability-consolidation)
    run_stage_b_duration_coverage_outside_soloing_repair_repeatability_consolidation
    ;;
  stage-b-duration-coverage-outside-soloing-repair-final-decision)
    run_stage_b_duration_coverage_outside_soloing_repair_final_decision
    ;;
  stage-b-generic-base-readiness-audit)
    run_stage_b_generic_base_readiness_audit
    ;;
  stage-b-generic-base-manifest-contract)
    run_stage_b_generic_base_manifest_contract
    ;;
  stage-b-generic-manifest-window-smoke)
    run_stage_b_generic_manifest_window_smoke
    ;;
  stage-b-generic-base-tiny-training-smoke)
    run_stage_b_generic_base_tiny_training_smoke
    ;;
  stage-b-generic-model-core-training-data-plan)
    run_stage_b_generic_model_core_training_data_plan
    ;;
  stage-b-generic-full-manifest-window-preparation)
    run_stage_b_generic_full_manifest_window_preparation
    ;;
  stage-b-generic-base-training-scale-smoke)
    run_stage_b_generic_base_training_scale_smoke
    ;;
  stage-b-generic-base-scale-checkpoint-generation-probe)
    run_stage_b_generic_base_scale_checkpoint_generation_probe
    ;;
  stage-b-generic-base-scale-checkpoint-grammar-representation-decision)
    run_stage_b_generic_base_scale_checkpoint_grammar_representation_decision
    ;;
  stage-b-generic-base-scale-checkpoint-density-coverage-repair-probe)
    run_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe
    ;;
  stage-b-generic-base-scale-checkpoint-density-coverage-remaining-blocker-decision)
    run_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision
    ;;
  stage-b-generic-base-scale-checkpoint-duration-long-note-repair-probe)
    run_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe
    ;;
  stage-b-generic-base-scale-checkpoint-duration-long-note-remaining-blocker-decision)
    run_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision
    ;;
  stage-b-generic-base-scale-checkpoint-sustained-coverage-dead-air-repair-probe)
    run_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe
    ;;
  stage-b-generic-base-scale-checkpoint-objective-gate-consolidation)
    run_stage_b_generic_base_scale_checkpoint_objective_gate_consolidation
    ;;
  stage-b-generic-base-scale-checkpoint-objective-gate-repeatability-sweep)
    run_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep
    ;;
  stage-b-generic-base-scale-checkpoint-repeatability-consolidation)
    run_stage_b_generic_base_scale_checkpoint_repeatability_consolidation
    ;;
  stage-b-midi-to-solo-mvp-contract)
    run_stage_b_midi_to_solo_mvp_contract
    ;;
  stage-b-midi-to-solo-context-extraction)
    run_stage_b_midi_to_solo_context_extraction
    ;;
  stage-b-midi-to-solo-training-resource-probe)
    run_stage_b_midi_to_solo_training_resource_probe
    ;;
  stage-b-midi-to-solo-conditioned-generation-probe)
    run_stage_b_midi_to_solo_conditioned_generation_probe
    ;;
  stage-b-midi-to-solo-phrase-bank-retrieval-baseline)
    run_stage_b_midi_to_solo_phrase_bank_retrieval_baseline
    ;;
  stage-b-midi-to-solo-phrase-bank-audio-render-package)
    run_stage_b_midi_to_solo_phrase_bank_audio_render_package
    ;;
  stage-b-midi-to-solo-phrase-bank-listening-review-package)
    run_stage_b_midi_to_solo_phrase_bank_listening_review_package
    ;;
  stage-b-midi-to-solo-phrase-bank-listening-review-input-guard)
    run_stage_b_midi_to_solo_phrase_bank_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-phrase-bank-objective-only-next-decision)
    run_stage_b_midi_to_solo_phrase_bank_objective_only_next_decision
    ;;
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-probe)
    run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe
    ;;
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-audio-package)
    run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package
    ;;
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-listening-review-package)
    run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_package
    ;;
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-listening-review-input-guard)
    run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-phrase-bank-dead-air-density-repair-objective-only-next-decision)
    run_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_objective_only_next_decision
    ;;
  stage-b-midi-to-solo-phrase-bank-cli-mvp-package)
    run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package
    ;;
  stage-b-midi-to-solo-phrase-bank-cli-user-input-smoke)
    run_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke
    ;;
  stage-b-midi-to-solo-phrase-bank-cli-audio-render-smoke)
    run_stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke
    ;;
  stage-b-midi-to-solo-phrase-bank-cli-listening-review-package)
    run_stage_b_midi_to_solo_phrase_bank_cli_listening_review_package
    ;;
  stage-b-midi-to-solo-phrase-bank-cli-listening-review-input-guard)
    run_stage_b_midi_to_solo_phrase_bank_cli_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-phrase-bank-cli-objective-only-next-decision)
    run_stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision
    ;;
  stage-b-midi-to-solo-candidate-audio-render-package)
    run_stage_b_midi_to_solo_candidate_audio_render_package
    ;;
  stage-b-midi-to-solo-mvp-execution-consolidation)
    run_stage_b_midi_to_solo_mvp_execution_consolidation
    ;;
  stage-b-midi-to-solo-model-direct-generation-repair)
    run_stage_b_midi_to_solo_model_direct_generation_repair
    ;;
  stage-b-midi-to-solo-model-direct-sequence-budget-repair-smoke)
    run_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke
    ;;
  stage-b-midi-to-solo-model-direct-8bar-generation-probe)
    run_stage_b_midi_to_solo_model_direct_8bar_generation_probe
    ;;
  stage-b-midi-to-solo-model-direct-monophonic-overlap-repair)
    run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair
    ;;
  stage-b-midi-to-solo-model-direct-audio-render-package)
    run_stage_b_midi_to_solo_model_direct_audio_render_package
    ;;
  stage-b-midi-to-solo-model-direct-audio-evidence-consolidation)
    run_stage_b_midi_to_solo_model_direct_audio_evidence_consolidation
    ;;
  stage-b-midi-to-solo-model-direct-phrase-quality-diagnostics)
    run_stage_b_midi_to_solo_model_direct_phrase_quality_diagnostics
    ;;
  stage-b-midi-to-solo-model-direct-pitch-contour-repair)
    run_stage_b_midi_to_solo_model_direct_pitch_contour_repair
    ;;
  stage-b-midi-to-solo-model-direct-timing-phrase-repair)
    run_stage_b_midi_to_solo_model_direct_timing_phrase_repair
    ;;
  stage-b-midi-to-solo-model-direct-listening-review-package)
    run_stage_b_midi_to_solo_model_direct_listening_review_package
    ;;
  stage-b-midi-to-solo-model-direct-user-listening-review-input-guard)
    run_stage_b_midi_to_solo_model_direct_user_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-model-direct-user-listening-review-fill)
    run_stage_b_midi_to_solo_model_direct_user_listening_review_fill
    ;;
  stage-b-midi-to-solo-model-direct-songlike-rejection-analysis)
    run_stage_b_midi_to_solo_model_direct_songlike_rejection_analysis
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-decision)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_decision
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-probe)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_probe
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-audio-package)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_audio_package
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-listening-review)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_listening_review
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-repair-objective-next)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_repair_objective_next
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repair)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repair
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-audio-package)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_audio_package
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-listening-review)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_listening_review
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-objective-next)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_objective_next
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-sweep)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_sweep
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-consolidation)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_consolidation
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-audio-package)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_audio_package
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-listening-review)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_listening_review
    ;;
  stage-b-midi-to-solo-model-direct-jazz-phrase-vocabulary-contour-phrase-shape-repeatability-objective-next)
    run_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next
    ;;
  stage-b-midi-to-solo-training-scale-expansion-decision)
    run_stage_b_midi_to_solo_training_scale_expansion_decision
    ;;
  stage-b-midi-to-solo-controlled-training-scale-smoke)
    run_stage_b_midi_to_solo_controlled_training_scale_smoke
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-generation-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-repair-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-density-collapse-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-remaining-blocker-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repair-repeatability-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-repair-consolidation)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-audio-review-package)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_audio_review_package
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-listening-review)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-dead-air-repeatability-temperature-guard-objective-next)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-expansion-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-smoke)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-generation-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-repair-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-density-grammar-collapse-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-density-grammar-collapse-repeatability-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-remaining-blocker-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repair-repeatability-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repeatability-temperature-guard-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repeatability-temperature-guard-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-dead-air-repeatability-temperature-guard-followup-decision)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-probe)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-consolidation)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-audio-review-package)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_audio_review_package
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-listening-review)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_listening_review
    ;;
  stage-b-midi-to-solo-controlled-scale-checkpoint-training-scale-postprocess-removal-dead-air-repair-objective-next)
    run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_objective_next
    ;;
  stage-b-midi-to-solo-mvp-current-evidence-consolidation)
    run_stage_b_midi_to_solo_mvp_current_evidence_consolidation
    ;;
  stage-b-midi-to-solo-mvp-completion-audit)
    run_stage_b_midi_to_solo_mvp_completion_audit
    ;;
  stage-b-midi-to-solo-quality-gap-decision)
    run_stage_b_midi_to_solo_quality_gap_decision
    ;;
  stage-b-midi-to-solo-listening-review-quality-gap)
    run_stage_b_midi_to_solo_listening_review_quality_gap
    ;;
  stage-b-midi-to-solo-mvp-delivery-package)
    run_stage_b_midi_to_solo_mvp_delivery_package
    ;;
  stage-b-midi-to-solo-final-status-audit)
    run_stage_b_midi_to_solo_final_status_audit
    ;;
  stage-b-midi-to-solo-post-mvp-quality-iteration-plan)
    run_stage_b_midi_to_solo_post_mvp_quality_iteration_plan
    ;;
  stage-b-midi-to-solo-quality-rubric-baseline)
    run_stage_b_midi_to_solo_quality_rubric_baseline
    ;;
  stage-b-midi-to-solo-candidate-failure-labeling)
    run_stage_b_midi_to_solo_candidate_failure_labeling
    ;;
  stage-b-midi-to-solo-targeted-quality-repair-sweep)
    run_stage_b_midi_to_solo_targeted_quality_repair_sweep
    ;;
  stage-b-midi-to-solo-targeted-quality-repair-audio-package)
    run_stage_b_midi_to_solo_targeted_quality_repair_audio_package
    ;;
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-review-decision)
    run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision
    ;;
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-probe)
    run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe
    ;;
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-audio-package)
    run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package
    ;;
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-listening-review-package)
    run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_package
    ;;
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-listening-review-input-guard)
    run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-model-conditioned-pitch-contour-changed-ratio-repair-objective-next)
    run_stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_objective_next
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-quality-alignment)
    run_stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-probe)
    run_stage_b_midi_to_solo_model_conditioned_input_path_probe
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-candidate-export)
    run_stage_b_midi_to_solo_model_conditioned_input_path_candidate_export
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-audio-render-package)
    run_stage_b_midi_to_solo_model_conditioned_input_path_audio_render_package
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-replacement-consolidation)
    run_stage_b_midi_to_solo_model_conditioned_input_path_replacement_consolidation
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-listening-review-package)
    run_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_package
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-listening-review-input-guard)
    run_stage_b_midi_to_solo_model_conditioned_input_path_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-objective-next)
    run_stage_b_midi_to_solo_model_conditioned_input_path_objective_next
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-decision)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-probe)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-audio-package)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio_package
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-objective-next)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-decision)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-probe)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_probe
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-audio-package)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_audio_package
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-listening-review-package)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_package
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-listening-review-input-guard)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_listening_review_input_guard
    ;;
  stage-b-midi-to-solo-model-conditioned-input-path-dead-air-timing-repair-pitch-contour-objective-next)
    run_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_objective_next
    ;;
  stage-b-generic-tiny-checkpoint-generation-probe)
    run_stage_b_generic_tiny_checkpoint_generation_probe
    ;;
  stage-b-generic-tiny-checkpoint-grammar-repair)
    run_stage_b_generic_tiny_checkpoint_grammar_repair
    ;;
  stage-b-generic-tiny-checkpoint-repair-repeatability)
    run_stage_b_generic_tiny_checkpoint_repair_repeatability
    ;;
  stage-b-generic-tiny-checkpoint-repair-review-package)
    run_stage_b_generic_tiny_checkpoint_repair_review_package
    ;;
  stage-b-generic-tiny-checkpoint-repair-listening-notes)
    run_stage_b_generic_tiny_checkpoint_repair_listening_notes
    ;;
  stage-b-generic-tiny-checkpoint-repair-listening-fill)
    run_stage_b_generic_tiny_checkpoint_repair_listening_fill
    ;;
  stage-b-generic-tiny-checkpoint-repair-audio-render-package)
    run_stage_b_generic_tiny_checkpoint_repair_audio_render_package
    ;;
  stage-b-generic-tiny-checkpoint-repair-local-audio-render-attempt)
    run_stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt
    ;;
  stage-b-generic-tiny-checkpoint-repair-user-listening-review)
    run_stage_b_generic_tiny_checkpoint_repair_user_listening_review
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-decision)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-sweep)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-audio-render-package)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-local-audio-render-attempt)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-midi-note-failure-review)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-decision)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sweep)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-audio-render-package)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-local-audio-render-attempt)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-user-listening-review)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-rejection-analysis)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-repair-decision)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-repair-sweep)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-audio-render-package)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-local-audio-render-attempt)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_local_audio_render_attempt
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-user-listening-review)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-rejection-analysis)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis
    ;;
  stage-b-generic-tiny-checkpoint-repair-phrase-continuation-range-interval-guard-sparse-phrase-model-core-review)
    run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_model_core_review
    ;;
  stage-b-constrained-probe)
    run_stage_b_constrained_probe
    ;;
  stage-b-overlap-gate)
    run_stage_b_overlap_gate
    ;;
  stage-b-stronger-probe)
    run_stage_b_stronger_probe
    ;;
  stage-b-collapse-sweep)
    run_stage_b_collapse_sweep
    ;;
  stage-b-2file-brad-probe)
    run_stage_b_2file_brad_probe
    ;;
  stage-b-coverage-aware-probe)
    run_stage_b_coverage_aware_probe
    ;;
  stage-b-coverage-ab-sweep)
    run_stage_b_coverage_ab_sweep
    ;;
  stage-b-candidate-ranking)
    run_stage_b_candidate_ranking
    ;;
  stage-b-chord-aware-probe)
    run_stage_b_chord_aware_probe
    ;;
  stage-b-longer-phrase-probe)
    run_stage_b_longer_phrase_probe
    ;;
  stage-b-pitch-mode-compare)
    run_stage_b_pitch_mode_compare
    ;;
  stage-b-8bar-approach-phrase)
    run_stage_b_8bar_approach_phrase
    ;;
  stage-b-swing-motif-phrase)
    run_stage_b_swing_motif_phrase
    ;;
  stage-b-reference-stats)
    run_stage_b_reference_stats
    ;;
  stage-b-reference-pitch-roles)
    run_stage_b_reference_pitch_roles
    ;;
  stage-b-motif-templates)
    run_stage_b_motif_templates
    ;;
  stage-b-data-motif-compare)
    run_stage_b_data_motif_compare
    ;;
  stage-b-data-motif-review-export)
    run_stage_b_data_motif_review_export
    ;;
  stage-b-review-context-grid)
    run_stage_b_review_context_grid
    ;;
  stage-b-guide-tone-cadence)
    run_stage_b_guide_tone_cadence
    ;;
  stage-b-data-guide-hybrid)
    run_stage_b_data_guide_hybrid
    ;;
  stage-b-overlap-free-review-export)
    run_stage_b_overlap_free_review_export
    ;;
  stage-b-duration-variation-review)
    run_stage_b_duration_variation_review
    ;;
  stage-b-phrase-cadence-review)
    run_stage_b_phrase_cadence_review
    ;;
  stage-b-phrase-recovery-review)
    run_stage_b_phrase_recovery_review
    ;;
  stage-b-data-motif-phrase-recovery-review)
    run_stage_b_data_motif_phrase_recovery_review
    ;;
  stage-b-contour-landing-repair)
    run_stage_b_contour_landing_repair
    ;;
  stage-b-rhythm-phrase-variation)
    run_stage_b_rhythm_phrase_variation
    ;;
  stage-b-clean-review-package)
    run_stage_b_clean_review_package
    ;;
  stage-b-proxy-keep-focused-package)
    run_stage_b_proxy_keep_focused_package
    ;;
  stage-b-clean-context-diagnostics)
    run_stage_b_clean_context_diagnostics
    ;;
  stage-b-clean-listening-review-notes)
    run_stage_b_clean_listening_review_notes
    ;;
  stage-b-focused-listening-review-notes)
    run_stage_b_focused_listening_review_notes
    ;;
  manifest-dry-run)
    run_manifest_dry_run
    ;;
  chord-coverage-audit)
    run_chord_coverage_audit
    ;;
  stage-b-chord-labeled-eval)
    run_stage_b_chord_labeled_eval
    ;;
  stage-b-generated-chord-eval)
    run_stage_b_generated_chord_eval
    ;;
  stage-b-data-guide-generated-chord-eval)
    run_stage_b_data_guide_generated_chord_eval
    ;;
  stage-b-review-markdown-chord-eval)
    run_stage_b_review_markdown_chord_eval
    ;;
  stage-b-listening-review-notes)
    run_stage_b_listening_review_notes
    ;;
  stage-b-full-review-notes)
    run_stage_b_full_review_notes
    ;;
  stage-b-objective-midi-review)
    run_stage_b_objective_midi_review
    ;;
  stage-b-objective-flags-review-flow)
    run_stage_b_objective_flags_review_flow
    ;;
  stage-b-listening-review-aggregate)
    run_stage_b_listening_review_aggregate
    ;;
  all)
    run_demo
    run_tiny_compare
    run_control_tiny
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown harness mode: $MODE"
    usage
    exit 1
    ;;
esac
