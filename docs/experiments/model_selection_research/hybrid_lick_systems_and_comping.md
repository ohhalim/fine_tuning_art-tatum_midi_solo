# Hybrid lick systems (model + rules / corpus + constraints) for "no wrong notes + bebop lick-ness", and clean real-time jazz comping

*How these notes were gathered (2026-10-03): WebFetch was blocked by the egress proxy for most academic hosts (arxiv.org, hal.science, transactions.ismir.net, archives.ismir.net, springer, openedition, cs.hmc.edu, ai.stanford.edu, ircam.fr, jazzomat/hfm-weimar, francoispachet.fr). GitHub pages could be read directly, and those findings are marked **[read directly]**. Every other finding comes from search-engine extracts of the cited page. Those are reliable for titles, authors, venues and abstract-level claims, but numbers taken from them should be spot-checked against the PDF before anyone quotes them. Items are dated as (year/month) wherever the year could be confirmed.*

## Q1. Corpus/concatenative and scenario-guided systems (ImproteK/DJazz/Dicy2, Somax2, Pachet's Markov constraints/Virtuoso/Continuator, Impro-Visor, GenJam, Weimar Bebop Alphabet, lick databases): how do they guarantee harmonic fit while keeping licks, and what did listeners say?

### Takeaway
The systems that keep "lick-ness" get harmonic correctness from where the material came from, not from checking each note. ImproteK/DJazz/Dicy2, Pachet's Virtuoso and Markov-constraint generators, and the Continuator copy contiguous fragments of real playing that were originally played over the same chord label, possibly transposed. They apply constraints to the search or navigation through that memory, not to individual pitches. Systems that pick each pitch from chord-scale rules (Impro-Visor grammars, Frieler's Weimar Bebop Alphabet model, GenJam) are harmonically safe but get rated as less idiomatic. This matches the user's failure #3 ("no licks, sounds like a melody"). Corpus evidence shows bebop lines are largely made of recurring 4-interval patterns, so lick identity lives at the interval n-gram level. Choosing pitches one note at a time destroys those patterns.

### Cited Findings
**ImproteK → DJazz → DYCI2/Dicy2 (IRCAM / EHESS; 2011–2022)**
- (2011–2013) ImproteK came out of research done with jazz musician Bernard Lubat specifically to build improvisation software. — [repmus.ircam.fr Nika page](http://repmus.ircam.fr/nika/improvisation_guidee)
- (2017, ACM Computers in Entertainment) A "scenario" is a specification of high-level musical structure that sets **hard constraints** on the generated sequences. "Improvising" means navigating an indexed memory to collect contiguous or disconnected sequences that match the successive parts of the scenario, for example a chord progression. — [Nika et al., ImproteK: Introducing Scenarios into Human-Computer Music Improvisation (HAL PDF)](https://hal.science/hal-01380163/file/ACM-CiE_ImproteK_Nika-et-al_2017.pdf)
- Constrained navigation reads successive labels of the input grid and looks for beats indexed by the same labels in the memory (oracle). It searches for continuity in the musical discourse by exploiting similar patterns. A prefix-indexing algorithm handles continuity with the *future* of the scenario, and anticipation comes from taking the required future labels into account. Transposition is handled when the scenario is a harmonic progression. — [ResearchGate: ImproteK (2017)](https://www.researchgate.net/publication/309575073_ImproteK_Introducing_Scenarios_into_Human-Computer_Music_Improvisation); [Nika 2012, "ImproteK, integrating harmonic controls…" (PDF)](http://architexte.ircam.fr/textes/Nika12a/index.pdf)
- (2012–2016) Evaluation was qualitative and long-term, with ten expert musicians: concerts, work sessions, filmed listening sessions and interviews. — [Nika et al. 2017 (HAL PDF)](https://hal.science/hal-01380163/file/ACM-CiE_ImproteK_Nika-et-al_2017.pdf)
- (2015, Cahiers d'ethnomusicologie, Chemillier & Nika) Bernard Lubat's judgments of ImproteK, "Étrangement musical" ("strangely musical"):
  - The software memorizes a musician's phrases **together with their phrasing and articulation** so it can reuse them.
  - Lubat's critiques fall into three themes: **phrasing (stiffness vs. flexibility)**, **conduct/direction of the improvisation**, and **rhythm/tempo accuracy**.
  - He also discussed "error" and transgressing the limits of the idiom.
  - Notably, harmony is not the main complaint once material is recombined from real playing.
  — [OpenEdition article](https://journals.openedition.org/ethnomusicologie/2496?lang=en); [ResearchGate](https://www.researchgate.net/publication/286342371_Etrangement_musical_les_jugements_de_gout_de_Bernard_Lubat_a_propos_du_logiciel_d'improvisation_ImproteK)
- DJazz is a variant of ImproteK by Marc Chemillier and Jérôme Nika. It relies on a database of musical sequences (audio or MIDI) associated with known chord changes. Mikhail Malt (ANR MERCI) was redesigning it with the aim of free distribution in 2022. — [ResearchGate/Academia: "Reflecting on the Musicality of ML-based Music Generators in Real-Time Jazz Improvisation: a case study of OMax-ImproteK-Djazz" (c. 2021)](https://www.researchgate.net/publication/355128945_Reflecting_on_the_Musicality_of_Machine_Learning_based_Music_Generators_in_Real-Time_Jazz_Improvisation_A_case_study_of_OMax-ImproteK-Djazz); [digitaljazz.fr research](http://digitaljazz.fr/research/); [IRCAM Forum Djazz](https://forum.ircam.fr/projects/detail/djazz/)
- **[read directly]** Dicy2, the successor library to the DYCI2 agents:
  - License **GPL v3**. Python generative core plus a Max package (macOS High Sierra+, Max 8+) and an Ableton Live device.
  - Current release v3 (2022), developed by Jérôme Nika's team at IRCAM (projects ANR-DYCI2, ANR-MERCI, ERC-REACH).
  — [GitHub DYCI2/Dyci2Lib](https://github.com/DYCI2/Dyci2Lib)

**Somax2 (IRCAM; 2019–2026)**
- (2022/06, SMC'22, Borg, Assayag & Malt) How Somax2 works:
  - The corpus is segmented, and each fragment gets multilayer analysis: harmony as chroma vectors, melody as pitch.
  - Players are "influenced" by external audio or MIDI streams.
  - The focus is the reactivity of **non-idiomatic** improvisation.
  — [Zenodo: Somax 2, a Reactive Multi-Agent Environment for Co-Improvisation](https://zenodo.org/records/6800805)
- **[read directly]** Somax2 license is **GPL-3.0**. It needs Max 8.6+ (9.0.3+ recommended), macOS 10.13+ or Windows 10+, and Python 3.9+ for manual install. The README does not mention Apple Silicon explicitly. — [GitHub DYCI2/Somax2](https://github.com/DYCI2/Somax2)
- (2022, Computer Music Journal 46(4)) "Cocreative Interaction: Somax2 and the REACH Project". — [MIT Press](https://direct.mit.edu/comj/article/46/4/7/119103/Cocreative-Interaction-Somax2-and-the-REACH)

**Pachet / Roy: Markov constraints, Virtuoso bebop generator, Continuator, Reflexive Looper (Sony CSL → independent; 2003–2026)**
- (2012, Springer chapter "Musical Virtuosity and Creativity", in *Computers and Creativity*) Architecture for generating virtuoso bebop phrases:
  - The core is a variable-order Markov chain of **maximum order 2**.
  - It uses **chord-specific training databases, selected at each beat according to the underlying chord sequence** through a simple set of chord/scale association rules.
  - Examples use transpositions of material.
  - **Side-slipping** is treated as the precisely defined bebop device for "playing out" the right way.
  - The authors claim output is "of the same musical quality as the ones human virtuosos produce" and rated comparable to a virtuoso. I could not find the evaluation protocol, so treat this as the authors' claim.
  — [Springer](https://link.springer.com/chapter/10.1007/978-3-642-31727-9_5); [ResearchGate](https://www.researchgate.net/publication/259383942_Musical_Virtuosity_and_Creativity)
- (2011) Pachet & Roy, "Markov constraints: steerable generation of Markov sequences" (*Constraints* journal) and "Finite-Length Markov Processes with Constraints" (IJCAI 2011). — [Semantic Scholar](https://www.semanticscholar.org/paper/Finite-Length-Markov-Processes-with-Constraints-Pachet-Roy/54347f25f2ac0bea4301b32cf040c52eaee8d730)
- (2015, CP'15) Papadopoulos, Pachet, Roy & Sakellariou, "Exact Sampling for Regular and Markov Constraints with Belief Propagation". It builds a linear factor graph whose factors combine Markov transitions with finite-automaton transitions. It samples **exactly** from the Markov distribution conditioned on acceptance by the automaton, with no distortion. — [ResearchGate](https://www.researchgate.net/publication/283343726_Exact_Sampling_for_Regular_and_Markov_Constraints_With_Belief_Propagation)
- **(2026/05, NEW)** Pachet, "Exact Regular-Constrained Variable-Order Markov Generation via Sparse Context-State Belief Propagation", arXiv 2605.07839, reported accepted to NeurIPS 2026. For a fixed context graph and automaton, inference is linear in the sequence horizon, and it gives the correct variable-order distribution conditioned on regular constraints. — [arXiv abs](https://arxiv.org/abs/2605.07839)
  - **[read directly]** Code `vo-regular-bp`: **MIT**, Python 3.10+.
    - Supported constraints: positional masks, **meter constraints**, **forbidden substrings**, and `most_probable_sequence()` (max-plus DP).
    - Examples constrain final pitch classes and avoid repeated n-grams from reference sequences.
    — [GitHub fpachet/vo-regular-bp](https://github.com/fpachet/vo-regular-bp)
- **(2026/06, NEW)** Pachet, "Attractive and Repulsive Pattern Control in Sequence Generation", arXiv 2606.24911:
  - Problem: variable-order Markov generators fall into "**tunnels**", attractor-like corridors of recurring high-order contexts, repeated suffixes and locally periodic passages.
  - Method: a weighted recurrence automaton plus belief propagation penalizes (negative coupling) or rewards (positive coupling) target patterns.
  - Result: the negative branch reduces generated 8-gram self-reuse and increases the effective number of distinct 8-grams.
  — [arXiv abs](https://arxiv.org/abs/2606.24911)
- **[read directly]** Continuator repo (Pachet):
  - License **MIT**, Python 3.11+.
  - Variable-order Markov plus finite-chain inference, with positional/unary constraints, regular constraints via `vo_regular_bp`, meter enforcement, and "**virtual transposition augmentation**".
  - Real-time MIDI through `python-rtmidi`, plus a Gradio UI.
  - Supports online/real-time learning and claims to need less data than transformer models.
  — [GitHub fpachet/Continuator](https://github.com/fpachet/Continuator)
- (2013, CHI, Best Paper Honorable Mention) Pachet, Roy, Moreira & d'Inverno, "Reflexive Loopers for Solo Musical Improvisation". The system classifies the musician's current playing mode (bass, chords or solo) and plays the "other members". For example, when the human solos, the system plays bass and chords. — [Goldsmiths PDF](https://research.gold.ac.uk/9888/1/Reflexive%20Loopers%20for%20Solo%20Music%20Improvisation.pdf)
- (2020) Overview of Flow Machines and Markov constraints in assisted composition. — [arXiv 2006.09232](https://arxiv.org/html/2006.09232v3)

**Impro-Visor (Harvey Mudd College; 2006–2019)**
- (2007, SMC) Keller & Morrison, "A Grammatical Approach to Automatic Improvisation". Probabilistic grammars instantiate note *categories* that correspond to jazz concepts, with range constraints applied at generation time. "Slopes" are the building blocks for contour. — [ResearchGate](https://www.researchgate.net/publication/228380885_A_grammatical_approach_to_automatic_improvisation)
- (2010, Computer Music Journal) Gillick, Tang & Keller, "Machine Learning of Jazz Grammars":
  - Terminal categories: **C** = chord tone; **L** = color tone; **A** = a tone that *chromatically approaches* a C or L; **S** = scale tone; **X** = arbitrary; **R** = rest. Each terminal carries a duration, e.g. A8 or C4.
  - Grammars are learned unsupervised from transcriptions by clustering plus Markov chains.
  - The paper reports blind comparisons of solos from grammars learned on different corpora.
  — [Stanford PDF](http://ai.stanford.edu/~kdtang/papers/cmj10-jazzgrammar.pdf)
- (2012, ICCC) "Continuous Improvisation and Trading with Impro-Visor". — [PDF](https://computationalcreativity.net/iccc2012/wp-content/uploads/2012/05/222-Keller.pdf). (2016, MUME) "Active Trading with Impro-Visor". — [PDF](https://musicalmetacreation.org/mume2016/proceedings/Kondak_Impro_Visor.pdf)
- (2017, ICCC) Johnson, Keller & Weintraut, "Learning to Create Jazz Melodies Using a Product of Experts". Two LSTMs, an **interval expert** and a **chord expert**, are multiplied to give a distribution over next notes. It was implemented inside Impro-Visor. — [PDF](https://www.danieldjohnson.com/files/2017jazzproduct.pdf)
- The roadmap editor analyzes chord changes for implied keys and idiomatic progressions ("bricks"). — [Wikipedia: Impro-Visor](https://en.wikipedia.org/wiki/Impro-Visor)
- **[read directly]** Status and license:
  - Impro-Visor is **GPL-2.0** and written in Java 1.8+.
  - The latest release is 10.2 (2019-06-12), and the project appears inactive.
  - Features include grammar generation, style accompaniment, roadmap, LSTM "connectome" files and trading.
  — [GitHub Impro-Visor/Impro-Visor](https://github.com/Impro-Visor/Impro-Visor)
  - Wikipedia lists GPL-2.0-or-later. — [Wikipedia](https://en.wikipedia.org/wiki/Impro-Visor)
  - A Swift iPad port, "Real imPro", is also GPL v2. — [GitHub onlyonear/RealimPro](https://github.com/onlyonear/RealimPro)
  - The Association for Computational Creativity published "Remembering Bob Keller", which mentions a user community of more than 8,000. — [ACC](https://computationalcreativity.net/home/remembering-bob/)

**GenJam (Biles; 1994–)**
- Two hierarchical populations (measures and phrases) are evolved with fitness from a human mentor. Measures and phrases are mapped to notes **through scales suggested by the chord progression**. GenJam trades fours or eights with a human soloist in real time. — [ResearchGate: GenJam](https://www.researchgate.net/publication/238992392_GenJam_An_interactive_genetic_algorithm_jazz_improviser); [Biles SMC'99 PDF](https://genjam.org/wp-content/uploads/2019/07/bilessmc99.pdf)

**Weimar Bebop Alphabet / analysis-by-synthesis (Frieler; 2019–2022)**
- (2019) The Weimar Bebop Alphabet (WBA) parses each phrase's interval sequence into nine classes of melodic "atoms": diatonic, chromatic, **approaches**, arpeggios, jump arpeggios, repetitions, trills, links and X (residual). — [TISMIR 2022](https://transactions.ismir.net/articles/10.5334/tismir.87); [Frieler 2019, "Constructing Jazz Lines" (PDF)](https://jazzforschung.hfm-weimar.de/wp-content/uploads/2019/06/JazzforschungHeute2019_Frieler-Constructing-Jazz-Lines.pdf)
- (2022/02, TISMIR 5(1):20–34) Frieler & Zaddach, "Evaluating an Analysis-by-Synthesis Model for Jazz Improvisation":
  - The model is a hierarchical Markov model. First it picks a mid-level unit (MLU). Then it picks a first-order Markov sequence of WBA atoms. **Pitches are "realized" from the chord context using chord-scale theory**, and rhythm comes from a first-order Markov model of duration classes.
  - "Line" and "lick" MLUs cover about 75% of MLUs.
  - For line MLUs, durations were **fixed to 8ths or 16ths**, because sampling from the real IOI distribution gave rhythms that were too inhomogeneous and syncopated.
  — [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87)
- Listening results from the same paper (search extract; verify against the PDF):
  - Jazz experts identified the computer solos with **64.4%** accuracy; non-experts with **41.7%**.
  - Algorithmic solos were ranked very low, except for the improved version. That version fooled raters at about 44% overall accuracy (experts 53%, non-experts 18%).
  - The authors conclude that rendition matters: timbre, articulation, micro-timing and band–soloist interaction were "equally if not more important" than tonal content.
  — [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87); [ResearchGate](https://www.researchgate.net/publication/358311603_Evaluating_an_Analysis-by-Synthesis_Model_for_Jazz_Improvisation)

**Pattern/lick statistics and databases (why licks should be retrieved whole)**
- (2014, Music Perception) Norgaard analyzed 48 Charlie Parker solos. **82.6% of notes begin a 4-interval pattern** that recurs in the corpus, and **57.6% begin a combined interval+rhythm pattern** (abstract-level figure from search extract). — [ResearchGate: "How Jazz Musicians Improvise"](https://www.researchgate.net/publication/278914755_How_Jazz_Musicians_Improvise)
- (2013, Psychomusicology 23(4):243–254) Norgaard, Spencer & Montiel built a pattern-based probabilistic algorithm for melody and rhythm in jazz improvisation. — [ResearchGate](https://www.researchgate.net/publication/263916053_Testing_cognitive_theories_by_creating_a_pattern-based_probabilistic_algorithm_for_melody_and_rhythm_in_jazz_improvisation)
- (2023, Musicae Scientiae) Cross & Goldman, "Interval patterns are dependent on metrical position in jazz solos" (title-level evidence only). — [DOI](https://doi.org/10.1177/10298649211033973)
- (2019–2024) Dig That Lick corpus sizes, and pattern search that includes transposition and interval transformations with edit-distance similarity:
  - DTL1000: about 300,000 tone events in 1,736 solos.
  - Weimar Jazz Database: about 200,000 tone events in 456 solos by 78 players.
  - Charlie Parker Omnibook: about 18,000 tones in 52 solos.
  — [ISMIR 2019 LBD PDF](https://archives.ismir.net/ismir2019/latebreaking/000037.pdf); [Dig That Lick home](https://dig-that-lick.eecs.qmul.ac.uk/)
- (2020, ISMIR) BebopNet, a neural baseline for comparison:
  - It quoted its training corpus at a **3.8%** rate, similar to the baseline rate at which human jazz giants quote each other.
  - Harmonic coherence on unseen progressions: chord match 0.53 and scale match 0.82, vs. Charlie Parker's 0.52 and 0.80 (search extract).
  — [ISMIR 2020 PDF](https://archives.ismir.net/ismir2020/paper/000132.pdf)
  - **[read directly]** Code is **MIT**, with LSTM and Transformer models. A pretrained Transformer is included. A `--score_model harmony` beam search "prefers notes coherent with the scale of the chord currently being played". — [GitHub shunithaviv/bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- (2025) "Phrase-Oriented Generative Rhythmic Patterns for Jazz Solos" (Applied Sciences; title-level only). — [DOI](https://doi.org/10.3390/app152011058)

### Inferences
- The user's failure #3 (rule realization sounds "like a melody, plunky") is predictable from these sources.
  - Realizing pitches note by note from chord-scale rules is exactly what Frieler's WBA model, Impro-Visor and GenJam do. Of these, only Frieler's model has a published listening test, and its unenhanced output was ranked very low.
  - Norgaard's 82.6% figure says bebop lick identity sits in specific ~4-interval sequences, which per-note pitch choice destroys.
  - The fix is to **retrieve and transpose whole interval-pattern fragments, then splice them under constraints**, as ImproteK and Virtuoso do. Generating contour and filling pitches does not get there.
- "Correct by provenance" means the following. Take a fragment Parker played over Dm7→G7 and transpose it to Em7→A7. Its chord-tone, approach and enclosure structure relative to the chords is preserved, so it is "right" in the same way the original was. The risks move elsewhere:
  - (a) seams between fragments;
  - (b) label mismatch, e.g. a G7alt fragment over a G7sus;
  - (c) rhythm/phrasing stiffness, which was Lubat's main critique;
  - (d) repetition "tunnels", which Pachet's 2026 repulsive control addresses.
- A feasible design for one person on a Mac with no GPU:
  - Build a **half-bar or beat-indexed lick memory** from WJazzD and Omnibook bebop solos.
  - Key each entry by chord quality, root-relative transposition, metric position of first and last note, and the pitch-class role of the landing note.
  - Recombine with `vo-regular-bp` or `Continuator` (MIT, CPU, Python) using meter, landing-note and forbidden-repeat automata.
  - Exact belief propagation is linear in horizon. That should fit a 0.4 s budget for a 1–2 half-bar horizon, but this has not been benchmarked.
- Licenses:
  - Somax2, Dicy2 and Impro-Visor are GPL. Copying their code means your system must be GPL, while reimplementing their ideas does not.
  - Continuator, vo-regular-bp, BebopNet and realchords-pytorch are MIT, so they are safe to embed.

### Gaps
- I could not read the Virtuoso evaluation protocol (Springer blocked), so "rated comparable to a virtuoso" is unverified.
- ImproteK and DJazz evaluations are qualitative only. I found no quantitative listening scores.
- I could not confirm whether DJazz is now publicly downloadable or under what license.
- I could not determine how Somax2's chroma "influence" behaves on fast bebop changes. It is designed for non-idiomatic reactivity, so idiomatic jazz performance is untested in the sources I found.
- I did not verify the license or terms of use for WJazzD and DTL1000 data.
- The Frieler TISMIR percentages come from a search extract and should be checked against the PDF, because the 64.4% and 53% figures refer to different conditions.

## Q2. Neural + constraint hybrids: constrained decoding, critics/rerankers, planner→realizer, infilling; why hard masks cause repetition collapse; better alternatives

### Takeaway
Per-token hard masking (mask, then renormalize at each step) is a known source of distribution distortion. Grammar-Aligned Decoding (NeurIPS 2024) formalizes this for LLMs: probability mass from banned continuations is pushed onto whichever legal tokens remain. In a melody model the safest remaining token is often the previous pitch, which plausibly explains the user's 20% same-note repetition. Methods that preserve the distribution or use constraints softly exist and are cheap at half-bar scale:
- exact belief propagation over a Markov model combined with an automaton (Pachet 2015/2026);
- conditioning on future constraints (Anticipation-RNN, 2017);
- distribution-aligned resampling (ASAp 2024, MCMC 2025);
- beam search with a critic (BebopNet 2020);
- interleaved infilling (Anticipatory Music Transformer 2023/24).

Optimizing a harmony reward alone collapses diversity (GAPT, ICLR 2026). A **planner (landing notes) → realizer (lick retrieval) → scorer** split avoids both failure modes.

### Cited Findings
- (2024, NeurIPS) Park, Wang, Berg-Kirkpatrick, Polikarpova & D'Antoni, "Grammar-Aligned Decoding". Grammar-constrained decoding "can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM". **ASAp** guarantees grammaticality while matching the LLM's distribution conditioned on the grammar. — [NeurIPS 2024 paper PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2bdc2267c3d7d01523e2e17ac0a754f3-Paper-Conference.pdf)
- (2025, NeurIPS) Gonzalez, Vaidya, Park, Ji, Berg-Kirkpatrick & D'Antoni, "Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective":
  - Method: Metropolis–Hastings that starts from a constrained-decoding sample, proposes by truncating the sequence and re-completing it, and accepts using model likelihoods.
  - Result: **2.11×–8.70× lower KL divergence** than constrained decoding or ASAp after 10 steps.
  — [mlanthology](https://mlanthology.org/neurips/2025/gonzalez2025neurips-constrained/); [alphaXiv overview](https://www.alphaxiv.org/overview/2506.05754v1)
- (2017/09) Hadjeres & Nielsen, Anticipation-RNN. The model conditions on a summary of upcoming positional constraints "to generate notes with a correct distribution". Sampling costs the same as an ordinary RNN, which suits real-time interactive use. — [arXiv 1709.06404](https://arxiv.org/abs/1709.06404)
- (2023 preprint; 2024 TMLR) Thickstun et al., Anticipatory Music Transformer:
  - It interleaves events and controls ("anticipation") so it can do infilling, including accompaniment.
  - It was trained on Lakh MIDI.
  - The authors claim human evaluators found its accompaniments comparable to human ones.
  — [Paper PDF](https://johnthickstun.com/assets/pdf/anticipatory-music-transformer.pdf); [mlanthology TMLR 2024](https://mlanthology.org/tmlr/2024/thickstun2024tmlr-anticipatory/)
- (2020, ISMIR) BebopNet:
  - It uses beam search to optimize a user-specific preference metric learned from that user's ratings. A harmony score variant prefers chord-scale-coherent notes.
  - It is a working example of "generate many candidates, rerank with a critic" for bebop.
  — [ISMIR 2020 PDF](https://archives.ismir.net/ismir2020/paper/000132.pdf); **[read directly]** [GitHub](https://github.com/shunithaviv/bebopnet-code)
- (2020, ISMIR) Wu & Yang, "The Jazz Transformer on the Front Line" (Transformer-XL trained on WJazzD). Training loss got low, but the **listening test showed a clear gap** between generated and real pieces. The authors built metrics for pitch use, groove, chord-progression consistency and structure to diagnose why. — [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307)
- (2025/11 arXiv; ICLR 2026) GAPT, "Generative Adversarial Post-Training Mitigates Reward Hacking in Live Human-AI Music Interaction":
  - With only a coherence reward, models produce "**harmonically coherent yet unnatural progressions with repetitive, trivial, and low-coverage** chord choices".
  - Adding a co-evolving discriminator restores diversity.
  - Musicians said GAPT "catches my key and chord changes faster".
  — [arXiv 2511.17879](https://arxiv.org/html/2511.17879v3); [mlanthology ICLR 2026](https://mlanthology.org/iclr/2026/wu2026iclr-generative/)
- (2026/06) Pachet's repulsive pattern control targets the repetition "tunnels" that appear under constrained Markov generation; see Q1. — [arXiv 2606.24911](https://arxiv.org/abs/2606.24911)
- (2017, ICCC) Impro-Visor's product of experts (interval LSTM × chord LSTM) is an early *soft* harmonic constraint: a multiplicative expert, not a mask. — [PDF](https://www.danieldjohnson.com/files/2017jazzproduct.pdf)
- (2025/10) Keating & Casey (Dartmouth), "A Graph Engine for Guitar Chord-Tone Soloing Education". Each chord in the progression gets chord-tone arpeggio nodes. Edge weights encode optimal transition tones between consecutive chords, and the shortest path yields a chord-tone line. This is effectively a **landing-note planner**. Code is on GitHub per the paper. — [arXiv 2510.19666](https://arxiv.org/abs/2510.19666)
- (2022) Frieler's model needed a *planner-level* rhythm constraint (line MLUs fixed to 8ths/16ths) to avoid incoherent rhythm. — [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87)
- (2025) ImprovNet, "Generating Controllable Musical Improvisations with Iterative Corruption Refinement" (title/abstract-level only; a corrupt-and-refine alternative to left-to-right decoding). — [arXiv 2502.04522](https://arxiv.org/pdf/2502.04522)

### Inferences
- **Likely mechanism of the user's repetition collapse (an inference, not measured in the literature for jazz):**
  - In a model trained on bebop, the top-k continuations after a chord tone often include chromatic approach and passing tones.
  - An out-of-scale mask removes them, and renormalization puts their mass on the in-scale tokens that remain.
  - The previous pitch is always "legal" and already moderately likely, so it gains disproportionately.
  - Grammar-Aligned Decoding is the formal statement of this distortion.
  - The fix is not a stronger mask. It is to make constraints **metric- and resolution-aware**, which bans far fewer tokens (see Q3), and to sample in a way that preserves the distribution.
- A practical hybrid for half-bar blocks (~4–8 notes per block at 3–5 notes/s):
  1. **Planner.** For each upcoming chord, choose landing targets: guide tones (3rd/7th) or chord tones on beats 1/3 and at chord-change points. Use a shortest path with a voice-leading cost, as in the Keating & Casey graph engine, and add breath/rest slots.
  2. **Realizer.** Fill each gap between landings by **retrieving real lick fragments** whose transposed start and end match the planned pitch classes and metric positions. If nothing matches, fall back to rule templates: Barry Harris descending scale, approach, enclosure.
  3. **Scorer.** Score whole half-bar candidates with the 13M Transformer's log-likelihood, a legality validator (Q3) and a repetition/self-reuse penalty, using beam search or rerank-N as in BebopNet. Do not sample the model token by token under masks.
  4. **Optional exact sampling.** Encode meter, landing and forbidden-repeat constraints as automata and sample with `vo-regular-bp` (exact, CPU).
- A soft alternative if the user keeps per-token neural sampling: a log-linear product of experts (model × legality expert) that penalizes only *strong-beat, unresolved* non-chord tones, combined with lookahead or backtracking over at most 1 beat. This keeps chromatic passing tones available.
- Because the candidate space per half-bar is small, generate-and-filter with exhaustive validation is affordable within 0.4 s on CPU (estimate, not benchmarked).

### Gaps
- I found no published jazz-specific measurement of repetition increase under hard pitch masks. The mechanism above is an inference by analogy to Grammar-Aligned Decoding and GAPT.
- I found no 2022–2026 "plan landing notes, then infill" jazz soloist with **jazz-musician listening evaluation**. The graph engine is educational and reports no listening test.
- I found no jazz-specific evaluation of Anticipatory Music Transformer or MMM infilling for bebop lines.
- Web pages claiming a "2024 MIT Media Lab study with 42 improvisers" and a "Duke Ellington Challenge" listening test appeared only on alibaba.com "product-insights" pages, which look machine-generated and cite no primary source. I **excluded them as unreliable**. — [example page](https://www.alibaba.com/product-insights/is-ai-generated-jazz-improvisation-passing-blind-listening-tests-or-still-betraying-statistical-patterns.html)

## Q3. Musicology of bebop line "legality": why some chromatic notes sound intentional and others sound wrong, and what rule sets operationalize this

### Takeaway
Pedagogy and the corpus studies I could find agree that chromaticism sounds intentional when **chord tones (especially 3rds and 7ths) fall on strong beats**, and when **non-chord tones sit on off-beats and resolve by step**: a half-step approach, an enclosure, or a bebop-scale passing tone that lands on a chord tone on the next beat. Avoid notes, such as the natural 11 over a major or dominant chord with a natural 3rd, appear only as passing tones. This translates into a small validator that is much less restrictive than an out-of-scale mask. Large-corpus analyses (WJazzD) show that scale choice depends on chord function and key context. I could not retrieve exact "chord-tone ratio by metrical position" numbers.

### Cited Findings
- **Barry Harris half-step rules:**
  - Added chromatic half-steps act as **rhythmic placeholders** so that chord tones 1-3-5-7 land on the downbeats in descending eighth-note lines.
  - The basic rule over a dominant is an extra half step between root and b7.
  - Where no half step is available (E–F, B–C), the line goes up a scale step before the half step.
  — [CEA notes on the method of Barry Harris](https://irfu.cea.fr/Pisp/frederic.galliano/Zique/barry_harris.html); [Fertile Minds: Barry Harris half-step rules](https://fertilemindsjazzacademy.com/barry-harris-major-scale-half-step-rules/); [jazzguitar.be forum: David Baker bebop scales vs. Barry Harris half-steps](https://www.jazzguitar.be/forum/improvisation/101660-david-baker-bebop-scales-barry-harris-half-steps.html)
- **Bebop scales:** the added passing tone means that when the scale is played in order, chord tones fall on on-beats and non-chord tones on off-beats. — [Wikipedia: Bebop scale](https://en.wikipedia.org/wiki/Bebop_scale)
- **Approach and enclosure definitions (pedagogy):**
  - A chromatic approach tone sits a half step above or below a target chord tone and resolves into it.
  - An enclosure surrounds the target from both sides, chromatically or diatonically, before landing on it.
  - Targets are placed on strong beats.
  — [Anton Schwartz: Approaches & Enclosures (2019)](https://antonjazz.com/2019/07/approaches-enclosures/); [Learn Jazz Standards: bebop scales](https://www.learnjazzstandards.com/blog/learning-jazz/jazz-theory/use-bebop-scales-like-pro/); [Jazz Etudes: chromatic enclosures](https://www.jazzetudes.net/post/chromatic-enclosures-for-jazz-stop-playing-straight-bebop-lines)
- **Keller's pedagogy (2007), behind Impro-Visor's categories:** in scale fragments, hit chord tones on the beat and put chromatic or unessential notes off the beat. — [Keller, "How to Improvise Jazz Melodies" (PDF)](https://www.cs.hmc.edu/~keller/jazz/improvisor/HowToImproviseJazz.pdf). The grammar's "A" category is defined as a tone that chromatically approaches a chord tone (C) or color tone (L). — [Gillick, Tang & Keller 2010 (PDF)](http://ai.stanford.edu/~kdtang/papers/cmj10-jazzgrammar.pdf)
- **Avoid notes:**
  - An avoid note is a tension a half step (minor 9th) above a chord tone, too dissonant to emphasize.
  - It is avoided, used as a passing tone, or altered, e.g. 11 raised to #11.
  - Over dominant chords "anything goes" more often. b9, #9 and b13 are standard colors on dominants because the chord is meant to be unstable.
  — [Wikipedia: Avoid note](https://en.wikipedia.org/wiki/Avoid_note); [The Jazz Piano Site: Avoid Notes](https://www.thejazzpianosite.com/jazz-piano-lessons/jazz-improvisation/avoid-notes/); [Interactive Chord Finder (2026): which tensions work](https://interactivechordfinder.com/articles/2026021507-extended-chords-jazz-harmony/)
- **General voice-leading principle:** passing and neighbor tones usually fall on weak beats and chord tones on strong beats. — [Tymoczko, MUS105 handout (2010)](https://dmitri.mycpanel.princeton.edu/files/pdfs/MUS105handouts.pdf)
- **(1995, Music Perception 12(4)) Järvinen, "Tonal Hierarchies in Jazz Improvisation":** statistical analysis of 18 bebop improvisations on Rhythm Changes found that the **metrical structure was used to emphasize or de-emphasize tones according to their tonal function**. This is the canonical corpus evidence for "chord tones on strong beats". — [Music Perception](https://mp.ucpress.edu/content/12/4/415)
- **WJazzD chord-scale study, Moss et al., "Inside or Outside: The Use of Chord, Scale and Chromatic Tones in Jazz Solo Improvisations":**
  - The corpus is 456 solos.
  - Scale choice over diatonically interpretable chords depends on local scale degree and key context.
  - Inherently non-diatonic chords allow more freedom.
  - Case studies cover tritone substitutes, diminished chords, and the 6th degree in pre-dominant chords.
  — [fabian-moss.de talk page](https://fabian-moss.de/talk/inside-or-outside-the-use-of-chord-scale-and-chromatic-tones-in-jazz-solo-improvisations/)
- **WJazzD metrical annotation:** events carry a metrical weight (0 = sub-beat, 1 = weak beat, 2 = strong beat). Chord-tone-by-metric-position statistics can be computed directly from the open database. — [Jazzomat: Database format](https://jazzomat.hfm-weimar.de/dbformat/dbformat.html)
- **WBA atoms:** "approaches" form their own atom class in the Weimar Bebop Alphabet, i.e. a unit of bebop line construction. — [TISMIR 2022](https://transactions.ismir.net/articles/10.5334/tismir.87)
- **(2023) FiloBass**, a corpus study of jazz basslines: semitone approaches into strong-beat targets are common. This supports the same "approach → strong-beat target" schema in another jazz voice. — [arXiv 2311.02023](https://arxiv.org/pdf/2311.02023)
- **Playing "out":** Pachet describes side-slipping as the main, precisely defined bebop device for going outside. Mastering it, i.e. returning "the right way", is a mark of virtuosity. — [ResearchGate: Musical Virtuosity and Creativity](https://www.researchgate.net/publication/259383942_Musical_Virtuosity_and_Creativity)
- **Low-confidence item:** a search extract said that in Charlie Parker solos "on-beat diatonic tones comprised 37.1% of notes, off-beat diatonic 46.6%". The source page was ambiguous; it was likely a 2019 blog analysis and not peer reviewed. **Do not rely on this without checking.** — [jazzido blog (2019)](https://blog.jazzido.com/2019/04/13/data-driven-stylistic-analysis-of-bird-solos)

### Inferences
- **Proposed validator** (synthesized from the sources above; not a published rule set). It assumes an 8th-note grid in 4/4, with strong = beat onsets and extra weight on beats 1 and 3 and on chord-change points.
  - **R1 (strong-beat legality).** A note on a strong beat must be a chord tone (1-3-5-7), an available tension for the chord type (9, 13, #11 on dominant/major; 9, 11 on minor 7), or an altered dominant tension on a V chord. Avoid notes, such as natural 11 over major or dominant with a natural 3rd, or b9 over major/minor-7, are forbidden on strong beats.
  - **R2 (resolution).** An off-beat non-chord tone must move by step (≤2 semitones) to a note that satisfies R1 on the next strong beat. A *chromatic* non-chord tone must either continue in the same direction by a semitone (passing) or be part of an enclosure.
  - **R3 (enclosure template).** The patterns `[t+1 or t+2(diatonic), t−1, t]` or `[t−1, t+1, t]` are allowed when t lands on a strong beat and satisfies R1. The off-beat entries are exempt from R1.
  - **R4 (chord-change arrival).** The first strong-beat note at or after a chord change should be a guide tone (3rd/7th), or be reached through an R2/R3 approach. An anticipation by one 8th is allowed if the anticipated note is a chord tone of the *new* chord.
  - **R5 (outside).** Side-slip is allowed only as a whole-fragment transposition of ±1 semitone that returns to R1-legal notes within ≤1 beat. This is how "intentional outside" can be permitted without random wrong notes.
  - **R6 (idiom guards).** Same-pitch repeats should stay near the ~5–6% of real bebop the user measured. Rests ("breathing") belong at phrase ends.
- This validator bans far fewer notes than an out-of-scale mask: chromatic tones stay legal on off-beats when they resolve. It also matches Impro-Visor's C/L/A/S category logic. When used as a *filter or scorer over whole fragments* rather than a per-token mask, it should avoid the repetition collapse described in Q2.
- The validator is cheap to calibrate. Compute, on WJazzD bebop-era solos, the fraction of strong-beat notes that pass R1 and the fraction of off-beat non-chord tones that pass R2. Use these as target rates instead of zero. Real solos will violate the rules occasionally, so a 100% pass rate is itself un-idiomatic.

### Gaps
- I could not retrieve published numbers for "chord-tone ratio by metrical position" in WJazzD (Frieler / *Inside the Jazzomat*); the book PDF was blocked. They are computable from the open database.
- I found no corpus statistics on how often bebop players hit the natural 11 over a dominant 7th, or in which metric position.
- I found no published validator that operationalizes "sounds wrong" *and* was validated by listening tests. The R1–R6 set is a synthesis.
- I could not access the full texts of Coker, Baker or Bergonzi pattern books, so their pattern taxonomies are not itemized here.

## Q4. Solo–comp clashes and proven real-time comping approaches

### Takeaway
Semitone and minor-9th collisions are the roughest intervals perceptually. That is why the user's measured 12% of solo time in semitone clash with the comping sounds "wrong" even when each part is legal against the chord on its own. The lowest-risk real-time comping design is rule-based:
- rootless A/B voicings (3-5-7-9 / 7-9-3-5) kept around C3–C5 with minimal voice-leading motion;
- short stabs and Charleston-type rhythms instead of sustained pads;
- staying out of the soloist's register;
- a **solo-aware clash filter**. Because both parts are generated by the AI, the solo for the block is known before the comp is rendered.

Learned accompaniment systems (AccoMontage 1/2, ReaLchords, the 2026 diffusion/retrieval models) are pop-oriented and need GPU training. Impro-Visor and Band-in-a-Box show that pattern-plus-voicing engines are the proven practical approach.

### Cited Findings
- **(1965) Plomp & Levelt, roughness:**
  - Pure-tone pairs are maximally dissonant at about **a quarter of the critical bandwidth** and consonant beyond it.
  - The critical band is roughly a minor third except at low frequencies, where it is wider.
  - So close semitones are rough, and low-register intervals get muddy faster.
  — [Plomp & Levelt 1965 (PDF)](https://www.math.miami.edu/~armstrong/592sp15/Plomp_Levelt_1965.pdf); [Tuning-list explanation of the Plomp–Levelt curve](https://yahootuninggroupsultimatebackup.github.io/tuning/topicId_3530.html)
- **Jazz voicing rules:**
  - Avoid a semitone between the melody and the second voice.
  - The minor 9th between an avoid tension and a chord tone is "the harshest sound in the system" and at close spacing "starts reading as a mistake".
  - The exception is dominant chords, where b9, #9 and b13 are standard.
  — [Taming the Saxophone: block voicing](https://tamingthesaxophone.com/theory/arranging/jazz-blockvoicing); [Interactive Chord Finder (2026)](https://interactivechordfinder.com/articles/2026021507-extended-chords-jazz-harmony/)
- **Rootless voicings:**
  - Type A = 3-5-7-9 and Type B = 7-9-3-5. They alternate through ii–V–I so the inner voices barely move.
  - The style was popularized in the mid-to-late 1950s by Bill Evans, Red Garland and Wynton Kelly.
  - The natural 11 is left out; on dominants the 13 replaces the 5.
  - Evans often comped with only 3rd, 7th and 9th.
  — [piano.org: Rootless voicings type A/B](https://piano.org/theory/rootless-voicings/); [Jazzedge: rootless voicings like Bill Evans](https://jazzedge.academy/how-to-play-rootless-voicings-like-bill-evans/); [Piano With Jonny: rootless voicings guide](https://pianowithjonny.com/piano-lessons/rootless-voicings-for-piano-the-complete-guide/)
- **Register:** Mark Levine (*The Jazz Piano Book*) recommends the left-hand pinky between C3 and C4, with the top note between middle C and C5 (overall about C3–A4). Lower voicings get muddy. — [Learn Jazz Standards: left-hand ii-V-I voicings](https://www.learnjazzstandards.com/blog/left-hand-piano-voicings-for-ii-v7-is/); [The Jazz Piano Site: rootless voicings](https://www.thejazzpianosite.com/jazz-piano-lessons/jazz-chord-voicings/rootless-voicings/)
- **Comping rhythm and role:**
  - The Charleston rhythm (dotted quarter + eighth) is a staple comping rhythm; Red Garland used it constantly with a light touch.
  - A "stab" is a short rhythmic hit of the chord.
  - Stay out of the soloist's range: when the soloist plays high, comp in the middle register.
  - The comper must never get in the soloist's way.
  — [PianoGroove: comping voicings & rhythms](https://www.pianogroove.com/jazz-piano-lessons/comping-voicings-rhythms/); [Jazz-Library: comping guide](https://jazz-library.com/articles/comping/); [MasterClass: how to comp](https://www.masterclass.com/articles/how-to-comp-when-playing-jazz-music)
- **Impro-Visor style accompaniment:** it uses both pre-planned and generated voicings, and **chooses voicings that fit the specified range and voice-lead from the previous voicing**. Comping and bass patterns use a textual pattern notation. — [Impro-Visor tutorial](https://www.cs.hmc.edu/~keller/jazz/improvisor/ImproVisorTutorial4.htm); [Style Editor tutorial (PDF)](https://www.cs.hmc.edu/~keller/jazz/improvisor/StyleEditorTutorial.pdf). **[read directly]** The GitHub README confirms style accompaniment, voicing tools and a "fluid voicing editor". — [GitHub](https://github.com/Impro-Visor/Impro-Visor)
- **Band-in-a-Box:**
  - MIDI styles contain chord patterns played by the keyboardist who authored the style.
  - RealTracks are recorded musicians.
  - There are more than 1,225 styles, including 2-handed piano styles with left-hand comping.
  - The algorithm is proprietary.
  — [PG Music forum: RealTracks vs MIDI](https://www.pgmusic.com/forums/ubbthreads.php?ubb=showthreaded&Number=145193); [PG Music: Band-in-a-Box Pro for Mac](https://www.pgmusic.com/bbmac.packages.pro.htm)
- **iReal Pro:** the workflow is the same as Band-in-a-Box (type chords, pick a style, play). I found no public technical description of its accompaniment engine. — [pract.is comparison](https://pract.is/blog/ireal-pro-alternatives-jazz-practice-backing-tracks)
- **(2020, ISMIR) "Chord Jazzification":** splits chord-symbol realization into **coloring** (which tensions) and **voicing**, with a dataset of pop-jazz interpretations. — [ISMIR 2020 PDF](https://archives.ismir.net/ismir2020/paper/000090.pdf)
- **AccoMontage and AccoMontage2:**
  - (2021, ISMIR) AccoMontage does accompaniment arrangement by **phrase retrieval** plus style transfer. — [arXiv 2108.11213](https://arxiv.org/pdf/2108.11213)
  - (2022, ISMIR) AccoMontage2 adds a harmonizer that balances note-wise dissonance, phrase-template matching and whole-piece coherence. It offers texture styles with voicing-density and rhythm-complexity levels, and it is open source. Both are pop/folk-oriented. — [arXiv 2209.00353](https://arxiv.org/pdf/2209.00353)
- **(2026) Recent pop-oriented accompaniment generators (not jazz):** style planning + dataset-aligned pattern retrieval — [arXiv 2602.15074](https://arxiv.org/html/2602.15074v1); D3PIA discrete diffusion from lead sheets — [arXiv 2602.03523](https://arxiv.org/pdf/2602.03523)
- **(2025) ReaLchords/ReaLJam** produce *chord-symbol* accompaniment to a melody. They are trained on Hooktheory pop data (19,086 songs), and training needs 48 GB GPUs. They are not jazz comping models. — **[read directly]** [GitHub lukewys/realchords-pytorch](https://github.com/lukewys/realchords-pytorch)
- **(2011) Shimon** (Hoffman & Weinberg, robotic marimba): improvises over a chord progression using **rules from canonical jazz improvisation textbooks** plus Markov style models of jazz masters, with **anticipatory beat-matched action** for synchronization. — [Hoffman & Weinberg, Autonomous Robots 2011 (PDF)](http://guyhoffman.com/publications/HoffmanAuRo11.pdf)
- **(2013) Reflexive Looper** "other members" principle: when the human solos, the system supplies bass and chords. — [Goldsmiths PDF](https://research.gold.ac.uk/9888/1/Reflexive%20Loopers%20for%20Solo%20Music%20Improvisation.pdf)

### Inferences
- **Why 12% clash time happens and how to cut it:** clash time is the product of overlap duration and collision probability, so sustained comp chords under an eighth-note line with off-beat chromatic passing tones collide often by construction. Ranked by expected impact:
  1. **Short stabs.** Duration ≤ an 8th or a quarter, on the Charleston pattern (beat 1 and the "and" of 2), on the "and" of 4 anticipating the next chord, or in solo rests. This reduces overlap time directly.
  2. **Solo-aware note removal.** For each comp hit, drop or re-voice any comp note within 1 semitone (mod 12, in close register) of a solo note sounding at that moment or within the next 8th. Exception: dominant altered tensions when the solo note is a chord tone.
  3. **Register separation.** Keep the top comp note at least a minor 3rd below the solo's lowest note in the block (critical-band logic). Keep the left-hand voicing roughly between C3 and C5.
  4. **Rootless A/B choice by voice leading.** Pick the inversion that minimizes total semitone motion from the previous voicing (Impro-Visor's approach), *subject to* rules 2 and 3.
  5. **Hit timing.** Place comp hits where the solo plan has strong-beat chord tones. The Q3 validator guarantees those exist, so comp notes coincide with consonant solo notes, not passing tones.
- Because the soloist is also the AI, the comp for block *k* can be rendered **after** the solo for block *k* is fixed within the same 0.4 s budget. That makes rule 2 exact, not predictive. This is the main architectural advantage over human-accompanying systems.
- For one person on a Mac, a rule engine is the robust choice: voicing table, voice-leading search over about 2–4 candidate voicings, rhythm templates, and a clash filter. The learned alternatives are pop-trained, GPU-trained, or proprietary.

### Gaps
- I found no learned **jazz piano comping** model from 2022–2026 with public weights *and* jazz-musician evaluation. PiJAMA (TISMIR, piano jazz with automatic MIDI annotations) could provide training data, but I saw it only at title level. — [TISMIR PiJAMA](https://transactions.ismir.net/articles/10.5334/tismir.162)
- iReal Pro's and Band-in-a-Box's comping-generation algorithms are not publicly documented.
- I found no published measurement of solo–comp semitone-clash rates in real jazz recordings to use as a target. It could be computed from multitrack jazz-piano transcriptions if available.
- The specific dissonance numbers depend on piano timbre and register. Plomp–Levelt applies to pure tones and is only approximate for piano partials.

## Q5. Real-time interaction with a human: late or changed chords, anticipation and commit horizons

### Takeaway
The best-documented protocol is ReaLJam's **lookahead + commit** scheme (CHI EA 2025):
- The agent continuously plans future beats.
- Everything inside a user-adjustable **commit window** is frozen.
- Everything beyond it can be revised when new input arrives.
- Cached future output is played while the server is slow.

ImproteK adds the idea of **rewriting anticipations** when the scenario or input changes. The 2026 StreamMUSE study ties real-time performance to music quality. For the user's half-bar blocks: commit the sounding block, prepare the next block speculatively, and re-plan only uncommitted material when a chord is typed.

### Cited Findings
- **(2025/02 arXiv; CHI EA '25, Yokohama) ReaLJam:**
  - **Lookahead** is the number of beats ahead in which the user sees upcoming chords.
  - Chords inside the lookahead up to the **commit time** are committed and cannot change. Beyond the commit time the agent may update its predictions, and uncommitted chords are drawn semi-transparently.
  - The commit time is user-adjustable, trading advance knowledge against responsiveness to the melody.
  - Synchronization is robust to server latency because the client plays cached future chords while waiting.
  - The client continuously sends history (melody plus past chords). The server predicts upcoming beats, and the client schedules them.
  - In the user study, participants cared a lot about interface settings but **disagreed about which settings were best**.
  — [arXiv 2502.21267](https://arxiv.org/html/2502.21267); [ACM DL](https://dl.acm.org/doi/10.1145/3706599.3720227)
- **[read directly]** The realchords-pytorch implementation (MIT) exposes **initial beats of silence (default 8)**, **lookahead beats** and **commit beats**. It has an **MLX backend for Apple Silicon** (`realjam-start-server --mlx`), ONNX acceleration, and MIDI or computer-keyboard input. — [GitHub lukewys/realchords-pytorch](https://github.com/lukewys/realchords-pytorch)
- **(2025/11 arXiv; ICLR 2026) GAPT:** musicians reported the post-trained model "catches my key and chord changes faster". Adaptation speed to the human's changes is a measurable target. — [arXiv 2511.17879](https://arxiv.org/html/2511.17879v3)
- **(2026/06) StreamMUSE** ("Real-Time Language Model Jamming", arXiv 2606.11886):
  - The client sends high-frequency inference requests based on the most recent input and receives outputs synchronized to an external clock.
  - It was evaluated locally, on a local server and on a remote server.
  - The authors report "a consistent correspondence between system real-time performance and music quality". It is open source.
  — [arXiv 2606.11886](https://arxiv.org/abs/2606.11886)
- **ImproteK's reactive architecture** combines long-term planning from the scenario with reactive listening. Anticipations are generated from future scenario labels and can be revised by reactive inputs. — [ResearchGate: ImproteK](https://www.researchgate.net/publication/309575073_ImproteK_Introducing_Scenarios_into_Human-Computer_Music_Improvisation); [ResearchGate figure: combining long-term planning and reactive listening](https://www.researchgate.net/figure/Combining-long-term-planning-and-reactive-listening_fig1_319528165)
- **(2011) Shimon** uses anticipatory, beat-matched action to stay synchronized with a human pianist. — [Hoffman & Weinberg (PDF)](http://guyhoffman.com/publications/HoffmanAuRo11.pdf)
- **(2012) Impro-Visor** supports continuous, non-repeating generation and phrase trading in real time. — [ICCC 2012 PDF](https://computationalcreativity.net/iccc2012/wp-content/uploads/2012/05/222-Keller.pdf)
- **(2026)** "A Design Space for Live Music Agents" (arXiv 2602.05064; title-level only). — [arXiv](https://arxiv.org/pdf/2602.05064)

### Inferences
- Mapping onto the user's constraints (half-bar = 0.94 s at 128 BPM; ~0.4 s p99 prep; chord change reflected within about one block):
  - **Commit horizon = the currently sounding block.** Block *k+1* is prepared speculatively using the last typed chord. Optionally predict the next chord with ii–V "brick" heuristics, as in Impro-Visor's roadmap.
  - **Re-plan deadline.** If a chord key arrives before (boundary − 0.4 s), re-plan block *k+1* with the new label. If it arrives later, apply it at block *k+2*, or patch only the *landing note* of block *k+1*. A landing-note-first plan makes a partial patch possible: swap the target and re-select one ending fragment.
  - **Grace window.** Treat a chord typed slightly *after* a boundary (e.g. within an 8th note) as intended for that boundary, and correct the comp hit immediately. Correct the solo from the next strong beat.
- Retrieval-based lines suit late changes well. Each fragment has known start and end pitch-class roles, so only the remainder of the block needs re-selection. A token-by-token neural sampler would need to regenerate from scratch.
- ReaLJam's finding that users disagree on the best commit/lookahead settings suggests exposing "commit = 1 block vs. 2 blocks" as a user setting rather than fixing it.

### Gaps
- I could not retrieve ReaLJam's numeric default for commit beats, the participant count or quantitative user-study results (arXiv was blocked for fetch).
- I found no published system that specifically handles chords **typed by a human on a computer keyboard** in a DAW with half-bar latency. The design above is an inference from ReaLJam and ImproteK.
- I did not find StreamMUSE's latency numbers or its specific synchronization policy.
