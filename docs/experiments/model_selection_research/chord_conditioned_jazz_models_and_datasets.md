# Chord-conditioned jazz (bebop) solo generation models and chord-annotated jazz datasets (research as of 2026-10-03)

Method note: The egress proxy blocked arxiv.org, archives.ismir.net, program.ismir2020.net, transactions.ismir.net, openreview.net, zenodo.org, huggingface.co, medium.com, eurecom.fr, qmul.ac.uk, jazzomat.hfm-weimar.de, github.io, and mdpi.com. Facts about those papers come from search-engine snippets of the cited pages, and some snippets may be paraphrased. They are marked "(snippet)" where it matters. Facts marked "(code-verified)" come from shallow `git clone` of the public GitHub repos, run 2026-10-03, so they are primary evidence about what the released code actually does.

## Q1. BebopNet (Haviv Hakimi, Bhonker, El-Yaniv; ISMIR 2020): chord conditioning, data, Transformer-XL, personalization, evaluation, repo state

### Takeaway
BebopNet is a monophonic, saxophone-derived bebop next-note model. Every note token carries the embedding of exactly 4 chord pitches, and every chord is forced into a 4-note seventh chord. An optional beam search reranks candidates with either a per-user preference regressor (trained on continuous like/dislike feedback from 4 amateur jazz musicians) or a chord-tone "harmony" score. The code (MIT, with a pretrained Transformer-XL checkpoint) is frozen at 2020 dependencies (torch 1.1.0), and the authors' published evaluation includes no blind listening test by jazz experts.

### Cited Findings
- **Paper and award.** "BebopNet: Deep Neural Models for Personalized Jazz Improvisations", ISMIR 2020. The project page says it won a Best Research Award at ISMIR 2020 — [project index.md](https://github.com/shunithaviv/bebopnet/blob/master/index.md); [ISMIR PDF](https://archives.ismir.net/ismir2020/paper/000132.pdf)
- **Task.** The model predicts the next note from past notes, the past harmonic progression, and the forthcoming harmony. Once trained, it can solo over any chord progression, including ones not in the training set — [ISMIR PDF (snippet)](https://program.ismir2020.net/static/final_papers/132.pdf); [Medium/TDS article (snippet)](https://medium.com/data-science/bebopnet-neural-models-for-jazz-improvisations-4a4d723d0b60)
- **Input representation.** The input is the note (pitch, duration) plus context (offset in the bar, chord). Pitch, duration, and offset each have learned embeddings. "The chord is encoded by using the embedding of the pitches comprising it" — [Medium/TDS (snippet)](https://medium.com/data-science/bebopnet-neural-models-for-jazz-improvisations-4a4d723d0b60)
- **(code-verified) Per-note vector layout:**
  - pitch (0–127, plus REST=128 and EOS=129)
  - duration (1.0 = quarter note)
  - offset within the bar on an 8th-note-based grid
  - chord root
  - 13 scale-pitch indicators
  - 13 chord-pitch indicators
  - chord-type index

  The data builder asserts `sum(chord_notes) == 4`, so every chord has exactly 4 pitches — [gather_data_from_xml.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/A_data_prep/gather_data_from_xml.py)
- **(code-verified) How the chord reaches the network.** In the Transformer-XL model, the 4 chord pitches are embedded with the same pitch embedding table and concatenated to the note embedding (`word_emb = torch.cat((word_emb, chord_emb), 2)`). This is per-note chord input, not cross-attention and not separate chord tokens. An alternative `chord_bias` path exists, but the released config sets `"chord_bias": false`. "No chord" is encoded by repeating a special pitch 4 times — [mem_transformer.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py); [model.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/B_next_note_prediction/model.py); [args.json](https://github.com/shunithaviv/bebopnet-code/tree/master/training_results/transformer/model)
- **(code-verified) Chord simplification.** `ensure_4_notes()` reduces every chord to a 4-note seventh chord:
  - A plain major triad is mapped to a dominant seventh: `kind == 'major'` → `ChordSymbol(kind='dominant')`.
  - Triads without a 7th get a "-seventh" added.
  - 9th, 11th, and 13th chords are reduced to the corresponding seventh chord.

  [vectorXmlConverter.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/utils/music/vectorXmlConverter.py)
- **(code-verified) Released Transformer-XL config:**
  - n_layer=4, n_head=8, d_model=400, d_head=50, d_inner=1028
  - mem_len=64, tgt_len=64
  - pitch embedding 64-d, duration embedding 64-d, offset embedding 16-d
  - pitch vocabulary 130, duration vocabulary 121, offset vocabulary 48
  - trained with fp16 and the Ranger optimizer, max_step 500k

  The shipped `model.pt` is 26.3 MB, and `optimizer.pt` is 52.6 MB — [train_model.yml / args.json](https://github.com/shunithaviv/bebopnet-code/tree/master/training_results/transformer/model)
- **Data.** 284 professionally transcribed solos, mostly bebop saxophonists: Charlie Parker, Sonny Stitt, Cannonball Adderley, Dexter Gordon, Sonny Rollins, Stan Getz, Phil Woods, and Gene Ammons. Only 4/4 solos whose transcription includes chords were used. The files are MusicXML purchased from saxsolos.com — [ISMIR PDF (snippet)](https://archives.ismir.net/ismir2020/paper/000132.pdf); [Medium/TDS (snippet)](https://medium.com/data-science/bebopnet-neural-models-for-jazz-improvisations-4a4d723d0b60)
- **Training data is not in the repo.** The repo ships only 36 lead-sheet XMLs ("heads"), and these include pop songs (Despacito, Dance Monkey). The paid saxsolos.com transcriptions are absent. (code-verified: `resources/xmls`) — [bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- **Pipeline.** Five steps:
  1. supervised language model on the corpus
  2. generation
  3. high-resolution user preference labeling
  4. user-preference metric learning
  5. optimized generation with beam search

  [ISMIR PDF (snippet)](https://archives.ismir.net/ismir2020/paper/000132.pdf); [README](https://github.com/shunithaviv/bebopnet-code)
- **Preference labels.** Users listen to computer-generated solos and give continuous good/bad feedback in real time through a digital variant of a Continuous Response Digital Interface (CRDI). A recurrent regression model is trained to predict that feedback. Selective prediction lets the preference model abstain when it is not confident — [ISMIR PDF (snippet)](https://program.ismir2020.net/static/final_papers/132.pdf); [Lacuna summary](https://lacuna.tiptreesystems.com/work/bebopnet-deep-neural-models-for-personalized-jazz-improvisations/wrk_abbbe983500f2bb9558a93b0d51d767d)
- **Who gave preferences.** The pipeline was applied to 4 users, all amateur jazz musicians with a few years of improvisation experience for whom music is not the main profession. The authors describe themselves as amateur jazz musicians — [ISMIR PDF (snippet)](https://archives.ismir.net/ismir2020/paper/000132.pdf); [Technion blog](https://www.technion.ac.il/en/blog/and-all-that-jazz/)
- **(code-verified) Personalization only works with the LSTM.** The README says: "The current version of user model training currently only supports the LSTM generative model." The pretrained checkpoint is the Transformer-XL, so it cannot be personalized out of the box — [README](https://github.com/shunithaviv/bebopnet-code)
- **(code-verified) Harmony-guided beam search** (`--score_model harmony`). `HarmonyScoreInference` scores each candidate step as (pitch class ∈ the 4 chord pitches) ÷ note duration, averages this per beam, and keeps the top-k distinct beams. It calls `.cuda()` unconditionally — [music_utils.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/utils/music_utils.py); [music_generator.py](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/B_next_note_prediction/music_generator.py)
- **Harmony-metric beam search works by design.** The project page has "harmony-guided" samples that use a harmonic coherence metric instead of the user preference score. The paper reports that optimized solos maximize this metric — [project index.md](https://github.com/shunithaviv/bebopnet/blob/master/index.md); [ISMIR PDF (snippet)](https://program.ismir2020.net/static/final_papers/132.pdf)
- **Plagiarism analysis.** Generated solos have an average largest common subsequence of 4.4 notes against the corpus. A plagiarism AUC metric gave 0.713 for BebopNet versus 0.680–0.746 for famous jazz musicians — [ISMIR PDF (snippet via search)](https://archives.ismir.net/ismir2020/paper/000132.pdf)
- **Samples.** The project page offers in-sample progressions, out-of-sample progressions, diversity samples ("Recorda Me" for user 4), per-user personalized samples, harmony-guided samples, and pop-song samples, across 18+ standards (e.g., Giant Steps, ATTYA) — [project index.md](https://github.com/shunithaviv/bebopnet/blob/master/index.md)
- **(code-verified) Repo status:**
  - License: MIT.
  - Last commit: 2020-11-09, a README edit.
  - Pinned dependencies: torch==1.1.0, torchvision==0.3.0, music21==5.1.0, numpy==1.17.3, pygame==1.9.6, tensorboard==2.0.0.

  [LICENSE / requirements.txt](https://github.com/shunithaviv/bebopnet-code)

### Inferences
- **What BebopNet does that the user's model doesn't.** It supplies the current chord as an input feature on every note during training. The user's Music Transformer only sees guide-tone voicings occasionally in the prompt. This is direct precedent for the user's hypothesis: chord-melody coupling has to be trained in through per-step conditioning, not prompted.
- **Triad pitfall.** Because plain major triads are mapped to dominant 7ths, typing "C" (triad) on the keyboard would make BebopNet solo over C7, so its b7 (Bb) would clash with a Cmaj7-type comp. For real-time use, send explicit 7th-chord symbols or patch `ensure_4_notes`.
- **"Harmony" beam score is only chord-tone membership.** It has no avoid-note or scale model and no strong-beat weighting, and dividing by duration favors short chord tones. It reduces clashes but does not encode bebop approach-note or enclosure logic, much like the user's own failed hard masks.
- **Running it today** (inferred from pins, not tested) needs a legacy environment (Python 3.6/3.7-era, torch 1.1 + CUDA 9/10) or porting to modern PyTorch. The hard-coded `.cuda()` in the harmony scorer requires a GPU or a small patch.
- **Idiom.** All training data is saxophone, so the output idiom is horn-like single lines (breath-length phrases, sax range), not pianistic.

### Gaps
- Exact Transformer-XL parameter count is not reported in what I could access. From the 26.3 MB `model.pt` it is roughly 6–7M parameters if stored as fp32; this is unverified.
- Not found: the number of solos each user labeled, the preference-model accuracy and AUC numbers, and any blinded listening/Turing test with outside jazz experts. The paper PDF was blocked, so I could only confirm the plagiarism and harmony-metric evaluations.
- Inference speed and real-time latency are not reported.
- No credible public follow-up or fork found that retrains BebopNet on piano data. Searches for piano adaptations returned nothing citable.

## Q2. MINGUS (Madaghiele, Lisena, Troncy; ISMIR 2021)

### Takeaway
MINGUS is two separate causal Transformers, one predicting pitch and one predicting duration. Each note is conditioned by concatenating embeddings of the current chord (4 pitches), the next chord (4 pitches), the bass pitch, the beat, and the offset. It is trained on WJazzD and Nottingham, reports MGEval and perplexity comparisons to BebopNet and a BiLSTM (SeqAttn), and ships pretrained weights under LGPL-3.0. Its human-evaluation results were not retrievable.

### Cited Findings
- **What it is.** A Transformer-based "Seq2Seq" architecture for monophonic jazz lines. Two dedicated models handle pitch and duration, and prediction uses chords (current and following), bass line, and position in the measure. Datasets are the Weimar Jazz Database and Nottingham — [ISMIR 2021 PDF (snippet)](https://archives.ismir.net/ismir2021/paper/000051.pdf); [GitHub](https://github.com/vincenzomadaghiele/MINGUS)
- **(code-verified) Architecture:**
  - Separate `nn.Embedding` tables for pitch, duration, bass, beat, and offset.
  - Current chord = 4 pitches looked up in the shared pitch embedding, then `nn.Linear(4*pitch_embed_dim, chord_encod_dim)`. The next chord gets its own linear encoder.
  - All features are concatenated, projected to `ninp`, given positional encoding, and passed through `nn.TransformerEncoder` with a square subsequent (causal) mask.
  - Separate pitch-model and duration-model checkpoints exist.

  [MINGUS_model.py](https://github.com/vincenzomadaghiele/MINGUS/blob/master/B_train/MINGUS_model.py)
- **(code-verified) Pretrained weights** are in the repo: `B_train/models/{pitchModel,durationModel}/MINGUS COND I-C-NC-B-BE-O Epochs {10,100}.pt` (conditioning flags I-C-NC-B-BE-O). WJazzD is preprocessed from its CSV export — [repo tree](https://github.com/vincenzomadaghiele/MINGUS)
- **(code-verified) License and activity.** LGPL-3.0. Last commit 2024-07-14, README edits — [LICENSE](https://github.com/vincenzomadaghiele/MINGUS)
- **Comparative results.** Compared against SeqAttn (a chord-conditioned BiLSTM) and BebopNet on WJazzD using perplexity/accuracy and MGEval. Performance was similar to BebopNet, and MINGUS was "largely better" on the pitch-class transition matrix and total pitch-class histogram — [ISMIR 2021 PDF (snippet)](https://archives.ismir.net/ismir2021/paper/000051.pdf)
- **Human evaluation.** A web app for user evaluation existed at mingus.tools.eurecom.fr — [search result summary of ISMIR 2021 materials](https://github.com/vincenzomadaghiele/MINGUS)

### Inferences
- MINGUS's design (current and next chord as 4-pitch embeddings, plus bass and beat position) matches the user's needs well. Conditioning on the next chord supports anticipatory approach notes into the next bar.
- Training is on WJazzD, which is mostly horn solos, so the idiom is again horn-like.
- LGPL-3.0 is fine for personal non-commercial use. If modified code is redistributed, the LGPL obligations apply.

### Gaps
- Exact layer, head, and embedding sizes and the WJazzD subset size were not retrieved; the paper PDF was blocked. The constructor arguments are in the repo, but default values live in `train.py`, which I did not inspect.
- Results of the mingus.tools.eurecom.fr user study (participants, expertise, blind or not) were not found.
- Inference speed is not reported.

## Q3. Jazz Transformer (Wu & Yang, ISMIR 2020) and its documented failures

### Takeaway
The Jazz Transformer is an unconditional Transformer-XL lead-sheet model trained on WJazzD. It generates both the chords and the melody itself and has no chord-input argument. The paper is valuable mainly as a failure analysis: high pitch-class entropy, low grooveness, irregular chord progressions, and no long-term structure. In a blind listening test, real pieces were rated higher on every criterion.

### Cited Findings
- **What it is.** A Transformer-XL model of jazz lead sheets using REMI events plus WJazzD structure events (mid-level units) — [GitHub README](https://github.com/slSeanWU/jazz_transformer); [arXiv 2008.01307](https://arxiv.org/abs/2008.01307)
- **(code-verified) Config:** n_layer=12, d_model=512, n_head=8, mem_len=512, implemented in TensorFlow 2.2.0. Chord vocabulary comes from a hand-written profile of 46 chord types — [model_aug.py](https://github.com/slSeanWU/jazz_transformer/blob/master/transformer_xl/model_aug.py); [chord_profile.txt](https://github.com/slSeanWU/jazz_transformer/blob/master/src/chord_profile.txt); [requirements.txt](https://github.com/slSeanWU/jazz_transformer)
- **(code-verified) No chord input at inference.** `inference.py` takes only `--temp` (default 1.2), `--n_bars` (default 32), and an output path; there is no argument to supply chords. The checkpoint downloads from Dropbox links in `download_model.sh`. License is MIT; last commit 2020-10-29 — [inference.py](https://github.com/slSeanWU/jazz_transformer/blob/master/inference.py); [download_model.sh](https://github.com/slSeanWU/jazz_transformer/blob/master/download_model.sh)
- **Diagnosed deficiencies.** "Erraticity of pitch usage (high entropy), lack of consistency in rhythm & harmony (low grooveness and high chord progression irregularity), and absence of longer-term structures" — [ISMIR 2020 PDF (snippet)](https://archives.ismir.net/ismir2020/paper/000339.pdf)
- **Blind listening test.** Each test-taker heard four one-minute pieces, two from the model and two real. Human pieces outperformed the model on overall quality, structureness, and richness: the model scored about 2.7–2.9/5 and humans about 3.0–3.4/5 — [arXiv PDF (snippet)](https://arxiv.org/pdf/2008.01307)

### Inferences
- The Jazz Transformer is not a chord-conditioned soloist. Its failures (scattered pitch classes, irregular harmony) closely resemble the user's symptoms when chords are not enforced. Its metrics, such as pitch-class entropy and grooveness, are useful offline diagnostics.
- The Dropbox checkpoint links may have rotted; this was not tested.

### Gaps
- Number and expertise of listening-test participants were not retrieved.

## Q4. Other chord-conditioned and real-time systems, and 2024–2026 work (incl. arXiv 2511.08755, Frieler & Zaddach, JazzGAN, LLM/ABC models)

### Takeaway
No 2024–2026 paper was found that presents a new jazz/bebop-specific, chord-symbol-conditioned solo generator with expert listening tests. Recent activity is in three other areas:
- general chord-conditioned melody/bass studies on Classical data (Salem et al. 2025)
- real-time co-creative piano and jam systems (Aria-Duet, jam_bot, ReaLJam)
- style-conditioned or genre-conditioned piano models (Edwards et al. ISMIR 2026; ImprovNet 2025)

The strongest jazz-specific expert evidence remains Frieler & Zaddach (TISMIR 2022), a rule-guided hierarchical Markov model built on the Weimar Bebop Alphabet.

### Cited Findings
- **arXiv 2511.08755: "Chord-conditioned Melody and Bass Generation"** (Alexandra C. Salem, Mohammad Shokri, Johanna Devaney; submitted 2025-11-11; NeurIPS 2025 AI4Music workshop):
  - Compares five Transformer strategies: (1) no chord conditioning, (2) independent chord-conditioned lines, (3) bass-first, (4) melody-first, (5) co-generation.
  - Metrics cover pitch content, interval size, and chord-tone usage.
  - Chord conditioning improves replication of stylistic pitch content and chord-tone usage, especially for the bass-first model.

  [arXiv abs](https://arxiv.org/abs/2511.08755); [OpenReview](https://openreview.net/forum?id=xFetnk5I6o)
- **2511.08755 is not jazz.** Its data is TAVERN, 27 sets of Mozart/Beethoven piano theme-and-variations in high Classical style — [arXiv html (snippet)](https://arxiv.org/html/2511.08755); [TAVERN paper](https://archives.ismir.net/ismir2015/paper/000261.pdf)
- **Related harmonic-fit metrics in a separate paper.** "Theory Structured Harmonic Embeddings for Chord Conditioned Melody Generation" (Francis Press) proposes:
  - Chord Tone Ratio (CTR): % of onsets on chord tones
  - Tension Correctness: tensions allowed by the chord symbol
  - Non-Chord-Tone Resolution Score: non-chord tones resolving by step to a chord tone

  The search engine blended these metrics into the Salem summary, so their attribution to Salem et al. is unverified — [Francis Press](https://francis-press.com/papers/20061)
- **Frieler & Zaddach, "Evaluating an Analysis-by-Synthesis Model for Jazz Improvisation"** (TISMIR 5(1):20–34, 2022):
  - Model: a generative model for monophonic jazz improvisation built from a hierarchical Markov model over mid-level units and the Weimar Bebop Alphabet, with statistics from WJazzD and chord-scale theory to choose pitches.
  - Listeners: a Turing-like test with 41 participants (convenience sample from social media and personal contacts; 7 female; mean age 27.0), 29 of whom were classified as jazz experts.
  - Detection rates: experts identified the computer solos at 64.4% accuracy and non-experts at 41.7%, "with a large margin of error."
  - One hand-selected, edited, and expressive rendition of a generated solo fooled the panel, judged slightly more often human than computer.
  - Rendition (timbre, articulation, micro-timing, band interaction) mattered as much as or more than tone content.

  [TISMIR article](https://transactions.ismir.net/articles/10.5334/tismir.87)
- **Weimar Bebop Alphabet.** Phrase-wise parsing of interval sequences into nine classes of melodic "atoms": diatonic, chromatic, approaches, arpeggios, jump arpeggios, repetitions, trills, links, and a residual X — [TISMIR (snippet)](https://transactions.ismir.net/articles/10.5334/tismir.87); [Frieler, "Constructing Jazz Lines"](https://jazzforschung.hfm-weimar.de/wp-content/uploads/2019/06/JazzforschungHeute2019_Frieler-Constructing-Jazz-Lines.pdf)
- **JazzGAN** (Trieu & Keller, MuMe 2018):
  - An RNN-based GAN that improvises monophonic jazz over chord progressions, built for the Impro-Visor teaching tool.
  - Trained on only 44 lead sheets (~1,700 bars).
  - Uses harmonic "bricks" for phrase segmentation and Impro-Visor note categories: chord tones, color tones, approach tones, other.
  - Compared favorably to Magenta's ImprovRNN on its defined metrics.

  [MuMe 2018 PDF](https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf)
- **Impro-Visor LSTM.** Impro-Visor 9.0 (2017) shipped an LSTM. Johnson, Keller et al. (2017) used two LSTM sub-networks ("product of experts") to predict future-note distributions conditioned on past melody and chords — [Kritsis et al., Frontiers AI 2021 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907589/); [ResearchGate: Learning to Create Jazz Melodies Using a Product of Experts](https://www.researchgate.net/publication/318311055_Learning_to_Create_Jazz_Melodies_Using_a_Product_of_Experts)
- **Real-time RNN accompaniment.** Kritsis et al. (2021) studied the adaptability of RNNs for real-time jazz improvisation accompaniment. This is the reverse direction: the system responds to a soloist — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7907589/)
- **ReaLJam** (CHI EA 2025, arXiv 2502.21267):
  - Real-time human-AI jamming in which the user plays melody and an RL-tuned Transformer plays chords. This is the inverse of the user's task.
  - Configurable lookahead and commit time; seeing upcoming chords helped users plan melodies.
  - Code is in realchords-pytorch.

  [arXiv](https://arxiv.org/abs/2502.21267); [ACM](https://dl.acm.org/doi/10.1145/3706599.3720227); [code](https://github.com/ikunabel/realchords-pytorch)
- **jam_bot** (ISMIR 2025; MIT Media Lab with Jordan Rudess):
  - Real-time free improvisation with symbolic music LMs, adapting context and conditioning for interaction modes, including the agent improvising lead over the musician's accompaniment.
  - Low-latency multithreaded scheduling.

  [ISMIR 2025 poster page](https://ismir2025program.ismir.net/poster_321.html); [MIT project page](https://www.media.mit.edu/projects/jordan-rudess-genai/overview/)
- **Aria-Duet / "The Ghost in the Keys"** (NeurIPS 2025 Creative AI): turn-taking real-time duet on a Disklavier using Aria, an autoregressive expressive-piano Transformer — [arXiv 2511.01663](https://arxiv.org/abs/2511.01663)
- **Aria.** LLaMA-3.2-1B-style architecture trained on ~60k hours of transcribed solo-piano MIDI; Apache-2.0 — [EleutherAI/aria](https://github.com/EleutherAI/aria); [aria-medium-base](https://huggingface.co/loubb/aria-medium-base)
- **"Learning Jazz Pianist Style with Cross-Attention Conditioning"** (Drew Edwards et al., QMUL, ISMIR 2026):
  - Built on Aria-medium (659M) with cross-attention over 12 learned pianist embeddings in the last 8 of 16 layers, fine-tuned on PiJAMA-12.
  - Conditioning persists through long generations. A sliding-window classifier attributes conditioned continuations to the intended pianist about 33 points more often than unconditioned baselines.
  - A classifier trained only on synthetic generations identifies real pianists with 87% chunk-level and 95% song-level accuracy.
  - It conditions on pianist identity, not chords.

  [paper PDF (snippet)](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2026/EdwardsEtAl-ISMIR2026.pdf); [HF model card (snippet)](https://huggingface.co/drewbie/jazz-pianist-style-generator)
- **ImprovNet** (Bhandari, Chang, Lu, Enus, Bradshaw, Herremans, Colton; IJCNN 2025):
  - A Transformer with self-supervised corruption-refinement that produces expressive, controllable improvisations of classical and jazz solo-piano pieces.
  - Capabilities: cross-genre and intra-genre improvisation, genre-specific harmonization of melodies, continuation, and infilling.
  - Outperforms the Anticipatory Music Transformer on short continuation and infilling.
  - 79% of participants correctly identified jazz-style improvisations of classical pieces.

  [arXiv 2502.04522](https://arxiv.org/abs/2502.04522)
- **Anticipatory Music Transformer** (Thickstun, Hall, Donahue, Liang; TMLR 2024):
  - Infilling control: given some parts (e.g., a melody), the model generates the rest (e.g., accompaniment). Trained on Lakh MIDI.
  - Apache-2.0, with pretrained models on HF. (code-verified) Last commit 2024-03-18.

  [GitHub](https://github.com/jthickstun/anticipation); [paper](https://arxiv.org/pdf/2306.08620)
- **Text2midi** (AAAI 2025): an LLM text encoder plus an autoregressive MIDI decoder, trained on MidiCaps (168,385 MIDI files with captions covering tempo, chord progression, key, instruments, genre, mood). Generations are controllable by captions that mention chords and keys — [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/34516/36671); [HF](https://huggingface.co/amaai-lab/text2midi); [MidiCaps](https://arxiv.org/html/2406.02255)
- **NotaGen** (IJCAI 2025): pretrained on 1.6M ABC pieces and fine-tuned on about 9K classical works with "period-composer-instrumentation" prompts and CLaMP-DPO. Not jazz-specific — [IJCAI PDF](https://www.ijcai.org/proceedings/2025/1134.pdf)
- **ChatMusician** (ACL Findings 2024): LLaMA2 continually pretrained and fine-tuned on ABC notation — [ACL Anthology](https://aclanthology.org/2024.findings-acl.373.pdf). **MuPT** (2024): ABC-based pretrained model with SMT-ABC multitrack notation — [arXiv 2404.06393](https://arxiv.org/pdf/2404.06393)
- **ISMIR 2024 chord-aware model.** "MMT-BERT: Chord-Aware Symbolic Music Generation Based on Multitrack Music Transformer and MusicBERT" (Zhu et al.) — [ISMIR 2024 accepted papers](https://ismir2024.ismir.net/accepted-papers/)
- **2025 WJazzD-derived rhythm model.** "Phrase-Oriented Generative Rhythmic Patterns for Jazz Solos" (Applied Sciences 15(20):11058) is a Markov-chain generator of phrase-level rhythm patterns from WJazzD statistics, evaluated statistically. It covers rhythm only — [DOI](https://doi.org/10.3390/app152011058)
- **Older related work, found by title only:**
  - "Improving Automatic Jazz Melody Generation by Transfer Learning Techniques" (2019) — [arXiv 1908.09484](https://arxiv.org/pdf/1908.09484)
  - "Comparative Assessment of Markov Models and RNNs for Jazz Music Generation" (2023) — [arXiv 2309.08027](https://arxiv.org/pdf/2309.08027)
  - Yang et al. "Learning to Generate Jazz & Pop Piano Music from Audio via MIR Techniques" (ISMIR 2019 LBD/slides), an audio→transcription→beat grid→composition pipeline — [slides](https://www.slideshare.net/affige/learning-to-generate-jazz-pop-piano-music-from-audio-via-mir-techniques)
- **2026 search.** A search for 2026 bebop/jazz chord-conditioned solo generators returned only older work (BebopNet, surveys) — [search-surfaced survey](https://arxiv.org/pdf/2011.06801)

### Inferences
- Across the literature, the reliable way to make a symbolic jazz model chord-aware is **per-step chord features** (BebopNet, MINGUS, SeqAttn) or **interleaved chord tokens** (REMI-style, as in Jazz Transformer). Prompting alone is not used.
- 2024–2026 conditioning mechanisms mostly use **cross-attention adapters on a large pretrained piano LM** (Edwards 2026 on Aria). That mechanism could carry a chord stream instead of a pianist ID. No paper doing exactly that was found.
- The **Anticipatory Music Transformer**'s control-event infilling is a plausible off-the-shelf route: put the comping or chord-voicing events in as "controls" and generate a melody. It is Lakh-trained, though, not jazz-trained.
- **Text/ABC LLMs** (Text2midi, NotaGen, ChatMusician) take chords only as caption-level or score-level text. None documents beat-accurate chord adherence or real-time latency for jazz soloing.
- Salem et al.'s **bass-first** finding hints that conditioning melody on a bass line plus chord helps chord-tone usage. MINGUS also uses bass as a feature.

### Gaps
- Not found: a Frieler & Zaddach code release, and detailed per-solo ratings or effect sizes beyond the accuracy figures (the article page was blocked).
- ImprovNet code and weights availability, listener expertise, and blinding were not verified.
- Edwards 2026: no listening test found; license of the fine-tuned weights not verified (base Aria is Apache-2.0, PiJAMA is CC BY-NC).
- "LK Jam" could not be identified; the closest match is ReaLJam. "JazzDreamer", "Bebop language model", or a jazz-specific LLM soloist yielded no credible hits.
- None of the real-time systems (ReaLJam, jam_bot, Aria-Duet) has published chord-symbol-conditioned bebop soloing with harmonic-fit metrics.

## Q5. Which systems have the strongest blind/expert listening evidence, and which produce piano (vs. sax) idiom?

### Takeaway
Only one jazz-specific generator was tested with a jazz-expert discrimination test: Frieler & Zaddach's Weimar Bebop Alphabet model (41 listeners, 29 experts). Experts could still detect it (64.4%) unless an expressive, hand-edited rendition was used. Every chord-conditioned neural bebop model (BebopNet, MINGUS) is saxophone/horn-trained, and the piano-idiom jazz models (Aria/PiJAMA-based, ImprovNet) are not chord-symbol-conditioned. Nothing found combines all three: piano idiom, chord conditioning, and expert validation.

### Cited Findings

| System (year) | Chord conditioning | Data / idiom | Listening evidence | Code / weights / license |
|---|---|---|---|---|
| Frieler & Zaddach (TISMIR 2022) | Chord-scale pitch selection inside a hierarchical Markov model over WBA atoms | WJazzD; monophonic horn-style lines | Turing-like test: 41 participants, 29 experts. Experts 64.4% vs non-experts 41.7% detection accuracy; one expressive edited rendition fooled the panel — [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87) | Not found |
| BebopNet (ISMIR 2020) | 4 chord pitches embedded and concatenated per note | 284 sax solos (Parker, Stitt, Adderley…) — [ISMIR](https://archives.ismir.net/ismir2020/paper/000132.pdf) | Preference data from 4 amateur jazz musicians (authors' circle); plagiarism and harmony metrics; no blind expert test found | MIT; pretrained Transformer-XL; torch 1.1 — [repo](https://github.com/shunithaviv/bebopnet-code) |
| MINGUS (ISMIR 2021) | Current and next chord (4 pitches each), bass, beat, offset per note | WJazzD + Nottingham — [ISMIR](https://archives.ismir.net/ismir2021/paper/000051.pdf) | Web user evaluation existed; results not retrieved | LGPL-3.0; pretrained pitch and duration models — [repo](https://github.com/vincenzomadaghiele/MINGUS) |
| Jazz Transformer (ISMIR 2020) | None at inference (generates its own chords) | WJazzD lead sheets | Blind test: model 2.7–2.9/5 vs human 3.0–3.4/5 — [arXiv](https://arxiv.org/pdf/2008.01307) | MIT; TF 2.2; Dropbox checkpoint — [repo](https://github.com/slSeanWU/jazz_transformer) |
| JazzGAN (MuMe 2018) | Chord progression input; Impro-Visor note categories | 44 lead sheets | Metric comparison vs ImprovRNN — [PDF](https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf) | Not verified |
| ImprovNet (IJCNN 2025) | Implicit, from source piece (corruption-refinement) | Classical + jazz solo piano | 79% of participants identified jazz style — [arXiv](https://arxiv.org/abs/2502.04522) | Not verified |
| Edwards et al. (ISMIR 2026) | None (pianist-ID cross-attention) | PiJAMA-12 jazz piano on Aria-medium 659M | Classifier-based only (87%/95%) — [PDF](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2026/EdwardsEtAl-ISMIR2026.pdf) | HF weights — [HF](https://huggingface.co/drewbie/jazz-pianist-style-generator) |

- Frieler & Zaddach found that expressive and performative aspects (timbre, articulation, micro-timing, band-soloist interaction) seem equally or more important than tone content for expert judgments — [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.87)

### Inferences
- For the user's "no audible wrong notes, real bebop licks" goal, the evidence base favors two things:
  - **(a) Explicit chord conditioning in training.** All chord-conditioned neural soloists do this, and none relies on prompts.
  - **(b) Bebop-vocabulary structure.** Frieler's WBA atoms (approaches, chromatic, arpeggios, enclosure-like links) are the only expert-tested approach, and they map directly onto the user's wish list.

  No neural chord-conditioned model has expert-blind evidence that beats this.
- Frieler's result implies the user's ears will also judge timing and velocity rendering. A good symbolic line can still sound "fake" if rendered with flat dynamics and timing.
- For piano idiom, the most practical path implied by the literature is a hybrid. Take a chord-conditioning recipe (BebopNet/MINGUS-style per-note chord features) and apply it to piano-derived single-line data, or add a chord cross-attention adapter to a piano LM (Edwards-style mechanism). Neither has published validation.

### Gaps
- MINGUS user-study results, BebopNet user-rating outcomes, and the Jazz Transformer listener demographics could not be retrieved because the PDFs were blocked.
- No study found reports avoid-note rate or semitone-clash rate against comping for any jazz soloist model. Only chord-tone and scale membership style metrics and MGEval histograms appear.

## Q6. Chord-annotated jazz datasets (size, instrument, annotations, license, access)

### Takeaway
Only a few datasets combine **solo notes + chord labels + beat grid**:
- WJazzD (456 horn-dominated solos, ODbL)
- FiloSax (48 tunes × 5 sax players; restrictive research-only agreement per the GitHub README, despite CC BY 4.0 shown on Zenodo)
- FiloBass (48 bass lines, CC BY 4.0)
- Charlie Parker Aligned Omnibook (50 Parker solos, CC BY 4.0)

Large **jazz piano** corpora have no chord labels:
- PiJAMA: 2,777 performances, automatic transcription, CC BY-NC 3.0. These numbers match the user's 2,777-piece pretraining set.
- Jazz Trio Database: MIT annotations, piano MIDI + beats.
- Doug McKenzie: ~250 hand-played MIDI, non-commercial.

Chord-only corpora are the iRealPro corpus / Jazz Harmony Treebank (CC BY-NC-SA 4.0 per the repo license file) and JAAH (CC BY-NC-SA 4.0).

### Cited Findings
- **Weimar Jazz Database (WJazzD).**
  - Size: 456 manually transcribed solos from 340 tracks (197 records), time-aligned to audio.
  - Annotations: meter, beats, measures, chord labels, phrases, mid-level units, structure, style, solo instrument.
  - Formats: SQLite DB, MIDI, PDF.
  - License: Open Data Commons ODbL.
  - Download: jazzomat.hfm-weimar.de/download.

  [Jazzomat download (snippet)](https://jazzomat.hfm-weimar.de/download/download.html); [JSD paper, TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.131)
- **DTL1000 (Dig That Lick).**
  - Size: 1,736 monophonic solos from 1,060 tracks spanning 1920–2020, about 300,000 tone events.
  - Transcription is automatic, by a jazz-specialized CRNN. Structure and style were manually annotated, and solo timestamps, instrument, and soloist are attached.
  - Hosted on UK Data Service ReShare. Search snippets cite a CC BY license, but this may refer to the article rather than the data.

  [ReShare record](https://reshare.ukdataservice.ac.uk/854781/); [Jazz Ontology article](https://www.sciencedirect.com/science/article/pii/S1570826822000245); [Dig That Lick](https://dig-that-lick.eecs.qmul.ac.uk/)
- **FiloSax.**
  - Size: 48 jazz standards (105–226 BPM) × 5 saxophonists, about 5 hours per stem.
  - Annotations: note-level MIDI (onset, offset, pitch), bar and mid-bar chord changes (.jams), beats, sections (head / written solo / improvised solo), and expressive features.
  - Restrictions: the GitHub README's agreement says use is limited to "the individual signing… and members of the research group" for "non-commercial research purposes," with no publication or distribution to third parties. Access goes through Zenodo with a permission agreement; a 2-track Lite version exists.

  [GitHub](https://github.com/dave-foster/filosax); [Zenodo](https://zenodo.org/records/5625643); [Filosax Lite](https://zenodo.org/records/5603104). This is contradicted by a search snippet stating CC BY 4.0, apparently from the Zenodo listing — [Zenodo (snippet)](https://zenodo.org/records/5625643)
- **FiloBass.** 48 manually verified transcriptions of professional jazz bassists over the FiloSax backing tracks, with 50,000+ notes, downbeats, chords, and MusicXML scores. CC BY 4.0 (ISMIR 2023) — [Zenodo](https://zenodo.org/records/10069709); [arXiv 2311.02023](https://arxiv.org/pdf/2311.02023)
- **Charlie Parker Omnibook, two versions:**
  - (a) MusicXML of 50 of the 60 Omnibook solos with chord progressions and themes — [LORIA page](https://homepages.loria.fr/evincent/omnibook/)
  - (b) **Charlie Parker Aligned Digital Omnibook** (Riley & Dixon, SMC 2024): 50 recordings with aligned MIDI, source-separated sax stems, downbeats, and MusicXML. CC BY 4.0 — [Zenodo](https://zenodo.org/records/14628467); [arXiv 2405.16687](https://arxiv.org/html/2405.16687v1)
- **iRealPro corpus / Jazz Harmony Treebank.**
  - Source: chord sequences from the iRealPro user community, first presented by Shanahan, Broze & Rodgers (2012).
  - Treebank (ISMIR 2020): hierarchical analyses of a subset.
  - (code-verified) `treebank.json` has 1,170 tune entries, 150 of them with tree analyses. The repo's `LICENSE.md` is CC BY-NC-SA 4.0.

  [GitHub](https://github.com/DCMLab/JazzHarmonyTreebank); [ISMIR 2020 paper](https://program.ismir2020.net/static/final_papers/80.pdf); [iRealPro corpus Zenodo](https://zenodo.org/records/3546040). This contradicts a search snippet claiming CC BY 4.0 for both.
- **JAAH** (Jazz Audio-Aligned Harmony): chord annotations aligned to jazz audio. CC BY-NC-SA 4.0 — [Zenodo](https://zenodo.org/records/1290737); [ISMIR 2018 paper](https://archives.ismir.net/ismir2018/paper/000206.pdf)
- **PiJAMA.**
  - Size: 200+ hours of solo jazz piano, automatically transcribed to MIDI: 2,777 unique performances by 120 pianists from 244 albums, studio and live.
  - Has audio tagging for applause and speech.
  - (code-verified) Repo LICENSE is CC BY-NC 3.0 Unported; last commit 2023-11-30.
  - MIDI is on Zenodo record 8354955.

  [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.162); [GitHub](https://github.com/almostimplemented/PiJAMA)
- **Jazz Trio Database (JTD)** (Cheston et al., TISMIR 2024):
  - 1,294 tracks / 44.5 h of piano trio, with onsets, beats, and downbeats for each performer plus MIDI for the piano soloist (2,174,833 notes).
  - (code-verified) MIT license for the dataset; YouTube audio is not covered and needs an access application. Last commit 2025-09-23.

  [GitHub](https://github.com/HuwCheston/Jazz-Trio-Database); [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.186)
- **Doug McKenzie** (bushgrafts.com): about 250 solo jazz piano performances, mostly standards, played into MIDI on a Yamaha P250. The original MIDI files were noted as not for commercial use, with contact for licensing — [Pianoteq forum](https://forum.modartt.com/viewtopic.php?id=1014); [bushgrafts.com](https://bushgrafts.com/)
- **Other / out of scope:**
  - Aria-MIDI (ICLR 2025): large solo-piano MIDI corpus with no chord labels — [arXiv 2504.15071](https://arxiv.org/pdf/2504.15071)
  - JazzSAMBA (2026): multi-take band audio, audio-domain — [arXiv](https://arxiv.org/html/2609.34931)
  - jazznet: 162,520 piano patterns (chords, arpeggios, scales, progressions) as audio — [GitHub](https://github.com/tosiron/jazznet)

### Inferences
- **Diagnosis of the user's model.** The user's "2,777 jazz pieces" almost certainly is PiJAMA, which has no chord or harmony labels. The model therefore had no supervised chord signal, which strongly supports the user's hypothesis that it never learned chord-melody relationships.
- **Training a chord-conditioned piano soloist** will likely need one of two things, neither validated in the literature:
  - (a) automatic chord labeling of PiJAMA/JTD MIDI, e.g., by a symbolic chord recognizer or by aligning to iRealPro charts of the same tunes
  - (b) pretraining or fine-tuning on chord-labeled horn data (WJazzD, Omnibook-aligned, FiloSax), then adapting to piano idiom
- **Best fully open chord-labeled sources for a bebop line model:** WJazzD (ODbL) and the Charlie Parker Aligned Omnibook (CC BY 4.0). FiloSax is useful but research-only.

### Gaps
- WJazzD per-instrument counts (how many piano solos) were not retrieved; the jazzomat site was blocked.
- JAAH track count was not verified.
- The exact license text for the DTL1000 data files was not verified (ReShare was blocked).
- The license of the LORIA Omnibook MusicXML was not found.
- No 2024–2026 dataset of piano solo transcriptions aligned to lead-sheet chords was found.

## Q7. Licensing implications for a non-commercial personal project and a public demo video

### Takeaway
Personal, non-commercial training and use is compatible with essentially all the sources above except FiloSax redistribution. A public demo video is fine for MIT, Apache, and LGPL code and for ODbL and CC BY data with attribution. CC BY-NC data (PiJAMA, JAAH, Jazz Harmony Treebank) and Doug McKenzie MIDI make any monetized video or commercial use risky, and nothing derived from FiloSax should be published beyond the signed research group.

### Cited Findings
- **Code licenses:**
  - BebopNet: MIT — [repo](https://github.com/shunithaviv/bebopnet-code)
  - Jazz Transformer: MIT — [repo](https://github.com/slSeanWU/jazz_transformer)
  - MINGUS: LGPL-3.0 — [repo](https://github.com/vincenzomadaghiele/MINGUS)
  - Anticipatory Music Transformer: Apache-2.0 — [repo](https://github.com/jthickstun/anticipation)
  - Aria: Apache-2.0 — [HF card (snippet)](https://huggingface.co/drewbie/jazz-pianist-style-generator)
- **Data licenses:**
  - WJazzD: ODbL — [Jazzomat (snippet)](https://jazzomat.hfm-weimar.de/download/download.html)
  - FiloBass: CC BY 4.0 — [Zenodo](https://zenodo.org/records/10069709)
  - Aligned Omnibook: CC BY 4.0 — [Zenodo](https://zenodo.org/records/14628467)
  - PiJAMA: CC BY-NC 3.0 — [repo](https://github.com/almostimplemented/PiJAMA)
  - JAAH: CC BY-NC-SA 4.0 — [Zenodo](https://zenodo.org/records/1290737)
  - Jazz Harmony Treebank: CC BY-NC-SA 4.0 per its LICENSE.md — [repo](https://github.com/DCMLab/JazzHarmonyTreebank)
  - JTD: MIT for annotations — [repo](https://github.com/HuwCheston/Jazz-Trio-Database)
  - FiloSax: research-only, non-distribution agreement — [repo](https://github.com/dave-foster/filosax)
  - Doug McKenzie MIDI: not for commercial use — [forum](https://forum.modartt.com/viewtopic.php?id=1014)
- **BebopNet's training transcriptions** were purchased from saxsolos.com and are not distributed with the code — [ISMIR (snippet)](https://archives.ismir.net/ismir2020/paper/000132.pdf); repo `resources/xmls` contains only heads (code-verified)

### Inferences
- **Demo video.**
  - Using pretrained BebopNet (MIT code) is low-risk legally for a non-monetized demo with attribution. The weights' training data were commercially purchased transcriptions, so a monetized use would be a gray area.
  - Playing copyrighted standards' chord changes is generally not the issue; reproducing copyrighted melodies or heads could be. This is a general copyright point, not verified for any jurisdiction.
- **Non-commercial terms flow into the user's own model.** Any model fine-tuned on PiJAMA (CC BY-NC) should be treated as carrying the non-commercial restriction. This already applies to the user's current model.
- **Attribution.** CC BY and ODbL sources require it, so the video description should credit them: WJazzD / Jazzomat, Riley & Dixon, Edwards et al. / PiJAMA, and so on.

### Gaps
- Whether model weights trained on CC BY-NC data are "adapted material" under CC is legally unsettled; no authoritative source was consulted.
- Licenses of the HF weights for drewbie/jazz-pianist-style-generator and ImprovNet were not verified.
