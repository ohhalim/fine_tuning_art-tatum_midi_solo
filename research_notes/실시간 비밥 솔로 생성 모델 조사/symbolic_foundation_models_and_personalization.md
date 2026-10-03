# Open symbolic-music foundation models for a chord-conditioned bebop piano solo generator, and LoRA/adapter style personalization (research notes, 2026-10-03)

Method and verification note (read first): the egress proxy in this research session blocked direct fetches of arxiv.org, huggingface.co, openreview.net, zenodo.org, ismir.net, nime.org, and most project/blog sites; only GitHub (github.com / raw.githubusercontent.com) could be fetched. Each finding carries one of two tags:
- **[V]** = verified this session by reading the primary artifact itself (repo README, LICENSE file, source code, config file on GitHub).
- **[S]** = taken from a search-engine summary of the linked page (the page itself could not be opened). Treat [S] as "reported by the source, not independently re-read". Numbers tagged [S] should be spot-checked in the PDF before they go into a final decision.
- Anything labelled "Inference" is my own reasoning, not a sourced fact.

Context recap for the reader (from the assignment, not a sourced fact): the target is a realtime system (FL Studio, keyboard-typed chords) that generates bebop right-hand lines in half-bar blocks (0.94 s at 128 BPM) with ~0.4 s p99 per-block preparation on an Apple Silicon Mac (MPS, no NVIDIA). The current 13.4M Music Transformer (note_on/off, 10 ms time-shift, velocity) gets chords only as a prepended guide-tone voicing and largely ignores them (10–37% token-identical outputs across chords; ~1% NLL gap between right and wrong harmony). Local data are PiJAMA-style transcribed two-hand jazz piano MIDI (13 bebop pianists, 235 pieces) with no chord labels and no beat grid.

## Q1. Anticipatory Music Transformer (AMT): tokenization, anticipation, sizes, data, license, checkpoints, demos/evals, and fit for "comping/chord tones = control, RH solo = events"

### Takeaway
AMT (Thickstun, Hall, Donahue, Liang; arXiv June 2023, TMLR 04/2024) is the most directly reusable open model for this project: it is Apache-2.0, has public 128M/360M/780M checkpoints, and its built-in "accompaniment" API already treats one set of notes as asynchronous *controls* and generates the rest as *events*. Swapping roles (comping/chord tones as controls, right-hand solo as events) needs no new architecture, only fine-tuning data in that layout. The main problems are the 5-second anticipation window, which assumes controls are known 5 s ahead and does not fit live chord typing, the absence of bar/beat tokens, and the decode cost of 360M on MPS.

### Cited Findings
- **Paper and dates.** "Anticipatory Music Transformer", by John Thickstun, David Hall, Chris Donahue, and Percy Liang. arXiv 2306.08620 (June 2023, v2 later); published in Transactions on Machine Learning Research (04/2024). — [GitHub README [V]](https://github.com/jthickstun/anticipation); [TMLR PDF [S]](https://openreview.net/pdf?id=EBNJ33Fcrl); [MLAnthology TMLR 2024 [S]](https://mlanthology.org/tmlr/2024/thickstun2024tmlr-anticipatory/)
- **License.** The code is Apache-2.0: "This project is licensed under the terms of the Apache License, Version 2.0" (LICENSE.txt present). — [README [V]](https://github.com/jthickstun/anticipation); [LICENSE.txt [V]](https://github.com/jthickstun/anticipation/blob/main/LICENSE.txt). Separate license metadata on the HF model cards could not be opened; this is listed under Gaps.
- **Tokenization (arrival-time triplets) [V from source].** In `config.py`, `EVENT_SIZE = 3` ("each event/control is encoded as 3 tokens"), `TIME_RESOLUTION = 100` ("10ms time resolution = 100 bins/second"), `MAX_TIME_IN_SECONDS = 100`, `MAX_DURATION_IN_SECONDS = 10`, `MAX_NOTE = MAX_PITCH*MAX_INSTR` (128 pitches × 129 instruments incl. drums, so one token encodes pitch and instrument together), `CONTEXT_SIZE = 1024` with `M = 341` events per context, and `DELTA = 5` ("anticipation time in seconds"). — [config.py [V]](https://github.com/jthickstun/anticipation/blob/main/anticipation/config.py)
- **Vocabulary layout [V].** `vocab.py` defines an event block (TIME, DUR, NOTE offsets plus a REST token) and a separate *control block* with its own anticipated time, duration, and note offsets (ATIME/ADUR/ANOTE), plus special tokens SEPARATOR, AUTOREGRESS, and ANTICIPATE. Controls therefore use a disjoint token range from events. A MIDI-like interarrival vocabulary (time-shift, note-on, note-off) also exists as an alternative encoding. — [vocab.py [V]](https://github.com/jthickstun/anticipation/blob/main/anticipation/vocab.py)
- **Anticipation mechanism.** Seq2Seq-style prefix conditioning "places time-localized controls far from the events they describe". AMT instead interleaves controls so that "a control appearing on time s_k appears close to events near time s_k − δ", which lets the model anticipate the control δ seconds in advance. Predictions combine a causal (filtering) estimate from local event history with a smoothing (bidirectional) estimate from local controls. — [arXiv PDF [S]](https://arxiv.org/pdf/2306.08620); [CSAIL talk page [S]](https://www.csail.mit.edu/event/anticipation-and-anticipatory-music-transformer)
- **Model sizes and data.** Small is 128M, Medium 360M, and Large 780M parameters, all trained on Lakh MIDI. The anticipatory models match the perplexity and bits-per-second of standard autoregressive baselines, so infilling ability did not cost prompted-generation quality. — [ResearchGate/arXiv summary [S]](https://www.researchgate.net/publication/371605312_Anticipatory_Music_Transformer). The `music-small-800k` card describes "a Small (128M parameter) Transformer trained for 800k steps … from the Lakh MIDI dataset … trained with anticipation" [S]. The `music-large-800k` card describes a 780M model trained on "Lakh MIDI dataset, MetaMidi dataset, and transcripts of the FMA audio dataset and 450k commercial music records" [S]. So the released Large checkpoint uses a broader corpus than Lakh alone. — [HF music-small-800k [S]](https://huggingface.co/stanford-crfm/music-small-800k); [HF music-large-800k [S]](https://huggingface.co/stanford-crfm/music-large-800k)
- **Checkpoints on HF (stanford-crfm).** The README loads `stanford-crfm/music-medium-800k` through `AutoModelForCausalLM` (a standard HF Transformers GPT-2-type model) [V]. `music-small-800k`, `music-large-800k`, and `music-large-100k` also appear in search results [S]. Intermediate checkpoints (every 10k steps) are "available on request" [S]. — [README [V]](https://github.com/jthickstun/anticipation); [HF music-large-100k [S]](https://huggingface.co/stanford-crfm/music-large-100k)
- **Accompaniment/infilling API [V].** "To generate an accompaniment to an isolated melody, call the `generate` function using the melody as control inputs": `accompaniment = generate(model, 5, 20, inputs=history, controls=melody, top_p=.98)`. `extract_instruments(events, [53])` splits one instrument out to serve as controls. `generate(model, start_time, end_time, ...)` works in seconds. — [README [V]](https://github.com/jthickstun/anticipation)
- **Training stack [V].** The repo has no trainer of its own. Training used Levanter (`split_lakh` branch) with `lakh-small/medium/large.yaml` configs. Preprocessing has an `--augment` factor ("multiple of 10") for anticipatory training data, and the anticipatory-preprocessed dataset is a multiple of the ~10 GB base Lakh token set. — [train/README [V]](https://github.com/jthickstun/anticipation/blob/main/train/README.md)
- **Human evaluation.** Two tasks were run. (1) Prompt continuation: 50 clips, a 3-bar Lakh prompt plus model or human continuation. (2) Accompaniment: 20 clips, a 5-second prompt plus a 15-second accompaniment infilled from the full 20-second melody. Evaluators reported that anticipatory accompaniments had "similar musicality to even music composed by humans over a 20-second clip." — [arXiv/TMLR summary [S]](https://arxiv.org/html/2306.08620v2); [CRFM blog [S]](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html). The `humaneval/` directory reproduces the procedure [V](https://github.com/jthickstun/anticipation).
- **Downstream reuse.** MIDI-LLM (ISMIR 2026) adds the AMT MIDI vocabulary (>55k tokens: onset, duration, instrument-pitch) to Llama 3.2 1B. jam_bot and its NIME 2026 follow-up are built on AMT. — [MIDI-LLM arXiv [S]](https://arxiv.org/html/2511.03942); [NIME 2026 jam_bot paper [S]](https://nime.org/proceedings/2026/nime2026_73.pdf)
- **Latency.** No per-token latency figure for AMT itself was found. The jam_bot authors needed ONNX Runtime in C++ with KV caching to run AMT "faster than real-time (RTF > 1)", and later a ggml port for more throughput (see Q2). — [NIME 2026 [S]](https://nime.org/proceedings/2026/nime2026_73.pdf)

### Inferences
- **Role swap is natively supported.** Controls are just notes with their own token range. LH comping notes, or synthetic chord-tone notes rendered on a dedicated "control instrument", can be the control stream, and the RH solo becomes the event stream. This mirrors the README's melody→accompaniment example with the roles reversed. It needs (a) hand separation of the two-hand PiJAMA transcriptions (pitch-split or voice-separation heuristics) and/or (b) chord labels converted into control notes (e.g., root + 3rd + 7th) from an automatic chord estimator.
- **The 5 s δ is the biggest mismatch with live chord typing.** With `DELTA = 5`, the pretrained model expects to see controls up to 5 s before the events they govern. In FL Studio the next chord is known only when pressed. Options:
  - Fine-tune with a re-tokenized dataset at δ ≈ 0.5–1 s (DELTA is a single preprocessing constant).
  - Impose a one-block lookahead, where the user's chord applies to the next half-bar.
  - Have the scheduler inject the current chord's control notes at block start and accept "late" controls.
  The jam_bot authors handled role and interaction changes "by modifying the context and conditioning signals" (Q2), which suggests this re-tokenization route is practical.
- **The 10 ms absolute-time grid fits the local data.** PiJAMA transcriptions have no beat grid, and AMT needs none. At 128 BPM a straight eighth note is ~234 ms ≈ 23 time bins and a 2:1 swing pair is ~156 + ~313 ms, both well resolved.
- **The cost of no beat grid.** AMT has no bar/beat tokens, so metric position (chord-tone landings on beats 1 and 3) must be inferred from the timing of the control notes. Including a metronome-like control (e.g., a ghost note on each beat) as part of the controls is a cheap way to inject the grid.
- **Context budget.** The 1024-token window holds 341 events (controls + events). At ~4 RH notes/s plus ~4–6 control notes/s that is roughly 35–40 s of musical context, ample for half-bar blocks.
- **Per-block token budget.** At 3–5 notes/s × 0.94 s, one block is 3–5 RH notes = 9–15 generated tokens (3 tokens per note), plus a prefill of ~12 tokens per 4-note chord control. Fitting in 0.4 s p99 means roughly ≤25–40 ms per decoded token if decoding is serial. That is plausible for 128M on MPS with a KV cache and uncertain for 360M without MLX/ggml-style optimization (compare Aria-Duet in Q3). This is an estimate, not a measurement.

### Gaps
- The model-card license fields for `stanford-crfm/music-*` and the exact model-card text for `music-medium-800k` (data mix) could not be opened, because huggingface.co was blocked. Repo code is verified Apache-2.0.
- No exact human-evaluation scores or significance tests were retrieved (arXiv blocked). A search summary from another paper mentioned AMT coherence of "3.70 ± 0.10" versus MuseCoco "3.11 ± 0.10", but the originating source was unclear, so it is not used here.
- No published per-token latency of AMT on Apple Silicon/MPS was found.

## Q2. jam_bot (ISMIR 2025) and related real-time AMT systems: authors, fine-tuning, roles, realtime optimization, musician evaluation, repo/license

### Takeaway
jam_bot is the ISMIR 2025 paper "The jam_bot, a Real-Time System for Collaborative Free Improvisation with Music Language Models". It fine-tuned AMT on Jordan Rudess's recorded playing and ran it in a low-latency multi-threaded engine (ONNX Runtime C++ with KV cache; a ggml port and an Apple M3 Max backend were added in a NIME 2026 follow-up). It switches between lead, accompaniment, and call-and-response roles by changing context and conditioning. Two items in the assignment's description could not be verified: I found no public jam_bot code repository or MIT license (the paper record is CC-BY-4.0), and I found no RTX-4090 int8 numbers.

### Cited Findings
- **Title, authors, venue.** "The jam_bot, a Real-Time System for Collaborative Free Improvisation with Music Language Models", by Lancelot Blanchard, Perry Naseck, Stephen Brade, Kimaya Lecamwasam, Jordan Rudess, Cheng-Zhi Anna Huang, and Joseph Paradiso. ISMIR 2025 (Daejeon), proceedings pp. 755–762. — [ISMIR 2025 program [S]](https://ismir2025program.ismir.net/poster_321.html); [Zenodo record [S]](https://zenodo.org/records/17706584). A companion music-program entry, "The JAM_BOT: An Insight Into An AI-Human Co-Created Musical Improvisation", also appeared at ISMIR 2025. — [ISMIR music_5 [S]](https://ismir2025program.ismir.net/music_5.html)
- **What it does.** It was built for the GRAMMY-winning keyboardist Jordan Rudess and debuted at a sold-out concert. To take different musical roles (lead, accompany, call-and-response), the "music language models are adapted to take on different interaction strategies by modifying the context and conditioning signals". It "includes optimizations needed to run music language models in real-time" inside "a low-latency multi-threaded system that listens, prompts, and schedules model generations." — [ISMIR program [S]](https://ismir2025program.ismir.net/poster_321.html); [MIT GenAI page [S]](https://genai.mit.edu/developing-jam_bots-real-time-collaborative-agents-for-live-human-ai-musical-improvisation/)
- **Fine-tuning data.** "Blanchard fine-tuned the model using Rudess' own playing of elements from bass lines to chords to melodies, variations of which Rudess recorded in his New York studio." — [MIT Media Lab article [S]](https://www.media.mit.edu/articles/a-model-of-virtuosity/); [Boston Globe, 2026-03-12 [S]](https://www.bostonglobe.com/2026/03/12/arts/jordan-rudess-jam-bot-mit/)
- **NIME 2026 follow-up.** "Enhancing Expressive Musical Conversation in the jam_bot" (Blanchard et al., NIME 2026, London, June 23–26) adds four things:
  - velocity modeling;
  - "increasing model throughput via a ggml implementation — required to accommodate the longer sequences induced by velocity modeling";
  - a call-and-response training modality across varying tempi, fine-tuned on "an expert dataset of MIDI sequences recorded on a piano with annotated calls and responses";
  - compensation for external MIDI output latency.

  It describes AMT as "GPT-2-based" and as sacrificing expression modeling for larger-scale pretraining. — [NIME 2026 PDF [S]](https://nime.org/proceedings/2026/nime2026_73.pdf); [MIT Media Lab publication page [S]](https://www.media.mit.edu/publications/enhancing-expressive-musical-conversation-in-the-jam_bot/)
- **Backends and hardware.** ONNX Runtime runs the ONNX models in C++ "faster than real-time (RTF > 1)", and a KV cache is "enabled to avoid redundant computation". The tested backends are ONNX Runtime and ggml on CPU (AMD Ryzen 9 7950X), CUDA (NVIDIA RTX 4090), and Apple Metal (M3 Max MacBook Pro). The tested models were fine-tuned from "a small pre-trained AMT model with 170M parameters and another … on a medium configuration". — [NIME 2026 [S]](https://nime.org/proceedings/2026/nime2026_73.pdf). The 170M figure conflicts with the AMT paper's 128M "Small". It may count embeddings or a different config; the PDF needs checking.
- **Repo and license.** Lancelot Blanchard's public GitHub lists 22 repositories, including forks `anticipation` (Apache-2.0), `levanterForAnticipation` (Apache-2.0), and `mlx` (MIT), but no jam_bot repository. — [GitHub profile [V via fetch]](https://github.com/lancelotblanchard?tab=repositories). The ISMIR/Zenodo paper record is CC-BY-4.0 [S](https://zenodo.org/records/17706584). jam_bot also received a MIDI Association innovation-award listing. — [midi.org [S]](https://midi.org/innovation-award/the-jam_bot)
- **StreamMUSE (related, AMT-style accompaniment), arXiv 2606.11886, June 2026.** "Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation", by Bowen Zheng, Andrew H. Yang, Jiaqi Ruan, Jia He, Xinyue Li, Yuan-Hsin Chen, Ziyu Wang, and Xiaosong Ma. It is a client-server system for frame-synchronous streaming inference: the model generates accompaniment conditioned on a live user melody, with frame intervals targeted "within 200 ms". It has two design variables: *future visibility* t_f (offset between the playback time and the latest conditioning input; t_f < 0 predicts ahead to absorb latency) and *output chunk duration* k. "System responsiveness is strongly correlated with music-quality metrics." — [arXiv HTML [S]](https://arxiv.org/html/2606.11886); [arXiv abs [S]](https://arxiv.org/abs/2606.11886). The GitHub repo `StreamMUSE/StreamMUSE-v1` exists (default branch `new_system_stanley`). Its README does not document the model ("LEKAI" checkpoint), and no license file was found. — [GitHub [V via fetch]](https://github.com/StreamMUSE/StreamMUSE-v1)
- **ReaLJam (CHI EA 2025).** Real-time human–AI jamming with RL-tuned transformers. It uses "anticipation": the agent continually predicts how the performance will unfold and visually shows its plan to the user. — [ACM DL [S]](https://dl.acm.org/doi/10.1145/3706599.3720227); [arXiv 2502.21267 [S]](https://arxiv.org/pdf/2502.21267)

### Inferences
- jam_bot is an existence proof that a fine-tuned AMT (Small or Medium) can improvise in tight real time against a live keyboardist. The ISMIR paper reportedly ran ONNX Runtime and KV cache on CUDA; the NIME 2026 follow-up adds a ggml port and an M3 Max backend. Both are the routes this project would need on Apple Silicon (ggml/Metal or MLX rather than PyTorch MPS).
- jam_bot fine-tuned on a single artist's recordings covering bass, chords, and melodies. That is the same shape as "player LoRA" in this project, but jam_bot apparently fine-tuned the whole model rather than using LoRA. The PDF needs checking to confirm.
- StreamMUSE's t_f/k decomposition maps directly onto the half-bar block scheduler: k = 0.94 s block, and t_f = how far the chord input must lead the generated audio.

### Gaps
- No public jam_bot code or MIT license was found. The assignment's "MIT license" claim is unverified; the only license found is CC-BY-4.0 on the paper.
- Exact latency/RTF tables (ONNX vs ggml vs Metal), whether int8 quantization was used, and the RTX 4090 numbers could not be retrieved (nime.org, zenodo, and arxiv were blocked).
- Whether jam_bot ran a formal evaluation with musicians (beyond the Rudess concert and the case study) is unconfirmed.
- Which AMT checkpoint the ISMIR version fine-tuned (360M medium vs small) is unconfirmed; NIME 2026 mentions both small (170M) and medium.

## Q3. Aria (EleutherAI, Bradshaw et al., ISMIR 2025): size, tokenizer, data, license, evaluations, embeddings, Aria-Duet realtime latency on Apple Silicon, and adding chord conditioning

### Takeaway
Aria is an Apache-2.0, LLaMA-style piano model (d_model 1536, 16 layers; ~0.65B parameters by my count from the released config, consistent with "650M") trained on ~60k hours of transcribed solo piano. Listeners preferred its continuations over AMT and MusicGen. Its MLX realtime duet demo reaches 200–300 ms time-to-first-note on M-series Macs. It has no chord control, but its code already contains an embedding-conditioned LM class, and its tokenizer config has a "jazz" genre prefix token. Adding chord conditioning is a fine-tuning job. Its training data (Aria-MIDI) is CC-BY-NC-SA.

### Cited Findings
- **Paper.** "Scaling Self-Supervised Representation Learning for Symbolic Piano Performance", by Louis Bradshaw, Honglu Fan, Alexander Spangher, Stella Biderman, and Simon Colton. ISMIR 2025; arXiv 2506.23869 (30 Jun 2025). — [GitHub README [V]](https://github.com/EleutherAI/aria); [arXiv PDF [S]](https://arxiv.org/pdf/2506.23869)
- **Model and data.** "Aria is a pretrained autoregressive generative model for symbolic music, based on the LLaMA 3.2 (1B) architecture, which was trained on ~60k hours of MIDI transcriptions of expressive solo-piano recordings." There are three checkpoints: `aria-medium-base`, `aria-medium-gen` (finetuned for continuation), and `aria-medium-embedding` (SimCSE-style contrastive). — [README [V]](https://github.com/EleutherAI/aria)
- **Exact config [V].** `medium.json`: `d_model 1536, n_heads 24, n_layers 16, ff_mult 4, max_seq_len 8192, vocab_size 17727`. `model.py` uses a gated MLP (`ff_gate_proj`, `ff_up_proj`, `ff_down_proj`) and an untied `lm_head`. — [medium.json [V]](https://github.com/EleutherAI/aria/blob/main/aria/config/models/medium.json); [model.py [V]](https://github.com/EleutherAI/aria/blob/main/aria/model.py). My count from this config is ≈ 658M parameters (16 × (4·1536² + 3·4·1536²) + 2 × 17727 × 1536), which matches the "~650M" in the assignment. The figure is computed, not quoted.
- **Tokenizer [V].** `AbsTokenizer`: "processes MIDI files in 5000ms segments, with each segment separated by a special <T> token." Each note is three tokens: [instrument, pitch, velocity], [onset in ms from segment start], [duration in ms]. Notes are ordered by onset, and "sustain pedal effects are incorporated directly into note durations". Prefix tokens for instrument, genre, composer, and form are prepended before `<S>`. The config sets `time_step_ms 10`, `max_dur_ms 5000`, `velocity_quantization_step 10`, `genre_names ["jazz","classical"]`, and composer names that are all classical. — [absolute.py [V]](https://github.com/EleutherAI/aria-utils/blob/main/ariautils/tokenizer/absolute.py); [config.json [V]](https://github.com/EleutherAI/aria-utils/blob/main/ariautils/config/config.json)
- **License.** "Our models and MIDI tooling are released under the Apache-2.0 license." — [README [V]](https://github.com/EleutherAI/aria). The Aria-MIDI dataset (1,186,253 MIDI files, ~100,629 h, ICLR 2025) is distributed under CC-BY-NC-SA 4.0. — [aria-midi GitHub [S]](https://github.com/loubbrad/aria-midi); [ICLR 2025 paper [S]](https://proceedings.iclr.cc/paper_files/paper/2025/file/f7f5f501282771c96bb3fedcc96bedfe-Paper-Conference.pdf)
- **Intended use and limits [V].** Aria "performs best when continuing existing piano MIDI files rather than generating music from scratch". It is "very sensitive to input quality" (no instruction tuning or RLHF) and may "memorize or closely reproduce" popular classical works. — [README [V]](https://github.com/EleutherAI/aria)
- **Listening evaluation.** 46 participants with ≥1 year of musical training compared 45-second continuations of 15-second solo-piano prompts on musical coherence (melodic development, rhythm, harmonic progression, style). They "consistently preferred" Aria over AMT and MusicGen. The paper also says it "remains competitive with proprietary audio generation models". — [arXiv HTML [S]](https://arxiv.org/html/2506.23869); [search summary [S]](https://arxiv.org/pdf/2506.23869)
- **Embeddings and fine-tuning efficiency.** Frozen contrastive embeddings reach SOTA linear-probe results on MIR classification, and direct fine-tuning "often requir[es] only a few hundred labeled examples to specialize to downstream tasks." — [arXiv [S]](https://arxiv.org/pdf/2506.23869)
- **Conditioning hook in code [V].** `TransformerLM_CND` is a "Transformer decoder with a language modeling head and optional conditioning". An `embedding_adapter` (Linear emb_size → d_model) prepends a conditioning embedding to the sequence. — [model.py [V]](https://github.com/EleutherAI/aria/blob/main/aria/model.py)
- **Fine-tuning recipes.** The README documents no LoRA/PEFT recipe; it only has generation and embedding CLIs and installs a `train` extra. — [README [V]](https://github.com/EleutherAI/aria)
- **Realtime demo [V].** "In `demo/` we provide an MLX (Apple Silicon) implementation of the real-time interactive piano-continuation demo". A demo-specific checkpoint (`model-demo.safetensors`) adds sustain-pedal control. Responsiveness "is dependent on … GPU memory bandwidth." The demo source sets `KV_CHUNK_SIZE = 256`, `PREFILL_CHUNK_SIZE = 16`, `RECALC_DUR_PREFILL_CHUNK_SIZE = 8`, `FIRST_ONSET_BUFFER_MS = -150`, and `MAX_STREAM_DELAY_MS = 50`. — [README [V]](https://github.com/EleutherAI/aria); [demo_mlx.py [V]](https://github.com/EleutherAI/aria/blob/main/demo/demo_mlx.py)
- **Aria-Duet ("The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity", arXiv 2511.01663, Nov 2025; NeurIPS 2025 demo slides).** The pianist plays, the model listens, and on a takeover signal it continues on the same Disklavier keys. Reported time-to-first-note is **200–300 ms on Apple Silicon (M-series)** and **10–20 ms on RTX 4090/5090**. Without mitigation, the compute-bound prefill gives "an unacceptable lag of 1000–2000 ms". The fix is "continuously fill[ing] chunks into the KV-Cache as the pianist performs" plus speculative re-evaluation of durations of notes cut short at takeover. — [arXiv abs [S]](https://arxiv.org/abs/2511.01663); [NeurIPS 2025 slides [S]](https://neurips.cc/media/neurips-2025/Slides/129323.pdf); [author PDF [S]](https://www.alexander-spangher.com/papers/aria_duet.pdf)

### Inferences
- **Chord conditioning is feasible by fine-tuning.** Either:
  - (a) Interleave chord-control tokens AMT-style: add new token IDs, initialize them randomly, and keep pretrained embeddings, as in Q5's "Equipping…" recipe.
  - (b) Prefix the chord progression per 5 s segment, after the `<T>` token.
  - (c) Reuse `TransformerLM_CND` to inject a per-block chord embedding.

  (a) and (b) need vocabulary growth plus LoRA on attention/MLP, and full-tuning the new embeddings. Because Aria's segments are 5 s with absolute onsets, chord tokens placed at each `<T>` boundary or interleaved near the notes they govern keep the AMT "locality" advantage.
- **Latency on a Mac.** Aria-Duet's 200–300 ms time-to-first-note on M-series uses MLX and continuous prefill with a ~650M model. A 128–360M AMT-sized model with the same tricks should fit the 0.4 s p99 block budget more comfortably. Aria-650M itself is borderline once per-block decoding of ~10–20 tokens is added to the prefill.
- **Style personalization hooks.** Aria's contrastive embedding model plus `TransformerLM_CND` gives a ready "style embedding → conditioned generation" path: embed Tatum or Mehldau clips and condition on them. It is a possible alternative or complement to per-player LoRA.
- **Data and license caution.** The weights are Apache-2.0 but were trained on CC-BY-NC-SA data. That is fine for a non-commercial hobby project but worth noting.

### Gaps
- Exact parameter count as stated in the paper (I computed ≈0.66B from config) and Aria's decode tokens/s on M-series could not be read (arxiv blocked).
- Whether the released base model was actually trained with the "jazz" genre prefix token active, and what share of Aria-MIDI is jazz, are not established.
- No public LoRA/adapter fine-tuning recipe for Aria was found.

## Q4. Other 2023–2026 open symbolic models: size, license, conditioning, data, weights, jazz evidence, inference cost

### Takeaway
Beyond AMT and Aria, the three most relevant candidates are:
- **Moonbeam** (Apache-2.0; 309M/839M; built-in LoRA recipes, a chord-conditioned generation recipe, and PiJAMA jazz-pianist experiments).
- **MIDI-GPT** (bar-level attribute controls including a pitch-class set, an "expressive" microtiming checkpoint, and macOS arm64 wheels; code MIT but weights possibly NC).
- **MIDI-RWKV** (edge-oriented, style adaptation by state tuning).

The ABC/LLM models (NotaGen, MuPT, ChatMusician, MIDI-LLM, Text2midi, MuseCoco) are text- or score-conditioned, classical- or pop-oriented, and too heavy for half-bar realtime. The diffusion models (Polyffusion, Whole-song) are chord-conditioned but pop-oriented and offline.

### Cited Findings
- **Moonbeam (Guo & Dixon, QMUL; arXiv 2505.15559, May 2025).**
  - Pretrained on "81.6K [hours] and 18 billion tokens" of MIDI; Apache-2.0 license file [V]. — [GitHub README/LICENSE [V]](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
  - Sizes ~309M (small) and ~839M (medium) [S].
  - Compound tokens (onset, duration, octave, pitch class, instrument, velocity) with Fundamental Music Embedding, and "Multidimensional Relative Attention" (RoPE-like rotations per musical dimension) [S]. — [review [S]](https://www.themoonlight.io/en/review/moonbeam-a-midi-foundation-model-using-both-absolute-and-relative-music-attributes)
  - Weights: the pretrained checkpoint is on HF [V link]; the conditional (CoMMU) and ATEPP-Bach fine-tuned checkpoints are listed as "TODO" [V].
  - Fine-tuning scripts use `--use_peft True --peft_method lora` with torchrun/bf16 [V].
  - A `conditional_gen_commu` branch has flags `--if_add_chords_in_transformer True --if_add_metadata_in_transformer True` [V].
  - A `finetune_player_classification` branch includes the **pijama30** dataset, with LoRA target modules q, k, v, o [V]. Moonbeam-M reached 67.9% accuracy on PiJAMA30 pianist classification [S].
  - Expert listeners rated the chord-conditioned fine-tune as fitting chord/metadata conditions better than a REMI transformer baseline, despite slightly lower objective pitch/velocity accuracy [S]. — [arXiv 2505.15559 [S]](https://arxiv.org/abs/2505.15559)
- **MIDI-GPT (Metacreation Lab; AAAI 2025, arXiv 2501.17011).**
  - GPT-2-style multitrack infilling model trained on GigaMIDI per the HF card summary [S]. — [HF card [S]](https://huggingface.co/Metacreation/MIDI-GPT)
  - The repo README (2026) lists checkpoints [V]:
    - `yellow_small/medium`: controls for note density, min/max polyphony, min/max note duration.
    - `prism_medium`: key signature, pitch range, silence proportion, note duration, bar-level note density and polyphony, **bar-level pitch class set**, and 18 genre groups; 4–16 bar contexts.
    - `expressive_medium`: adds sub-grid microtiming `delta` tokens, 128 velocity bins, and a `nomml` control. The docs recommend it for "jazz, solo piano".
  - Installs via `pip install "midigpt[inference]"` with prebuilt wheels for macOS arm64 [V]. — [README [V]](https://github.com/Metacreation-Lab/MIDI-GPT); [docs/models.md [V]](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md)
  - **License conflict:** the GitHub LICENSE is MIT (©2026) [V](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/LICENSE), while the HF model card was summarized as CC-BY-NC-4.0 [S](https://huggingface.co/Metacreation/MIDI-GPT).
- **MIDI-RWKV ("Adaptable/Personalizable Symbolic Music Infilling with MIDI-RWKV", arXiv 2506.13001, June 2025).**
  - A small RWKV-7 (linear-recurrent) model for multitrack long-context infilling "on edge devices" [S].
  - It outperforms MIDI-Mistral on 2/4/8-bar infilling [S].
  - Personalization by *state tuning* is covered in Q6. — [arXiv [S]](https://arxiv.org/abs/2506.13001)
- **MIDI-LLM (Wu, Carlton, Mikayawa, Kim, Donahue, Huang; ISMIR 2026; arXiv 2511.03942).**
  - Llama 3.2 1B with an extended AMT vocabulary, for text-to-MIDI [V README]. It runs on vLLM with FP8; the README recommends "a GPU with 16GB+ VRAM and CUDA 12.x" [V].
  - License: Llama 3.2 Community License [V](https://github.com/slSeanWU/MIDI-LLM/blob/main/LICENSE.md). — [README [V]](https://github.com/slSeanWU/MIDI-LLM)
  - Reported 7–13× faster than Text2midi [S]. — [arXiv [S]](https://arxiv.org/html/2511.03942)
- **Composer's Assistant 2 (Malandro; arXiv 2407.14700, July 2024).**
  - Interactive multitrack MIDI infilling inside REAPER; runs locally.
  - Main model: encoder-decoder, 512-dim, 16 + 16 layers (~3.5× the original CA).
  - Trained only on public-domain and permissively licensed MIDI [S].
  - Repo LICENSE is MIT [V](https://github.com/m-malandro/composers-assistant-REAPER/blob/main/LICENSE); a search summary called the paper CC-BY-4.0 [S](https://arxiv.org/html/2407.14700v1).
- **Whole-Song Hierarchical Generation (Wang, Min, Xia; ICLR 2024).**
  - Cascaded diffusion over form → counterpoint → lead sheet → accompaniment; "long-term control of chord progression" via cross-attention [S](https://proceedings.iclr.cc/paper_files/paper/2024/hash/39d239a30bb40536a5e4e78b5780ba82-Abstract-Conference.html).
  - Code is MIT [V](https://github.com/ZZWaang/whole-song-gen/blob/main/LICENSE). The README (Apr 2024) says generation with external control is "not released" and only "a portion of the model checkpoints" are public [V](https://github.com/ZZWaang/whole-song-gen).
- **Polyffusion (ISMIR 2023; arXiv 2307.10304; authors not verified in this session).**
  - Piano-roll diffusion; chord and texture conditions are encoded by pretrained VAEs and cross-attended in the UNet; it "can generate music scores that follow a given chord progression".
  - Paper CC-BY-4.0; the repo was reported as MIT. — [search summary [S]](https://github.com/aik2mlj/polyffusion); [demo [S]](https://polyffusion.github.io/). The repo README/LICENSE could not be fetched at the guessed paths (404), so the license is unverified.
- **NotaGen (IJCAI 2025; arXiv 2502.18008).**
  - ABC notation, classical. Pretrained on 1.6M pieces, fine-tuned on ~9K classical works with "period-composer-instrumentation" prompts, then CLaMP-DPO.
  - Sizes 110M/244M/516M [S](https://arxiv.org/pdf/2502.18008). Code MIT [V](https://github.com/ElectricAlexis/NotaGen/blob/main/LICENSE).
- **MuPT (ICLR 2025).** ABC; 190M–4.23B parameters on 33.6B tokens [S](https://arxiv.org/pdf/2404.06393). License not verified.
- **ChatMusician (ACL Findings 2024).** LLaMA2-7B continually pretrained on ABC (MusicPile) [S](https://aclanthology.org/2024.findings-acl.373/). License not verified.
- **Text2midi (AAAI 2025).** 272M total (159M trainable): frozen FLAN-T5 encoder plus an 18-layer decoder emitting REMI+; pretrained on SymphonyNet, fine-tuned on MidiCaps [S](https://arxiv.org/html/2412.16526v2). Code MIT [V](https://github.com/AMAAI-Lab/Text2midi/blob/main/LICENSE).
- **MuseCoco (2023).** 1.2B parameters; text → attributes → music, with attributes incl. instrument, rhythm, key, time signature, emotion [S](https://arxiv.org/pdf/2306.00110). `microsoft/muzic` is MIT [V](https://github.com/microsoft/muzic/blob/main/LICENSE).
- **SymPAC (ISMIR 2024).** "First to demonstrate … training symbolic generation models solely from auto-transcribed audio data", using MIR models for transcription, beats, and structure. Uses prompt bars plus *constrained generation via finite state machines* at inference [S](https://arxiv.org/abs/2409.03055). No public weights found.
- **Amadeus (ACL 2026; arXiv 2508.20665).** AR note sequence plus a bidirectional discrete-diffusion attribute level; "speedup of at least 4x"; Amadeus-S and inference code released 2025-08-28 [S](https://github.com/lingyu123-su/Amadeus); [ACL Anthology [S]](https://aclanthology.org/2026.acl-long.1898/). License not verified.
- **Tooling: MidiTok** (MIT [V](https://github.com/Natooz/MidiTok/blob/main/LICENSE)). Supports REMI and others with `use_chords=True`, plus "Attribute Controls … at the track-level or bar-level" with custom controls via `add_attribute_control` [V](https://github.com/Natooz/MidiTok/blob/main/docs/attribute_controls.rst).
- **BebopNet (ISMIR 2020; pre-window baseline).** Monophonic, harmony-constrained jazz LM trained on bebop-giant solos; "able to generate improvisations based on any given chord progression". Personalized via a learned per-note user-preference metric plus personalized beam search [S](https://archives.ismir.net/ismir2020/paper/000132.pdf). Code: [GitHub](https://github.com/shunithaviv/bebopnet-code) (license not checked).

### Inferences
- **Jazz competence evidence is thin everywhere.** The only direct jazz-piano touchpoints found are Moonbeam's PiJAMA30 classification fine-tune, Aria's "jazz" genre token and transcribed-YouTube corpus, MIDI-GPT `expressive` being recommended for jazz, and BebopNet (2020, monophonic). None of the open foundation models reports chord-conditioned *jazz solo* generation quality.
- **Moonbeam's compound-token design** emits one note per decoding step with multiple attribute heads, versus 3 tokens per note for AMT and Aria. It could be ~3× fewer sequential steps per block, attractive for the 0.4 s budget. This needs checking against its actual decoding loop. It is the only candidate with LoRA, chord-conditioning, and PiJAMA recipes all in one Apache-2.0 repo. Its trainers are torchrun/bf16/CUDA-oriented, so Apple-MPS training is unproven.
- **MIDI-GPT's** bar-level pitch-class-set control can act as a "chord-scale" control (set the per-bar PC set to the chord tones plus tensions). It requires a bar grid, which PiJAMA lacks (fixed 120 BPM transcription). Its infilling granularity is whole bars in 4–16-bar windows, not half-bar realtime blocks.
- **Diffusion models** (Polyffusion, Whole-song) are chord-conditioned out of the box but pop- and POP909-oriented, full-segment (non-streaming), and partly unreleased. They are unsuitable as the realtime core.

### Gaps
- Licenses of the MuPT, ChatMusician, Amadeus, and Polyffusion weights, and of the HF cards for MIDI-GPT and Moonbeam, could not be verified (HF blocked).
- XMusic, GigaMIDI-trained models other than MIDI-GPT, and PerTok-based models were not researched; I found no reliable details in this session.
- Composer's Assistant 2's exact user controls (pitch, rhythm, chord?) were not verified.

## Q5. Which models support chord/harmonic controls out of the box, and what techniques make controls stick

### Takeaway
Out-of-the-box harmonic control among open models:
- MIDI-GPT `prism`/`expressive`: bar-level pitch-class set plus key.
- Moonbeam: chord-conditioned fine-tuning recipe; weights not yet released.
- Polyffusion and Whole-song: chord cross-attention, pop data.
- AMT: arbitrary note-level controls, so chord tones can be given as control notes.
- Aria: none.

The literature also shows that prefix chord tokens are easy for a model to ignore, and that interleaving (locality), counterfactual/contrastive losses, and chord-first conditioning orders measurably improve adherence.

### Cited Findings
- **MIDI-GPT `prism_medium` / `expressive_medium`.** Key signature, pitch range, bar-level pitch class set, bar-level density/polyphony, and genre controls. — [docs/models.md [V]](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md)
- **Moonbeam.** A chord/metadata-conditioned generation and infilling recipe on CoMMU (`--if_add_chords_in_transformer True`), but its checkpoint is marked TODO. — [README [V]](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model)
- **Polyffusion.** Chord condition via a pretrained VAE plus cross-attention [S]. — [Polyffusion repo [S]](https://github.com/aik2mlj/polyffusion)
- **Whole-song.** Chord progression as an external condition; control generation is not released. — [README [V]](https://github.com/ZZWaang/whole-song-gen)
- **AMT.** Any note stream can be a control; the README example uses a melody instrument as controls. — [README [V]](https://github.com/jthickstun/anticipation)
- **MuseBarControl (arXiv 2407.04331, July 2024).** Plain bar-level fine-tuning (BFT) gave 65.27% chord accuracy. Adding prompt augmentation plus a **counterfactual loss** raised it to 78.33% (+13.06% bar-level controllability). The counterfactual loss swaps the correct bar prompt for a wrong one and penalizes the model if the likelihood of the true notes does not drop. It is "designed to ensure the model correctly responds to the prefix bar-level prompts and avoids the trivial solution of generating the next tokens solely based on previous tokens." — [arXiv [S]](https://arxiv.org/html/2407.04331)
- **"Chord-conditioned Melody and Bass Generation"** (Salem, Shokri, Devaney; NeurIPS 2025 AI4Music workshop; arXiv 2511.08755). Five Transformer strategies were compared (none, independent, bass-first, melody-first, co-generation). Chord conditioning "improves the replication of stylistic pitch content and chord tone usage", "particularly for the bass-first model." — [arXiv [S]](https://arxiv.org/abs/2511.08755)
- **AMT locality argument.** Prefix/Seq2Seq "places time-localized controls far from the events they describe"; interleaving keeps them near. — [arXiv [S]](https://arxiv.org/pdf/2306.08620)
- **Adding controls to a pretrained unconditional model ("Equipping Pretrained Unconditional Music Transformers with Instrument and Genre Controls", arXiv 2311.12257, Nov 2023).** The model was pretrained on 1.5M songs, then fine-tuned with prefix control tokens. Embeddings of existing tokens were initialized from the pretrained weights and new tokens randomly. It beat the base model in listening tests on coherence, harmony, and overall quality. — [arXiv [S]](https://arxiv.org/abs/2311.12257)
- **Classifier-free guidance.** It is used for chord conditioning in symbolic discrete-diffusion work (ViTex: chord inputs nulled with p = 0.5 in training). — [ViTex arXiv 2603.01984 [S]](https://arxiv.org/pdf/2603.01984)
- **SymPAC.** Finite-state-machine constrained decoding enforces structural/prompt constraints at inference. — [arXiv [S]](https://arxiv.org/abs/2409.03055)
- **Interpretability.** "Do Music Transformers Represent and Use Musical Key?" (public code release) trains GPT-2-style symbolic models, linearly probes key subspaces, and uses activation patching to swap the prompt key for a target key. It tests whether models represent key internally without explicit key or chord tokens. — [GitHub README [V]](https://github.com/masuyama-genki-eng/music-key-patching). Its quantitative results were not available.

### Inferences
- The project's symptom (a ~1% NLL gap between right and wrong harmony, and identical outputs across chords) is the textbook failure that MuseBarControl's counterfactual loss targets: the model minimizes NLL from local note history and treats the prefix chord as noise.
- Cheap fixes that do not need a bigger model:
  - (1) Move chord information next to the notes it governs: AMT-style interleaving, or chord tokens at every half-bar.
  - (2) Add a counterfactual or contrastive term (true chord vs. a transposed/wrong chord) to the loss.
  - (3) Use CFG-style condition dropout so that conditional-minus-unconditional logits can be amplified at inference.
  - (4) Keep a constrained-decoding or ranking safety net (FSM / chord-tone masks) for "no wrong notes" (SymPAC, BebopNet beam search).

### Gaps
- No open model was found that ships *chord-symbol*-conditioned jazz solo generation weights.
- Direct measurements of chord adherence (e.g., chord-tone ratio on strong beats) for AMT/Aria/MIDI-GPT under chord controls were not found.

## Q6. Style personalization: LoRA/adapters/prefix/state tuning on symbolic music transformers; data needs; forgetting of chord-following; recommended order

### Takeaway
LoRA on symbolic music models is now routine:
- Moonbeam ships LoRA recipes, including PiJAMA pianist experiments.
- Function-alignment (ISMIR 2025) uses LoRA plus adapters for music-for-music tasks.
- MIDI-RWKV shows that tiny state-tuning can beat LoRA when only a few samples exist.

I found no study that quantifies "minutes of a pianist needed" or measures forgetting of chord adherence after style LoRA. The safest recipe the evidence supports is to train the chord/control-conditioned base first and then apply the player LoRA on player data re-encoded with the *same* control format (inferred chords and LH controls), optionally keeping a counterfactual-control loss term during the LoRA.

### Cited Findings
- **Moonbeam LoRA recipes [V].** Unconditional style fine-tune on ATEPP-Bach with `--use_peft True --peft_method lora --lr 3e-4 --context_length 2048`. Conditional (chord + metadata) LoRA fine-tune on CoMMU with `--context_length 848`. Player classification with LoRA target modules q, k, v, o on pijama30, pianist8, and Giant_Piano_MIDI. — [README [V]](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model). In the classification fine-tune, "LoRA is applied while keeping the embedding layer for the classification token and the linear layer for classification fully trainable" [S]. — [review [S]](https://www.themoonlight.io/en/review/moonbeam-a-midi-foundation-model-using-both-absolute-and-relative-music-attributes)
- **MIDI-RWKV (June 2025).** *State tuning* (294k trainable parameters) was compared against LoRA r = 4 (331k) and r = 32 (2.7M), with matched training times of "~4–6 minutes on a single RTX 4090 or CPU backend". State tuning "generally achieves significantly better objective scores" (CP, GS, PCHE) "in the low-sample regime", with comparable F1. The authors note state tuning and LoRA are complementary: one changes the initial state, the other the weights. — [arXiv [S]](https://arxiv.org/abs/2506.13001)
- **"Versatile Symbolic Music-for-Music Modeling via Function Alignment" (ISMIR 2025; arXiv 2506.15548).** Uses a pretrained LM for both reference and target sequences, linked by a lightweight adapter: either cross-attentive adapters between two LMs or a self-attentive adapter in a shared LM, with "a trainable LoRA module … appended to the pretrained LM". Tasks include chord recognition, melody generation, and drum-track generation; "all demos, code and model weights are publicly available." — [arXiv [S]](https://arxiv.org/html/2506.15548v1); [ISMIR 2025 poster [S]](https://ismir2025program.ismir.net/poster_25.html)
- **Aria fine-tuning efficiency.** Pretrained representations specialize with "only a few hundred labeled examples" for classification tasks (not generation). — [arXiv [S]](https://arxiv.org/pdf/2506.23869)
- **jam_bot.** Personalization to one artist was done by fine-tuning AMT on that artist's recordings (bass, chords, melodies; later call/response annotations). — [MIT Media Lab [S]](https://www.media.mit.edu/articles/a-model-of-virtuosity/); [NIME 2026 [S]](https://nime.org/proceedings/2026/nime2026_73.pdf)
- **BebopNet.** Personalization via a learned per-user preference metric used inside beam search, rather than via weights: an inference-time reranking alternative to LoRA. — [ISMIR 2020 [S]](https://archives.ismir.net/ismir2020/paper/000132.pdf)
- **Adding control tokens at fine-tuning time.** Initialize new-token embeddings randomly and keep pretrained ones; listening quality improved over the base. — [arXiv 2311.12257 [S]](https://arxiv.org/abs/2311.12257)
- **LoRA forgetting, general NLP evidence (not music).** LoRA "still suffer[s] from catastrophic forgetting", with mitigation work such as Bayesian PEFT and SLIM. — [arXiv 2402.12220 [S]](https://arxiv.org/pdf/2402.12220); [SLIM, NAACL 2025 [S]](https://aclanthology.org/2025.naacl-long.246.pdf)
- **Audio-domain note.** MusicGen genre LoRA is few-shot. — [Complex & Intelligent Systems 2026 [S]](https://link.springer.com/article/10.1007/s40747-026-02285-5)

### Inferences
- **Recommended order, supported by the cited mechanisms:**
  - (1) Build a chord/control-conditioned base: AMT-360M or Moonbeam fine-tuned on all 235 bebop pieces plus any larger jazz corpus, with automatic chord/LH controls and a counterfactual-control loss.
  - (2) Train a player LoRA (Tatum/Mehldau) on that player's pieces encoded with the same controls (inferred chords), so that the LoRA learns style *conditional on* harmony.
  - (3) Regularize against forgetting chord-following: keep the counterfactual term active, mix ~10–30% base-style data into LoRA batches, use small rank (r = 4–16, attention-only q, k, v, o as in Moonbeam), and gate on a chord-adherence metric (strong-beat chord-tone ratio, wrong-chord NLL gap) before accepting a LoRA.
- **Data needs.** Very small per-player corpora (a few pieces) argue for the smallest adapters (state tuning / prefix / low-rank attention-only), per MIDI-RWKV's low-sample result, and against full fine-tuning.
- **Compute.** A LoRA on a 128–360M model at 1–2k context is plausibly trainable on an Apple Silicon Mac with ≥16 GB unified memory in fp32/bf16 at batch 1–4 with gradient accumulation, or on a free 16 GB Colab/Kaggle T4/P100. Full fine-tuning of 360M+ is more comfortable on the free cloud GPUs. This is an estimate; none of the cited repos documents MPS training.

### Gaps
- No paper found quantifies how many minutes of a pianist's playing a symbolic LoRA needs for recognizable style.
- No music study directly measures catastrophic forgetting of chord/control adherence caused by a style LoRA. The recommendation above is inferred from NLP evidence plus the control-adherence mechanisms in Q5.
- Free Colab/Kaggle GPU quotas and their current terms were not verified in this session.

## Q7. Is a 13M Music Transformer fundamentally too small, or is the bottleneck data/representation? Evidence on scale vs. conditioning adherence

### Takeaway
The evidence says scale clearly improves general musicality: best results at ~0.95B parameters, and Aria at ~0.65B beats AMT. The evidence on *conditioning adherence* points mainly to representation and objective (control locality, counterfactual/contrastive losses, conditioning order), not size. The project's 1% NLL gap is better explained by prefix-only guide-tone conditioning on a 10 ms event stream with no adherence pressure than by the 13M size alone. Moving to a pretrained 128–360M base would still likely improve lick vocabulary and phrase quality.

### Cited Findings
- **Lehmkuhl et al. (NeurIPS 2025 AI4Music workshop; arXiv 2511.07268).** "Generating Piano Music with Transformers: A Comparative Study of Scale, Data, and Metrics." A systematic comparison of datasets, architectures, sizes, and training strategies for symbolic piano. The best model is "a 950M-parameter transformer trained on 80K MIDI files from diverse genres", often rated human-composed in a Turing-style test. "Larger models and pre-training on diverse datasets significantly improve musical quality." It uses scaled-down Mistral architectures and pretrains on an Aria-MIDI subset. — [arXiv [S]](https://arxiv.org/abs/2511.07268)
- **Aria (~0.65B, 60k h).** Preferred over AMT (≤780M, Lakh-centric) and MusicGen for continuation coherence. — [arXiv [S]](https://arxiv.org/html/2506.23869)
- **MuPT.** Scaled ABC models from 190M to 4.23B (33.6B tokens). — [arXiv [S]](https://arxiv.org/pdf/2404.06393)
- **Adherence driven by objective and representation, not size:**
  - MuseBarControl: +13 points chord accuracy from prompt augmentation and counterfactual loss at fixed model size. — [arXiv [S]](https://arxiv.org/html/2407.04331)
  - AMT: interleaving keeps controls local versus prefix conditioning. — [arXiv [S]](https://arxiv.org/pdf/2306.08620)
  - Salem et al.: conditioning order (bass-first) changes chord-tone usage. — [arXiv [S]](https://arxiv.org/abs/2511.08755)
- **Small models can be made controllable.** MIDI-RWKV is explicitly a "small foundation model" for edge devices with effective infilling and personalization. — [arXiv [S]](https://arxiv.org/abs/2506.13001)
- **Pretraining matters for a small target corpus.** Pretraining plus control-token fine-tuning beat the base model on harmony in listening tests. — [arXiv 2311.12257 [S]](https://arxiv.org/abs/2311.12257)

### Inferences
- **Diagnosis for this project.** The 13M model's weak chord following is most likely an objective/representation problem: the chord is a prefix far from the notes in a dense 10 ms note_on/off/velocity stream, NLL is dominated by local continuation, and there is no counterfactual pressure. Scaling alone would not fix it. Evidence: MuseBarControl's gains at fixed size, and AMT's locality argument.
- **Where scale does matter.** Musical vocabulary (licks, enclosures, phrase shape) does improve with scale and diverse pretraining (Lehmkuhl et al.; Aria vs AMT). 2,777 pieces is small compared with Lakh (~170k files) or Aria-MIDI (~1.19M files).
- **Practical path.** Start from AMT-128M or 360M (Apache-2.0, pretrained, control-native, realtime-proven by jam_bot), re-tokenize the jazz data with chord/LH controls at a short δ, and add a counterfactual-chord loss. Then add a player LoRA and a constrained or ranking safety layer. This is a hypothesis to test with the existing harness metrics (token-identical rate across chords, wrong-chord NLL gap), not a proven result.

### Gaps
- No controlled study was found that varies model size while holding the conditioning scheme fixed and measures chord/control adherence in symbolic music.
- No published evaluation was found of any open foundation model on bebop-specific criteria (chromatic approach and enclosure rates, chord-tone landings on strong beats).
