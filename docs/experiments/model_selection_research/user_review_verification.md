# Verification of the pasted model-choice review (real-time chord-conditioned bebop piano solo, Apple Silicon)

Research date: 2026-10-03. Scope: check the factual claims in the user-pasted review, and judge its 1st/2nd recommendation order against the evidence and against this folder's earlier notes. The earlier notes are `chord_conditioned_jazz_models_and_datasets.md`, `symbolic_foundation_models_and_personalization.md`, `apple_silicon_realtime_inference.md`, `hybrid_lick_systems_and_comping.md` and `unlabeled_midi_chord_conditioning_data.md`.

**Tags**
- **[V]** = read directly from a primary page or file. Most [V] items come from git clones of public GitHub repos; papers that the authors committed into those repos count as primary.
- **[S]** = taken only from a search-engine snippet of the cited page.
- **[X]** = could not be found, or was contradicted.
- **[calc]** = my own arithmetic on [V] data. It is an inference, not a published number.

**Access.** The egress proxy blocked these hosts: nime.org, zenodo.org, media.mit.edu, arxiv.org, link.springer.com, eurecom.fr, huggingface.co, johnthickstun.com, jazzomat.hfm-weimar.de, dezrann.net, pure.mpg.de, api.semanticscholar.org, dblp.org, api.crossref.org, researchgate.net, hal.science and web.archive.org. GitHub (git clone, raw.githubusercontent.com, github.com pages) worked.

**Repo commits read:**

| Repo | Commit | Date |
|---|---|---|
| MINGUS | `3c4ac12` | 2024-07-14 |
| bebopnet-code | `9dfa800` | 2020-11-09 |
| BACHI_Chord_Recognition | `f0bead1` | 2026-05-16 |
| PiJAMA | `8498db5` | 2023-11-30 |
| anticipation | `af37397` | 2024-03-18 |

## Q1. jam_bot NIME 2026: ggml port, ONNX Runtime on macOS, and the M3 Max per-token numbers

### Takeaway
The paper's identity and the reason for the ggml port are confirmed. The paper is "Enhancing Expressive Musical Conversation in the jam_bot", NIME 2026. The authors ported to ggml partly because ONNX Runtime does not support macOS Metal acceleration well, and ggml lets the models run on Apple GPUs. The specific M3 Max figures could not be retrieved, so they are neither confirmed nor contradicted [X]:
- 170M: 10.13 ms/token, 98.76 tok/s
- 416M: 26.17 ms/token, 38.21 tok/s
- benchmark context of 120 tokens ≈ 30 notes

The conflict with our earlier notes resolves in the review's favour. "Only the ONNX Runtime CUDA backend can be used" describes the *first-iteration ONNX Runtime setup* (the problem). It is not the paper's conclusion about ggml.

### Cited Findings
- **Paper record.**
  - Title "Enhancing Expressive Musical Conversation in the jam_bot".
  - Lead author Lancelot Blanchard, with Perry Naseck, Joseph A. Paradiso and Cheng-Zhi Anna Huang among the authors.
  - Proceedings of NIME 2026, pp. 621–626.
  - [S] Sources: [NIME PDF](https://nime.org/proceedings/2026/nime2026_73.pdf); [Zenodo record 20784228](https://zenodo.org/records/20784228); [MIT Media Lab page](https://www.media.mit.edu/publications/enhancing-expressive-musical-conversation-in-the-jam_bot/).
- **Four contributions.** Velocity modeling; "increasing model throughput via a ggml implementation – required to accommodate the longer sequences induced by velocity modeling"; call-and-response training across tempi; MIDI output latency compensation. — [S] [NIME PDF](https://nime.org/proceedings/2026/nime2026_73.pdf); [Zenodo](https://zenodo.org/records/20784228)
- **Why ggml (this is the conflict resolution).**
  - A snippet paraphrases the paper: "In the first iteration of this work, they use ONNX Runtime … faster than real-time. However, only the ONNX Runtime CUDA backend can be used for the jam_bot initially due to limitations with macOS acceleration. To address this, they use the ggml framework."
  - A second snippet: "ONNX Runtime does not well support Apple macOS Metal acceleration, which is why they switched to ggml."
  - [S] [NIME PDF](https://nime.org/proceedings/2026/nime2026_73.pdf)
- **ggml effect.**
  - ggml "improves the model throughput by a factor of 2 for both model sizes when using CUDA."
  - On CPU it improves throughput "to a lesser extent".
  - It "allows the models to run on Apple accelerated AI hardware through an implementation of operators using the Metal framework"; the "ggml implementation for Apple Metal allows for the jam_bot to now perform on common Apple laptop and desktop computers."
  - [S] [NIME PDF](https://nime.org/proceedings/2026/nime2026_73.pdf)
- **Model sizes and hardware.** The paper tests "a small pre-trained AMT model (170 M parameters)" and "a medium pre-trained AMT model (416 M parameters)" on a Ryzen 9 7950X CPU, an RTX 4090 (CUDA) and an M3 Max MacBook Pro (Metal). — [S] [NIME PDF](https://nime.org/proceedings/2026/nime2026_73.pdf); earlier note [S] in `symbolic_foundation_models_and_personalization.md`
- **M3 Max numbers [X].** These four figures did not appear in any retrievable snippet, and the PDF, Zenodo and Media Lab pages are blocked:
  - 10.13 ms/token
  - 98.76 tok/s
  - 26.17 ms/token
  - 38.21 tok/s

  Searches for the exact strings returned nothing — [search](https://nime.org/proceedings/2026/nime2026_73.pdf). The "120 tokens ≈ 30 notes with velocity" benchmark context was also not found [X].
- **Earlier-notes figures (still [S], not re-retrieved):** ORT CUDA 4.00 ms/token (249.75 tok/s, 416M) and ORT CPU 118.55 ms/token. Throughput was a 15-s running average. — [NIME PDF](https://nime.org/proceedings/2026/nime2026_73.pdf) via `apple_silicon_realtime_inference.md`
- **No public code.** Blanchard's GitHub lists 22 repos: forks of `anticipation`, `levanterForAnticipation`, `mlx`, and `rvc-mlx`. None is a jam_bot or ggml AMT repo. — [V] [github.com/lancelotblanchard](https://github.com/lancelotblanchard?tab=repositories)

### Inferences
- **The 170M/416M figures are AMT Small/Medium [calc].**
  - `anticipation/vocab.py` gives `VOCAB_SIZE = 55028` [V]. The training configs give Small d=768/12 layers, Medium d=1024/24, Large d=1280/36 [V] — [lakh-small.yaml](https://github.com/jthickstun/anticipation/blob/main/train/lakh-small.yaml), [vocab.py](https://github.com/jthickstun/anticipation/blob/main/anticipation/vocab.py).
  - With tied input/output embeddings, the parameter counts come to ≈128M / 359M / 780M. These match the paper's 128M / 360M / 780M.
  - With untied embeddings (a separate LM head, e.g. after export), they come to ≈170M / 416M.
  - So jam_bot's 170M and 416M are almost certainly the public AMT Small and Medium counted with an untied head, not different models. This resolves the "170M vs 128M" question left open in `symbolic_foundation_models_and_personalization.md`.
- **The review's numbers are internally consistent.** 1000/10.13 = 98.7 and 1000/26.17 = 38.2. Velocity adds a 4th token per note (3 AMT tokens + 1 velocity token), and 120 tokens / 4 = 30 notes. Consistency is not verification, but nothing contradicts them.
- **If the figures are right:**
  - A 170M AMT on an M3 Max (a high-end, 400 GB/s chip) would decode a 9–15-token half-bar block in roughly 0.1–0.15 s plus prefill.
  - Base M1/M2 chips have about 1/4–1/6 the bandwidth (see `apple_silicon_realtime_inference.md`) and could be 2–4× slower, which is borderline against 0.4 s p99.
  - A 15–50M model would be comfortably inside budget on any M-series chip.
  - All of this is an estimate, not a measurement.

### Gaps
- The full benchmark table (per backend, device and size), whether int8/fp16 was used, and the exact context length all need the NIME PDF ([nime.org](https://nime.org/proceedings/2026/nime2026_73.pdf) or [Zenodo 20784228](https://zenodo.org/records/20784228)). All hosts were blocked.

## Q2. MINGUS (ISMIR 2021): harmonic-coherence table, listening test, and whether BebopNet was retrained

### Takeaway
Every number in the review's harmonic-coherence table is exact, read from the paper PDF committed in the MINGUS repo. The listening-test design is correct in substance, with three corrections:
- Listeners rated how much they *liked* each clip on a 1–5 scale. It was not a forced real-vs-generated choice.
- The group label is "music lover", not "enthusiast".
- Each clip had a drum beat and chords added.

BebopNet *was retrained on WJazzD* for the comparison: a reduced WJazzD with 28 songs removed. The raw ratings JSON in the repo shows mean liking of original 3.54, MINGUS 2.74 and BebopNet 2.21.

### Cited Findings
- **Harmonic coherence (Table 4, "Harmonic coherence on WjazzDB", chord % / scale %):**

  | Source | Chord-tone % | Chord-scale % |
  |---|---|---|
  | Original | 49.17 | 72.16 |
  | MINGUS | 51.81 | 77.49 |
  | BebopNet | 40.66 | 64.55 |
  | SeqAttn | 35.92 | 60.26 |

  It is defined as "the percentage of generated notes that are tones belonging to the current chord, or to the scale associated with it." — [V] [MINGUS_ISMIR2021.pdf in repo](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)
- **Retraining.**
  - "To obtain comparable results MINGUS, BebopNet and SeqAttn have been re-trained on the two datasets [WjazzDB, NottinghamDB]."
  - Footnote 7: "28 songs have been removed from the original dataset due to incompatibility with BebopNet, which was not recognising chords outside its internal dictionary; all models have been trained on this reduced version."
  - BebopNet could not be run on NottinghamDB at all ("code incompatibilities").
  - [V] [paper PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf); [supplementary PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)
- **Blind quiz design.**
  - 15 short melodies, average duration 20 s: 5 originals from WJazzD, 5 by MINGUS and 5 by BebopNet.
  - Each was rated "with a score from 1 to 5 based on how much they liked it".
  - All were rendered to audio "with a shuffle drum beat and chords".
  - "Users were unaware of which melodies were original."
  - Participants: "music lover (8 participants), music student (9), professional musician (11)".
  - Web app: mingus.tools.eurecom.fr.
  - [V] [paper PDF §6.1](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)
- **Results (authors' words).**
  - "listeners are capable of identifying the original musical phrases with different degrees of confidence, proportional to their level of musical expertise."
  - "MINGUS generations tend to be preferred by the users with respect to BebopNet generations, probably due to the greater harmonic coherence."
  - "There is still a clear difference between machine learning generated samples and original ones, especially when evaluated by high-skilled musicians."
  - The difference "may probably be more evident on long tracks."
  - [V] [paper PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)
- **Mean ratings [calc on V data].** From `D_evaluate/user_eval/TUNES_STATS.json`: 15 tunes × 28 ratings each.

  | Group | Original | MINGUS | BebopNet |
  |---|---|---|---|
  | All | 3.54 | 2.74 | 2.21 |
  | Professionals | 3.67 | 2.45 | 2.02 |
  | Students | 3.58 | 2.73 | 2.22 |
  | Music lovers | 3.33 | 3.15 | 2.45 |

  — [TUNES_STATS.json](https://github.com/vincenzomadaghiele/MINGUS/blob/master/D_evaluate/user_eval/TUNES_STATS.json)
- **All five "original" clips are horn solos:**
  - Lee Konitz (ATTYA)
  - Charlie Parker (Donna Lee)
  - Pepper Adams (How High the Moon)
  - Sidney Bechet (Summertime)
  - Chet Baker (There Will Never Be Another You)

  — [V] [E_docs/melodies](https://github.com/vincenzomadaghiele/MINGUS/tree/master/E_docs/melodies)
- **Representation.**
  - Per note: pitch, duration, current chord ×4 pitches, next chord ×4, bass, beat [0–3], offset [0–95].
  - The measure is split into **96 parts**, i.e. 24 per quarter in 4/4.
  - "Chords … are always represented by their four fundamental notes … chords with more than four notes have been cropped, the VII degree has been added to chords with less than four notes."
  - Two separate encoder-only (causal-masked) transformers for pitch and duration, each with 4 layers, 4 heads, hidden size 200, sequence length 35.
  - [V] [paper PDF §3](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf); [supplementary Table 1](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf)
- **Ablation (supplementary Table 4; WJazzD pitch model, 15 epochs).**
  - Pitch accuracy is 13.57% with no conditioning and 14.99% with the best set (D-C-B-BE-O); perplexity goes from 12.30 to 11.96.
  - The accuracy-optimal pitch model **excludes next chord (NC)**.
  - The released checkpoint nevertheless uses all features (`I-C-NC-B-BE-O`).
  - [V] [supplementary PDF](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf); [paper §3.2](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)

### Inferences
- **The chord-tone gap is between two per-note-conditioned models.** BebopNet also gets the chord on every note. The gap therefore reflects next chord + bass + split pitch/duration models + training details, not "conditioning vs. none".
- **A higher-than-real chord-tone rate is not proof of quality.** MINGUS's 51.81% versus 49.17% for real solos can also mean a conservative, "inside" line.
- **Per-note conditioning alone moved likelihood very little.** Accuracy rose about 1.4 points and perplexity fell about 3% at 15 epochs. That resembles this project's "~1% NLL gap between right and wrong chord" symptom. It is direct evidence that MINGUS-style features help but do not by themselves force strong chord dependence. This supports adding the review's contrastive/counterfactual term rather than relying on features alone.
- **The listening test is weak evidence.**
  - Small n (28) and 5 clips per system.
  - It measures liking, not harmonic correctness.
  - All clips are sax/trumpet idiom.
  - It shows MINGUS > BebopNet when both are trained on the same reduced WJazzD, not that either is adequate.
- **Size check.** MINGUS's models are tiny: d=200, 4 layers, roughly a few M parameters each [calc estimate]. They are an order of magnitude below the review's 15–50M.

### Gaps
- Per-group real-vs-generated identification rates (Figure 2) are a chart; no numeric table was published. The JSON ratings are the closest raw data.
- The total parameter count of MINGUS was not computed exactly.

## Q3. "Chord-Transformer" (2026, Springer)

### Takeaway
The paper exists. Its title is "A chord-controlled transformer for controllable and coherent music generation", published by Springer (DOI 10.1007/s40747-025-02210-2; the s40747 prefix is *Complex & Intelligent Systems*), around January 2026. It uses POP909 and LMD (pop/Lakh), not jazz. The architecture differs from the review's description: snippets describe *parallel fusion inside the decoder* of chord cross-attention and music self-attention, plus a chord-aligned positional encoding and a DP chord-extraction step. None of the following could be verified [X]:
- "separate chord encoder"
- "30 participants, 15 trained / 15 untrained, double-blind"
- "coherence 4.3/5 vs FIGARO 3.9"
- authors
- code or weights

### Cited Findings
- **Title, venue and data.**
  - "Chord-Transformer … for chord-conditioned symbolic music generation".
  - Published in a Springer journal in January 2026.
  - Experiments on "the Pop909 and LMD datasets".
  - [S] [link.springer.com/article/10.1007/s40747-025-02210-2](https://link.springer.com/article/10.1007/s40747-025-02210-2)
- **Method.**
  - "(1) a dynamic programming algorithm based on an energy function to extract the most salient chord progression from a given sequence."
  - "(2) a parallel fusion architecture within the Transformer decoder that synergistically combines chord cross-attention with music self-attention … enhanced by a chord-aligned positional encoding."
  - [S] [Springer](https://link.springer.com/article/10.1007/s40747-025-02210-2)
- **Evaluation and baselines [X].** Searches combining the paper with FIGARO, participant counts, double-blind design or coherence scores returned nothing — [search results](https://link.springer.com/article/10.1007/s40747-025-02210-2). The page is blocked, as are Crossref and Semantic Scholar.

### Inferences
- Even if the 4.3 vs 3.9 figure is real, it is a pop-music MOS against FIGARO. It says nothing about bebop chord-tone/avoid-note behaviour. It supports "cross-attention to an explicit chord stream helps adherence" only in general terms.
- Its DP chord extraction from note content is a possible alternative to BACHI for pseudo-labeling, but it is untested on jazz.

### Gaps
- Authors, participant design, the score table, model size, and code/weights availability. All need the Springer full text, which was blocked.

## Q4. BACHI (ICASSP 2026) symbolic chord recognizer

### Takeaway
Most of the review's claims are confirmed:
- MIT license
- pretrained checkpoints and inference code
- MIDI/MusicXML input
- root/quality/bass output
- trained on piano data with a piano pitch range

The checkpoints, however, are **classical (When-in-Rome + DCML) and pop (POP909-CL)** only. **No jazz training or evaluation** appears anywhere in the repo, and the quality vocabulary has no extensions, alterations or 6th chords.

### Cited Findings
- **Paper.** "BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked Iterative Decoding on Pop and Classical Music", by Mingyang Yao, Ke Chen, Shlomo Dubnov and Taylor Berg-Kirkpatrick, ICASSP 2026; arXiv 2510.06528. — [V] [README citation](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition); [S] [arXiv 2510.06528](https://arxiv.org/abs/2510.06528)
- **License.** "MIT License, Copyright (c) 2025 AndyWeasley2004". — [V] [LICENSE](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition/blob/main/LICENSE)
- **Checkpoints.**
  - "Classical Model: Trained on When-in-Rome + DCML corpus"
  - "Pop Model: Trained on POP909-CL"
  - Both are hosted as a Hugging Face *dataset* repo (`Itsuki-music/BACHI_Chord_Recognition`).
  - "The model is trained on piano data only and only supports pitch ranges of piano (21-108) on MIDI."
  - [V] [README](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition)
- **I/O.** Supported inputs are `.musicxml`, `.mxl`, `.xml`, `.mid` and `.midi`. Each output line is a chord change: "Beat position (in quarter notes)", root, quality, bass (e.g. `2.50 F_M_F`). CPU works (`--device cpu`). — [V] [README](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition)
- **Method.** Boundary detection, then iterative decoding of root, quality and bass in confidence order. — [S] [arXiv](https://arxiv.org/abs/2510.06528); [V] [README](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition)
- **Quality vocabulary:** M, m, o, +, D7, M7, m7, o7, /o7, mM7, +7, sus2, sus4, other, N. — [V] [data_process/vocab.py](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition/blob/main/data_process/vocab.py)
- **Grid:** `beat_resolution: 12`, `label_resolution: 2` (piano-roll frames per beat; labels at half-beat resolution). — [V] [config/film_kdec.yaml](https://github.com/AndyWeasley2004/BACHI_Chord_Recognition/blob/main/config/film_kdec.yaml)
- **Jazz.** A full-text grep of the repo for "jazz" returns nothing [V].

### Inferences
- **Mismatch with this project's data.** BACHI works on beat positions. PiJAMA MIDI has no beat grid (fixed-tempo transcription, no beats; see `unlabeled_midi_chord_conditioning_data.md` §1). BACHI would first need beat tracking or a lead-sheet alignment. Its classical/pop priors and 7th-chord-capped vocabulary will mislabel jazz voicings (rootless, 9/13/alt) as "other" or as the wrong root.
- **Useful at best for coarse tiers.** It could supply the review's coarser confidence tiers (root+quality) after beat alignment, with agreement checks against a simple LH/bass pitch-class template. It should not be trusted as a full-chord labeler for jazz piano without a validation set.

### Gaps
- Weight license on the HF dataset repo, and reported accuracy numbers. The arXiv and HF pages were blocked.

## Q5. BebopNet chord representation: "12-dim four-hot" vs "4 pitches through the pitch embedding"

### Takeaway
Both descriptions are partly right, at different layers:
- **Stored data:** a **13-dim** indicator vector (12 pitch classes + a no-chord slot) with exactly **4 ones**.
- **Network input:** the 4 indices are turned into **4 pitch IDs in the 5th octave (+72)**. These are embedded with **the same `encode_pitch` table** as melody notes and concatenated to the note embedding.

Our earlier code-verified note is the precise one for what the model sees. The review's "four-hot" is a loose description of the on-disk data.

### Cited Findings
- **Data layout.** "[18:31] 13 ints - chord pitches - indicators of participating pitches", alongside 13 scale-pitch indicators, the chord root and a chord-type index. — [V] [gather_data_from_xml.py L15–23](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/A_data_prep/gather_data_from_xml.py)
- **`chord_2_vec`.**
  - Sets `chord_notes[chord_pitch_indices[:4]] = 1` and asserts `sum(chord_notes) == 4`.
  - "No chord" sets the last (13th) slot.
  - Non-4-note chords go through `ensure_4_notes`:
    - `major` → `dominant` (i.e. C → C7)
    - triads → `+'-seventh'`
    - 9th/11th/13th → 7th
    - major-minor → minor-major-seventh

  — [V] [vectorXmlConverter.py L80–145](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/utils/music/vectorXmlConverter.py)
- **Model input.**
  - `get_chord_pitch_emb` takes `chord_pitches.view(-1, 13).nonzero()[:, 1] + OFFSET_TO_5OCT` (with `OFFSET_TO_5OCT = 72`), reshapes to (·, 4), and calls `self.encode_pitch(...)`, the same embedding used for melody pitch.
  - Then `word_emb = torch.cat((word_emb, chord_emb), 2)` unless `chord_bias` is on.
  - [V] [mem_transformer.py L521–538, L614–628](https://github.com/shunithaviv/bebopnet-code/blob/master/jazz_rnn/B_next_note_prediction/transformer/mem_transformer.py)
- **Released config.**
  - `"chord_bias": false`, `n_layer 4`, `d_model 400`, `n_head 8`, `d_inner 1028`, `mem_len 64`.
  - `pitch_sizes [130, 64]`, `duration_sizes [121, 64]`, `offset_sizes [48, 16]`.
  - `model.pt` is 26.3 MB.
  - [V] [training_results/transformer/model/args.json](https://github.com/shunithaviv/bebopnet-code/tree/master/training_results/transformer/model)

### Inferences
- **The chord dominates the input [calc].** Pitch embeddings are 64-d, so the chord contributes 4×64 = 256 of the 400 input dimensions, against 64 + 64 + 16 = 144 for pitch, duration and offset. That is a strong per-step chord signal.
- **The offset grid is 48 per bar (12 per quarter)** [V via `offset_sizes`]. That is coarser than MINGUS's 96 per bar.
- **Small checkpoint.** 26.3 MB in fp32 is about 6–7M parameters [calc]. The "BebopNet pretrained baseline" is a ~7M sax-data model, not a 15–50M one.
- **Practical caveat (repeated from earlier notes).** Typed triads become dominant 7ths. Send explicit 7th-chord symbols to the BebopNet baseline.

### Gaps
- The ISMIR 2020 paper's own wording of the chord encoding was not re-read; archives.ismir.net was blocked.

## Q6. Weimar Jazz Database: beat table fields, Dezrann exposure, license, piano share

### Takeaway
The beats-table fields are confirmed directly from WJazzD CSV exports committed in the MINGUS repo: `bar, bass_pitch, beat, chord, chorus_id, form, onset, signature`. "333 scores on Dezrann vs 456 solos" and the ODbL license are supported by snippets only. **Piano is about 6 of 456 solos** (Herbie Hancock ×5, Red Garland ×1), counted by performer name. That is about 1.3%.

### Cited Findings
- **Beats CSV header** (456 beats files + 456 melody files):
  - Beats: `bar,bass_pitch,beat,chord,chorus_id,form,onset,signature`
  - Melody: `bar,beat,beatdur,denom,division,duration,num,onset,period,pitch,subtatum,tatum`
  - [V] [MINGUS A_preprocessData/data/WjazzDBcsv](https://github.com/vincenzomadaghiele/MINGUS/tree/master/A_preprocessData/data/WjazzDBcsv); parser use of `chord`, `bass_pitch`, `signature`, `onset` in [wjazzDB_csv_to_json.py](https://github.com/vincenzomadaghiele/MINGUS/blob/master/A_preprocessData/wjazzDB_csv_to_json.py)
- **Jazzomat schema.** "The beats table is solely used for WJD-type melodies to store (tapped) beats along with chord, form, bass pitch and other annotations"; fields include `beatid, melid, onset, bar, beat`, and `bass_pitch` ("fractional MIDI"). — [S] [Jazzomat dbformat](https://jazzomat.hfm-weimar.de/dbformat/dbformat.html)
- **Size.** "456 instrumental jazz solos from 343 different recordings" with "meter, structural segmentation, measures, beats, chord labels, style, solo instrument". — [S] [JSD paper, TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.131); MINGUS paper also says "456 improvisations" [V] [paper](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf)
- **Dezrann.** The WJazzD corpus there is described as "333 transcriptions of jazz solos from 1925–2009" / "330+ high-quality jazz transcriptions, some of them with synchronized audio." — [S] [Dezrann TISMIR paper](https://transactions.ismir.net/articles/10.5334/tismir.212); [dezrann.net](https://www.dezrann.net/)
- **License.** "released under the Open Data Commons Open DataBase License (ODbL) … version 2.1 with 456 solo transcriptions." — [S] [Jazzomat download](https://jazzomat.hfm-weimar.de/download/download.html)
- **Performer counts from the 456 file names [V data, calc classification]:**
  - Piano: Herbie Hancock 5, Red Garland 1.
  - Vibraphone: Milt Jackson 6, Lionel Hampton 6.
  - Guitar: Pat Metheny 4, Pat Martino 1, John Abercrombie 1.
  - The rest are horns, e.g. Coltrane 20, Miles Davis 19, Parker 17.
  - [MINGUS csv_beats listing](https://github.com/vincenzomadaghiele/MINGUS/tree/master/A_preprocessData/data/WjazzDBcsv/csv_beats)

### Inferences
- **Usable fields.** WJazzD gives exactly the per-beat fields the review wants: chord, bass pitch, beat/bar, form. But it is ~98% non-piano, monophonic lines, and earlier notes put it at about 200k notes. Training a "piano" soloist on it means learning horn phrasing (breath-length phrases, horn range), which is the weakness the review itself acknowledges.
- **Piano share is a name-based count.** The ~6 piano solos come from performer names, not the DB's instrument field. Treat it as approximate.

### Gaps
- The exact Dezrann page counts (456 listed vs 333 with scores) and the WJazzD instrument field could not be read directly; both hosts were blocked.

## Q7. Stanford CRFM AMT checkpoints: training data and license

### Takeaway
The review is correct for Small (128M) and Medium (360M): both are Lakh-MIDI-trained, and 800k-step checkpoints exist (`music-small-800k`, `music-medium-800k`). The one checkpoint trained on *more* than Lakh is **`music-large-800k` (780M)**. Its card snippet lists Lakh + MetaMIDI + FMA transcripts + 450k transcribed commercial recordings. The code is Apache-2.0 [V]. The model cards say "all code and models" are Apache-2.0 [S], with a caveat about copyrighted training data.

### Cited Findings
- **`music-small-800k`.** "Small (128M parameter) Transformer trained for 800k steps on arrival-time encoded music from the Lakh MIDI dataset." "All code and models are released under the Apache License, Version 2.0", with a note that the Apache license "may not fully reflect the legal status" because Lakh files are often derivative of copyrighted music. — [S] [HF card](https://huggingface.co/stanford-crfm/music-small-800k)
- **`music-medium-800k`.** "Medium (360M parameter) Transformer trained for 800k steps … from the Lakh MIDI dataset". Lakh is described as 178,561 MIDI files, 1.99B arrival-time tokens, 8,943 h. — [S] [HF card](https://huggingface.co/stanford-crfm/music-medium-800k)
- **`music-large-800k`.** "Large (780M parameter) Transformer trained for 800k steps on … the Lakh MIDI dataset, MetaMidi dataset, and transcripts of the FMA audio dataset and 450k commercial music records (transcribed using … Magenta's ISMIR 2022 music transcription model)". Also listed: `music-large-100k` and `music-small-ar-100k`. — [S] [HF card](https://huggingface.co/stanford-crfm/music-large-800k); [HF search result](https://huggingface.co/stanford-crfm/music-small-ar-100k)
- **Repo.**
  - License: "licensed under the terms of the Apache License, Version 2.0".
  - The README loads `stanford-crfm/music-medium-800k`.
  - The repo's Levanter configs (`lakh-small/medium/large.yaml`) all point at Lakh data with `num_train_steps: 100000` and `seq_len: 1024`. These are the paper-era 100k configs; the 800k runs are not in the repo.
  - [V] [anticipation README](https://github.com/jthickstun/anticipation); [train/](https://github.com/jthickstun/anticipation/tree/main/train)

### Inferences
- **Correction to an earlier note.** `symbolic_foundation_models_and_personalization.md` said all sizes were Lakh-trained. Large-800k is not Lakh-only [S].
- **Both live-feasible AMT sizes are pop/Lakh-heavy.** For a small or medium model on a Mac, the AMT route starts from a Lakh-only prior, which has little bebop.

### Gaps
- The HF license tag fields and full card text were not opened (huggingface.co blocked).

## Q8. FiloSax and PiJAMA quick confirmations

### Takeaway
Both are confirmed from primary files:
- **FiloSax:** individual/research-group, non-commercial research only, no redistribution.
- **PiJAMA:** CC BY-NC 3.0 Unported, 2,777 performances, 120 distinct artist names, about 219–224 h.

### Cited Findings
- **FiloSax license text:**
  1. "may only be used by the individual signing below and by members of the research group or organisation of this individual. This permission is not transferable."
  2. "may be used only for non-commercial research purposes."
  3. "may not be sold, leased, published or distributed to any third party without written permission."

  — [V] [filosax README, License](https://github.com/dave-foster/filosax)
- **PiJAMA.**
  - LICENSE is "Attribution-NonCommercial 3.0 Unported".
  - `pijama.csv` has 2,777 rows and 120 unique `artist` values (107 unique MusicBrainz artist IDs).
  - The sum of `duration_sec` is 223.6 h (219.4 h within performance start/end windows).
  - 1,994 studio and 783 live.

  — [V/calc] [PiJAMA repo](https://github.com/almostimplemented/PiJAMA)

### Inferences
- "120 pianists / 200+ h" holds. The 120 is an artist-name count; MusicBrainz IDs give 107, so some names are variants or groups.

### Gaps
- None material.

## Q9. Precedent or counter-evidence for the review's design proposals

### Takeaway
Each proposal has partial precedent, but none has been validated on jazz:
- **24/quarter grid + microtiming residual:** MINGUS uses 96 per bar (= 24/quarter). PerTok and MIDI-GPT "expressive" encode quantized time plus a microtiming remainder.
- **Wrong-chord margin loss:** close to MuseBarControl's counterfactual loss.
- **Frozen-condition-path LoRA with replay:** only generic precedent (Moonbeam LoRA recipes, replay mixing). No study measures chord-adherence forgetting.
- **Resolution-aware validator:** supported by pedagogy and by an NCT-resolution metric in Salem et al. No study tests it as a reranker.

### Cited Findings
- **Grid.**
  - MINGUS: "sampling each measure into 96 equally sized parts", offset [0–95], "precision up to 8th note triplets and dotted 16th notes" — [V] [MINGUS paper §3.1](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/MINGUS_ISMIR2021.pdf).
  - BebopNet: 48 offsets per bar — [V] [args.json](https://github.com/shunithaviv/bebopnet-code/tree/master/training_results/transformer/model).
  - AMT: absolute 10 ms bins, no beat grid (earlier notes) — [V] [anticipation config](https://github.com/jthickstun/anticipation).
- **Microtiming residual.**
  - PerTok: "Timeshift tokens are expressed as the nearest quantized value based upon beat_res … The microtiming shift is then characterized as the remainder from this quantized value." Options include `use_microtiming`, `max_microtiming_shift` and `num_microtiming_bins`.
  - Notes take 2–5 tokens (TimeShift, Pitch, Velocity, MicroTiming, Duration).
  - [V] [MidiTok pertok.py](https://github.com/Natooz/MidiTok/blob/main/src/miditok/tokenizations/pertok.py)
  - MIDI-GPT `expressive_medium` adds sub-grid microtiming `delta` tokens and is recommended for "jazz, solo piano" — [V via earlier notes] [MIDI-GPT docs/models.md](https://github.com/Metacreation-Lab/MIDI-GPT/blob/main/docs/models.md).
- **Counterfactual / contrastive chord loss.**
  - MuseBarControl reports 65.27% chord accuracy with plain bar-level fine-tuning, rising to 78.33% with prompt augmentation + a counterfactual loss that penalizes the model if true-note likelihood does not drop under a wrong bar prompt.
  - [S] [arXiv 2407.04331](https://arxiv.org/html/2407.04331) (via earlier notes)
  - The review's hinge form `max(0, m + NLL(correct) − NLL(corrupt))` is a margin variant of the same idea. I found no paper using exactly this hinge, nor strong-beat or phrase-end weighting, in music.
- **Classifier-free guidance.**
  - MIDI-LLM applies CFG per token because MIDI context dominates the text prompt — [S] [arXiv 2511.03942](https://arxiv.org/pdf/2511.03942).
  - The MIDI-LLM README does not mention CFG — [V] [README](https://github.com/slSeanWU/MIDI-LLM).
  - ViTex nulls chord inputs with p = 0.5 for CFG in discrete diffusion — [S] [arXiv 2603.01984](https://arxiv.org/pdf/2603.01984).
- **Resolution-aware validation.** Salem et al. define a "Non-Chord-Tone Resolution Score" (non-chord tones resolving by step to a chord tone) as an evaluation metric — [S] [arXiv 2511.08755](https://arxiv.org/abs/2511.08755). Earlier notes give the strong-beat chord-tone / off-beat stepwise-resolution rules and the repetition collapse caused by hard masks — [hybrid_lick_systems_and_comping.md] citing [TISMIR 2022](https://transactions.ismir.net/articles/10.5334/tismir.87).
- **LoRA after a conditioned base.** Moonbeam ships LoRA recipes, including a chord-conditioned LoRA fine-tune — [V via earlier notes] [Moonbeam repo](https://github.com/guozixunnicolas/Moonbeam-MIDI-Foundation-Model). No study was found that measures forgetting of chord adherence after a style LoRA (earlier notes Q6).
- **Counter-evidence that features alone are weak.** MINGUS's ablation shows only 13.57% → 14.99% pitch accuracy from all conditioning features — [V] [supplementary](https://github.com/vincenzomadaghiele/MINGUS/blob/master/E_docs/Supplementary_material_MINGUS_ISMIR21.pdf). This argues *for* adding the contrastive term rather than against it.

### Inferences
- **Confidence-tiered pseudo-labels and the comping cost** are new as stated. I found no precedent for confidence tiers (full chord / root+quality / pitch-class mask). For the comping cost (voice movement + register overlap + solo semitone collision, computed after the solo block), earlier notes found rule-based voice-leading comping but no solo-collision-aware cost. Both are reasonable engineering, not evidence-backed.
- **Grid vs. PiJAMA timing.** A 24/quarter grid needs a beat grid, which PiJAMA lacks. The ±40 ms residual is only meaningful after beat alignment. This adds pipeline risk not mentioned in the review.

### Gaps
- No jazz-specific test of any of the four proposals was found.

## Q10. Overall assessment of the review's position and ordering

### Takeaway
The review is factually solid where it could be checked:
- MINGUS numbers are exact.
- BACHI facts are correct, apart from the missing caveat "pop/classical only".
- AMT Small/Medium being Lakh-only is correct.
- The jam_bot ggml/Metal rationale is correct.

Its unverified items are the jam_bot M3 Max per-token numbers and every Chord-Transformer evaluation detail. Most of its design is consistent with our notes:
- per-note chord conditioning
- a counterfactual loss
- LoRA after the base, with replay
- a soft, resolution-aware validator instead of masks
- MLX/ggml rather than MPS

Its **ordering** (from-scratch small model first, AMT second) is **not more strongly supported** than AMT-first or corpus-recombination-first. The decisive unknown is data, not architecture, and a cheap A/B is warranted before committing.

### Cited Findings
- **Agrees with earlier notes:**
  - Per-note chord features (BebopNet/MINGUS) are the documented way to make jazz models chord-aware — [chord_conditioned_jazz_models_and_datasets.md] citing [MINGUS](https://github.com/vincenzomadaghiele/MINGUS) and [BebopNet code](https://github.com/shunithaviv/bebopnet-code).
  - Counterfactual loss as the fix for "chord treated as noise" — [S] [MuseBarControl](https://arxiv.org/html/2407.04331).
  - Avoid MPS for batch-1 decode; prefer MLX or ggml — [apple_silicon_realtime_inference.md].
- **Differs from earlier notes.** Our "practical path" started from AMT-128M/360M (Apache-2.0, control-native, realtime-proven) — [symbolic_foundation_models_and_personalization.md Q7]. The review demotes AMT to second.
- **Data reality for the from-scratch route:**
  - Chord-labeled data: WJazzD 456 solos, about 6 of them piano [V/calc above], plus Omnibook (50 Parker solos, CC BY 4.0) and FiloSax (research-only) [V].
  - Piano data with no chord labels: PiJAMA, 2,777 performances, ~220 h [V].
- **Reference model sizes:** MINGUS is d=200, 4 layers per model [V]. BebopNet is d=400, 4 layers, ~6–7M parameters [V/calc]. Neither comes close to 15–50M, and neither was validated as adequate by experts: in the MINGUS quiz, professionals rated MINGUS 2.45 versus 3.67 for real solos [calc on V].
- **Latency:** jam_bot ran AMT 170M/416M on M3 Max via ggml/Metal [S]. Exact speeds are unverified [X].

### Inferences
- **Where the review is right to prefer a small purpose-built model.**
  - It can use a beat-relative grid and explicit current/next chord + bass + beat features. AMT's absolute 10 ms grid has no notion of beat or chord.
  - Latency is trivially inside 0.4 s on any M-series chip.
  - BebopNet is a cheap baseline: weights exist, MIT license. MINGUS is a clean design reference: LGPL weights, exact feature recipe.
- **Weak points the review under-weights:**
  - **(a) Parameter count versus data.** 15–50M parameters on ~0.2–0.3M chord-labeled horn notes invites memorization. The precedent models are 2–7M. The size only makes sense if the model is pretrained on PiJAMA with pseudo-labels, and that pipeline (beat tracking → labels) is itself unproven. BACHI is pop/classical, has no extensions, and needs beats.
  - **(b) Idiom.** WJazzD (~98% non-piano), MINGUS and BebopNet are all horn idiom. Our notes found no system that is simultaneously piano-idiom, chord-conditioned and expert-validated.
  - **(c) MINGUS's own ablation.** Feature conditioning alone moves likelihood only slightly, so success hinges on the contrastive loss, which is unvalidated in jazz.
  - **(d) Quality claims.** No evidence shows a 13–50M model reaching adequate bebop quality. The only expert-tested jazz generator with near-passing results is rule-guided (Frieler & Zaddach 2022, in earlier notes).
- **Arguments for AMT-first.**
  - It has a pretrained musical prior.
  - It handles "LH or comp = control, RH = events" natively, so it can train on all PiJAMA without chord labels by using the LH as control. At runtime, the review's deterministic shell comping can generate that control.
  - It has a working realtime precedent on Apple GPUs (jam_bot ggml/Metal).
  - Against it: the Small/Medium prior is Lakh pop, the pretrained δ = 5 s needs re-tokenization, and per-token cost is 3–10× a 15–50M model.
- **Arguments for corpus-recombination-first** (earlier hybrid notes):
  - It is the only route where harmonic correctness comes "by provenance".
  - It is near-zero latency and needs no training.
  - Its weaknesses are seams and limited novelty.
  - For a personal realtime demo it may deliver usable output soonest.
- **Suggested way to decide (inference, not evidence).** Run the same harness metrics on four arms, all measuring the wrong-chord NLL gap, token-identity across chords, strong-beat chord-tone rate, and salient-clash rate:
  1. BebopNet pretrained with explicit 7th chords: hours of work.
  2. A small (5–20M) MINGUS-style model with the contrastive term on WJazzD + Omnibook.
  3. AMT-Small fine-tuned on PiJAMA with LH-as-control at short δ.
  4. A lick-retrieval baseline with the same validator.

  Pick the ordering from those numbers, not from the review's priors. The review's validator, comping and runtime pieces are shared by all arms and can be built first.

### Gaps
- jam_bot M3 Max per-token numbers [X].
- Chord-Transformer evaluation details [X].
- BACHI's accuracy on jazz piano (no data exists).
- Any published model of 15–50M with chord conditioning evaluated on bebop.
