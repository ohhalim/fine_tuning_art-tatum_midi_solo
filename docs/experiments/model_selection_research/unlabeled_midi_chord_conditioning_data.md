# Turning unlabeled, audio-transcribed jazz piano MIDI into chord-conditioned training data (+ dataset licensing)

Research date: 2026-10-03. Scope: PiJAMA-family MIDI (Kong-style high-res transcription, fixed 120 BPM, no chords, no beats) to training data for a chord-conditioned bebop solo generator that runs on Apple Silicon.

Verification labels used below:
- **[V-file]**: I read the primary file myself (GitHub repo README/LICENSE/CSV/code, cloned in this session).
- **[V-snip]**: taken from a search-engine extract of the named primary page. Direct fetches of zenodo.org, arxiv.org, transactions.ismir.net, archives.ismir.net, jazzomat.hfm-weimar.de and huggingface.co were **blocked by this session's egress proxy**, so I could not read those pages in full. Treat these numbers as likely but not fully verified.
- **[Inf]**: my inference. No source states it.

---

## 1. Beat/downbeat tracking: performance MIDI vs. original audio; is PiJAMA audio available; do PiJAMA or JTD already provide beats?

### Takeaway
PiJAMA does **not** ship beats, but every one of its 2,777 performances has a YouTube URL. That makes it practical to re-acquire the audio and run an audio beat tracker (Beat This!, MIT-licensed), then map the MIDI onto that beat grid. Beat tracking on jazz is reliable at roughly F≈0.96 on trio mixtures. **Downbeat** tracking is not reliable (F≈0.41 in JTD's own check), so plan on beat-level grids plus a separate way to find bar or chord phase. The Jazz Trio Database (MIT license) already provides piano-solo MIDI with beats and manually anchored downbeats. It is the best ready-made "piano MIDI + beat grid" jazz resource.

### Cited Findings
**PiJAMA (Edwards, Dixon & Benetos, TISMIR 2023)**
- Over 200 h of solo jazz piano: 2,777 performances by 120 pianists from 244 albums, transcribed automatically to MIDI. Studio and live recordings are mixed, and audio tagging marks applause and speech. [V-snip] — [TISMIR article](https://transactions.ismir.net/articles/10.5334/tismir.162); [Zenodo record 8354955](https://zenodo.org/records/8354955); [project page](https://almostimplemented.github.io/PiJAMA/)
- The metadata `pijama.csv` has these columns: `id, artist, album, title, recording_condition, midi_filepath, mp3_filepath, youtube_url, duration_sec, performance_start_sec, performance_end_sec, split, acoust_id, mb_album_id, mb_artist_id, mb_track_id, mb_recording_id, agreement, discogs_idx`. **All 2,777/2,777 rows have a `youtube_url`.** There is **no beat, tempo or chord column**. [V-file, counted myself] — [PiJAMA GitHub repo, pijama.csv](https://github.com/almostimplemented/PiJAMA)
- The repo includes `scripts/download_youtube_audio.py`, which uses `youtube_dl` to fetch audio from each row's `youtube_url`, plus `transcribe_hr.py` (high-res transcription) and `compute_start_end_times.py`. [V-file] — [PiJAMA repo scripts/](https://github.com/almostimplemented/PiJAMA)
- Per-artist counts in `pijama.csv`: Art Tatum 122, Brad Mehldau 72, Oscar Peterson 56, Hank Jones 44, Barry Harris 35, Thelonious Monk 31, Tommy Flanagan 11, Red Garland 8, Bill Evans 6. Recording conditions: 1,994 studio and 783 live. Splits: 2,227 train, 272 val, 278 test. [V-file, counted myself] — [PiJAMA repo](https://github.com/almostimplemented/PiJAMA)
- I found no later work that released beat or downbeat timestamps for PiJAMA. (See Gaps.)

**Jazz Trio Database, JTD (Cheston, Schlichting, Cross & Harrison, TISMIR 2024)**
- 1,294 recordings, chosen from 4,659 evaluated, totaling 44.5 h of jazz **piano solos with bass and drums**. Annotations are automatic: onset, beat and downbeat timestamps for each performer, plus **MIDI for the piano soloist**, produced through source separation. [V-snip] — [TISMIR 10.5334/tismir.186](https://transactions.ismir.net/articles/10.5334/tismir.186); [Cambridge repository](https://www.repository.cam.ac.uk/items/cf8f6655-2f15-49e7-a08b-ba258d3c63b1)
- Each track directory holds `*_onset.csv` for piano, bass and drums, a `beats.csv` (onsets matched to beats, estimated metre and downbeats), `piano_midi.mid` and `metadata.json`. Metadata includes `time_signature`, a **manually annotated `first_downbeat`**, a YouTube link, and solo start/end timestamps. [V-file] — [JTD docs: database-structure (repo source)](https://github.com/HuwCheston/Jazz-Trio-Database)
- JTD exposes both `downbeats_auto` (from the beat tracker's metre) and `downbeats_manual`, which is built by extrapolating the manual `first_downbeat`. [V-file] — [JTD docs: onsetmaker-classes](https://github.com/HuwCheston/Jazz-Trio-Database)
- Beat and downbeat tracking in JTD uses madmom `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor` on the full mixture. [V-file] — [JTD src/detect/onset_utils.py](https://github.com/HuwCheston/Jazz-Trio-Database)
- In JTD's parameter-optimisation results (`references/parameter_optimisation/beat_track_mix.csv`, 34 manually annotated tracks), mean **beat F-measure ≈ 0.958** but mean **automatic downbeat F-measure ≈ 0.408**. These are in-sample tuning figures, not a held-out test. [V-file, computed myself from the CSV] — [JTD repo](https://github.com/HuwCheston/Jazz-Trio-Database)
- Onset annotations scored a mean F-measure of 0.94 against ground truth. [V-snip] — [TISMIR 10.5334/tismir.186](https://transactions.ismir.net/articles/10.5334/tismir.186)
- JTD is now in `mirdata` (`mirdata.initialize('jtd')`). Audio (mixed and unmixed) is on a Zenodo record and is granted **only on request for valid research projects**. [V-file] — [JTD README](https://github.com/HuwCheston/Jazz-Trio-Database); [Zenodo audio record 13828030](https://zenodo.org/records/13828030)
- JTD includes bebop-relevant pianists, for example Bud Powell ("Confirmation", "Salt Peanuts"), Kenny Drew, Thelonious Monk, Oscar Peterson, Tommy Flanagan, Hank Jones, Bill Evans and Brad Mehldau (track IDs in the docs data explorer). [V-file] — [JTD repo docs/_static/data-explorer](https://github.com/HuwCheston/Jazz-Trio-Database)

**Audio beat trackers (to run on re-acquired PiJAMA audio)**
- **Beat This!** (Foscarin, Schlüter & Widmer, ISMIR 2024) is a transformer beat/downbeat tracker that needs no DBN post-processing. Models: `final0-2` (~78 MB each) and `small0-2` (~8.1 MB). It falls back to CPU when CUDA is absent. **Code and published weights are MIT-licensed**, though the README warns that some training data is copyrighted or under limited CC licenses. [V-file] — [CPJKU/beat_this README](https://github.com/CPJKU/beat_this)
- Beat This! reports **80.7% downbeat F1 on RWC Jazz** without a DBN. [V-snip] — [arXiv 2407.21658](https://arxiv.org/pdf/2407.21658)
- **madmom**: source code is BSD-style, but **all model and data files (pickled processors included) are CC BY-NC-SA 4.0**. Commercial use requires contacting JKU. [V-file] — [madmom LICENSE](https://github.com/CPJKU/madmom)
- **BeatNet** (ISMIR 2021) does joint beat, downbeat, tempo and meter tracking with streaming, real-time, online and offline modes. It is **CC BY 4.0**. [V-file] — [mjhydri/BeatNet](https://github.com/mjhydri/BeatNet). A successor, BeatNet+, appears in TISMIR. [V-snip] — [BeatNet+ TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.198)

**Beat tracking directly on performance MIDI**
- **PM2S** (Liu, Kong, Morfi & Benetos, ISMIR 2022, "Performance MIDI-to-Score Conversion by Neural Beat Tracking") uses RNN heads for beats/downbeats, key, time signature, note value and **hand part**. It was trained on **A-MAPS, Classical Piano MIDI and ASAP**, all classical, and is MIT-licensed. [V-file] — [cheriell/PM2S README](https://github.com/cheriell/PM2S) and [dev/README](https://github.com/cheriell/PM2S); [ISMIR 2022 paper PDF](https://archives.ismir.net/ismir2022/paper/000047.pdf)
- **Murgul & Heizmann (SMC 2025; arXiv 2507.00466, Jul 2025)** describe an end-to-end encoder-decoder transformer that translates performance MIDI into beat annotations. It uses dynamic augmentation and optimized tokenization, and it beats an HMM baseline and PM2S on almost every dataset (A-MAPS, ASAP, GuitarSet, **Leduc: 239 jazz guitar performances**) for both beat and downbeat F1. Reported GuitarSet scores are beat F1 92.0 and downbeat F1 88.1. PM2S was still about 6% better on ASAP beats. [V-snip] — [arXiv 2507.00466](https://arxiv.org/abs/2507.00466); [Zenodo 15838779](https://zenodo.org/records/15838779)
- The same group followed with transformer **rhythm quantization of performance MIDI using beat annotations** (arXiv 2508.19262, Aug 2025; arXiv 2604.22290, Apr 2026), reporting onset F1 97.3% and note-value accuracy 83.3% on ASAP. [V-snip] — [arXiv 2604.22290](https://arxiv.org/html/2604.22290v1); [arXiv 2508.19262](https://arxiv.org/html/2508.19262)
- **Beyer & Dai (arXiv 2410.00210, 2024)** do end-to-end piano performance-MIDI-to-score conversion with transformers and compound tokens (3.5× shorter sequences). They predict staff assignment and notational details. [V-snip] — [arXiv 2410.00210](https://arxiv.org/pdf/2410.00210)
- A 2026 preprint, "Masked diffusion enables coherent beat tracking", exists (arXiv 2608.04624). I did not read it. [V-snip, title only] — [arXiv 2608.04624](https://arxiv.org/pdf/2608.04624)

### Inferences
- [Inf] **Recommended route: audio beat tracking, then map onto the MIDI.** PiJAMA's MIDI was transcribed from the same audio its `youtube_url` points to, so audio beats (Beat This!, MIT) and MIDI onsets share a timeline up to a constant offset. `performance_start_sec` and a cross-correlation of onset envelopes can recover that offset. This is more robust than MIDI-only trackers, which were trained on classical piano or guitar and have not been evaluated on swung solo jazz piano.
- [Inf] Re-downloading YouTube audio is a grey area under YouTube's Terms of Service. Keep the audio local and transient (derive beats, then discard), and never redistribute it.
- [Inf] Solo piano is harder than trio. Out-of-tempo intros and codas (very common in Tatum and in Mehldau ballads) and stride or rubato passages will defeat any beat tracker. Gate segments by local tempo stability, for example coefficient of variation of inter-beat intervals within a window, and by tracker activation confidence. Train only on segments that pass.
- [Inf] Downbeats: JTD's 0.41 automatic downbeat F shows that bar phase is unreliable even with bass and drums present. For solo piano, assume worse. Two options: (a) use a beat-relative representation with no bar lines, or (b) infer bar and chord phase by aligning a lead-sheet chord chart to the beat sequence (see Section 3), which fixes phase and chord labels in one step.
- [Inf] JTD is a strong **supplementary** corpus. Its trio piano solos are more metrically regular than solo piano and come with beats and manual downbeat anchors. Its MIDI comes from source-separated audio, so expect lower transcription precision than PiJAMA, especially in the left hand under bass.

### Gaps
- I found no released beat or downbeat annotations for PiJAMA, and no paper evaluating beat tracking on PiJAMA specifically.
- I could not obtain JTD's held-out beat and downbeat accuracy from the paper (fetch blocked). The 0.958/0.408 figures are in-sample optimisation numbers.
- I could not get per-dataset Beat This! numbers beyond RWC-Jazz downbeat F1. MIDI-only beat trackers have no published accuracy on swung jazz piano.
- Not verified: that PiJAMA MIDI carries a constant 120 BPM header. This is the user's statement; the default of Kong et al.'s inference package was not checked.

---

## 2. Hand / voice separation (melody vs. left-hand comping)

### Takeaway
PiJAMA has no hand labels. Neural hand-separation models trained on classical piano reach about 91–94% note accuracy. PM2S ships an open (MIT) hand-part head. Skyline melody extraction is only about 80% accurate even on pop piano. For bebop piano (single-line right hand over left-hand shells) a learned or heuristic split is workable. For stride (Tatum) it is fragile.

### Cited Findings
- PiJAMA's metadata has no hand or voice fields. Each performance is one transcribed piano MIDI file. [V-file] — [PiJAMA pijama.csv](https://github.com/almostimplemented/PiJAMA)
- **PM2S** includes an `RNNHandPartModel` (binary per-note hand prediction, evaluated with frame-wise F-measure) in its open MIT code. It was trained only on classical datasets. [V-file] — [PM2S dev/evaluation/hand_part.py](https://github.com/cheriell/PM2S)
- **Hadjakos et al. 2019, "Detecting Hands from Piano MIDI Data"**: RNNs assigning notes to hands reach **93.25% (real-time)** and **94.47% (non-real-time)** accuracy and beat earlier heuristics. [V-snip] — [GI Digital Library](https://dl.gi.de/items/969b1f5c-6eb0-4761-a35c-ac1b067c4991); [PDF](http://www.cemfi.de/wp-content/papercite-data/pdf/hadjakos-2019-detectinghands.pdf)
- A search extract (provenance unclear, possibly the 2026 fingering paper below) reports LSTM 93.02%, Transformer-decoder 91.51% and a zone-based heuristic 73.14% for hand separation. A 2026 statistical model for fingering-annotated piano transcription reports 90.8% hand-separation accuracy. [V-snip] — [arXiv 2609.28787](https://arxiv.org/html/2609.28787v1)
- **Skyline melody extraction** (keep the highest note) is the most common MIDI melody method. One evaluation reported skyline accuracy 79.52% (precision 81.42%, recall 56.57%). The extract groups this with the MidiBERT-Piano work (POP909 pop piano, not jazz). A "revised skyline" adds a time-overlap parameter. [V-snip] — [MidiBERT arXiv 2107.05223](https://arxiv.org/pdf/2107.05223); [Melody extraction on MIDI files](https://ieeexplore.ieee.org/document/1565863/definitions)
- Beyer & Dai (2024) predict staff assignment, a close proxy for hand, from performance MIDI with a transformer. [V-snip] — [arXiv 2410.00210](https://arxiv.org/pdf/2410.00210)

### Inferences
- [Inf] For bebop-style material (Bud Powell or Barry Harris lineage: single-note right hand plus sparse left-hand shells or rootless voicings), use a **two-stage split**: (1) a learned hand model (PM2S head, or a small BiLSTM trained on synthetic data), then (2) monophonic voice extraction inside the right hand (skyline with an overlap tolerance and octave-doubling merge). This should yield a usable "solo line" stream.
- [Inf] For **Art Tatum and stride**, the left hand leaps between bass notes and mid-register chords, and right-hand runs cross the whole keyboard. Pitch-split heuristics will mislabel many notes. Either exclude stride or rubato sections from "solo line" training, or treat the whole texture as accompaniment context (Section 4).
- [Inf] Hand-separation errors matter less if the conditioning signal is a per-beat pitch-class set of the accompaniment (Section 4). A few melody notes leaking into the "accompaniment" chroma add small noise, whereas a wrong explicit chord label is categorical noise.

### Gaps
- I found no hand-separation or melody-extraction accuracy figures on **jazz** piano MIDI.
- I could not verify PM2S's published hand-part F-measure (paper fetch blocked).

---

## 3. Automatic chord recognition (symbolic and audio) on jazz; lead-sheet alignment; iReal Pro status

### Takeaway
Off-the-shelf chord recognition is weak on jazz. JAAH's baseline was about 42%, and ChordSync's chord-to-audio alignment F1 on jazz was 0.435. Symbolic Roman-numeral models are trained on classical scores and need quantized input. A more practical route for PiJAMA is **title-matched lead-sheet charts aligned to the beat grid**. An iReal-derived chart set already matches **31.4% of PiJAMA performances, including 87 of 122 Art Tatum tracks**, by exact normalized title. Chord progressions themselves are generally treated as uncopyrightable, but iReal-derived charts carry no explicit open license.

### Cited Findings
**Jazz-specific accuracy**
- **JAAH** (Eremenko, Demirel, Bozkurt & Serra, ISMIR 2018): 113 jazz recordings (Smithsonian collections) annotated with beats, beat-aligned chords and structure labels. Jazz chord estimation by existing algorithms was **about 42%**, lower than on other datasets, because of extensive seventh chords and improvised harmonic interpretation. [V-snip] — [ISMIR 2018 PDF](https://archives.ismir.net/ismir2018/paper/000206.pdf); [Zenodo 1290737](https://zenodo.org/record/1290737); [MTG/JAAH repo](https://github.com/MTG/JAAH)
- The JAAH repo provides JSON annotations, chroma features, `labs.zip`, and a "Jazz5Functions" chord comparison in a MusOOEvaluator fork. [V-file] — [MTG/JAAH README](https://github.com/MTG/JAAH)
- "Transcribing Lead Sheet-Like Chord Progressions of Jazz Recordings" (Computer Music Journal 44(4), 2020) addresses lead-sheet-style jazz chord output. [V-snip, title and venue only] — [MIT Press CMJ](https://direct.mit.edu/comj/article/44/4/26/108550/Transcribing-Lead-Sheet-Like-Chord-Progressions-of)
- **ChordSync** (Poltronieri, Presutti & Rocamora, SMC 2024) is a Conformer that aligns chord annotations to audio without prior weak alignment. On **jazz**: precision 0.4663, recall 0.4129, **F1 0.4350**. Performance drops for under-represented genres such as jazz and classical. [V-snip] — [SMC 2024 paper](https://smcnetwork.org/smc2024/papers/SMC2024_paper_id205.pdf); [arXiv 2408.00674](https://arxiv.org/pdf/2408.00674)
- **ChordFormer** (arXiv 2502.11840, Feb 2025) is a Conformer for large-vocabulary audio chord recognition. It reaches 72.28% on the "sevenths" metric versus 67.84% for the prior best. These are pop/rock benchmarks, not jazz. [V-snip] — [arXiv 2502.11840](https://arxiv.org/html/2502.11840)
- Other 2025 audio chord-estimation work exists: training on artificially generated audio (arXiv 2508.05878) and consonance-based training (arXiv 2509.01588). I found no jazz numbers for either. [V-snip, titles] — [arXiv 2508.05878](https://arxiv.org/pdf/2508.05878); [arXiv 2509.01588](https://arxiv.org/pdf/2509.01588)

**Open models**
- **BTC** (Park et al., ISMIR 2019) is a bi-directional transformer chord recognizer. `--voca True` enables a large vocabulary, and output is lab files plus MIDI. MIT license. [V-file] — [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19)
- **Harmony Transformer** (Chen & Su, ISMIR 2019) does multi-task chord recognition with chord segmentation. A search extract says it beats BLSTM baselines by 5–6% on symbolic datasets. [V-file for description; V-snip for the 5–6%] — [Tsung-Ping/Harmony-Transformer](https://github.com/Tsung-Ping/Harmony-Transformer); [ISMIR 2019 PDF](https://archives.ismir.net/ismir2019/paper/000030.pdf)
- Functional harmony recognition with multi-task BLSTM (Chen & Su, ISMIR 2018) reached about 25.69% on full Roman-numeral labels (classical). [V-snip] — [ISMIR 2018 PDF](https://archives.ismir.net/ismir2018/paper/000178.pdf)
- **AugmentedNet** (Nápoles López et al., ISMIR 2021; MIT) does Roman-numeral analysis from MusicXML. On its full classical test set it scores Key 82.9, Degree 67.0, **Quality 79.7**, Inversion 78.8, **Root 83.0**, and full RN 46.4. Training uses synthetic block-chord templates re-texturized at every transposition. [V-file] — [napulen/AugmentedNet README](https://github.com/napulen/AugmentedNet)
- **ChordGNN** (Karystinaios, ISMIR 2023; MIT) is a graph neural network for Roman-numeral analysis that takes score files such as MusicXML. [V-file] — [manoskary/chordgnn](https://github.com/manoskary/chordgnn)
- A semi-CRF symbolic chord recognizer reports 83.2% event-level accuracy on classical data. [V-snip] — [ResearchGate: Segmental CRF chord recognition](https://www.researchgate.net/publication/330129068_Chord_Recognition_in_Symbolic_Music_A_Segmental_CRF_Model_Segment-Level_Features_and_Comparative_Evaluations_on_Classical_and_Popular_Music)

**Lead-sheet charts and alignment**
- **ChoCo** (Chord Corpus): 20,080 JAMS files with 20,530 Harte-notation chord annotations. Jazz-relevant partitions: JAAH (113, audio-aligned), **The Real Book (2,486, symbolic)**, **Weimar Jazz Database (456, audio-aligned leadsheet)**, **iReal Pro (2,000+)**, Band-in-a-Box (5,000+), Jazz Corpus (76). [V-file] — [smashub/choco README](https://github.com/smashub/choco)
- **iRb corpus** (Broze & Shanahan): 1,186 jazz standards in a Humdrum `**jazz` encoding, CC BY 4.0. [V-snip] — [Zenodo 3546040](https://zenodo.org/records/3546040); [iRb release note](http://www.stacoscimus.com/irb-corpus-released/)
- **mikeoliphant/JazzStandards** is a JSON chord set (title, key, rhythm, time signature, sections with bar-level chords and endings) pulled from the iReal Pro main playlists. 1,377 unique titles. [V-file] — [mikeoliphant/JazzStandards](https://github.com/mikeoliphant/JazzStandards)
- **Title-match coverage (computed myself):** exact normalized-title matching of PiJAMA titles against JazzStandards matched **873/2,777 performances (31.4%)**. By artist: **Art Tatum 87/122**, Thelonious Monk 24/31, Barry Harris 13/35, Oscar Peterson 12/56, Hank Jones 12/44, Brad Mehldau 7/72 (mostly originals or pop covers). [V-file, my computation over the two repos] — [PiJAMA](https://github.com/almostimplemented/PiJAMA); [JazzStandards](https://github.com/mikeoliphant/JazzStandards)
- **iReal Pro's position:** iReal Pro deals only in chord changes (no melodies, no lyrics) on the basis that chord progressions, rhythms and titles are not copyrightable. It still received a publisher takedown threat, which concerned **song titles**. The forum removes posts with lyrics. [V-snip] — [iReal Pro forum "Copyrights and rules"](https://forums.irealpro.com/threads/copyrights-and-rules.48/); [iReal Pro forum "Melody?"](https://forums.irealpro.com/threads/melody.12751/)

### Inferences
- [Inf] **Pipeline for chord labels (preferred over blind ACR):**
  1. Match the PiJAMA title to an iRb or iReal chart.
  2. Get beats from audio (Section 1).
  3. Unroll the chart form into choruses.
  4. Align the chart's bar-level chord sequence to the beat sequence with DTW or an HMM. Use, per beat, a bass-weighted pitch-class profile from the MIDI (left-hand notes weighted higher) scored against chord templates (root, 3rd, 7th) as the local cost. Allow chorus repeats, intro and coda skipping, and tempo-free starts.
  5. Keep only segments whose alignment cost and chord-template fit pass a threshold.

  This yields bar phase and chord labels together, and fixes the downbeat problem from Section 1.
- [Inf] Performers reharmonize (tritone subs, ii–V insertions, pedal points), so chart labels describe what was implied, not what was played. That still matches the runtime task: the user types lead-sheet chords and the soloist should play over them.
- [Inf] **How label errors affect training:** at about 42–45% chord accuracy (JAAH baseline, ChordSync jazz F1), blind ACR labels are wrong more often than right at fine vocabulary. A model trained on them learns a weak or contradictory chord–note mapping, which reproduces the current "ignores chord prompts" problem. Mitigations: (a) collapse labels to root plus a coarse quality class (maj6/maj7, dom7, min7, m7b5, dim7, alt), where accuracy is much higher; (b) keep per-segment confidence and drop or down-weight low-confidence segments; (c) prefer chart alignment for standards, and use ACR only as a local consistency check.
- [Inf] Symbolic Roman-numeral models (AugmentedNet, ChordGNN) need quantized score input and classical-style harmony, so they fit unquantized swung jazz MIDI poorly. A simple beat-synchronous template matcher over left-hand and bass pitch classes is probably as good and far cheaper on a Mac.

### Gaps
- I found no published evaluation of symbolic chord recognition on **jazz piano performance MIDI**.
- I could not get a 2023–2026 audio ACR model evaluated on JAAH with MIREX-style sevenths or majmin7 scores (paper pages blocked). ChordFormer's numbers are pop-only.
- Not verified: the license on the iReal Pro forum content itself and the iReal Pro app ToS text. The JazzStandards repo has **no LICENSE file** (fetch returned 404).
- Not verified: the WJD/JAAH alignment format, which I could not read directly.

---

## 4. Using the left hand / accompaniment itself as the control signal (anticipation-style) instead of explicit chord symbols

### Takeaway
Conditioning on the accompaniment (or a pitch-class summary of it) removes the need for chord labels and uses all of PiJAMA. The Anticipatory Music Transformer (Apache-2.0) is a proven open framework for "generate stream A given interleaved controls from stream B". At runtime the typed chord can be rendered as a left-hand voicing or a pitch-class set. I found **no paper that directly compares chord-symbol conditioning with accompaniment conditioning**. My recommended hybrid is a **per-beat pitch-class-set control**: it can be extracted from data without labels, and it can be produced exactly from typed chords at runtime.

### Cited Findings
- **Anticipatory Music Transformer** (Thickstun, Hall, Donahue & Liang; arXiv 2306.08620, Jun 2023) is an autoregressive symbolic model that conditions on arbitrary subsets of future "control" events by interleaving them ("anticipation"), designed for infilling and accompaniment. [V-snip] — [arXiv 2306.08620](https://arxiv.org/pdf/2306.08620)
- Human evaluation of 15-second accompaniments conditioned on a prompt plus the full melody, against human compositions: **18 wins, 31 ties, 11 losses (p = 0.194)**, so no significant preference. [V-snip] — [arXiv 2306.08620](https://arxiv.org/pdf/2306.08620); [author PDF](https://johnthickstun.com/assets/pdf/anticipatory-music-transformer.pdf)
- The anticipation repo builds anticipatory training datasets and runs sampling but does not train (the authors used Levanter). Pretrained models such as `stanford-crfm/music-medium-800k` are on the Hugging Face Hub. Code is **Apache-2.0**. [V-file] — [jthickstun/anticipation README](https://github.com/jthickstun/anticipation)
- **Aria** (EleutherAI; Bradshaw & Colton, ICLR 2025) is a pretrained piano model (`aria-medium-base`, `-gen`, `-embedding`) with **an MLX implementation for Apple Silicon** and a real-time interactive piano-continuation demo in MLX. Models and tooling are **Apache-2.0**. [V-file] — [EleutherAI/aria README](https://github.com/EleutherAI/aria)
- **Aria-MIDI** has 1,186,253 MIDI files (~100,629 h) of transcribed solo piano with genre, composer and performer metadata, under **CC BY-NC-SA 4.0** plus a disclaimer. [V-file] — [loubbrad/aria-midi README](https://github.com/loubbrad/aria-midi)
- **Chord-symbol baselines:** the Chord-Conditioned Melody Transformer (CMT) reached chord-tone ratio 72.53% overall and 80.63% on beat 1, against a dataset baseline of 71.42% and 79.61%. [V-snip] — [Chord Conditioned Melody Generation With Transformer Based Decoders](https://www.researchgate.net/publication/350023802_Chord_Conditioned_Melody_Generation_With_Transformer_Based_Decoders)
- Salem (arXiv 2511.08755, Nov 2025) compares transformer strategies for chord-conditioned melody and bass with music-theory metrics (pitch content, interval size, chord-tone usage). [V-snip] — [arXiv 2511.08755](https://arxiv.org/pdf/2511.08755)
- **Jazz Transformer** (Wu & Yang, ISMIR 2020) models WJD lead sheets with chord events decomposed into CHORD-TONE, CHORD-TYPE and CHORD-SLASH. [V-snip] — [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307)
- **BebopNet** (Haviv Hakimi, Bhonker & El-Yaniv, ISMIR 2020 best research paper) is a chord-conditioned monophonic bebop improvisation model trained on XML transcriptions of bebop players. The code is released, but the dataset must be supplied by the user. [V-snip + V-file] — [ISMIR 2020 PDF](https://archives.ismir.net/ismir2020/paper/000132.pdf); [shunithaviv/bebopnet-code](https://github.com/shunithaviv/bebopnet-code)
- **JazzSAMBA** (arXiv 2609.34931, Sep 2026): new multitrack jazz-combo recordings of 76 standards by 8 musicians (drums, bass, piano, horns), with timed bar, chord, section and soloist annotations. Intended to support "chart-conditioned accompaniment". A `jazzsamba` package is on PyPI. [V-snip] — [arXiv 2609.34931](https://arxiv.org/abs/2609.34931); [PyPI jazzsamba](https://pypi.org/project/jazzsamba/0.1.0/)

### Inferences
- [Inf] **Pros of accompaniment conditioning:** no chord recognition needed; uses 100% of PiJAMA, JTD and Aria-MIDI jazz; captures real voice-leading and comping rhythm; anticipation already supports "melody given accompaniment" (just swap which stream is the control).
- [Inf] **Cons for this product:** the runtime input is typed chord symbols. Rendering them into a left-hand voicing creates a train/test mismatch, because real pianists' left hands (Tatum stride, Powell shells, Mehldau counter-lines) differ from a canned voicing. The model may also learn to copy or echo the accompaniment rhythm rather than the harmony.
- [Inf] **Recommended hybrid: "pitch-class-set per beat" control.**
  - *Training:* compute, per beat, a 12-dimensional multi-hot (or duration-weighted) pitch-class vector from left-hand or non-melody notes (Section 2) plus the lowest note as a separate "bass PC" token.
  - *Runtime:* convert the typed chord symbol to its pitch-class set and bass. This is deterministic, so there is no voicing mismatch.
  - *Optional label smoothing:* snap each training vector to the nearest chord template in a jazz vocabulary, giving a chord token derived without labels while keeping the raw vector as a soft feature.

  This is "explicit chord conditioning" whose labels come from the data, not from a noisy ACR model or from charts.
- [Inf] Anticipation-style interleaving of control tokens a fixed lookahead ahead of the generated note also suits **real-time** use: the user types the next chord about a beat ahead, and the model sees it as an anticipated control.
- [Inf] Aria (MLX, Apache-2.0) is the most practical open pretrained base for Apple Silicon. Fine-tuning it with added control tokens (or a LoRA) on PiJAMA bebop segments is feasible locally. The Aria-MIDI pretraining data is NC-SA (see Section 6).

### Gaps
- No direct empirical comparison found between chord-symbol and accompaniment (or pitch-class) conditioning for melody adherence.
- No measured skyline or hand-split accuracy on jazz to estimate how clean the derived control stream would be.
- JazzSAMBA's license and download terms are not verified.

---

## 5. Data augmentation and training tricks that increase chord-conditioning adherence

### Takeaway
12-key transposition, applied to training data only, is standard practice (Jazz Transformer; AugmentedNet re-texturizes per transposition). Classifier-free guidance is now used to **strengthen control adherence in autoregressive MIDI models when context dominates** (MIDI-LLM, 2025). I found no jazz-specific evidence for contrastive "wrong-chord" negatives or auxiliary chord-prediction losses. Those remain plausible, untested levers.

### Cited Findings
- Jazz Transformer: training sequences are augmented by **twelve-key transposition**; validation and test sequences are not transposed. [V-snip] — [arXiv 2008.01307](https://arxiv.org/pdf/2008.01307)
- AugmentedNet: each time a synthetic example is transposed to another key, it is **re-texturized**, so the network sees many more distinct examples. [V-file] — [napulen/AugmentedNet README](https://github.com/napulen/AugmentedNet)
- **MIDI-LLM** (arXiv 2511.03942, Nov 2025): in MIDI infilling, the surrounding MIDI context often dominates text prompts and weakens adherence, so the authors adopt **classifier-free guidance at each token step** to strengthen text control. [V-snip] — [arXiv 2511.03942](https://arxiv.org/pdf/2511.03942)
- CFG has been applied to autoregressive MIDI transformers for composer conditioning (a GitHub project). [V-snip] — [Ururu1000/midi-transformer](https://github.com/Ururu1000/midi-transformer)
- Reward-weighted CFG for autoregressive models (arXiv 2604.15577, Apr 2026) modifies next-token logits at inference. [V-snip] — [arXiv 2604.15577](https://arxiv.org/html/2604.15577)
- Joint audio-and-symbolic conditioning work (arXiv 2406.10970, Jun 2024) uses multi-source CFG and **Subjective Condition Adherence** tests in which listeners rank harmonic alignment with chord conditioning. [V-snip] — [arXiv 2406.10970](https://arxiv.org/pdf/2406.10970)
- MusiConGen (arXiv 2407.15060, Jul 2024) adds rhythm and chord control to a transformer text-to-music model (audio domain). [V-snip] — [arXiv 2407.15060](https://arxiv.org/pdf/2407.15060)
- Chord-tone ratio (CTnCTR) is the common objective adherence metric (CMT: 72.53% vs. dataset 71.42%). [V-snip] — [CMT paper](https://www.researchgate.net/publication/350023802_Chord_Conditioned_Melody_Generation_With_Transformer_Based_Decoders)

### Inferences
- [Inf] **Transposition:** transpose melody, control stream and chord labels together by −5…+6 semitones, with range clamping (drop notes outside 21–108, or fold octave). For a bebop right hand, restrict to transpositions that keep the line inside about C3–C7 to avoid unrealistic registers.
- [Inf] **Condition dropout and CFG:** drop the chord/PC control in about 10–20% of training windows (replace with a NULL token). At inference compute logits as `l_uncond + w·(l_cond − l_uncond)` with w ≈ 1.5–3. This costs two forward passes per token, so on Apple Silicon use a small model, or batch the cond and uncond passes together, to stay real-time.
- [Inf] **Wrong-chord negatives:** in about 10% of windows, substitute a random or tritone-related chord in the control stream while keeping the true melody. Train with an auxiliary binary "does the melody fit the control?" head, or with an unlikelihood loss on the mismatched pair. This penalizes ignoring the control. I found no published evidence in symbolic jazz, so treat it as an experiment.
- [Inf] **Auxiliary chord-from-melody loss:** add a head that predicts the current beat's chord or PC set from the melody hidden state. This forces the representation to encode harmony. It is cheap to add and should be measured by the change in chord-tone ratio on strong beats and by CFG-free adherence.
- [Inf] **Evaluation for the repo's quality gate:** chord-tone ratio on downbeats and strong beats, guide-tone (3rd/7th) landing rate on chord changes, and a "control-swap test": generate the same seed under two different progressions and measure how much the melody's pitch-class distribution follows the control.

### Gaps
- No published study found that measures adherence gains from contrastive wrong-chord negatives or auxiliary chord-prediction losses in symbolic music.
- No CFG ablation found specifically for chord adherence in autoregressive symbolic transformers. MIDI-LLM covers text controls.

---

## 6. Licensing and access of all relevant datasets, and implications for personal use and a public demo video

### Takeaway
Most jazz-relevant data is **non-commercial**: PiJAMA (CC BY-NC), Aria-MIDI and JAAH (CC BY-NC-SA), FiloSax (research-only, no redistribution), madmom model weights (CC BY-NC-SA), and Doug McKenzie MIDI ("not for commercial use"). Permissive options are JTD annotations (MIT), Lakh MIDI, FiloBass, DTL1000, iRb and most of ChoCo (CC BY 4.0), Weimar Jazz Database (ODbL), and the tools Beat This!, PM2S, BTC, AugmentedNet, ChordGNN (MIT) and Aria/anticipation (Apache-2.0). Personal research is fine under all of them. A **non-monetized** demo video is broadly consistent with the NC licenses. Monetizing, or releasing model weights, raises NC and ShareAlike questions.

### Cited Findings
| Resource | License / access | What it contains | Verification |
|---|---|---|---|
| **PiJAMA** (TISMIR 2023) | MIDI and metadata **CC BY-NC**. GitHub repo LICENSE is **CC BY-NC 3.0 Unported**. Audio not distributed; YouTube URLs given. | ~200 h solo jazz piano MIDI, 2,777 performances; no beats or chords | [V-snip: TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.162); [V-file: repo LICENSE and CSV](https://github.com/almostimplemented/PiJAMA); [Zenodo 8354955](https://zenodo.org/records/8354955) |
| **Jazz Trio Database** (TISMIR 2024) | **MIT** for dataset annotations and code. YouTube audio not covered. Separated audio on Zenodo by request, research only. | 1,294 trio piano solos, 44.5 h; piano MIDI, onsets, beats, downbeats (manual first downbeat) | [V-file: README](https://github.com/HuwCheston/Jazz-Trio-Database); [Zenodo 13828337](https://zenodo.org/records/13828337) |
| **Weimar Jazz Database** v2.1 | **ODbL** (Open Data Commons Open Database License) | 456 manually transcribed monophonic solos; beat track with tapped beats, **chords**, form, bass pitch | [V-snip: Jazzomat download](https://jazzomat.hfm-weimar.de/download/download.html); [V-snip: DB format](https://jazzomat.hfm-weimar.de/dbformat/dbformat.html) |
| **FiloSax** (ISMIR 2021) | **Research-only, non-commercial, non-transferable, no redistribution** (signed agreement on Zenodo). Backing tracks must be **bought** (Aebersold via jazzbooks.com). Repo code is CC0. | 48 tunes × 5 sax players; beats per bar and chord change, chords per bar, sections, note-level `chord_changes`, `scale_changes` | [V-file: README and LICENSE](https://github.com/dave-foster/filosax) |
| **FiloBass** (ISMIR 2023) | **CC BY 4.0** | 48 bass transcriptions (>50k notes) on FiloSax backings; audio stems, scores, aligned MIDI, beats, downbeats, **chord symbols**, form | [V-snip: Zenodo 10069709](https://zenodo.org/records/10069709); [arXiv 2311.02023](https://arxiv.org/pdf/2311.02023) |
| **DTL1000** (Dig That Lick) | **CC BY 4.0** | 1,000 tracks (100 per decade 1920–2019), ~1,700 solos, segmentation, instruments, soloists; pattern n-grams | [V-snip: DigThatLick repo](https://github.com/ppquadrat/DigThatLick); [DTL slides](https://dig-that-lick.eecs.qmul.ac.uk/Docs/DiD2020-slides.pdf) |
| **JAAH** (ISMIR 2018) | ChoCo states JAAH-derived data is **CC BY-NC-SA 4.0** | 113 recordings; beats, beat-aligned chords, structure | [V-file: ChoCo LICENSE](https://github.com/smashub/choco); [MTG/JAAH](https://github.com/MTG/JAAH) |
| **ChoCo** | **CC BY 4.0**, except JAAH, Chordify and Mozart-derived data (**CC BY-NC-SA 4.0**) | 20,080 JAMS; Real Book 2,486, iReal 2,000+, WJD 456, BiaB 5,000+ | [V-file](https://github.com/smashub/choco) |
| **Jazz Harmony Treebank** | **CC BY-NC-SA 4.0** | Jazz chord-sequence tree analyses | [V-file: LICENSE](https://github.com/DCMLab/JazzHarmonyTreebank) |
| **iRb corpus** (Broze & Shanahan) | **CC BY 4.0** | 1,186 standards, Humdrum `**jazz` | [V-snip: Zenodo 3546040](https://zenodo.org/records/3546040) |
| **iReal Pro playlists / forum** | No open license. iReal Pro asserts chord progressions are not copyrightable; publisher pressure concerned titles. | User-made chord charts | [V-snip: forum rules](https://forums.irealpro.com/threads/copyrights-and-rules.48/) |
| **mikeoliphant/JazzStandards** | **No LICENSE file** (404). Data pulled from iReal Pro main playlists. | 1,377 titles, JSON bar-level chords | [V-file](https://github.com/mikeoliphant/JazzStandards) |
| **Doug McKenzie MIDI** (bushgrafts.com) | **"Not for commercial use"**; contact the author for licensing | ~250 solo jazz piano standards played on a Yamaha P250 (true performance MIDI, not transcribed) | [V-snip: bushgrafts.com/midi](https://bushgrafts.com/midi/); [Pianoteq forum](https://forum.modartt.com/viewtopic.php?id=1014) |
| **Lakh MIDI** | **CC BY 4.0** (cite Raffel thesis) | 176,581 unique MIDI; 45,129 matched to MSD | [V-snip: LMD page](https://colinraffel.com/projects/lmd/) |
| **Aria-MIDI** (ICLR 2025) | **CC BY-NC-SA 4.0** plus disclaimer | 1.19M transcribed solo-piano MIDI (~100k h) | [V-file](https://github.com/loubbrad/aria-midi) |
| **Aria models** | **Apache-2.0** (MLX for Apple Silicon) | Pretrained piano LM | [V-file](https://github.com/EleutherAI/aria) |
| **Anticipatory Music Transformer** code | **Apache-2.0**. HF model-card license not verified. | AMT code and data builders | [V-file](https://github.com/jthickstun/anticipation) |
| **Beat This!** | **MIT** (code and weights) | Audio beat/downbeat | [V-file](https://github.com/CPJKU/beat_this) |
| **madmom** | Code BSD; **models CC BY-NC-SA 4.0** | Audio beat/downbeat/onset | [V-file](https://github.com/CPJKU/madmom) |
| **BeatNet** | **CC BY 4.0** | Online and offline beat/downbeat/meter | [V-file](https://github.com/mjhydri/BeatNet) |
| **PM2S** / **BTC** / **AugmentedNet** / **ChordGNN** | **MIT** each | MIDI beats and hands / audio ACR / Roman numerals / Roman numerals | [PM2S](https://github.com/cheriell/PM2S), [BTC](https://github.com/jayg996/BTC-ISMIR19), [AugmentedNet](https://github.com/napulen/AugmentedNet), [ChordGNN](https://github.com/manoskary/chordgnn) [V-file] |

- Beat This! README: some training files are fully copyrighted or under limited CC licenses, and users must assess the impact on their own use. [V-file] — [CPJKU/beat_this](https://github.com/CPJKU/beat_this)
- madmom: "Please note that pickled Processors (i.e. saved models) fall into this category" (CC BY-NC-SA). Commercial products must contact JKU. [V-file] — [madmom LICENSE](https://github.com/CPJKU/madmom)

### Inferences
- [Inf] **Personal, local training:** every dataset above allows non-commercial research use. FiloSax additionally requires a signed agreement and purchased Aebersold backings, and forbids redistribution.
- [Inf] **Public demo video:** CC "NonCommercial" covers use "primarily intended for or directed toward commercial advantage or monetary compensation". A non-monetized portfolio or demo video is generally consistent with it. A monetized channel, sponsored content or a paid product is not. Credit PiJAMA, Aria-MIDI and others in the description (the BY requirement).
- [Inf] **ShareAlike risk:** if the model is trained or fine-tuned on Aria-MIDI or JAAH (CC BY-NC-SA) and the **weights are published**, it is legally unsettled whether the weights are "adapted material" that must carry BY-NC-SA. To stay safe, keep NC-SA-trained weights private, or publish them under CC BY-NC-SA.
- [Inf] **Underlying copyrights:** PiJAMA MIDI are transcriptions of copyrighted recordings of often-copyrighted compositions. A demo should avoid having the model reproduce copyrighted **melodies (heads)** or long memorized licks from specific recordings. Use chord progressions only (generally regarded as uncopyrightable), and add a memorization check (n-gram overlap against training data) before publishing.
- [Inf] **Charlie Parker Omnibook:** it is a commercially published transcription book (not verified here), so do not use it as training data for anything public without permission. The permissive alternative for chord-annotated bebop lines is WJD (ODbL), which includes Parker solos as monophonic transcriptions with chords and beats (inferred from WJD's bebop coverage; not verified per solo here).
- [Inf] **Recommended licensing-clean core:**
  - PiJAMA bebop subset (CC BY-NC, private use)
  - JTD (MIT)
  - WJD (ODbL) for explicit chord–melody pairs
  - iRb (CC BY 4.0) for charts
  - Beat This! (MIT) for beats
  - Aria (Apache-2.0) as the base model

  Avoid madmom weights if any commercial use is ever planned (prefer Beat This! or BeatNet).

### Gaps
- Could not open the PiJAMA Zenodo page directly to confirm the exact license version on the MIDI archive. The paper extract says CC-BY-NC; the GitHub repo says CC BY-NC 3.0.
- Weimar Jazz Database's ODbL claim is search-extract-level only. Per-solo instrument breakdown (how many piano solos, which Parker solos) was not verified.
- The Charlie Parker Omnibook's copyright holder and terms were not researched (no source retrieved).
- License of the Anticipatory Music Transformer HF checkpoints, of JazzSAMBA, and the iReal Pro app/forum ToS text were not verified.
- No legal authority (court ruling or statute) was retrieved for "chord progressions are not copyrightable". The only source is iReal Pro's own statement.
