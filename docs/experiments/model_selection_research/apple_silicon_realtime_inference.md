# Real-time autoregressive symbolic-music transformer inference on Apple Silicon, and latency engineering in existing real-time AI improvisation systems

Scope note for the report writer: the network policy in this research session blocked arxiv.org, zenodo.org, nime.org, machinelearning.apple.com, image-line.com, llmcheck.net and several blogs, so those sources could not be opened in full. Where a number comes only from a search-engine snippet of such a page (not a full-page read), it is marked **[snippet]**. Numbers read directly from GitHub pages or source code are marked **[verified page read]**. Anything I derived myself is marked **ESTIMATE** and is listed under Inferences, never under Cited Findings. Today is 2026-10-03.

Design constants used throughout (from the user's brief): 128 BPM, so a half-bar block = 2 beats = 0.9375 s; block-preparation budget about 0.4 s at p99; about 3–5 notes per block, i.e. about 10–40 generated tokens per block depending on tokenization (AMT = 3 tokens/note; event-based ≈ 4+ tokens/note), optionally × N reranking candidates.

---

## Q1. Frameworks and measured tokens/sec for 10M–1B decoder-only transformers on M1/M2/M3/M4

### Takeaway
Batch-1 decoding on Apple Silicon is limited by memory bandwidth for models of a few hundred MB and up, and by a fixed per-token dispatch overhead (a few ms per token) for small models. MLX is the best-measured fast path for small models: it is 4–5× faster than GGUF/llama.cpp for a 0.5B model in one LM Studio test, and it already supports GPT-2-, GPT-NeoX- and Llama-architecture files. Avoid PyTorch MPS for small batch-1 decoding (high per-op launch overhead, and a reported GPT-2 output-divergence bug), and avoid the Neural Engine and ONNX Runtime's CoreML EP for token-by-token decoding.

### Cited Findings

**Memory bandwidth and llama.cpp reference table (the canonical Apple Silicon benchmark)** [verified page read]
- llama.cpp "Performance of llama.cpp on Apple Silicon M-series" (Discussion #4167, opened 2023-11-22, rows added later incl. M4 family; reference build 8e672ef). LLaMA-2-7B. Format: BW GB/s / GPU cores / PP512 and TG128 tokens/s — [llama.cpp #4167](https://github.com/ggml-org/llama.cpp/discussions/4167):
  - M1: 68 GB/s, 7-core: Q8_0 PP 108.21 / TG 7.92; Q4_0 PP 107.81 / TG 14.19
  - M1 Pro: 200 GB/s, 16-core: F16 TG 12.75; Q8_0 TG 22.34; Q4_0 PP 266.25 / TG 36.41
  - M1 Max: 400 GB/s, 32-core: F16 TG 23.03; Q8_0 TG 40.20; Q4_0 PP 530.06 / TG 61.19
  - M2: 100 GB/s, 10-core: F16 TG 6.72; Q8_0 TG 12.21; Q4_0 PP 179.57 / TG 21.91
  - M2 Pro: 200 GB/s, 19-core: F16 TG 13.06; Q8_0 TG 23.01; Q4_0 PP 341.19 / TG 38.86
  - M2 Max: 400 GB/s, 38-core: Q4_0 PP 671.31 / TG 65.95
  - M3: 100 GB/s, 10-core: the row as extracted shows "F16 PP 187.52 / TG 12.27, Q8_0 PP 186.75 / TG 21.34, Q4_0 —". This looks column-shifted, because F16 TG 12.27 would need about 165 GB/s on a 100 GB/s chip. More likely Q8_0 TG = 12.27 and Q4_0 TG = 21.34. Re-check the page before quoting.
  - M3 Pro: 150 GB/s, 18-core: F16 TG 9.89; Q8_0 TG 17.53; Q4_0 PP 341.67 / TG 30.74
  - M3 Max: 400 GB/s, 40-core: Q4_0 PP 759.70 / TG 66.31
  - M4: 120 GB/s, 10-core: F16 TG 7.43; Q8_0 TG 13.54; Q4_0 PP 221.29 / TG 24.11
  - M4 Pro: 273 GB/s, 20-core: F16 TG 17.18; Q8_0 TG 30.69; Q4_0 PP 439.78 / TG 50.74
  - M4 Max: 546 GB/s, 40-core: F16 TG 31.64; Q8_0 TG 54.05; Q4_0 PP 885.68 / TG 83.06
- llama.cpp README: "Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks"; 1.5- to 8-bit integer quantization — [llama.cpp README](https://github.com/ggml-org/llama.cpp) [verified page read]

**MLX / mlx-lm vs llama.cpp**
- Small models: on an M2 Ultra Mac Studio (192 GB), LM Studio 0.3.10 / MLX engine v0.6 (2025-02-18), Qwen2.5-0.5B 4-bit ran at **317 tok/s on MLX vs 79 tok/s on GGUF**. The reporter's summary: "for very small models, MLX is 4-5x faster. For large models, it is 10x slower" (the large-model slowdown starts around 22B+) — [lmstudio-ai/mlx-engine #101](https://github.com/lmstudio-ai/mlx-engine/issues/101) [verified page read]
- 8B model on M3 Max 64 GB (2024-10-10), Llama-3.1-8B-Instruct. Generation: MLX 4-bit 23.43 tok/s, MLX 8-bit 19.02; llama.cpp Q4_K_M with flash-attention 32.00 / without FA 9.41; Q8_0 with FA 25.90 / without FA 8.66. Prompt processing: MLX 4-bit 421.6, MLX 8-bit 401.6, llama.cpp FA Q4_K_M 385.6 tok/s — [mlx-examples #1029](https://github.com/ml-explore/mlx-examples/issues/1029) [verified page read]. Takeaway: llama.cpp numbers depend heavily on flash-attention being enabled.
- mlx-lm ships model files `gpt2.py`, `gpt_neox.py`, `llama.py` and a `cache.py` (KV cache) in `mlx_lm/models`; there is no Transformer-XL — [mlx-lm models dir](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models) [verified page read]. So GPT-2-architecture AMT checkpoints and Llama-architecture Aria are architecturally covered. BebopNet's Transformer-XL is not.
- Llama-3.2-1B 4-bit on MLX, M4 Max: 461.9 tok/s — [starmorph guide](https://blog.starmorph.com/blog/apple-silicon-llm-inference-optimization-guide) [snippet; secondary blog]
- Ollama (llama.cpp Metal backend) on a MacBook Air M1 8 GB: llama3.2:1b 77.02 tok/s, qwen2.5:1.5b 59.94 tok/s — [agenticwire M1 Air test](https://www.agenticwire.news/article/run-local-llm-8gb-ram) [snippet; secondary blog, unverified methodology]
- A 2026 comparison claims MLX is 20–87% faster than llama.cpp for generation below 14B params on short contexts, and that the lead reverses at very long contexts (~146K) — [yage.ai MLX vs llama.cpp](https://yage.ai/share/mlx-apple-silicon-en-20260331.html) [snippet; secondary]
- Magenta RealTime 2 ships a **C++ inference engine on MLX** that loads a compiled `.mlxfn` graph (weights plus graph). Its 230M "small" model runs real-time on any Apple Silicon including Air models; the 2.4B "base" needs M4 Pro / M5 Max class. Compatibility table: M5 Max ✅/✅, M4 Pro ✅/✅, M2 Pro ✅/❌, M1 Pro ✅/❌, M4 Air ✅/❌ — [magenta-realtime GitHub](https://github.com/magenta/magenta-realtime) [verified page read]; Air claim and `.mlxfn` detail from [Mervin Praison MRT2 on Mac](https://mer.vin/2026/06/magenta-realtime-2-open-live-music-ai-on-mac-with-midi-audio-and-text-control/) [snippet]

**PyTorch MPS**
- Every MPS op carries a constant setup cost (command-buffer creation, kernel dispatch, Metal sync). For small models, small batches and light ops, this overhead can exceed the compute time, and an unsupported op mid-graph breaks optimization of that region — [GeekChamp: PyTorch MPS slower than CPU](https://geekchamp.com/pytorch-mps-slower-than-cpu/) [snippet; secondary]
- PyTorch issue: "MPS device appears much slower than CPU on M1 Mac Pro" (BERT inference about 2× slower on MPS than CPU) — [pytorch #77799](https://github.com/pytorch/pytorch/issues/77799) [snippet]
- Correctness risk: on MPS, GPT-2 small produced "noticeably different and often degraded greedy generations compared to CPU" — [ARENA_materials #264](https://github.com/ARENA-education/ARENA_materials/issues/264) [snippet]

**CPU-only and Apple Neural Engine (ANE)**
- GPT-2 124M on M4 Max: CPU decode **283 tok/s (3.48 ms/token)**; the ANE full-forward path decodes at only 170 tok/s because of about **2.3 ms IOSurface round-trip overhead per ANE dispatch** — Orion paper (arXiv 2603.06728, Mar 2026) [Orion](https://arxiv.org/pdf/2603.06728) [snippet]

**Core ML (stateful KV cache)**
- Apple's "On Device Llama 3.1 with Core ML": Llama-3.1-8B-Instruct reaches about **33 tok/s decode on an M1 Max** after optimizations (stateful KV cache via the macOS Sequoia "states" input type, plus int4 block-wise quantization and fused SDPA). The baseline without a KV cache managed only 0.19 tok/s at 2048 max context and 3.23 tok/s at 128 context. Without stateful buffers, an 8192 context copies up to ~1 GB per step — [Apple ML Research](https://machinelearning.apple.com/research/core-ml-on-device-llama) [snippet; page fetch blocked]

**ONNX Runtime on macOS**
- The CoreML execution provider warns "Dynamic shape is not supported" and falls back to CPU, which hits a growing KV cache directly — [onnxruntime #14212](https://github.com/microsoft/onnxruntime/issues/14212); [ORT CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) [snippet]
- An experimental community **MLX execution provider for ONNX Runtime** maps supported ONNX regions (incl. attention, quantized ops, KV-cache ops) to MLX and leaves the rest on ORT CPU — [justinchuby/onnxruntime-mlx](https://github.com/justinchuby/onnxruntime-mlx) [snippet]
- jam_bot (MIT Media Lab; ISMIR 2025 and NIME 2026) runs AMT in **ONNX Runtime from C++**. The NIME 2026 paper benchmarks AMT medium ("around 400M parameters") on three backends: ORT plus a new **ggml implementation**, on CPU (AMD Ryzen 9 7950X), CUDA (RTX 4090) and **Apple Metal (M3 Max MacBook Pro)**. Throughput was measured as a 15-s running average during performance. The paper states that "in practice, only the ONNX Runtime CUDA backend can be used for the jam_bot." Reported figures: ORT CUDA **4.00 ms/token, 249.75 tok/s** (416M model); ORT CPU **118.55 ms/token, 8.44 tok/s** (model variant unclear in snippet) — [jam_bot NIME 2026 PDF](https://nime.org/proceedings/2026/nime2026_73.pdf); [ISMIR 2025 poster page](https://ismir2025program.ismir.net/poster_321.html) [snippet; full table, including the M3 Max Metal number, not retrieved]

### Inferences
- **ESTIMATE: effective decode bandwidth** = TG(Q4_0 7B) × ~3.82 GB of weights, using the #4167 table: M1 ≈ 54 GB/s, M2 ≈ 84, M3 ≈ 82 (if the shifted reading is right), M4 ≈ 92, M3 Pro ≈ 117, M1 Pro ≈ 139, M2 Pro ≈ 148, M4 Pro ≈ 194, M1 Max ≈ 234, M3 Max ≈ 253, M4 Max ≈ 317 GB/s. That is about 70–85% of nominal.
- **ESTIMATE: batch-1 bandwidth ceiling** (tok/s ≈ effective BW / weight bytes):

  | Model (weights) | M1 | M2 | M4 Pro | M4 Max |
  |---|---|---|---|---|
  | AMT-128M fp16 (0.26 GB) | ~210 | ~320 | ~750 | ~1200 |
  | AMT-360M fp16 (0.72 GB) | ~75 | ~117 | ~270 | ~440 |
  | AMT-360M 8-bit (0.36 GB) | ~150 | ~230 | ~540 | ~880 |
  | Aria-650M bf16 (1.3 GB) | ~42 | ~65 | ~150 | ~245 |
  | Aria-650M 8-bit (0.65 GB) | ~83 | ~130 | ~300 | ~490 |
  | AMT-780M fp16 (1.56 GB) | ~35 | ~54 | ~124 | ~200 |

- **ESTIMATE: per-token overhead floor of about 3 ms.** Qwen2.5-0.5B 4-bit on MLX reached "only" 317 tok/s on an M2 Ultra (about 3.2 ms/token), although bandwidth alone would allow more than 2000 tok/s. GPT-2-124M on the M4 Max CPU sits at 3.48 ms/token. So below about 150M params, per-token latency is set by dispatch and Python overhead, not by bandwidth. Use t_token ≈ weights/BW_eff + ~3 ms as a conservative planning model. `mx.compile` or a compiled `.mlxfn` C++ path (as in MRT2) is the known way to cut this floor.
- **ESTIMATE: prefill compute.** Q4_0 PP512 on M1 (107.8 tok/s × ~2 × 6.7 GFLOP) ≈ 1.4 TFLOPS effective, M2 ≈ 2.4, M3 Pro ≈ 4.6, M4 Pro ≈ 5.9, M4 Max ≈ 11.9. A 360M model prefilling a 1024-token context therefore costs about 0.35–0.7 s on M1 but about 0.06–0.12 s on M4 Pro. A full re-prefill on every chord change would blow the 0.4 s budget on base chips. Keep the KV cache and append or roll back instead (see Q4).
- The 13M Music Transformer is overhead-bound everywhere. On CPU (PyTorch or plain NumPy/ORT) it is expected to run well under 1 ms/token, extrapolated from GPT-2-124M's 3.48 ms/token on CPU. MPS would likely be slower than CPU for it.
- Framework ranking for this project (ESTIMATE, from the evidence above):
  1. MLX (Python with `mx.compile`, or C++ `.mlxfn`) for 128M–1B models.
  2. CPU (PyTorch, or ORT CPU int8) for models up to ~30M.
  3. llama.cpp/ggml only if GPT-2 GGUF conversion and custom-vocab handling are acceptable; jam_bot already wrote a ggml AMT port.
  4. Core ML stateful models: viable on macOS 15+ but heavy tooling.
  5. PyTorch MPS: last resort for batch-1 decoding.

### Gaps
- No measured tokens/sec found for exactly GPT-2 124M/355M/774M on MLX or llama.cpp on M1/M2/M3, nor for Qwen2.5-0.5B or Llama-3.2-1B on base M1/M2 with mlx-lm. The closest are 317 tok/s (0.5B, M2 Ultra, MLX) and 461.9 tok/s (1B 4-bit, M4 Max, secondary blog). The user should run `mlx_lm.generate --verbose` or `llama-bench` locally.
- llama.cpp GPT-2/GGUF support could not be confirmed from the README excerpt (the model list sits in docs/models.md, which was not fetched). Whether a custom-vocab AMT checkpoint converts cleanly with `convert_hf_to_gguf.py` is unverified.
- jam_bot's M3 Max (Metal/ggml) tokens/sec and any int8 ONNX numbers were not retrievable (NIME PDF blocked). The brief's claim of "AMT-360M ONNX int8 + KV cache" is not confirmed by the sources I could read: they say "medium ~400M" and "416M", and the snippets mention no int8.
- No measured HF `generate()` + KV-cache tokens/sec on MPS for GPT-2-class models was found. Whether static cache / `torch.compile` work on MPS in 2025–2026 PyTorch versions is unverified.
- No time-to-first-token figures for 512–1024-token prompts on small models; only the prefill-compute estimates above.

---

## Q2. Aria / Aria-Duet (EleutherAI): framework, latency on Apple Silicon, pipelining

### Takeaway
Aria (Llama-3.2-1B-style, about 650M–1B params) runs real time on Apple Silicon through a dedicated MLX path (`demo/demo_mlx.py`): bf16 weights, optional 8-bit quantization, a 4096-token KV cache, and five threads. The decisive trick is **continuous chunked prefill while the human plays**. Without it, prefill alone causes 1000–2000 ms of lag on Apple Silicon. Note scheduling uses epoch timestamps plus calibrated per-velocity output latency.

### Cited Findings
- Aria-Duet ("The Ghost in the Keys", Nov 2025) is a real-time human-AI duet on a Yamaha Disklavier using Aria, "based on the LLaMA 3.2 (1B) architecture", trained on ~60k hours of piano MIDI transcriptions. It is turn-taking: the human plays, signals handover, the model continues — [arXiv 2511.01663](https://arxiv.org/html/2511.01663) [snippet]; [EleutherAI/aria](https://github.com/EleutherAI/aria)
- On Apple Silicon, high memory bandwidth suits decoding, but "the compute-bound prefill phase creates a significant bottleneck, introducing an unacceptable lag of **1000–2000 ms** between the user's takeover signal and the model's first note." The fix is a **continuous prefill strategy** that "proactively updates the KV-cache in small chunks as the user plays", "virtually eliminating the prefill-induced delay." Speculative re-evaluation of note durations adds "**~100–200 ms**" — [Ghost in the Keys (OpenReview PDF)](https://openreview.net/pdf?id=yL8BrlEqHQ) [snippet]
- Repo `demo/` contains `demo_mlx.py`, `calibrate.py`, `demo-tokenizer-config.json` and `hardware/`, i.e. "an MLX (Apple Silicon) implementation of the real-time interactive piano-continuation demo" — [aria/demo](https://github.com/EleutherAI/aria/tree/main/demo) [verified page read]
- Source details of `demo_mlx.py` — [raw source](https://raw.githubusercontent.com/EleutherAI/aria/main/demo/demo_mlx.py) [verified page read]:
  - **Model, precision and quantization:** `DTYPE = mx.bfloat16`; optional `nn.quantize(model.model, group_size=32, bits=8)`.
  - **KV cache:** `model.setup_cache(batch_size=1, max_seq_len=4096, dtype=DTYPE)`, `KV_CHUNK_SIZE = 256`.
  - **Prefill:** `continuous_prefill()` processes incoming MIDI in batches of ≥10 messages. `chunked_prefill()` uses `PREFILL_CHUNK_SIZE = 16` and `PREFILL_CHUNK_SIZE_L = 128`. Duration recalculation uses `RECALC_DUR_PREFILL_CHUNK_SIZE = 8` and `RECALC_DUR_BUFFER_MS = 100`.
  - **Sampling:** temperature 0.95, `min_p` 0.03. The opening uses `BEAM_WIDTH = 3` and `TIME_TOK_WEIGHTING = -5`, which penalizes time tokens. Per-token decode time is logged in ms.
  - **Threads:** five, connected by `queue.Queue`: `capture_midi_input` → `continuous_prefill` → `generate_tokens` → `decode_tokens_to_midi` → `stream_midi`.
  - **Scheduler:** `stream_midi` sorts pending messages by epoch send-time, with `MAX_STREAM_DELAY_MS = 50`.
  - **Latency calibration:** `VELOCITY_OUTPUT_LATENCY_MS` (per velocity bucket), `BASE_OUTPUT_LATENCY_MS`, and `FIRST_ONSET_BUFFER_MS = -150`. `set_calibration_settings()` loads a hardware JSON profile.
  - **MIDI I/O:** `mido.open_input()` / `mido.open_output()`.
- In the paper's framing, the latency handling is a "custom, zero-latency streaming layer that modifies the playback schedule in real-time", scheduling each note-on's send-time to absorb calibrated velocity-specific latency — [Ghost in the Keys](https://arxiv.org/html/2511.01663) [snippet]

### Inferences
- The Aria design maps directly onto the user's FL Studio setup: prefill the chord and solo context continuously as chords arrive, keep the KV cache hot, and decode only the next block. Its quantization recipe (8-bit, group 32, MLX) is a ready template for running Aria-sized models on base chips.
- Aria's 1000–2000 ms un-optimized prefill lag agrees with the Q1 prefill-compute estimate. That is strong evidence that a re-prefill-on-every-chord-change design would fail on M1/M2.

### Gaps
- No measured Aria tokens/sec or first-note latency on a named M-series chip was retrievable (paper body blocked). Repo code logs per-token ms but publishes no numbers.
- Which Mac the demo was performed on is not stated in the material I could read.

---

## Q3. Latency engineering in other real-time AI music systems

### Takeaway
Every working system hides model latency behind **musical lookahead**. ReaLJam generates frames ahead and commits them; Magenta RT streams fixed 2-s chunks faster than real time; jam_bot runs multi-threaded C++ ORT with scheduled generations; Aria prefills continuously and schedules by timestamp; BachDuet steps on a 16th-note grid. None of them requires the model to answer within one musical event of the input.

### Cited Findings
- **ReaLJam** (Google DeepMind et al., CHI EA 2025, arXiv 2502.21267):
  - Its two named problems are **anticipation** (each side predicts the other) and **synchronization** (notes played at intended times without delay) — [ReaLJam (ACM)](https://dl.acm.org/doi/10.1145/3706599.3720227); [arXiv PDF](https://arxiv.org/pdf/2502.21267) [snippet]
  - **Commit protocol:** chords "in the immediate future up to a commit time (before the end of the lookahead) are said to be committed; that is, the agent can no longer change those predictions. Beyond the commit time, the agent is free to update its predictions." The UI renders uncommitted chords semi-transparently — [ReaLJam HTML](https://arxiv.org/html/2502.21267v1) [snippet]
  - **Stateless requests:** each request sends the full session history plus the chords in the lookahead, the target frame to start generating in, and the lookahead/commit settings — same source [snippet]
  - **Latency hiding:** "scheduled chords play while waiting for responses, ensuring no client-side delay or interruption as long as the number of lookahead frames is higher than the number of frames spent waiting." On a single inference device, "a majority of responses return within **100 milliseconds**, fast enough for single-frame round trips at **150 beats per minute**" — same source [snippet]
- **Magenta RealTime** (audio, June 2025):
  - Fixed **2-s chunks**, each predicted autoregressively from 5 previous coarse chunks (10 s of history), with **crossfading** at boundaries and only the first 4 RVQ levels kept in context to cut compute. RTF **1.8 on an H100** (T5 Large) — [Emergent Mind summary](https://www.emergentmind.com/topics/magenta-realtime-magenta-rt); [Magenta RT blog](https://magenta.withgoogle.com/magenta-realtime) [snippet]
  - MRT2 moves to a compiled MLX C++ engine for local Macs (see Q1) — [magenta-realtime](https://github.com/magenta/magenta-realtime) [verified page read]
- **jam_bot** (MIT, Jordan Rudess; ISMIR 2025):
  - Adapts music LMs to lead, accompany, or call-and-response roles by changing context and conditioning signals. Needs optimizations to run in real time inside "a low-latency multi-threaded system that listens, and prompts and schedules model generations seamlessly" — [ISMIR 2025 poster 321](https://ismir2025program.ismir.net/poster_321.html) [snippet]
  - Uses ORT in C++ to reach real-time factor > 1; only ORT CUDA was usable in practice — [NIME 2026](https://nime.org/proceedings/2026/nime2026_73.pdf) [snippet]
- **BachDuet** (Rochester, AAAI-20 demo): multi-task LSTM; time is quantized to **16th-note steps**, and at each step the RNN predicts one token — [BachDuet PDF](http://labsites.rochester.edu/air/publications/benetatos20bachduet.pdf) [snippet]
- **LK Jam** (arXiv 2606.21018, 2026): a real-time human-AI interactive music system using a role-aware GRU, i.e. a small recurrent model chosen for latency — [LK Jam](https://arxiv.org/pdf/2606.21018) [snippet: title only]
- **"Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation"** (arXiv 2606.11886, 2026) exists but could not be read — [arXiv](https://arxiv.org/html/2606.11886) [title only]

### Inferences
- In ReaLJam, "single-frame round trip at 150 BPM within 100 ms" implies a frame of about one 16th note (100 ms at 150 BPM). The user's half-bar block (0.94 s) is about 9–10× coarser, so a 0.4 s budget is generous by comparison. The binding constraint is chord-reaction latency, not throughput.
- Common patterns across these systems:
  1. Lookahead buffer at least as long as the worst-case model wait.
  2. A commit horizon beyond which output may still be revised.
  3. Stateless or rollback-able context so late input can be absorbed.
  4. Timestamp-based scheduling decoupled from generation.
  5. Small or recurrent models when per-event response is needed (BachDuet, LK Jam).

### Gaps
- No primary latency numbers found for Google "AI Duet", Shimon, Somax2 or Magenta Studio (Ableton/Max). These systems were not researched in depth within the tool budget.
- ReaLJam's inference hardware (TPU/GPU) and exact frame and lookahead sizes were not confirmed from full text.

---

## Q4. Speculative / lookahead generation design and batched N-candidate generation

### Takeaway
Generate block k+1 during block k; commit it at a fixed point before its downbeat; keep the KV cache append-only with cheap rollback to the last committed boundary. Handle late chords by (a) hedging: batch-decoding candidates under 2–4 chord hypotheses, which is nearly free in the bandwidth-bound regime, and (b) a short pickup/patch region that is regenerated after the chord lands. Batching N candidates on Apple Silicon costs far less than N× because batch-1 decoding is bandwidth-bound.

### Cited Findings
- ReaLJam's commit/lookahead protocol and its "lookahead frames > waiting frames" invariant — [ReaLJam](https://arxiv.org/html/2502.21267v1) [snippet]
- Aria's continuous chunked prefill (16/128-token chunks) and a ~100–200 ms speculative duration re-evaluation step — [Ghost in the Keys](https://openreview.net/pdf?id=yL8BrlEqHQ) [snippet]; [demo_mlx.py](https://raw.githubusercontent.com/EleutherAI/aria/main/demo/demo_mlx.py) [verified page read]
- Bandwidth-bound decoding evidence: on M1, llama.cpp processes 512-token batches at 107.81 tok/s versus 14.19 tok/s for batch-1 generation (Q4_0 7B). On M4 Pro the figures are 439.78 vs 50.74 — [llama.cpp #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) [verified page read]
- The mlx-lm KV-cache module (`cache.py`) exists alongside the GPT-2 and Llama model files — [mlx-lm models](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models) [verified page read]

### Inferences
- **ESTIMATE: batching cost.** PP/TG ≈ 7.6× on M1 and ≈ 8.7× on M4 Pro. So processing about 8 tokens per weight-read costs roughly the same wall time as 1 token. Decoding N = 2–8 candidates in one batch should cost about 1.1–2× a single candidate, not N×. Real gains depend on the framework's batched-attention kernels. Measure with mlx-lm batched generation.
- Suggested timeline per half-bar block at 128 BPM (0.9375 s), as engineering inference:
  1. t = 0 (downbeat of block k): block k plays. Block k+1 decoding starts immediately, using the current chord estimate for k+1 plus 1–3 alternative chord hypotheses as a batch.
  2. Chord input for k+1 arrives up to t ≈ 0.94 − 0.4 − safety. Select the matching hypothesis candidate. If none matches, regenerate only the first beat (a pickup patch, about 5–20 tokens) from the committed KV boundary.
  3. Commit block k+1 about 50–100 ms before its downbeat and hand it to the MIDI scheduler as timestamped events. This mirrors Aria's 50 ms max stream delay.
  4. On rollback, truncate the KV cache to the committed position. Never re-prefill the whole context (see the Q1 prefill estimate and Aria's 1000–2000 ms finding).
- Since "chord reflected within ~1 block" is the requirement, hedged batch generation over the few diatonic or ii-V likely next chords lets most chord changes be absorbed with zero extra decode. That is the same idea as ReaLJam's uncommitted-but-revisable horizon.
- For tokenizations where chords are interleaved as control tokens (AMT anticipation-style), appending a late chord costs a few prefill tokens (< 10). That is negligible compared with decode.

### Gaps
- No published measurements of batched-N decode scaling on MLX or MPS for 100M–1B models were found.
- Whether mlx-lm's KV cache exposes trim/rollback for GPT-2 models was not verified in source. Check `cache.py` for `trim()`/`is_trimmable()`.

---

## Q5. MIDI plumbing on macOS into FL Studio (IAC, clock, jitter, processes)

### Takeaway
Use the macOS IAC Driver: one bus from FL Studio to Python for chords and clock, a separate bus from Python to FL Studio for solo notes routed to Serum. FL Studio can send MIDI clock (master sync) but not receive it, so FL must be the master. python-rtmidi on CoreMIDI adds sub-millisecond latency (median ~0.14 ms, p99 ~0.9 ms in one benchmark), so MIDI transport is not the bottleneck. Python GIL contention and scheduler design are the real risks.

### Cited Findings
- FL Studio can **send** MIDI Clock ("Send master sync") so external gear can follow it, but **cannot receive** MIDI clock, so FL must be master. "Send master sync" sends FL's transport (start/stop/pause). Each output interface has independent "Send master sync" and "Port number" settings — [FL Studio MIDI settings manual](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/envsettings_midi.htm); [instrument.bible: Sync to FL Studio](https://instrument.bible/guide/sync/fl-studio/) [snippet]
- Image-Line warns: "Don't enable 'Send master sync' if the device does not use transport control as it can cause unpredictable behavior" — [FL Studio MIDI settings](https://www.image-line.com/fl-studio-learning/fl-studio-online-manual/html/envsettings_midi.htm) [snippet]
- On macOS, select "IAC Driver Bus" in FL's MIDI settings and set a port number (e.g., 0) — [Syntheway FL MIDI settings](https://www.syntheway.net/FL_Studio_System_Settings_MIDI.htm) [snippet]
- A macOS MIDI routing latency benchmark compared C, Python (rtmidi/mido), Java and Bome over the same path. Native CoreMIDI C: median **0.125 ms**, p99 **0.881 ms**. python-rtmidi callback mode: median **0.141 ms**, p99 **0.893 ms** — [Beennnn/midi-bench-routing](https://github.com/Beennnn/midi-bench-routing) [snippet; the repo URL returned 404 when fetched, so treat as unverified]
- CoreMIDI processing adds < 1 ms normally — [midilize: MIDI latency on Mac](https://midilize.com/guides/midi-latency-mac) [snippet; secondary]
- Aria's demo uses mido in/out ports, threads plus queues, and an epoch-sorted sender with `MAX_STREAM_DELAY_MS = 50` — [demo_mlx.py](https://raw.githubusercontent.com/EleutherAI/aria/main/demo/demo_mlx.py) [verified page read]

### Inferences
- Recommended topology (engineering inference):
  - **IAC Bus 1, FL → Python:** FL MIDI Out with "Send master sync" for 24-PPQN clock and start/stop, plus chord notes from the typing keyboard, echoed by FL's MIDI Out plugin or controller routing.
  - **IAC Bus 2, Python → FL:** solo notes into a MIDI input port mapped to the Serum channel.
  - Keep the buses separate to avoid feedback loops.
- Derive tempo and phase from clock ticks (smoothed, e.g., by averaging the inter-tick interval). If FL does not send Song Position Pointer reliably, count bars from the Start message, or set the tempo statically and use only Start/Stop for phase.
- Run the model in a **separate process** (multiprocessing) from the MIDI scheduler. A Python sampling loop holds the GIL between kernel calls and can delay a scheduler thread by several ms. Aria gets away with threads, but its 50 ms stream-delay tolerance is large. The scheduler process should receive whole committed blocks as timestamped events and fire them from a high-priority loop. Timestamped sends make generation jitter irrelevant as long as blocks commit ahead of their downbeat.

### Gaps
- Whether FL Studio's master sync emits Song Position Pointer messages (vs only Clock + Start/Stop/Continue) could not be confirmed (manual fetch blocked).
- No measured jitter of FL Studio's outgoing MIDI clock on macOS was found.
- Whether python-rtmidi on macOS supports future-timestamped CoreMIDI sends (MIDITimeStamp) rather than immediate sends was not verified. Assume immediate send and do scheduling in user code.

---

## Q6. Rule of thumb: which model size fits a 0.4 s p99 block budget?

### Takeaway
**ESTIMATE**, built from the measured bandwidth table, the ~3 ms/token overhead floor, and the planning model t_token ≈ weights/BW_eff + 3 ms:

- **M1/M2 base:** comfortable up to ~128M fp16 or ~360M 8-bit for 15–25 tokens per block. Aria-650M or AMT-780M fits only at 8-bit with about 15 tokens per block and no extra candidates.
- **M3/M4 Pro and above:** 360M–780M at 8-bit handles 40 tokens per block, with 2–4 batched candidates.
- **13M Music Transformer:** fits anywhere on CPU, even with N = 8 candidates.

### Cited Findings
- Effective inputs: per-chip bandwidth and TG figures [llama.cpp #4167](https://github.com/ggml-org/llama.cpp/discussions/4167); overhead floor evidence from 317 tok/s for a 0.5B 4-bit model on MLX [mlx-engine #101](https://github.com/lmstudio-ai/mlx-engine/issues/101) and 3.48 ms/token for GPT-2-124M on CPU [Orion](https://arxiv.org/pdf/2603.06728) [snippet]
- Real-system calibration points:
  - The MRT2 230M model runs real time on M1 Pro, M2 Pro and Air-class chips; the 2.4B needs M4 Pro or above — [magenta-realtime](https://github.com/magenta/magenta-realtime)
  - Aria (~1B Llama-arch) runs a live duet on Apple Silicon via MLX bf16/8-bit — [EleutherAI/aria](https://github.com/EleutherAI/aria)
  - jam_bot found only CUDA usable for its ~400M AMT at ~250 tok/s, but its budget was tighter (free improvisation, per-event responsiveness) — [NIME 2026](https://nime.org/proceedings/2026/nime2026_73.pdf) [snippet]

### Inferences
- **ESTIMATE: per-block decode time** (single candidate) = tokens × (weights/BW_eff + 3 ms). ✓ = under 0.4 s with margin; ~ = marginal; ✗ = over budget.

  | Model / precision | M1 15 tok | M1 40 tok | M2 15 tok | M2 40 tok | M4 Pro 15 tok | M4 Pro 40 tok |
  |---|---|---|---|---|---|---|
  | 13M (CPU) | ✓ ~0.02–0.05 s | ✓ | ✓ | ✓ | ✓ | ✓ |
  | AMT-128M fp16 | ✓ 0.12 s | ~ 0.31 s | ✓ 0.09 s | ✓ 0.24 s | ✓ 0.06 s | ✓ 0.17 s |
  | AMT-360M fp16 | ✓ 0.25 s | ✗ 0.65 s | ✓ 0.17 s | ✗ 0.46 s | ✓ 0.10 s | ✓ 0.27 s |
  | AMT-360M 8-bit | ✓ 0.15 s | ~ 0.39 s | ✓ 0.11 s | ✓ 0.29 s | ✓ 0.07 s | ✓ 0.19 s |
  | Aria-650M bf16 | ✗ 0.41 s | ✗ | ~ 0.28 s | ✗ 0.74 s | ✓ 0.15 s | ~ 0.39 s |
  | Aria-650M 8-bit | ✓ 0.23 s | ✗ 0.60 s | ✓ 0.16 s | ✗ 0.43 s | ✓ 0.10 s | ✓ 0.25 s |
  | AMT-780M fp16 | ✗ 0.48 s | ✗ | ~ 0.33 s | ✗ | ✓ 0.17 s | ~ 0.44 s |
  | AMT-780M 8-bit | ~ 0.26 s | ✗ | ✓ 0.18 s | ✗ 0.49 s | ✓ 0.11 s | ✓ 0.28 s |

- The table above is p50-style. For p99, add about 30–50% headroom for thermal throttling on fanless Airs, GC pauses and DAW contention, so treat "~" as failing on a base chip. N-candidate batching adds an ESTIMATED 10–100% (not N×) per the PP/TG ratio argument in Q4.
- With AMT's 3 tokens/note and about 5 notes per half-bar, a block is about 15 tokens. That makes AMT-360M 8-bit or AMT-128M the sweet spot for M1/M2 base, and AMT-780M or Aria 8-bit realistic on M3/M4 Pro.
- Event-based vocabularies (4+ tokens/note) push toward 20–40 tokens and toward the smaller models.
- Practical first step: benchmark on the user's actual Mac with mlx-lm (GPT-2 or Llama class files) and record per-token p50/p99 over 1000 blocks, including chord-change rollbacks.

### Gaps
- All Q6 per-model figures are estimates. No source measured AMT-128M/360M/780M, BebopNet or Aria tokens/sec on M1/M2/M3/M4.
- The thermal-throttling impact on a MacBook Air over a multi-minute session is unmeasured in the sources found.
- BebopNet's parameter count and Transformer-XL memory-cache behaviour on MLX/MPS were not researched. mlx-lm has no Transformer-XL implementation, so it would need a custom port or a PyTorch CPU run.
