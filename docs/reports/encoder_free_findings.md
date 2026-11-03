# Encoder-free SCRIPT: findings

Status: current as of 2026-11-03. All numbers are from `eval_leaderboard.sh` on the
same seven-corpus suite (74 842 utterances) unless marked `val_wer`, which is the
in-training metric on mcv11 dev (≈2 000 utterances).

---

## 1. Headline

**One parameter set can serve the whole latency axis, and encoder-free is far more
viable offline than streaming.**

The unified model trains on `chunk_size: [14, 1024]` — a 1.12 s streaming point and a
whole-utterance offline point — with 100 % shared parameters. At its best it:

- **ties** the streaming-only specialist at streaming, and
- **beats** the offline-only specialist at offline by 27 %.

| model | chunk 14 (streaming) | chunk 1024 (offline) |
|---|---|---|
| pretrained-encoder baseline | **5.84** | — |
| unified + modality-local attn, n=8 @ 24 k | **10.02** | 8.12 |
| unified + modality-local attn, n=4 @ 8 k | 10.06 | **8.08** |
| unified (plain), n=0 @ 8 k | 10.22 | 8.18 |
| streaming specialist (`nocoder_sa`) | 10.51 | — |
| offline specialist | — | 11.66 |

The encoder-free gap to a pretrained 24-layer Conformer is ~72 % relative at streaming
and ~39 % offline. It has not been closed.

---

## 2. Setup

`ScriptSTTModel`, Qwen3-1.7B. Encoder-free means the pretrained Conformer is replaced
by `StackingEncoder` — a parameter-free reshape of 8 × 128 log-mel frames into one
1024-dim vector — followed by a 1024→8192→2048 MLP. The non-LLM audio stack is
25.2 M parameters, essentially identical in size to Mel-LLM's.

Layout is `[prompt][audio window][emitted text]` per chunk, with history carried as
**text** and re-prefilled each chunk, so there is no acoustic memory across chunks.
Offline is the degenerate case: one chunk covering the utterance.

---

## 3. What we learned

### 3.1 Most of the encoder-free penalty is the *streaming* constraint

From scratch, same architecture and data:

| setting | reaches | at step |
|---|---|---|
| offline (chunk 1024) | 0.2311 `val_wer` | 24 000 |
| streaming (chunk 14) | 0.2323 `val_wer` | 112 000 |

**~4.7× more step-efficient offline.** Streaming asks the LLM to do all acoustic
modelling inside a 42-frame (3.36 s) window with no acoustic history; an encoder is
far more load-bearing in that regime. This is why offline was worth testing first.

### 3.2 Multi-operating-point training is free at streaming and *helps* offline

The unified model beats the dedicated offline arm (8.56 vs 11.66 when first compared)
while matching the streaming specialist. The streaming draws act as useful auxiliary
supervision rather than competing for capacity.

The same effect appears from scratch: `scratch_ctrl` (chunk `[14,1024]`) reached
0.3889 `val_wer` by step 14 000, faster than any single-point from-scratch arm.

### 3.3 Simpler front-ends won

| front-end | best `val_wer` |
|---|---|
| stacking + MLP + SpecAugment | **0.1888** |
| stacking + MLP | 0.2046 |
| 3 Conformer layers (random init) | 0.2104 |

Parameter-free stacking beat three Conformer layers. This cuts against Mel-LLM, which
retains a 12.6 M-parameter conv front-end. SpecAugment is worth ~8 % relative.

### 3.4 Initialisation dominates

Warm-started arms reach ~0.19 in a few thousand steps; from scratch needs 38 k–112 k.
Consistent with Mel-LLM (7.12 with multimodal init vs 7.44 random).

### 3.5 Modality-local attention: a small, regime-specific win

`modality_local_layers: n` — for the first *n* LLM layers, audio attends **only** to
audio in its own branch, losing access to the emitted-text prefix. Text is already
modality-local at every layer (a structural invariant that keeps its KV cache
reusable). Zero added parameters.

Encoder-free, matched at step 8 000:

| n | chunk 14 | chunk 1024 |
|---|---|---|
| 0 | 10.22 | 8.18 |
| **4** | **10.06** | **8.08** |
| 8 | 10.09 | 8.14 |
| 14 | 10.28 | 8.33 |

A consistent ~1–2 % with a shallow optimum at n≈4–8; n=14 is worse than no local
attention at all. n=8 trained to 24 k reaches 10.02 / 8.12.

**It does not transfer to encoder-based models.** On `p84` (pretrained Conformer),
four matched leaderboard points split 2–2, and within each point the seven corpora
split 3–3–1 — the signature of no effect rather than a small one:

| step | n=0 | n=4 |
|---|---|---|
| 16 000 | 6.02 | 6.12 |
| 20 000 | 5.95 | **5.88** |
| 24 000 | 6.01 | **5.93** |
| 36 000 | **5.94** | 5.97 |

The one regime where it helps is the one where the LLM has **no encoder at all** — not
merely less lookahead (see §4.2).

---

## 4. Negative results

### 4.1 Modality-specific audio q/k/v — clearly harmful

Giving audio its own `q/k/v` in the modality-local layers (+33.6 M params,
initialised as exact copies of the text projections, so step 0 is bit-identical):

| step | shared (control) | private audio q/k/v |
|---|---|---|
| 10 000 | 0.5662 | 0.7887 |
| 20 000 | 0.3047 | 0.4023 |
| 30 000 | **0.2568** | 0.3380 |

**20–32 % worse across ten consecutive checkpoints, from scratch** (so not a
warm-start artifact). Verified *not* a plumbing bug: the audio projections diverged
~20 % from their text counterparts and appear in the optimizer.

Best explanation is gradient dilution — splitting the projections halves the signal
each copy receives per step. The capacity was not the binding constraint; the
`StackingEncoder` + projector ahead of the LLM already do the modality-specific work.

**The constraint helps; the capacity does not.**

### 4.2 Causal / fixed encoder output — does not work

`encoder_chunk_size: 1` pins the encoder to zero lookahead independently of the LLM
chunk, with `bidirectional_audio: true` so the LLM can see a chunk's frames both ways.
The hypothesis: a chunk's frames all arrive together, so the LLM can supply the
within-chunk context itself and latency comes only from the chunk boundary.

It does not. Warm-started from 0.0863, both arms plateau at ~0.107–0.108 and stay
there from step 24 k to 34 k — **~25 % worse, never recovering.** This reproduces an
earlier OCI run (0.0865 → 0.1068 over 104 k steps). Modality-local attention made no
difference within this setting either (sign alternated across all four checkpoints).

---

## 5. Measurement hazards

Several bugs produced confidently wrong numbers before being caught. Recorded because
each looked like a model result at first.

- **cuDNN SDPA returns NaN from its *backward*** while the forward stays finite
  (`ScaledDotProductCudnnAttentionBackward0`). Symptom: healthy loss and logits, then
  every gradient NaN on the same step. Worked around with `disable_cudnn_sdpa`.
  The streaming arms were immune only because they set `attn_backend: flex`.
- **Per-chunk token budgets silently truncate WER.** `val_max_new_tokens_per_chunk` is
  *per chunk*; a value sized for chunk 14 (24 tokens) truncates a whole-utterance
  decode past ~8 s. Produced `val_wer=1.3984`, which measured the cap, not the model.
- **`resume_if_exists` inherits poisoned checkpoints.** A cancelled NaN run writes a
  `-last.ckpt` that the next run silently resumes. Park checkpoints before any
  fix-verification relaunch.
- **`save_top_k` evicts checkpoints you intend to evaluate.** A matched step-32 000
  comparison became impossible because one arm's checkpoint was rotated away while
  the eval was queued. Evaluate promptly, or snapshot first.
- **Launchers forked from older recipes carry fixed bugs forward.** The `p84` arms
  re-hit both the missing `NEMO_USERS` mount and `NUM_WORKERS=8` (DFW has 128 CPUs vs
  OCI's 248) — two bugs this project had already fixed elsewhere.

### `val_wer` is unreliable below ~2 %

On mcv11 dev (~2 000 utterances) it repeatedly disagreed with the 74 842-utterance
leaderboard, in both directions:

- encoder-free n=0 vs n=4: `val_wer` preferred n=0 (0.1791 vs 0.1806); the leaderboard
  preferred n=4 (10.22 vs 10.06).
- `p84` n=0 vs n=4: `val_wer` preferred n=4 at three of four steps; the leaderboard
  split 2–2.

**Rank variants on the leaderboard at matched steps, with the per-corpus breakdown.**
Differences that come from datasets splitting 3–3–1 are noise, however clean the macro
average looks. Comparing each arm's *best* checkpoint is also invalid — doing so once
inflated a 1–2 % effect into an apparent 5 % win.

---

## 6. Open questions

- **Nothing has closed the gap to 5.84.** The untried levers are the ones Mel-LLM
  names as future work: distillation from the frozen Conformer (EVE-style MSE to its
  activations), an auxiliary CTC loss, and staged freezing.
- **Token rate.** We run 12.5 Hz (8× stacking). Mel-LLM's ablation finds 100 Hz and
  50 Hz give *better* WER (6.58 / 6.71 vs 7.12) at 3× the cost. Untested here.
- **Depth sweep resolution.** n=2 and n=6 would pin the modality-local optimum; we
  only know it lies near 4–8.
- **Late-training decay.** Every warm-started arm peaks early (step ~8 k) then drifts
  ~5 % worse. Cause not investigated; checkpoint selection currently compensates.

---

## 7. Reproducing

| arm | config | launcher |
|---|---|---|
| unified | `streaming_stt_granary2_full_script_nocoder_unified.yaml` | `dfw_script_nocoder_unified_8node.sh` |
| modality-local sweep | `..._script_mlocal{0,4,8,14}.yaml` | `dfw_script_mlocal{0,4,8,14}_8node.sh` |
| private q/k/v (negative) | `..._script_scratch_mqkv4.yaml` | `dfw_script_scratch_mqkv4_8node.sh` |
| encoder-based pair | `..._script_p84_mlocal{0,4}.yaml` | `dfw_script_p84_mlocal{0,4}_8node.sh` |
| causal encoder (negative) | `..._script_p84_causalenc_mlocal{0,4}.yaml` | `dfw_script_p84_causalenc_mlocal{0,4}_8node.sh` |

Evals: `launch/dfw_eval_*_cs14.sh` and `*_cs1024.sh`. The chunk-1024 variants need
`MAX_NEW_TOKENS=256`; the default of 64 truncates.
