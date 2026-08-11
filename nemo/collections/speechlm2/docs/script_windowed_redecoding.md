# Windowed re-decoding self-correction for SCRIPT

Status: design + reference for the `redecode` feature in the SCRIPT streaming
SpeechLM (`nemo/collections/speechlm2/{parts/script.py, data/script_dataset.py,
models/script_model.py}`).

This document specifies a **second** self-correction mechanism for SCRIPT that
replaces the in-stream `<del>` backspace token with a **windowed re-decoding**
scheme. It is self-contained; familiarity with the base SCRIPT layout (spine +
per-chunk branches) is assumed.

---

## 1. Motivation

The existing self-correction mechanisms (`self_correction` forced/DAgger, and
`self_correction_prefix`) teach the model a `<del>` token: it keeps a
mis-committed word `W'` in its context and learns the non-monotonic semantics
"the token before `<del>` is void, here is the fix". Two weaknesses:

1. **The corrected re-emission conditions on the wrong token.** The history
   transiently reads `... W' <del> w_prev ...`; the model has to actively negate
   its own context, and `W'` can bias the re-emission.
2. **The redo sees no new evidence.** The re-emission is produced from
   essentially the *same acoustic window* that caused the original error, so
   there is little reason for the second attempt to be correct. Streaming errors
   are overwhelmingly caused by committing a word before enough **right-context
   (lookahead)** audio has arrived — the `<del>` design fixes the transcript but
   not the cause.

**Windowed re-decoding** attacks the cause directly: every chunk is transcribed
several times, each time with **one more chunk of lookahead audio**, always
conditioning on **clean (believed-correct) history**. Early emissions are
low-latency provisional previews; later re-decodes — with more lookahead —
progressively refine and finally *lock* each chunk. There is no `<del>` token
and, in the delayed-commit regime, no need to manufacture synthetic errors at
all: correction falls out of re-decoding with more context.

---

## 2. Formulation

Let the utterance be a sequence of audio chunks with word targets
`y_0, y_1, ..., y_{T-1}` (a chunk may be empty/silent). Fix a window size of
`N` chunks; write `M = N - 1` (this is exactly the existing
`audio_history_chunks`). Fix a maximum **rollback depth** `R` with `1 <= R <= M`
(default `R = M`).

For a chunk `c` and a **lookahead level** `j in {0, 1, ..., R}` we model

```
p( y_c | instruction, y_0 .. y_{c-1},  audio[ max(0, c+j-M) .. c+j ] )
```

i.e. predict chunk `c`'s words given (a) the **clean text history** of all
earlier chunks and (b) an `N`-chunk audio window that ends `j` chunks **after**
`c`. `j` chunks of that window are pure lookahead (their words are *not* in the
history and are *not* predicted here); the model must transcribe only chunk `c`
and then stop (emit `<eot>`).

- `j = 0` is exactly the base SCRIPT branch with `audio_history_chunks = M`
  (window ends at `c`, zero lookahead).
- `j = R` is the maximal-lookahead re-decode used to **lock** chunk `c`.

The key property: **the history is always the clean prefix `y_0..y_{c-1}`** for
every `j`. Nothing wrong ever sits in the context; "correction" is simply
re-decoding chunk `c` with more lookahead. If an early (low-`j`) preview was
wrong, the later (high-`j`) re-decode overwrites it, because both are produced
from the same clean history — the higher-`j` one just had more evidence.

### 2.1 Why single-chunk targets (not spans)

An earlier idea was to have level `j` predict the *whole span* `y_{c} .. y_{c+j}`
jointly. We deliberately use **single-chunk targets** (`y_c` + `<eot>`) instead,
because a flat span target concatenates several chunks' words with a single
trailing `<eot>` and provides **no per-chunk boundary** to segment at inference
(we could not tell where `y_c` ends and `y_{c+1}` begins to lock just `y_c`).
Single-chunk targets keep the clean "words + `<eot>`" segmentation of base
SCRIPT, and at inference we still cover every chunk each step by running one
branch per lookahead level. The two formulations induce the same per-step
inference behaviour; single-chunk targets are simply implementable.

---

## 3. Training layout

Reuse the base spine: `spine = [instruction] y_0 y_1 ... y_{T-1}` (pure text,
loss-free, computed once). Then, instead of one branch per chunk, emit **one
branch per `(c, j)` pair**:

```
branch(c, j):  <vs> [ audio window  max(0,c+j-M) .. c+j ] <ve>  y_c <eot>
   history prefix = instruction + y_0..y_{c-1}   (prefix_len = |spine[:start of c]|)
   positions      = pref, pref+1, ...            (default contiguous convention)
   supervised     = y_c then <eot>               (single chunk + end-of-turn)
```

with `j in {0 .. min(R, T-1-c)}` (a level is only valid when the window end
`c+j` is a real chunk). `j = 0` reproduces the base branch, so
`redecode` is a strict superset of `audio_history_chunks = M`.

Because branches remain mutually non-attending (each attends only its own
history prefix of the spine + its own audio + its own earlier tokens), the base
4D mask (`build_script_mask`) and the parity argument carry over unchanged: a
`(c, j)` branch's logits equal those of the standalone example
`[instruction y_0..y_{c-1}] <vs> window <ve> y_c <eot>`.

Builder: `build_packed_redecode_example(...)` in `parts/script.py`
(oracle: `build_separate_redecode_examples(...)`).

**Cost / subsampling.** Adding levels `j = 1..R` multiplies the branch count by
up to `R+1` and lengthens the packed sequence. `redecode_train_prob` (default
`1.0`) includes each `j >= 1` branch with that probability, so training cost is
tunable; `j = 0` is always kept (the base objective).

**No training-step change.** The model's `training_step` already takes the CE
loss over *all* supervised branch positions; the extra `(c, j)` branches are
just more branches. Only inference (`generate`) needs a new path.

---

## 4. Inference

Window `N = M+1` chunks, rollback depth `R`. Per stream keep the transcript
**segmented per chunk**: `committed[c]` is the current best token list for chunk
`c` (finalized once `c` is locked). History for decoding chunk `c` is always
`concat(committed[0..c-1])`.

At audio step `t` (chunk `t` has just arrived) the audio window is
`frames[max(0,t-M) .. t]`. Process lookahead levels **deepest first**
(`j = min(R,t) down to 0`, `c = t - j`):

```
for j in [min(R, t) .. 0]:            # deepest (most lookahead / lock) first
    c = t - j
    history = concat(committed[0..c-1])         # clean, mostly locked
    words   = greedy_decode( [instruction | history] <vs> window <ve> -> ... <eot> )
    committed[c] = words                        # lock if j == R, else provisional preview
```

- **Locked stream** (`j = R`, chunk `c = t-R`): its history `committed[0..c-1]`
  is fully locked (those chunks locked in earlier steps), so the locked stream
  is self-consistent and needs no speculation. This is plain **delayed decoding
  with `R` chunks of lookahead**; the finalized transcript lags the newest audio
  by `R` chunks.
- **Previews** (`j < R`): display-only refinements of the trailing `R` chunks.
  They may be revised on subsequent steps; they never feed the locked stream, so
  their (mildly speculative) conditioning cannot corrupt the final output.

Deepest-first ordering means that within a step, chunk `t-R` is locked before it
is used as history for the shallower previews.

At the final step the trailing chunks that never reached `j = R` are locked at
the largest lookahead the audio allows (`j = T-1-c`).

**Latency story (adaptive display, delayed lock).** Each chunk is *shown*
immediately as a `j=0` preview (zero added latency) and *locked* `R` chunks
later. If previews and their eventual lock agree — the common case — the user
perceives no revision. `R` trades finalization latency for accuracy: larger `R`
= more lookahead + deeper corrections, but a longer lock lag and ~`R+1`x
training branches / decodes per step (batched, one shared audio encode).

**Two operating points from one model.** Reporting the `j=0` predictions gives a
*streaming/provisional WER* (no lookahead); the locked `j=R` predictions give a
*finalized WER* (`R`-chunk lookahead). Same weights.

Decoder: `batched_stream_decode_redecode(...)` in `parts/script.py`; parity with
training is checked the same way as the base decoder (greedy stream == argmax of
a teacher-forced packed forward of the emitted tokens).

---

## 5. Configuration

Model **and** dataset config (both must agree; the model re-reads them after
`from_pretrained`):

| Key | Where | Meaning | Default |
|-----|-------|---------|---------|
| `redecode` | model + dataset | enable windowed re-decoding | `false` |
| `audio_history_chunks` (`M`) | model + dataset | window size `N = M+1`; must be `>= 1` when `redecode` | `0` |
| `redecode_depth` (`R`) | model + dataset | max rollback / lookahead depth, `1 <= R <= M` | `M` |
| `redecode_train_prob` | dataset | probability of including each `j >= 1` branch | `1.0` |

**Mutual exclusions** (guarded): `redecode` requires `audio_history_chunks >= 1`
and `audio_window_frames == 0`, and is incompatible with `shared_audio_track`,
`contiguous_text_positions`, `self_correction`, `self_correction_prefix`,
`history_word_recovery_prob > 0`, and `script_last_layer_history_tokens > 0`.

Example: `examples/speechlm2/conf/streaming_stt_granary2_lora_script_redecode.yaml`.

---

## 6. Relationship to the other mechanisms

- **vs `<del>` self-correction:** no delete token, no synthetic-error
  manufacturing (in the delayed-commit regime), corrected output always
  conditions on clean history, and the redo actually sees new audio. Cost is a
  fixed `R`-chunk finalization lag and `R+1`x decode passes.
- **vs `history_word_recovery`:** recovery drops a word to force re-emission
  from the *current* chunk's audio; re-decoding instead re-emits the whole chunk
  from a *wider* window with real lookahead, and does so for every chunk.
- **vs base SCRIPT (`audio_history_chunks = M`):** identical `j=0` branch;
  windowed re-decoding adds the `j=1..R` lookahead branches and the
  provisional/locked inference loop.
