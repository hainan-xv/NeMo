# Streaming Speech LLM with CHAT-Based Self-Alignment

## 1. Motivation

The current Streaming STT model requires **external word-level forced alignment** to construct its training data. This alignment tells the model which words have been "completed" by each audio chunk, enabling the multi-turn conversation structure (user=audio chunk, assistant=transcribed words or blank).

Two alignment sources exist today:
- **Pre-computed alignments** stored in Lhotse manifests (requires offline preprocessing)
- **Online forced alignment** via `QwenForcedAligner` (requires a separate 0.6B model at training time)

Both add pipeline complexity and introduce a dependency on an external alignment system whose chunk/frame granularity doesn't inherently match the streaming model's own chunk structure.

**Proposed change**: Replace the external aligner with the **CHAT (Chunk-wise Attention Transducer)** model's own RNNT forward-backward algorithm. Since the CHAT model operates on the same chunked encoder output that the streaming speech LLM uses, its alignment is chunk-aligned by construction — no conversion needed.

## 2. Background: How CHAT Works

The CHAT model (`RNNTAttJoint`) is an RNN-T variant where the joint network uses cross-attention over chunks of encoder frames rather than the standard additive joint over individual frames.

Key properties:
- **Encoder**: Streaming Conformer producing frame-level features `(B, D, T)`.
- **Chunking**: `chunk_concat_audio()` groups frames into chunks of size `chunk_size`, producing `(B, num_chunks, chunk_size * D)`.
- **Joint**: Cross-attention from decoder queries `(B, U, D)` into each chunk's encoder frames, producing an RNNT lattice of shape `(B, num_chunks, U, V+1)`.
- **Critical**: The RNNT lattice's time axis is **chunks**, not individual frames.

Because of this, when we run forward-backward or Viterbi on the CHAT lattice, we get a **chunk-level alignment** — each output token is assigned to a chunk, which maps directly to the streaming speech LLM's turn structure.

## 3. Architecture Overview

### Components

| Component | Source | Frozen? | Purpose |
|-----------|--------|---------|---------|
| Conformer Encoder | Pretrained ASR (e.g., `nemotron-speech-streaming-en-0.6b`) | Configurable | Audio → frame features |
| CHAT Decoder (RNN prediction net) | Pretrained CHAT model | **Yes** (always) | Part of alignment oracle |
| CHAT Joint (`RNNTAttJoint`) | Pretrained CHAT model | **Yes** (always) | Chunk-level RNNT lattice for alignment |
| Modality Adapter + Projection | Initialized or from pretrained | Configurable | Adapt encoder dim → LLM dim |
| LLM (Qwen3-1.7B) | Pretrained HuggingFace | Configurable | Text generation backbone |
| LLM Tokenizer | Qwen3 tokenizer | — | Tokenizes assistant turns for LLM |
| CHAT Tokenizer | From pretrained CHAT model | — | Tokenizes ground-truth text for RNNT alignment |

### Shared Encoder

The Conformer encoder is **shared** between the CHAT alignment path and the LLM perception path. During training:
1. The encoder processes the full audio → frame features `(B, D, T)`
2. The CHAT path chunks these frames and computes the RNNT lattice (for alignment extraction)
3. The LLM path projects these frames to LLM dimension and interleaves with text embeddings (for language modeling loss)

This means the encoder receives gradients from the LLM loss but NOT from the CHAT alignment (which runs under `torch.no_grad()`).

**Open question**: Should the CHAT alignment path and the LLM perception path share the same encoder forward pass, or should we run the encoder twice (once for each)? Sharing is more efficient but means the CHAT decoder/joint must accept the same encoder output dimension and configuration as the LLM perception path.

**Decision**: Share the encoder. The pretrained CHAT model's encoder IS the same architecture (streaming Conformer) that the speech LLM uses. We load the full CHAT model and use its encoder as the speech LLM's encoder. The CHAT decoder and joint are kept as frozen auxiliary modules.

### Tokenizer Setup

Two tokenizers coexist:
- **CHAT BPE tokenizer** (~1024 tokens): Used to tokenize ground-truth text into RNNT labels for alignment extraction.
- **Qwen3 tokenizer** (~150K tokens): Used to tokenize assistant turn content for LLM training/inference.

The alignment flow converts between them: CHAT tokenizer → alignment → detokenize to text → Qwen tokenizer.

## 4. Training Pipeline

### Step-by-step for each batch

#### 4.1 Encode Audio
```
audio (B, T_samples) → Conformer Encoder → encoder_output (B, D, T_frames), encoder_lengths (B,)
```

#### 4.2 Extract Chunk-Level Alignment (frozen, no_grad)

**4.2.1** Tokenize ground-truth text with CHAT BPE tokenizer:
```
"unfortunately is a word" → [▁un, for, tun, ate, ly, ▁is, ▁a, ▁word]
```

**4.2.2** Chunk the encoder output:
```
chunk_concat_audio(encoder_output, encoder_lengths, chat_chunk_size)
  → chunked_encoder (B, num_chunks, chunk_size * D)
  → chunk_frame_lengths (B, num_chunks)
```

**4.2.3** Run CHAT decoder (prediction net) on ground-truth tokens:
```
chat_decoder(ground_truth_token_ids, token_lengths)
  → decoder_output (B, U, D_pred)
```

**4.2.4** Compute CHAT joint:
```
chat_joint(chunked_encoder, decoder_output, chunk_frame_lengths)
  → rnnt_logits (B, num_chunks, U+1, V+1)
```

**4.2.5** Extract Viterbi alignment from the RNNT lattice:
```
viterbi(rnnt_logits, encoder_lengths_in_chunks, token_lengths)
  → alignment: for each ground-truth token, which chunk it was emitted at
```

**4.2.6** Convert subword alignment to word-level chunk assignment:
- Group CHAT BPE tokens into words using word-boundary markers (e.g., `▁` prefix)
- Assign each word to the chunk where its **last subword** was emitted
- Result: `{chunk_idx: [list of words]}` per sample

#### 4.3 Build Multi-Turn Conversation

Using the word-to-chunk mapping and the **ground-truth transcript**:
```
system: "Transcribe the audio into text."
user: <audio> * chunk_size          # chunk 0
assistant: <blank>
user: <audio> * chunk_size          # chunk 1
assistant: <blank>
...
user: <audio> * chunk_size          # chunk 6
assistant: "unfortunately"
...
user: <audio> * chunk_size          # chunk 8
assistant: "is a"
user: <audio> * chunk_size          # chunk 9
assistant: "word"
```

The assistant turn text comes from the **ground-truth transcript** (not CHAT's predictions), preserving punctuation, casing, etc. The CHAT model only provides the timing.

**Important detail**: We need to map CHAT's word-level assignment back to the ground-truth transcript. Since we tokenized the ground-truth text with the CHAT tokenizer, the words recovered from detokenization should match the ground-truth words (modulo tokenizer normalization). We then locate these words in the original transcript to preserve original casing and punctuation — similar to how `compute_word_spans()` works in the current code.

#### 4.4 Tokenize for LLM and Compute Loss

Same as the current pipeline:
1. Tokenize the multi-turn conversation with Qwen's tokenizer
2. Replace audio tag tokens with `AUDIO_TOKEN_IDX`
3. Build input/target token sequences with `IGNORE_INDEX` masking on non-assistant positions
4. Run encoder output through modality adapter + projection → audio embeddings
5. Interleave audio and text embeddings
6. Forward through LLM, compute cross-entropy loss on assistant turns

## 5. Chunk Size Alignment

### The Problem

The CHAT model and the streaming speech LLM may use different chunk sizes:
- CHAT model: `chunk_size` is typically `right_context + 1` from `att_context_size`. E.g., `att_context_size=[70, 13]` → `chunk_size=14`.
- Speech LLM: `chunk_size=7` in the current config (from `mine.sh`).

The CHAT's chunk-level alignment operates at the CHAT chunk size. If the LLM uses a different (smaller) chunk size, we need to map CHAT chunks to LLM chunks.

### Options

**Option A: Use the same chunk size for both.**
- Simplest. Set the speech LLM's chunk_size = CHAT's chunk_size.
- May require retraining the CHAT model if you need a specific chunk size.

**Option B: CHAT chunk is a multiple of LLM chunk.**
- E.g., CHAT chunk=14, LLM chunk=7 → each CHAT chunk spans 2 LLM chunks.
- If CHAT assigns a word to CHAT chunk `c`, we assign it to LLM chunk `2*c + 1` (the last LLM chunk within that CHAT chunk).
- Straightforward integer mapping.

**Option C: Independent chunk sizes with frame-level remapping.**
- Convert CHAT chunk indices back to frame ranges, then map to LLM chunk indices.
- CHAT chunk `c` spans frames `[c * chat_chunk_size, (c+1) * chat_chunk_size)`.
- The word's last subword at CHAT chunk `c` → frame range end = `(c+1) * chat_chunk_size`.
- LLM chunk = `floor(((c+1) * chat_chunk_size - 1) / llm_chunk_size)`.
- Most general, works for any combination.

**Decision**: Implement Option C (most general), but in practice encourage Option A or B.

## 6. Posterior-Mode Alignment Extraction

### Approach

We use **posterior-mode alignment** to extract chunk assignments from the RNNT lattice. This reuses the existing RNNT forward (alpha) and backward (beta) CUDA kernels from `nemo.collections.asr.parts.numba.rnnt_loss` — no new Viterbi pass or backpointer tensor is needed.

### How It Works

1. Run `compute_alphas_kernel` → forward variable `alpha[b, t, u]`
2. Run `compute_betas_kernel` → backward variable `beta[b, t, u]`
3. For each ground-truth token `u`, compute the posterior emission probability at each chunk `t`:
   ```
   posterior(t, u) = alpha[t, u] + log_prob(t, u, labels[u]) + beta[t, u+1]
   ```
4. The best chunk for token `u` is `argmax_t posterior(t, u)`.

This is implemented in `compute_posterior_alignment()` in `chat_aligner.py`. The alpha/beta kernels run on GPU via Numba CUDA, and the argmax is a simple batched PyTorch operation.

### Why Not Full Viterbi?

Full Viterbi requires materializing a backpointer tensor of shape `(B, T, U)` and a sequential backtracking step. The posterior-mode approach instead:
- Reuses the **exact same** `compute_alphas_kernel` and `compute_betas_kernel` that the RNNT loss already uses (no new CUDA code).
- Needs only a single `argmax` per token (O(T) per token, O(T * U) total).
- Gives results that are equivalent to Viterbi in practice (the argmax over a smooth posterior converges to the Viterbi path for well-trained models).

### Computational Cost

The RNNT lattice for CHAT has shape `(num_chunks, U+1, V+1)`. For a 10-second utterance with chunk_size=7 and 80ms frames: `ceil(125/7) = 18 chunks`. With U ≈ 30 tokens and V ≈ 1024, the lattice is small. The alpha/beta computation is `O(num_chunks * U)` which is negligible.

The expensive part is the CHAT joint forward pass to compute the lattice, which is `O(num_chunks * U * V)`. This is comparable to one RNNT loss computation.

## 7. Word Boundary Detection

### How BPE Word Boundaries Work

SentencePiece BPE (used by the CHAT model) marks word-initial tokens with a `▁` (U+2581) prefix. For example:
```
"the cat sat" → ["▁the", "▁cat", "▁sat"]
"unfortunately" → ["▁un", "for", "tun", "ate", "ly"]
```

A new word starts whenever a token begins with `▁`. Tokens without `▁` are continuations of the current word.

### Algorithm

Given a Viterbi alignment producing `[(token_id, chunk_idx), ...]`:

```python
words = []
current_word_tokens = []
current_word_last_chunk = -1

for token_id, chunk_idx in alignment:
    token_text = chat_tokenizer.id_to_token(token_id)
    if token_text.startswith('▁') and current_word_tokens:
        # New word boundary — flush previous word
        words.append({
            'text': chat_tokenizer.decode(current_word_tokens),
            'chunk': current_word_last_chunk
        })
        current_word_tokens = [token_id]
        current_word_last_chunk = chunk_idx
    else:
        current_word_tokens.append(token_id)
        current_word_last_chunk = chunk_idx

# Don't forget the last word
if current_word_tokens:
    words.append({
        'text': chat_tokenizer.decode(current_word_tokens),
        'chunk': current_word_last_chunk
    })
```

### Mapping Back to Ground-Truth Transcript

The words recovered from CHAT tokenization may differ slightly from the original transcript (e.g., casing, punctuation that the BPE doesn't encode). We reuse the existing `compute_word_spans()` logic to locate each alignment word in the ground-truth transcript and preserve the original text.

## 8. Handling Edge Cases

### 8.1 Empty Chunks (No Tokens Emitted)
If no word has its last subword at a given chunk, the assistant turn is `<blank>`. This is identical to the current behavior.

### 8.2 Many Words at One Chunk
If several words complete at the same chunk (e.g., short words like "is a" both ending within the same chunk), the assistant turn contains all of them: `"is a"`. Same as current behavior.

### 8.3 Very Long Words
A long word's subwords may span many chunks. The whole word is assigned to the last chunk. Earlier chunks that contained partial subwords will show as `<blank>` (unless other words also complete there). This is slightly more conservative (higher latency) than the forced-aligner approach for such words, but ensures acoustic completeness.

### 8.4 Tokenizer Normalization Mismatch
The CHAT BPE tokenizer may normalize text differently than expected (e.g., lowercasing, stripping punctuation). The ground-truth transcript may contain punctuation and mixed case. We handle this by:
- Using the CHAT tokenizer only for alignment timing
- Using the original ground-truth transcript text for assistant turn content
- Matching alignment words to transcript words using case-insensitive fuzzy matching (similar to `compute_word_spans()`)

### 8.5 CHAT Model Accuracy
If the CHAT model's encoder is weak for certain audio (noisy, accented, etc.), the Viterbi alignment may be poor — assigning words to wrong chunks. This is the same failure mode as any forced aligner. Since the CHAT model is pretrained on large-scale ASR data, it should be robust. Additionally, since we run forward-backward with ground-truth labels (not free decoding), alignment quality is generally high even when recognition would fail.

### 8.6 Chunk Size Mismatch
See Section 5. When CHAT and LLM chunk sizes differ, we convert CHAT chunk indices to LLM chunk indices using frame-level arithmetic.

## 9. Configuration

### New Config Fields

```yaml
model:
  # Existing fields...
  pretrained_chat: "path/to/chat_model.nemo"  # Pretrained CHAT model for alignment
  chat_chunk_size: null  # If null, inferred from CHAT model's config
  # chunk_size: 7  # LLM's chunk size (can differ from chat_chunk_size)

  # Remove forced_aligner section — no longer needed
```

### What Gets Loaded from the CHAT Model

From the pretrained CHAT `.nemo` checkpoint:
- `encoder` → becomes the shared encoder (also used by LLM perception)
- `decoder` (prediction net) → frozen, used only for alignment
- `joint` (RNNTAttJoint) → frozen, used only for alignment
- `tokenizer` → CHAT BPE tokenizer, used to tokenize ground-truth for alignment

The LLM, modality adapter, projection, and Qwen tokenizer are configured separately as before.

## 10. Inference

At inference time, the CHAT alignment machinery is **not used**. The streaming inference path (`generate_streaming`) remains unchanged:
1. Audio chunk → encoder → modality adapter → projection → audio embeddings
2. Construct turn template → interleave with text embeddings → forward through LLM
3. Autoregressive decode → tokens or blank

The CHAT decoder and joint are not needed at inference time and can be dropped from the model.

## 11. Summary of Changes

| File | Change |
|------|--------|
| `streaming_stt_model.py` | Load CHAT decoder+joint as frozen modules; add alignment extraction in `training_step` |
| `streaming_stt_dataset.py` | Add method to build conversation from chunk-level word assignments (not `WordAlignment` objects) |
| `pretrained.py` | Add function to load full CHAT model and extract components |
| New: `chat_aligner.py` | RNNT Viterbi alignment extraction, word boundary detection, chunk-to-word mapping |
| Config YAML | Add `pretrained_chat` field, remove `forced_aligner` section |
| `mine.sh` (launch script) | Update config path/name |

## 12. Open Questions

1. **Should the encoder be shared or separate?** (Decided: shared, see Section 3)

2. **Gradient flow through encoder**: The CHAT alignment runs under `no_grad`, so the encoder only gets gradients from the LLM loss. Is this desirable? The alternative would be to also backprop through the RNNT loss to keep the encoder "aligned" with the CHAT model, but this adds complexity and couples two loss functions.

3. **Online vs. offline alignment**: Should alignment be computed every training step (online), or once as a preprocessing step (offline)? Online is simpler (no preprocessing pipeline) and allows the alignment to adapt as the encoder trains. Offline is faster (no CHAT forward pass during training) but produces stale alignments if the encoder changes.

4. **What if the CHAT model uses a different encoder architecture or configuration than desired?** E.g., the CHAT model was trained with `att_context_size=[70, 13]` but you want the speech LLM to use `[70, 6]`. Since we share the encoder, we'd need a CHAT model trained with the desired encoder config, or accept the CHAT model's encoder config.

5. **Blank token semantics**: The current speech LLM uses a `<blank>` token added to the Qwen vocabulary. The CHAT model has its own blank token (index 0 in RNNT convention). These are separate concepts — the CHAT blank indicates "no emission at this chunk" in the RNNT lattice, while the LLM blank is a special token in the assistant vocabulary meaning "no speech this turn." The mapping is straightforward: CHAT blank at chunk → LLM `<blank>` in assistant turn.
