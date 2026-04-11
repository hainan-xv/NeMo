# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CHAT-based self-alignment for streaming speech LLM.

Uses a frozen CHAT (Chunk-wise Attention Transducer) model to extract
chunk-level word alignments from encoder output, replacing the need for
an external forced aligner.

The alignment is computed via posterior-mode decoding: the RNNT forward
(alpha) and backward (beta) variables are computed using existing Numba
CUDA kernels, then the most likely emission chunk for each token is
determined by argmax over the per-token posterior.
"""

from __future__ import annotations

import multiprocessing
from typing import List, Optional, Tuple

import torch
from numba import cuda
from torch import Tensor

from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants, rnnt_helper
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt, gpu_rnnt_kernel
from nemo.collections.speechlm2.parts.alignments import WordAlignment


def compute_posterior_alignment(
    acts: Tensor,
    labels: Tensor,
    input_lengths: Tensor,
    label_lengths: Tensor,
    blank_id: int,
) -> Tensor:
    """Compute posterior-mode chunk assignment for each token using RNNT alpha/beta.

    Runs the RNNT forward and backward passes (via existing Numba CUDA kernels)
    to obtain alpha and beta tensors, then for each ground-truth token u finds
    the chunk t that maximises::

        alpha[t, u] + log_prob(t, u, labels[u]) + beta[t, u+1]

    Args:
        acts: Raw logits from the CHAT joint, shape ``(B, T_chunks, U, V+1)``.
            Must **not** be log-softmaxed — the RNNT kernels handle normalisation
            internally via a denominator tensor.
        labels: Padded ground-truth CHAT token IDs, shape ``(B, U_max)``.
            ``U_max`` equals ``max(label_lengths)`` (the labels themselves,
            without the SOS/blank prepended).
        input_lengths: Number of valid chunks per utterance, shape ``(B,)``.
        label_lengths: Number of valid tokens per utterance, shape ``(B,)``.
        blank_id: Index of the RNNT blank token in the CHAT vocabulary.

    Returns:
        ``chunk_assignments`` — ``LongTensor`` of shape ``(B, U_max)`` where
        ``chunk_assignments[b, u]`` is the chunk index at which token ``u``
        was most likely emitted. Values beyond ``label_lengths[b]`` are
        undefined.
    """
    # The Numba RNNT kernels operate in float32.
    acts = acts.float()
    labels = labels.int()
    input_lengths = input_lengths.int()
    label_lengths = label_lengths.int()

    B, maxT, maxU, V1 = acts.shape

    # --- 1. Allocate workspace and launch kernels --------------------------
    gpu_size, status = rnnt_helper.get_workspace_size(maxT, maxU, B, gpu=True)
    assert status == global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    cuda.select_device(acts.device.index)
    gpu_workspace = torch.zeros(gpu_size, device=acts.device, dtype=torch.float32)

    if hasattr(cuda, 'external_stream'):
        stream = cuda.external_stream(torch.cuda.current_stream(acts.device).cuda_stream)
    else:
        stream = cuda.default_stream()

    acts_flat = acts.contiguous().view(-1)

    wrapper = gpu_rnnt.GPURNNT(
        minibatch=B,
        maxT=maxT,
        maxU=maxU,
        alphabet_size=V1,
        workspace=gpu_workspace,
        blank=blank_id,
        fastemit_lambda=0.0,
        clamp=-1.0,
        num_threads=min(multiprocessing.cpu_count(), 4),
        stream=stream,
    )

    _, (denom_ws, alphas_ws, betas_ws, llForward_ws, llBackward_ws) = wrapper._prepare_workspace()

    # log_softmax denominator
    wrapper.log_softmax(cuda.as_cuda_array(acts_flat), denom_ws)

    # Forward pass (alphas)
    gpu_rnnt_kernel.compute_alphas_kernel[B, maxU, stream, 0](
        cuda.as_cuda_array(acts_flat),
        denom_ws,
        alphas_ws,
        llForward_ws,
        cuda.as_cuda_array(input_lengths),
        cuda.as_cuda_array(label_lengths),
        cuda.as_cuda_array(labels),
        B,
        maxT,
        maxU,
        V1,
        blank_id,
    )

    # Backward pass (betas)
    gpu_rnnt_kernel.compute_betas_kernel[B, maxU, stream, 0](
        cuda.as_cuda_array(acts_flat),
        denom_ws,
        betas_ws,
        llBackward_ws,
        cuda.as_cuda_array(input_lengths),
        cuda.as_cuda_array(label_lengths),
        cuda.as_cuda_array(labels),
        B,
        maxT,
        maxU,
        V1,
        blank_id,
    )

    stream.synchronize()

    # --- 2. Extract alphas, betas, denom from workspace --------------------
    # The workspace is a flat PyTorch tensor; the Numba kernels wrote into it
    # in-place, so we can read it back through the same tensor.
    BTU = B * maxT * maxU
    denom = gpu_workspace[:BTU].reshape(B, maxT, maxU)
    alphas = gpu_workspace[BTU : 2 * BTU].reshape(B, maxT, maxU)
    betas = gpu_workspace[2 * BTU : 3 * BTU].reshape(B, maxT, maxU)

    # --- 3. Compute per-token posterior and argmax -------------------------
    # log_prob(b, t, u, v) = denom[b, t, u] + acts[b, t, u, v]
    # For each token u we need log_prob at v = labels[b, u].
    # labels has shape (B, U_max) where U_max = maxU - 1 (the kernel uses maxU = U_labels + 1).
    U_labels = maxU - 1
    if labels.shape[1] < U_labels:
        # Pad labels if needed
        labels = torch.nn.functional.pad(labels, (0, U_labels - labels.shape[1]), value=0)

    label_indices = labels[:, :U_labels].long()  # (B, U_labels)
    # Gather the activation at each label position: acts[b, t, u, labels[b, u]]
    # acts shape: (B, maxT, maxU, V1)
    label_idx_expanded = label_indices.unsqueeze(1).unsqueeze(-1).expand(B, maxT, U_labels, 1)
    # We need acts[:, :, :U_labels, :] → gather along dim 3
    acts_at_labels = torch.gather(acts[:, :, :U_labels, :], dim=3, index=label_idx_expanded).squeeze(-1)
    # acts_at_labels: (B, maxT, U_labels)

    # log_prob at label positions
    log_prob_labels = denom[:, :, :U_labels] + acts_at_labels  # (B, maxT, U_labels)

    # Posterior: alpha[b, t, u] + log_prob(b, t, u, labels[b, u]) + beta[b, t, u+1]
    posterior = alphas[:, :, :U_labels] + log_prob_labels + betas[:, :, 1:maxU]
    # posterior shape: (B, maxT, U_labels)

    # Mask invalid time steps and token positions
    t_valid = torch.arange(maxT, device=acts.device).unsqueeze(0) < input_lengths.unsqueeze(1)  # (B, maxT)
    u_valid = torch.arange(U_labels, device=acts.device).unsqueeze(0) < label_lengths.unsqueeze(1)  # (B, U_labels)
    mask = t_valid.unsqueeze(2) & u_valid.unsqueeze(1)  # (B, maxT, U_labels)
    posterior = posterior.masked_fill(~mask, float('-inf'))

    # Argmax over time dimension
    chunk_assignments = posterior.argmax(dim=1)  # (B, U_labels)

    del gpu_workspace, wrapper
    return chunk_assignments


def tokens_to_word_chunks(
    token_ids: List[int],
    chunk_indices: List[int],
    tokenizer_vocab: List[str],
    chunk_size: int,
    frame_length_in_secs: float,
) -> List[WordAlignment]:
    """Convert BPE token-level chunk assignments to word-level WordAlignment objects.

    Groups consecutive BPE tokens into words using the SentencePiece ``▁``
    (U+2581) word-boundary prefix.  Each word is assigned to the chunk
    where its **last** subword was emitted.

    The returned ``WordAlignment`` objects have ``start_time`` / ``end_time``
    set so that the existing ``get_llm_messages_for_sample`` function assigns
    the word to the correct chunk (assuming ``num_delay_frames = 0``).

    Args:
        token_ids: CHAT BPE token IDs for the utterance (length U).
        chunk_indices: Per-token chunk assignments (length U).
        tokenizer_vocab: The CHAT tokenizer's vocabulary list, indexed by token ID.
        chunk_size: Number of encoder frames per chunk.
        frame_length_in_secs: Duration of a single encoder output frame.

    Returns:
        List of :class:`WordAlignment` objects.
    """
    if not token_ids:
        return []

    words: List[WordAlignment] = []
    current_pieces: List[str] = []
    current_last_chunk = chunk_indices[0]

    for token_id, chunk_idx in zip(token_ids, chunk_indices):
        piece = tokenizer_vocab[token_id]
        is_word_start = piece.startswith('▁')

        if is_word_start and current_pieces:
            # Flush the accumulated word.
            word_text = ''.join(current_pieces).replace('▁', '')
            if word_text:
                # Place the word so its end_time falls within current_last_chunk.
                start_time = current_last_chunk * chunk_size * frame_length_in_secs
                end_time = (current_last_chunk * chunk_size + 1) * frame_length_in_secs
                words.append(WordAlignment(text=word_text, start_time=start_time, end_time=end_time))
            current_pieces = [piece]
            current_last_chunk = chunk_idx
        else:
            current_pieces.append(piece)
            current_last_chunk = chunk_idx

    # Flush last word.
    if current_pieces:
        word_text = ''.join(current_pieces).replace('▁', '')
        if word_text:
            start_time = current_last_chunk * chunk_size * frame_length_in_secs
            end_time = (current_last_chunk * chunk_size + 1) * frame_length_in_secs
            words.append(WordAlignment(text=word_text, start_time=start_time, end_time=end_time))

    return words


class ChatAligner(torch.nn.Module):
    """Chunk-level word aligner based on a frozen CHAT transducer model.

    Holds the CHAT prediction network (decoder), joint network, and BPE
    tokenizer. Given encoder output and ground-truth text, it:

    1. Tokenizes text with the CHAT BPE tokenizer.
    2. Runs the CHAT prediction network on ground-truth tokens.
    3. Computes the CHAT joint to get the RNNT lattice.
    4. Extracts posterior-mode alignment via RNNT alpha/beta.
    5. Converts subword-level chunk assignments to word-level alignments.

    The encoder is **shared** with the streaming speech LLM — this class
    does not own or run the encoder.  Registered as an ``nn.Module`` so
    that PyTorch Lightning automatically moves the frozen sub-modules to
    the correct device.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        joint: torch.nn.Module,
        tokenizer,
        blank_id: int,
        chunk_size: int,
        frame_length_in_secs: float,
    ):
        super().__init__()
        self.chat_decoder = decoder
        self.chat_joint = joint
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.chunk_size = chunk_size
        self.frame_length_in_secs = frame_length_in_secs

        # Build vocabulary list for word boundary detection.
        if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'id_to_piece'):
            self._vocab = [tokenizer.tokenizer.id_to_piece(i) for i in range(tokenizer.tokenizer.get_piece_size())]
        elif hasattr(tokenizer, 'vocab'):
            self._vocab = tokenizer.vocab
        else:
            raise ValueError("Cannot extract vocabulary from CHAT tokenizer")

        self.chat_decoder.eval()
        self.chat_joint.eval()
        # The RNNT kernels need raw (un-normalised) logits from the joint.
        self.chat_joint.log_softmax = None

        for p in self.chat_decoder.parameters():
            p.requires_grad_(False)
        for p in self.chat_joint.parameters():
            p.requires_grad_(False)

    def _tokenize_batch(self, texts: List[str], device: torch.device) -> Tuple[Tensor, Tensor]:
        """Tokenize a batch of texts with the CHAT BPE tokenizer.

        Returns:
            labels: (B, U_max) padded token IDs.
            label_lengths: (B,) actual lengths.
        """
        all_ids = []
        for text in texts:
            if hasattr(self.tokenizer, 'text_to_ids'):
                ids = self.tokenizer.text_to_ids(text)
            else:
                ids = self.tokenizer.encode(text)
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids) if all_ids else 0
        labels = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
        label_lengths = torch.zeros(len(texts), dtype=torch.long, device=device)

        for i, ids in enumerate(all_ids):
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            label_lengths[i] = len(ids)

        return labels, label_lengths

    @torch.no_grad()
    def align(
        self,
        encoder_output: Tensor,
        encoder_lengths: Tensor,
        texts: List[str],
    ) -> List[List[WordAlignment]]:
        """Compute chunk-level word alignments for a batch.

        Args:
            encoder_output: Raw encoder output, shape ``(B, D, T)``
                (channel-first, as returned by the Conformer encoder).
            encoder_lengths: Frame-level lengths, shape ``(B,)``.
            texts: Ground-truth transcript strings.

        Returns:
            Per-sample word alignments, compatible with
            ``get_llm_messages_for_sample``.
        """
        device = encoder_output.device
        B = encoder_output.shape[0]

        # 1. Tokenize ground-truth with CHAT tokenizer.
        labels, label_lengths = self._tokenize_batch(texts, device)

        if labels.shape[1] == 0:
            return [[] for _ in range(B)]

        # 2. Prepend blank (SOS) for the CHAT decoder input.
        sos = torch.full((B, 1), self.blank_id, dtype=torch.long, device=device)
        decoder_input = torch.cat([sos, labels], dim=1)  # (B, U+1)
        decoder_input_lengths = label_lengths + 1

        # 3. Run the frozen CHAT prediction network.
        decoder_output, _, _ = self.chat_decoder(
            targets=decoder_input,
            target_length=decoder_input_lengths,
        )  # decoder_output: (B, D_pred, U+1)

        # 4. Run the frozen CHAT joint (handles chunking internally).
        # joint() expects encoder (B, T, D) and decoder (B, U, D);
        # when f_len is 1D it chunks internally via chunk_concat_audio.
        encoder_bt = encoder_output.transpose(1, 2)   # (B, T, D)
        decoder_bt = decoder_output.transpose(1, 2)    # (B, U+1, D)
        logits = self.chat_joint.joint(encoder_bt, decoder_bt, encoder_lengths)
        # logits: (B, num_chunks, U+1, V+1) — raw logits (not log-softmaxed on GPU)

        # The joint stores the number of valid chunks per utterance.
        input_lengths_chunks = self.chat_joint.num_chunks_per_utterance  # (B,)

        # 5. Compute posterior alignment.
        chunk_assignments = compute_posterior_alignment(
            acts=logits,
            labels=labels,
            input_lengths=input_lengths_chunks,
            label_lengths=label_lengths,
            blank_id=self.blank_id,
        )

        # 6. Convert to word-level WordAlignment objects.
        batch_alignments: List[List[WordAlignment]] = []
        for b in range(B):
            n_tokens = label_lengths[b].item()
            if n_tokens == 0:
                batch_alignments.append([])
                continue
            tok_ids = labels[b, :n_tokens].tolist()
            tok_chunks = chunk_assignments[b, :n_tokens].tolist()
            word_aligns = tokens_to_word_chunks(
                token_ids=tok_ids,
                chunk_indices=tok_chunks,
                tokenizer_vocab=self._vocab,
                chunk_size=self.chunk_size,
                frame_length_in_secs=self.frame_length_in_secs,
            )
            batch_alignments.append(word_aligns)

        return batch_alignments
