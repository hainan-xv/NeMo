# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import json
import os
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import numpy as np
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.aligner_loss import AlignerCrossEntropyLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss, resolve_rnnt_default_loss_name
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.modules.aligner import AlignerCTCHead, AlignerJoint
from nemo.collections.asr.modules.chunk_channel_token_mixer import ChunkChannelTokenMixer
from nemo.collections.asr.modules.chunk_token_extractor import ChunkTokenExtractor
from nemo.collections.asr.modules.rnnt import RNNTDecoderJoint
from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import ChunkedAlignerLossNumba, ChunkedAlignerNarLossNumba
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.aligner_decoding import AlignerDecoding
from nemo.collections.asr.parts.submodules.chunked_aligner_decoding import (
    ChunkedAlignerDecoding,
    ChunkedAlignerNarDecoding,
)
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecoding, RNNTDecodingConfig
from nemo.collections.asr.parts.utils.asr_batching import get_semi_sorted_batch_sampler
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing.parsers import make_parser
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import AcousticEncodedRepresentation, AudioSignal, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging


class _RemapInputEmbedding(torch.nn.Module):
    """Embedding wrapper that remaps incoming label ids before lookup.

    Used by the multi-target model when the shared prediction network is driven by
    a *pronunciation* label stream while the rest of the RNN-T pipeline (joint /
    decoding) still operates on the character/token vocabulary. The prediction net
    is fed character ids (during both training and greedy decoding), and this
    wrapper converts each char id to its pronunciation class id (toneless or tonal)
    before the underlying embedding lookup. The trailing slot of ``remap`` maps the
    token blank/pad id to the embedding's pad row, so padded positions are safe.
    """

    def __init__(self, base_embedding: torch.nn.Embedding, remap: torch.Tensor):
        super().__init__()
        self.base = base_embedding
        self.register_buffer('remap', remap, persistent=False)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.base(self.remap[y.long()])


class _MultiTargetConsistencyJoint(torch.nn.Module):
    """Joint wrapper that folds the multi-target consistency score into the joiner.

    The multi-target model has one shared encoder + prediction net and several joint
    heads (token + pronunciation), and every char maps deterministically to a
    toneless / tonal syllable. This wrapper presents the *token* joint interface but,
    on each joint evaluation, also evaluates the aux (pron) heads and returns, over
    the char vocabulary, the consistency score::

        s(c) = w_tok*logP_token(c) + sum_aux w_aux*logP_aux(map_aux[c])

    (the trailing blank entry sums each head's own blank log-prob). Because the
    combination lives *inside* the joiner, the standard (batched, CUDA-graphed)
    greedy decoder runs unchanged and stays fast -- no custom decode loop.

    The batched decoder pre-projects the encoder once via ``project_encoder`` and the
    prednet via ``project_prednet`` and then calls ``joint_after_projection``. Each
    head has its *own* projections, so here ``project_*`` are identity and every head
    re-projects internally inside ``joint_after_projection`` / ``joint``. Other
    attributes (vocab size, chunking, ...) delegate to the token head via __getattr__.
    """

    def __init__(self, token_joint, aux_joints, aux_maps, weights):
        super().__init__()
        # index 0 == token head; the rest are the aux (pron) heads, same order as aux_maps.
        self.joints = torch.nn.ModuleList([token_joint, *aux_joints])
        self._num_aux = len(aux_joints)
        self.weights = [float(w) for w in weights]
        for i, m in enumerate(aux_maps):
            # char_id -> pron_class_id for aux head i (non-persistent: this is an
            # eval-only wrapper, never saved).
            self.register_buffer(f'_map_{i}', m, persistent=False)

    def __getattr__(self, name):
        # nn.Module.__getattr__ handles params/buffers/submodules + normal attrs;
        # anything else (num_classes_with_blank, num_extra_outputs, vocabulary,
        # encoder_hidden, chunk_size, chunk_encoder_for_decoding, ...) delegates to
        # the token head so the decoding pipeline sees a normal token joint.
        try:
            return super().__getattr__(name)
        except AttributeError:
            joints = self._modules.get('joints', None)
            if joints is not None and len(joints) > 0:
                return getattr(joints[0], name)
            raise

    # Projections are identity; each head re-projects internally (see class doc).
    def project_encoder(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return encoder_output

    def project_prednet(self, prednet_output: torch.Tensor) -> torch.Tensor:
        return prednet_output

    def _combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Combine per-head raw logits (..., V_head+1) into char-vocab log-probs."""
        tok = logits_list[0].float().log_softmax(dim=-1)
        v_char = tok.shape[-1] - 1
        combined = self.weights[0] * tok
        for i in range(self._num_aux):
            aux = logits_list[i + 1].float().log_softmax(dim=-1)
            char2pron = getattr(self, f'_map_{i}')  # [V_char] -> pron id
            w = self.weights[i + 1]
            combined[..., :v_char] = combined[..., :v_char] + w * aux[..., char2pron]
            combined[..., v_char] = combined[..., v_char] + w * aux[..., -1]
        return combined

    def joint_after_projection(self, f, g, f_len=None):
        # f, g are *unprojected* (project_* are identity); re-project per head.
        logits_list = [
            j.joint_after_projection(j.project_encoder(f), j.project_prednet(g), f_len) for j in self.joints
        ]
        # Combine in float32 (log_softmax stability) then cast back to the encoder
        # dtype: the batched decoder allocates its score buffers from f.dtype, and a
        # normal joint returns logits in that dtype -- mismatching it (e.g. bf16 buf
        # vs float32 scores) trips an out-dtype error in BatchedHyps.
        return self._combine(logits_list).to(f.dtype)

    def joint(self, f, g, f_len=None):
        # Full path (used by the non-batched greedy decoder); each head handles its
        # own projection / chunking inside .joint().
        return self._combine([j.joint(f, g, f_len) for j in self.joints]).to(f.dtype)


class EncDecRNNTModel(ASRModel, ASRModuleMixin, ExportableEncDecModel, ASRTranscriptionMixin):
    """Base class for encoder decoder RNNT-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.loss_type = str(self.cfg.get('loss_type', 'rnnt')).lower()
        if self.loss_type not in ('rnnt', 'aligner', 'chunked_aligner'):
            raise ValueError(
                f"model.loss_type must be one of ['rnnt', 'aligner', 'chunked_aligner'], got '{self.loss_type}'."
            )

        # Initialize components
        self.preprocessor = EncDecRNNTModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = EncDecRNNTModel.from_config_dict(self.cfg.encoder)

        # Optional chunked frame reduction at the end of the encoder (only used
        # by the Chunked-Aligner; set up in the chunked-aligner setup methods).
        self.token_extractor = None
        self.first_k_frames_per_chunk = -1
        # Optional reshape + channel-axis attention token mixer (Chunked-Aligner only,
        # chunk_channel_attn=true): emits M tokens/chunk of dim chunk_size*d_model/M.
        self.channel_token_mixer = None
        # Optional learned per-chunk query embedding (added at the last conformer
        # layer; only used by the Chunked-Aligner first-k method when chunk_query=true).
        self.chunk_query_emb = None
        # Non-autoregressive (NAR) Chunked-Aligner flag (no prediction net / joint).
        self.nar = False

        if self.loss_type == 'aligner':
            self._setup_aligner_model_components()
        elif self.loss_type == 'chunked_aligner' and self._is_chunked_aligner_nar():
            # Non-autoregressive Chunked-Aligner: NO prediction net and NO joint.
            # A single per-frame projection head replaces the joint (computed after
            # token extraction), which removes the U-axis from the activations and
            # is a large training-memory win. Built with no unused modules.
            self._setup_chunked_aligner_nar_components()
        else:
            # Standard RNN-T decoder + joint (shared by 'rnnt' and AR 'chunked_aligner').
            # Update config values required by components dynamically
            with open_dict(self.cfg.decoder):
                self.cfg.decoder.vocab_size = len(self.cfg.labels)

            with open_dict(self.cfg.joint):
                self.cfg.joint.num_classes = len(self.cfg.labels)
                self.cfg.joint.vocabulary = self.cfg.labels
                # When the chunk channel-attention mixer is on, the encoder output that
                # feeds the joint has dim chunk_size*d_model/M, not d_model.
                self.cfg.joint.jointnet.encoder_hidden = self._chunked_aligner_enc_out_hidden()
                self.cfg.joint.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden

            self.decoder = EncDecRNNTModel.from_config_dict(self.cfg.decoder)
            self.joint = EncDecRNNTModel.from_config_dict(self.cfg.joint)

            # CHAT: infer chunk_size for the attention joint (RNNTAttJoint) when not set explicitly.
            if hasattr(self.joint, 'chunk_size') and self.joint.chunk_size <= 0:
                self.joint.chunk_size = self._infer_chat_chunk_size()

            if self.loss_type == 'chunked_aligner':
                self._setup_chunked_aligner_loss_and_decoding()
            else:
                # Setup RNNT Loss
                loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

                num_classes = self.joint.num_classes_with_blank - 1  # for standard RNNT and multi-blank

                if loss_name == 'tdt':
                    num_classes = num_classes - self.joint.num_extra_outputs

                self.loss = RNNTLoss(
                    num_classes=num_classes,
                    loss_name=loss_name,
                    loss_kwargs=loss_kwargs,
                    reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
                )

        if hasattr(self.cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncDecRNNTModel.from_config_dict(self.cfg.spec_augment)
        else:
            self.spec_augmentation = None

        # Optional 2-D (time x pitch) feature-domain warp (SpectrogramTimePitchWarp).
        # Unlike spec_augment, this changes the sequence length, so forward() must
        # propagate the updated length to the encoder.
        if hasattr(self.cfg, 'spec_warp') and self._cfg.spec_warp is not None:
            self.spec_warp = EncDecRNNTModel.from_config_dict(self.cfg.spec_warp)
        else:
            self.spec_warp = None

        # 'aligner' and 'chunked_aligner' use their own decoding objects (set up
        # above) plus a manual edit-distance WER; only standard RNN-T uses the
        # RNNTDecoding + WER metric objects here.
        if self.loss_type == 'rnnt':
            self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
            # Setup decoding objects
            self.decoding = RNNTDecoding(
                decoding_cfg=self.cfg.decoding,
                decoder=self.decoder,
                joint=self.joint,
                vocabulary=self.joint.vocabulary,
            )
            # Setup WER calculation
            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=0,
                use_cer=self._cfg.get('use_cer', False),
                log_prediction=self._cfg.get('log_prediction', True),
                dist_sync_on_step=True,
            )

        # Whether to compute loss during evaluation
        if 'compute_eval_loss' in self.cfg:
            self.compute_eval_loss = self.cfg.compute_eval_loss
        else:
            self.compute_eval_loss = True

        # Setup fused Joint step if flag is set
        if self.loss_type == 'rnnt' and (self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        )):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup optimization normalization (if provided in config)
        self.setup_optim_normalization()

        # Setup optional Optimization flags
        self.setup_optimization_flags()

        # Setup encoder adapters (from ASRAdapterModelMixin)
        self.setup_adapters()

        # Optional multi-target heads (token + pronunciation). Built last so the
        # standard RNN-T token head (decoder/joint/loss/decoding/wer) is fully set
        # up first and remains the primary inference/eval path.
        self._setup_multi_target()

    def _setup_multi_target(self):
        """Build auxiliary pronunciation joint heads that share the encoder + decoder.

        Enabled via ``model.multi_target.enabled=true``. There is a SINGLE shared
        prediction network (``self.decoder``) and one ``(joint + loss)`` per target:
        the token head (``self.joint``/``self.loss``) plus auxiliary heads predicting
        a *pronunciation* relabeling of the char target -- ``notone`` (toneless
        syllable) and, if ``enable_tone`` (default True), ``tone`` (tonal syllable).

        The shared predictor's input stream is configurable via
        ``model.multi_target.predictor_input`` (``token`` | ``notone`` | ``tone``):
        the predictor conditions on that label stream while the joint heads (and the
        decoded output vocabulary) are unchanged.

        Because Mandarin is monosyllabic the pron target is an element-wise relabel
        of the char target (same length), so no dataloader change is needed, and the
        shared char-driven decoder gives every head an identical U-axis -- keeping
        the per-head RNN-T alignments in sync.

        Per training step a single head is sampled and trained; the inactive joint
        heads are kept in the autograd graph via a zero-weighted parameter term (see
        ``_multi_target_training_step``) to avoid DDP unused-parameter errors. The
        token head remains the only inference/eval path.
        """
        mt_cfg = self.cfg.get('multi_target', None)
        self.multi_target_enabled = bool(mt_cfg is not None and mt_cfg.get('enabled', False))
        if not self.multi_target_enabled:
            return
        if self.loss_type != 'rnnt':
            raise ValueError("model.multi_target is only supported with loss_type='rnnt'.")

        data = self._load_or_inline_pron_map(mt_cfg)
        if list(data['labels']) != list(self.cfg.labels):
            raise ValueError(
                "multi_target pron map 'labels' do not match model.labels "
                f"({len(data['labels'])} vs {len(self.cfg.labels)}); regenerate pron_map with build_pron_map.py."
            )

        enable_tone = bool(mt_cfg.get('enable_tone', True))
        head_specs = [('notone', list(data['notone_vocab']), list(data['labels_to_notone']))]
        if enable_tone:
            head_specs.append(('tone', list(data['tone_vocab']), list(data['labels_to_tone'])))

        # char_id -> pron_class_id lookups. Registered for BOTH notone and tone
        # regardless of which aux heads are trained, because they are also used to
        # report multi-representation error rates (char / notone / tone) from a
        # single decode (see ``_mt_repr_error_counts``). Derived from the (inlined)
        # cfg, so non-persistent -- rebuilt at every construction/restore.
        self.register_buffer(
            '_mt_map_notone', torch.tensor(list(data['labels_to_notone']), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            '_mt_map_tone', torch.tensor(list(data['labels_to_tone']), dtype=torch.long), persistent=False
        )
        # tone_class_id -> notone_class_id, so the tonal head's prediction can be
        # projected into the toneless space for cross-head agreement (every tonal
        # syllable comes from some char, whose toneless form is known).
        tone2notone = [0] * len(data['tone_vocab'])
        for tone_id, notone_id in zip(data['labels_to_tone'], data['labels_to_notone']):
            tone2notone[int(tone_id)] = int(notone_id)
        self.register_buffer('_mt_tone_to_notone', torch.tensor(tone2notone, dtype=torch.long), persistent=False)

        self.aux_head_names: List[str] = []
        self.aux_joints = torch.nn.ModuleList()
        self.aux_losses = torch.nn.ModuleList()
        rnnt_reduction = self.cfg.get('rnnt_reduction', 'mean_batch')

        # The shared prediction net (self.decoder) is driven by the char/token
        # labels; its output (pred_hidden) feeds every joint. So each aux head only
        # needs its own joint (with its pron vocab) + loss -- no extra decoder.
        for name, vocab, _mapping in head_specs:
            joint_cfg = copy.deepcopy(self.cfg.joint)
            with open_dict(joint_cfg):
                joint_cfg.num_classes = len(vocab)
                joint_cfg.vocabulary = list(vocab)
                joint_cfg.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
                joint_cfg.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden
                # Aux heads always train with a non-fused joint+loss.
                joint_cfg.fuse_loss_wer = False
                joint_cfg.fused_batch_size = None

            joint = EncDecRNNTModel.from_config_dict(joint_cfg)
            # CHAT: if this is an attention joint (RNNTAttJoint) and chunk_size was
            # not set explicitly, infer it the same way the token head does.
            if hasattr(joint, 'chunk_size') and joint.chunk_size <= 0:
                joint.chunk_size = self._infer_chat_chunk_size()
            loss = RNNTLoss(
                num_classes=joint.num_classes_with_blank - 1,
                loss_name='default',
                reduction=rnnt_reduction,
            )

            self.aux_head_names.append(name)
            self.aux_joints.append(joint)
            self.aux_losses.append(loss)
            # The char->pron map for this head was already registered above as
            # ``_mt_map_{notone,tone}`` (shared with the multi-representation
            # reporting), so no per-head buffer is needed here.

        n_heads = 1 + len(self.aux_head_names)  # token + aux heads
        weights = mt_cfg.get('sample_weights', None)
        weights = [1.0] * n_heads if weights is None else [float(w) for w in weights]
        if len(weights) != n_heads:
            raise ValueError(
                f"model.multi_target.sample_weights must have {n_heads} entries "
                f"(token + {len(self.aux_head_names)} aux heads), got {len(weights)}."
            )
        self.register_buffer('_mt_sample_weights', torch.tensor(weights, dtype=torch.float), persistent=False)

        # Configurable input to the shared prediction network. By default it is fed
        # the char/token labels ('token'); it can instead be driven by the toneless
        # ('notone') or tonal ('tone') pronunciation stream. The joint heads (and
        # hence the model output vocab) are unchanged -- only the autoregressive
        # context the predictor conditions on changes.
        predictor_input = str(mt_cfg.get('predictor_input', 'token')).lower()
        if predictor_input not in ('token', 'notone', 'tone'):
            raise ValueError(
                f"model.multi_target.predictor_input must be 'token', 'notone' or 'tone', got '{predictor_input}'."
            )
        self.predictor_input = predictor_input
        if predictor_input == 'notone':
            self._rebuild_shared_predictor(list(data['notone_vocab']), list(data['labels_to_notone']))
        elif predictor_input == 'tone':
            self._rebuild_shared_predictor(list(data['tone_vocab']), list(data['labels_to_tone']))

        logging.info(
            f"[multi_target] enabled with predictor_input='{predictor_input}'; heads: "
            f"['token'(+{len(self.cfg.labels)}) "
            + ", ".join(f"'{n}'(+{j.num_classes_with_blank - 1})" for n, j in zip(self.aux_head_names, self.aux_joints))
            + f"]; sample_weights={weights}"
        )

    def _rebuild_shared_predictor(self, pred_vocab: List[str], char2pred: List[int]):
        """Rebuild ``self.decoder`` so it embeds a pronunciation label stream.

        The new prediction network's embedding spans the pronunciation vocab
        (``pred_vocab``), but it is wrapped so its public interface still accepts
        character ids: each char id is remapped to its pron class id (``char2pred``)
        before the embedding lookup. This keeps the training step and the greedy
        decoding loop (which feed char ids) completely unchanged, while the
        autoregressive context the predictor sees is the pronunciation stream.

        Decoding/WER are rebuilt afterwards so the greedy strategy references the new
        predictor. The token joint head remains the inference/eval output path.
        """
        token_vocab_size = len(self.cfg.labels)
        pred_vocab_size = len(pred_vocab)

        decoder_cfg = copy.deepcopy(self.cfg.decoder)
        with open_dict(decoder_cfg):
            decoder_cfg.vocab_size = pred_vocab_size
        new_decoder = EncDecRNNTModel.from_config_dict(decoder_cfg)

        # char id -> pron class id; the trailing slot maps the token blank/pad id to
        # the embedding's pad row (index == pred_vocab_size).
        remap = torch.full((token_vocab_size + 1,), pred_vocab_size, dtype=torch.long)
        remap[:token_vocab_size] = torch.tensor(char2pred, dtype=torch.long)
        new_decoder.prediction["embed"] = _RemapInputEmbedding(new_decoder.prediction["embed"], remap)
        # Keep the predictor's blank id aligned with the token vocab so any external
        # blank comparison / feed matches the rest of the (char-vocab) pipeline; the
        # remap routes that id to the embedding pad row.
        new_decoder.blank_idx = token_vocab_size

        self.decoder = new_decoder

        # Rebuild decoding + WER so the (greedy/beam) strategy points at the new
        # predictor. The token joint head stays the output vocabulary.
        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=self.joint,
            vocabulary=self.joint.vocabulary,
        )
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

    def _load_or_inline_pron_map(self, mt_cfg) -> dict:
        """Return the pron-map dict, preferring the copy inlined in cfg.

        On first construction the map is read from ``pron_map_path`` and inlined
        into ``self.cfg.multi_target.pron_map`` so the saved ``.nemo`` is
        self-contained (no pypinyin / external file needed at restore time).
        """
        if mt_cfg.get('pron_map', None) is not None:
            return OmegaConf.to_container(mt_cfg.pron_map, resolve=True)
        path = mt_cfg.get('pron_map_path', None)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"model.multi_target.pron_map_path not found: {path!r}. Generate it with "
                "rno/build_pron_map.py and mount it into the container."
            )
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open_dict(self.cfg):
            self.cfg.multi_target.pron_map = data
        return data

    def _mt_sample_head_index(self) -> int:
        """Pick which head to train this step.

        Seeded by ``global_step`` so every DDP rank (and every grad-accumulation
        microbatch within a step) selects the *same* head without communication --
        otherwise different ranks would grad different params and desync.
        """
        step = int(self.trainer.global_step) if self.trainer is not None else 0
        g = torch.Generator()
        g.manual_seed((0x9E3779B9 ^ step) & 0x7FFFFFFF)
        idx = torch.multinomial(self._mt_sample_weights.detach().cpu(), 1, generator=g).item()
        return int(idx)

    def _mt_repr_error_counts(self, encoded, encoded_len, transcript, transcript_len):
        """Edit-distance error counts for char / notone / tone, from ONE decode.

        The token head is decoded once (greedy) into char id sequences; both the
        hypotheses and the char references are then projected into each
        representation via the precomputed char->pron maps and scored with edit
        distance. So no representation requires a separate decode: a homophone
        substitution counts as a char error but not a (toneless/tonal) pinyin error.

        Returns ``{rep: [edits, ref_len]}`` for rep in ('char', 'notone', 'tone').
        """
        notone_map = self._mt_map_notone.detach().cpu()
        tone_map = self._mt_map_tone.detach().cpu()

        def to_seq(ids: List[int], rep: str) -> List[int]:
            if rep == 'char':
                return list(ids)
            m = notone_map if rep == 'notone' else tone_map
            return m[torch.as_tensor(ids, dtype=torch.long)].tolist() if len(ids) else []

        hyps = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded.detach(), encoded_lengths=encoded_len, return_hypotheses=True
        )
        transcript = transcript.long().cpu()
        transcript_len = transcript_len.long().cpu()

        reps = ('char', 'notone', 'tone')
        counts = {r: [0, 0] for r in reps}
        for b, hyp in enumerate(hyps):
            ref_ids = transcript[b, : int(transcript_len[b])].tolist()
            hyp_ids = hyp.y_sequence
            if torch.is_tensor(hyp_ids):
                hyp_ids = hyp_ids.tolist()
            for r in reps:
                ref_seq = to_seq(ref_ids, r)
                hyp_seq = to_seq(hyp_ids, r)
                counts[r][0] += editdistance.eval(hyp_seq, ref_seq)
                counts[r][1] += len(ref_seq)
        return counts

    @torch.no_grad()
    def _mt_head_agreement_counts(self, encoded, encoded_len, decoder_out, target_len):
        """Top-1 cross-head agreement *counts* on a (teacher-forced) lattice.

        Every head shares the same encoder + prediction-net context, so for each
        lattice cell ``(b, t, u)`` we take each head's argmax and project it into a
        common space via the char->pron maps (token head) and the tone->notone map
        (tonal head). For each head pair we count, over cells where *both* heads emit
        a non-blank symbol, how many have matching projected predictions (so a
        homophone counts as agreement in the toneless space). This is a pure
        diagnostic: it runs under ``no_grad`` and never touches the loss.

        Returns ``{pair: (agree, denom)}`` (0-dim long tensors) for whichever
        pairings the active heads support, so callers can either turn them into a
        per-batch rate (training) or accumulate them across a dataset (inference).
        """
        # name -> joint. Token head is always present; aux heads depend on cfg.
        # When consistency decoding is active ``self.joint`` is the combining
        # wrapper, so reach through to the underlying token head for its raw argmax.
        token_joint = self.joint
        if isinstance(token_joint, _MultiTargetConsistencyJoint):
            token_joint = token_joint.joints[0]
        joints = {'token': token_joint}
        for name, j in zip(self.aux_head_names, self.aux_joints):
            joints[name] = j

        preds: Dict[str, torch.Tensor] = {}
        blanks: Dict[str, int] = {}
        valid = None  # [B, T, U] bool mask of in-bounds cells
        for name, j in joints.items():
            out = j(encoder_outputs=encoded, decoder_outputs=decoder_out, encoder_lengths=encoded_len)
            preds[name] = out.argmax(dim=-1)  # [B, T, U]
            blanks[name] = out.shape[-1] - 1
            if valid is None:
                B, T, U = out.shape[0], out.shape[1], out.shape[2]
                # CHAT (RNNTAttJoint) time-axis is #chunks; plain joints use frames.
                t_len = getattr(j, 'num_chunks_per_utterance', None)
                if t_len is None:
                    t_len = encoded_len
                t_len = t_len.to(encoded.device).view(B, 1, 1)
                u_len = (target_len.to(encoded.device) + 1).view(B, 1, 1)  # +1 for the SOS/blank row
                t_idx = torch.arange(T, device=encoded.device).view(1, T, 1)
                u_idx = torch.arange(U, device=encoded.device).view(1, 1, U)
                valid = (t_idx < t_len) & (u_idx < u_len)
            del out

        # Projections into common spaces (blanks handled by the non-blank masks).
        def proj(name: str, mapping: Optional[torch.Tensor]) -> torch.Tensor:
            p = preds[name]
            if mapping is None:
                return p
            return mapping.to(p.device)[p.clamp(max=mapping.numel() - 1)]

        def pair_counts(a_name, a_map, b_name, b_map) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            if a_name not in preds or b_name not in preds:
                return None
            both = valid & (preds[a_name] != blanks[a_name]) & (preds[b_name] != blanks[b_name])
            agree = (proj(a_name, a_map) == proj(b_name, b_map)) & both
            return agree.sum(), both.sum()

        counts: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        tn = pair_counts('token', self._mt_map_notone, 'notone', None)
        if tn is not None:
            counts['token_notone'] = tn
        tt = pair_counts('token', self._mt_map_tone, 'tone', None)
        if tt is not None:
            counts['token_tone'] = tt
        nt = pair_counts('notone', None, 'tone', self._mt_tone_to_notone)
        if nt is not None:
            counts['notone_tone'] = nt

        # All-three agreement: cells where all heads are non-blank and the token
        # head's projected pron matches both aux heads.
        if {'token', 'notone', 'tone'} <= set(preds):
            both = (
                valid
                & (preds['token'] != blanks['token'])
                & (preds['notone'] != blanks['notone'])
                & (preds['tone'] != blanks['tone'])
            )
            agree = (
                (proj('token', self._mt_map_notone) == preds['notone'])
                & (proj('token', self._mt_map_tone) == preds['tone'])
                & both
            )
            counts['all3'] = (agree.sum(), both.sum())
        return counts

    @torch.no_grad()
    def multi_target_agreement_counts(self, encoded, encoded_len):
        """Cross-head agreement counts on the model's *own* decoded output.

        Inference-time sibling of the teacher-forced training diagnostic: the token
        head is greedily decoded, the shared prediction net is re-run on that decoded
        char context, and every head's argmax is compared on the resulting lattice
        (see :meth:`_mt_head_agreement_counts`). So this reports how consistent the
        heads are given the autoregressive context the model actually produces.

        Returns ``{pair: (agree, denom)}`` (0-dim long tensors), empty if nothing was
        emitted in the batch.
        """
        hyps = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )
        seqs = []
        for h in hyps:
            y = h.y_sequence
            if torch.is_tensor(y):
                y = y.tolist()
            seqs.append([int(t) for t in y])
        lens = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=encoded.device)
        if int(lens.max().item()) == 0:
            return {}
        max_u = int(lens.max().item())
        transcript = torch.zeros(len(seqs), max_u, dtype=torch.long, device=encoded.device)
        for i, s in enumerate(seqs):
            if s:
                transcript[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=encoded.device)
        decoder_out, target_len, _ = self.decoder(targets=transcript, target_length=lens)
        return self._mt_head_agreement_counts(encoded, encoded_len, decoder_out, target_len)

    def _multi_target_training_step(self, encoded, encoded_len, transcript, transcript_len):
        """Single-sampled-head training step for the multi-target model.

        The shared prediction net runs once on the char/token labels, so every head
        sees the same U-axis. Only the sampled head's (joint + loss) is computed;
        its targets are the char labels (token head) or their pron relabeling.
        """
        # The shared predictor is always *fed* char ids; when predictor_input is a
        # pronunciation stream its embedding remaps char -> pron internally
        # (_RemapInputEmbedding), so this call is identical for every input mode.
        decoder_out, target_len, _ = self.decoder(targets=transcript, target_length=transcript_len)

        head_idx = self._mt_sample_head_index()
        if head_idx == 0:
            head_name = 'token'
            targets = transcript
            joint, loss_fn = self.joint, self.loss
        else:
            aux = head_idx - 1
            head_name = self.aux_head_names[aux]
            # Element-wise relabel char ids -> pron class ids (same length / U-axis).
            targets = getattr(self, f'_mt_map_{head_name}')[transcript.long()]
            joint, loss_fn = self.aux_joints[aux], self.aux_losses[aux]

        joint_out = joint(encoder_outputs=encoded, decoder_outputs=decoder_out, encoder_lengths=encoded_len)
        # For CHAT (RNNTAttJoint) the joint time-axis is the number of chunks, not
        # encoder frames, so the loss input length must come from the joint's
        # per-utterance chunk count (set during its forward). Plain RNN-T/TDT joints
        # leave it None / unset, so fall back to encoded_len in that case.
        effective_len = getattr(joint, 'num_chunks_per_utterance', None)
        if effective_len is None:
            effective_len = encoded_len
        loss_value = loss_fn(
            log_probs=joint_out, targets=targets, input_lengths=effective_len, target_lengths=target_len
        )

        # Keep every parameter in the autograd graph this step. The inactive heads
        # contribute 0*sum(params) -> a real (zero) grad rather than None, which is
        # what DDP requires (no need for find_unused_parameters).
        loss_value = loss_value + 0.0 * sum(p.sum() for p in self.parameters())
        loss_value = self.add_auxiliary_losses(loss_value)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_loss': loss_value,
            f'train_loss_{head_name}': loss_value.detach(),
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        # Training error rates. Like the standard RNN-T step, compute every
        # ``log_every_n_steps`` -- otherwise nothing populates ``training_batch_wer``
        # and it never appears in TensorBoard/W&B. A SINGLE token-head decode yields
        # all three representations (char / notone / tone) via the char->pron maps,
        # so a homophone substitution shows up as a char error but not a pinyin one.
        # Always measured on the token head (the eval/output path) regardless of
        # which head was sampled, so the curves stay comparable across the
        # token/notone/tone predictor variants.
        if self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = 0
        if log_every_n_steps > 0 and (sample_id + 1) % log_every_n_steps == 0:
            counts = self._mt_repr_error_counts(encoded, encoded_len, transcript, transcript_len)
            for rep, (edits, ref_len) in counts.items():
                key = 'training_batch_wer' if rep == 'char' else f'training_batch_wer_{rep}'
                tensorboard_logs[key] = torch.tensor(
                    edits / max(ref_len, 1), dtype=torch.float32, device=encoded.device
                )

        self.log_dict(tensorboard_logs)
        return {'loss': loss_value}

    def enable_consistency_decoding(self, head_weights=None):
        """Switch decoding to the "consistency-maintaining" joint (in-place).

        Rebuilds ``self.decoding`` so the joiner is the :class:`_MultiTargetConsistencyJoint`
        wrapper: at every joint evaluation the token score for each char ``c`` is
        combined with the (toneless/tonal) pronunciation log-probs of the syllables it
        implies, ``s(c) = w_tok*logP_token(c) + sum_aux w_aux*logP_aux(map[c])``. The
        combination lives inside the joiner, so the standard fast (batched, CUDA-graph)
        greedy decoder is reused with no algorithm change -- ``transcribe`` just works.

        Output stays the char vocabulary, so CER is directly comparable to the plain
        token-head decode.

        Args:
            head_weights: per-head weights ordered ``[token, notone, tone?]``;
                defaults to all-ones over (token + active aux heads).
        """
        if not getattr(self, 'multi_target_enabled', False):
            raise RuntimeError("consistency decoding requires a multi_target model.")
        if self.loss_type != 'rnnt':
            raise RuntimeError("consistency decoding is only supported for loss_type='rnnt'.")

        n_heads = 1 + len(self.aux_head_names)
        weights = [1.0] * n_heads if head_weights is None else [float(w) for w in head_weights]
        if len(weights) != n_heads:
            raise ValueError(
                f"head_weights must have {n_heads} entries (token + {len(self.aux_head_names)} aux heads), "
                f"got {len(weights)}."
            )

        aux_maps = [getattr(self, f'_mt_map_{name}') for name in self.aux_head_names]
        cons_joint = _MultiTargetConsistencyJoint(self.joint, list(self.aux_joints), aux_maps, weights)

        # Rebuild decoding + WER around the consistency joint. The token vocabulary
        # (and hence blank id) is unchanged, so greedy/beam decoding is identical
        # except that each step's scores already encode cross-head agreement.
        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        self.decoding = RNNTDecoding(
            decoding_cfg=self.cfg.decoding,
            decoder=self.decoder,
            joint=cons_joint,
            vocabulary=self.joint.vocabulary,
        )
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )
        logging.info(
            f"[multi_target] consistency decoding enabled; heads=['token', "
            + ", ".join(f"'{n}'" for n in self.aux_head_names)
            + f"]; weights={weights}"
        )
        return cons_joint

    def _setup_aligner_model_components(self):
        """Set up Aligner-Encoder components inside the RNNT model lifecycle."""
        vocabulary = list(self.cfg.labels)
        self.eos_id = len(vocabulary)
        num_classes = self.eos_id + 1  # real tokens + EOS, no blank

        self.aligner_type = self.cfg.get('aligner_type', 'ar')
        if self.aligner_type not in ('ar', 'nonar'):
            raise ValueError(f"model.aligner_type must be 'ar' or 'nonar', got '{self.aligner_type}'.")

        with open_dict(self.cfg.decoder):
            self.cfg.decoder.vocab_size = num_classes

        with open_dict(self.cfg.joint):
            self.cfg.joint.num_classes = num_classes
            self.cfg.joint.vocabulary = ListConfig(vocabulary)
            self.cfg.joint.jointnet.encoder_hidden = self.cfg.model_defaults.enc_hidden
            self.cfg.joint.jointnet.pred_hidden = self.cfg.model_defaults.pred_hidden

        self.decoder = EncDecRNNTModel.from_config_dict(self.cfg.decoder)
        self.joint = AlignerJoint(
            jointnet=self.cfg.joint.jointnet,
            num_classes=num_classes,
            vocabulary=vocabulary,
            log_softmax=self.cfg.joint.get('log_softmax', None),
        )

        self.aux_nonar_loss_weight = float(self.cfg.get('aux_nonar_loss_weight', 0.0))
        self.ctc_head = None
        if self.aligner_type == 'nonar' or self.aux_nonar_loss_weight > 0:
            head_cfg = self.cfg.get('ctc_head', {}) or {}
            self.ctc_head = AlignerCTCHead(
                feat_in=self.cfg.model_defaults.enc_hidden,
                num_classes=num_classes,
                hidden=head_cfg.get('hidden', None),
                activation=head_cfg.get('activation', 'relu'),
                dropout=head_cfg.get('dropout', 0.0),
            )

        self.loss = AlignerCrossEntropyLoss(
            num_classes=num_classes,
            label_smoothing=self.cfg.get('label_smoothing', 0.1),
        )

        self.decoding = AlignerDecoding(
            decoding_cfg=self.cfg.get('decoding', None),
            decoder=self.decoder,
            joint=self.joint,
            eos_id=self.eos_id,
            vocabulary=vocabulary,
            tokenizer=getattr(self, 'tokenizer', None),
            ctc_head=self.ctc_head,
        )

    def _append_aligner_eos(
        self, transcript: torch.Tensor, transcript_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = torch.nn.functional.pad(transcript, (0, 1), value=0)
        batch_idx = torch.arange(transcript.size(0), device=transcript.device)
        targets[batch_idx, transcript_len.long()] = self.eos_id
        return targets, transcript_len + 1

    def _aligner_loss(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> torch.Tensor:
        targets_eos, target_eos_len = self._append_aligner_eos(transcript, transcript_len)

        # The one-to-one joint needs at least one encoder frame per target token,
        # plus one for the appended EOS (i.e. T >= U + 1). In the very rare case
        # that an utterance has too few frames (T < U + 1), there are not enough
        # acoustic frames to supervise every token, so we drop it from the loss
        # (mask its target length to 0 so it contributes nothing) and warn.
        insufficient_frames = encoded_len < target_eos_len.to(encoded_len.device)
        if torch.any(insufficient_frames):
            bad_idx = torch.nonzero(insufficient_frames, as_tuple=False).flatten().tolist()
            details = ", ".join(
                f"(batch_idx={i}, T={int(encoded_len[i])}, U+1={int(target_eos_len[i])})" for i in bad_idx
            )
            logging.warning(
                f"[aligner] Skipping {len(bad_idx)}/{int(encoded_len.size(0))} utterance(s) with too few "
                f"encoder frames for the one-to-one joint (T < U+1): {details}. If this is not rare, lower "
                f"model.train_ds.max_duration or the encoder subsampling_factor."
            )
            target_eos_len = target_eos_len.clone()
            target_eos_len[insufficient_frames] = 0

        total_loss = encoded.new_zeros(())

        if self.aligner_type == 'ar' or self.aux_nonar_loss_weight > 0:
            decoder_outputs, _, _ = self.decoder(targets=transcript, target_length=transcript_len)
            ar_logits = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_outputs)
            ar_loss = self.loss(log_probs=ar_logits, targets=targets_eos, target_lengths=target_eos_len)
            if self.aligner_type == 'ar':
                total_loss = total_loss + ar_loss

        if self.ctc_head is not None:
            nonar_logits = self.ctc_head(encoder_output=encoded)
            nonar_loss = self.loss(log_probs=nonar_logits, targets=targets_eos, target_lengths=target_eos_len)
            if self.aligner_type == 'nonar':
                total_loss = total_loss + nonar_loss
            elif self.aux_nonar_loss_weight > 0:
                total_loss = total_loss + self.aux_nonar_loss_weight * nonar_loss

        return total_loss

    def _is_chunked_aligner_nar(self) -> bool:
        """Whether the Chunked-Aligner is configured in non-autoregressive mode."""
        ca_cfg = self.cfg.get('chunked_aligner', None)
        return bool(ca_cfg.get('nar', False)) if ca_cfg is not None else False

    def _chunked_aligner_enc_out_hidden(self) -> int:
        """Encoder-output hidden dim that feeds the joint / NAR head / lattice.

        Normally this is ``model_defaults.enc_hidden`` (= encoder ``d_model``). With
        the chunk channel-attention mixer (``chunk_channel_attn=true``) the per-chunk
        reshape changes the per-token feature dim to ``chunk_size * d_model / M``
        (``M = first_k_frames_per_chunk``), so the joint must be built for that dim.
        """
        base = int(self.cfg.model_defaults.enc_hidden)
        if self.loss_type != 'chunked_aligner':
            return base
        ca_cfg = self.cfg.get('chunked_aligner', None)
        if ca_cfg is None or not bool(ca_cfg.get('chunk_channel_attn', False)):
            return base
        C = int(ca_cfg.get('chunk_size', 12))
        M = int(ca_cfg.get('first_k_frames_per_chunk', -1))
        if M < 1:
            raise ValueError(
                "model.chunked_aligner.chunk_channel_attn=true requires "
                "model.chunked_aligner.first_k_frames_per_chunk >= 1 (= tokens/chunk M)."
            )
        if (C * base) % M != 0:
            raise ValueError(
                f"model.chunked_aligner.chunk_channel_attn: chunk_size*d_model ({C * base}) must be "
                f"divisible by first_k_frames_per_chunk ({M})."
            )
        return (C * base) // M

    def _setup_chunked_token_extractor(self, ca_cfg):
        """Set chunk geometry and optional chunk-level frame reduction.

        ``token_extraction_size`` (-1 disables) compresses each acoustic chunk of
        ``chunk_size`` encoder frames into that many learned tokens via
        cross-attention. ``first_k_frames_per_chunk`` (-1 disables) is a cheaper
        alternative that simply keeps the left-most ``k`` encoder frames from each
        chunk and drops the rest. In either case the downstream lattice runs over
        the reduced sequence, so the effective ``chunk_size`` becomes the reduced
        number of frames/tokens per chunk.
        """
        # Acoustic chunk: number of encoder frames covered by one chunk.
        self.frames_per_chunk = int(ca_cfg.get('chunk_size', 12)) if ca_cfg is not None else 12
        if self.frames_per_chunk < 1:
            raise ValueError(f"model.chunked_aligner.chunk_size must be >= 1, got {self.frames_per_chunk}.")

        token_extraction_size = int(ca_cfg.get('token_extraction_size', -1)) if ca_cfg is not None else -1
        first_k_frames_per_chunk = (
            int(ca_cfg.get('first_k_frames_per_chunk', -1)) if ca_cfg is not None else -1
        )
        chunk_query = bool(ca_cfg.get('chunk_query', False)) if ca_cfg is not None else False
        chunk_channel_attn = bool(ca_cfg.get('chunk_channel_attn', False)) if ca_cfg is not None else False
        if token_extraction_size >= 1 and first_k_frames_per_chunk >= 1:
            raise ValueError(
                "model.chunked_aligner.token_extraction_size and "
                "model.chunked_aligner.first_k_frames_per_chunk are mutually exclusive."
            )
        if chunk_query and first_k_frames_per_chunk < 1:
            raise ValueError(
                "model.chunked_aligner.chunk_query=true requires "
                "model.chunked_aligner.first_k_frames_per_chunk >= 1 (the chunk-query "
                "embedding augments the first-k frame-selection method)."
            )
        if chunk_channel_attn:
            if first_k_frames_per_chunk < 1:
                raise ValueError(
                    "model.chunked_aligner.chunk_channel_attn=true requires "
                    "model.chunked_aligner.first_k_frames_per_chunk >= 1 (= tokens/chunk M)."
                )
            if token_extraction_size >= 1:
                raise ValueError(
                    "model.chunked_aligner.chunk_channel_attn and token_extraction_size are mutually exclusive."
                )
            if chunk_query:
                raise ValueError(
                    "model.chunked_aligner.chunk_channel_attn and chunk_query are mutually exclusive."
                )
        if token_extraction_size >= 1:
            self.token_extractor = ChunkTokenExtractor(
                d_model=int(self.cfg.model_defaults.enc_hidden),
                frames_per_chunk=self.frames_per_chunk,
                tokens_per_chunk=token_extraction_size,
            )
            # Downstream the lattice runs over extracted tokens.
            self.chunk_size = token_extraction_size
            logging.info(
                f"[chunked-aligner] Token-extraction enabled: {self.frames_per_chunk} encoder frames/chunk -> "
                f"{token_extraction_size} tokens/chunk (effective chunk_size={self.chunk_size})."
            )
        elif chunk_channel_attn:
            # Reshape each [chunk_size, d_model] chunk into M = first_k tokens of dim
            # chunk_size*d_model/M and self-attend over the M (token) axis. The M rows
            # become the per-chunk tokens, so the lattice runs with chunk_size = M and
            # the encoder-out hidden dim seen by the joint becomes chunk_size*d_model/M.
            m_tokens = first_k_frames_per_chunk
            self.channel_token_mixer = ChunkChannelTokenMixer(
                d_model=int(self.cfg.model_defaults.enc_hidden),
                frames_per_chunk=self.frames_per_chunk,
                tokens_per_chunk=m_tokens,
                num_heads=int(ca_cfg.get('chunk_channel_attn_heads', 4)) if ca_cfg is not None else 4,
            )
            self.chunk_size = m_tokens
            logging.info(
                f"[chunked-aligner] Channel-attention token mixer enabled: {self.frames_per_chunk} encoder "
                f"frames/chunk -> {m_tokens} tokens/chunk of dim {self.channel_token_mixer.out_dim} "
                f"(num_heads={self.channel_token_mixer.num_heads}, effective chunk_size={self.chunk_size})."
            )
        elif first_k_frames_per_chunk >= 1:
            if first_k_frames_per_chunk > self.frames_per_chunk:
                raise ValueError(
                    "model.chunked_aligner.first_k_frames_per_chunk must be <= "
                    f"model.chunked_aligner.chunk_size ({self.frames_per_chunk}), got {first_k_frames_per_chunk}."
                )
            self.first_k_frames_per_chunk = first_k_frames_per_chunk
            self.chunk_size = first_k_frames_per_chunk
            logging.info(
                f"[chunked-aligner] First-k frame selection enabled: keeping first {first_k_frames_per_chunk}/"
                f"{self.frames_per_chunk} encoder frames per chunk (effective chunk_size={self.chunk_size})."
            )
            if chunk_query:
                self._setup_chunk_query_emb()
        else:
            self.chunk_size = self.frames_per_chunk

    def _setup_chunk_query_emb(self):
        """Learned per-chunk query embedding injected at the LAST conformer layer.

        A trainable ``[frames_per_chunk, d_model]`` tensor (one row per position in the
        full acoustic chunk) is tiled over the time axis and added to the INPUT of the
        final conformer layer -- i.e. on top of the second-to-last layer's output,
        before that layer runs its self-attention. The downstream first-k selection
        then keeps only the first ``k`` frames of each chunk for the logits/loss, so
        these learned per-position queries give the kept frames a chance to summarize
        the whole chunk through the last self-attention. Zero-initialized, so at the
        start of training it is identical to plain first-k (warm-start friendly).

        Implemented as a forward pre-hook on ``encoder.layers[-1]`` (rather than via the
        encoder config) so it survives the warm-start encoder injection untouched and
        does not require any ConformerEncoder changes. NOTE: this targets the full,
        non-cached training/validation forward; cache-aware streaming export would need
        the chunk offset threaded through and is out of scope here.
        """
        d_model = int(self.encoder.d_model)
        self.chunk_query_emb = torch.nn.Parameter(torch.zeros(int(self.frames_per_chunk), d_model))
        self.encoder.layers[-1].register_forward_pre_hook(self._chunk_query_pre_hook, with_kwargs=True)
        logging.info(
            f"[chunked-aligner] Chunk-query embedding enabled: trainable "
            f"[{self.frames_per_chunk}, {d_model}] added per chunk at the input of the last "
            f"conformer layer (before first-k truncation to {self.first_k_frames_per_chunk})."
        )

    def _add_chunk_query(self, x: torch.Tensor) -> torch.Tensor:
        """Tile the ``[frames_per_chunk, d_model]`` query over time and add to ``x``.

        ``x`` is the last conformer layer input ``[B, T, d_model]``. Chunks are aligned
        to multiples of ``frames_per_chunk`` from frame 0 (matching the chunked-limited
        streaming attention and the post-encoder first-k selection).
        """
        C = self.chunk_query_emb.shape[0]
        T = x.shape[1]
        n_chunks = (T + C - 1) // C
        q = self.chunk_query_emb.repeat(n_chunks, 1)[:T]  # [T, d_model]
        return x + q.unsqueeze(0).to(dtype=x.dtype)

    def _chunk_query_pre_hook(self, module, args, kwargs):
        """Add the tiled chunk-query embedding to the last conformer layer input."""
        if 'x' in kwargs:
            kwargs = dict(kwargs)
            kwargs['x'] = self._add_chunk_query(kwargs['x'])
        elif len(args) > 0:
            args = (self._add_chunk_query(args[0]),) + tuple(args[1:])
        return args, kwargs

    def _select_first_k_frames_per_chunk(
        self, encoded: torch.Tensor, encoded_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep the left-most ``k`` frames from every fixed-size encoder chunk.

        Args:
            encoded: Encoder output ``[B, D, T]``.
            encoded_len: Valid encoder lengths ``[B]``.

        Returns:
            The compacted encoder sequence and updated lengths. The frame order is
            preserved chunk-by-chunk, e.g. for ``chunk_size=12, k=4`` we keep
            offsets ``0..3, 12..15, 24..27, ...`` and remove the remaining offsets.
        """
        k = int(self.first_k_frames_per_chunk)
        if k < 1:
            return encoded, encoded_len

        C = int(self.frames_per_chunk)
        if k == C:
            return encoded, encoded_len

        B, D, T = encoded.shape
        n_chunks = (T + C - 1) // C
        padded_T = n_chunks * C
        if padded_T != T:
            encoded = torch.nn.functional.pad(encoded, (0, padded_T - T))

        encoded = encoded.view(B, D, n_chunks, C)[:, :, :, :k].contiguous().view(B, D, n_chunks * k)

        encoded_len = encoded_len.to(device=encoded.device, dtype=torch.long)
        full_chunks = encoded_len // C
        remainder = encoded_len % C
        selected_len = full_chunks * k + torch.minimum(remainder, remainder.new_full(remainder.shape, k))
        return encoded, selected_len

    def _setup_chunked_aligner_nar_components(self):
        """Set up the non-autoregressive (NAR) Chunked-Aligner.

        NAR drops the prediction net and the joint entirely. After (optional) token
        extraction, a single per-frame projection head maps each encoder frame to a
        distribution over the vocabulary (blank/EOC included): ``[B, T, D] -> [B, T, V]``.
        Because there is no dependency on previously emitted tokens, the activations
        have no U-axis, which is a large training-memory win versus the AR joint.

        The loss is intentionally left for a later step; only the architecture and
        the (cheap) greedy decoder are set up here.
        """
        self.nar = True
        ca_cfg = self.cfg.get('chunked_aligner', None)
        self._setup_chunked_token_extractor(ca_cfg)

        vocabulary = list(self.cfg.labels)
        # Output space = real tokens + a single blank / end-of-chunk (EOC) symbol.
        self.num_classes_with_blank = len(vocabulary) + 1
        self.blank_id = self.num_classes_with_blank - 1

        # The "joiner" is just a projection + softmax (softmax applied in loss /
        # decoding). Operates on the (token-extracted) encoder frames; with the channel-
        # attention mixer the per-token dim is chunk_size*d_model/M, else enc_hidden.
        enc_hidden = (
            self.channel_token_mixer.out_dim
            if self.channel_token_mixer is not None
            else int(self.cfg.model_defaults.enc_hidden)
        )
        self.nar_head = torch.nn.Linear(enc_hidden, self.num_classes_with_blank)

        # NAR full-sum loss over the per-frame head logits (no joint, no U axis):
        # the CUDA/Numba kernel (acts [B, T, V]). See ChunkedAlignerNarLossPytorch
        # for the reference / cross-checked implementation.
        reduction = str(ca_cfg.get('reduction', 'mean_volume')) if ca_cfg is not None else 'mean_volume'
        self.loss = ChunkedAlignerNarLossNumba(
            blank=self.blank_id,
            chunk_size=self.chunk_size,
            reduction=reduction,
            clamp=float(ca_cfg.get('clamp', -1.0)) if ca_cfg is not None else -1.0,
        )

        self.decoding = ChunkedAlignerNarDecoding(
            decoding_cfg=self.cfg.get('decoding', None),
            head=self.nar_head,
            blank_id=self.blank_id,
            chunk_size=self.chunk_size,
            vocabulary=vocabulary,
            tokenizer=getattr(self, 'tokenizer', None),
        )

    def _nar_logits(self, encoded: torch.Tensor) -> torch.Tensor:
        """Per-frame NAR projection head: ``[B, D, T] -> [B, T, V]`` (raw logits)."""
        return self.nar_head(encoded.transpose(1, 2))

    def _setup_chunked_aligner_loss_and_decoding(self):
        """Set up the streaming Chunked-Aligner loss + greedy decoding.

        The Chunked Aligner reuses the standard RNN-T prediction net and joint
        (built by the caller). Compared to RNN-T it only swaps in the full-sum
        chunked-aligner loss and a chunked greedy decoder; the blank symbol
        doubles as the end-of-chunk (EOC) signal.
        """
        ca_cfg = self.cfg.get('chunked_aligner', None)
        self._setup_chunked_token_extractor(ca_cfg)

        blank_id = self.joint.num_classes_with_blank - 1

        # Default to per-token normalization ('mean_volume') so the loss reads like
        # a cross-entropy (~ln V at init) and is comparable in scale to the aligner
        # / TDT recipes -- this keeps a shared learning rate transferable. Use
        # 'mean' for the (length-scaled) total-sequence NLL averaged over the batch.
        reduction = str(ca_cfg.get('reduction', 'mean_volume')) if ca_cfg is not None else 'mean_volume'

        self.loss = ChunkedAlignerLossNumba(
            blank=blank_id,
            chunk_size=self.chunk_size,
            reduction=reduction,
            clamp=float(ca_cfg.get('clamp', -1.0)) if ca_cfg is not None else -1.0,
        )

        self.decoding = ChunkedAlignerDecoding(
            decoding_cfg=self.cfg.get('decoding', None),
            decoder=self.decoder,
            joint=self.joint,
            blank_id=blank_id,
            chunk_size=self.chunk_size,
            vocabulary=self.joint.vocabulary,
            tokenizer=getattr(self, 'tokenizer', None),
        )

    def _chunked_aligner_nar_loss(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> torch.Tensor:
        # Feasibility: the chunked lattice needs at least one (extracted) frame per
        # target token (T >= U). Drop infeasible utterances (label length -> 0).
        transcript_len = transcript_len.to(encoded_len.device)
        insufficient_frames = encoded_len < transcript_len
        if torch.any(insufficient_frames):
            bad_idx = torch.nonzero(insufficient_frames, as_tuple=False).flatten().tolist()
            details = ", ".join(
                f"(batch_idx={i}, T={int(encoded_len[i])}, U={int(transcript_len[i])})" for i in bad_idx
            )
            logging.warning(
                f"[chunked-aligner-nar] Skipping {len(bad_idx)}/{int(encoded_len.size(0))} utterance(s) with too "
                f"few (extracted) frames for the chunked lattice (T < U): {details}."
            )
            transcript_len = transcript_len.clone()
            transcript_len[insufficient_frames] = 0

        max_logit_len = int(encoded_len.max().item())
        max_targets_len = int(transcript_len.max().item())
        if transcript.shape[1] != max_targets_len:
            transcript = transcript.narrow(dim=1, start=0, length=max_targets_len).contiguous()

        # Per-frame head logits [B, T, V] (no joint, no U-axis). Trim padding frames.
        logits = self._nar_logits(encoded)
        if logits.shape[1] != max_logit_len:
            logits = logits.narrow(dim=1, start=0, length=max_logit_len).contiguous()

        loss_value = self.loss(
            acts=logits,
            labels=transcript,
            act_lens=encoded_len,
            label_lens=transcript_len,
        )
        if loss_value.dim() > 0:
            loss_value = loss_value.squeeze()
        return loss_value

    def _chunked_aligner_loss(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
    ) -> torch.Tensor:
        if self.nar:
            return self._chunked_aligner_nar_loss(encoded, encoded_len, transcript, transcript_len)
        # The chunked lattice needs at least one encoder frame per target token
        # (T >= U). Utterances with too few frames cannot host all tokens, so we
        # drop them from the loss (mask label length to 0) and warn -- mirroring
        # the offline Aligner's feasibility guard.
        transcript_len = transcript_len.to(encoded_len.device)
        insufficient_frames = encoded_len < transcript_len
        if torch.any(insufficient_frames):
            bad_idx = torch.nonzero(insufficient_frames, as_tuple=False).flatten().tolist()
            details = ", ".join(
                f"(batch_idx={i}, T={int(encoded_len[i])}, U={int(transcript_len[i])})" for i in bad_idx
            )
            logging.warning(
                f"[chunked-aligner] Skipping {len(bad_idx)}/{int(encoded_len.size(0))} utterance(s) with too "
                f"few encoder frames for the chunked lattice (T < U): {details}. If this is not rare, lower "
                f"model.train_ds.max_duration or the encoder subsampling_factor."
            )
            transcript_len = transcript_len.clone()
            transcript_len[insufficient_frames] = 0

        # The encoder / decoder may emit a few extra padding steps relative to the
        # valid lengths. The kernel requires acts to be exactly [B, max_T, max_U+1, V],
        # so trim the transcript (-> U) before the joint and the joint (-> T) after,
        # mirroring the standard RNNTLoss wrapper.
        max_logit_len = int(encoded_len.max().item())
        max_targets_len = int(transcript_len.max().item())
        if transcript.shape[1] != max_targets_len:
            transcript = transcript.narrow(dim=1, start=0, length=max_targets_len).contiguous()

        decoder_outputs, _, _ = self.decoder(targets=transcript, target_length=transcript_len)
        # Raw joint logits [B, T, U, V]; the loss applies log_softmax internally.
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_outputs)
        if joint.shape[1] != max_logit_len:
            joint = joint.narrow(dim=1, start=0, length=max_logit_len).contiguous()

        loss_value = self.loss(
            acts=joint,
            labels=transcript,
            act_lens=encoded_len,
            label_lens=transcript_len,
        )
        if loss_value.dim() > 0:
            loss_value = loss_value.squeeze()
        return loss_value

    def _references_from_targets(self, transcript: torch.Tensor, transcript_len: torch.Tensor) -> List[str]:
        transcript = transcript.long().cpu()
        transcript_len = transcript_len.long().cpu()
        refs = []
        for b in range(transcript.size(0)):
            ids = transcript[b, : int(transcript_len[b].item())].tolist()
            refs.append(self.decoding.decode_ids_to_str(ids))
        return refs

    def _wer_counts(self, hypotheses: List[str], references: List[str]) -> Tuple[int, int]:
        # Honor ``use_cer``: character-level edit distance for character/CER models
        # (e.g. Mandarin), word-level (whitespace split) otherwise. This mirrors
        # NeMo's ``word_error_rate`` so the aligner / chunked-aligner WER is
        # comparable to the standard ``WER`` metric used by the RNN-T / TDT recipes.
        # Without this, character models (no word spaces) treat the whole utterance
        # as a single "word", so any imperfect utterance counts as 100% error.
        use_cer = self._cfg.get('use_cer', False)
        scores = 0
        words = 0
        for hyp, ref in zip(hypotheses, references):
            if use_cer:
                hyp_tokens = list(hyp)
                ref_tokens = list(ref)
            else:
                hyp_tokens = hyp.split()
                ref_tokens = ref.split()
            scores += editdistance.eval(hyp_tokens, ref_tokens)
            words += len(ref_tokens)
        return scores, words

    def _infer_chat_chunk_size(self) -> int:
        """
        Infer chunk_size for RNNTAttJoint from the encoder's attention context config.

        Requires the encoder to use 'chunked_limited' attention with a single
        [left, right] context size. Returns right_context + 1 as the chunk size.
        """
        encoder_cfg = self.cfg.get('encoder', {})
        att_context_style = encoder_cfg.get('att_context_style', 'regular')
        att_context_size = encoder_cfg.get('att_context_size', None)

        if att_context_style != 'chunked_limited':
            raise ValueError(
                f"RNNTAttJoint requires encoder with att_context_style='chunked_limited', "
                f"got '{att_context_style}'. Set chunk_size explicitly in the joint config "
                f"or configure the encoder for chunked attention."
            )
        if att_context_size is None:
            raise ValueError(
                "RNNTAttJoint requires encoder att_context_size to be set. "
                "Set chunk_size explicitly in the joint config."
            )

        ctx = list(att_context_size) if hasattr(att_context_size, '__iter__') else att_context_size
        if not isinstance(ctx, (list, tuple)) or len(ctx) != 2 or not isinstance(ctx[0], int):
            raise ValueError(
                f"RNNTAttJoint requires a single [left, right] attention context, "
                f"got {att_context_size}. Set chunk_size explicitly in the joint config."
            )

        right_context = ctx[1]
        chunk_size = right_context + 1
        logging.info(f"CHAT mode: inferred chunk_size={chunk_size} from encoder att_context_size={ctx}")
        return chunk_size

    def setup_optim_normalization(self):
        """
        Helper method to setup normalization of certain parts of the model prior to the optimization step.

        Supported pre-optimization normalizations are as follows:

        .. code-block:: yaml

            # Variation Noise injection
            model:
                variational_noise:
                    std: 0.0
                    start_step: 0

            # Joint - Length normalization
            model:
                normalize_joint_txu: false

            # Encoder Network - gradient normalization
            model:
                normalize_encoder_norm: false

            # Decoder / Prediction Network - gradient normalization
            model:
                normalize_decoder_norm: false

            # Joint - gradient normalization
            model:
                normalize_joint_norm: false
        """
        # setting up the variational noise for the decoder
        if hasattr(self.cfg, 'variational_noise'):
            self._optim_variational_noise_std = self.cfg['variational_noise'].get('std', 0)
            self._optim_variational_noise_start = self.cfg['variational_noise'].get('start_step', 0)
        else:
            self._optim_variational_noise_std = 0
            self._optim_variational_noise_start = 0

        # Setup normalized gradients for model joint by T x U scaling factor (joint length normalization)
        self._optim_normalize_joint_txu = self.cfg.get('normalize_joint_txu', False)
        self._optim_normalize_txu = None

        # Setup normalized encoder norm for model
        self._optim_normalize_encoder_norm = self.cfg.get('normalize_encoder_norm', False)

        # Setup normalized decoder norm for model
        self._optim_normalize_decoder_norm = self.cfg.get('normalize_decoder_norm', False)

        # Setup normalized joint norm for model
        self._optim_normalize_joint_norm = self.cfg.get('normalize_joint_norm', False)

    def extract_rnnt_loss_cfg(self, cfg: Optional[DictConfig]):
        """
        Helper method to extract the rnnt loss name, and potentially its kwargs
        to be passed.

        Args:
            cfg: Should contain `loss_name` as a string which is resolved to a RNNT loss name.
                If the default should be used, then `default` can be used.
                Optionally, one can pass additional kwargs to the loss function. The subdict
                should have a keyname as follows : `{loss_name}_kwargs`.

                Note that whichever loss_name is selected, that corresponding kwargs will be
                selected. For the "default" case, the "{resolved_default}_kwargs" will be used.

        Examples:
            .. code-block:: yaml

                loss_name: "default"
                warprnnt_numba_kwargs:
                    kwargs2: some_other_val

        Returns:
            A tuple, the resolved loss name as well as its kwargs (if found).
        """
        if cfg is None:
            cfg = DictConfig({})

        loss_name = cfg.get("loss_name", "default")

        if loss_name == "default":
            loss_name = resolve_rnnt_default_loss_name()

        loss_kwargs = cfg.get(f"{loss_name}_kwargs", None)

        logging.info(f"Using RNNT Loss : {loss_name}\n" f"Loss {loss_name}_kwargs: {loss_kwargs}")

        return loss_name, loss_kwargs

    def set_decoding_type_according_to_loss(self, decoding_cfg):
        loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))

        if loss_name == 'tdt':
            decoding_cfg.durations = loss_kwargs.durations
        elif loss_name == 'multiblank_rnnt':
            decoding_cfg.big_blank_durations = loss_kwargs.big_blank_durations

        return decoding_cfg

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        use_lhotse: bool = True,
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: Optional[bool] = None,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:
            audio: (a single or list) of paths to audio files or a np.ndarray/tensor audio array or path
                to a manifest file.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            use_lhotse: (bool) If audio is not a dataloder, defines whether to create a lhotse dataloader or a
                non-lhotse dataloader.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            partial_hypothesis: Optional[List['Hypothesis']] - A list of partial hypotheses to be used during rnnt
                decoding. This is useful for streaming rnnt decoding. If this is not None, then the length of this
                list should be equal to the length of the audio list.
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels
                from multi-channel audio. If set to `'average'`, it performs averaging across channels.
                Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            verbose: (bool) whether to display tqdm progress bar
            timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis object
                (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class for more details.
                Default is None and would retain the previous state set by using self.change_decoding_strategy().
            override_config: (Optional[TranscribeConfig]) override transcription config pre-defined by the user.
                **Note**: All other arguments in the function will be ignored if override_config is passed.
                You should call this argument as `model.transcribe(audio, override_config=TranscribeConfig(...))`.

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """

        timestamps = timestamps or (override_config.timestamps if override_config is not None else None)
        if timestamps is not None:
            need_change_decoding = False
            if timestamps or (override_config is not None and override_config.timestamps):
                logging.info(
                    "Timestamps requested, setting decoding timestamps to True. Capture them in Hypothesis object, \
                        with output[0][idx].timestep['word'/'segment'/'char']"
                )
                return_hypotheses = True
                if self.cfg.decoding.get("compute_timestamps", None) is not True:
                    # compute_timestamps None, False or non-existent -> change to True
                    need_change_decoding = True
                    with open_dict(self.cfg.decoding):
                        self.cfg.decoding.compute_timestamps = True
            else:
                return_hypotheses = False
                if self.cfg.decoding.get("compute_timestamps", None) is not False:
                    # compute_timestamps None, True or non-existent -> change to False
                    need_change_decoding = True
                    with open_dict(self.cfg.decoding):
                        self.cfg.decoding.compute_timestamps = False

            if need_change_decoding:
                self.change_decoding_strategy(self.cfg.decoding, verbose=False)

        return super().transcribe(
            audio=audio,
            use_lhotse=use_lhotse,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=override_config,
            # Additional arguments
            partial_hypothesis=partial_hypothesis,
        )

    def change_vocabulary(self, new_vocabulary: List[str], decoding_cfg: Optional[DictConfig] = None):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a
        pre-trained model. This method changes only decoder and leaves encoder and pre-processing
        modules unchanged. For example, you would use it if you want to use pretrained encoder when
        fine-tuning on data in another language, or when you'd need model to learn capitalization,
        punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.

        Returns: None

        """
        if self.joint.vocabulary == new_vocabulary:
            logging.warning(f"Old {self.joint.vocabulary} and new {new_vocabulary} match. Not changing anything.")
        else:
            if new_vocabulary is None or len(new_vocabulary) == 0:
                raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')

            joint_config = self.joint.to_config_dict()
            new_joint_config = copy.deepcopy(joint_config)
            new_joint_config['vocabulary'] = new_vocabulary
            new_joint_config['num_classes'] = len(new_vocabulary)
            del self.joint
            self.joint = EncDecRNNTModel.from_config_dict(new_joint_config)

            decoder_config = self.decoder.to_config_dict()
            new_decoder_config = copy.deepcopy(decoder_config)
            new_decoder_config.vocab_size = len(new_vocabulary)
            del self.decoder
            self.decoder = EncDecRNNTModel.from_config_dict(new_decoder_config)

            del self.loss
            loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get('loss', None))
            self.loss = RNNTLoss(
                num_classes=self.joint.num_classes_with_blank - 1, loss_name=loss_name, loss_kwargs=loss_kwargs
            )

            if decoding_cfg is None:
                # Assume same decoding config as before
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(RNNTDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
            decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

            self.decoding = RNNTDecoding(
                decoding_cfg=decoding_cfg,
                decoder=self.decoder,
                joint=self.joint,
                vocabulary=self.joint.vocabulary,
            )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer or (
                self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
            ):
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            # Update config
            with open_dict(self.cfg.joint):
                self.cfg.joint = new_joint_config

            with open_dict(self.cfg.decoder):
                self.cfg.decoder = new_decoder_config

            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            ds_keys = ['train_ds', 'validation_ds', 'test_ds']
            for key in ds_keys:
                if key in self.cfg:
                    with open_dict(self.cfg[key]):
                        self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

            logging.info(f"Changed decoder to output to {self.joint.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig, verbose=True):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            verbose: (bool) whether to display logging information
        """
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        self.decoding = RNNTDecoding(
            decoding_cfg=decoding_cfg,
            decoder=self.decoder,
            joint=self.joint,
            vocabulary=self.joint.vocabulary,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        self.joint.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                # During transcription, the model is initially loaded on the CPU.
                # To ensure the correct global_rank and world_size are set,
                # these values must be passed from the configuration.
                global_rank=self.global_rank if not config.get("do_transcribe", False) else config.get("global_rank"),
                world_size=self.world_size if not config.get("do_transcribe", False) else config.get("world_size"),
                dataset=LhotseSpeechToTextBpeDataset(
                    tokenizer=make_parser(
                        labels=config.get('labels', None),
                        name=config.get('parser', 'en'),
                        unk_id=config.get('unk_index', -1),
                        blank_id=config.get('blank_index', -1),
                        do_normalize=config.get('normalize_transcripts', False),
                    ),
                    return_cuts=config.get("do_transcribe", False),
                ),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_char_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            preprocessor_cfg=self._cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToCharDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        batch_sampler = None
        if config.get('use_semi_sorted_batching', False):
            if not isinstance(dataset, _AudioTextDataset):
                raise RuntimeError(
                    "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
                    f"but found dataset of type {type(dataset)}"
                )
            # set batch_size and batch_sampler to None to disable automatic batching
            batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
            config['batch_size'] = None
            config['drop_last'] = False
            shuffle = False

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            sampler=batch_sampler,
            batch_sampler=None,
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.

        if (
            self._train_dl is not None
            and hasattr(self._train_dl, 'dataset')
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )
            elif self._trainer is None:
                logging.warning(
                    "Model Trainer was not set before constructing the dataset, incorrect number of "
                    "training batches will be used. Please set the trainer and rebuild the dataset."
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()

        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        # Feature-domain time/pitch warp (training only); updates the sequence length.
        if self.spec_warp is not None and self.training:
            processed_signal, processed_signal_length = self.spec_warp(
                input_spec=processed_signal, length=processed_signal_length
            )

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        # Optional chunked frame reduction (Chunked-Aligner): either compress each
        # acoustic chunk into learned tokens, or keep the left-most k frames.
        if self.token_extractor is not None:
            encoded, encoded_len = self.token_extractor(encoded, encoded_len)
        elif self.channel_token_mixer is not None:
            encoded, encoded_len = self.channel_token_mixer(encoded, encoded_len)
        elif self.first_k_frames_per_chunk >= 1:
            encoded, encoded_len = self._select_first_k_frames_per_chunk(encoded, encoded_len)

        return encoded, encoded_len

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if self.loss_type in ('aligner', 'chunked_aligner'):
            if self.loss_type == 'aligner':
                loss_value = self._aligner_loss(encoded, encoded_len, transcript, transcript_len)
            else:
                loss_value = self._chunked_aligner_loss(encoded, encoded_len, transcript, transcript_len)
            loss_value = self.add_auxiliary_losses(loss_value)

            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            # All periodic logging keys off the trainer ``global_step`` so that the
            # batch-WER computation, the stats summary and the hyp/ref sample logging
            # stay in sync. (Previously the WER/decoding gate used ``batch_nb``, which
            # resets every epoch and so almost never lined up with the
            # ``global_step``-based stats/prediction gates -- the sampled hyp/ref
            # comparison was therefore effectively never printed.)
            global_step = self.trainer.global_step if self.trainer is not None else batch_nb
            is_global_zero = self.trainer is not None and self.trainer.is_global_zero

            log_every_n_steps = self._trainer.log_every_n_steps if self._trainer is not None else 1
            log_stats_every_n_steps = int(self.cfg.get('log_stats_every_n_steps', 10))
            log_prediction_every_n_steps = int(self.cfg.get('log_prediction_every_n_steps', 100))

            log_train_wer = log_every_n_steps > 0 and global_step % log_every_n_steps == 0
            log_stats = log_stats_every_n_steps > 0 and is_global_zero and global_step % log_stats_every_n_steps == 0
            log_prediction = (
                log_prediction_every_n_steps > 0
                and is_global_zero
                and global_step % log_prediction_every_n_steps == 0
            )

            # Decode once if either the batch-WER value or the hyp/ref samples are needed this step.
            train_wer = None
            hypotheses = None
            references = None
            if log_train_wer or log_prediction:
                with torch.no_grad():
                    hypotheses, _ = self.decoding.decode_encoder_output(encoded.detach(), encoded_len)
                    references = self._references_from_targets(transcript, transcript_len)
                    scores, words = self._wer_counts(hypotheses, references)
                    train_wer = torch.tensor(scores / max(words, 1), dtype=torch.float32, device=encoded.device)
                if log_train_wer:
                    tensorboard_logs['training_batch_wer'] = train_wer

            if log_stats:
                logging.info(
                    f"[{self.loss_type}-train] "
                    f"step={self.trainer.global_step} "
                    f"epoch={self.trainer.current_epoch} "
                    f"loss={float(loss_value.detach().float().cpu()):.4f} "
                    f"lr={self._optimizer.param_groups[0]['lr']:.3e} "
                    f"bs={encoded.size(0)} "
                    f"enc_frames_mean={float(encoded_len.float().mean().detach().cpu()):.1f} "
                    f"enc_frames_max={int(encoded_len.max().detach().cpu())} "
                    f"tgt_len_mean={float(transcript_len.float().mean().detach().cpu()):.1f} "
                    f"tgt_len_max={int(transcript_len.max().detach().cpu())}"
                    + (f" training_batch_wer={float(train_wer.detach().cpu()):.4f}" if train_wer is not None else "")
                )

            if log_prediction and hypotheses is not None and references is not None:
                max_samples = min(int(self.cfg.get('log_prediction_num_samples', 2)), len(hypotheses))
                for sample_idx in range(max_samples):
                    logging.info("\n")
                    logging.info(f"WER reference:{references[sample_idx]}")
                    logging.info(f"WER predicted:{hypotheses[sample_idx]}")

            self.log_dict(tensorboard_logs)
            return {'loss': loss_value}

        # Multi-target (token + pronunciation) path: sample one head and train it.
        if getattr(self, 'multi_target_enabled', False):
            return self._multi_target_training_step(encoded, encoded_len, transcript, transcript_len)

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, encoder_lengths=encoded_len)
            effective_len = getattr(self.joint, 'num_chunks_per_utterance', encoded_len)
            loss_value = self.loss(
                log_probs=joint,
                targets=transcript,
                input_lengths=effective_len,
                target_lengths=target_length,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if self.loss_type in ('aligner', 'chunked_aligner'):
            texts, _ = self.decoding.decode_encoder_output(encoded, encoded_len)
            if isinstance(sample_id, torch.Tensor):
                sample_id = sample_id.cpu().detach().numpy()
            return list(zip(sample_id, texts))

        best_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )

        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        if self.loss_type in ('aligner', 'chunked_aligner'):
            if self.compute_eval_loss:
                if self.loss_type == 'aligner':
                    tensorboard_logs['val_loss'] = self._aligner_loss(
                        encoded, encoded_len, transcript, transcript_len
                    )
                else:
                    tensorboard_logs['val_loss'] = self._chunked_aligner_loss(
                        encoded, encoded_len, transcript, transcript_len
                    )

            hypotheses, _ = self.decoding.decode_encoder_output(encoded, encoded_len)
            references = self._references_from_targets(transcript, transcript_len)
            scores, words = self._wer_counts(hypotheses, references)

            tensorboard_logs['val_wer_num'] = torch.tensor(scores, dtype=torch.float32, device=encoded.device)
            tensorboard_logs['val_wer_denom'] = torch.tensor(words, dtype=torch.float32, device=encoded.device)
            tensorboard_logs['val_wer'] = torch.tensor(
                scores / max(words, 1), dtype=torch.float32, device=encoded.device
            )
            self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))
            return tensorboard_logs

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, encoder_lengths=encoded_len)
                effective_len = getattr(self.joint, 'num_chunks_per_utterance', encoded_len)
                loss_value = self.loss(
                    log_probs=joint,
                    targets=transcript,
                    input_lengths=effective_len,
                    target_lengths=target_length,
                )

                tensorboard_logs['val_loss'] = loss_value

            if getattr(self, 'multi_target_enabled', False):
                # One decode -> char / notone / tone error rates. val_wer (char) stays
                # the monitored metric; the pinyin reps are extra diagnostics.
                counts = self._mt_repr_error_counts(encoded, encoded_len, transcript, transcript_len)
                for rep, (edits, ref_len) in counts.items():
                    prefix = 'val_wer' if rep == 'char' else f'val_wer_{rep}'
                    tensorboard_logs[f'{prefix}_num'] = torch.tensor(
                        edits, dtype=torch.float32, device=encoded.device
                    )
                    tensorboard_logs[f'{prefix}_denom'] = torch.tensor(
                        ref_len, dtype=torch.float32, device=encoded.device
                    )
                tensorboard_logs['val_wer'] = torch.tensor(
                    counts['char'][0] / max(counts['char'][1], 1), dtype=torch.float32, device=encoded.device
                )
            else:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                wer, wer_num, wer_denom = self.wer.compute()
                self.wer.reset()

                tensorboard_logs['val_wer_num'] = wer_num
                tensorboard_logs['val_wer_denom'] = wer_denom
                tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        # Aggregate any additional representation-level error rates (e.g.
        # val_wer_notone / val_wer_tone from the multi-target model), each logged as
        # summed num/denom per step so the epoch value is a proper micro-average.
        for base in sorted({k[:-4] for k in outputs[0] if k.endswith('_num') and k != 'val_wer_num'}):
            if f'{base}_denom' not in outputs[0]:
                continue
            num = torch.stack([x[f'{base}_num'] for x in outputs]).sum()
            denom = torch.stack([x[f'{base}_denom'] for x in outputs]).sum()
            tensorboard_logs[base] = num.float() / torch.clamp(denom, min=1.0)
        if (
            self.loss_type in ('aligner', 'chunked_aligner')
            and self.trainer is not None
            and self.trainer.is_global_zero
        ):
            val_loss_msg = (
                f" val_loss={float(val_loss_mean.detach().float().cpu()):.4f}" if self.compute_eval_loss else ""
            )
            logging.info(
                f"[{self.loss_type}-val] "
                f"step={self.trainer.global_step} "
                f"epoch={self.trainer.current_epoch}"
                f"{val_loss_msg} "
                f"val_wer={float(tensorboard_logs['val_wer'].detach().float().cpu()):.4f} "
                f"words={int(wer_denom.detach().cpu())}"
            )
        return {**val_loss_log, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}
        for base in sorted({k[:-4] for k in outputs[0] if k.endswith('_num') and k != 'test_wer_num'}):
            if f'{base}_denom' not in outputs[0]:
                continue
            num = torch.stack([x[f'{base}_num'] for x in outputs]).sum()
            denom = torch.stack([x[f'{base}_denom'] for x in outputs]).sum()
            tensorboard_logs[base] = num.float() / torch.clamp(denom, min=1.0)
        return {**test_loss_log, 'log': tensorboard_logs}

    """ Transcription related methods """

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
        output = dict(encoded=encoded, encoded_len=encoded_len)
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Union[List['Hypothesis'], List[List['Hypothesis']]]:
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')

        if self.loss_type in ('aligner', 'chunked_aligner'):
            texts, token_ids = self.decoding.decode_encoder_output(encoded, encoded_len)
            del encoded, encoded_len
            return [
                Hypothesis(score=0.0, y_sequence=torch.tensor(ids, dtype=torch.long), text=text)
                for text, ids in zip(texts, token_ids)
            ]

        hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=trcfg.return_hypotheses,
            partial_hypotheses=trcfg.partial_hypothesis,
        )
        del encoded, encoded_len

        if trcfg.timestamps:
            hyp = process_timestamp_outputs(
                hyp, self.encoder.subsampling_factor, self.cfg['preprocessor']['window_stride']
            )

        return hyp

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'labels': self.joint.vocabulary,
            'batch_size': batch_size,
            'trim_silence': False,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio=audio, trcfg=trcfg)
        # add biasing requests to the decoding computer
        try:
            biasing_multi_model = self.decoding.decoding.decoding_computer.biasing_multi_model
        except AttributeError:
            biasing_multi_model = None
        if trcfg.partial_hypothesis:
            for partial_hyp in trcfg.partial_hypothesis:
                if (
                    isinstance(partial_hyp, Hypothesis)
                    and partial_hyp.has_biasing_request()
                    and partial_hyp.biasing_cfg.auto_manage_multi_model
                    and partial_hyp.biasing_cfg.multi_model_id is None
                ):
                    if biasing_multi_model is not None:
                        partial_hyp.biasing_cfg.add_to_multi_model(
                            tokenizer=self.tokenizer, biasing_multi_model=biasing_multi_model
                        )
                    else:
                        logging.warning("Requested biasing for hypothesis, but multi-model is not found, skipping.")

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg=trcfg)
        try:
            biasing_multi_model = self.decoding.decoding.decoding_computer.biasing_multi_model
        except AttributeError:
            biasing_multi_model = None

        # remove biasing requests from the decoding computer
        if biasing_multi_model is not None and trcfg.partial_hypothesis:
            for partial_hyp in trcfg.partial_hypothesis:
                if (
                    isinstance(partial_hyp, Hypothesis)
                    and partial_hyp.has_biasing_request()
                    and partial_hyp.biasing_cfg.auto_manage_multi_model
                ):
                    partial_hyp.biasing_cfg.remove_from_multi_model(biasing_multi_model=biasing_multi_model)

    def on_after_backward(self):
        super().on_after_backward()
        if self._optim_variational_noise_std > 0 and self.global_step >= self._optim_variational_noise_start:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0.0,
                        std=self._optim_variational_noise_std,
                        size=param.size(),
                        device=param.device,
                        dtype=param.dtype,
                    )
                    param.grad.data.add_(noise)

        if self._optim_normalize_joint_txu:
            T, U = self._optim_normalize_txu
            if T is not None and U is not None:
                for param_name, param in self.encoder.named_parameters():
                    if param.grad is not None:
                        param.grad.data.div_(U)

                for param_name, param in self.decoder.named_parameters():
                    if param.grad is not None:
                        param.grad.data.div_(T)

        if self._optim_normalize_encoder_norm:
            for param_name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

        if self._optim_normalize_decoder_norm:
            for param_name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

        if self._optim_normalize_joint_norm:
            for param_name, param in self.joint.named_parameters():
                if param.grad is not None:
                    norm = param.grad.norm()
                    param.grad.data.div_(norm)

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        return ['encoder', 'decoder_joint']

    # for export
    @property
    def decoder_joint(self):
        return RNNTDecoderJoint(self.decoder, self.joint)

    def set_export_config(self, args):
        if 'decoder_type' in args:
            if hasattr(self, 'change_decoding_strategy'):
                self.change_decoding_strategy(decoder_type=args['decoder_type'])
            else:
                raise Exception("Model does not have decoder type option")
        super().set_export_config(args)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_zh_conformer_transducer_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_zh_conformer_transducer_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_zh_conformer_transducer_large/versions/1.8.0/files/stt_zh_conformer_transducer_large.nemo",
        )
        results.append(model)

        return results

    @property
    def wer(self):
        return self._wer

    @wer.setter
    def wer(self, wer):
        self._wer = wer
