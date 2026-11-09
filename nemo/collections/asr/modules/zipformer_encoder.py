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

"""NeMo-facing wrapper around icefall's Zipformer.

The vendored encoder under ``parts/submodules/zipformer`` is upstream code kept
as close to verbatim as possible so it stays diffable. Everything that makes it
look like a NeMo encoder lives here:

  * SHAPES.   NeMo passes ``(B, D, T)`` and expects ``(B, D, T)`` back. icefall
              works in ``(T, B, C)`` and takes features as ``(B, T, D)``.
  * STACKING. Upstream pairs ``Zipformer2`` with a separate ``Conv2dSubsampling``
              front-end; a NeMo encoder is expected to own its subsampling.
  * PADDING.  Zipformer wants an explicit ``src_key_padding_mask``; NeMo passes
              lengths.
  * ACCOUNTING. ``subsampling_factor`` is read by RNNT timestamp code and by
              CHAT to derive its frame length, so it must report the TOTAL factor
              (conv front-end x the encoder's own output downsampling).
"""

from typing import Optional, Tuple

import torch
from torch import nn

from nemo.collections.asr.parts.submodules.zipformer.subsampling import Conv2dSubsampling
from nemo.collections.asr.parts.submodules.zipformer.zipformer import Zipformer2
from nemo.core.classes.common import typecheck
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging

__all__ = ["ZipformerEncoder"]


def _to_tuple(x, n: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * n
    t = tuple(int(v) for v in x)
    if len(t) == 1:
        return t * n
    return t


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """True where padded. ``(B, max_len)``."""
    n = lengths.size(0)
    idx = torch.arange(max_len, device=lengths.device).expand(n, max_len)
    return idx >= lengths.unsqueeze(-1)


class ZipformerEncoder(NeuralModule):
    """Zipformer encoder with NeMo's input/output conventions.

    Args:
        feat_in: input mel bins.
        encoder_dim: per-stack width. The module's OUTPUT width is ``max(encoder_dim)``.
        downsampling_factor: per-stack downsampling INSIDE the encoder (the U-net).
        output_downsampling_factor: applied to the encoder's output.
        causal: enable chunked/streaming attention.
        chunk_size: chunk sizes in encoder frames; ``[-1]`` = full context.
        left_context_frames: left context in encoder frames; ``[-1]`` = unlimited.
    """

    @property
    def input_types(self):
        return {
            "audio_signal": NeuralType(("B", "D", "T"), SpectrogramType()),
            "length": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "outputs": NeuralType(("B", "D", "T"), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(
        self,
        feat_in: int = 80,
        encoder_dim=(192, 256, 384, 512, 384, 256),
        num_encoder_layers=(2, 2, 3, 4, 3, 2),
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
        encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),
        num_heads=(4, 4, 4, 8, 4, 4),
        query_head_dim: int = 32,
        pos_head_dim: int = 4,
        value_head_dim: int = 12,
        pos_dim: int = 48,
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        output_downsampling_factor: int = 2,
        dropout: Optional[float] = None,
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size=(-1,),
        left_context_frames=(-1,),
    ):
        super().__init__()
        n = len(_to_tuple(num_encoder_layers, 1)) if not isinstance(num_encoder_layers, int) else 1
        n = max(n, len(encoder_dim) if not isinstance(encoder_dim, int) else 1)

        enc_dim = _to_tuple(encoder_dim, n)
        self._feat_in = int(feat_in)
        self._feat_out = max(enc_dim)

        # Upstream's recipe: Conv2dSubsampling (2x) feeds Zipformer2, whose own
        # output_downsampling_factor applies on top. NeMo expects one module, so
        # both live here and subsampling_factor reports the product.
        self.encoder_embed = Conv2dSubsampling(
            in_channels=self._feat_in,
            out_channels=enc_dim[0],
            dropout=dropout if dropout is not None else 0.1,
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=int(output_downsampling_factor),
            downsampling_factor=_to_tuple(downsampling_factor, n),
            encoder_dim=enc_dim,
            num_encoder_layers=_to_tuple(num_encoder_layers, n),
            encoder_unmasked_dim=_to_tuple(encoder_unmasked_dim, n),
            query_head_dim=_to_tuple(query_head_dim, n),
            pos_head_dim=_to_tuple(pos_head_dim, n),
            value_head_dim=_to_tuple(value_head_dim, n),
            num_heads=_to_tuple(num_heads, n),
            feedforward_dim=_to_tuple(feedforward_dim, n),
            cnn_module_kernel=_to_tuple(cnn_module_kernel, n),
            pos_dim=int(pos_dim),
            warmup_batches=float(warmup_batches),
            causal=bool(causal),
            chunk_size=list(_to_tuple(chunk_size, 1)),
            left_context_frames=list(_to_tuple(left_context_frames, 1)),
            **({"dropout": dropout} if dropout is not None else {}),
        )

        # CHUNK SIZE MUST DIVIDE EVERY STACK'S DOWNSAMPLING FACTOR.
        #
        # Zipformer's U-net runs middle stacks at reduced rates, and the chunked
        # attention mask is built by folding the sequence by each stack's factor --
        # so a chunk that is not a multiple of them cannot be expressed. Upstream
        # asserts this deep inside _get_attn_mask with a bare AssertionError; catch
        # it here where the config is visible.
        #
        # NOTE chunk_size and left_context_frames are counted at the encoder's
        # INTERNAL rate -- after Conv2dSubsampling (2x) but BEFORE
        # output_downsampling_factor. With a 10 ms hop that is 20 ms per frame,
        # NOT the 1/subsampling_factor rate of the module's output.
        # Upstream hard-asserts this inside forward (zipformer.py: "assert
        # self.output_downsampling_factor == 2"), so the TOTAL subsampling is fixed
        # at 2 (Conv2dSubsampling) x 2 = 4 -> 40 ms frames from a 10 ms hop. That is
        # FINER than the Conformer donors here (8x, 80 ms), so anything deriving a
        # frame length from subsampling_factor must be told, not assumed.
        if int(output_downsampling_factor) != 2:
            raise ValueError(
                f"output_downsampling_factor must be 2 (upstream Zipformer asserts it); "
                f"got {output_downsampling_factor}. Use downsampling_factor to change the "
                f"INTERNAL stack rates instead."
            )

        ds = _to_tuple(downsampling_factor, n)
        for cs in _to_tuple(chunk_size, 1):
            if cs > 0 and any(cs % d for d in ds):
                raise ValueError(
                    f"chunk_size={cs} must be divisible by every downsampling_factor {ds} "
                    f"(lcm={__import__('math').lcm(*ds)}). At a 20 ms internal frame, "
                    f"{cs} frames = {cs * 0.02:.2f}s."
                )

        # Conv2dSubsampling is 2x; the encoder applies output_downsampling_factor.
        self.subsampling_factor = 2 * int(output_downsampling_factor)
        logging.info(
            "ZipformerEncoder: feat_in=%d out_dim=%d subsampling_factor=%d (2 x %d) causal=%s "
            "chunk_size=%s left_context_frames=%s",
            self._feat_in,
            self._feat_out,
            self.subsampling_factor,
            output_downsampling_factor,
            causal,
            chunk_size,
            left_context_frames,
        )

    @property
    def feat_in(self) -> int:
        return self._feat_in

    @property
    def d_model(self) -> int:
        return self._feat_out

    @typecheck()
    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        # NeMo (B, D, T) -> icefall (B, T, D)
        x = audio_signal.transpose(1, 2)
        if length is None:
            length = torch.full((x.size(0),), x.size(1), device=x.device, dtype=torch.long)

        # ZERO THE PADDED TAIL BEFORE THE CONV FRONT-END.
        #
        # Zipformer masks ATTENTION via src_key_padding_mask, but Conv2dSubsampling
        # and the CNN modules convolve straight across the padding boundary. Measured
        # against this encoder: perturbing the padded region changed the last 5 output
        # frames (~200 ms) while frames 0..55 stayed bit-identical. NeMo's
        # ConformerEncoder shows zero contamination on the same test because it masks
        # first, so without this the two are not comparable -- and the corrupted tail
        # is exactly where CHAT's banded loss places an utterance's final words.
        x = x.masked_fill(make_pad_mask(length, x.size(1)).unsqueeze(-1), 0.0)

        x, x_lens = self.encoder_embed(x, length)

        # Zipformer2 works in (T, B, C) and needs an explicit padding mask.
        x = x.permute(1, 0, 2)
        mask = make_pad_mask(x_lens, x.size(0))
        enc, enc_lens = self.encoder(x, x_lens, src_key_padding_mask=mask)

        # back to NeMo (B, D, T), with the padded tail zeroed so downstream
        # consumers see the same contract the Conformer gives them.
        enc = enc.permute(1, 2, 0)
        enc = enc.masked_fill(make_pad_mask(enc_lens, enc.size(2)).unsqueeze(1), 0.0)
        return enc, enc_lens
