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

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch

from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
from nemo.collections.asr.parts.submodules.spectr_augment import SpecAugment, SpecCutout
from nemo.collections.audio.parts.utils.transforms import MFCC
from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import (
    AudioSignal,
    LengthsType,
    MelSpectrogramType,
    MFCCSpectrogramType,
    NeuralType,
    SpectrogramType,
)
from nemo.core.utils.optional_libs import NUMBA_CUDA_AVAILABLE
from nemo.utils import logging, logging_mode

if NUMBA_CUDA_AVAILABLE:
    from nemo.collections.asr.parts.numba.spec_augment import SpecAugmentNumba, spec_augment_launch_heuristics

__all__ = [
    'AudioToMelSpectrogramPreprocessor',
    'AudioToMFCCPreprocessor',
    'SpectrogramAugmentation',
    'SpectrogramTimePitchWarp',
    'MaskedPatchAugmentation',
    'CropOrPadSpectrogramAugmentation',
]


class AudioPreprocessor(NeuralModule, ABC):
    """
    An interface for Neural Modules that performs audio pre-processing,
    transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

        # Normally, when you call to(dtype) on a torch.nn.Module, all
        # floating point parameters and buffers will change to that
        # dtype, rather than being float32. The AudioPreprocessor
        # classes, uniquely, don't actually have any parameters or
        # buffers from what I see. In addition, we want the input to
        # the preprocessor to be float32, but need to create the
        # output in appropriate precision. We have this empty tensor
        # here just to detect which dtype tensor this module should
        # output at the end of execution.
        self.register_buffer("dtype_sentinel_tensor", torch.tensor((), dtype=torch.float32), persistent=False)

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        if input_signal.dtype != torch.float32:
            logging.warning(
                f"AudioPreprocessor received an input signal of dtype {input_signal.dtype}, rather than torch.float32. In sweeps across multiple datasets, we have found that the preprocessor is not robust to low precision  mathematics. As such, it runs in float32. Your input will be cast to float32, but this is not necessarily enough to recovery full accuracy. For example, simply casting input_signal from torch.float32 to torch.bfloat16, then back to torch.float32 before running AudioPreprocessor causes drops in absolute WER of up to 0.1%. torch.bfloat16 simply does not have enough mantissa bits to represent enough values in the range [-1.0,+1.0] correctly.",
                mode=logging_mode.ONCE,
            )
        processed_signal, processed_length = self.get_features(input_signal.to(torch.float32), length)
        processed_signal = processed_signal.to(self.dtype_sentinel_tensor.dtype)
        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor, Exportable):
    """Featurizer module that converts wavs to mel spectrograms.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        exact_pad (bool): If True, sets stft center to False and adds padding, such that num_frames = audio_length
            // hop_length. Defaults to False.
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
        rng : Random number generator
        nb_augmentation_prob (float) : Probability with which narrowband augmentation would be applied to
            samples in the batch.
            Defaults to 0.0
        nb_max_freq (int) : Frequency above which all frequencies will be masked for narrowband augmentation.
            Defaults to 4000
        mel_norm: Normalization used for mel filterbank weights.
            Defaults to 'slaney' (area normalization)
        stft_exact_pad: Deprecated argument, kept for compatibility with older checkpoints.
        stft_conv: Deprecated argument, kept for compatibility with older checkpoints.
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(
                tuple('B'), LengthsType()
            ),  # Please note that length should be in samples not seconds.
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.

        processed_signal:
            0: AxisType(BatchTag)
            1: AxisType(MelSpectrogramSignalTag)
            2: AxisType(ProcessedTimeTag)
        processed_length:
            0: AxisType(BatchTag)
        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        use_torchaudio: bool = False,  # Deprecated arguments; kept for config compatibility
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)
        super().__init__(n_window_size, n_window_stride)

        # Given the long and similar argument list, point to the class and instantiate it by reference
        self.featurizer = FilterbankFeatures(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=pad_value,
            mag_power=mag_power,
            rng=rng,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
            stft_exact_pad=stft_exact_pad,  # Deprecated arguments; kept for config compatibility
            stft_conv=stft_conv,  # Deprecated arguments; kept for config compatibility
        )

    def input_example(self, max_batch: int = 8, max_dim: int = 32000, min_length: int = 200):
        dev = self.filter_banks.device

        signals = torch.randn(size=[max_batch, max_dim], device=dev)
        lengths = torch.randint(low=min_length, high=max_dim, size=[max_batch], device=dev)
        lengths[0] = max_dim
        return signals, lengths

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram.
            Defaults to 0.02
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect.
            Defaults to 0.01
        n_window_size: Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride: Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null'].
            Defaults to 'hann'
        n_fft: Length of FT window. If None, it uses the smallest power of 2
            that is larger than n_window_size.
            Defaults to None
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        n_mels: Number of mel filterbanks.
            Defaults to 64
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MFCCSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window='hann',
        n_fft=None,
        lowfreq=0.0,
        highfreq=None,
        n_mels=64,
        n_mfcc=64,
        dct_type=2,
        norm='ortho',
        log=True,
    ):
        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        # Get win_length (n_window_size) and hop_length (n_window_stride)
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        super().__init__(n_window_size, n_window_stride)

        mel_kwargs = {}

        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels

        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))

        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None."
            )
        mel_kwargs['window_fn'] = window_fn

        # Use torchaudio's implementation of MFCCs as featurizer
        self.featurizer = MFCC(
            sample_rate=self._sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log,
            melkwargs=mel_kwargs,
        )

    def get_features(self, input_signal, length):
        features = self.featurizer(input_signal)
        seq_len = torch.ceil(length.to(torch.float32) / self.hop_length).to(dtype=torch.long)
        return features, seq_len


class SpectrogramAugmentation(NeuralModule):
    """
    Performs time and freq cuts in one of two ways.
    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.
    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
        use_numba_spec_augment: use numba code for Spectrogram augmentation
        use_vectorized_spec_augment: use vectorized code for Spectrogram augmentation

    """

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
        rect_masks=0,
        rect_time=5,
        rect_freq=20,
        rng=None,
        mask_value=0.0,
        use_vectorized_spec_augment: bool = True,
        use_numba_spec_augment: bool = False,
    ):
        super().__init__()

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng,
            )
            # self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda input_spec: input_spec
        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
                use_vectorized_code=use_vectorized_spec_augment,
            )
        else:
            self.spec_augment = lambda input_spec, length: input_spec

        # Check if numba is supported, and use a Numba kernel if it is
        if use_numba_spec_augment and NUMBA_CUDA_AVAILABLE:
            logging.info('Numba CUDA SpecAugment kernel is being used')
            self.spec_augment_numba = SpecAugmentNumba(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng,
                mask_value=mask_value,
            )
        else:
            self.spec_augment_numba = None

    @typecheck()
    def forward(self, input_spec, length):
        augmented_spec = self.spec_cutout(input_spec=input_spec)

        # To run the Numba kernel, correct numba version is required as well as
        # tensor must be on GPU and length must be provided
        if self.spec_augment_numba is not None and spec_augment_launch_heuristics(augmented_spec, length):
            augmented_spec = self.spec_augment_numba(input_spec=augmented_spec, length=length)
        else:
            augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)
        return augmented_spec


class SpectrogramTimePitchWarp(NeuralModule):
    """Independent 2-D (time x pitch) warp of a (mel-)spectrogram.

    Decouples the two axes that ordinary speed perturbation forces together:

      * Time factor ``alpha`` -> a *tempo* change. The valid region of each
        sample is stretched/compressed to ``round(alpha * length)`` frames
        (pitch unchanged). This changes the returned sequence ``length``.
      * Pitch factor ``beta`` -> a *frequency-axis* warp (VTLP-style). A feature
        originally at bin ``b`` is mapped to output bin ``beta * b`` (duration
        unchanged). NOTE: this warps the mel-bin axis directly, a cheap
        approximation of true linear-Hz VTLP.

    Both factors are sampled *independently per sample* as the average of two
    i.i.d. ``Uniform[1 - dev, 1 + dev]`` draws -- a symmetric triangular
    distribution centered at 1.0 that favors mild warps over the extremes. Set a
    per-axis ``dev`` of 0 to disable that axis.

    Implemented as a single ``grid_sample`` (one bilinear resample handles both
    axes at once). Applied on the spectrogram, so no waveform resynthesis is
    needed. Intended for training only; the caller should guard with
    ``model.training`` and feed the returned ``length`` to the encoder/loss.

    Args:
        time_warp (float): max deviation ``dev`` for the time factor. Must be in
            ``[0.0, 1.0)``. ``0`` disables time warping. Default 0.3 -> [0.7, 1.3].
        pitch_warp (float): max deviation ``dev`` for the pitch factor. Must be in
            ``[0.0, 1.0)``. ``0`` disables pitch warping. Default 0.3 -> [0.7, 1.3].
        prob (float): per-batch probability of applying the warp. Default 1.0.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            "augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, time_warp: float = 0.3, pitch_warp: float = 0.3, prob: float = 1.0):
        super().__init__()
        for name, dev in (("time_warp", time_warp), ("pitch_warp", pitch_warp)):
            if not (0.0 <= dev < 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0), got {dev}")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be in [0.0, 1.0], got {prob}")
        self.time_dev = float(time_warp)
        self.pitch_dev = float(pitch_warp)
        self.prob = float(prob)

    def _sample_factor(self, dev, n, device):
        """Per-sample factor = mean of two U[1-dev, 1+dev] draws (triangular)."""
        if dev <= 0.0:
            return torch.ones(n, device=device)
        lo, hi = 1.0 - dev, 1.0 + dev
        u = torch.rand(n, 2, device=device) * (hi - lo) + lo
        return u.mean(dim=1)

    @typecheck()
    def forward(self, input_spec, length):
        if (self.time_dev <= 0.0 and self.pitch_dev <= 0.0) or (self.prob < 1.0 and random.random() >= self.prob):
            return input_spec, length

        B, D, T = input_spec.shape
        device = input_spec.device
        alpha = self._sample_factor(self.time_dev, B, device)  # time / tempo
        beta = self._sample_factor(self.pitch_dev, B, device)  # pitch / freq

        # New per-sample valid lengths after the time warp (factor < 2 by construction).
        new_length = torch.clamp((length.to(torch.float32) * alpha).round().long(), min=1)
        T_out = int(new_length.max().item())

        # align_corners=True index<->normalized mapping, guard size-1 dims.
        t_den = max(T - 1, 1)
        d_den = max(D - 1, 1)

        w = torch.arange(T_out, device=device, dtype=torch.float32)  # [T_out]
        h = torch.arange(D, device=device, dtype=torch.float32)  # [D]
        # output time index w samples input time w / alpha; output bin h samples input bin h / beta.
        ti = w[None, :] / alpha[:, None]  # [B, T_out]
        fi = h[None, :] / beta[:, None]  # [B, D]
        gx = 2.0 * ti / t_den - 1.0  # normalized x (time/width)
        gy = 2.0 * fi / d_den - 1.0  # normalized y (freq/height)
        grid_x = gx[:, None, :].expand(B, D, T_out)
        grid_y = gy[:, :, None].expand(B, D, T_out)
        grid = torch.stack((grid_x, grid_y), dim=-1)  # [B, D, T_out, 2]

        inp = input_spec.unsqueeze(1).to(torch.float32)  # [B, 1, D, T]
        out = torch.nn.functional.grid_sample(
            inp, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        out = out.squeeze(1).to(input_spec.dtype)  # [B, D, T_out]
        return out, new_length


class MaskedPatchAugmentation(NeuralModule):
    """
    Zeroes out fixed size time patches of the spectrogram.
    All samples in batch are guaranteed to have the same amount of masked time steps.
    Optionally also performs frequency masking in the same way as SpecAugment.
    Args:
        patch_size (int): up to how many time steps does one patch consist of.
            Defaults to 48.
        mask_patches (float): how many patches should be masked in each sample.
            if >= 1., interpreted as number of patches (after converting to int)
            if <1.,   interpreted as fraction of total tokens to be masked (number of patches is rounded up)
            Defaults to 10.
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in a segment.
            Defaults to 0.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        patch_size: int = 48,
        mask_patches: float = 10.0,
        freq_masks: int = 0,
        freq_width: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        if mask_patches >= 1:
            self.mask_patches = int(mask_patches)
        elif mask_patches >= 0:
            self._mask_fraction = mask_patches
            self.mask_patches = None
        else:
            raise ValueError('mask_patches cannot be negative')

        if freq_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=0,
                freq_width=freq_width,
                time_width=0,
            )
        else:
            self.spec_augment = None

    @typecheck()
    def forward(self, input_spec, length):
        augmented_spec = input_spec

        min_len = torch.min(length)

        if self.mask_patches is None:
            # masking specified as fraction
            len_fraction = int(min_len * self._mask_fraction)
            mask_patches = len_fraction // self.patch_size + int(len_fraction % self.patch_size != 0)
        else:
            mask_patches = self.mask_patches

        if min_len < self.patch_size * mask_patches:
            mask_patches = min_len // self.patch_size

        for idx in range(input_spec.shape[0]):
            cur_len = length[idx]
            patches = range(cur_len // self.patch_size)
            masked_patches = random.sample(patches, mask_patches)

            for mp in masked_patches:
                augmented_spec[idx, :, mp * self.patch_size : (mp + 1) * self.patch_size] = 0.0

        if self.spec_augment is not None:
            augmented_spec = self.spec_augment(input_spec=augmented_spec, length=length)

        return augmented_spec


class CropOrPadSpectrogramAugmentation(NeuralModule):
    """
    Pad or Crop the incoming Spectrogram to a certain shape.

    Args:
        audio_length (int): the final number of timesteps that is required.
            The signal will be either padded or cropped temporally to this
            size.
    """

    def __init__(self, audio_length):
        super(CropOrPadSpectrogramAugmentation, self).__init__()
        self.audio_length = audio_length

        if self.audio_length < 0:
            raise ValueError(
                'audio_length must be non-negative. If using a dataclass with OmegaConf, '
                'please call OmegaConf.to_object(cfg) to call appropriate __post_init__ methods.'
            )

    @typecheck()
    @torch.no_grad()
    def forward(self, input_signal, length):
        image = input_signal
        num_images = image.shape[0]

        audio_length = self.audio_length
        image_len = image.shape[-1]

        # Crop long signal
        if image_len > audio_length:  # randomly slice
            cutout_images = []
            offset = torch.randint(low=0, high=image_len - audio_length + 1, size=[num_images])

            for idx, offset in enumerate(offset):
                cutout_images.append(image[idx : idx + 1, :, offset : offset + audio_length])

            image = torch.cat(cutout_images, dim=0)
            del cutout_images

        else:  # symmetrically pad short signal with zeros
            pad_left = (audio_length - image_len) // 2
            pad_right = (audio_length - image_len) // 2

            if (audio_length - image_len) % 2 == 1:
                pad_right += 1

            image = torch.nn.functional.pad(image, [pad_left, pad_right], mode="constant", value=0)

        # Replace dynamic length sequences with static number of timesteps
        length = (length * 0) + audio_length

        return image, length

    @property
    def input_types(self):
        """Returns definitions of module output ports."""
        return {
            "input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass


@dataclass
class AudioToMelSpectrogramPreprocessorConfig:
    _target_: str = "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor"
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    n_window_size: Optional[int] = None
    n_window_stride: Optional[int] = None
    window: str = "hann"
    normalize: str = "per_feature"
    n_fft: Optional[int] = None
    preemph: float = 0.97
    features: int = 64
    lowfreq: int = 0
    highfreq: Optional[int] = None
    log: bool = True
    log_zero_guard_type: str = "add"
    log_zero_guard_value: float = 2**-24
    dither: float = 1e-5
    pad_to: int = 16
    frame_splicing: int = 1
    exact_pad: bool = False
    pad_value: int = 0
    mag_power: float = 2.0
    rng: Optional[str] = None
    nb_augmentation_prob: float = 0.0
    nb_max_freq: int = 4000
    mel_norm: str = "slaney"
    use_torchaudio: bool = False  # Deprecated argument, kept for compatibility with older checkpoints.
    stft_exact_pad: bool = False  # Deprecated argument, kept for compatibility with older checkpoints.
    stft_conv: bool = False  # Deprecated argument, kept for compatibility with older checkpoints.


@dataclass
class AudioToMFCCPreprocessorConfig:
    _target_: str = 'nemo.collections.asr.modules.AudioToMFCCPreprocessor'
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    n_window_size: Optional[int] = None
    n_window_stride: Optional[int] = None
    window: str = 'hann'
    n_fft: Optional[int] = None
    lowfreq: Optional[float] = 0.0
    highfreq: Optional[float] = None
    n_mels: int = 64
    n_mfcc: int = 64
    dct_type: int = 2
    norm: str = 'ortho'
    log: bool = True


@dataclass
class SpectrogramAugmentationConfig:
    _target_: str = "nemo.collections.asr.modules.SpectrogramAugmentation"
    freq_masks: int = 0
    time_masks: int = 0
    freq_width: int = 0
    time_width: Optional[Any] = 0
    rect_masks: int = 0
    rect_time: int = 0
    rect_freq: int = 0
    mask_value: float = 0
    rng: Optional[Any] = None  # random.Random() type
    use_numba_spec_augment: bool = False
    use_vectorized_spec_augment: bool = True


@dataclass
class CropOrPadSpectrogramAugmentationConfig:
    audio_length: int
    _target_: str = "nemo.collections.asr.modules.CropOrPadSpectrogramAugmentation"


@dataclass
class MaskedPatchAugmentationConfig:
    patch_size: int = 48
    mask_patches: float = 10.0
    freq_masks: int = 0
    freq_width: int = 0
    _target_: str = "nemo.collections.asr.modules.MaskedPatchAugmentation"
