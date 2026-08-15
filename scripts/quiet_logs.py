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
"""Console helpers for the local Parakeet eval scripts: one-call log quieting plus a
tqdm ``total`` shim for NeMo's transcribe bar.

``silence()`` -- the tqdm progress bar and NeMo's warnings both go to stderr, so a
shell pipe can't drop the warnings without also mangling the (in-place) bar. Instead
we silence the noisy loggers IN-PROCESS, which leaves the bar and our own ``print``
output intact. The one non-obvious bit: ``model.transcribe()`` force-lowers the NeMo
verbosity to WARNING for its duration (and Lightning sets its own level at import),
so a plain ``set_verbosity(ERROR)`` gets undone -> per-batch dataloader spam. A
logging Filter runs regardless of a logger's level, so we pin a hard "ERROR floor"
filter that drops every sub-ERROR record no matter what the level is later set to.

``add_transcribe_total()`` -- NeMo wraps a length-less Lhotse dataloader in tqdm with
no ``total``, so its bar only shows it/s, not overall progress. We inject the total.
"""
import logging
import os

# Loggers that emit the once-per-process noise (Lightning "GPU/TPU/HPU available",
# nv_one_logger telemetry, captured python warnings). Filtering the exact emitters
# works even when they set their own level at import (ancestor filters don't apply
# to records propagated up from children).
_NOISY = (
    "lightning",
    "lightning.pytorch",
    "lightning.pytorch.utilities.rank_zero",
    "lightning_fabric",
    "pytorch_lightning",
    "pytorch_lightning.utilities.rank_zero",
    "nv_one_logger",
    "py.warnings",
)


class _ErrorFloor(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib Filter API
        return record.levelno >= logging.ERROR


def silence() -> None:
    """Drop all sub-ERROR log output (keeps tqdm bars + explicit prints)."""
    import warnings

    warnings.filterwarnings("ignore")

    floor = _ErrorFloor()
    for name in _NOISY:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.addFilter(floor)

    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.set_verbosity(nemo_logging.ERROR)
        base = getattr(nemo_logging, "_logger", None)
        if base is not None:
            base.addFilter(floor)
    except Exception:  # noqa: BLE001 - best-effort; nemo may not be importable yet
        pass


def add_transcribe_total(manifest_path: str, batch_size: int) -> None:
    """Give NeMo's 'Transcribing' tqdm bar a ``total`` so it shows overall progress.

    transcribe() builds a fixed-batch, ``shuffle=False`` dataloader (no dynamic
    bucketing), so the number of batches is exactly ``ceil(#utts / batch_size)``.
    We read the utterance count from the manifest we're about to decode and inject
    that total into the bar via a thin ``tqdm`` shim in the transcription mixin.
    """
    import math

    try:
        if not manifest_path or not os.path.isfile(manifest_path) or batch_size <= 0:
            return
        with open(manifest_path, encoding="utf-8", errors="replace") as f:
            n_utts = sum(1 for line in f if line.strip())
        total = math.ceil(n_utts / batch_size)

        from nemo.collections.asr.parts.mixins import transcription as _tr

        _orig_tqdm = _tr.tqdm

        def _tqdm_with_total(iterable=None, *args, **kwargs):
            kwargs.setdefault("total", total)
            kwargs.setdefault("unit", "batch")
            return _orig_tqdm(iterable, *args, **kwargs)

        _tr.tqdm = _tqdm_with_total
    except Exception:  # noqa: BLE001 - progress total is best-effort cosmetics
        pass
