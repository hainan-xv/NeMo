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
"""Vendored subset of the Open ASR Leaderboard text normalizer.

Mirrors github.com/huggingface/open_asr_leaderboard/normalizer so our leaderboard
eval scores with the SAME normalization the public board uses (as of 2026-08),
which diverged from the stock ``whisper_normalizer`` we use for training val_wer:
an expanded disfluency list, acronym de-spacing ("b b c" -> "bbc"), name-variant
folding, and compound-word joining ("wi fi" -> "wifi").

Only the English path is vendored (the leaderboard suite we run is English). The
heavy, well-tested Whisper number/spelling logic is reused from the installed
``whisper_normalizer`` package rather than re-vendored -- see normalizer.py.
"""
from .normalizer import EnglishTextNormalizer

__all__ = ["EnglishTextNormalizer"]
