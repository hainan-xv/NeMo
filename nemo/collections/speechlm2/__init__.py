# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from .data import StreamingSTTDataset
from .models import StreamingSTTModel

# Keep Streaming STT importable even when optional duplex/TTS dependencies
# are unavailable in lightweight environments (e.g. import smoke checks).
try:
    from .data import DataModule, DuplexEARTTSDataset, DuplexS2SDataset, DuplexSTTDataset, SALMDataset
    from .models import (
        SALM,
        DuplexEARTTS,
        DuplexS2SModel,
        DuplexS2SSpeechDecoderModel,
        DuplexSTTModel,
        NemotronVoiceChat,
        SALMWithAsrDecoder,
    )
except (ImportError, AttributeError):
    DataModule = None
    DuplexS2SDataset = None
    DuplexSTTDataset = None
    DuplexEARTTSDataset = None
    SALMDataset = None
    DuplexEARTTS = None
    DuplexS2SModel = None
    DuplexS2SSpeechDecoderModel = None
    DuplexSTTModel = None
    SALM = None
    SALMWithAsrDecoder = None
    NemotronVoiceChat = None

__all__ = [
    'DataModule',
    'DuplexS2SDataset',
    'DuplexSTTDataset',
    'DuplexEARTTSDataset',
    'SALMDataset',
    'StreamingSTTDataset',
    'DuplexEARTTS',
    'DuplexS2SModel',
    'DuplexS2SSpeechDecoderModel',
    'DuplexSTTModel',
    'SALM',
    'SALMWithAsrDecoder',
    'NemotronVoiceChat',
    'StreamingSTTModel',
]
