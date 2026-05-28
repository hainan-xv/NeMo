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
from .streaming_stt_dataset import StreamingSTTDataset

# Keep Streaming STT dataset importable even when optional TTS deps are absent.
try:
    from .datamodule import DataModule
    from .duplex_ear_tts_dataset import DuplexEARTTSDataset
    from .duplex_stt_dataset import DuplexSTTDataset
    from .s2s_dataset import DuplexS2SDataset
    from .salm_dataset import SALMDataset
except (ImportError, AttributeError):
    DataModule = None
    DuplexS2SDataset = None
    DuplexSTTDataset = None
    DuplexEARTTSDataset = None
    SALMDataset = None

__all__ = [
    'DataModule',
    'DuplexS2SDataset',
    'DuplexSTTDataset',
    'DuplexEARTTSDataset',
    'SALMDataset',
    'StreamingSTTDataset',
]
