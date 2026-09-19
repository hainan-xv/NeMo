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
"""The purpose-built-vocabulary CHAT arms (8k / 16k / 32k).

These replace Qwen3's 151.7k multilingual LLM vocabulary with SentencePiece
vocabularies trained on the Granary v2 transcripts. Everything else is held at
dfw_chat_banded1_both_nodelay_v2's settings, so the arms isolate vocabulary size
-- and these tests pin the two things that would silently break that.
"""

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).parents[3]
CONF = ROOT / "examples/asr/conf/fastconformer/cache_aware_streaming"
SPE = CONF / "nemotron_chat_transducer_granary2_spe.yaml"
QWEN = CONF / "nemotron_chat_transducer_granary2_qwen.yaml"
ARMS = {"8k": 8192, "16k": 16384, "32k": 32768}


@pytest.fixture(scope="module")
def spe():
    return yaml.safe_load(SPE.read_text())["model"]


@pytest.mark.unit
def test_tokenizer_routes_to_sentencepiece(spe):
    """chat_bpe_models._setup_tokenizer intercepts ONLY type: huggingface; any
    other value falls through to the ASR collection's SentencePiece path."""
    assert spe["tokenizer"]["type"] == "bpe"


@pytest.mark.unit
def test_objective_matches_the_reference_arm(spe):
    """The arms isolate VOCABULARY. If the loss or band settings drifted, a WER
    difference could not be attributed to the vocabulary at all."""
    q = yaml.safe_load(QWEN.read_text())["model"]
    assert spe["loss_type"] == q["loss_type"]
    for k in ("band_chunks", "num_delay_frames", "target_construction"):
        assert spe["forced_alignment"].get(k) == q["forced_alignment"].get(k), k
    assert spe["encoder"]["att_context_size"] == q["encoder"]["att_context_size"]


@pytest.mark.unit
@pytest.mark.parametrize("tag,size", sorted(ARMS.items()))
def test_launcher_points_at_its_own_vocabulary(tag, size):
    sh = (ROOT / f"launch/dfw_chat_spe{tag}_both.sh").read_text()
    assert f"granary2_en_spe/v{size}" in sh
    assert f"dfw_granary2_chat_spe{tag}_both" in sh


@pytest.mark.unit
@pytest.mark.parametrize("tag", sorted(ARMS))
def test_warm_start_is_encoder_only(tag):
    """decoder.* and joint.* are VOCABULARY-SHAPED. Loading them from a 151.7k
    Qwen arm into an 8k-32k model cannot work -- and init_from_nemo_model skips
    shape mismatches silently, so the tensors would stay at init while the load
    reported success."""
    sh = (ROOT / f"launch/dfw_chat_spe{tag}_both.sh").read_text()
    assert """INIT_INCLUDE='["encoder."]'""" in sh
    assert '"decoder."' not in sh and '"joint."' not in sh


@pytest.mark.unit
def test_vocab_builder_keeps_case_and_punctuation():
    """These models emit cased, punctuated text; a lowercased vocabulary would
    force every capital through a separate piece."""
    sh = (ROOT / "launch/dfw_build_spe_vocabs.sh").read_text()
    assert "--no_lower_case" in sh
    assert "8192 16384 32768" in sh


@pytest.mark.unit
def test_corpus_is_sampled_by_training_weight():
    """Uniform sampling would fit the vocabulary to a distribution the model
    never trains on -- corpus weights span three orders of magnitude."""
    src = (ROOT / "scripts/build_chat_spe_corpus.py").read_text()
    assert "weight" in src and "total_w" in src
    assert 'json.loads(line).get("text")' in src
