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
"""The full-context CHAT arm must match its donor encoder, field for field.

``init_from_nemo_model`` matches tensors by NAME and SHAPE and skips what does
not fit. A config that disagrees with the donor therefore does not fail -- it
loads a subset and reports success, and the arm trains from a half-random
encoder while looking warm-started. Two fields do this silently:

  conv_norm_type  batch_norm vs layer_norm renames and reshapes every Conformer
                  conv module's normalisation parameters.
  xscaling        not a weight at all, so nothing mismatches -- it just scales
                  activations by sqrt(d_model) away from where the donor trained.

The values below are parakeet-tdt-0.6b-v2's own, read from its model_config.yaml.
"""

import pathlib

import pytest
import yaml

_CONF = (
    pathlib.Path(__file__).parents[3]
    / "examples/asr/conf/fastconformer/cache_aware_streaming"
    / "nemotron_chat_transducer_granary2_qwen_fullctx.yaml"
)

# parakeet-tdt-0.6b-v2 model_config.yaml, encoder block.
PARAKEET_ENCODER = {
    "n_layers": 24,
    "d_model": 1024,
    "n_heads": 8,
    "subsampling": "dw_striding",
    "subsampling_factor": 8,
    "subsampling_conv_channels": 256,
    "conv_kernel_size": 9,
    "ff_expansion_factor": 4,
    "self_attention_model": "rel_pos",
    "pos_emb_max_len": 5000,
    "untie_biases": True,
    "use_bias": False,
    "conv_norm_type": "batch_norm",
    "xscaling": False,
    "causal_downsampling": False,
    "att_context_style": "regular",
    "conv_context_size": None,
}


@pytest.fixture(scope="module")
def cfg():
    assert _CONF.is_file(), f"missing config: {_CONF}"
    return yaml.safe_load(_CONF.read_text())["model"]


@pytest.mark.unit
@pytest.mark.parametrize("field,expected", sorted(PARAKEET_ENCODER.items(), key=lambda kv: kv[0]))
def test_encoder_field_matches_the_parakeet_donor(cfg, field, expected):
    got = cfg["encoder"].get(field)
    assert got == expected, f"encoder.{field}={got!r} but the donor has {expected!r}; weights will not transfer"


@pytest.mark.unit
def test_attention_is_unlimited(cfg):
    assert cfg["encoder"]["att_context_size"] == [-1, -1]


@pytest.mark.unit
def test_chunk_size_is_explicit(cfg):
    """It is normally inferred as right_context + 1 from att_context_size.
    att_context_size is [-1,-1] here, so inference raises -- the joint must say
    14 itself, and 14 is what keeps this arm comparable to the streaming ones."""
    assert cfg["joint"]["chunk_size"] == 14


@pytest.mark.unit
def test_feature_count_matches_the_donor(cfg):
    """The donor's first conv layer is shaped by this; 80 vs 128 would silently
    drop the subsampling stack from the warm start."""
    assert cfg["preprocessor"]["features"] == 128


@pytest.mark.unit
def test_the_streaming_sibling_is_left_alone(cfg):
    """This arm is a COPY. If the edits had landed on the shared streaming config
    instead, every other CHAT arm would silently change objective."""
    sib = _CONF.with_name("nemotron_chat_transducer_granary2_qwen.yaml")
    s = yaml.safe_load(sib.read_text())["model"]
    assert s["encoder"]["att_context_size"] == [70, 13]
    assert s["encoder"]["conv_norm_type"] == "layer_norm"
    assert s["encoder"]["causal_downsampling"] is True
    assert "chunk_size" not in s["joint"], "the streaming arm must keep INFERRING chunk_size"
