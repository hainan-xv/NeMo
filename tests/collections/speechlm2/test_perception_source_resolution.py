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

"""Resolution of a perception front-end against a pretrained ASR donor.

The case that motivated this file: a convolution-only SCRIPT encoder warm-started
from Citrinet. Donor and user ``_target_`` are then the SAME class, so the
"replace wholesale" branch does not fire -- and the merge branch used to push
``att_context_size`` (a Conformer-only key) into ``ConvASREncoder.__init__``.
"""

import pytest
from omegaconf import OmegaConf

from nemo.collections.speechlm2.parts.pretrained import resolve_perception_sources

CONFORMER = "nemo.collections.asr.modules.ConformerEncoder"
CONVASR = "nemo.collections.asr.modules.ConvASREncoder"


def _asr_cfg(target, **enc):
    return OmegaConf.create(
        {"preprocessor": {"_target_": "pre", "normalize": "per_feature"}, "encoder": {"_target_": target, **enc}}
    )


def test_matching_conformer_target_receives_att_context_size():
    p = OmegaConf.create({"encoder": {"_target_": CONFORMER, "n_layers": 3}})
    resolve_perception_sources(p, _asr_cfg(CONFORMER, n_layers=17, att_context_size=[70, 13]), [84, 1])
    assert p.encoder.att_context_size == [84, 1]
    assert p.encoder.n_layers == 3  # user key still wins


def test_matching_conv_target_does_not_receive_att_context_size():
    """ConvASREncoder has no attention; the donor declares no look-ahead to override."""
    p = OmegaConf.create({"encoder": {"_target_": CONVASR, "feat_in": 80}})
    resolve_perception_sources(p, _asr_cfg(CONVASR, feat_in=80, jasper=[]), [84, 1])
    assert "att_context_size" not in p.encoder


def test_differing_target_replaces_wholesale():
    p = OmegaConf.create({"encoder": {"_target_": CONVASR, "feat_in": 80}})
    resolve_perception_sources(p, _asr_cfg(CONFORMER, n_layers=17, att_context_size=[70, 13]), [84, 1])
    assert p.encoder._target_ == CONVASR
    assert "att_context_size" not in p.encoder
    assert "n_layers" not in p.encoder


def test_user_preprocessor_keys_survive():
    p = OmegaConf.create({"preprocessor": {"normalize": "NA"}, "encoder": {"_target_": CONVASR}})
    resolve_perception_sources(p, _asr_cfg(CONVASR), None)
    assert p.preprocessor.normalize == "NA"
