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
"""The WER-delta panels: corrected MINUS chat, both teacher-forced.

NEGATIVE means the corrector is winning. The metric exists because the effect is
a couple of WER points on top of rates around 0.05, which two overlaid curves
hide. The methods are exercised against a stand-in object rather than a real
model: they touch only ``self._val`` / ``self._last`` / ``self.log``, and
building a Qwen+Conformer to check subtraction would be absurd.
"""

import types

import pytest

from nemo.collections.speechlm2.models.script_corrector_model import (
    _METRIC_KEYS,
    _VAL_METRIC_KEYS,
    ScriptCorrectorModel,
)


def _recorder():
    seen = {}

    def log(k, v, **kw):
        seen[k] = v

    return seen, log


@pytest.mark.unit
def test_val_delta_is_registered_for_rank_uniform_logging():
    """Absent from the tuple it would never be logged; logged conditionally it
    would desync the sync_dist collective."""
    assert "val_oracle_gain" in _VAL_METRIC_KEYS


@pytest.mark.unit
def test_train_delta_is_NOT_in_the_every_step_key_set():
    """train_corrected_wer only exists every N steps; forcing the delta into the
    every-step set would post 0.0 between recomputations and saw-tooth the panel."""
    assert "train_oracle_gain" not in _METRIC_KEYS


@pytest.mark.unit
def test_val_delta_differences_accumulated_counts():
    seen, log = _recorder()
    obj = types.SimpleNamespace(
        _val={
            "pred": [],
            "label": [],
            "chat_e": 10,
            "chat_n": 100,
            "corr_e": 4,
            "corr_n": 100,
            "gen_e": 0,
            "gen_n": 0,
            "gen_utts": 0,
        },
        log=log,
    )
    ScriptCorrectorModel.on_validation_epoch_end(obj)
    assert seen["val_chat_wer_tf"] == pytest.approx(0.10)
    assert seen["val_oracle_wer"] == pytest.approx(0.04)
    assert seen["val_oracle_gain"] == pytest.approx(-0.06)


@pytest.mark.unit
def test_val_delta_is_corpus_weighted_not_a_mean_of_batches():
    """Two batches, one long and one short: the delta must follow total edits
    over total words, so the long batch dominates."""
    seen, log = _recorder()
    # batch A: 990 words, chat 99 errors, corrected 99 (no gain)
    # batch B:  10 words, chat  1 error,  corrected  0 (total gain)
    obj = types.SimpleNamespace(
        _val={
            "pred": [],
            "label": [],
            "chat_e": 100,
            "chat_n": 1000,
            "corr_e": 99,
            "corr_n": 1000,
            "gen_e": 0,
            "gen_n": 0,
            "gen_utts": 0,
        },
        log=log,
    )
    ScriptCorrectorModel.on_validation_epoch_end(obj)
    # A mean of per-batch deltas would be about (0 + -0.1)/2 = -0.05.
    assert seen["val_oracle_gain"] == pytest.approx(-0.001)


@pytest.mark.unit
def test_val_delta_survives_zero_denominators():
    seen, log = _recorder()
    obj = types.SimpleNamespace(
        _val={
            "pred": [],
            "label": [],
            "chat_e": 0,
            "chat_n": 0,
            "corr_e": 0,
            "corr_n": 0,
            "gen_e": 0,
            "gen_n": 0,
            "gen_utts": 0,
        },
        log=log,
    )
    ScriptCorrectorModel.on_validation_epoch_end(obj)
    assert seen["val_oracle_gain"] == pytest.approx(0.0)


def _train_obj(log, *, corrected, chat, raises=False):
    def _cw(last):
        if raises:
            raise RuntimeError("boom")
        return corrected

    return types.SimpleNamespace(
        core_cfg=types.SimpleNamespace(corrected_wer_every_n_steps=200),
        global_step=200,
        _last={"stub": True},
        _last_chat_wer=chat,
        _corrected_wer=_cw,
        log=log,
    )


@pytest.mark.unit
def test_train_delta_pairs_the_same_batch():
    seen, log = _recorder()
    ScriptCorrectorModel.on_train_batch_end(_train_obj(log, corrected=0.03, chat=0.05), None, None, 0)
    assert seen["train_oracle_wer"] == pytest.approx(0.03)
    assert seen["train_oracle_gain"] == pytest.approx(-0.02)


@pytest.mark.unit
def test_train_delta_sign_is_positive_when_the_corrector_hurts():
    seen, log = _recorder()
    ScriptCorrectorModel.on_train_batch_end(_train_obj(log, corrected=0.08, chat=0.05), None, None, 0)
    assert seen["train_oracle_gain"] == pytest.approx(0.03)


@pytest.mark.unit
def test_failed_corrected_wer_posts_no_spurious_gain():
    """w defaults to 0.0 on failure, so a naive w - chat would report a huge win
    exactly when the measurement broke."""
    seen, log = _recorder()
    ScriptCorrectorModel.on_train_batch_end(_train_obj(log, corrected=0.0, chat=0.05, raises=True), None, None, 0)
    assert seen["train_oracle_wer"] == pytest.approx(0.0)
    assert seen["train_oracle_gain"] == pytest.approx(0.0)


@pytest.mark.unit
def test_no_delta_logged_on_a_non_measurement_step():
    """The branch is rank-uniform on global_step; off-step it must log NEITHER."""
    seen, log = _recorder()
    obj = _train_obj(log, corrected=0.03, chat=0.05)
    obj.global_step = 199
    ScriptCorrectorModel.on_train_batch_end(obj, None, None, 0)
    assert seen == {}


@pytest.mark.unit
def test_stash_is_cleared_after_each_batch():
    seen, log = _recorder()
    obj = _train_obj(log, corrected=0.03, chat=0.05)
    ScriptCorrectorModel.on_train_batch_end(obj, None, None, 0)
    assert obj._last is None and obj._last_chat_wer is None


# --------------------------------------------------------------------------
# The oracle metric CANNOT report failure; the real one can. Pin both.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_the_real_metric_is_logged():
    assert "val_gen_wer" in _VAL_METRIC_KEYS and "val_gen_delta" in _VAL_METRIC_KEYS


@pytest.mark.unit
def test_oracle_and_real_are_separately_named():
    """They were the same key. The oracle substitutes the reference and so can
    only improve; conflating them is what hid a real regression."""
    assert "val_oracle_wer" in _VAL_METRIC_KEYS
    assert "val_corrected_wer" not in _VAL_METRIC_KEYS, "the ambiguous name must be gone"


@pytest.mark.unit
def test_gen_delta_can_be_POSITIVE():
    """The whole point: unlike the oracle gain, this one can report that the
    corrector made things worse."""
    seen, log = _recorder()
    obj = types.SimpleNamespace(
        _val={
            "pred": [],
            "label": [],
            "chat_e": 10,
            "chat_n": 100,  # chat 0.10
            "corr_e": 4,
            "corr_n": 100,  # oracle 0.04  -> gain -0.06
            "gen_e": 15,
            "gen_n": 100,  # generated 0.15 -> delta +0.05
            "gen_utts": 5,
        },
        log=log,
    )
    ScriptCorrectorModel.on_validation_epoch_end(obj)
    assert seen["val_oracle_gain"] == pytest.approx(-0.06), "oracle still looks good"
    assert seen["val_gen_delta"] == pytest.approx(0.05), "real metric reports the regression"
    assert seen["val_gen_wer"] == pytest.approx(0.15)


@pytest.mark.unit
def test_checkpoints_select_on_the_real_metric():
    import pathlib

    import yaml

    c = yaml.safe_load(
        (
            pathlib.Path(__file__).parents[3]
            / "examples/speechlm2/conf/streaming_stt_granary2_lora_script_corrector.yaml"
        ).read_text()
    )
    k = c["exp_manager"]["checkpoint_callback_params"]
    assert k["monitor"] == "val_gen_wer", "selecting on the oracle ranks by a number that cannot fail"
    assert "val_gen_wer" in k["filename"]
