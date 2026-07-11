# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTBatch
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
from nemo.collections.speechlm2.parts.alignments import WordAlignment


@pytest.mark.parametrize(
    "chunk_size,configured,expected",
    [
        (14, None, 14),
        (14, 20, 20),
        (0, None, 64),
        (-1, None, 64),
    ],
)
def test_val_max_new_tokens_per_chunk(chunk_size, configured, expected):
    mock = SimpleNamespace(
        core_cfg=SimpleNamespace(
            chunk_size=chunk_size,
            val_max_new_tokens_per_chunk=configured,
        )
    )
    assert StreamingSTTModel.val_max_new_tokens_per_chunk.__get__(mock) == expected


def test_val_max_new_tokens_must_be_positive():
    mock = SimpleNamespace(
        core_cfg=SimpleNamespace(
            chunk_size=14,
            val_max_new_tokens_per_chunk=0,
        )
    )
    with pytest.raises(ValueError, match="must be positive"):
        StreamingSTTModel.val_max_new_tokens_per_chunk.__get__(mock)


def test_validation_epoch_logs_corpus_wer_not_accuracy():
    logged = {}
    mock = SimpleNamespace(
        _partial_wer_refs={"clean": ["hello world"]},
        _partial_wer_hyps={"clean": ["hello"]},
        device=torch.device("cpu"),
        log=lambda name, value, **kwargs: logged.__setitem__(name, float(value)),
    )

    StreamingSTTModel.on_validation_epoch_end(mock)

    assert logged["val_wer_clean"] == pytest.approx(0.5)
    assert logged["val_wer"] == pytest.approx(0.5)
    assert "val_loss" not in logged
    assert not any(name.startswith("val_acc") for name in logged)


def test_train_autoregressive_preview_decodes_first_sample_once_per_step():
    calls = []
    logged = {}
    module = torch.nn.Linear(1, 1).train()
    mock = SimpleNamespace(
        core_cfg=SimpleNamespace(train_decode_every_n_steps=100),
        trainer=SimpleNamespace(global_step=100, is_global_zero=True),
        device=torch.device("cpu"),
        train_decode_max_new_tokens_per_chunk=14,
        dataset=SimpleNamespace(
            cfg=SimpleNamespace(
                system_role="system",
                audio_tag="<audio>",
                blank_token="<blank>",
                chunk_size=2,
                num_delay_frames=0,
                sample_rate=16,
                frame_length_in_secs=0.5,
                words_per_group=1,
                chunk_step=1,
            )
        ),
        _latest_train_alignments=[
            [
                WordAlignment("reference", 0.0, 0.5),
                WordAlignment("one", 1.0, 1.5),
            ]
        ],
        modules=lambda: [module],
        eval=module.eval,
        _validation_system_prompts=lambda batch: ["prompt one", "prompt two"],
        generate=lambda **kwargs: calls.append(kwargs) or ["decoded | hypothesis"],
        log=lambda name, value, **kwargs: logged.__setitem__(name, float(value)),
    )
    batch = StreamingSTTBatch(
        audios=torch.zeros(2, 32),
        audio_lens=torch.tensor([32, 24]),
        text=["reference one", "reference two"],
    )

    StreamingSTTModel.on_train_batch_end(mock, outputs=None, batch=batch, batch_idx=0)
    StreamingSTTModel.on_train_batch_end(mock, outputs=None, batch=batch, batch_idx=1)

    assert len(calls) == 1
    assert calls[0]["audios"].shape[0] == 1
    assert calls[0]["audio_lens"].tolist() == [32]
    assert calls[0]["system_prompt"] == "prompt one"
    assert calls[0]["max_new_tokens"] == 14
    assert calls[0]["chunk_separator"] == "|"
    assert logged["training_wer"] == pytest.approx(1.0)
    assert module.training


def test_training_reference_uses_aligned_chunk_boundaries():
    mock = SimpleNamespace(
        dataset=SimpleNamespace(
            cfg=SimpleNamespace(
                system_role="system",
                audio_tag="<audio>",
                blank_token="<blank>",
                chunk_size=2,
                num_delay_frames=0,
                sample_rate=16,
                frame_length_in_secs=0.5,
                words_per_group=1,
                chunk_step=1,
            )
        ),
        _latest_train_alignments=[
            [
                WordAlignment("hello", 0.0, 0.5),
                WordAlignment("world", 1.0, 1.5),
            ]
        ],
        _validation_system_prompts=lambda batch: ["prompt"],
    )
    batch = StreamingSTTBatch(
        audios=torch.zeros(1, 48),
        audio_lens=torch.tensor([48]),
        text=["hello world"],
    )

    chunked_ref = StreamingSTTModel._training_reference_with_chunk_boundaries(mock, batch)

    assert chunked_ref == "hello | world | "
