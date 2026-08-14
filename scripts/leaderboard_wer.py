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
"""WER scorer matching the Open ASR Leaderboard's English scoring (2026-08).

Two things differ from our training-time ``val_wer`` (nemo speechlm2 ``WER``),
and both are reproduced here so leaderboard-eval numbers line up with the board:

1. Normalization: the vendored leaderboard ``EnglishTextNormalizer``
   (scripts/leaderboard_normalizer) instead of stock ``whisper_normalizer`` --
   expanded fillers, acronym de-spacing, name folding, compound joining.
2. Edit distance: ``kaldialign.batch_error_rate(..., merge_compounds=True)``
   instead of plain word error rate, so split compounds ("white paper" vs
   "whitepaper") count as 0 errors either way.

Drop-in for the eval driver: same ``WER(normalize=..., verbose=...)`` ctor,
``update(name, refs=, hyps=)`` and ``compute() -> {"wer": <fraction>}`` surface
(the driver multiplies by 100), so swapping it in touches only the import.
"""
from collections import defaultdict
from difflib import SequenceMatcher

_WARNED_FALLBACK = False


def _identity(s: str) -> str:
    return s


def normalize_compound_pairs(refs, preds):
    """Align compound word boundaries between ref/pred pairs (vendored verbatim
    from open_asr_leaderboard/normalizer/eval_utils.py).

    When a mismatch region has identical characters ignoring whitespace, normalize
    both sides to the joined form. Used only by the fallback path below to
    approximate kaldialign's ``merge_compounds`` when that API is unavailable.
    """
    new_refs, new_preds = [], []
    for ref_text, pred_text in zip(refs, preds):
        ref_words = ref_text.split()
        pred_words = pred_text.split()
        sm = SequenceMatcher(None, ref_words, pred_words)
        new_rw, new_pw = [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                new_rw.extend(ref_words[i1:i2])
                new_pw.extend(pred_words[j1:j2])
            else:
                rc = "".join(ref_words[i1:i2])
                pc = "".join(pred_words[j1:j2])
                if rc == pc:
                    new_rw.append(rc)
                    new_pw.append(pc)
                else:
                    new_rw.extend(ref_words[i1:i2])
                    new_pw.extend(pred_words[j1:j2])
        new_refs.append(" ".join(new_rw))
        new_preds.append(" ".join(new_pw))
    return new_refs, new_preds


class LeaderboardWER:
    """Leaderboard-faithful WER. Interface-compatible with speechlm2's ``WER``."""

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = False):
        if normalize:
            if normalizer is None:
                from leaderboard_normalizer import EnglishTextNormalizer

                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity
        self.verbose = verbose
        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def update(self, name: str, refs, hyps) -> None:
        for ref, hyp in zip(refs, hyps):
            self._refs[name].append(self.normalizer(ref))
            self._hyps[name].append(self.normalizer(hyp))

    @staticmethod
    def _batch_error_rate(refs, hyps):
        """Corpus error rate matching the leaderboard.

        Primary path (faithful): ``kaldialign.batch_error_rate(merge_compounds=True)``
        -- the exact call the board uses. The eval launcher upgrades kaldialign so
        this is what runs on the cluster.

        Fallback (only if that API is unavailable, e.g. an old kaldialign that
        couldn't be upgraded offline): approximate compound merging with the
        vendored ``normalize_compound_pairs`` and score with NeMo's
        ``word_error_rate``. This yields near-identical numbers but is flagged with
        a one-time warning so a silent divergence can't slip by.
        """
        global _WARNED_FALLBACK
        batch_error_rate = None
        try:
            from kaldialign import batch_error_rate  # noqa: F811
        except (ImportError, AttributeError):
            batch_error_rate = None

        if batch_error_rate is not None:
            refs_split = [tuple(r.split()) for r in refs]
            hyps_split = [tuple(h.split()) for h in hyps]
            try:
                r = batch_error_rate(refs_split, hyps_split, merge_compounds=True)
            except TypeError:
                # Present but older signature without merge_compounds.
                r = batch_error_rate(refs_split, hyps_split)
            return float(r["err_rate"])

        # ---- Fallback ----
        if not _WARNED_FALLBACK:
            print(
                "[leaderboard_wer] WARNING: kaldialign.batch_error_rate unavailable; "
                "using normalize_compound_pairs + NeMo word_error_rate (numbers may "
                "differ slightly from the board). Upgrade kaldialign to fix.",
                flush=True,
            )
            _WARNED_FALLBACK = True
        from nemo.collections.asr.metrics.wer import word_error_rate

        merged_refs, merged_hyps = normalize_compound_pairs(refs, hyps)
        return float(word_error_rate(hypotheses=merged_hyps, references=merged_refs))

    def compute(self):
        """Return ``{"wer": corpus_fraction, "wer_<name>": per-dataset_fraction}``.

        Corpus WER is computed over the POOLED refs/hyps across all names (additive
        over utterances -- the correct way to combine, not an average of per-name
        rates). ``verbose`` prints a few normalized ref/hyp pairs for spot-checking.
        """
        all_refs, all_hyps = [], []
        out = {}
        for name in self._refs:
            refs = self._refs[name]
            hyps = self._hyps[name]
            all_refs.extend(refs)
            all_hyps.extend(hyps)
            if refs:
                out[f"wer_{name}"] = self._batch_error_rate(refs, hyps)
                if self.verbose:
                    for r, h in list(zip(refs, hyps))[:3]:
                        print(f"[{name}] ref: {r!r}\n[{name}] hyp: {h!r}", flush=True)
        out["wer"] = self._batch_error_rate(all_refs, all_hyps) if all_refs else 0.0
        return out


# Export under the name the eval driver imports.
WER = LeaderboardWER
