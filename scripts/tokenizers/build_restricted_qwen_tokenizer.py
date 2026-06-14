# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Build a restricted ("English-only") subset of a HuggingFace LLM (e.g. Qwen) tokenizer.

Two selection strategies are supported; both always also keep the 256 base byte tokens and all
special tokens so any string remains representable via byte fallback. The output is a JSON artifact
listing ``kept_ids`` (original HF token ids), consumed at training time by
``nemo.collections.common.tokenizers.huggingface.restricted_auto_tokenizer.RestrictedAutoTokenizer``
(``model.tokenizer.type=huggingface_restricted``), which exposes a compact, contiguous id space of
size ``len(kept_ids)`` to the model while keeping the original merge rules for encoding.

(A) SPE character-coverage mode (deterministic, corpus-free) -- pass ``--spe_vocab`` or ``--spe_model``:
    Keep every HF token whose decoded text uses only characters that appear in the base
    SentencePiece vocab (e.g. parakeet's 1024-piece SPE). This keeps the full English-character
    subword inventory the acoustic model already covers, while dropping the CJK/other-script bulk.
    Add ``--ascii_only`` to further restrict the character set to ASCII (drops Greek/Cyrillic/accented).

    Example:
        python scripts/tokenizers/build_restricted_qwen_tokenizer.py \
            --hf_model Qwen/Qwen3-1.7B \
            --spe_vocab /path/to/tokenizer.vocab \
            --output /results/qwen3_english_kept_ids.json

(B) Corpus-frequency mode -- pass ``--manifests`` / ``--text_files``:
    Tokenize the corpus and keep token ids seen at least ``--min_count`` times.

    Example:
        python scripts/tokenizers/build_restricted_qwen_tokenizer.py \
            --hf_model Qwen/Qwen3-1.7B \
            --manifests /data/.../mcv11_train.json \
            --text_files /data/.../extra_english.txt \
            --min_count 1 \
            --output /results/qwen3_english_kept_ids.json
"""

import argparse
import json
import os
from collections import Counter

from transformers import AutoTokenizer


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _spe_charset(spe_vocab=None, spe_model=None, ascii_only=False):
    """Collect the set of characters used by a SentencePiece vocab.

    The SentencePiece meta symbol U+2581 ('lower one eighth block') marks a leading space and is
    mapped back to a regular space. ``<...>`` control/special pieces (e.g. ``<unk>``) are skipped.
    """
    pieces = []
    if spe_vocab:
        with open(spe_vocab, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                pieces.append(line.split('\t')[0])
    elif spe_model:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.Load(spe_model)
        pieces = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]
    else:
        raise ValueError("Provide --spe_vocab or --spe_model for character-coverage mode")

    charset = set()
    for piece in pieces:
        if piece.startswith('<') and piece.endswith('>'):
            continue
        charset.update(piece.replace('\u2581', ' '))
    if ascii_only:
        charset = {c for c in charset if ord(c) < 128}
    return charset


def _build_byte_decoder():
    """Inverse of GPT-2/Qwen byte->unicode map: surface char -> raw byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _kept_ids_by_charset(tok, full_vocab, charset):
    """Keep every HF token whose decoded text uses only characters in ``charset``."""
    u2b = _build_byte_decoder()
    kept = set()
    n_fragment = 0
    for surf, idx in full_vocab.items():
        try:
            raw = bytes(u2b[c] for c in surf)
        except KeyError:
            # added/special token whose surface isn't in byte-unicode space; handled via specials
            continue
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            n_fragment += 1  # partial multi-byte fragment; not needed for in-charset text
            continue
        if text and all(ch in charset for ch in text):
            kept.add(idx)
    return kept, n_fragment


def _iter_texts(manifests, text_files, text_key):
    for mf in manifests or []:
        with open(mf, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = obj.get(text_key, None)
                if txt:
                    yield txt
    for tf in text_files or []:
        with open(tf, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line:
                    yield line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hf_model', required=True, help="HF tokenizer name or path, e.g. Qwen/Qwen3-1.7B")
    # (A) SPE character-coverage mode
    ap.add_argument('--spe_vocab', default=None, help="SentencePiece .vocab file (char-coverage mode)")
    ap.add_argument('--spe_model', default=None, help="SentencePiece .model file (char-coverage mode)")
    ap.add_argument('--ascii_only', action='store_true', help="Restrict the SPE charset to ASCII chars")
    # (B) corpus-frequency mode
    ap.add_argument('--manifests', nargs='*', default=[], help="NeMo ASR manifest .json files (one JSON per line)")
    ap.add_argument('--text_files', nargs='*', default=[], help="Plain-text corpus files (one sentence per line)")
    ap.add_argument('--text_key', default='text', help="Manifest field holding the transcript (default: text)")
    ap.add_argument('--min_count', type=int, default=1, help="Keep token ids seen at least this many times")
    ap.add_argument('--max_lines', type=int, default=-1, help="Optional cap on number of corpus lines (debug)")
    ap.add_argument('--output', required=True, help="Output JSON path for the kept-id artifact")
    ap.add_argument('--trust_remote_code', action='store_true')
    args = ap.parse_args()

    spe_mode = bool(args.spe_vocab or args.spe_model)
    corpus_mode = bool(args.manifests or args.text_files)
    if spe_mode == corpus_mode:
        ap.error("Choose exactly one mode: SPE char-coverage (--spe_vocab/--spe_model) "
                 "OR corpus frequency (--manifests/--text_files)")

    tok = AutoTokenizer.from_pretrained(args.hf_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    full_vocab = tok.get_vocab()  # token_str -> id

    byte_chars = set(_bytes_to_unicode().values())
    byte_ids = {full_vocab[c] for c in byte_chars if c in full_vocab}
    special_ids = {i for i in tok.all_special_ids if i is not None}

    artifact = {
        'hf_model': args.hf_model,
        'original_vocab_size': len(full_vocab),
        'num_byte_ids': len(byte_ids),
        'num_special_ids': len(special_ids),
    }

    if spe_mode:
        charset = _spe_charset(args.spe_vocab, args.spe_model, ascii_only=args.ascii_only)
        selected, n_fragment = _kept_ids_by_charset(tok, full_vocab, charset)
        artifact.update({
            'mode': 'spe_charset',
            'spe_vocab': args.spe_vocab,
            'spe_model': args.spe_model,
            'ascii_only': args.ascii_only,
            'spe_charset_size': len(charset),
            'num_selected_ids': len(selected),
            'num_utf8_fragments_skipped': n_fragment,
        })
        progress = (f"charset={len(charset)} chars, selected={len(selected)} "
                    f"(skipped {n_fragment} utf-8 fragments)")
    else:
        counts = Counter()
        n_lines = 0
        for txt in _iter_texts(args.manifests, args.text_files, args.text_key):
            counts.update(tok.encode(txt, add_special_tokens=False))
            n_lines += 1
            if args.max_lines > 0 and n_lines >= args.max_lines:
                break
            if n_lines % 100000 == 0:
                print(f"  ...tokenized {n_lines} lines, {len(counts)} distinct ids so far")
        selected = {i for i, c in counts.items() if c >= args.min_count}
        artifact.update({
            'mode': 'corpus',
            'min_count': args.min_count,
            'num_corpus_lines': n_lines,
            'num_selected_ids': len(selected),
        })
        progress = f"used={len(selected)} from {n_lines} lines"

    kept = sorted(selected | byte_ids | special_ids)
    artifact['compact_vocab_size'] = len(kept)
    artifact['kept_ids'] = kept

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(artifact, f)

    print(
        f"Wrote {args.output}: kept {len(kept)} / {len(full_vocab)} tokens "
        f"({100.0 * len(kept) / len(full_vocab):.1f}%) [{artifact['mode']}] "
        f"({progress}, byte={len(byte_ids)}, special={len(special_ids)})."
    )


if __name__ == '__main__':
    main()
