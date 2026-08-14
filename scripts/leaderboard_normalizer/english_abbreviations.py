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
"""Word/name/compound mappings for the vendored leaderboard normalizer.

These mirror the three dicts in
github.com/huggingface/open_asr_leaderboard/normalizer/english_abbreviations.py.

The British->American SPELLING dict (``english_spelling_normalizer``) is the
canonical ~1.7k-entry Whisper "tysto" list. Rather than re-vendor 1.7k literals
(and risk drifting from the exact list the container already ships), we reuse the
mapping from the installed ``whisper_normalizer`` and add the three leaderboard
prepends ("ok"/"kay"/"etcetera"). This keeps us byte-identical to whatever Whisper
spelling list the eval container has, plus the leaderboard's intentional extras.

``english_name_normalizer`` (name-variant folding) and
``english_compound_normalizer`` (multi-word -> single token) are leaderboard
additions with no Whisper equivalent, so they are vendored verbatim.
"""

# ---------------------------------------------------------------------------
# Spelling: canonical Whisper list (from the installed whisper_normalizer) plus
# the leaderboard's three prepended entries. If whisper_normalizer is somehow
# unavailable we still function with just the extras (spelling folding is a minor
# WER contributor relative to the fillers/acronyms/compounds below).
# ---------------------------------------------------------------------------
_LEADERBOARD_SPELLING_EXTRAS = {
    "ok": "okay",
    "kay": "okay",
    "etcetera": "etc",
}


def _build_spelling_normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer as _WhisperETN

        base = dict(_WhisperETN().standardize_spellings.mapping)
    except Exception:  # noqa: BLE001 - fall back to just the extras
        base = {}
    base.update(_LEADERBOARD_SPELLING_EXTRAS)
    return base


english_spelling_normalizer = _build_spelling_normalizer()

# ---------------------------------------------------------------------------
# Name-variant folding (leaderboard-specific; verbatim).
# ---------------------------------------------------------------------------
english_name_normalizer = {
    # -- Double-letter variants --
    "alan": "allen",
    "allan": "allen",
    "bridgette": "bridget",
    "charly": "charlie",
    "charley": "charlie",
    "garry": "gary",
    "gregg": "greg",
    "jacky": "jackie",
    "joann": "joanne",
    "joane": "joanne",
    "kellye": "kelly",
    "kelli": "kelly",
    "kelley": "kelly",
    "lilly": "lily",
    "micheal": "michael",
    "michele": "michelle",
    "mollie": "molly",
    "phillip": "philip",
    "sallie": "sally",
    "stacey": "stacy",
    "stacie": "stacy",
    "tracey": "tracy",
    "tracie": "tracy",
    "bret": "brett",
    "carrol": "carol",
    "carole": "carol",
    "carroll": "carol",
    "allison": "alison",
    "alyson": "alison",
    "russel": "russell",
    "douglass": "douglas",
    "dominick": "dominic",
    "robb": "rob",
    # -- Chr/Kr variants --
    "kris": "chris",
    "kristopher": "christopher",
    "cristopher": "christopher",
    "kristina": "christina",
    "kristen": "kristin",
    # -- C/K variants --
    "karl": "carl",
    "kathy": "cathy",
    "katherine": "catherine",
    "kathryn": "catherine",
    "catharine": "catherine",
    "erik": "eric",
    "erick": "eric",
    "caren": "karen",
    "caryn": "karen",
    "karin": "karen",
    "katelyn": "caitlin",
    "kaitlyn": "caitlin",
    "kaitlin": "caitlin",
    "nikole": "nicole",
    "veronika": "veronica",
    "viktor": "victor",
    "viktoria": "victoria",
    "kevan": "kevin",
    "patrik": "patrick",
    "frederik": "frederick",
    "fredrick": "frederick",
    "lukas": "lucas",
    # -- Silent letters / alternate spellings --
    "ann": "anne",
    "jon": "john",
    "johnathan": "jonathan",
    "jonathon": "jonathan",
    "sara": "sarah",
    "mathew": "matthew",
    "nicolas": "nicholas",
    "rachael": "rachel",
    "rebekah": "rebecca",
    "devorah": "deborah",
    "theresa": "teresa",
    "suzanne": "susanne",
    "antony": "anthony",
    "martyn": "martin",
    "denis": "dennis",
    "laurence": "lawrence",
    "tomas": "thomas",
    "tobey": "toby",
    # -- Mac/Mc extensions --
    "macarthur": "mcarthur",
    "macartney": "mccartney",
    "macarthy": "mccarthy",
    "maccarthy": "mccarthy",
    "macdonald": "mcdonald",
    "mackay": "mckay",
    "mackenzie": "mckenzie",
    "macleod": "mcleod",
    "maclean": "mclean",
    "macmillan": "mcmillan",
    "macintosh": "mcintosh",
    "macintyre": "mcintyre",
    "macnamara": "mcnamara",
    "macgowan": "mcgowan",
    # -- International --
    "mohamad": "mohammed",
    "mohamed": "mohammed",
    "mohammad": "mohammed",
    "muhammad": "mohammed",
    "muhamad": "mohammed",
    "muhammed": "mohammed",
    "mouhamed": "mohammed",
    "mouhamad": "mohammed",
    "mahomet": "mohammed",
    "fatimah": "fatima",
    "yusuf": "yousef",
    "yusef": "yousef",
    "myriam": "miriam",
    "rajeev": "rajiv",
    # -- Miscellaneous homophones --
    "alphonso": "alfonso",
    "bryan": "brian",
    "geoffrey": "jeffrey",
    "jeffery": "jeffrey",
    "geoff": "jeff",
    "neal": "neil",
    "shaun": "sean",
    "shawn": "sean",
    "shayne": "shane",
    "stephen": "steven",
    "toni": "tony",
    "leigh": "lee",
    "lewis": "louis",
    "marc": "mark",
    "meghan": "megan",
    "nathalie": "natalie",
    "robyn": "robin",
    "rodger": "roger",
    "linsey": "lindsay",
    "lindsey": "lindsay",
    "zackary": "zachary",
    "zachery": "zachary",
    "zak": "zach",
    "sheri": "sherry",
    "cheri": "sherry",
    "sherrie": "sherry",
    "terri": "terry",
    "lori": "laurie",
    "jaime": "jamie",
    "jayson": "jason",
    "lesley": "leslie",
    "lynda": "linda",
    "lynne": "lynn",
    "gayle": "gail",
    "rhonda": "ronda",
    "yvonne": "ivonne",
    "stewart": "stuart",
    "walther": "walter",
    "symon": "simon",
    "collin": "colin",
    "dillon": "dylan",
    "aron": "aaron",
    "artur": "arthur",
    "henri": "henry",
    "josef": "joseph",
    "pieter": "peter",
}

# ---------------------------------------------------------------------------
# Compound-word joining (leaderboard-specific; verbatim). Regex keys applied with
# re.sub AFTER symbol removal, so hyphens/punctuation are already stripped.
# ---------------------------------------------------------------------------
english_compound_normalizer = {
    r"\bet\s+cetera\b": "etc",
    r"\bal\s+right\b": "alright",
    r"\ball\s+right\b": "alright",
    r"\bhow\s+ever\b": "however",
    r"\bwi\s+fi\b": "wifi",
    r"\bhi\s+fi\b": "hifi",
    r"\blo\s+fi\b": "lofi",
    r"\bsci\s+fi\b": "scifi",
    r"\be\s+mail\b": "email",
    r"\be\s+book\b": "ebook",
    r"\be\s+commerce\b": "ecommerce",
    r"\bx\s+ray\b": "xray",
    r"\bt\s+shirt\b": "tshirt",
    r"\ba\s+m\b": "am",
    r"\bp\s+m\b": "pm",
    r"\bo\s+k\b": "okay",
}
