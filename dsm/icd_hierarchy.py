"""ICD-10-CM hierarchy expansion for the disease feature.

Loads the compact lookup built by `scripts/build_icd_hierarchy.py` and expands each ICD code into
its hierarchy ancestors — full code, 3-char category, block/section, chapter — as multi-hot tokens:

    E74.02 -> ["ICD:E74.02", "ICD3:E74", "BLOCK:E70-E88", "CHAPTER:IV"]

`icd_ancestor_tokens` is module-level (picklable by reference) and is wired in as the disease
MultiHot `token_fn`, so training and serving share one featurization path. Codes absent from the
table fall back to range inference over the chapter/section letter ranges, so at least the ICD3
category and chapter are always emitted. Normalization upper-cases and strips dots for lookup; the
dotted form is preserved for the `ICD:` display token.
"""

from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "assets" / "icd10cm_hierarchy.json"
_H: dict | None = None


def _hierarchy() -> dict:
    global _H
    if _H is None:
        with open(_PATH) as f:
            _H = json.load(f)
    return _H


def _norm(code: str) -> str:
    return str(code).strip().upper().replace(".", "")


def _display(norm: str) -> str:
    """Canonical dotted display: 'E7402' -> 'E74.02', 'E74' -> 'E74'."""
    return norm if len(norm) <= 3 else f"{norm[:3]}.{norm[3:]}"


def _range_lookup(ranges: list[list[str]], cat3: str) -> str | None:
    """Label of the first [label, start, end] range whose [start, end] covers `cat3` (ICD
    categories are uniformly letter+2 chars, so plain string comparison is a correct ordering)."""
    for label, start, end in ranges:
        if start <= cat3 <= end:
            return label
    return None


def icd_ancestor_tokens(codes: list[str]) -> list[str]:
    """Expand a row's ICD codes to the union of their hierarchy ancestor tokens (de-duped downstream
    by MultiHot's per-row set)."""
    h = _hierarchy()
    cat_map, ch_ranges, sec_ranges = h["cat_map"], h["chapter_ranges"], h["section_ranges"]
    out: list[str] = []
    for code in codes:
        norm = _norm(code)
        if not norm:
            continue
        cat3 = norm[:3]
        out.append("ICD:" + _display(norm))
        out.append("ICD3:" + cat3)
        block, chapter = cat_map.get(cat3, [None, None])
        if block is None:                                  # code not in the table -> infer by range
            block = _range_lookup(sec_ranges, cat3)
            chapter = _range_lookup(ch_ranges, cat3)
        if block:
            out.append("BLOCK:" + block)
        if chapter:
            out.append("CHAPTER:" + chapter)
    return out
