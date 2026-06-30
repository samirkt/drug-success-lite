#!/usr/bin/env python3
"""Download the official ICD-10-CM Tabular XML (CDC/NCHS, public domain) and parse its
chapter / section / category hierarchy into a compact lookup at dsm/assets/icd10cm_hierarchy.json.

Both training and the web tool consume the JSON via dsm/icd_hierarchy.py, so the disease
featurization path is identical. Run once; re-run with --force (or bump ICD_YEAR) to refresh.

    uv run python scripts/build_icd_hierarchy.py

Network is needed only for this one-time build; the committed JSON has no runtime dependency.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from dsm.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ICD_YEAR = 2025
ZIP_URL = (f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/{ICD_YEAR}/"
           f"icd10cm-table-index-{ICD_YEAR}.zip")
RAW_ZIP = PROJECT_ROOT / "data" / f"icd10cm-table-index-{ICD_YEAR}.zip"
OUT = PROJECT_ROOT / "dsm" / "assets" / "icd10cm_hierarchy.json"

_ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII",
          "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII"]
_RANGE_RE = re.compile(r"\(([A-Z][0-9A-Z]*)-([A-Z][0-9A-Z]*)\)")


def _fetch_zip(force: bool) -> bytes:
    if RAW_ZIP.exists() and not force:
        logger.info("using cached %s", RAW_ZIP)
        return RAW_ZIP.read_bytes()
    logger.info("downloading %s", ZIP_URL)
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "drug-success-lite/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    RAW_ZIP.write_bytes(data)
    logger.info("cached %s (%.1f MB)", RAW_ZIP, len(data) / 1e6)
    return data


def _tabular_xml(zip_bytes: bytes) -> bytes:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    member = next((n for n in zf.namelist() if n.lower().endswith(".xml") and "tabular" in n.lower()),
                  None)
    if member is None:
        raise SystemExit(f"no *tabular*.xml in {RAW_ZIP}; members: {zf.namelist()}")
    logger.info("parsing %s", member)
    return zf.read(member)


def _split_range(token: str) -> tuple[str, str]:
    """'E70-E88' -> ('E70','E88'); 'C50' / 'C7A' -> ('C50','C50')."""
    parts = token.split("-")
    return (parts[0], parts[1]) if len(parts) == 2 else (token, token)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if the zip is cached")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = ET.fromstring(_tabular_xml(_fetch_zip(args.force)))

    cat_map: dict[str, list[str]] = {}          # 'E74' -> ['E70-E88', 'IV']
    chapter_ranges: list[list[str]] = []        # ['IV', 'E00', 'E89']
    section_ranges: list[list[str]] = []        # ['E70-E88', 'E70', 'E88'] (leaf blocks only)

    for chapter in root.findall("chapter"):
        num = (chapter.findtext("name") or "").strip()
        roman = _ROMAN[int(num)] if num.isdigit() and int(num) < len(_ROMAN) else num
        m = _RANGE_RE.search(chapter.findtext("desc") or "")
        if m:
            chapter_ranges.append([roman, m.group(1), m.group(2)])
        for section in chapter.findall("section"):
            cats = section.findall("diag")          # leaf blocks hold the category diags directly
            if not cats:
                continue                            # skip broad organizational "block-of-blocks"
            block = section.get("id")
            start, end = _split_range(block)
            section_ranges.append([block, start, end])
            for diag in cats:
                cat3 = (diag.findtext("name") or "").strip().upper()
                if cat3:
                    cat_map[cat3] = [block, roman]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "year": ICD_YEAR,
        "cat_map": cat_map,
        "chapter_ranges": chapter_ranges,
        "section_ranges": section_ranges,
    }))
    logger.info("wrote %s: %d categories, %d chapters, %d blocks (%.0f KB)",
                OUT, len(cat_map), len(chapter_ranges), len(section_ranges),
                OUT.stat().st_size / 1e3)


if __name__ == "__main__":
    main()
