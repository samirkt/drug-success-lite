"""Build data/sentence2embedding.pkl with REAL BioBERT embeddings of the
eligibility-criteria sentences.

The shipped pkl was empty (0 sentences), which silently disabled HINT's protocol
encoder (protocol2feature zero-filled every trial). This repopulates it so the
criteria branch actually contributes.

By default it embeds every cleaned sentence in the benchmark phase splits
(data/phase_{I,II,III}_{train,valid,test}.csv) — the exact set the dataloader
looks up. Pass --raw to embed the full data/raw_data.csv corpus instead.

Run from the repo root:  uv run python build_sentence_embeddings.py
"""

import argparse
import csv
import sys
import time

csv.field_size_limit(10 ** 9)
sys.path.insert(0, ".")

from HINT.protocol_encode import (  # noqa: E402
    collect_cleaned_sentence_set,
    save_sentence_bert_dict_pkl,
    split_protocol,
    _pick_device,
)


def phase_sentences():
    s = set()
    for p in ["I", "II", "III"]:
        for sp in ["train", "valid", "test"]:
            rows = list(csv.reader(open(f"data/phase_{p}_{sp}.csv")))[1:]
            for r in rows:
                if len(r) > 9 and r[9].strip():
                    res = split_protocol(r[9])
                    s.update(res[0])
                    if len(res) == 2:
                        s.update(res[1])
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="embed full raw_data.csv corpus instead of phase splits")
    args = ap.parse_args()

    sents = collect_cleaned_sentence_set() if args.raw else phase_sentences()
    device = _pick_device()
    print(f"[build] device={device}  unique sentences={len(sents)}", flush=True)

    t0 = time.time()
    d = save_sentence_bert_dict_pkl(sents, device=device)
    print(f"[build] wrote data/sentence2embedding.pkl: {len(d)} sentences in {time.time() - t0:.0f}s", flush=True)
