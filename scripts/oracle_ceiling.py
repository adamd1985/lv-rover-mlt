"""Oracle ceiling for the router candidate set.

Two oracles, both on the consistently-generated candidate streams in
outputs/audit_v12/ (same psm-6 config):

1. best-single-stream: per image, pick the whole paragraph (anchor or any
   candidate) with the lowest CER to gold. Conservative lower bound.
2. per-word: align each candidate to gold at word level, pick the best word
   per gold position. True upper bound on router stream-selection gain.

The gap between oracle and v14's CER bounds how much further candidate
stacking can help.
"""
from __future__ import annotations

import json
from pathlib import Path

from jiwer import cer

import importlib.util
spec = importlib.util.spec_from_file_location("ct", "competition_transcriber.py")
ct = importlib.util.module_from_spec(spec); import sys; sys.modules['ct'] = ct
spec.loader.exec_module(ct)
from competition_transcriber import _WS_SPLIT, _align_word_seqs


def load(p):
    return {json.loads(l)["id"]: json.loads(l).get("pred", json.loads(l).get("gold"))
            for l in Path(p).read_text().splitlines()}


def main():
    gold = load("outputs/campaign/dev_gold.jsonl")
    streams = {
        "mlt+ita": load("outputs/tess_lang_chain/mlt_ita.jsonl"),
        "mlt": load("outputs/tess_lang_chain/mlt.jsonl"),
        "mlt+ita+fra": load("outputs/audit_v12/dev_mlt_ita_fra.jsonl") if Path("outputs/audit_v12/dev_mlt_ita_fra.jsonl").exists() else load("outputs/tess_lang_chain/mlt_ita_fra.jsonl"),
        "mltstock": load("outputs/audit_v12/dev_mltstock.jsonl"),
        "mlt+ita+spa": load("outputs/audit_v12/dev_mlt_ita_spa.jsonl"),
        "easy": load("outputs/easyocr/preds_dev.jsonl"),
    }
    common = sorted(set(gold) & set.intersection(*[set(s) for s in streams.values()]))
    print(f"n={len(common)} streams={list(streams.keys())}\n")

    # Per-stream standalone CER
    for name, s in streams.items():
        c = cer([gold[k] for k in common], [s[k] for k in common])
        print(f"  standalone {name:14s}: {c:.5f}")

    # Oracle 1: best single stream per image
    best_hyps = []
    for k in common:
        cands = [s[k] for s in streams.values()]
        best = min(cands, key=lambda h: cer([gold[k]], [h]))
        best_hyps.append(best)
    print(f"\noracle (best single stream / image): {cer([gold[k] for k in common], best_hyps):.5f}")

    # Oracle 2: per-word, anchor = mlt+ita
    anchor = streams["mlt+ita"]
    word_hyps = []
    for k in common:
        g_words = [w for w in _WS_SPLIT.split(gold[k].replace('\n', ' ').strip()) if w]
        a_words = [w for w in _WS_SPLIT.split(anchor[k].replace('\n', ' ').strip()) if w]
        # collect all candidate words aligned to anchor positions, plus anchor itself
        all_streams = [anchor[k]] + [streams[n][k] for n in streams if n != "mlt+ita"]
        # For the per-word oracle we pick, for each anchor position, the candidate
        # word that best matches the aligned gold word.
        # Align anchor to gold to get target per anchor position.
        ag = _align_word_seqs(a_words, g_words)
        target = []
        for a, g in ag:
            if a is not None:
                target.append(g)
        while len(target) < len(a_words):
            target.append(None)
        target = target[: len(a_words)]
        # candidate words per anchor position
        cand_aligned = []
        for st in all_streams:
            st_words = [w for w in _WS_SPLIT.split(st.replace('\n', ' ').strip()) if w]
            al = _align_word_seqs(a_words, st_words)
            per = []
            for a, b in al:
                if a is None:
                    continue
                per.append(b)
            while len(per) < len(a_words):
                per.append(None)
            cand_aligned.append(per[: len(a_words)])
        out = []
        for i, a in enumerate(a_words):
            tgt = target[i]
            choices = [a] + [cand_aligned[j][i] for j in range(len(cand_aligned)) if cand_aligned[j][i]]
            if tgt is None:
                out.append(a)
            else:
                best = min(choices, key=lambda w: ct._edit_distance(w, tgt))
                out.append(best)
        word_hyps.append(" ".join(out))
    print(f"oracle (per-word best vs gold):     {cer([gold[k] for k in common], word_hyps):.5f}")
    print(f"\nv14 actual wrapper dev CER:          0.01357")


if __name__ == "__main__":
    main()
