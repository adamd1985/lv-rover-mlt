"""LV-ROVER style multi-candidate voting (Stuner et al. 2017).

For each anchor word not in lexicon, take in-lexicon candidates across
all streams and pick the most-frequent one (ties broken by the v11
guard set: edit-dist bound, diacritic preserve, alpha-floor, length).
Falls back to anchor when no candidate beats the anchor floor.

Compare against v12's sequential first-match router.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from jiwer import cer

import importlib.util
spec = importlib.util.spec_from_file_location("ct", "competition_transcriber.py")
ct = importlib.util.module_from_spec(spec); import sys; sys.modules['ct'] = ct
spec.loader.exec_module(ct)


class LVRoverRouter:
    def __init__(self, lex: set, max_swap_dist: int = 2) -> None:
        self.lex = lex
        self.max_swap_dist = max_swap_dist

    def _in_lex(self, w):
        n = ct._norm_lookup(w)
        return bool(n) and (n in self.lex or n.lower() in self.lex)

    def _passes_guards(self, anchor, cand):
        if not cand or cand == anchor: return False
        if self._in_lex(anchor) or not self._in_lex(cand): return False
        a_alpha = sum(c.isalpha() for c in anchor)
        c_alpha = sum(c.isalpha() for c in cand)
        if c_alpha < a_alpha or a_alpha < 3 or c_alpha < 3: return False
        if len(cand) < len(anchor) - 1 or abs(len(anchor) - len(cand)) > 2: return False
        if ct._non_ascii_alpha(cand) < ct._non_ascii_alpha(anchor): return False
        d = ct._edit_distance(anchor, cand)
        if d == 0 or d > self.max_swap_dist: return False
        return True

    def combine(self, anchor: str, candidate_streams: List[str]) -> str:
        from competition_transcriber import _WS_SPLIT, _align_word_seqs
        # Align each candidate to anchor's lines
        cand_aligned_per_line = []
        cand_token_lists = [[w for w in _WS_SPLIT.split(c.replace("\n", " ").strip()) if w]
                            for c in candidate_streams]
        cursors = [0] * len(candidate_streams)
        out_lines = []
        for anchor_line in anchor.split("\n"):
            a_words = [w for w in _WS_SPLIT.split(anchor_line.strip()) if w]
            if not a_words:
                out_lines.append("")
                continue
            # For each candidate stream, slice a window matched to anchor length and align
            aligned_per_stream = []
            for k, ct_tokens in enumerate(cand_token_lists):
                window = ct_tokens[cursors[k]: cursors[k] + 2 * len(a_words)]
                alignment = _align_word_seqs(a_words, window)
                per_pos = []
                for a, b in alignment:
                    if a is None: continue
                    per_pos.append(b)
                while len(per_pos) < len(a_words):
                    per_pos.append(None)
                aligned_per_stream.append(per_pos[: len(a_words)])
                cursors[k] += len(a_words)
            # Vote per position
            line_out = []
            for i, anc in enumerate(a_words):
                cands = [aligned_per_stream[k][i] for k in range(len(aligned_per_stream))]
                votes = Counter()
                for c in cands:
                    if c and self._passes_guards(anc, c):
                        votes[c] += 1
                if votes:
                    line_out.append(votes.most_common(1)[0][0])
                else:
                    line_out.append(anc)
            out_lines.append(" ".join(line_out))
        return "\n".join(out_lines)


def main():
    confusion = json.loads(Path('data/tess_confusion.json').read_text())
    lex_data = json.loads(Path('data/maltese_en_it_lexicon.json').read_text())
    lex = set(lex_data.keys() if isinstance(lex_data, dict) else lex_data); lex |= {w.lower() for w in lex}
    corr = ct._ConfusionCorrector(confusion, lex, tau=0.05)
    seq_router = ct._CrossEngineRouter(lex, max_swap_dist=2)
    lv_router = LVRoverRouter(lex, max_swap_dist=2)

    def load(p): return {json.loads(l)['id']: json.loads(l).get('pred', json.loads(l).get('gold')) for l in Path(p).read_text().splitlines()}

    DS = {
        'dev': {
            'gold': 'outputs/campaign/dev_gold.jsonl',
            'mlt+ita': 'outputs/tess_lang_chain/mlt_ita.jsonl',
            'cands': {
                'easy': 'outputs/easyocr/preds_dev.jsonl',
                'mlt': 'outputs/tess_lang_chain/mlt.jsonl',
                'mlt+ita+fra': 'outputs/tess_lang_chain/mlt_ita_fra.jsonl',
                'mlt+ita+spa': 'outputs/audit_v12/dev_mlt_ita_spa.jsonl',
                'mltstock': 'outputs/audit_v12/dev_mltstock.jsonl',
                'mltstock+ita': 'outputs/audit_v12/dev_mltstock_ita.jsonl',
            },
        },
        'synth_val': {
            'gold': 'outputs/audit_v12/synth_val_gold.jsonl',
            'mlt+ita': 'outputs/audit_v12/synth_val_mlt_ita.jsonl',
            'cands': {
                'mlt': 'outputs/audit_v12/synth_val_mlt.jsonl',
                'mlt+ita+fra': 'outputs/audit_v12/synth_val_mlt_ita_fra.jsonl',
                'mlt+ita+spa': 'outputs/audit_v12/synth_val_mlt_ita_spa.jsonl',
                'mltstock': 'outputs/audit_v12/synth_val_mltstock.jsonl',
                'mltstock+ita': 'outputs/audit_v12/synth_val_mltstock_ita.jsonl',
            },
        },
        'hard_synth': {
            'gold': 'outputs/audit_v12/hard_gold.jsonl',
            'mlt+ita': 'outputs/audit_v12/hard_mlt_ita.jsonl',
            'cands': {
                'mlt': 'outputs/audit_v12/hard_mlt.jsonl',
                'mlt+ita+fra': 'outputs/audit_v12/hard_mlt_ita_fra.jsonl',
                'mlt+ita+spa': 'outputs/audit_v12/hard_mlt_ita_spa.jsonl',
                'mltstock': 'outputs/audit_v12/hard_mltstock.jsonl',
                'mltstock+ita': 'outputs/audit_v12/hard_mltstock_ita.jsonl',
            },
        },
    }

    # Configs: list of (label, candidate_keys, router_type)
    configs = [
        ('v12 (seq):    easy + mlt + fra',                       ['easy', 'mlt', 'mlt+ita+fra'], 'seq'),
        ('LV-ROVER:     easy + mlt + fra',                       ['easy', 'mlt', 'mlt+ita+fra'], 'lv'),
        ('LV-ROVER:     easy + mlt + fra + spa',                 ['easy', 'mlt', 'mlt+ita+fra', 'mlt+ita+spa'], 'lv'),
        ('LV-ROVER:     easy + mlt + fra + stock',               ['easy', 'mlt', 'mlt+ita+fra', 'mltstock'], 'lv'),
        ('LV-ROVER:     easy + mlt + fra + stock+ita',           ['easy', 'mlt', 'mlt+ita+fra', 'mltstock+ita'], 'lv'),
        ('LV-ROVER:     easy + mlt + fra + spa + stock',         ['easy', 'mlt', 'mlt+ita+fra', 'mlt+ita+spa', 'mltstock'], 'lv'),
        ('LV-ROVER:     6 cands (easy+mlt+fra+spa+stock+stock_ita)',
                                                                  ['easy', 'mlt', 'mlt+ita+fra', 'mlt+ita+spa', 'mltstock', 'mltstock+ita'], 'lv'),
    ]

    print(f"{'config':60s} {'dev':>10s} {'synth_val':>10s} {'hard_synth':>10s}")
    for label, cand_keys, kind in configs:
        row = [label]
        for sn in ['dev', 'synth_val', 'hard_synth']:
            s = DS[sn]
            gold = load(s['gold'])
            anc = load(s['mlt+ita'])
            common = sorted(set(gold) & set(anc))
            refs = [gold[k] for k in common]
            cand_streams_per_id: Dict[str, List[str]] = {k: [] for k in common}
            for ck in cand_keys:
                if ck not in s['cands']:
                    continue
                d = load(s['cands'][ck])
                for k in common:
                    cand_streams_per_id[k].append(d.get(k, '') or '')
            hyps = []
            for k in common:
                a = corr.correct(anc[k]) if len(anc[k]) >= 100 else anc[k]
                streams = cand_streams_per_id[k]
                if kind == 'seq':
                    for c in streams:
                        a = seq_router.combine(a, c)
                else:
                    a = lv_router.combine(a, streams)
                hyps.append(a)
            c = cer(refs, hyps)
            row.append(f"{c:.5f}")
        print(f"  {row[0]:60s} {row[1]:>10s} {row[2]:>10s} {row[3]:>10s}")


if __name__ == "__main__":
    main()
