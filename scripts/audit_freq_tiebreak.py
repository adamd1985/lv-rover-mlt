"""LV-ROVER with korpus_malti unigram frequency tie-break (Rijhwani 2021 light).

Among lexicon-valid candidate words that pass the v11 guards, ties on vote
count break by unigram frequency from korpus_malti. Higher-frequency word
wins. Falls back to stream order when neither is in the frequency table.
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


class LVRoverFreqRouter:
    def __init__(self, lex: set, freq: Dict[str, int], max_swap_dist: int = 2) -> None:
        self.lex = lex
        self.freq = freq
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

    def _word_freq(self, w):
        n = ct._norm_lookup(w)
        if not n: return 0
        return max(self.freq.get(n, 0), self.freq.get(n.lower(), 0))

    def combine(self, anchor: str, candidate_streams: List[str]) -> str:
        from competition_transcriber import _WS_SPLIT, _align_word_seqs
        cand_token_lists = [[w for w in _WS_SPLIT.split(c.replace("\n", " ").strip()) if w]
                             for c in candidate_streams]
        cursors = [0] * len(candidate_streams)
        out_lines = []
        for anchor_line in anchor.split("\n"):
            a_words = [w for w in _WS_SPLIT.split(anchor_line.strip()) if w]
            if not a_words:
                out_lines.append("")
                continue
            aligned_per_stream = []
            for k, tokens in enumerate(cand_token_lists):
                window = tokens[cursors[k]: cursors[k] + 2 * len(a_words)]
                alignment = _align_word_seqs(a_words, window)
                per_pos = []
                for a, b in alignment:
                    if a is None: continue
                    per_pos.append(b)
                while len(per_pos) < len(a_words):
                    per_pos.append(None)
                aligned_per_stream.append(per_pos[: len(a_words)])
                cursors[k] += len(a_words)
            line_out = []
            for i, anc in enumerate(a_words):
                votes = Counter()
                stream_order = []
                for k in range(len(aligned_per_stream)):
                    c = aligned_per_stream[k][i]
                    if c and self._passes_guards(anc, c):
                        if c not in votes:
                            stream_order.append(c)
                        votes[c] += 1
                if votes:
                    # Tie-break: vote count first, then unigram freq, then stream order
                    best = max(stream_order, key=lambda w: (votes[w], self._word_freq(w), -stream_order.index(w)))
                    line_out.append(best)
                else:
                    line_out.append(anc)
            out_lines.append(" ".join(line_out))
        return "\n".join(out_lines)


def main():
    confusion = json.loads(Path('data/tess_confusion.json').read_text())
    lex_data = json.loads(Path('data/maltese_en_it_lexicon.json').read_text())
    lex = set(lex_data.keys() if isinstance(lex_data, dict) else lex_data); lex |= {w.lower() for w in lex}
    freq_data = json.loads(Path('data/maltese_lexicon.json').read_text())
    freq = freq_data if isinstance(freq_data, dict) else {w: 1 for w in freq_data}
    print(f'lex size: {len(lex)}, freq table size: {len(freq)}')

    corr = ct._ConfusionCorrector(confusion, lex, tau=0.05)
    lv_router = ct._CrossEngineRouter(lex, max_swap_dist=2)
    freq_router = LVRoverFreqRouter(lex, freq, max_swap_dist=2)

    def load(p): return {json.loads(l)['id']: json.loads(l).get('pred', json.loads(l).get('gold')) for l in Path(p).read_text().splitlines()}

    DS = {
        'dev': {
            'gold': 'outputs/campaign/dev_gold.jsonl',
            'mlt+ita': 'outputs/tess_lang_chain/mlt_ita.jsonl',
            'cands': {
                'easy': 'outputs/easyocr/preds_dev.jsonl',
                'mlt': 'outputs/tess_lang_chain/mlt.jsonl',
                'mlt+ita+fra': 'outputs/tess_lang_chain/mlt_ita_fra.jsonl',
                'mltstock': 'outputs/audit_v12/dev_mltstock.jsonl',
                'mlt+ita+spa': 'outputs/audit_v12/dev_mlt_ita_spa.jsonl',
            },
        },
        'synth_val': {
            'gold': 'outputs/audit_v12/synth_val_gold.jsonl',
            'mlt+ita': 'outputs/audit_v12/synth_val_mlt_ita.jsonl',
            'cands': {
                'mlt': 'outputs/audit_v12/synth_val_mlt.jsonl',
                'mlt+ita+fra': 'outputs/audit_v12/synth_val_mlt_ita_fra.jsonl',
                'mltstock': 'outputs/audit_v12/synth_val_mltstock.jsonl',
                'mlt+ita+spa': 'outputs/audit_v12/synth_val_mlt_ita_spa.jsonl',
            },
        },
        'hard_synth': {
            'gold': 'outputs/audit_v12/hard_gold.jsonl',
            'mlt+ita': 'outputs/audit_v12/hard_mlt_ita.jsonl',
            'cands': {
                'mlt': 'outputs/audit_v12/hard_mlt.jsonl',
                'mlt+ita+fra': 'outputs/audit_v12/hard_mlt_ita_fra.jsonl',
                'mltstock': 'outputs/audit_v12/hard_mltstock.jsonl',
                'mlt+ita+spa': 'outputs/audit_v12/hard_mlt_ita_spa.jsonl',
            },
        },
    }

    configs = [
        ('v13 (LV, stream-order tiebreak): easy+mlt+fra+stock',     ['easy', 'mlt', 'mlt+ita+fra', 'mltstock'], 'lv'),
        ('v14 (LV+freq):                   easy+mlt+fra+stock',     ['easy', 'mlt', 'mlt+ita+fra', 'mltstock'], 'lvf'),
        ('v14+ (LV+freq):                  easy+mlt+fra+stock+spa', ['easy', 'mlt', 'mlt+ita+fra', 'mltstock', 'mlt+ita+spa'], 'lvf'),
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
            cand_streams_per_id = {k: [] for k in common}
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
                if kind == 'lv':
                    a = lv_router.combine_lv(a, streams) if hasattr(lv_router, 'combine_lv') else streams[0]
                    if not hasattr(lv_router, 'combine_lv'):
                        # fallback: sequential
                        for c in streams: a = lv_router.combine(a, c)
                else:
                    a = freq_router.combine(a, streams)
                hyps.append(a)
            c = cer(refs, hyps)
            row.append(f"{c:.5f}")
        print(f"  {row[0]:60s} {row[1]:>10s} {row[2]:>10s} {row[3]:>10s}")


if __name__ == "__main__":
    main()
