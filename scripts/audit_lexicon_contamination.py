"""Task 3 diagnostic: dev-set contamination check for the korpus_malti text
that feeds the synth pipeline / vote lexicon.

The 422-image dev set was contamination-checked on the image side
(dedup_against_devset.py, zero flags, claim-audit row "Dev-set
contamination check"). The *text* side was never checked: korpus_malti
paragraphs feed src/datagen/generate_korpus_shard.py's synth renders and,
indirectly, data/maltese_en_it_lexicon.json (the vote lexicon). If a
korpus_malti domain paragraph is textually identical or near-identical to a
dev source paragraph, the router's vote is silently biased toward the dev
gold on that item.

Two independent checks, resumable (checkpoint after each domain):

1. REAL: stream a bounded sample of each of the 11 domain configs actually
   used by generate_korpus_shard.py (parliament, wiki, government_gazzette,
   law_mt, nonfiction, theses, legal, speeches, blogs, umlib_oar,
   web_general). Build 5-gram sets per corpus doc, compare against the
   union of dev-text 5-grams (cheap membership check) as a prefilter, then
   run the dedup_against_devset.py-style pairwise Jaccard only on docs that
   clear the prefilter (>=1 shared 5-gram) against every dev text.
   This is real and reproducible, but bounded to a sample (SAMPLE_PER_DOMAIN
   docs per config, not the full ~470M-token corpus) - disclosed, not
   exhaustive.

2. PROXY: how much of the dev-gold vocabulary (word types and 5-grams) is
   already inside the shipped vote lexicon (data/maltese_en_it_lexicon.json)?
   High OOV would mean the lexicon under-covers dev; this does not by itself
   prove or disprove contamination, it is a coverage proxy run regardless of
   whether (1) succeeds, per the task's own fallback instruction.

Usage:
    PYTHONPATH=. python scripts/audit_lexicon_contamination.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NGRAM_N = 5
JACCARD_FLAG_THRESHOLD = 0.5
SAMPLE_PER_DOMAIN = 8000
MIN_DOC_CHARS = 40

DOMAIN_CONFIGS = (
    "parliament", "wiki", "government_gazzette", "law_mt", "nonfiction",
    "theses", "legal", "speeches", "blogs", "umlib_oar", "web_general",
)

CKPT = ROOT / "outputs" / "campaign" / "lexicon_contamination_ckpt.json"
REPORT = ROOT / "outputs" / "campaign" / "lexicon_contamination_report.json"


def normalize(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return s.split()


def ngrams(words: list[str], n: int) -> set[tuple]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_ckpt() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text(encoding="utf-8"))
    return {"domains_done": [], "total_docs_scanned": 0, "text_flags": [],
            "domain_doc_counts": {}}


def save_ckpt(state: dict) -> None:
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    CKPT.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_real_check() -> dict:
    from datasets import load_dataset

    dev = json.loads((ROOT / "competition_files" / "dev" / "texts.json").read_text(encoding="utf-8"))
    dev_entries = []
    dev_ngram_union: set = set()
    inverted: dict = {}  # 5-gram -> set of dev indices, to avoid all-pairs Jaccard
    for idx, d in enumerate(dev):
        words = normalize(d["text"])
        ng = ngrams(words, NGRAM_N)
        dev_entries.append({"text": d["text"], "ngrams": ng})
        dev_ngram_union |= ng
        for g in ng:
            inverted.setdefault(g, set()).add(idx)
    print(f"[contam] dev: {len(dev_entries)} texts, {len(dev_ngram_union)} unique 5-grams", flush=True)

    state = load_ckpt()
    t0 = time.time()
    for domain in DOMAIN_CONFIGS:
        if domain in state["domains_done"]:
            print(f"[contam] {domain}: already done, skipping", flush=True)
            continue
        try:
            ds = load_dataset("MLRS/korpus_malti", domain, split="train", streaming=True)
        except Exception as e:
            print(f"[contam] {domain}: SKIP ({e})", flush=True)
            state["domains_done"].append(domain)
            state["domain_doc_counts"][domain] = 0
            save_ckpt(state)
            continue

        n_scanned = 0
        n_prefilter_hits = 0
        max_jac = 0.0
        for row in ds:
            if n_scanned >= SAMPLE_PER_DOMAIN:
                break
            raw = row.get("text") or ""
            if isinstance(raw, list):
                raw = " ".join(str(x) for x in raw)
            text = raw.strip()
            if len(text) < MIN_DOC_CHARS:
                continue
            n_scanned += 1
            words = normalize(text)
            pg = ngrams(words, NGRAM_N)
            cand = set()
            for g in pg:
                hit = inverted.get(g)
                if hit:
                    cand |= hit
            if not cand:
                continue
            n_prefilter_hits += 1
            for i in cand:
                score = jaccard(pg, dev_entries[i]["ngrams"])
                if score > max_jac:
                    max_jac = score
                if score >= JACCARD_FLAG_THRESHOLD:
                    state["text_flags"].append({
                        "domain": domain, "corpus_text": text[:150],
                        "dev_index": i, "dev_text": dev_entries[i]["text"][:150],
                        "jaccard": round(score, 3),
                    })
        state["domains_done"].append(domain)
        state["domain_doc_counts"][domain] = n_scanned
        state["total_docs_scanned"] = state.get("total_docs_scanned", 0) + n_scanned
        state.setdefault("max_jaccard", {})[domain] = round(max_jac, 4)
        save_ckpt(state)
        print(f"[contam] {domain}: scanned={n_scanned} prefilter_hits={n_prefilter_hits} "
              f"max_jaccard={max_jac:.3f} flags_so_far={len(state['text_flags'])} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    return {
        "method": "real",
        "domains": list(DOMAIN_CONFIGS),
        "sample_per_domain": SAMPLE_PER_DOMAIN,
        "total_docs_scanned": state["total_docs_scanned"],
        "domain_doc_counts": state["domain_doc_counts"],
        "n_dev_texts": len(dev_entries),
        "n_dev_5grams": len(dev_ngram_union),
        "jaccard_flag_threshold": JACCARD_FLAG_THRESHOLD,
        "n_text_flags": len(state["text_flags"]),
        "max_jaccard_per_domain": state.get("max_jaccard", {}),
        "text_flags": state["text_flags"][:50],
    }


def run_proxy_check() -> dict:
    dev = json.loads((ROOT / "competition_files" / "dev" / "texts.json").read_text(encoding="utf-8"))
    lex_data = json.loads((ROOT / "data" / "maltese_en_it_lexicon.json").read_text(encoding="utf-8"))
    lex = set(lex_data.keys() if isinstance(lex_data, dict) else lex_data)
    lex |= {w.lower() for w in lex}

    all_words: set = set()
    all_ngrams: set = set()
    total_word_tokens = 0
    oov_word_tokens = 0
    for d in dev:
        words = normalize(d["text"])
        total_word_tokens += len(words)
        for w in words:
            all_words.add(w)
            if w not in lex:
                oov_word_tokens += 1
        all_ngrams |= ngrams(words, NGRAM_N)

    oov_types = [w for w in all_words if w not in lex]

    return {
        "method": "proxy",
        "description": "fraction of dev-gold vocabulary already present in the shipped vote lexicon "
                        "(data/maltese_en_it_lexicon.json, 2.1M entries) - a coverage proxy, not a "
                        "direct leakage test",
        "n_dev_texts": len(dev),
        "n_lexicon_entries": len(lex_data),
        "n_dev_word_types": len(all_words),
        "n_dev_word_tokens": total_word_tokens,
        "oov_word_types": len(oov_types),
        "oov_word_type_rate": len(oov_types) / len(all_words) if all_words else 0.0,
        "oov_word_token_rate": oov_word_tokens / total_word_tokens if total_word_tokens else 0.0,
        "n_dev_5grams": len(all_ngrams),
    }


def main() -> None:
    proxy = run_proxy_check()
    print(f"[proxy] dev word types={proxy['n_dev_word_types']} "
          f"OOV-vs-lexicon type rate={proxy['oov_word_type_rate']:.4f} "
          f"token rate={proxy['oov_word_token_rate']:.4f}", flush=True)

    real = run_real_check()
    print(f"[real] scanned {real['total_docs_scanned']} corpus docs across "
          f"{len(real['domains'])} domains, {real['n_text_flags']} Jaccard>= "
          f"{JACCARD_FLAG_THRESHOLD} flags", flush=True)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"real": real, "proxy": proxy}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"[contam] wrote -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
