"""Anti-overfit audit of the v16-v20 post-processing rules.

Each rule added in submission v16 through v20 is gated off in turn and the
full v20 pipeline is re-run on three distributions:

  - dev            real competition crops (curly-quote gold)
  - synth_val      synthetic paragraphs (ASCII-quote gold)
  - hard_synth_val augmented synthetic (ASCII-quote gold, dev-overfit detector)

A rule that lowers CER on dev but raises it on synth_val / hard is overfit to
the dev label convention and is a test-set liability.

Scoring is RAW jiwer cer - the organizer's exact metric, no curly
normalization. Raw scoring is the point: it is the only lens under which a
curly-quote rule that helps real dev gold but regresses ASCII-quote synth gold
shows up as a regression.

The five rules and where they live in competition_transcriber.py:
  v16 _fix_lead_marker           leading "N -" -> "N — "
  v17 _fix_apostrophe            ASCII ' -> curly (closing U+2019 default)
  v18 _fix_apostrophe (opening)  positional opening U+2018 branch
  v19 _CrossEngineRouter._decide diacritic-restoration cross-stream vote
  v20 _fix_doublequote           ASCII " -> curly U+201C / U+201D

Rules are all post-Tesseract, so the 5 Tesseract streams are run once per
image and cached; the 6 ablation configs then replay in pure Python.

Usage:
    PYTHONPATH=. python scripts/audit_postproc_overfit.py
    PYTHONPATH=. python scripts/audit_postproc_overfit.py --limit 20  # smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import PIL.Image
import pytesseract
from jiwer import cer

TESSDATA = "data/tesseract/tessdata"
PSM_CONFIG = f'--tessdata-dir "{TESSDATA}" --psm 6'
CONF_PATH = "data/tess_confusion.json"
LEX_PATH = "data/maltese_en_it_lexicon.json"
EPS = 1e-5

RULES = ["v16_lead", "v17_apostrophe", "v18_positional_quote",
         "v19_diacritic_vote", "v20_doublequote"]
SETS = ["dev", "synth_val", "hard"]


def _load_module():
    spec = importlib.util.spec_from_file_location("ct", "competition_transcriber.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ct = _load_module()


def _load_corrector_router():
    with open(CONF_PATH, encoding="utf-8") as f:
        cdata = json.load(f)
    with open(LEX_PATH, encoding="utf-8") as f:
        ldata = json.load(f)
    lex = set(ldata.keys() if isinstance(ldata, dict) else ldata)
    lex |= {w.lower() for w in lex}
    corrector = ct._ConfusionCorrector(cdata["by_tess_char"], lex, tau=ct._CORRECTOR_TAU)
    router = ct._CrossEngineRouter(lex, max_swap_dist=ct._CROSS_ENGINE_MAX_SWAP_DIST)
    return corrector, router


# ----- ablated rule variants -------------------------------------------------

def _fix_apostrophe_closing_only(text: str) -> str:
    # v17 without v18: every ASCII apostrophe becomes the closing/clitic U+2019,
    # the opening positional U+2018 branch is removed.
    return text.replace("'", "’")


def _decide_no_diac(router, anchor: str, candidate: Optional[str]) -> str:
    # ct._CrossEngineRouter._decide with the v19 diacritic-restoration block
    # (the first conditional) removed. Everything else is byte-identical.
    if not candidate or candidate == anchor:
        return anchor
    if router._in_lex(anchor):
        return anchor
    if not router._in_lex(candidate):
        return anchor
    a_alpha = sum(c.isalpha() for c in anchor)
    c_alpha = sum(c.isalpha() for c in candidate)
    if c_alpha < a_alpha or a_alpha < 3 or c_alpha < 3:
        return anchor
    if len(candidate) < len(anchor) - 1 or abs(len(anchor) - len(candidate)) > 2:
        return anchor
    if ct._non_ascii_alpha(candidate) < ct._non_ascii_alpha(anchor):
        return anchor
    d = ct._edit_distance(anchor, candidate)
    if d == 0 or d > router.max_swap_dist:
        return anchor
    return candidate


def _combine_lv_no_diac(router, anchor: str, candidate_streams: List[str]) -> str:
    # Copy of ct._CrossEngineRouter.combine_lv that calls _decide_no_diac.
    from collections import Counter
    ws = ct._WS_SPLIT
    cand_token_lists = [[w for w in ws.split(c.replace("\n", " ").strip()) if w]
                        for c in candidate_streams]
    cursors = [0] * len(candidate_streams)
    out_lines = []
    for anchor_line in anchor.split("\n"):
        a_words = [w for w in ws.split(anchor_line.strip()) if w]
        if not a_words:
            out_lines.append("")
            continue
        aligned_per_stream = []
        for k, tokens in enumerate(cand_token_lists):
            window = tokens[cursors[k]: cursors[k] + 2 * len(a_words)]
            alignment = ct._align_word_seqs(a_words, window)
            per_pos: List[Optional[str]] = []
            for a, b in alignment:
                if a is None:
                    continue
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
                swap = _decide_no_diac(router, anc, c)
                if swap != anc:
                    if swap not in votes:
                        stream_order.append(swap)
                    votes[swap] += 1
            if votes:
                best = max(stream_order, key=lambda w: (votes[w], -stream_order.index(w)))
                line_out.append(best)
            else:
                line_out.append(anc)
        out_lines.append(" ".join(line_out))
    return "\n".join(out_lines)


# ----- pipeline replay -------------------------------------------------------

def _base_select(streams: Dict[str, str]) -> str:
    mlt, ita = streams["mlt"], streams["ita"]
    present = [s for s in streams.values() if s]
    base = ita if ita else mlt
    if present:
        longest = max(present, key=len)
        if len(longest) > 10 and len(base) < ct._ITA_FALLBACK_RATIO * len(longest):
            base = longest
    return base


def replay(streams: Dict[str, str], corrector, router, off: Optional[str]) -> str:
    """Replay the post-Tesseract pipeline with rule `off` gated out.

    `streams` holds the joined+normalised Tesseract outputs keyed mlt, ita,
    romance, stock, up. EasyOCR is absent on the dev box (1 fewer candidate).
    """
    present = [s for s in (streams["mlt"], streams["ita"], streams["romance"],
                           streams["stock"], streams["up"]) if s]
    if not present:
        return ""
    base = _base_select(streams)
    joined = base
    if corrector is not None and len(joined) >= ct._CORRECTOR_LEN_THR:
        joined = corrector.correct(joined)
    candidates = []
    for cand in (streams["mlt"], streams["ita"], streams["romance"],
                 streams["stock"], streams["up"]):
        if cand and cand != base and cand not in candidates:
            candidates.append(cand)
    if candidates:
        if off == "v19_diacritic_vote":
            joined = _combine_lv_no_diac(router, joined, candidates)
        else:
            joined = router.combine_lv(joined, candidates)

    lead = (lambda t: t) if off == "v16_lead" else ct._fix_lead_marker
    if off == "v17_apostrophe":
        apos = (lambda t: t)
    elif off == "v18_positional_quote":
        apos = _fix_apostrophe_closing_only
    else:
        apos = ct._fix_apostrophe
    dquote = (lambda t: t) if off == "v20_doublequote" else ct._fix_doublequote
    return dquote(apos(lead(joined)))


# ----- Tesseract streams (run once per image, cached) ------------------------

def _tess(joiner, img, lang):
    raw = pytesseract.image_to_string(img, lang=lang, config=PSM_CONFIG)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    return ct._normalise(joiner.join_lines(lines, fix_hyphenated_words=True))


def streams_for(joiner, img) -> Dict[str, str]:
    def safe(lang, image=img):
        try:
            return _tess(joiner, image, lang)
        except Exception:
            return ""
    up = ""
    try:
        w, h = img.size
        up = _tess(joiner, img.resize((w * 2, h * 2), PIL.Image.LANCZOS), "mlt+ita")
    except Exception:
        up = ""
    return {
        "mlt": safe("mlt"),
        "ita": safe("mlt+ita"),
        "romance": safe("mlt+ita+fra"),
        "stock": safe("mltstock"),
        "up": up,
    }


# ----- dataset loaders -------------------------------------------------------

def load_dev(limit: int) -> List[dict]:
    rows = [json.loads(l) for l in Path("outputs/campaign/dev_gold.jsonl").read_text().splitlines()]
    rows.sort(key=lambda d: d["id"])
    items = [{"id": d["id"], "image": f"competition_files/dev/{d['id']}", "gold": d["gold"]} for d in rows]
    return items[:limit] if limit else items


def load_hard(limit: int) -> List[dict]:
    data = json.loads(Path("data/synth_val_hard/meta.json").read_text())
    items = [{"id": Path(d["hard_image"]).name, "image": d["hard_image"], "gold": d["gold"]} for d in data]
    return items[:limit] if limit else items


def load_synth_val(limit: int, n: int = 120) -> List[dict]:
    out = []
    for bucket in ["L0", "L1", "L2", "L3"]:
        p = Path(f"data/mira_pairs/{bucket}/shard_000.jsonl")
        if not p.exists():
            continue
        for ln in p.read_text().splitlines()[:n // 4]:
            d = json.loads(ln)
            img = str(Path("data/mira_pairs") / bucket / d["image"].split("/", 1)[1])
            out.append({"id": f"{bucket}/{Path(img).name}", "image": img, "gold": d["gold"]})
    out = out[:n]
    return out[:limit] if limit else out


def _trigger_counts(items: List[dict]) -> Dict[str, int]:
    n_apos = sum("'" in it["gold"] or "‘" in it["gold"] or "’" in it["gold"] for it in items)
    n_dq = sum('"' in it["gold"] or "“" in it["gold"] or "”" in it["gold"] for it in items)
    import re
    lead = re.compile(r"^\s*\d{1,2}\s*[-–—]")
    n_lead = sum(bool(lead.match(it["gold"])) for it in items)
    return {"v16_lead": n_lead, "v17_apostrophe": n_apos, "v18_positional_quote": n_apos,
            "v19_diacritic_vote": len(items), "v20_doublequote": n_dq}


# ----- driver ----------------------------------------------------------------

def run_set(name: str, items: List[dict], corrector, router) -> Dict:
    from malti.line_joiner import RBLineJoiner
    joiner = RBLineJoiner()
    refs = [it["gold"] for it in items]
    configs = [None] + RULES  # None == full pipeline
    preds: Dict[Optional[str], List[str]] = {c: [] for c in configs}
    t0 = time.time()
    for it in items:
        img = PIL.Image.open(it["image"]).convert("RGB")
        st = streams_for(joiner, img)
        for c in configs:
            preds[c].append(replay(st, corrector, router, c))
    cers = {("full" if c is None else c): cer(refs, preds[c]) for c in configs}
    return {"cer": cers, "n": len(items), "wall_s": time.time() - t0,
            "triggers": _trigger_counts(items)}


def verdict_for(dev_delta: float, synth_delta: float, hard_delta: float) -> str:
    helps_dev = dev_delta > EPS
    hurts_synth = synth_delta < -EPS or hard_delta < -EPS
    inert = abs(dev_delta) <= EPS and abs(synth_delta) <= EPS and abs(hard_delta) <= EPS
    if inert:
        return "NEUTRAL"
    if helps_dev and hurts_synth:
        return "OVERFIT"
    if helps_dev:
        return "SAFE"
    if not helps_dev and not hurts_synth:
        return "NEUTRAL"
    return "GATE_TIGHTER"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap items per set (smoke)")
    args = ap.parse_args()

    corrector, router = _load_corrector_router()
    easyocr_available = False  # absent on the dev box; logged as caveat

    datasets = {
        "dev": load_dev(args.limit),
        "synth_val": load_synth_val(args.limit),
        "hard": load_hard(args.limit),
    }
    results = {}
    for name in SETS:
        items = datasets[name]
        print(f"[{name}] n={len(items)} running 6 configs (Tesseract once/img)...")
        results[name] = run_set(name, items, corrector, router)
        full = results[name]["cer"]["full"]
        print(f"  full CER={full:.5f}  wall={results[name]['wall_s']:.1f}s")

    rows = []
    for rule in RULES:
        d_full = results["dev"]["cer"]["full"]
        s_full = results["synth_val"]["cer"]["full"]
        h_full = results["hard"]["cer"]["full"]
        d_off = results["dev"]["cer"][rule]
        s_off = results["synth_val"]["cer"][rule]
        h_off = results["hard"]["cer"][rule]
        dev_delta = d_off - d_full
        synth_delta = s_off - s_full
        hard_delta = h_off - h_full
        v = verdict_for(dev_delta, synth_delta, hard_delta)
        rows.append((rule, d_full, s_full, h_full, dev_delta, synth_delta, hard_delta, v))

    print("\n| Rule | Dev CER | SynthVal CER | HardSynthVal CER | Dev delta | SynthVal delta | HardSynth delta | Verdict |")
    print("|------|---------|--------------|------------------|-----------|----------------|-----------------|---------|")
    for (rule, d, s, h, dd, sd, hd, v) in rows:
        print(f"| {rule} | {d:.5f} | {s:.5f} | {h:.5f} | {dd:+.5f} | {sd:+.5f} | {hd:+.5f} | {v} |")

    print("\ntriggers (items whose gold exercises the rule):")
    for name in SETS:
        tr = results[name]["triggers"]
        print(f"  {name:10s} n={results[name]['n']:4d}  " +
              "  ".join(f"{k.split('_')[0]}={tr[k]}" for k in RULES))
    print(f"\neasyocr_available={easyocr_available} (absent on dev box -> 4-stream router, not 5)")
    print("delta = CER_without_rule - CER_with_rule. positive = rule helps that set.")


if __name__ == "__main__":
    main()
