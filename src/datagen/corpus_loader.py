"""Streamed corpus loader.

Pulls paragraph-shaped chunks from `MLRS/korpus_malti` v4.2 domain-split configs
per the design decision (no global shuffle), interleaves with 10-15 percent English from a
small clean source. Falls back to an embedded Maltese fixture when the dataset
cache is not present and the run is offline.

`public_fallback_stream` is the unblock path while korpus access is pending. It
pulls from `wikimedia/wikipedia` config `20231101.mt` (CC-BY-SA-3.0, public,
ungated) and the UD Maltese MUDT treebank (CC-BY-SA-4.0, public). Both license-
clean.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

DOMAINS = ("news", "law", "literature", "religion", "academic", "parliament", "europarl")

CANARY_CHARS: Tuple[str, ...] = ("ċ", "ġ", "ħ", "ż", "Ċ", "Ġ", "Ħ", "Ż")

PUBLIC_SOURCES_DEFAULT: List[Dict] = [
    {"name": "wikipedia_mt", "kind": "hf", "path": "wikimedia/wikipedia",
     "config": "20231101.mt", "split": "train", "text_field": "text", "weight": 0.8},
    {"name": "ud_mudt", "kind": "ud_conllu", "weight": 0.2,
     "urls": [
         "https://raw.githubusercontent.com/UniversalDependencies/UD_Maltese-MUDT/master/mt_mudt-ud-train.conllu",
         "https://raw.githubusercontent.com/UniversalDependencies/UD_Maltese-MUDT/master/mt_mudt-ud-dev.conllu",
         "https://raw.githubusercontent.com/UniversalDependencies/UD_Maltese-MUDT/master/mt_mudt-ud-test.conllu",
     ]},
]

MT_FIXTURE: List[str] = [
    "Il-Gvern Malti qed jaħdem fuq strateġija ġdida għall-iżvilupp ekonomiku. "
    "Il-Ministru tal-Finanzi ħabbar miżuri li jolqtu lil kulħadd, b'mod partikulari "
    "lill-familji bi dħul baxx. Il-pjan jinkludi investiment fis-saħħa u fl-edukazzjoni.",
    "Fl-okkażjoni ta' Jum ir-Repubblika, il-President ta diskors qasir fuq is-solidarjetà "
    "u l-għaqda nazzjonali. Hu fakkar lis-soċjetà fl-importanza tal-valuri ċivili u "
    "fil-ħtieġa li nibqgħu ngħożżu d-demokrazija.",
    "Ir-riċerkaturi tal-Università ta' Malta ppubblikaw studju ġdid dwar il-bidla "
    "fil-klima fil-Mediterran. Ir-riżultati juru li t-temperaturi tal-baħar qed jogħlew "
    "b'rata mgħaġġla, ħaġa li tolqot direttament lis-sajd u t-turiżmu.",
    "Fis-seħħ illum jidħlu regoli ġodda dwar it-traffiku fil-bliet ewlenin. "
    "Il-pulizija sejra tinforza l-limiti tal-veloċità b'aktar serjetà. Ix-xufiera huma "
    "mħeġġa jirrispettaw is-sinjali tat-triq u joqogħdu attenti għall-pedestrijani.",
    "L-għada filgħodu, il-folla nġabret fil-pjazza prinċipali biex tesprimi s-solidarjetà "
    "tagħha mal-ħaddiema. Ir-rappreżentanti tat-trade unions tkellmu favur paga "
    "minima diċenti u kundizzjonijiet xierqa tax-xogħol.",
    "Il-każ tressaq quddiem il-qorti tal-maġistrati ilbieraħ — l-imputat wieġeb mhux "
    "ħati. L-avukat tad-difiża talab li jingħata l-ħelsien mill-arrest bil-kundizzjonijiet "
    "tas-soltu, inkluż il-firma fl-għassa l-aktar viċin.",
    "L-orizzont kien imdallam meta s-sajjieda telqu mill-port qabel is-sebħ. Id-dgħajjes "
    "tagħhom kienu mgħobbija bix-xbieki u l-armar kollu meħtieġ għal ġurnata twila fuq "
    "il-baħar miftuħ.",
    "Il-konferenza tinżamm bejn it-12 u l-14 ta' Ġunju fil-Belt Valletta. Fost il-kelliema "
    "hemm akkademiċi minn diversi pajjiżi Ewropej. Il-programm jinkludi sessjonijiet "
    "dwar l-istorja, il-letteratura u l-arti Maltija.",
    "ĊENSURA, ĠID, ĦAJJA u ŻGUR huma kelmiet b'ittri kapitali bid-djakritiċi Maltin. "
    "À tort jew À raġun, dan it-test isemmi ismijiet Taljani bħal È vero u perspettivi "
    "soċjali fil-Mediterran. Il-każ jeħtieġ studju aktar profond.",
]

EN_FIXTURE: List[str] = [
    "The conference will run from 12 to 14 June in Valletta. Speakers include "
    "academics from several European countries. The programme covers history, "
    "literature, and the arts.",
    "Researchers at the University of Malta published a new study on climate "
    "change in the Mediterranean. Sea temperatures are rising faster than the "
    "global mean, with direct consequences for fishing and tourism.",
    "New traffic rules came into force today in the main towns. Police will "
    "enforce speed limits more strictly. Drivers are urged to respect road "
    "signs and watch out for pedestrians.",
]


def _hf_token() -> Optional[str]:
    import os
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    from pathlib import Path
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            return v or None
    return None


def _try_streaming_korpus(domains: Sequence[str], min_len: int = 60):
    try:
        from datasets import load_dataset
    except Exception:
        return None
    token = _hf_token()
    out = []
    for d in domains:
        try:
            ds = load_dataset(
                "MLRS/korpus_malti",
                d,
                split="train",
                streaming=True,
                token=token,
            )
            out.append(("mt", iter(ds)))
        except Exception:
            continue
    return out or None


# ------------------------------------------------------------------
# Public fallback path
# ------------------------------------------------------------------

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_WS = re.compile(r"\s+")


def _chunk_text_into_paragraphs(
    text: str, min_chars: int = 200, max_chars: int = 1200
) -> Iterator[str]:
    """Yield paragraph-shaped chunks. Splits on blank lines, then re-packs
    sentence-wise until each chunk lands in [min_chars, max_chars]."""
    for block in _PARA_SPLIT.split(text):
        block = block.strip()
        if not block:
            continue
        block = _WS.sub(" ", block)
        if min_chars <= len(block) <= max_chars:
            yield block
            continue
        if len(block) < min_chars:
            continue
        # too long: split on sentence boundary, repack
        sents = re.split(r"(?<=[.!?])\s+", block)
        buf = ""
        for s in sents:
            if not s:
                continue
            if len(buf) + 1 + len(s) <= max_chars:
                buf = (buf + " " + s).strip() if buf else s
                if len(buf) >= min_chars:
                    yield buf
                    buf = ""
            else:
                if len(buf) >= min_chars:
                    yield buf
                buf = s
        if min_chars <= len(buf) <= max_chars:
            yield buf


def _iter_wikipedia_mt(source: Dict) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(
        source["path"], source["config"], split=source.get("split", "train"),
        streaming=True,
    )
    field = source.get("text_field", "text")
    for row in ds:
        text = row.get(field) or ""
        if not text:
            continue
        for chunk in _chunk_text_into_paragraphs(text):
            yield chunk


def _iter_ud_conllu(source: Dict) -> Iterator[str]:
    """Pull sentences from a UD CoNLL-U file via raw URL, repack into paragraphs."""
    import urllib.request
    sents: List[str] = []
    for url in source["urls"]:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            LOG.warning("ud_conllu fetch failed: %s (%s)", url, e)
            continue
        for line in raw.splitlines():
            if line.startswith("# text = "):
                sents.append(line[len("# text = "):].strip())
    if not sents:
        return
    # Re-pack consecutive sentences into paragraph-shaped chunks.
    buf = ""
    for s in sents:
        if len(buf) + 1 + len(s) <= 1000:
            buf = (buf + " " + s).strip() if buf else s
        else:
            if len(buf) >= 150:
                yield buf
            buf = s
    if len(buf) >= 150:
        yield buf


def _canary_freqs(text: str) -> Dict[str, float]:
    n = max(1, len(text))
    return {c: 1000.0 * text.count(c) / n for c in CANARY_CHARS}


def _open_source(source: Dict) -> Iterator[str]:
    kind = source.get("kind")
    if kind == "hf":
        return _iter_wikipedia_mt(source)
    if kind == "ud_conllu":
        return _iter_ud_conllu(source)
    raise ValueError(f"unknown public source kind: {kind!r}")


def _scan_canaries(
    sources: List[Dict], sample_chars: int = 10000
) -> Dict[str, Dict[str, float]]:
    """Sample up to ~sample_chars per source and report canary frequency per
    1k chars. Maltese baseline: hroughly 5x ċ. Logs a warning on miss."""
    report: Dict[str, Dict[str, float]] = {}
    for source in sources:
        name = source["name"]
        buf = ""
        try:
            for chunk in _open_source(source):
                buf += " " + chunk
                if len(buf) >= sample_chars:
                    break
        except Exception as e:
            LOG.warning("public source %s canary scan failed: %s", name, e)
            report[name] = {"error": -1.0}
            continue
        freqs = _canary_freqs(buf[:sample_chars])
        report[name] = freqs
        h, c = freqs.get("ħ", 0.0), freqs.get("ċ", 0.0)
        ratio = (h / c) if c > 0.01 else float("inf")
        LOG.info(
            "canary[%s] chars=%d hbar/cdot=%.1f ; %s",
            name, min(len(buf), sample_chars), ratio,
            ", ".join(f"{k}={v:.2f}" for k, v in freqs.items()),
        )
        if h < 0.5:
            LOG.warning("public source %s has suspiciously low h-bar rate (%.2f/1k)", name, h)
    return report


def public_fallback_stream(
    config: Dict,
    rng: Optional[random.Random] = None,
    english_frac: float = 0.12,
    do_canary_scan: bool = True,
) -> Iterator[dict]:
    """Interleave paragraphs from public Maltese sources by weight.

    config keys:
      sources: list of source dicts (see PUBLIC_SOURCES_DEFAULT)
      english_frac: optional override; intermixes EN_FIXTURE at this rate
    """
    rng = rng or random.Random()
    sources: List[Dict] = list(config.get("sources") or PUBLIC_SOURCES_DEFAULT)
    if not sources:
        raise ValueError("public_fallback_stream: no sources configured")
    english_frac = float(config.get("english_frac", english_frac))

    if do_canary_scan:
        try:
            _scan_canaries(sources)
        except Exception as e:
            LOG.warning("canary scan skipped: %s", e)

    iters: List[Iterator[str]] = []
    weights: List[float] = []
    names: List[str] = []
    for source in sources:
        try:
            iters.append(_open_source(source))
            weights.append(float(source.get("weight", 1.0)))
            names.append(source["name"])
        except Exception as e:
            LOG.warning("public source %s unavailable: %s", source.get("name"), e)
    if not iters:
        raise RuntimeError("public_fallback_stream: all sources failed to open")

    while True:
        if rng.random() < english_frac:
            yield {"text": rng.choice(EN_FIXTURE), "lang": "en", "domain": "fixture-en"}
            continue
        idx = _weighted_choice(weights, rng)
        try:
            text = next(iters[idx])
        except StopIteration:
            iters[idx] = _open_source(sources[idx])
            try:
                text = next(iters[idx])
            except StopIteration:
                continue
        yield {"text": text, "lang": "mt", "domain": names[idx]}


def _weighted_choice(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if r <= upto:
            return i
    return len(weights) - 1


# ------------------------------------------------------------------
# Top-level iterator
# ------------------------------------------------------------------


def iter_paragraphs(
    english_frac: float = 0.12,
    use_streaming: bool = False,
    rng: Optional[random.Random] = None,
    mode: str = "korpus_malti",
    public_config: Optional[Dict] = None,
) -> Iterator[dict]:
    """Yield paragraph dicts.

    mode:
      - "korpus_malti": primary path. Streams gated MLRS/korpus_malti when
        HF_TOKEN has access; otherwise falls through to the embedded fixture.
      - "public_fallback": pulls from license-clean public sources only
        (Wikipedia mt + UD-MUDT by default). No HF_TOKEN required.
      - "fixture": forces the embedded fixture path. Used by smoke tests.
    """
    rng = rng or random.Random()

    if mode == "public_fallback":
        yield from public_fallback_stream(
            public_config or {}, rng=rng, english_frac=english_frac
        )
        return

    if mode == "fixture":
        while True:
            if rng.random() < english_frac:
                yield {"text": rng.choice(EN_FIXTURE), "lang": "en", "domain": "fixture"}
            else:
                yield {"text": rng.choice(MT_FIXTURE), "lang": "mt", "domain": "fixture"}
        return

    streams = _try_streaming_korpus(DOMAINS) if use_streaming else None

    if streams is None:
        while True:
            if rng.random() < english_frac:
                yield {"text": rng.choice(EN_FIXTURE), "lang": "en", "domain": "fixture"}
            else:
                yield {"text": rng.choice(MT_FIXTURE), "lang": "mt", "domain": "fixture"}
        return

    si = 0
    while True:
        if rng.random() < english_frac:
            yield {"text": rng.choice(EN_FIXTURE), "lang": "en", "domain": "fixture-en"}
            continue
        lang, it = streams[si % len(streams)]
        si += 1
        try:
            row = next(it)
        except StopIteration:
            continue
        raw = row.get("text") or row.get("content") or ""
        if isinstance(raw, list):
            raw = "\n".join(str(x) for x in raw)
        text = raw.strip()
        if len(text) < 60:
            continue
        yield {"text": text, "lang": lang, "domain": "korpus_malti"}
