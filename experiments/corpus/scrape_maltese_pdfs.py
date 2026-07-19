"""I1 PDF scrape infrastructure.

Crawl Maltese government and news sites for PDFs. Fetch with 1s sleep,
respect robots.txt, hash-dedupe, save to data/unlabelled_real/<source>/<hash>.pdf.
Cap: 10k PDFs, 4h wall.

Sources:
  gov_mt:     https://www.gov.mt
  parlament:  https://parlament.mt
  justice:    https://justice.gov.mt
  tom_news:   https://timesofmalta.com/section/news/ (limited scraping)

Robots.txt is checked before crawling each domain.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_BASE = ROOT / "data" / "unlabelled_real"
MAX_PDFS = 10000
WALL_LIMIT_H = 4.0
SLEEP_S = 1.0
USER_AGENT = "MaltaOCR-research/1.0 (academic)"

SEEDS = {
    "gov_mt": [
        "https://www.gov.mt",
        "https://www.gov.mt/en/Government/DOI/",
        "https://www.gov.mt/en/Government/",
    ],
    "parlament": [
        "https://parlament.mt",
        "https://parlament.mt/en/parliamentary-business/",
        "https://parlament.mt/mt/parliamentary-business/",
    ],
    "justice": [
        "https://justice.gov.mt",
        "https://justice.gov.mt/en/Pages/Home.aspx",
    ],
}

PDF_RE = re.compile(r'href=["\']([^"\']+\.pdf)["\']', re.IGNORECASE)
LINK_RE = re.compile(r'href=["\']([^"\'#?]+)["\']', re.IGNORECASE)


def _get_robots(domain: str) -> RobotFileParser:
    rp = RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        pass
    return rp


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, timeout: int = 30) -> Optional[bytes]:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def scrape_source(
    source_name: str,
    seeds: list[str],
    robots: RobotFileParser,
    seen_hashes: Set[str],
    t_start: float,
    max_total: int,
    n_downloaded: list,
) -> None:
    out_dir = OUT_BASE / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = urlparse(seeds[0]).netloc
    queue: list[str] = list(seeds)
    visited: Set[str] = set()
    pdf_queue: list[str] = []

    while (queue or pdf_queue) and n_downloaded[0] < max_total:
        if (time.time() - t_start) / 3600 >= WALL_LIMIT_H:
            print(f"[scrape] wall limit reached at {n_downloaded[0]} PDFs")
            break

        if queue:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            parsed = urlparse(url)
            if parsed.netloc and parsed.netloc != domain:
                continue
            if not robots.can_fetch(USER_AGENT, url):
                continue

            html = _fetch(url)
            time.sleep(SLEEP_S)
            if html is None:
                continue

            try:
                text = html.decode("utf-8", errors="replace")
            except Exception:
                continue

            # Collect PDF links.
            for m in PDF_RE.finditer(text):
                pdf_url = urljoin(url, m.group(1))
                if pdf_url not in visited:
                    pdf_queue.append(pdf_url)

            # Collect page links on same domain.
            for m in LINK_RE.finditer(text):
                href = m.group(1)
                full = urljoin(url, href)
                fp = urlparse(full)
                if fp.netloc == domain and full not in visited:
                    queue.append(full)

        elif pdf_queue:
            pdf_url = pdf_queue.pop(0)
            if pdf_url in visited:
                continue
            visited.add(pdf_url)

            if not robots.can_fetch(USER_AGENT, pdf_url):
                continue

            data = _fetch(pdf_url)
            time.sleep(SLEEP_S)
            if data is None or not data.startswith(b"%PDF"):
                continue

            h = _sha256(data)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            out_file = out_dir / f"{h[:16]}.pdf"
            out_file.write_bytes(data)
            n_downloaded[0] += 1
            if n_downloaded[0] % 10 == 0:
                print(f"[scrape] {source_name}: {n_downloaded[0]} PDFs saved, queue={len(pdf_queue)}")


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seen_hashes: Set[str] = set()
    n_downloaded = [0]

    for source_name, seeds in SEEDS.items():
        if n_downloaded[0] >= MAX_PDFS:
            break
        if (time.time() - t0) / 3600 >= WALL_LIMIT_H:
            break
        domain = urlparse(seeds[0]).netloc
        print(f"[scrape] starting {source_name} (domain={domain})")
        robots = _get_robots(domain)
        scrape_source(source_name, seeds, robots, seen_hashes, t0, MAX_PDFS, n_downloaded)

    total_s = time.time() - t0
    print(f"[scrape] done. total PDFs={n_downloaded[0]} wall={total_s/3600:.2f}h")


if __name__ == "__main__":
    main()
