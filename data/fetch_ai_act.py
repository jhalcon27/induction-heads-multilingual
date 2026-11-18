"""Fetch the recitals of the EU AI Act, Regulation (EU) 2024/1689, in EN/ES/DE/SV.

Source: the EU Publications Office's machine-access endpoint (cellar) for CELEX 32024R1689, which
serves the official XHTML of each language version (the EUR-Lex web pages block scripted downloads).
Writes data/ai_act/recitals_<lang>.txt: one recital per line, recitals 1-180 in order, whitespace
normalised, footnote reference markers such as "(11)" removed.

    python data/fetch_ai_act.py                 # download
    python data/fetch_ai_act.py --raw-dir DIR   # use saved ai_act_32024R1689_<eng|spa|deu|swe>.xhtml files
"""
import argparse
import html
import re
import urllib.request
from pathlib import Path

CELEX_URL = "http://publications.europa.eu/resource/celex/32024R1689"
LANGS = {"en": "eng", "es": "spa", "de": "deu", "sv": "swe"}
N_RECITALS = 180
OUT = Path(__file__).parent / "ai_act"


def download(code: str) -> str:
    req = urllib.request.Request(CELEX_URL, headers={"Accept": "application/xhtml+xml", "Accept-Language": code})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def recitals(xhtml: str) -> list[str]:
    found = {}
    for m in re.finditer(r'<div class="eli-subdivision" id="rct_(\d+)">(.*?)</table>', xhtml, re.S):
        body = re.findall(r"<td[^>]*>(.*?)</td>", m.group(2), re.S)[-1]       # [number, text]
        body = re.sub(r'<a id="ntc[^"]*"[^>]*>.*?</a>', "", body, flags=re.S)  # footnote references
        found[int(m.group(1))] = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", body)).split())
    assert sorted(found) == list(range(1, N_RECITALS + 1)), "unexpected recital numbering"
    return [found[i] for i in range(1, N_RECITALS + 1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for lang, code in LANGS.items():
        if args.raw_dir:
            xhtml = (args.raw_dir / f"ai_act_32024R1689_{code}.xhtml").read_text(encoding="utf-8")
        else:
            xhtml = download(code)
        rec = recitals(xhtml)
        (OUT / f"recitals_{lang}.txt").write_text("\n".join(rec) + "\n", encoding="utf-8")
        print(lang, len(rec), "recitals,", sum(len(r.split()) for r in rec), "words")
