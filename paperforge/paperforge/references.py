"""Step 6: the LLM proposes related work; only entries confirmed by Crossref or arXiv are kept."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import requests

from .llm import LLM, ask_json

Fetch = Callable[[str], "str | dict | None"]   # url -> parsed JSON (Crossref) or text (arXiv), None if unreachable
UA = "paperforge/0.1 (reference verification)"


def http_fetch(url: str):
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.json() if "json" in resp.headers.get("content-type", "") else resp.text


def offline_fetch(url: str):
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _esc(s: str) -> str:
    return re.sub(r"([&%#_$])", r"\\\1", s)


def _bib_from_crossref(key: str, w: dict) -> str:
    authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in w.get("author", [])) or "[TODO: authors]"
    year = next((str(w[k]["date-parts"][0][0]) for k in ("published", "published-print", "issued")
                 if w.get(k, {}).get("date-parts", [[None]])[0][0]), "")
    venue = (w.get("container-title") or [""])[0]
    kind = "inproceedings" if w.get("type") == "proceedings-article" else "article"
    fields = {"author": authors, "title": "{" + _esc((w.get("title") or [""])[0]) + "}",
              "booktitle" if kind == "inproceedings" else "journal": _esc(venue), "year": year,
              "volume": w.get("volume", ""), "pages": w.get("page", ""), "doi": w.get("DOI", "")}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}"


def _verify(cand: dict, fetch: Fetch) -> tuple[str | None, str]:
    """Return (bibtex without key placeholder resolved, reason)."""
    title, key = str(cand.get("title", "")), cand["key"]
    doi, arxiv = cand.get("doi"), cand.get("arxiv_id")
    if doi:
        data = fetch("https://api.crossref.org/works/" + quote(str(doi), safe="/"))
        if isinstance(data, dict) and similar(title, (data["message"].get("title") or [""])[0]) >= 0.85:
            return _bib_from_crossref(key, data["message"]), "Crossref DOI"
    if arxiv:
        text = fetch("https://export.arxiv.org/api/query?id_list=" + quote(str(arxiv)))
        if isinstance(text, str):
            ns = {"a": "http://www.w3.org/2005/Atom"}
            try:
                entry = ET.fromstring(text).find("a:entry", ns)
            except ET.ParseError:
                entry = None
            if entry is not None and entry.find("a:title", ns) is not None \
                    and similar(title, entry.findtext("a:title", "", ns)) >= 0.85:
                authors = " and ".join(a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns))
                year = entry.findtext("a:published", "", ns)[:4]
                real = " ".join(entry.findtext("a:title", "", ns).split())
                return (f"@misc{{{key},\n  author = {{{authors}}},\n  title = {{{{{_esc(real)}}}}},\n  year = {{{year}}},\n"
                        f"  eprint = {{{arxiv}}},\n  archivePrefix = {{arXiv}}\n}}"), "arXiv id"
    if title:
        data = fetch("https://api.crossref.org/works?rows=3&query.bibliographic=" + quote(title))
        if isinstance(data, dict):
            for item in data["message"].get("items", []):
                if similar(title, (item.get("title") or [""])[0]) >= 0.9:
                    return _bib_from_crossref(key, item), "Crossref title match"
    return None, "not found in Crossref/arXiv (or lookup unreachable)"


@dataclass
class Bibliography:
    text: str
    keys: dict[str, str]                       # key -> short description for the writer
    rejected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


PROMPT = """Propose up to {n} published works this manuscript should cite as background and related work, based on the repository material and plan.
Only propose works you are confident exist; an unverifiable entry is dropped automatically, so guessing gains nothing.
Reply with JSON: [{{"key": "authorYEARword", "title": "...", "authors": "...", "year": 2020, "doi": "10.xxxx/..." or null, "arxiv_id": "2003.08934" or null, "why": "one line on what it is cited for"}}]"""


def build(llm: LLM, context: str, repo_bibs: list[Path], fetch: Fetch = http_fetch, n: int = 20) -> Bibliography:
    chunks, keys, rejected, notes = [], {}, [], []
    for bib in repo_bibs:  # the authors' own .bib files are trusted as-is
        text = bib.read_text(errors="replace")
        new = []
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
            if m.group(1).lower() not in {"comment", "string", "preamble"} and m.group(2) not in keys:
                title = re.search(r"title\s*=\s*[{\"]+([^}\"]+)", text[m.end(): m.end() + 1500])
                keys[m.group(2)] = "from the repo's .bib" + (f": {title.group(1)}" if title else "")
                new.append(m.group(2))
        if new:
            chunks.append(f"% ---- from repository file {bib.name} ----\n{text.strip()}")
            notes.append(f"{len(new)} entries taken from {bib.name}")

    candidates = ask_json(llm, "refs", PROMPT.format(n=n), context)
    for cand in candidates if isinstance(candidates, list) else []:
        if not isinstance(cand, dict) or not cand.get("title"):
            continue
        cand["key"] = re.sub(r"[^A-Za-z0-9:_-]", "", str(cand.get("key") or "")) or f"ref{len(keys) + 1}"
        if cand["key"] in keys:
            continue
        entry, reason = _verify(cand, fetch)
        if entry:
            chunks.append(f"% verified via {reason}\n{entry}")
            keys[cand["key"]] = f"{cand['title']} ({cand.get('year', '')}) - {cand.get('why', '')}"
        else:
            rejected.append(f"{cand['key']}: \"{cand['title']}\" - {reason}")
    return Bibliography("\n\n".join(chunks) + "\n", keys, rejected, notes)
