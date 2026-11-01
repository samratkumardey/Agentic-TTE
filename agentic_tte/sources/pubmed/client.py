"""PubMed E-utilities client (esearch, efetch) with an on-disk cache, for the Clinician agent.

Searches and articles are cached under ``<cache_dir>/pubmed``, so a corpus can be rebuilt
against the exact records it used. Abstracts are publishers' text: the cache stays out of git
(``.cache/`` is ignored). No user identity is sent: only the tool name, as NCBI asks.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from agentic_tte.sources.ctgov.client import USER_AGENT

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_NCT = re.compile(r"\bNCT\d{8}\b")


class PubMedError(RuntimeError):
    pass


class Article(BaseModel):
    pmid: str
    title: str
    abstract: str
    year: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    nct_ids: list[str] = Field(default_factory=list)  # registry numbers in the record

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


def parse_articles(xml: str) -> list[Article]:
    """Articles with an abstract from an efetch ``PubmedArticleSet``."""
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        title = "".join(art.find(".//ArticleTitle").itertext()).strip() \
            if art.find(".//ArticleTitle") is not None else ""  # fmt: skip
        parts = []
        for ab in art.iter("AbstractText"):
            body = "".join(ab.itertext()).strip()
            label = ab.get("Label")
            parts.append(f"{label}: {body}" if label else body)
        abstract = " ".join(p for p in parts if p)
        if not abstract:
            continue
        year = art.findtext(".//PubDate/Year") or (art.findtext(".//PubDate/MedlineDate") or "")[:4]
        nct = {a.text.strip() for a in art.iter("AccessionNumber")
               if a.text and _NCT.match(a.text.strip())}  # fmt: skip
        nct |= set(_NCT.findall(f"{title} {abstract}"))
        out.append(Article(pmid=pmid, title=title, abstract=abstract,
                           year=int(year) if year.isdigit() else None,
                           publication_types=[t.text for t in art.iter("PublicationType")
                                              if t.text],
                           nct_ids=sorted(nct)))  # fmt: skip
    return out


class PubMedClient:
    def __init__(self, cache_dir: Path | None = None, *, base_url: str = EUTILS,
                 timeout_s: float = 30.0, max_retries: int = 3, pause_s: float = 0.4,
                 http: httpx.Client | None = None):  # fmt: skip
        self.cache_dir = Path(cache_dir) / "pubmed" if cache_dir else None
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.pause_s = pause_s  # NCBI allows 3 requests a second without an API key
        self._http = http or httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT})

    def search(self, term: str, *, retmax: int = 40, maxdate: str | None = None) -> list[str]:
        """PMIDs for ``term``, by relevance; ``maxdate`` (YYYY/MM/DD) limits publication date."""
        params = {"db": "pubmed", "term": term, "retmax": str(retmax), "retmode": "json",
                  "sort": "relevance", "tool": "agentic-tte"}  # fmt: skip
        if maxdate:
            params |= {"datetype": "pdat", "mindate": "1900/01/01", "maxdate": maxdate}
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache_dir / "search" / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["pmids"]
        data = json.loads(self._get("esearch.fcgi", params))
        pmids = data.get("esearchresult", {}).get("idlist", [])
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"params": params, "pmids": pmids}, indent=1),
                            encoding="utf-8")
        return pmids

    def fetch(self, pmids: list[str]) -> list[Article]:
        """Articles with an abstract, in the order asked; cached one file per PMID."""
        have: dict[str, Article | None] = {}
        todo = []
        for pmid in dict.fromkeys(pmids):
            path = self.cache_dir / "articles" / f"{pmid}.json" if self.cache_dir else None
            if path and path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                have[pmid] = Article.model_validate(d) if d else None
            else:
                todo.append(pmid)
        for i in range(0, len(todo), 100):
            chunk = todo[i:i + 100]
            got = {a.pmid: a for a in parse_articles(self._get("efetch.fcgi", {
                "db": "pubmed", "id": ",".join(chunk), "retmode": "xml",
                "tool": "agentic-tte"}))}  # fmt: skip
            for pmid in chunk:
                have[pmid] = got.get(pmid)
                if self.cache_dir:  # an empty file records "no abstract"
                    path = self.cache_dir / "articles" / f"{pmid}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    a = got.get(pmid)
                    path.write_text(a.model_dump_json(indent=1) if a else "{}", encoding="utf-8")
        return [have[p] for p in dict.fromkeys(pmids) if have.get(p)]

    def _get(self, endpoint: str, params: dict) -> str:
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(self.max_retries + 1):
            time.sleep(self.pause_s)
            try:
                resp = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise PubMedError(f"GET {url} failed: {exc}") from exc
            else:
                if resp.status_code < 400:
                    return resp.text
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise PubMedError(f"GET {url} -> HTTP {resp.status_code}")
            time.sleep(min(2**attempt, 30))
        raise PubMedError(f"GET {url} failed")  # unreachable

    def close(self) -> None:
        self._http.close()
