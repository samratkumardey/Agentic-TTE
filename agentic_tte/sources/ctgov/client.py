"""ClinicalTrials.gov API v2 client with an on-disk cache.

Ported from EmulatRx ``trialistUtils/web_utils.get_trial_info`` (MIT, TrialLab). Changes: an
honest User-Agent instead of a spoofed browser one, retries with backoff on 429/5xx, NCT ID
validation, and a cache so Replay runs are reproducible against the exact record they used.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from agentic_tte import __version__

NCT_RE = re.compile(r"^NCT\d{8}$")
USER_AGENT = f"agentic-tte/{__version__} (research; target trial emulation)"


class CTGovError(RuntimeError):
    pass


def normalize_nct_id(nct_id: str) -> str:
    nid = nct_id.strip().upper()
    if not NCT_RE.match(nid):
        raise ValueError(f"not a ClinicalTrials.gov NCT ID: {nct_id!r}")
    return nid


def record_sha256(study: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(study, sort_keys=True).encode()).hexdigest()


class CTGovClient:
    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2",
        cache_dir: Path | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        http: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) / "ctgov" if cache_dir else None
        self.max_retries = max_retries
        self._http = http or httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT})

    def get_study(self, nct_id: str, *, refresh: bool = False) -> dict[str, Any]:
        """Full v2 study record. Served from the cache unless ``refresh``; the fetch time is
        stored next to the record as ``_retrieved_at``."""
        nid = normalize_nct_id(nct_id)
        path = self.cache_dir / f"{nid}.json" if self.cache_dir else None
        if path and path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))

        study = self._get(f"/studies/{nid}")
        study["_retrieved_at"] = datetime.now(UTC).isoformat()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(study, indent=2), encoding="utf-8")
        return study

    def _get(self, endpoint: str) -> dict[str, Any]:
        url = self.base_url + endpoint
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.get(url)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise CTGovError(f"GET {url} failed: {exc}") from exc
            else:
                if resp.status_code == 404:
                    raise CTGovError(f"study not found: {url}")
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise CTGovError(f"GET {url} -> HTTP {resp.status_code}")
            time.sleep(min(2**attempt, 30))
        raise CTGovError(f"GET {url} failed")  # unreachable

    def close(self) -> None:
        self._http.close()
