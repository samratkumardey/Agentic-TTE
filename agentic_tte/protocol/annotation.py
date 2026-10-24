"""Parser for the tagged annotation format used by EmulatRx / Criteria2Query 3.0.

Example: ``[Annotation] <Demographic>Age</Demographic> <Value>18 years old or older</Value>``.
EmulatRx's Trialist and Clinician exchange this format, and upstream's cached parses use it,
so we read it into typed fields. Our own agents emit JSON instead (see agents/trialist.py).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agentic_tte.protocol.models import ConceptRef, Domain

# Upstream outputs are sometimes malformed (``<Demographic>aged<Demographic>``), so the closing
# slash is optional.
_TAG_RE = re.compile(r"<([A-Za-z_]+)>(.*?)</?\1>", re.DOTALL)
_DOMAIN_TAGS = {d.value.lower(): Domain(d.value) for d in Domain}
_DOMAIN_TAGS["demographics"] = Domain.DEMOGRAPHIC


class Annotation(BaseModel):
    concepts: list[ConceptRef] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    temporals: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    other: dict[str, list[str]] = Field(default_factory=dict)  # Provider, Quantity, ...


def parse_annotation(markup: str) -> Annotation:
    out = Annotation()
    for tag, raw in _TAG_RE.findall(markup or ""):
        text = " ".join(raw.split())
        if not text:
            continue
        key = tag.lower()
        if key in _DOMAIN_TAGS:
            out.concepts.append(ConceptRef(name=text, domain=_DOMAIN_TAGS[key], source_text=text))
        elif key == "value":
            out.values.append(text)
        elif key == "temporal":
            out.temporals.append(text)
        elif key == "negation_cue":
            out.negations.append(text)
        else:
            out.other.setdefault(tag, []).append(text)
    return out
