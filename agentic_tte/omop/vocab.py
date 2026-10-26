"""OMOP vocabulary search: protocol concept text -> candidate standard concepts.

Concept IDs come only from the vocabulary tables, never from an LLM's memory (PLAN.md §1). The
search is deterministic: exact name or synonym matches first, then all-words matches, within
the protocol concept's domain. Non-standard concepts (e.g. ICD-10-CM codes) are followed through
``concept_relationship`` 'Maps to' to their standard concept. Every candidate records how it was
found, so a reviewer can judge it.

Vocabulary tables: ``concept`` (required), ``concept_synonym``, ``concept_relationship``,
``concept_ancestor`` (optional, used when present). ``load_athena`` reads a vocabulary download
from athena.ohdsi.org, which each user downloads under their own account and licences.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from agentic_tte.protocol.models import Domain

OMOP_DOMAIN = {
    Domain.CONDITION: "Condition",
    Domain.DRUG: "Drug",
    Domain.MEASUREMENT: "Measurement",
    Domain.PROCEDURE: "Procedure",
    Domain.OBSERVATION: "Observation",
    Domain.DEVICE: "Device",
    Domain.VISIT: "Visit",
}

MATCH_RANK = {"exact_name": 0, "exact_synonym": 1, "all_words": 2}


class Candidate(BaseModel):
    concept_id: int
    concept_name: str
    vocabulary_id: str
    domain_id: str
    concept_class_id: str | None = None
    standard: bool
    match: str  # exact_name, exact_synonym, all_words
    matched_text: str  # the query variant that matched
    via: int | None = None  # non-standard source concept mapped through 'Maps to'
    via_name: str | None = None
    score: float  # 1.0 exact; lower for looser matches

    @property
    def rank(self) -> tuple:
        return (MATCH_RANK[self.match], self.via is not None, -self.score, self.concept_id)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


# Words that rarely appear in standard concept names: dropped as an extra query variant.
_GENERIC = {"level", "levels", "value", "values", "test", "result", "results", "concentration",
            "use", "therapy", "treatment", "history", "of", "known", "patient", "patients"}
# Salt and form words: "hydrocortisone sodium succinate" should find the ingredient.
_SALTS = {"sodium", "succinate", "hydrochloride", "hcl", "phosphate", "sulfate", "sulphate",
          "acetate", "chloride", "potassium", "citrate", "tartrate", "maleate", "mesylate",
          "calcium", "besylate", "bitartrate", "fumarate", "bromide", "injection", "infusion",
          "intravenous", "iv", "oral", "tablet", "tablets"}  # fmt: skip


def query_variants(name: str, domain: str | None = None) -> list[str]:
    """"T2DM (type 2 diabetes mellitus)" -> the full text, the text outside the brackets and
    the bracketed expansion; then the same without generic words ("serum lactate level" ->
    "serum lactate") and, for drugs, without salt and form words. Normalized, de-duplicated,
    most specific first."""
    out = [name]
    inner = re.findall(r"\(([^)]+)\)", name)
    outer = re.sub(r"\([^)]*\)", " ", name)
    out += inner + [outer]
    drop = _GENERIC | (_SALTS if domain == Domain.DRUG else set())
    out += [" ".join(w for w in norm(v).split() if w not in drop) for v in list(out)]
    seen, variants = set(), []
    for v in map(norm, out):
        if v and v not in seen:
            seen.add(v)
            variants.append(v)
    return variants


def _tables(con) -> set[str]:
    return {r[0].lower() for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}  # fmt: skip


def _norm_sql(col: str) -> str:
    return f"trim(regexp_replace(lower({col}), '[^a-z0-9]+', ' ', 'g'))"


def search(con, name: str, domain: str, *, limit: int = 10) -> list[Candidate]:
    """Candidate standard concepts for a protocol concept, best first."""
    omop_domain = OMOP_DOMAIN.get(Domain(domain))
    if omop_domain is None:
        return []
    tables = _tables(con)
    has_syn = "concept_synonym" in tables
    has_rel = "concept_relationship" in tables
    cols = {r[0].lower() for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE lower(table_name) = 'concept'").fetchall()}  # fmt: skip
    valid = "(c.invalid_reason IS NULL OR c.invalid_reason = '')" \
        if "invalid_reason" in cols else "TRUE"  # fmt: skip
    # concept_class_id is required in CDM v5.4, but tolerate older or minimal tables.
    cls_s = "c.concept_class_id" if "concept_class_id" in cols else "NULL"
    cls_t = "t.concept_class_id" if "concept_class_id" in cols else "NULL"
    found: dict[int, Candidate] = {}

    def add(rows, match: str, text: str) -> None:
        for cid, cname, vocab, dom, cls, std, via, via_name in rows:
            score = 1.0 if match.startswith("exact") else round(
                min(0.9, len(text) / max(len(norm(cname)), 1)), 3)  # fmt: skip
            cand = Candidate(concept_id=cid, concept_name=cname, vocabulary_id=vocab,
                             domain_id=dom, concept_class_id=cls, standard=std == "S",
                             match=match, matched_text=text, via=via, via_name=via_name,
                             score=score)  # fmt: skip
            if cid not in found or cand.rank < found[cid].rank:
                found[cid] = cand

    for text in query_variants(name, domain):
        words = text.split()
        exact_name = f"{_norm_sql('c.concept_name')} = ?"
        like_all = " AND ".join([f"{_norm_sql('c.concept_name')} LIKE ?"] * len(words))
        likes = [f"%{w}%" for w in words]
        conds = [("exact_name", exact_name, [text]), ("all_words", like_all, likes)]
        if has_syn:
            conds.insert(1, ("exact_synonym",
                             f"c.concept_id IN (SELECT concept_id FROM concept_synonym s "
                             f"WHERE {_norm_sql('s.concept_synonym_name')} = ?)", [text]))
        for match, where, params in conds:
            # Standard concepts in the right domain.
            rows = con.execute(f"""
                SELECT c.concept_id, c.concept_name, c.vocabulary_id, c.domain_id,
                       {cls_s}, c.standard_concept, NULL, NULL
                FROM concept c
                WHERE c.standard_concept = 'S' AND c.domain_id = ? AND {valid} AND ({where})
                LIMIT {limit * 3}""", [omop_domain, *params]).fetchall()  # fmt: skip
            add(rows, match, text)
            if has_rel:
                # Non-standard source concepts, followed to their standard target.
                rows = con.execute(f"""
                    SELECT t.concept_id, t.concept_name, t.vocabulary_id, t.domain_id,
                           {cls_t}, t.standard_concept, c.concept_id, c.concept_name
                    FROM concept c
                    JOIN concept_relationship r ON r.concept_id_1 = c.concept_id
                         AND r.relationship_id = 'Maps to'
                    JOIN concept t ON t.concept_id = r.concept_id_2 AND t.standard_concept = 'S'
                    WHERE (c.standard_concept IS NULL OR c.standard_concept <> 'S')
                      AND t.domain_id = ? AND {valid} AND ({where})
                    LIMIT {limit * 3}""", [omop_domain, *params]).fetchall()  # fmt: skip
                add(rows, match, text)
    ranked = sorted(found.values(), key=lambda c: c.rank)
    # Drugs: prefer the ingredient, which covers every product through concept_ancestor.
    if omop_domain == "Drug":
        ranked.sort(key=lambda c: (c.rank[0], c.concept_class_id != "Ingredient", c.rank[1:]))
    return ranked[:limit]


ATHENA_FILES = ("CONCEPT", "CONCEPT_SYNONYM", "CONCEPT_RELATIONSHIP", "CONCEPT_ANCESTOR",
                "VOCABULARY", "DOMAIN", "CONCEPT_CLASS", "RELATIONSHIP")  # fmt: skip


def load_athena(con, directory: str | Path) -> dict[str, int]:
    """Load an Athena vocabulary download (tab-separated CSVs) into the connected database,
    replacing existing vocabulary tables. Returns row counts per table loaded."""
    d = Path(directory)
    counts = {}
    for name in ATHENA_FILES:
        f = d / f"{name}.csv"
        if not f.exists():
            continue
        table = name.lower()
        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT * FROM read_csv(?, delim = '\t', header = true, quote = '',
                                   all_varchar = false, dateformat = '%Y%m%d')""",
                    [str(f)])  # fmt: skip
        counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    if "concept" not in counts:
        raise FileNotFoundError(f"no CONCEPT.csv in {d}")
    return counts
