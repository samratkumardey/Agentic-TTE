"""OMOP vocabulary search: protocol concept text -> candidate standard concepts.

Concept IDs come only from the vocabulary tables, never from an LLM's memory (PLAN.md §1). The
search is deterministic: exact name or synonym matches first, then all-words matches, within
the protocol concept's domain. Non-standard concepts (e.g. ICD-10-CM codes) are followed through
``concept_relationship`` 'Maps to' to their standard concept. Every candidate records how it was
found, so a reviewer can judge it.

Abbreviations ("ADHF", "MAP") are expanded only from the protocol's own definitions and a
reviewer-approved list (abbreviations.py); a candidate found that way says so.

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
    abbreviation: str | None = None  # "ADHF = acute decompensated heart failure (source)"
    score: float  # 1.0 exact; lower for looser matches
    penalty: int = 0  # signs of a different concept than asked for (see search)

    @property
    def rank(self) -> tuple:
        # Signs of a different concept come first: an exact synonym of the wrong variant
        # ("Hematocrit ... adjusted ... by calculation") must not beat the plain test found by
        # its words. Then the match level, a direct match over one through an abbreviation,
        # the candidate found by the fullest query ("pain level" over "pain"), and length.
        return (self.penalty, MATCH_RANK[self.match], self.abbreviation is not None,
                self.via is not None, -len(self.matched_text), -self.score, self.concept_id)


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


# Words that rarely appear in standard concept names: dropped as an extra query variant.
_GENERIC = {"level", "levels", "value", "values", "test", "result", "results", "concentration",
            "use", "therapy", "treatment", "history", "of", "known", "patient", "patients",
            "diagnosis", "diagnosed"}
# Salt and form words: "hydrocortisone sodium succinate" should find the ingredient.
_SALTS = {"sodium", "succinate", "hydrochloride", "hcl", "phosphate", "sulfate", "sulphate",
          "acetate", "chloride", "potassium", "citrate", "tartrate", "maleate", "mesylate",
          "calcium", "besylate", "bitartrate", "fumarate", "bromide", "injection", "infusion",
          "intravenous", "iv", "oral", "tablet", "tablets"}  # fmt: skip
# LOINC names read "Analyte [Property] in System": system words are not another analyte.
_LOINC_SYSTEM = {"in", "of", "by", "or", "and", "blood", "serum", "plasma", "urine", "total"}
# Variants of a test rather than the test itself.
_VARIANTS = {"adjusted", "calculation", "calculated", "corrected", "goal", "ratio", "post",
             "pre", "fetal", "maternal", "donor", "challenge", "stimulation", "measurement"}
# Words that turn a concept into its opposite or absence.
_NEGATIONS = {"non", "not", "no", "without", "absent", "absence", "negative", "free"}
# Words in ICU drug and fluid labels that name an administration, not a substance
# (MIMIC-IV inputevents: "KCL (Bolus)", "Replete with Fiber (Full)", "Fentanyl (Concentrate)").
_DRUG_NOISE = {"solution", "bolus", "full", "concentrate", "prophylaxis", "premix", "piggyback",
               "drip", "crrt", "intake", "flush", "push", "gtt", "infusion", "mini", "bag",
               "with", "and", "in"}  # fmt: skip


def query_variants(name: str, domain: str | None = None) -> list[str]:
    """"T2DM (type 2 diabetes mellitus)" -> the full text, the text outside the brackets and
    the bracketed expansion; then the same without generic words ("serum lactate level" ->
    "serum lactate"). For drugs, also without salt, form and administration words and without
    strengths ("Albumin 5%" -> "albumin"); a variant left with only such words is dropped, so
    "(Full)" never becomes a query on its own. Normalized, de-duplicated, most specific
    first."""
    drug = domain == Domain.DRUG
    out = [name]
    # Bracketed text is a query of its own only when it can name the concept: an expansion
    # ("type 2 diabetes mellitus") or, for drugs, a brand ("Zosyn"). "(serum)" alone proposed
    # the concept "Serum" for MIMIC's Potassium (serum); "(s)" matched every synonym "S".
    inner = [i for i in re.findall(r"\(([^)]+)\)", name)
             if len(norm(i).split()) >= 2 or (drug and len(norm(i)) >= 4)]  # fmt: skip
    outer = re.sub(r"\([^)]*\)", " ", name)
    out += inner + [outer]
    drop = _GENERIC | (_SALTS | _DRUG_NOISE if drug else set())
    if domain == Domain.MEASUREMENT:
        drop = drop | {"count", "absolute"}  # LOINC says "[#/volume]", not "count"
    out += [" ".join(w for w in norm(v).split() if w not in drop) for v in list(out)]
    if drug:
        out += [" ".join(w for w in norm(v).split() if w not in drop and not w.isdigit())
                for v in list(out)]  # fmt: skip
    seen, variants = set(), []
    for v in map(norm, out):
        words = v.split()
        # Nothing left that names a substance ("(Full)", "Solution"). Salt words do name one:
        # "Potassium Chloride" is a query.
        if drug and all(w in _DRUG_NOISE or w.isdigit() for w in words):
            continue
        if len(v) < 2:
            continue  # single letters: MIMIC's lab indices "H", "L", "I"
        if v and v not in seen:
            seen.add(v)
            variants.append(v)
    return variants


def _tables(con) -> set[str]:
    return {r[0].lower() for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}  # fmt: skip


def _norm_sql(col: str) -> str:
    return f"trim(regexp_replace(lower({col}), '[^a-z0-9]+', ' ', 'g'))"


def _other_number(word: str) -> str:
    """The plural or singular of a query word, for matching: "platelet" <-> "platelets"."""
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word if word.isdigit() or len(word) < 3 else word + "s"


def _prefer(alias: str) -> str:
    """SQL ordering before the cut: LOINC tests with a measured property, then short names."""
    return (f"({alias}.vocabulary_id = 'LOINC' AND {alias}.concept_name LIKE '%[%') DESC, "
            f"length({alias}.concept_name)")


def _search_index(con, has_syn: bool) -> None:
    """Normalized names, built once per connection: ``atte_cnorm`` (concept_id, ' name ') and
    ``atte_snorm`` (concept_id, name) for synonyms. Normalizing 2 million names with a regular
    expression in every query made item-map proposals take hours. Rebuilt when the concept
    table changes (row count and highest ID)."""
    key = con.execute("SELECT count(*), max(concept_id) FROM concept").fetchone()
    try:
        built = con.execute("SELECT n, top FROM atte_cnorm_key").fetchone()
    except Exception:  # noqa: BLE001 - not built yet on this connection
        built = None
    if built == key:
        return
    con.execute(f"""CREATE OR REPLACE TEMP TABLE atte_cnorm AS
        SELECT concept_id, ' ' || {_norm_sql('concept_name')} || ' ' AS padded FROM concept""")
    if has_syn:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE atte_snorm AS
            SELECT concept_id, {_norm_sql('concept_synonym_name')} AS name
            FROM concept_synonym""")  # fmt: skip
    con.execute("CREATE OR REPLACE TEMP TABLE atte_cnorm_key AS SELECT ? AS n, ? AS top",
                list(key))


def search(con, name: str, domain: str, *, limit: int = 10,
           abbreviations: list | None = None) -> list[Candidate]:  # fmt: skip
    """Candidate standard concepts for a protocol concept, best first. ``abbreviations``
    (usable ``abbreviations.Abbreviation`` entries) add the name with each abbreviation it
    contains expanded; such candidates say which expansion found them."""
    from agentic_tte.omop.abbreviations import expansions_for

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
    _search_index(con, has_syn)
    # Retrieve many matches and rank them all: a common lab name ("creatinine") matches
    # hundreds of LOINC tests, and keeping the first few the database returns lost the right
    # one (MIMIC-IV proposals, 2026-10-26).
    fetch = max(limit * 3, 500)

    asked = set(norm(name).split())

    def penalty(cname: str, vocab: str, cls: str | None) -> int:
        """Signs that a candidate is a different concept than the one asked for, found in the
        first MIMIC-IV item proposals (2026-10-26)."""
        words = set(norm(cname).split())
        p = 0
        if (words & _NEGATIONS) - asked:  # "Non-invasive ventilation" for Invasive Ventilation
            p += 2
        if omop_domain == "Measurement":
            if cls == "Clinical Finding" or {"reference", "finding", "detected"} & words - asked:
                p += 2  # a finding ("above reference range"), not the measurement
            if "cord" in words - asked:
                p += 1  # newborn cord-blood tests
            if vocab != "LOINC":
                p += 1  # LOINC is OMOP's standard vocabulary for lab tests and vital signs
            else:
                analyte = set(norm(cname.split("[")[0]).split())
                if "[" not in cname and "panel" not in asked:
                    p += 1  # a panel or narrative, not a measured quantity
                elif analyte - asked - _LOINC_SYSTEM:
                    p += 1  # another analyte: "Calcium/Sodium", "Hemoglobin F1", "CD19 cells"
            if _VARIANTS & words - asked:
                p += 1  # a calculated, adjusted, ratio or goal variant
        if omop_domain == "Drug" and vocab == "RxNorm Extension":
            p += 1  # prefer RxNorm; Extension names include codes such as "LR-90"
        return p

    def add(rows, match: str, text: str, abbr: str | None) -> None:
        for cid, cname, vocab, dom, cls, std, via, via_name in rows:
            score = 1.0 if match.startswith("exact") else round(
                min(0.9, len(text) / max(len(norm(cname)), 1)), 3)  # fmt: skip
            cand = Candidate(concept_id=cid, concept_name=cname, vocabulary_id=vocab,
                             domain_id=dom, concept_class_id=cls, standard=std == "S",
                             match=match, matched_text=text, via=via, via_name=via_name,
                             abbreviation=abbr, score=score,
                             penalty=penalty(cname, vocab, cls))  # fmt: skip
            if cid not in found or cand.rank < found[cid].rank:
                found[cid] = cand

    queries = [(v, None) for v in query_variants(name, domain)]
    seen = {v for v, _ in queries}
    for entry, exp, expanded in expansions_for(name, domain, abbreviations or []):
        label = f"{entry.abbr} = {exp} ({entry.source})"
        for v in query_variants(expanded, domain):
            if v not in seen:
                seen.add(v)
                queries.append((v, label))
    for text, abbr in queries:
        words = text.split()
        exact_name = "n.padded = ?"
        # Whole words only: "nacl" must not match inside "linaclotide", nor "full" inside
        # "buckminsterfullerene" (found in the MIMIC-IV inputevents proposals, 2026-10-26).
        # Singular and plural both match ("platelet count" finds "Platelets [#/volume]").
        like_all = " AND ".join(["(n.padded LIKE ? OR n.padded LIKE ?)"] * len(words))
        likes = [x for w in words for x in (f"% {w} %", f"% {_other_number(w)} %")]
        conds = [("exact_name", exact_name, [f" {text} "])]
        # A short abbreviation ("LR", "KCL", "GT") gets exact matches only: as a word it found
        # RxNorm Extension code names ("LR-90", "KCL-1999000269").
        if not (len(words) == 1 and len(text) <= 3):
            conds.append(("all_words", like_all, likes))
        if has_syn:
            conds.insert(1, ("exact_synonym", "c.concept_id IN (SELECT concept_id FROM "
                                              "atte_snorm WHERE name = ?)", [text]))
        for match, where, params in conds:
            # Standard concepts in the right domain.
            rows = con.execute(f"""
                SELECT c.concept_id, c.concept_name, c.vocabulary_id, c.domain_id,
                       {cls_s}, c.standard_concept, NULL, NULL
                FROM concept c JOIN atte_cnorm n ON n.concept_id = c.concept_id
                WHERE c.standard_concept = 'S' AND c.domain_id = ? AND {valid} AND ({where})
                ORDER BY {_prefer('c')} LIMIT {fetch}""",
                               [omop_domain, *params]).fetchall()  # fmt: skip
            add(rows, match, text, abbr)
            if has_rel:
                # Non-standard source concepts, followed to their standard target.
                rows = con.execute(f"""
                    SELECT t.concept_id, t.concept_name, t.vocabulary_id, t.domain_id,
                           {cls_t}, t.standard_concept, c.concept_id, c.concept_name
                    FROM concept c JOIN atte_cnorm n ON n.concept_id = c.concept_id
                    JOIN concept_relationship r ON r.concept_id_1 = c.concept_id
                         AND r.relationship_id = 'Maps to'
                    JOIN concept t ON t.concept_id = r.concept_id_2 AND t.standard_concept = 'S'
                    WHERE (c.standard_concept IS NULL OR c.standard_concept <> 'S')
                      AND t.domain_id = ? AND {valid} AND ({where})
                    ORDER BY {_prefer('t')} LIMIT {fetch}""",
                                   [omop_domain, *params]).fetchall()  # fmt: skip
                add(rows, match, text, abbr)
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
    if not (d / "CONCEPT.csv").exists():
        hint = (" (the path is empty: is the variable set? on Hellbender, `source hpc/env.sh`)"
                if str(directory).strip() in ("", ".") else "")  # fmt: skip
        raise FileNotFoundError(f"no CONCEPT.csv in {d.resolve()}{hint}")
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
    return counts
