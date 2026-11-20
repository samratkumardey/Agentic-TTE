"""Concordance agent: agreement of each emulation with its trial's published result, by the
metrics pre-registered in eval/SCORING.md (Phase 3, R4), as in RCT-DUPLICATE.

Deterministic; no LLM. Every number comes from the emulation's signed-off (or auto-approved,
labelled) report and the user-verified reference values; nothing is tuned after an estimate
is seen.

Per trial (R4), with *E* the emulation and *R* the trial, on the log scale for ratios and in
percentage points for risk differences, and standard errors from the 95% CIs:
- standardized difference agreement: |z| < 1.96, z = (θ_E − θ_R) / √(SE_E² + SE_R²);
- estimate agreement: θ_E inside the trial's 95% CI;
- regulatory agreement: same direction and same significance at 0.05;
- direction agreement: same sign, or both CIs containing the null;
- ratio of ratios (ratio measures): exp(θ_E − θ_R) with its 95% CI;
- the emulation's own CI width, next to every call.

Primary endpoints over a set: the share with standardized difference agreement, with an exact
Clopper-Pearson 95% CI, and the pooled ROR (random effects, REML) over ratio-measure trials,
with τ². A trial in the set that did not reach an estimate counts as no agreement on every
binary metric, with the reason (R4 "Failures count"). Sets are never pooled.

Measure and orientation:
- The emulation reports the trial's own measure: HR from the primary weighted Cox model, RR,
  OR or RD from the bootstrap risk measures (R4). A published estimate is never converted.
- The trial's contrast ("A vs B") is matched to the protocol's arms. If the protocol's
  treatment arm is the trial's B, the emulation's contrast is reversed (1/ratio, −RD); if
  the match is unclear, the trial is not scored until a person confirms the orientation.
- A risk difference in survival is compared with −RD of the emulation's event risk.

A run whose first gate found S8 not met (status ``excluded_s8``) leaves the set: it is listed,
kept out of the denominator, and the set must be recomputed (SCORING.md item 9). S8 not met at
a later gate is a failure.
"""

from __future__ import annotations

import math
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy import optimize, stats

Z = stats.norm.ppf(0.975)
SCORED_BOOTSTRAP = {"n_bootstrap": 500, "seed": 20261027}  # R4
Kind = Literal["HR", "RR", "OR", "RD"]
_STOP = {"with", "and", "plus", "the", "versus", "group", "arm", "therapy", "treatment",
         "early", "standard", "usual"}  # fmt: skip
_NULL_ARM = re.compile(r"^(placebo|control|usual care|standard care|no .+|no intervention)$")


def measure_kind(measure: str) -> tuple[Kind | None, bool]:
    """The measure's kind, and whether a risk difference is in survival."""
    m = measure.strip().upper()
    for k in ("HR", "RR", "OR", "RD"):
        if m.startswith(k):
            return k, k == "RD" and "SURVIVAL" in m
    return None, False


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) >= 4 and w not in _STOP}


def _null(name: str) -> bool:
    return bool(_NULL_ARM.match(" ".join(name.lower().split())))


def orientation(contrast: str, treatment: list[str], comparator: list[str]) -> str:
    """"same", "reversed" or "unconfirmed": the trial's "A vs B" against the protocol's
    treatment and comparator arm names (labels and component concepts). Words that both arms
    share are ignored; "placebo", "control", "usual care" or "No X" mark a null arm. With no
    treatment arm in the protocol, the orientation is never guessed."""
    parts = re.split(r"\s+vs\.?\s+", re.sub(r"\(.*?\)", "", contrast).strip(), maxsplit=1)
    if len(parts) != 2 or not treatment:
        return "unconfirmed"
    a, b = (_words(x) for x in parts)
    t = set().union(*(_words(x) for x in treatment))
    c = set().union(*(_words(x) for x in comparator)) if comparator else set()
    tu, cu = t - c, c - t
    c_null = any(_null(x) for x in comparator)
    a_t, a_c, b_t, b_c = a & tu, a & cu, b & tu, b & cu
    if (a_t or b_c or (_null(parts[1]) and c_null)) and not a_c and not b_t:
        return "same"
    if (b_t or a_c or (_null(parts[0]) and c_null)) and not b_c and not a_t:
        return "reversed"
    return "unconfirmed"


class Reference(BaseModel):
    nct_id: str
    trial: str | None = None
    measure: str
    value: float
    ci: tuple[float, float]
    contrast: str
    outcome: str | None = None
    verified_by: str | None = None


class Emulation(BaseModel):
    """What the Replay wrote at the end of a run (``result.json``)."""

    nct_id: str
    status: str  # signed_off, auto_approved, returned, blocked, stopped, done
    protocol_version: int | None = None
    protocol_hash: str | None = None
    treatment: list[str] = Field(default_factory=list)  # arm label and component names
    comparator: list[str] = Field(default_factory=list)
    analysis: dict | None = None  # AnalysisResult (JSON)
    stop_reason: str | None = None


class TrialAgreement(BaseModel):
    nct_id: str
    trial: str | None = None
    kind: Kind | None = None
    reference: str
    emulation: str | None = None
    orientation: str = "unconfirmed"
    scored: bool = False  # False: a failure (no estimate) or orientation unconfirmed
    excluded: bool = False  # left the set (S8 not met at the first gate): not a failure
    reason: str = ""  # why there is no estimate, or what was adjusted
    theta_e: float | None = None
    se_e: float | None = None
    theta_r: float | None = None
    se_r: float | None = None
    z: float | None = None
    standardized: bool = False
    estimate: bool = False
    regulatory: bool = False
    direction: bool = False
    ror: tuple[float, float, float] | None = None  # value, lower, upper
    ci_width: float | None = None  # ratio of the CI limits (ratios) or points (RD)
    notes: list[str] = Field(default_factory=list)


def _scale(kind: Kind, v: float) -> float:
    return math.log(v) if kind != "RD" else v


def _se(lo: float, hi: float) -> float:
    return (hi - lo) / (2 * Z)


def _emulated(kind: Kind, analysis: dict) -> tuple[float, float, float] | str:
    """The emulation's estimate and 95% CI in the trial's measure (RD in points), or why
    there is none."""
    if kind == "HR":
        est = next((e for e in analysis.get("estimates") or []
                    if e["method"] == analysis.get("primary_method")), None)  # fmt: skip
        if est is None:
            return "no primary estimate"
        return est["hr"], est["lower"], est["upper"]
    rm = analysis.get("risk_measures")
    if not rm:
        return "no risk measures (bootstrap not run)"
    key = {"RR": "rr", "OR": "or_", "RD": "rd"}[kind]
    v, ci = rm.get(key), (rm.get("ci") or {}).get(key)
    if v is None or not ci:
        return f"{kind} undefined or without a bootstrap CI"
    f = 100.0 if kind == "RD" else 1.0
    return f * v, f * ci[0], f * ci[1]


def agree(ref: Reference, emu: Emulation | None, *, confirmed: str | None = None
          ) -> TrialAgreement:  # fmt: skip
    """The R4 metrics for one trial. ``confirmed``: an orientation ("same" or "reversed")
    a person confirmed, used when the automatic match is unclear."""
    kind, survival = measure_kind(ref.measure)
    out = TrialAgreement(nct_id=ref.nct_id, trial=ref.trial, kind=kind,
                         reference=f"{ref.measure} {ref.value:g} ({ref.ci[0]:g} to "
                                   f"{ref.ci[1]:g}), {ref.contrast}")  # fmt: skip
    if kind is None:
        out.reason = f"measure {ref.measure!r} not recognised"
        return out
    if emu is None:
        out.reason = "no run"
        return out
    if emu.status == "excluded_s8":
        out.excluded = True
        out.reason = ("S8 not met at the first gate: the trial leaves the agreement set, and "
                      "the set is recomputed by the outcome-blind rule (SCORING.md item 9)")
        return out
    if emu.status not in ("signed_off", "auto_approved"):
        out.reason = f"no signed-off report (status {emu.status}" + (
            f": {emu.stop_reason}" if emu.stop_reason else "") + ")"
        return out
    if emu.status == "auto_approved":
        out.notes.append("auto-approved run: not reviewed by a person")
    got = _emulated(kind, emu.analysis or {})
    if isinstance(got, str):
        out.reason = got
        return out
    v, lo, hi = got
    if kind != "RD" and min(v, lo, hi) <= 0:
        out.reason = "the emulation's ratio or its CI is not positive"
        return out
    rm = (emu.analysis or {}).get("risk_measures") or {}
    if kind != "HR" and any(rm.get(k) != x for k, x in SCORED_BOOTSTRAP.items()):
        out.notes.append(f"bootstrap {rm.get('n_bootstrap')} resamples, seed {rm.get('seed')}: "
                         "not the pre-registered 500 / 20261027")
    auto = orientation(ref.contrast, emu.treatment, emu.comparator)
    out.orientation = auto if auto != "unconfirmed" else (confirmed or "unconfirmed")
    if auto == "unconfirmed" and confirmed:
        out.notes.append(f"orientation {confirmed}, confirmed by a person")
    if out.orientation == "unconfirmed":
        out.reason = ("orientation unconfirmed: the trial's contrast does not match the "
                      "protocol's arms; confirm it before scoring")
        out.emulation = f"{v:.3g} ({lo:.3g} to {hi:.3g})"
        return out
    te, se_e = _scale(kind, v), _se(_scale(kind, lo), _scale(kind, hi))
    if out.orientation == "reversed":
        te = -te
        out.notes.append("the protocol's arms are the trial's in reverse: contrast reversed")
    if survival:
        te = -te
        out.notes.append("the trial's RD is in survival: compared with −RD of the event risk")
    tr, se_r = _scale(kind, ref.value), _se(_scale(kind, ref.ci[0]), _scale(kind, ref.ci[1]))
    lo_e, hi_e = te - Z * se_e, te + Z * se_e
    back = (lambda x: math.exp(x)) if kind != "RD" else (lambda x: x)  # noqa: E731
    out.emulation = f"{back(te):.3g} ({back(lo_e):.3g} to {back(hi_e):.3g})"
    out.theta_e, out.se_e, out.theta_r, out.se_r = te, se_e, tr, se_r
    out.z = (te - tr) / math.sqrt(se_e ** 2 + se_r ** 2)
    out.standardized = abs(out.z) < Z
    r_lo, r_hi = _scale(kind, ref.ci[0]), _scale(kind, ref.ci[1])
    out.estimate = r_lo <= te <= r_hi
    sig_e, sig_r = not (lo_e <= 0 <= hi_e), not (r_lo <= 0 <= r_hi)
    out.regulatory = (sig_e == sig_r) and (not sig_e or np.sign(te) == np.sign(tr))
    out.direction = (np.sign(te) == np.sign(tr) and te != 0) or (not sig_e and not sig_r)
    if kind != "RD":
        d, s = te - tr, math.sqrt(se_e ** 2 + se_r ** 2)
        out.ror = (math.exp(d), math.exp(d - Z * s), math.exp(d + Z * s))
    out.ci_width = math.exp(hi_e - lo_e) if kind != "RD" else hi_e - lo_e
    out.scored = True
    return out


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def reml(y: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    """Random-effects pooled mean, its SE and τ² by REML."""
    if len(y) == 1:
        return float(y[0]), float(math.sqrt(v[0])), 0.0

    def nll(t2: float) -> float:
        w = 1 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + t2)) + math.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))

    hi = max(10 * float(np.var(y)), 1e-6)
    t2 = float(optimize.minimize_scalar(nll, bounds=(0.0, hi), method="bounded").x)
    if nll(0.0) <= nll(t2):
        t2 = 0.0
    w = 1 / (v + t2)
    return float(np.sum(w * y) / np.sum(w)), float(math.sqrt(1 / np.sum(w))), t2


class SetSummary(BaseModel):
    name: str
    n: int
    failures: int  # no estimate, counted as no agreement
    unconfirmed: int  # orientation awaiting a person: not scored yet
    excluded: int = 0  # left the set: S8 not met at the first gate
    standardized: tuple[int, int]  # agreeing, denominator
    standardized_ci: tuple[float, float]
    estimate: int
    regulatory: int
    direction: int
    pooled_ror: tuple[float, float, float] | None = None
    tau2: float | None = None
    n_ror: int = 0


def summarize(name: str, rows: list[TrialAgreement]) -> SetSummary:
    """The R4 primary endpoints for one set. Unconfirmed orientations are left out of the
    denominator and counted, so the endpoint is not final until they are confirmed."""
    wait = [r for r in rows if r.orientation == "unconfirmed" and r.emulation is not None]
    out = [r for r in rows if r.excluded]
    base = [r for r in rows if r not in wait and r not in out]
    k = sum(r.standardized for r in base)
    s = SetSummary(name=name, n=len(rows), failures=sum(not r.scored for r in base),
                   unconfirmed=len(wait), excluded=len(out), standardized=(k, len(base)),
                   standardized_ci=clopper_pearson(k, len(base)),
                   estimate=sum(r.estimate for r in base),
                   regulatory=sum(r.regulatory for r in base),
                   direction=sum(r.direction for r in base))  # fmt: skip
    ratio = [r for r in base if r.scored and r.kind != "RD"]
    if ratio:
        y = np.array([r.theta_e - r.theta_r for r in ratio])
        v = np.array([r.se_e ** 2 + r.se_r ** 2 for r in ratio])
        mu, se, t2 = reml(y, v)
        s.pooled_ror = (math.exp(mu), math.exp(mu - Z * se), math.exp(mu + Z * se))
        s.tau2, s.n_ror = t2, len(ratio)
    return s


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def to_markdown(sets: dict[str, list[TrialAgreement]], *, run: str) -> str:
    lines = ["# Agreement with the trials (eval/SCORING.md R4)", "",
             f"Run `{run}`. Each set is reported on its own and never pooled. A trial that did "
             "not reach an estimate counts as no agreement on every binary metric.", ""]
    for name, rows in sets.items():
        s = summarize(name, rows)
        k, n = s.standardized
        lines += [f"## {name}", "",
                  f"**Standardized difference agreement:** {_pct(k, n)}, exact 95% CI "
                  f"{100 * s.standardized_ci[0]:.0f}–{100 * s.standardized_ci[1]:.0f}%"
                  if n else "**Standardized difference agreement:** no trial scored yet"]
        if s.pooled_ror:
            r = s.pooled_ror
            lines.append(f"**Pooled ratio of ratios** ({s.n_ror} ratio-measure trials, random "
                         f"effects, REML): {r[0]:.2f} ({r[1]:.2f}–{r[2]:.2f}), τ² {s.tau2:.3f}")
        lines += ["", f"Secondary: estimate agreement {_pct(s.estimate, n)}, regulatory "
                      f"{_pct(s.regulatory, n)}, direction {_pct(s.direction, n)}. Failures "
                      f"{s.failures}; orientation awaiting confirmation {s.unconfirmed} (not in "
                      f"the denominator); left the set by S8 {s.excluded}"
                      + (" (the set must be recomputed)" if s.excluded else "") + ".", "",
                  "| Trial | Trial result | Emulation | z | Std. diff | Estimate | Regulatory "
                  "| Direction | ROR | CI width | Notes |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for r in rows:
            def yn(b: bool) -> str:
                return "yes" if b else "no"
            ror = f"{r.ror[0]:.2f} ({r.ror[1]:.2f}–{r.ror[2]:.2f})" if r.ror else "–"
            note = "; ".join(([r.reason] if r.reason else []) + r.notes) or "–"
            lines.append(
                f"| {r.trial or r.nct_id} | {r.reference} | {r.emulation or '–'} | "
                f"{f'{r.z:.2f}' if r.z is not None else '–'} | {yn(r.standardized)} | "
                f"{yn(r.estimate)} | {yn(r.regulatory)} | {yn(r.direction)} | {ror} | "
                f"{f'{r.ci_width:.2f}' if r.ci_width is not None else '–'} | {note} |")
        lines.append("")
    return "\n".join(lines)
