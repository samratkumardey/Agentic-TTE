# E2 (a): compiled OMOP criteria vs mimic-code derived concepts (MIMIC-IV 3.1)

Run: job 19373181 on Hellbender, 2026-10-28, commit 3dbc95c; CDM `mimiciv-3.1-cdm54-19365483`;
mimic-code derived tables exported by job 19370433 (database `mimiciv`). Pre-registered in
eval/SCORING.md, Phase 3 changelog item 17. First ICU stay per patient; counts below 11
suppressed.

| Definition | Compared | Both | Ours only | mimic-code only | Neither | Kappa | Only in ours | Only in mimic-code |
|---|---|---|---|---|---|---|---|---|
| age65 | 65361 | 34723 | 0 | 773 | 29865 | 0.976 | 0 | <11 |
| creat2 | 64703 | 8836 | 0 | 0 | 55867 | 1.0 | 0 | 663 |
| lact2 | 64703 | 19414 | 5269 | 0 | 40020 | 0.82 | 0 | 663 |
| ivent24 | 64703 | 15982 | 4386 | 3484 | 40851 | 0.715 | 0 | 663 |
| norepi24 | 64703 | 8102 | 732 | 0 | 55869 | 0.95 | 0 | 663 |

## Reading

- **creat2: exact agreement** (kappa 1.0, no disagreement in 64,703 patients). With a source and
  window that match mimic-code's, the compiled criterion reproduces the validated definition:
  the compiler's measurement window, value test and time zero (ICU admission + 24 h, a 30-hour
  look-back) are right.
- **Every other disagreement goes one way or the other, as the sources predict** (named before
  the run):
  - age65: 773 patients are 65 by mimic-code and not by us, none the other way. mimic-code's age
    is anchor_age plus calendar years at hospital admission (whole years); ours is the exact age
    at ICU admission from the year of birth with a mid-year default, so ours is slightly lower.
  - lact2: 5,269 positive by us only, none the other way. Our lactate concept collects every
    source mapped to it (charted and point-of-care values as well as blood-gas labs); mimic-code's
    first_day_bg reads blood-gas lactate only. To be confirmed by a breakdown by source.
  - norepi24: 732 positive by us only, none the other way: prescriptions, which mimic-code's
    infusion table does not count.
  - ivent24: disagreement both ways (kappa 0.715). Ours is the procedureevents "Invasive
    Ventilation" episode start; mimic-code's is ventilator settings in chartevents, with its own
    episode rules (tracheostomy is a separate status). The two sources define different events.
- **663 patients** have mimic-code first-day data but no day-1 time zero on our side: in the
  24-hour designs, time zero needs the patient alive and observed at ICU admission + 24 h (death
  or discharge in the first day).

## What this shows

Where the source matches, the compiler is exact. The other gaps are differences of definition
between data sources, not compiler errors, and they are the kind a concept review must decide
(which lactate, which ventilation source). They are reported, not tuned away.

## Breakdown of the disagreements (job 19374062, 2026-10-28)

| Definition | Group | Category | Patients |
|---|---|---|---|
| lact2 | ours only | qualifying values from labevents item 50813 | 5269 |
| ivent24 | ours only | mimic-code invasive episode already running at ICU admission | 1952 |
| ivent24 | ours only | mimic-code invasive episode starting after 24 h | 130 |
| ivent24 | ours only | mimic-code tracheostomy on day 1 | 15 |
| ivent24 | ours only | mimic-code other support only on day 1 (NIV, HFNC, oxygen) | 1705 |
| ivent24 | ours only | no mimic-code ventilation row for the stay | 451 |
| ivent24 | ours only | other | 133 |
| ivent24 | mimic-code only | our invasive ventilation starts before ICU admission | 709 |
| ivent24 | mimic-code only | our invasive ventilation starts after 24 h | 228 |
| ivent24 | mimic-code only | no invasive ventilation record of ours | 2547 |
| norepi24 | ours only | day-1 records from ICU item 221906 (inputevents) | 49 |

- **Lactate: the predicted explanation was wrong.** All 5,269 patients positive by us only have their qualifying value from lab item 50813, the same blood-gas item mimic-code uses. The gap is not about sources: it is in how mimic-code's `bg` / `first_day_bg` select blood gases. A second breakdown splits it: is mimic-code's lactate_max missing or <= 2, is our value only before ICU admission, and is it outside a hospital admission (no hadm_id)?
- **Ventilation: explained by timing and source.**
  - Ours only: mostly patients whose invasive episode, by mimic-code, began before ICU admission (1,952: intubated in the OR or ED, with the procedureevents entry made on arrival), or who are classified by mimic-code's ventilator settings as non-invasive support only (1,705).
  - Mimic-code only: mostly ventilated patients with no procedureevents "Invasive Ventilation" entry at all (2,547), or one starting before ICU admission (709).
  - **Consequence for the core trials that need ventilation (ACURASYS, ROSE):** procedureevents alone misses patients that ventilator settings show. Their concept review should consider the charted ventilator settings (PEEP set, tidal volume, already mapped) as evidence of invasive ventilation.
- **Norepinephrine: the breakdown undercounted** (49 of 732 classified). The other patients' day-1 records are date-only, mostly prescriptions. The compiler places those at midnight, but the first breakdown read only datetimes. Fixed: date-only records are placed at midnight as the compiler places them, and labelled. A prescription dated the day of ICU admission counts as day-1 exposure from midnight, which can precede the actual start: a timing limitation of date-only records, to be reported.
