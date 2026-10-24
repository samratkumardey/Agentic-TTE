# Replay reference: the 10 MIMIC-IV acute-care trials

Compiled 2026-10-24. The machine-readable source of truth is [replay_trials.json](replay_trials.json); this table must match it.

**Sources:**
- Cached ClinicalTrials.gov v2 records
- PubMed abstracts (E-utilities), searched by NCT number and then by title and design
- The EmulatRx paper's main text (Li et al., *Nat Commun* 17:5501, 2026): the Case 1–3 showcases and Table 2. Its Supplementary Information has not been checked.

**Verification labels:**
- **V**: verified. The publication cites the NCT ID, or the registry posts the value.
- **P**: probable. Matched on sponsor, design and sample size, but the PubMed record does not cite the NCT ID. Confirm in the full text.
- **NF**: searched, nothing found.
- **NR**: the source does not report it.
- † computed by us from reported counts (Wald 95% CI). Not an author-reported estimate.

| NCT | Trial | Results on CT.gov | Primary publication | Primary outcome (publication) | Reported effect (95% CI) | EmulatRx estimate |
|---|---|---|---|---|---|---|
| NCT00475852 | ASCEND-HF | Yes: RD −0.7 (−2.1, 0.7), p = 0.313 | O'Connor, NEJM 2011; PMID 21732835; doi 10.1056/NEJMoa1100171 (V) | 30-day HF rehospitalization or death | RD −0.7 pp (−2.1, 0.7); 9.4% vs 10.1% (V) | HR 0.5913 (0.46, 0.76); n = 13,942 (V) |
| NCT02856698 | MIMO | Yes: 7/55 vs 10/56, no analysis | Domínguez-Rodríguez, Eur J Heart Fail 2022; PMID 35780488; doi 10.1002/ejhf.2602 (V, by trial name) | In-hospital all-cause mortality | RR 0.71 (0.29, 1.74); 12.7% vs 17.9% (V) | NR |
| NCT03872011 | Lyu 2022 | No | Lyu, Crit Care 2022; PMID 36171582; doi 10.1186/s13054-022-04175-x (V) | 90-day mortality (per-protocol) | 39.9% vs 39.0%, p = 0.86; no CI in abstract. RD +0.9 pp (−8.6, 10.4)† | NR |
| NCT04134403 | STASIS | No (status unknown, never recruited?) | NF | In-hospital or 28-day mortality (registry) | NF | HR 1.2983 (1.07, 1.58); n = 2,306 (V) |
| NCT06091982 | Harapan Kita cohort (observational) | Yes: OR 10.6 (7.0, 16.2), 96/302 vs 34/811 | NF | In-hospital mortality (registry) | Registry crude OR only (V) | HR 0.6443, p = 0.0008, CI NR; n = 11,124 (V) |
| NCT03258684 | HYVCTTSSS | No | Chang, Chest 2020; PMID 32243943; doi 10.1016/j.chest.2020.02.065 (V) | 28-day all-cause mortality | 27.5% vs 35%, p = 0.47; no CI. RD −7.5 pp (−27.7, 12.7)† | NR |
| NCT03592693 | Athens HAT | No (terminated, n = 24) | NF | Hospital mortality, 90 d (registry) | NF | NR |
| NCT01560650 | CRRT intensity, cardiac surgery | No | NF (only a Cochrane review cites it) | Death from any cause (registry) | NF | NR |
| NCT01223690 | Clarithromycin, Gram-negative sepsis | No | Giamarellos-Bourboulis, J Antimicrob Chemother 2014; PMID 24292991; doi 10.1093/jac/dkt475 (P) | 28-day mortality | 18.5% vs 17.1%, p = 0.671 (P). RD +1.4 pp (−4.7, 7.6)† | NR |
| NCT00953303 | COPE-ADHF | No | Liu, J Cardiovasc Pharmacol 2014; PMID 24710470; doi 10.1097/FJC.0000000000000048 (P) | 30-day cardiovascular death | OR 0.26, p < 0.05; 3 vs 10 deaths; no CI (P) | NR |

## What this means for the paper

- **Only one trial has a like-for-like comparison:** ASCEND-HF has an author-reported RCT estimate with a CI *and* an EmulatRx estimate. Even there, the scales differ (RD vs HR). The RCT result is null; the emulation reports a significant benefit.
- **NCT04134403** has no RCT result at all. **NCT06091982** is observational, and its posted crude association runs in the opposite direction to EmulatRx's estimate (confounded by indication, so this is not like-for-like). Neither can serve as an "RCT benchmark".
- **The other 7 trials** have no EmulatRx estimate in the main text. Check the Supplementary Information before saying none exists.
- **Phase 0 doesn't need this table:** it scores protocol extraction, not effect estimates. This table becomes the outcome benchmark from Phase 3, when effects are estimated on MIMIC-IV.
- **Criterion counts:** the EmulatRx Table 2 counts (#EC, simple and complex) are in the JSON for the split comparison. Our deterministic split matches the paper's #EC in 5 of 10 trials using free-text items only (NCT00475852, NCT02856698, NCT03872011, NCT06091982, NCT03592693). NCT00953303 matches only if the structured age criterion is counted. NCT04134403 (12 vs 10), NCT03258684 (4 vs 5), NCT01560650 (5 vs 8) and NCT01223690 (6 vs 8) differ. The paper's split was done by GPT-4o, so a difference isn't automatically our error.

## Still to verify

1. The full texts of PMID 24292991 and 24710470: do they cite the NCT IDs?
2. Effect estimates with CIs in the full texts of Lyu 2022 (open access, PMC9520942) and Chang 2020.
3. The EmulatRx Supplementary Information: emulated estimates for the 7 trials not shown in the main text.
4. Whether NCT03592693, NCT01560650 and NCT06091982 have publications not indexed in PubMed (Embase, CNKI, conference abstracts).
