# E1 fault injection on synthetic OMOP

Synthetic ICU cohort, n = 3000 (seed 20261027), treatment started within 3 days of time zero. Truth: marginal HR 0.590 (initiate within the grace period vs never, potential outcomes).

**Correct design:** checks failed: none.

| Fault | Planted error | Target check | Caught | Checks failed | Primary estimate (HR, 95% CI) | Bias (log HR) |
|---|---|---|---|---|---|---|
| F-a immortal time | immortal-time handling set to 'none' | A-a immortal time | yes | A-a | 0.496 (0.433–0.569) iptw | -0.173 |
| F-b look-ahead in eligibility | I1 now reads the 24 hours after time zero | A-b look-ahead in eligibility | yes | A-b | 0.612 (0.538–0.696) clone-censor-weight | +0.036 |
| F-c time zero after treatment start | time zero moved 2 days later | A-c time zero after treatment start | yes | A-c | none: fewer than 11 patients in an arm or events in total | – |
| F-d arms swapped | treatment and comparator flags swapped | A-d treatment and comparator arms | yes | A-d | none: frame not compiled: active comparators are not supported yet | – |
| F-e outcome horizon changed | primary outcome horizon set to 90 days | A-e outcome horizon | yes | A-e | 0.612 (0.538–0.696) clone-censor-weight | +0.036 |
| F-f prevalent users | exclusion of prior treatment removed | A-f new users (prior treatment excluded) | yes | A-f | 0.612 (0.538–0.696) clone-censor-weight | +0.036 |
| F-g death censored | death added to the censoring events | A-g death not censored for a mortality outcome | yes | A-g | 0.612 (0.538–0.696) clone-censor-weight | +0.036 |
| F-h treatment as a confounder | the treatment added as confounder 'study drug exposure' | A-h confounders measured before treatment | yes | A-h, A-j | 0.612 (0.538–0.696) clone-censor-weight | +0.036 |
| F-i grace period removed | grace period removed (the trial's initiation window is kept) | A-i grace period for a stated initiation window | yes | A-a, A-i | 0.496 (0.433–0.569) iptw | -0.173 |
| F-j wrong concept set | treatment concept set replaced by concept 2000000101 | A-j reviewed concept sets | yes | A-j | none: fewer than 11 patients in an arm or events in total | – |

Correct design: 0.612 (0.538–0.696), clone-censor-weight; bias +0.036.

**Caught 10 of 10 applicable faults** by their target check; false alarms on the correct design: 0.
