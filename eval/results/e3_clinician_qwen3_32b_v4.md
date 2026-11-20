# Trialist + Clinician: qwen3_32b confounders vs the adjudicated references

Run `runs/exit-A100-core-clinician-v4`, replicate 1; replicate 2 identical in 12/12 trials. Prompt clinician-v4. Lenient one-to-one name matching. The frozen E3 result (`e3_qwen3_32b.md`, confounder recall 0) is unchanged.

**Disclosure (changelog item 29):** the Clinician was built by the LLM that wrote the second annotation and adjudicated these references, after E3 showed the confounders missing. Its prompt holds no trial-specific content, but its developer had seen the reference lists, so this score is not blind to development. The references await the clinician audit.

**Over the 12 core trials:** precision 28%, recall 48%, F1 35% (67 matched; 243 proposed, 139 in the references).

| Trial | Proposed (cited) | Reference | Matched | Precision | Recall | Dropped by the checks |
|---|---|---|---|---|---|---|
| ACURASYS | 21 (20) | 10 | 5 | 24% | 50% | – |
| CRISTAL | 20 (19) | 10 | 6 | 30% | 60% | time from ICU admission to randomization (measured after time zero (post-baseline adjustment)) |
| APROCCHSS | 20 (13) | 12 | 5 | 25% | 42% | – |
| CHEST | 21 (20) | 12 | 5 | 24% | 42% | – |
| ADRENAL | 18 (11) | 10 | 4 | 22% | 40% | time from onset to vasopressor initiation (measured after time zero (post-baseline adjustment)) |
| HYDRAREA | 21 (20) | 12 | 5 | 24% | 42% | – |
| BICAR-ICU | 21 (17) | 12 | 8 | 38% | 67% | – |
| CAPE COD | 20 (19) | 12 | 7 | 35% | 58% | – |
| STARRT-AKI | 20 (13) | 14 | 4 | 20% | 29% | – |
| EvK | 20 (19) | 10 | 5 | 25% | 50% | Mechanical Ventilation (the treatment itself (F-h)) |
| ECLS-SHOCK | 21 (20) | 13 | 6 | 29% | 46% | – |
| LOVIT | 20 (20) | 12 | 7 | 35% | 58% | – |
