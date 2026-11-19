# Trialist + Clinician: qwen3_32b confounders vs the adjudicated references

Run `runs/exit-A100-core-clinician-v3`, replicate 1; replicate 2 identical in 12/12 trials. Prompt clinician-v3. Lenient one-to-one name matching. The frozen E3 result (`e3_qwen3_32b.md`, confounder recall 0) is unchanged.

**Disclosure (changelog item 29):** the Clinician was built by the LLM that wrote the second annotation and adjudicated these references, after E3 showed the confounders missing. Its prompt holds no trial-specific content, but its developer had seen the reference lists, so this score is not blind to development. The references await the clinician audit.

**Over the 12 core trials:** precision 25%, recall 43%, F1 32% (60 matched; 241 proposed, 139 in the references).

| Trial | Proposed (cited) | Reference | Matched | Precision | Recall | Dropped by the checks |
|---|---|---|---|---|---|---|
| ACURASYS | 20 (19) | 10 | 4 | 20% | 40% | – |
| CRISTAL | 20 (12) | 10 | 5 | 25% | 50% | – |
| APROCCHSS | 20 (14) | 12 | 3 | 15% | 25% | – |
| CHEST | 19 (19) | 12 | 5 | 26% | 42% | – |
| ADRENAL | 20 (16) | 10 | 6 | 30% | 60% | – |
| HYDRAREA | 20 (17) | 12 | 5 | 25% | 42% | – |
| BICAR-ICU | 20 (19) | 12 | 6 | 30% | 50% | – |
| CAPE COD | 20 (16) | 12 | 6 | 30% | 50% | – |
| STARRT-AKI | 20 (19) | 14 | 8 | 40% | 57% | – |
| EvK | 21 (20) | 10 | 3 | 14% | 30% | – |
| ECLS-SHOCK | 20 (17) | 13 | 3 | 15% | 23% | – |
| LOVIT | 21 (20) | 12 | 6 | 29% | 50% | – |
