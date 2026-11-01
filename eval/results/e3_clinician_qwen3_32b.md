# Trialist + Clinician: qwen3_32b confounders vs the adjudicated references

Run `runs/exit-H100-core-clinician-v2b`, replicate 1; replicate 2 not run yet. Prompt clinician-v2. Lenient one-to-one name matching. The frozen E3 result (`e3_qwen3_32b.md`, confounder recall 0) is unchanged.

**Disclosure (changelog item 29):** the Clinician was built by the LLM that wrote the second annotation and adjudicated these references, after E3 showed the confounders missing. Its prompt holds no trial-specific content, but its developer had seen the reference lists, so this score is not blind to development. The references await the clinician audit.

**Over the 12 core trials:** precision 26%, recall 46%, F1 33% (64 matched; 246 proposed, 139 in the references).

| Trial | Proposed (cited) | Reference | Matched | Precision | Recall | Dropped by the checks |
|---|---|---|---|---|---|---|
| ACURASYS | 21 (20) | 10 | 4 | 19% | 40% | – |
| CRISTAL | 21 (2) | 10 | 5 | 24% | 50% | – |
| APROCCHSS | 21 (3) | 12 | 5 | 24% | 42% | – |
| CHEST | 20 (17) | 12 | 6 | 30% | 50% | – |
| ADRENAL | 20 (8) | 10 | 5 | 25% | 50% | – |
| HYDRAREA | 20 (19) | 12 | 5 | 25% | 42% | – |
| BICAR-ICU | 21 (18) | 12 | 6 | 29% | 50% | – |
| CAPE COD | 21 (0) | 12 | 6 | 29% | 50% | – |
| STARRT-AKI | 20 (4) | 14 | 6 | 30% | 43% | – |
| EvK | 21 (4) | 10 | 3 | 14% | 30% | – |
| ECLS-SHOCK | 20 (18) | 13 | 7 | 35% | 54% | – |
| LOVIT | 20 (7) | 12 | 6 | 30% | 50% | – |
