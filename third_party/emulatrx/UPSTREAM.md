# Vendored: TrialLab/EmulatRx

- Source: https://github.com/TrialLab/EmulatRx
- Commit: `e353ee46e189` (2026-05-01)
- License: MIT (see `LICENSE`, kept unchanged)
- Paper: Li, Pan, Rajendran, Zang & Wang, Nat. Commun. 17:5501 (2026), doi:10.1038/s41467-026-74501-2
- Code DOI: 10.5281/zenodo.20016649

This is a read-only reference snapshot. Do not edit it except as listed below. Ported and extended code lives in `agentic_tte/`; each ported module names its upstream source in its docstring.

## Local modifications

1. **Credentials redacted.** `trialistUtils/main.py` and `trialistUtils/parse_info.py` contained a hardcoded OpenAI API key, organization and project ID. They now read `OPENAI_API_KEY`, `OPENAI_ORG` and `OPENAI_PROJECT` from the environment (lines tagged `[agentic-tte]`).
2. **`trials/` excluded.** Upstream `trials/NCT00475852/` holds MIMIC-IV-derived patient-level CSVs and pickles (git-lfs). These fall under the PhysioNet data use agreement and are never vendored. The per-criterion reference SQL in the same folder is also git-lfs, and upstream's LFS server refused downloads at snapshot time; re-fetch it on Hellbender when it becomes available.
3. **LFS pickle excluded:** `trialistUtils/results/trial_info/NCT03872011/trial_info_NCT03872011.pkl` (same reason). The JSON files next to it are kept; they are public ClinicalTrials.gov content plus upstream's parse output, and we use them as a regression fixture.
4. Upstream `.gitignore` dropped.
