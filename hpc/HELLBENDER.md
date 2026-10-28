# Hellbender facts (confirmed 2026-10-24; storage and CPU partitions 2026-10-26)

The source of truth for paths, quotas, Slurm settings and known pitfalls on MU's Hellbender cluster. Everything here was checked on the cluster, not assumed. Update this file whenever something on the cluster changes, and keep `hpc/env.example.sh` in line with it.

## Locations

| What | Path | Notes |
|---|---|---|
| Repo checkout | `~/data/Agentic-TTE` | Cloned from GitHub. Never edit code here; see "Sync rules" |
| Per-project caches | `~/data/Agentic-TTE/.hpc-data/` (`$ATTE_PROJECT_DIR`) | CT.gov cache, LLM cache, uv cache. Git-ignored. Exit-run replicates each get a fresh cache in `.hpc-data/exit/<GPU>[-<tag>]/<model>/rep<k>/`, and reproducibility-test replicates in `.hpc-data/repro/<GPU>/<config>/<model>/rep<k>/` |
| Ollama models | `/cluster/pixstor/data/$USER/tte-project/ollama-models` (`$OLLAMA_MODELS`) | Shared with older projects. Do not move |
| Ollama container | `/cluster/pixstor/data/$USER/tte-project/containers/ollama-0.40.3.sif` (`$OLLAMA_SIF`) | 3.2 GB |
| Run outputs | `~/data/Agentic-TTE/runs/` | Git-ignored. `runs/replay0-<jobid>/`, `runs/interactive/`, `runs/exit-<GPU>[-<tag>]/<model>/rep<k>/` for the exit run (`runs/exit-<GPU>.jobs` lists batch job IDs), and `runs/repro-<GPU>/<config>/<model>/rep<k>/` for the reproducibility test |
| Slurm logs | `~/data/Agentic-TTE/logs/` | Git-ignored |
| Apptainer build scratch | `/local/scratch/$USER/apptainer-{tmp,cache}` | Node-local, temporary |

## Storage and quotas

| Area | Path | Quota | Use |
|---|---|---|---|
| Home | `/home/$USER` (on `/cluster/pixstor`) | 50 GB | Small config only |
| Personal Pixstor | `~/data` → `/mnt/pixstor/data/$USER` | 500 GB, **nearly full** | Repo, caches, outputs |
| Older project area | `/cluster/pixstor/data/$USER/tte-project` | **The same quota as `~/data`**: `/cluster/pixstor/data/$USER` *is* `~/data` (same device and inode, checked 2026-10-26) | Models, container, MIMIC, OMOP |
| Local scratch | `/local/scratch` | 1.6–3.2 TB per node | Temporary build and job files; not shared between nodes, not kept: never store models or data here |
| Lab group storage | `/cluster/pixstor/shyuc-hpc-lab` (checked 2026-10-26) | Not known: `df` shows only the whole filesystem (15 TB, 2.5 TB free); ask RCSS | Writable by the user. Mode `drwxrws---`, group `rss-rde-shyuc-lab-rw`: every lab member can read what is put there. Has `.snapshots`, unlike personal space. Other members keep per-user folders. The user's folder is `/cluster/pixstor/shyuc-hpc-lab/$USER` (`$ATTE_LAB_DIR`), **mode 700**: only the user can enter it. The user confirmed on 2026-10-26 that the lab holds a PhysioNet licence for MIMIC-IV, so MIMIC-IV, its Postgres build, the OMOP CDM and the vocabulary live there. PhysioNet credentialing is per person: open a subfolder to the group (`chmod`) only for credentialed members |

- `df` reports whole-filesystem totals, not your quota, so it cannot tell you whether you are near your limit.
- Nothing is backed up. Code lives on GitHub; models can be re-pulled; results must be copied off the cluster if they matter.
- Cluster storage is approved only for lower data classification levels. DCL3/DCL4 data (identifiable EHR) needs an arrangement with RCSS before it comes to Hellbender. Phase 0 uses only public ClinicalTrials.gov text.

## Data already on Hellbender (found 2026-10-26)

All of this is under `/cluster/pixstor/data/$USER/` (`$P` below). The user confirmed on 2026-10-26 that the MIMIC-IV data use agreement covers them and that RCSS has no issue with MIMIC-IV on Hellbender.

| What | Path | Size | Notes |
|---|---|---|---|
| MIMIC-IV 3.1, raw (`hosp`, `icu` only: no `ed` or `note`) | `$P/mimic-source/mimic-iv-3.1/` | hosp 5.9 GB, icu 4.1 GB | PhysioNet release, with `SHA256SUMS.txt`. The build logs list it as `/home/$USER/data/mimic-source/...` |
| MIMIC-IV 3.1, zipped | `$P/emulatrx-work/EmulatRx/mimic-iv-3.1.zip` | 9.9 GB | The same release again |
| MIMIC-IV 3.1 in Postgres (native schemas `mimiciv_hosp`, `mimiciv_icu`) | `$P/mimic-postgres/` (data directory; moved 2026-10-26 to `$ATTE_LAB_DIR/mimic-postgres`, with a link at the old path), env `$P/conda_envs/mimic-pg`, build code `$P/mimic-code` (MIT-LCP) | 127 GB | Built and validated: row counts match the source for all 28 tables (`$P/mimic-validate.log`). It served on `127.0.0.1:5433` on node g030 when built; it is not a running service. **The mimic-code derived concepts (schema `mimiciv_derived`) are built:** the log (`$P/mimic-concepts-build.log`) ends with the first-day tables (94,458 rows = ICU stays) and has 0 errors (checked 2026-10-28). Started only for exports, by `hpc/export_mimic_derived.sbatch` (env `ATTE_MIMIC_PG`, `ATTE_MIMIC_PG_ENV`), on a private socket in `/local/scratch` with no TCP |
| mimic-code derived exports for E2 (a) | `$ATTE_MIMIC_DERIVED` = `$ATTE_LAB_DIR/mimic-derived/` (gzipped CSVs: icustay_detail, first_day_lab, first_day_bg, ventilation, norepinephrine) | about 50–100 MB (estimate) | Patient-level, in the mode-700 lab folder; never copied out. Written by `hpc/export_mimic_derived.sbatch`, read by `eval/e2a_mimic_code.py` |
| Athena vocabulary, unpacked | `$P/tte-project/athena_vocab/` | 1.5 GB | CONCEPT, CONCEPT_RELATIONSHIP, CONCEPT_ANCESTOR, CONCEPT_SYNONYM, DOMAIN, VOCABULARY, RELATIONSHIP, CONCEPT_CLASS, DRUG_STRENGTH, CONCEPT_CPT4 (2026-07-07). Version not recorded |
| Athena vocabulary, frozen package | `$P/ct_tte_fm/data/external/ct_tte_sources/S07_athena_omop/raw/` (zip), manifest `$P/ct_tte_fm/data/manifests/sources/S07/` | 263 MB zipped | Frozen 2026-09-12, SHA256 `ffab28df4048…`: RxNorm 20260105, LOINC 2.80, SNOMED 2025 (Int/US/UK), ICD10CM FY2026, ICD10PCS 2026, CPT4 2025; 2,191,251 concepts, 1,028,153 standard. The reproducible choice |
| OHDSI MIMIC-IV ETL mappings | `$P/ohdsi-mimic/` (`git clone https://github.com/OHDSI/MIMIC`), env `ATTE_OHDSI_MIMIC` | ~3 MB | `custom_mapping_csv/gcpt_meas_lab_loinc_mod.csv`: curated lab itemid → LOINC for 1,401 items (Apache-2.0). The ETL uses it for labevents, with reviewed overrides; no patient data |
| EmulatRx working copy | `$P/emulatrx-work/EmulatRx/` | – | `trials/NCT00475852` (24 KB), `results/runs/NCT04134403_mimiciv_v1/` (an earlier run's JSON reports). Patient-level CSVs there never go into git |

**`$P` is `~/data`** (checked 2026-10-26: `~/data` → `/mnt/pixstor/data/$USER`, and `~/data/mimic-source` and `$P/mimic-source` have the same device and inode). So everything above counts against the one nearly full 500 GB quota, including the Ollama models and any new OMOP database. The 2026-10-26 search of `$P` therefore covered `~/data` too. Candidates to free space, both the user's call:
- the duplicate 9.9 GB zip, once the raw copy passes its checksums;
- the 127 GB Postgres build, if the native-schema comparison is not needed.

**Athena S07, unpacked** (2026-10-26) to `$P/tte-project/athena/S07/Athena Verison/`. The zip's folder name has a space and a typo; quote the path.

**Both Athena copies are incomplete for MIMIC-IV** (checked 2026-10-26 against `VOCABULARY.csv`). S07 and `tte-project/athena_vocab` each hold only ICD10CM, ICD10PCS, LOINC, RxNorm and SNOMED. **ICD9CM, ICD9Proc and NDC are missing.** In the 1% ETL pilot, this left every ICD-9 diagnosis and procedure (about half of MIMIC-IV's coding) and every prescription (NDC) unmapped. A new Athena download with those vocabularies is needed before the full ETL.

**Athena release v5.0 29-AUG-26** (downloaded 2026-10-26; zip 722 MB, about 4.2 GB unpacked) goes to `$P/tte-project/athena/v5-2026-08-29/` (`ATTE_VOCAB`).
- **Vocabularies:** ICD9CM, ICD9Proc, ICD10CM, ICD10PCS, ICD10, SNOMED (2026), RxNorm (20260601), RxNorm Extension, NDC (20260823), ATC, HCPCS, CVX, LOINC, UCUM, ICDO3, NUCC, Medicare Specialty, CMS Place of Service, UB04, and the OMOP infrastructure vocabularies.
- **CPT4** is shipped as `CONCEPT_CPT4.csv` and only joins `CONCEPT.csv` after `./cpt.sh <UMLS API key>` (needs Java and internet). The MIMIC ETL does not need CPT4; do it before an EHR source with CPT procedure codes.

### Moving a folder to the lab space

The procedure is copy, verify, then delete. `hpc/copy_to_lab.sbatch` copies with `rsync -a`, rechecks, and prints `VERIFIED`; it never deletes the source.
1. Create the user folder once, as `mkdir -p "$ATTE_LAB_DIR" && chmod 700 "$ATTE_LAB_DIR"`. Many source files are world-readable, so privacy comes from this folder's mode.
2. Run `sbatch -p "$SLURM_CPU_PARTITION" --export=ALL,SRC=<old>,DST=<new> hpc/copy_to_lab.sbatch`, and wait for `VERIFIED` in `logs/copy-<job>.out`.
3. Remove the old folder and leave a link at its path (`ln -s <new> <old>`), so older projects that use the old path keep working.

Never copy a Postgres data directory while its server runs.

## Slurm

- Account `general`, QOS `normal`.
- GRES type names are **case-sensitive**: `gpu:H100:1` works; `gpu:h100:1` fails with "Requested node configuration is not available".
- Never request MIG slices (`1g.10gb`, `2g.20gb`); they are too small for the models.

| Partition | GPUs | Limits | When to use |
|---|---|---|---|
| `gpu` | 3 nodes × 2 H100 (64 CPU, ~490 GB RAM); nodes g032–g034 report **H100 NVL, 95830 MiB** (2026-10-25) | 2 days, not preemptible | Final benchmark runs. Queue waits can be many hours. OnDemand g020 is a different type (H100 80GB HBM3), so never pool runs from both |
| `gpu_requeue` | H100 (×8, ×4, ×2 nodes), A100 ×4 on 15 nodes, H200, L40S | 2 days, **preemptible** | Development and fast turnaround; `--gres=gpu:A100:1` usually starts quickly |

CPU partitions (`sinfo -s`, 2026-10-26). `general` nodes come in three sizes (`sinfo -p general -N -o "%N %c %m"`): 128 cores with about 490 GB (c001, c022 types), and 48 cores with about 363 GB (c149 type). The ETL's 16 CPUs and 96 GB fit every node.

| Partition | Nodes | Limits | When to use |
|---|---|---|---|
| `general` | 56 CPU nodes (`c001–c049`, …) | 2 days, not preemptible | The MIMIC-to-OMOP ETL (`hpc/etl_mimic.sbatch`) |
| `requeue` (the default) | 258 CPU and GPU nodes | 2 days, preemptible | Short CPU jobs that may restart |
| `interactive` | 10 CPU nodes | 4 hours | Interactive work, item-map proposals |
| `logical_cpu`, `logical_cpu2` | 4 nodes, 1 node | 2 days | Not used by this project |
| `rss-class` | 26 nodes | 8 hours | Class use; not this project |

Jobs request 8 CPUs and 64 GB RAM, which fits every GPU node. The model lives in GPU memory, not system RAM.

**GPU consistency:** results can differ slightly between GPU types (floating point). Every run writes the GPU name to `manifest.txt`. Run each final benchmark for all models on one GPU type, and report it in the methods.

## Models present in `$OLLAMA_MODELS`

`qwen3:32b` (20 GB, project default), `gpt-oss:20b` (13 GB), `alibayram/medgemma:27b` (16 GB), `qwen3:8b`, `mistral-small:24b`, `deepseek-r1:14b`, `phi4:14b`, `gemma3:12b`, `llama3.1:8b`, `qwen2.5:7b-instruct`, `mistral:7b`, `meditron:7b`, `cniongolo/biomistral`, `llama3.2:3b`, `nomic-embed-text`. All fit on a single A100, H100 or H200.

`ATTE_THINK=false` for qwen3; `ATTE_THINK=none` for models without a reasoning switch (gpt-oss, medgemma, the rest).

## Known pitfalls and their fixes

| Symptom | Cause | Fix (already in the scripts) |
|---|---|---|
| `listen tcp 0.0.0.0:11434: bind: address already in use` | The `ollama/ollama` image sets `OLLAMA_HOST=0.0.0.0:11434`, which overrides the host value inside Apptainer | Pass every `OLLAMA_*` setting with `--env` on `apptainer exec` |
| gpt-oss:20b drafts end in `ReadTimeout: timed out` (exit run, 9 of 20) | Runaway generation with the default reasoning level: about 1,000 completion tokens per call, stalling on the same criteria | Output capped at 8,192 tokens (`ATTE_NUM_PREDICT`); timeouts fail one item, not the draft; use `ATTE_THINK=low` for a gpt-oss arm |
| Replicates differ at temperature 0 (exit run: qwen3 6/10, medgemma 4/10 substantively identical) | **Server state** (repro test 2026-10-24): on a restarted server, replicates are identical at both parallelism settings. The parallelism setting itself also changes outputs, and outputs appear to depend on earlier requests | `OLLAMA_NUM_PARALLEL=1` by default; Ollama restarted per replicate; model unloaded before every trial (`ATTE_UNLOAD_PER_TRIAL=1`). All recorded in `manifest.txt`. Order independence confirmed 2026-10-25: protocols identical with trials forward or reversed |
| Every batch manifest says `restart_count=2` though the jobs ran once (final run, 2026-10-25) | Submitted from inside the OnDemand job, whose own `SLURM_RESTART_COUNT` (it had been requeued twice) was passed on by `--export=ALL` | `exit_run_submit.sh` unsets `SLURM_RESTART_COUNT` before submitting. To tell a real restart apart, check that the job ID matches `runs/exit-<label>.jobs` and that start and finish are continuous |
| `interactive_ollama.sh` refuses: "runs with '…', not '…'" | A server started with other parallelism or flash-attention settings is still running | `source hpc/interactive_ollama.sh stop`, then start again |
| `no CONCEPT.csv in .` (or `missing ./hosp/...`) from `agentic-tte etl ...` | A new shell without `source hpc/env.sh`: `$ATTE_VOCAB` / `$ATTE_MIMIC_SOURCE` are empty, and an empty path is `.` | `source hpc/env.sh` in every new shell; the commands now say so |
| `srun -p general ... uv run ...` from an Open OnDemand session failed with `execve(): uv: No such file or directory`, on the session's node (g034) (2026-10-28) | Inside an allocation, `srun` starts a step in that allocation (`-p` is ignored), and the step's PATH lacks `~/.local/bin` | Run CPU work with `sbatch` scripts, which add `~/.local/bin` to PATH and get their own node (`hpc/e2a_mimic_code.sbatch`) |
| `copy_to_lab.sbatch` said NOT VERIFIED for `mimic-postgres` with one difference, `.S...p..... socket/.s.PGSQL.5433` (2026-10-26) | A leftover Postgres socket file: a runtime file, not data. All 1,875 data files matched | Moved by hand after checking there was no `postmaster.pid`; the script now leaves sockets out of its check |
| Port 11434 busy on OnDemand nodes | A personal Ollama 0.13.5 often runs there | Project uses port 11500 interactively and a per-job port in batch |
| `apptainer pull` → "Disk quota exceeded" | Build files and cache went to `~/data` | `APPTAINER_TMPDIR`/`APPTAINER_CACHEDIR` on `/local/scratch`, and `apptainer pull --disable-cache` |
| `WARNING: Error changing the container working directory` | `~/data` is a symlink to `/mnt/pixstor`, which is not bound into the container | Harmless. Never pass paths under `~/data` into the container |
| `Requested node configuration is not available` | Lowercase GRES (`h100`) | Use `gpu:H100:1` |
| Clone/push "Write access to repository not granted" (403) | Fine-grained token without access to this repo | Token with Agentic-TTE → Contents: Read and write |

## Environment notes

- `uv` is installed at `~/.local/bin`. The conda `(base)` environment is active by default and is ignored by `uv run`.
- `TMPDIR` is set to `~/data/tmp` in the user's shell profile.
- The OnDemand node g020 has outbound internet (Docker Hub and ClinicalTrials.gov both worked). Batch compute nodes are not yet confirmed, so trial records are pre-fetched into the cache.
- The OnDemand Jupyter session runs on `gpu_requeue` and can be preempted.

## Sync rules

1. Code changes are made on the PC, committed and pushed to GitHub. On Hellbender, only `git pull`.
2. `hpc/env.sh` is per-user and git-ignored. When `hpc/env.example.sh` changes, copy the new lines into `env.sh` by hand.
3. If something has to be fixed on Hellbender first, record it here and in the scripts on the PC, so the next `git pull` matches what works.
4. Never write into tracked folders on Hellbender (`eval/`, `agentic_tte/`, `hpc/`, ...). Run outputs, scores and archives go under `runs/`, which is git-ignored, e.g. `uv run python eval/score_phase0.py runs/exit-H100-final --out runs/results`. Untracked files in tracked folders block `git pull` (it happened with `eval/results/repro/` on 2026-10-25).
