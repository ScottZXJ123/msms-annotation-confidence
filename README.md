# msms-annotation-confidence

Calibrated trust tiers for MS/MS spectral-library annotations. The system sorts each query into one of three tiers: **trusted**, **suspicious**, or **likely absent from library**. It is CPU only and evaluated under the structure-disjoint MassSpecGym split.

- `CLAUDE.md` — the task specification (goal, known failure modes, prohibitions).
- `REPORT.md` — methods and all results, including negative results and baselines. It is rendered from `results/*.json`.
- `LOG.md` — chronological record of every decision, variant and abandoned direction.
- `WETLAB_PREREG.md` — the pre-registration handed to the collaborating laboratory.
- `config.yaml` — every setting. Each result JSON records the config hash, seed, script and commit.
- `cosmic/` — the SIRIUS 6 / COSMIC baseline package. It runs on the owner's machine.
  - `cosmic/customdb/` is the main comparison: SIRIUS searches our own library's molecules, so both systems face the same candidate set.
  - `cosmic/README.md` describes the supplementary PubChem run.

## Reproduce

```bash
pip install -r requirements.txt
make all        # download -> prepare -> search -> features -> train -> calibrate -> evaluate -> shifts
                # -> external source -> added baselines -> post-hoc analyses -> COSMIC package -> figures -> report -> audits
```

Approximate time: ~3 h on 2 CPU cores, ~8 GB RAM. Individual stages are available as Make targets (`make train`, `make evaluate`, …).

## Layout

| path | content |
|---|---|
| `src/msconf/` | library: chemistry, spectra, features, models, evaluation, engine-agnostic interfaces |
| `scripts/01–24` | pipeline stages (numbered in execution order; see `Makefile`; 17–24 were added after the main test evaluation) |
| `scripts/dev/` | validation-only diagnostics cited in `LOG.md` |
| `scripts/cosmic/` | COSMIC subset, custom structure databases, result import |
| `results/` | every number in the reports, with provenance |
| `reports/tables/`, `figures/` | generated tables and figures |
| `models/` | frozen primary model (for the pre-registration) |
| `data/cache/classyfire.jsonl` | ClassyFire lookups (public structures only) |

## Laboratory data

The files currently in `data/lab/` are **not used** in any experiment, result or text of this project (owner's instruction). No script reads that directory.

Laboratory files belong in `data/lab/`. That directory is git-ignored. Laboratory data must never be committed, uploaded to any web service (including SIRIUS and ClassyFire), or quoted in logs or reports. `src/msconf/interfaces.py` reads engine outputs (GNPS, SIRIUS, MS-DIAL, generic CSV) without network access (`lab_mode`).

## Scope

- Positive-ion ESI, [M+H]+ / [M+Na]+, Orbitrap and QTOF.
- 2-D identity only (InChIKey first block); no stereochemistry claims.
- "Absent from library" is not a claim of a new metabolite.
