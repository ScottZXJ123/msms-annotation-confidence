# COSMIC head-to-head on the same candidate set

This is the main COSMIC comparison. The earlier PubChem run (`../results`) is supplementary background.

Here CSI:FingerID/COSMIC searches **custom structure databases made of the molecules in our spectral library**. Both systems therefore choose among the same molecules:

| database | molecules | used for |
|---|---|---|
| `msg_test_a` (`db_test_a.tsv`) | test-phase library (train + val + test folds), incl. all 300 test query molecules | scenario (a): truth in library |
| `msg_test_b` (`db_test_b.tsv`) | `msg_test_a` minus the 300 test query molecules | scenario (b): truth absent |
| `msg_val_a`, `msg_val_b` | the same, built for the 200 validation queries (train + val folds) | calibration of COSMIC scores only |

- On our side the same removals are applied. In (a), each query spectrum itself is left out. In (b), all spectra of all 300 query molecules are removed.
- The databases keep only library molecules whose mass is within 0.05 Da of some query's neutral mass. CSI:FingerID can only retrieve structures whose formula matches a query formula candidate, so this changes no retrieval. It only keeps the databases small. Sizes are in `db_manifest.json`.
- Files contain public structures only (SMILES, InChIKey first block, name). There are no spectra and no laboratory data.

## Run (on the Mac, local SIRIUS 6.5.4, stored login)

```bash
conda deactivate
cd ~/msms-annotation-confidence/cosmic/customdb
bash run_sirius_customdb.sh 2>&1 | tee run_console.txt
```

The script does the following:

1. **Builds and checks the four databases.** It runs `sirius custom-db create` → `import` → `show`, and stops if a database has fewer structures than expected.
2. **Runs the chemistry once per split and instrument.** It runs `formulas -p <orbitrap|qtof>` → `fingerprints` → `classes` on the same `../input/*.ms` spectra as before.
3. **Searches each database.** It copies that project once per scenario and runs `structures --database msg_<split>_<a|b> --exp OFF`. Expansive search is off, so SIRIUS never falls back to PubChem.
4. **Writes the summaries.**

The script is resumable: finished steps are skipped. It needs no password or token. Expected time is roughly 30–60 min: database import takes a few minutes per database, and the rest is about twice the earlier run.

## Output (`results/`)

- `summaries_<split>_<instrument>_<a|b>/structure_identifications.tsv` — needed for import.
- `dbinfo_msg_*.txt` — database sizes as reported by SIRIUS.
- `log_*.txt`, `environment.txt`, `run_console.txt`.
- `*.sirius` projects and `dbs/*.siriusdb` can stay on the Mac.

## Import (analysis side)

Run from the repository root:

```bash
python scripts/cosmic/ingest_cosmic_customdb.py cosmic/customdb/results
```

- It checks that every returned structure belongs to the searched database.
- It stratifies by the published CSI:FingerID training structures. "Not in training" is primary; "in training" is a leaky reference.
- It evaluates (a) and (b) and their mixtures at the pre-declared in-library fractions π = 0.2 and 0.5.
- COSMIC scores are calibrated on the validation subset with the same rule as every other system.
- It reports positive and negative counts for every stratum and system, with bootstrap 95% CIs.
