# COSMIC baseline (SIRIUS 6: CSI:FingerID + COSMIC)

Runs **locally on the project owner's Mac** with the installed, logged-in SIRIUS 6.5.4 (academic licence).
No password or token is needed or stored here. Only public MassSpecGym spectra are processed. Never run this
on laboratory data.

## Input (`input/`)
| file | spectra | purpose |
|---|---|---|
| `test_orbitrap.ms`, `test_qtof.ms` | 300 test-fold molecules, one spectrum each | evaluation |
| `val_orbitrap.ms`, `val_qtof.ms` | 200 validation-fold molecules | calibration of COSMIC scores only (same validation-only rule as every other system) |

- Multi-compound SIRIUS `.ms` files: `>compound <MassSpecGym identifier>`, `>parentmass` (precursor m/z),
  `>ionization` ([M+H]+ / [M+Na]+), `>instrumentation`, `>collision <eV>`, then the raw MS/MS peak list.
  No formula or structure is given to SIRIUS.
- Test molecules are stratified by the published CSI:FingerID positive-mode training structures
  (`csi.bright-giant.com/v3.0/api/fingerid/trainingstructures?predictor=1`, 28,000 structures): 150 in the list
  (**leaky reference only**) and 150 not in the list (**primary**). All validation molecules are not in the list.
  Note: the published list may not contain proprietary (e.g. NIST) training structures.
- `subset_manifest.csv`: identifier, split, instrument, adduct, precursor m/z, collision energy, stratum.
  (No structures; truth is joined only at import time.)

## Run
`run_sirius.sh` runs, per split and instrument: `formulas -p <orbitrap|qtof>` → `fingerprints` → `classes`
(CANOPUS; needed by COSMIC) → `structures --database pubchem`, then writes the summary tables.
It is resumable: re-running skips finished compounds.

## Output (`results/`)
- `summaries_<split>_<instrument>/` — SIRIUS summary tables; the importer needs `structure_identifications.tsv`
  (top structure per compound, with `ConfidenceScoreExact` and `ConfidenceScoreApproximate`).
- `<split>_<instrument>.sirius` — SIRIUS projects (can stay local; not needed for import).
- `log_*.txt`, `environment.txt` — logs and SIRIUS version.

## Expected running time
Roughly 1–3 h in total for 500 spectra on a recent MacBook (formula computation is local; fingerprint,
class and structure steps use the SIRIUS web service, so time depends on server load and network).
High-mass spectra dominate the running time. The test subset runs first (~60% of the time).

## Import (done on the analysis side, not on the Mac)
`ingest_cosmic.py` in this folder is a copy of `scripts/cosmic/ingest_cosmic.py`; it needs the full repository
(derived data and frozen models) and is run from the repository root as
`python scripts/cosmic/ingest_cosmic.py <path to cosmic/results>`. Nothing needs to be run on the Mac for the import. It reports:
- per stratum (not in CSI training = primary; in CSI training = leaky reference);
- COSMIC **exact** mode as the main result, **approximate** mode ("correct or MCES distance ≤ 2") as a supplement;
- each system's confidence ranking of **its own** annotations (AURC, e-AURC, AUROC) and calibration (ECE,
  raw and validation-calibrated), with bootstrap 95% CIs. COSMIC ranks structure-database (PubChem) hits,
  the proposed system ranks spectral-library hits — the comparison is about confidence quality, not about
  which annotation engine finds more structures.
