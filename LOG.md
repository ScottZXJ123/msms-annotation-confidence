# Experiment log

Chronological. Every attempted variant, its configuration and the decision taken, including abandoned
directions. Times are US Eastern. Numbers quoted here are copied from the named result files
(`results/*.json`), which carry the script / config hash / seed / commit that produced them.

---

## 2026-09-26

### 19:40 Setup
- Environment: CPU only (2 cores, 7.8 GB RAM in the build container; pipeline designed for 16 GB).
- Data: MassSpecGym **v1.5** (`MassSpecGym1.5.tsv`), not v1. Reason: v1.5 fixes evaluation pitfalls
  reported in "MassSpecGym in the Wild" (2025/26); same schema, RDKit-canonical SMILES. Official folds used
  unchanged (MCES-based, structure-disjoint).
- Git identity set to the project owner; no AI co-author trailers (§6).

### 19:50 Spectrum processing fixed a priori (config `spectrum_processing`)
- Same processing for query and library: drop peaks at m/z > precursor − 1.6 Da, noise < 1% of base
  peak, merge peaks closer than 0.05 Da, sum-normalize, min 3 peaks after processing.
- Consequence: 11.8% of spectra have < 3 peaks after cleaning and are dropped everywhere (train 11.4%,
  val 14.5%, test 13.0%) — `results/prepare_audit.json`. Not tuned; this is a population restriction and is
  reported as such.

### 19:55 Structure standardization
- One RDKit pipeline (Cleanup → LargestFragment → Uncharge; no tautomer canonicalization because the
  standard InChI already normalizes mobile-H tautomers and RDKit tautomer canonicalization can alter the
  InChIKey skeleton unpredictably). Identity = InChIKey first block of the standardized structure.
- 33 of 28,929 provided first-block keys change after standardization; 8 molecules merge; **0 merged
  molecules span folds** (so no fold contamination) → 28,921 molecules.

### 19:58 Precursor tolerance (decided on TRAIN fold only)
- Measured |Δm/z| between spectra of the same molecule and same adduct in the train fold.
  Coverage: 5 ppm 95.1%, 10 ppm 97.3%, 15 ppm 98.4%, **20 ppm 99.2%**, 30 ppm 99.7%.
- Rule (fixed before looking): smallest tolerance in {5,10,15,20,30} ppm with ≥ 99% coverage → **20 ppm**.
- Engine does not filter by adduct (like GNPS library search); an [M+Na]+ query can retrieve [M+H]+
  library spectra of other molecules if the m/z coincides.

### 20:00 ClassyFire access
- `classyfire.wishartlab.com` over HTTPS is unreachable from the build environment (connection reset);
  the Fiehn-lab ClassyFire cache (`cfb.fiehnlab.ucdavis.edu/entities/<InChIKey>.json`) returns existing
  ClassyFire results by InChIKey → used as the primary lookup (it is ClassyFire output, so §4 is unchanged).
  HTTP entity endpoint of classyfire.wishartlab.com as second lookup. Cache: `data/cache/classyfire.jsonl`.
- MassSpecGym also ships `MassSpecGym_classyfire_all.tsv`, but as an unordered bag of ancestor terms
  (class level not recoverable reliably) → not used for the family definition.
- Handling of partial classifications (logged, not silent): class missing → `<superclass> [no class]`;
  subclass missing → `<class> [no subclass]`; not found anywhere → `unclassified`.

### 20:02 Structural neighbour graph
- Morgan r=2 / 2048 bits; neighbour lists with Tanimoto ≥ 0.4 for all molecules
  (`results/struct_neighbors_summary.json`). Median max-Tanimoto of test molecules to the train fold: 0.375
  (95th pct 0.70) — consistent with the MCES split.

### 20:04 Engine / search design
- Flash entropy search (ms_entropy 1.5.2). Checked that the open-search score equals the identity-search
  score for every library spectrum inside the precursor window (max |Δ| 3e-7 over 1,142 pairs, also equal
  to the exact pairwise function). Hence one open search per spectrum gives: candidate scores (window),
  analog hits (outside window) and neighbourhood densities; library composition is applied as masks.
- Nested reference libraries (config `scenarios.phases`): train queries see the train fold only;
  val queries see train+val; test queries see train+val+test. So **no test-fold spectrum is ever a
  library entry or a candidate for any training query** (§5.1), while each query's own fold is present
  (needed for scenario (a)).
- Scenarios per query: (a) leave-one-spectrum-out (the query spectrum itself is never in the library);
  (b) remove every spectrum of the true molecule; (c) additionally remove library molecules with
  Tanimoto ≥ 0.7 to the truth.
- **Query pool construction (§5.2)**: only query spectra for which (a) is possible (another spectrum of the
  same molecule falls in the precursor window) are used, and each such query is used in BOTH (a) and (b).
  Rationale: if absent examples came from a different molecule population (e.g. singletons) than
  in-library examples, metadata alone could separate them. With paired (a)/(b), the query spectrum and
  its metadata are identical across the two labels; only the library differs.
- Train fold: at most 5 query spectra per molecule (seeded) + weights 1/n per molecule; val/test: all
  eligible spectra, weights 1/n per molecule so every molecule counts once.
- Search pass: 6 min on 2 cores. Eligible queries: train 165,897 (61,724 used after the cap, 17,681 mol),
  val 15,999 (2,626 mol), test 14,706 (2,490 mol) — `results/search_summary.json`.

### 20:10 Evaluation harness written before any model (§2)
- `src/msconf/evaluation.py`: deployment-like batches (one spectrum per molecule, molecule either in or
  absent, exact prevalence π), frozen tiers, wet-lab simulation (40 per tier, repeated), Wilson CIs,
  risk–coverage/AURC, calibration metrics, per-group calibration with minimum group size, Bayes
  prior-shift + EM prevalence estimation, molecule-cluster bootstrap.

### 20:15 Early observation (from train/val only)
- In scenario (a), top-1 accuracy of plain entropy library search is very high (~94–97% in a debug subset):
  MassSpecGym molecules usually have several spectra (different collision energies) of the same
  compound, so LOO retrieval of the right molecule is easy. Implication: in-library mis-annotations are
  a small fraction; most wrong top-1 annotations in a realistic batch come from library absence. The
  crowding hypothesis concerns the in-library errors, so it is tested on the P(correct | in) component
  separately as well as on the combined output.

### 20:20 Training pipeline trial (interim, NOT used)
- Ran with an incomplete family table (ClassyFire lookup 25% done → 77% "unclassified") only to test code
  paths and timing. Observed (val-B, in-sample isotonic, not a result): meta-only floor log loss 0.674
  (≈ chance for a balanced mix → no obvious metadata shortcut); sim-only 0.205; LR(s1, gap) 0.205;
  base (no crowding) 0.119; full 0.125; full minus family 0.110. Family features built on the interim
  table hurt. Everything is re-run after the ClassyFire lookup completes.

### 20:28 External source for the data-source shift test
- MassSpecGym has no per-spectrum source field → leave-one-source is done with an external library.
- MassBank-data 2026.03 (Zenodo 10.5281/zenodo.19590880): 139,240 records → 62,823 positive-mode MS2
  [M+H]+/[M+Na]+ high-resolution records mapped to QTOF/Orbitrap; 57,622 of them are molecules already in
  MassSpecGym and are removed (cross-library de-duplication at the molecule level); of 999 new molecules,
  748 have Tanimoto < 0.7 to every train molecule (MCES not computed — proxy, stated in REPORT);
  3,928 spectra (3,307 after the min-peak filter). Contributors: Eawag 2,308, BAFG 506, mFam 300, …
  (`results/external_prepare.json`). These spectra are added to the library as a new source.
