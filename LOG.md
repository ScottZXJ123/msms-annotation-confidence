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

### 21:05 ClassyFire lookup complete
- 28,666 / 28,921 molecules (99.1%) found by InChIKey (existing ClassyFire results). For the 255 others a
  new ClassyFire classification request was attempted (public structures only); the ClassyFire query
  service answered HTTP 500 → these 255 molecules form the `unclassified` group (0.9% overall; train 0.6%,
  val 1.5%, test 2.2%). 0.3% of molecules are classified but lack a class-level node
  (→ `<superclass> [no class]`). 320 class-level groups, 20 superclasses, 754 subclasses
  (`results/families_summary.json`).
- External MassBank molecules: all 748 found in the ClassyFire cache.

### 21:08 Final features computed; one test-fold statistic seen
- `07_features.py` printed per-fold engine top-1 accuracy, including the test fold (scenario a, unweighted).
  This is an engine-level descriptive statistic, not a method result, and no decision was based on it;
  recorded here for transparency. The script now withholds test-fold accuracy.
- Descriptive (train/val only): engine top-1 accuracy in scenario (a) is 93.5% on train queries (train-only
  library) and 98.6% on val queries (train+val library). Structure-disjoint folds make val/test queries
  less crowded by near-isomers from other clusters than train queries; P(correct | in) therefore has a much
  lower error base rate at val/test time. Calibration is fitted on val-B, which matches the test condition.
- (b): 13.3% (train) / 18.6% (val) of queries have no candidate once the truth is removed → outside the
  study population (§4), counted.
- External set: 2,955 eligible query spectra from 348 molecules.

### 21:10 Pre-registered before touching the test fold
- Supplementary "near-correct" definition: top-1 Tanimoto to truth ≥ 0.8 (Morgan r2/2048)
  (`evaluation.approx_correct_tanimoto`). Primary definition unchanged (InChIKey first block).
- Tier thresholds frozen in config: trusted iff p_correct ≥ 0.9; absent iff p_in ≤ 0.2; otherwise suspicious.
- Primary system = M_full (decomposed P(in)·P(correct|in), label-shift correction with EM-estimated
  prevalence), with per-family calibration offsets only if the val-B cross-fit rule in 08_train.py says so.
- Headline prevalences π ∈ {0.2, 0.5}; sweep {0.05 … 0.9}.
- COSMIC: SIRIUS ≥5 needs an account + license for CSI:FingerID/COSMIC web services; not available in the
  build environment. A seeded 300-spectrum public test subset (one spectrum per molecule) is exported
  (`scripts/cosmic/make_subset.py`) with a runner and an ingestion script; the baseline is reported as
  NOT RUN until executed with the project owner's academic SIRIUS login.

### 22:00 Full training run #1 (all 17 systems, entropy + cosine) — `results/train_entropy.json` (superseded entries noted below)
- Val-A (hyperparameter-selection split) log losses of the two components (lower is better):
  | system | P(correct\|in) | P(in) |
  |---|---|---|
  | floor: metadata only | 0.0986 | 0.6689 (≈ chance 0.693) |
  | floor: similarity only | 0.0328 | 0.1825 |
  | logistic(s1, gap), decomposed | 0.0370 | 0.2055 |
  | base (no crowding) | 0.0219 | 0.0819 |
  | full (all crowding) | 0.0187 | **0.1034** |
  | full − family crowding | 0.0187 | 0.0753 |
- Crowding helps P(correct | in) (0.0219 → 0.0187), but the family-crowding group made P(in) much WORSE.

### 22:15 Diagnosis: a construction shortcut in `fam_lib_size` (§5.2) — `results/dev_feature_variants.json`
- Validation-only diagnostic with fixed hyperparameters, varying only the feature set
  (`scripts/dev/diag_feature_variants.py`). Adding `fam_lib_size` alone to the base set: P(in) train log
  loss 0.084 → 0.033 while val-A log loss 0.083 → 0.171. Removing only `fam_lib_size` from the family group
  restores P(in) (0.0813 with base + other family features; 0.0767 with base + spectral + structural +
  family-without-libsize).
- Mechanism: the fixed family tables (`fam_spec_crowd_train`, `fam_struct_crowd_train`) identify the family;
  in the training phase the library count of that family is a fixed number N_c in scenario (a) and N_c − 1
  in the paired scenario (b) whenever the top-1 shares the truth's family. The GBDT memorizes N_c per family
  and learns "count one short of full ⇒ absent". This is exactly the collaborator's earlier failure
  (the pool construction leaks the answer); it does not transfer because val/test libraries have other
  counts. It is invisible in a random split of the training data.
- Decision: `fam_lib_size` is removed from every proposed feature set (kept as `M_full_with_libsize`
  only to document the artifact). The same logic predicts a weaker version of the artifact for window
  counts (`n_win_mol`, `n_win_spec` together with `precursor_mz`); removing them improves val-A P(in)
  slightly (0.0753 → 0.0739) but hurts P(correct|in) slightly (0.0201 → 0.0205) → added as candidate
  `M_full_nocounts` rather than decided by hand. Library-count features tied to the observable top-1
  (`top1_struct_nbrs`, `top1_specdens`) do not show the artifact: removing them worsens val-A P(in)
  (0.0753 → 0.0828).
- Primary selection rule (validation only, fixed now): among {M_full, M_full_nocounts, M_full_minus_family,
  M_base}, the lowest val-A log loss of the combined p_correct at π = 0.5 (calibration fitted on val-B);
  M_base is a candidate so that "no crowding" can win. Then the family-offset rule as before.
- Retraining the affected systems (training run #2 for: M_full, M_full_nocounts, M_full_with_libsize,
  M_sim_plus_crowd, M_full_minus_spec, M_full_minus_struct, M_full_direct, M_full_lr; cosine M_full).

### 22:55 Training run #2 + validation decisions (`results/selection.json`, `results/train_entropy.json`)
- Val-A log loss of combined p_correct (calibrated on val-B) at π = 0.5 / 0.2:
  M_full_minus_family 0.0691 / 0.0585; M_full 0.0706 / 0.0608; M_full_nocounts 0.0716 / 0.0612;
  M_base 0.0780 / 0.0655. → **Primary = M_full_minus_family** (meta + similarity + analog + spectral
  crowding + structural crowding; no family-level features). Family features (after removing the
  artifact) do not add out-of-sample value on validation; they remain in the ablations and family
  groupings remain the unit of the calibration analysis.
- Family-offset rule: cross-fitted mean |obs − pred| over sufficient top-1-class groups 0.0264 (global
  isotonic) vs 0.0264 (with offsets) → no 10% improvement → **no per-family offsets**.
- Tier sanity on val-B (importance-reweighted, in-sample calibration): at π = 0.2 trusted 14.1% of queries
  (accuracy 97.1%), suspicious 9.4% (55.2%), absent 76.5% (0.7%); at π = 0.5 trusted 39.7% (98.7%),
  suspicious 13.9% (62.6%), absent 46.4% (1.3%). Thresholds 0.9 / 0.2 kept as frozen.
- Cosine engine: M_full_minus_family added so the engine-swap check uses the same primary configuration.
- FROZEN from here on: models in data/derived/models/, config, thresholds, primary system.

### 23:05 Harness dry run on val-A (code debugging only; `results/dryrun_valA_*.json`)
- Found and fixed a bug in the wet-lab batch sampler (row positions taken relative to a filtered frame),
  which had made every simulated batch ≈ 50% in-library. After the fix, EM prevalence estimates track the
  truth on val-A (π = 0.1 → 0.092, 0.2 → 0.19, 0.5 → 0.49).
- Dry-run behaviour at π = 0.1: without label-shift correction the trusted tier is over-confident
  (direct full-feature model: predicted 98.5%, observed 91%); with EM correction predicted 96%, observed 99%.
- The suspicious tier was under-predicted on val-A (predicted 0.54–0.65 vs observed 0.65–0.72).

### 23:20 Calibrator choice (validation only; `results/dev_calibrators.json`, `results/calibration_*.json`)
- Cross-fitting val-A ↔ val-B for the primary system: P(correct|in) log loss isotonic 0.0312, Platt 0.0273;
  P(in) isotonic 0.1100, Platt 0.1072. The mid-range (0.2–0.9) calibration gap of P(in) flips sign with the
  fitting direction (−0.11 vs +0.10): the two structurally disjoint halves of val genuinely differ in the
  middle range — between-set heterogeneity, not calibrator noise. Recorded as a limitation for the
  suspicious tier.
- Rule adopted for ALL systems (baselines included, same calibration budget): per component choose isotonic
  vs Platt by 2-fold cross-fit log loss on val-A/val-B, then refit on the whole validation fold
  (`scripts/08b_recalibrate.py`). Result: all LightGBM components → Platt; single-score baselines → isotonic
  for P(in) / direct, Platt for P(correct|in). Prevalence of the calibration set π_cal = 0.539.
- Family-offset rule re-checked under the new calibrators: gap 0.0181 (global) vs 0.0183 (offsets) → still no
  offsets. Primary stays M_full_minus_family.
- Validation data are now fully used for calibration; the val-A dry run is no longer an independent check.
  The next step is the single final evaluation on the test fold.
