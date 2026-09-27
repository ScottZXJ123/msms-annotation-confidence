# Experiment log

Chronological. Every attempted variant, its configuration and the decision taken, including abandoned
directions. Times are US Eastern (container clock; result files carry exact UTC stamps). Numbers quoted here are copied from the named result files
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

### 20:55 ClassyFire lookup complete
- 28,666 / 28,921 molecules (99.1%) found by InChIKey (existing ClassyFire results). For the 255 others a
  new ClassyFire classification request was attempted (public structures only); the ClassyFire query
  service answered HTTP 500 → these 255 molecules form the `unclassified` group (0.9% overall; train 0.6%,
  val 1.5%, test 2.2%). 0.3% of molecules are classified but lack a class-level node
  (→ `<superclass> [no class]`). 320 class-level groups, 20 superclasses, 754 subclasses
  (`results/families_summary.json`).
- External MassBank molecules: all 748 found in the ClassyFire cache.

### 20:57 Final features computed; one test-fold statistic seen
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

### 21:00 Pre-registered before touching the test fold
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

### 21:15 Full training run #1 (all 17 systems, entropy + cosine) — `results/train_entropy.json` (superseded entries noted below)
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

### 21:24 Diagnosis: a construction shortcut in `fam_lib_size` (§5.2) — `results/dev_feature_variants.json`
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

### 21:38 Training run #2 + validation decisions (`results/selection.json`, `results/train_entropy.json`)
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

### 21:46 Harness dry run on val-A (code debugging only; `results/dryrun_valA_*.json`)
- Found and fixed a bug in the wet-lab batch sampler (row positions taken relative to a filtered frame),
  which had made every simulated batch ≈ 50% in-library. After the fix, EM prevalence estimates track the
  truth on val-A (π = 0.1 → 0.092, 0.2 → 0.19, 0.5 → 0.49).
- Dry-run behaviour at π = 0.1: without label-shift correction the trusted tier is over-confident
  (direct full-feature model: predicted 98.5%, observed 91%); with EM correction predicted 96%, observed 99%.
- The suspicious tier was under-predicted on val-A (predicted 0.54–0.65 vs observed 0.65–0.72).

### 21:53 Calibrator choice (validation only; `results/dev_calibrators.json`, `results/calibration_*.json`)
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

### 21:57 FINAL test evaluation (single run of the frozen procedure) — `results/eval_sweep.json`, `eval_bootstrap.json`, `eval_wetlab.json`, `eval_groups.json`, `eval_incremental.json`, `eval_extra.json`
- Ran `scripts/09_evaluate.py` once on the test fold with the frozen models/thresholds.
- Metric bug found while reading the output: the oracle ordering inside `risk_coverage` (used only for
  e-AURC) sorted by the wrong array, giving negative e-AURC. AURC, risk@coverage, coverage@risk,
  calibration and all tier/wet-lab numbers are unaffected. Fixed (unit check: oracle ranking → e-AURC 0)
  and the identical frozen evaluation is re-run so that every stored e-AURC is correct. Nothing about the
  method, models, thresholds or selection changed.
- Second harness bug found in the same read-through: `summarize_wetlab` averaged an object-dtype column of
  numpy booleans; when a tier was empty in some repeats (only B_lr_dec at π ≤ 0.2 — its trusted tier is
  almost always empty there), pandas summed numpy booleans with logical OR and the "prediction inside
  95% CI" rate collapsed to 1/1000. Fixed by casting to float. Rates for systems without empty tiers
  (including the proposed system) were not affected. The frozen evaluation is re-run once more.

### 22:17 Main test evaluation complete (after the two harness-bug re-runs); frozen-procedure results stored
- The proposed system and every pre-registered baseline were evaluated once more with identical frozen
  models; only the two reporting bugs above differ from the first pass.
- Pre-planned shift tests launched in this order: library-level cross-instrument scenario (a_x,
  `16_scenario_cross_instrument.py`), external MassBank source (`12_external_eval.py`), leave-one-instrument
  model training (`10_shift_instrument.py`). None of them fits or selects anything on test data (the
  instrument test re-trains on train/val rows of one instrument type with the frozen recipe).

### 22:20 Two further baselines requested by the project owner — ADDED AFTER THE MAIN TEST EVALUATION
The proposed method is not modified in any way. Both baselines are configured on train/validation only,
calibrated with exactly the same rule/budget as every other system (`choose_calibrator`: isotonic vs Platt by
val-A ↔ val-B cross-fit, refit on the whole validation fold), and evaluated once on the test fold
(`scripts/19_new_baselines.py` → `results/eval_new_baselines.json`).

**MS2Query.** Route chosen: re-implement its neighbourhood-consistency mechanism with our retrieval engine
and train its random forest on the train fold only (`scripts/17_ms2query_features.py`). Reasons: (i) the
official Spec2Vec / MS2Deepscore / random-forest models were trained on public libraries that overlap the
MassSpecGym test molecules — using them leaks; (ii) retraining MS2Deepscore/Spec2Vec is a deep-embedding
training job, outside the CPU-only budget and outside what would make the comparison about the
neighbourhood mechanism; (iii) with one shared similarity (entropy) for all systems, any difference is
attributable to how neighbourhood information is used, not to a better embedding. Faithful parts: 5
features (similarity, query precursor m/z, precursor m/z difference, mean similarity between the query and
ALL spectra of the 10 library molecules most Tanimoto-similar to the candidate, mean Tanimoto of those 10),
random forest regression on Tanimoto(candidate, truth), 250 trees, max depth 5, ~500k training pairs
(train fold, both scenarios), candidates = precursor-window molecules + 10 best analogue molecules. The
10 structural neighbours are taken from the scenario library only (the truth is never a neighbour in (b)).
Deviation: Spec2Vec/MS2Deepscore → entropy similarity. Three uses are reported: MS2Query's own top-1
(`B_ms2q_rerank_direct`), its score for our top-1 (`B_ms2q_conf_direct`), and its neighbourhood features
inside our decomposed/label-shift framework (`B_ms2q_dec`, same LightGBM budget) — the last isolates the
neighbourhood mechanism from the absence/prevalence machinery. The official pre-trained MS2Query was NOT
run (it would only be a leaky reference; not needed for the comparison).

**Target-decoy.** Decoy construction fixed a priori (`scripts/18_target_decoy.py`): one decoy per library
spectrum; same precursor m/z, peak count and intensity multiset; fragment m/z values drawn from real
fragments of random library spectra of other molecules with precursor within ±5 Da (conditional,
spectrum-based decoys), processed like targets; decoys mirror the scenario library. Two uses:
(i) per-query score s1 − d1 (`B_tda_margin_direct`, calibrated like everything else); (ii) set-level FDR:
within each simulated batch, concatenated target–decoy competition with (decoys+1)/targets q-values,
accept q ≤ 0.10, compared with the proposed trusted tier (nominal ≥ 90% correct) on the SAME batches
(realized error, coverage, estimated vs realized error). This is the fair comparison because TDA certifies
a set, not individual queries.

### 22:35 Results of the two added baselines (test fold, single run; `results/eval_new_baselines.json`)
- The proposed method was not modified. Every added baseline is significantly worse than the proposed system at π = 0.2 and 0.5
  (paired molecule bootstrap, all ΔAURC and Δlog-loss CIs exclude 0). No baseline exceeded the proposed system.
- MS2Query-style: its own top-1 is an analogue (outside the precursor window) for 35% of queries — expected
  for an analogue-search tool, and it lowers exact top-1 accuracy. Its neighbourhood features inside our
  decomposed framework (`B_ms2q_dec`) beat logistic(s1, gap) but not the proposed system.
- Target–decoy at the set level: at nominal q ≤ 0.10 the realized error of the accepted set was far above
  10% at every prevalence (e.g. π = 0.2: estimated FDR ≈ 0.10, realized ≈ 0.32), because decoys do not
  reproduce the high scores of wrong-but-real library compounds or of absent compounds with close
  analogues. The proposed trusted tier had realized error ≈ 1–2% at the same prevalences.

### 22:40 Instrument shift (pre-planned; `results/eval_shift_instrument.json`)
- First launch used the all-family `M_full` recipe instead of the validation-selected primary; stopped,
  corrected to the primary configuration (and the same calibration rule, fitted on the instrument's
  validation rows) and re-run. Only the corrected run is reported.
- Train on Orbitrap → test QTOF: calibration roughly holds (π = 0.2: ECE 0.013 vs 0.010 for the
  all-instrument model; trusted tier 93% observed vs 96% predicted).
- Train on QTOF (≈ 28k training rows) → test Orbitrap: calibration fails (π = 0.2: predicted 0.12 vs observed 0.20;
  trusted tier 83% vs 96%).

### 22:45 Tautomer audit and external set rebuild (`results/tautomer_audit.json`)
- Reading the external-set failure: many wrong-but-perfect-score annotations were the same compound stored
  as different tautomers (e.g. tetracyclines, amide/iminol and keto/enol forms), i.e. the InChIKey-block
  de-duplication across libraries missed them (§5.11). RDKit canonical-tautomer keys: 25 external
  molecules are tautomers of MassSpecGym molecules, 4 of other external molecules. Within MassSpecGym,
  28 tautomer groups (56 molecules) exist, none across folds (no fold contamination).
- External set rebuilt as v2 (+ canonical-tautomer de-duplication) and v3 (+ removal of molecules with an
  ≥ 0.99-similar spectrum of a different MassSpecGym molecule). Frozen models only; v1 kept and reported.

### 22:50 COSMIC package delivered to the project owner's Mac (`cosmic/`)
- SIRIUS 6.5.4 with an academic licence is installed on the owner's Mac; COSMIC is run there, locally.
- 300 test molecules stratified by the published CSI:FingerID positive-mode training structures
  (`csi.bright-giant.com/v3.0/api/fingerid/trainingstructures?predictor=1`; 28,000 structures; 24% of
  MassSpecGym test molecules are in it): 150 in (leaky reference), 150 not in (primary). Plus 200 validation
  molecules not in the list, used only to calibrate COSMIC scores with the shared validation rule (SIRIUS
  documents that COSMIC scores are not probabilities).
- Structure database: PubChem. Main result: exact mode; supplement: approximate mode against
  "correct or MCES ≤ 2" (myopic_mces). Comparison = each system's confidence on its own annotations.
- Noticed while listing the delivery folder: `data/lab/` now contains laboratory raw files. They were not
  opened, read, copied or uploaded; only file names were listed. They stay on the owner's machine (§6).

### 23:20 Owner's instruction on `data/lab/`
- The project owner instructed that the files currently in `data/lab/` must not be used — neither in
  experiments nor in writing the paper. They have not been opened or used at any point; no script in this
  repository reads that directory, and the reports contain nothing derived from it.

### 23:45 COSMIC baseline imported (`results/eval_cosmic_subset.json`)
- SIRIUS 6.5.4 (academic licence) ran on the owner's Mac in ~8 min: `formulas -p <orbitrap|qtof>` →
  `fingerprints` → `classes` → `structures --database pubchem` → `write-summaries`. Only the four
  `structure_identifications.tsv` summaries (public-data results) were copied into the analysis environment.
- 294/300 test spectra received a structure; 1 had a `-Infinity` confidence (treated as missing, like the
  6 unannotated ones: lowest confidence, counted wrong). First import crashed on the `-Infinity` values;
  fixed by treating non-finite scores as missing.
- COSMIC scores calibrated on the 200 validation molecules with the shared rule (Platt chosen for both modes).
- Primary stratum (not in the published CSI:FingerID training structures): CSI:FingerID exact top-1
  accuracy 8.7%; COSMIC AUROC 0.882 vs proposed 0.995 (difference +0.113, CI [0.035, 0.209]). Leaky stratum:
  top-1 accuracy 67.3% — the leakage effect is large. The comparison concerns confidence quality on each
  system's own annotations (PubChem search vs spectral library containing the compound), not annotation power.

## 2026-09-27

### 00:05 Owner's review of the COSMIC import; plan (nothing about the method changes)
- Requested after the reproduction check: (1) report positive/negative counts per stratum and system, mark the
  proposed system's (a)-only AUROC / e-AURC / ECE as unstable (only a handful of wrong top-1 annotations),
  drop the AUROC difference as a main conclusion, and evaluate the proposed system on the same 300 queries in
  (a) and (b) mixed at pre-declared in-library fractions; (2) a head-to-head on the SAME candidate set using
  SIRIUS custom structure databases; (3) a completeness check of the report.
- **Pre-declared rules for both COSMIC comparisons** (written into `scripts/cosmic/cosmic_common.py` before the
  custom-database results existed): mixtures at π = 0.2 and 0.5 (the headline prevalences of the whole
  report); (a) rows total weight π, (b) rows 1 − π within each stratum; a metric is "unstable" when its
  minority class has < 20 rows (the minimum already used for group calibration); no annotation = confidence 0
  and not correct, for both systems; query bootstrap with both scenario rows of a query kept together.
  For COSMIC: validation calibration on (a)+(b) at the proposed system's calibration prevalence, then the same
  Saerens EM prior correction that the proposed system uses.
- (b) for the COSMIC comparisons removes all spectra of all 300 subset molecules at once ("bset" in
  `features.py`), so that our library and the SIRIUS (b) database lack exactly the same molecules. (a) rows are
  bit-identical to the stored scenario-(a) rows (checked).

### 00:10 Custom structure databases for the head-to-head (`scripts/cosmic/make_customdb.py`)
- `msg_test_a`: 6,721 test-phase library molecules within 0.05 Da of some test query's neutral mass (CSI:FingerID
  retrieves only formula-matching structures, so the mass filter changes no retrieval); `msg_test_b`: minus
  the 300 query molecules (6,421). Validation: 4,160 / 3,960.
- One validation query's recorded precursor m/z is 82 ppm off its structure (MassSpecGym metadata); its
  molecule is added to `msg_val_a` explicitly (first build asserted and stopped on it).
- SIRIUS 6.5.4 CLI read from the source of the stable branch: `custom-db create/import/show`, and
  `structures --database <name> --exp OFF`. Expansive search defaults to APPROXIMATE in 6.5 and would fall back
  to PubChem for low-confidence compounds, returning structures outside our library; it is switched off, and
  the import checks every returned structure against the searched database.
- Formulas, fingerprints and classes are computed once per split/instrument; the project is copied and
  searched once per database, so (a) and (b) share identical fingerprints.

### 00:30 Post-hoc analyses added for report completeness (test fold; no feedback into the method)
- `scripts/23_lead_attribution.py`: where the lead over logistic(s1, gap) comes from, by feature group.
  Eight extra ablation systems (drop-one from the primary; two intermediate steps of the path
  sim → +sim_extra → +analog → +meta → +crowding), trained and calibrated exactly like every other system.
- `scripts/24_baselines_em.py`: every direct baseline (incl. MS2Query-style and TDA margin) additionally with
  the same Saerens EM prior correction applied to its P(correct). `scripts/19_new_baselines.py` now also
  writes the rows it evaluates (`rows_new_baselines.parquet`) so that 24 needs no refit.

### 00:50 Same-candidate-set COSMIC run imported (`results/eval_cosmic_customdb.json`) — main COSMIC comparison
- SIRIUS 6.5.4 on the owner's Mac, 00:15–00:49 (`cosmic/customdb/run_console.txt`). The four databases hold
  exactly the expected numbers of structures (6,721 / 6,421 / 4,160 / 3,960). Only the eight
  `structure_identifications.tsv` summaries, the database info and the console log were copied back.
- Checks: 0 returned structures outside the searched database (every hit's InChIKey 2-D block is a database key;
  every `links` entry names the searched database); 0 (b) rows correct. Test queries with a structure:
  (a) 253/300, (b) 146/300 (in (b), fewer formulas have any candidate).
- Primary stratum (150 molecules not in CSI:FingerID's published training structures): CSI:FingerID top-1
  accuracy 62% on our library's molecules (8.7% against PubChem). (a) alone: COSMIC 93 correct / 57 wrong
  (e-AURC 0.037); proposed 147 / 3 → unstable by the pre-declared rule, not compared.
  (a)+(b) at π = 0.2 (EM for both): e-AURC 0.0015 vs 0.0577, paired Δ −0.056 [−0.080, −0.033]; ECE 0.015 vs
  0.063, Δ −0.038 [−0.078, −0.009]. π = 0.5: Δe-AURC −0.056 [−0.091, −0.031].
- Leaky stratum (reference): COSMIC ranks its (a) annotations almost perfectly (e-AURC 0.0009); mixtures still
  favour the proposed system in exact mode (π = 0.2: Δ −0.012 [−0.025, −0.003]); in approximate mode COSMIC's
  e-AURC is lower at π = 0.2 (Δ +0.003 [−0.008, +0.018], n.s.). Reported as observed.
- COSMIC calibrator (validation, shared rule): isotonic (exact), Platt (approximate).
- The PubChem run is re-presented as supplementary background with counts and instability flags; the AUROC
  difference from 23:45 is no longer reported as a conclusion (COSMIC had only 13 correct and the proposed
  system only 3 wrong annotations in that comparison, both below the pre-declared minimum of 20).

### 01:45 Owner's follow-up questions on the same-candidate-set comparison (answered; `results/cosmic_customdb_audit.json`)
- **Why the validation database pair differs by only 200.** (b) = (a) minus exactly the subset's query molecules
  of that split: 300 test molecules, 200 validation molecules (one spectrum per molecule). Verified: test (a) − (b)
  = the 300 query molecules, validation (a) − (b) = the 200 query molecules, (b) − (a) empty in both.
- **Was the removal tautomer-insensitive?** No. Molecules were removed by our identity key (InChIKey first block
  after RDKit Cleanup → LargestFragment → Uncharge), for SIRIUS's (b) databases and for our (b) library alike.
  Re-checked with the tautomer-insensitive key of `scripts/20_tautomer_keys.py`:
  - test (b) database: 0 molecules are tautomers of a test query molecule; no test top-1 of either system, in
    (a) or (b), is a non-identical tautomer of the truth;
  - validation (b) database: 2 molecules are tautomers of validation query molecules (queries
    MassSpecGymID0193145, MassSpecGymID0089707); for 0089707 COSMIC's (b) top-1 is that tautomer and was labelled
    wrong in COSMIC's calibration data. Main analysis unchanged; a sensitivity analysis removes both queries from
    COSMIC's calibration (`sensitivity_val_tautomer` in `results/eval_cosmic_customdb.json`).
- **When were the main metric and π of the head-to-head fixed?** Honest record:
  - π = 0.2 and 0.5 and the metrics (e-AURC as the ranking metric, ECE / log loss for calibration, trusted tier)
    are the report's headline choices, fixed in `config.yaml` / the main evaluation on 2026-09-26, long before any
    COSMIC result.
  - The instability rule (minority class < 20 rows) and the EM correction for COSMIC were written into
    `scripts/cosmic/cosmic_common.py` at ~00:10–00:30, **after** the PubChem COSMIC results had been seen (they had
    13 correct COSMIC and 3 wrong proposed annotations, which prompted the owner's request) but **before** the
    custom-database run had finished (00:49) or been imported (00:50). The threshold 20 is the existing
    minimum-molecules rule for group calibration.
  - Seen before the metrics were computed: the SIRIUS console log, file sizes and the header and first two rows of
    one summary file (00:40, to check the `links` format). This led to one change before the import: a hit's
    identity is taken from SIRIUS's InChIKey 2-D block (checked to be a database key) instead of parsing `links`.
  - Decided after seeing the head-to-head results: showing the "COSMIC, val-calibrated, no prior correction" row
    in the report table (presentation only), the a_x version (owner's request) and the tautomer sensitivity above.
- **Wording.** The head-to-head is a pipeline-level result (library search + confidence model vs CSI:FingerID +
  COSMIC) and cannot be attributed to the confidence model alone, because the engines differ. Report wording
  changed accordingly.
- **a_x version** (owner's request): the same 300 queries and SIRIUS results; on our side the library keeps the
  truth only through other-instrument spectra. Only 66 of the 300 queries are a_x-eligible (5 primary, 61 leaky
  stratum), so the primary stratum is too small; reported pooled and per stratum with counts.

### 02:00 a_x version of the head-to-head and tautomer sensitivity (`results/eval_cosmic_customdb.json`)
- Re-import with the two additions; all earlier numbers in the file are unchanged (checked, 1,486 values).
- a_x: 66 of 300 queries eligible (61 leaky stratum, 5 primary). Pooled, π = 0.2: e-AURC proposed-a_x 0.0073,
  proposed-default 0.0023, COSMIC 0.0251; Δ(a_x − COSMIC) −0.017 [−0.043, +0.001] (not significant),
  Δ(default − COSMIC) −0.022 [−0.049, −0.004]; ΔECE(a_x − COSMIC) −0.073 [−0.155, −0.001]. The pipeline-level
  ranking advantage shrinks under a_x and is not significant in the pooled set; primary stratum (5) not interpreted.
- Tautomer sensitivity (2 validation queries removed from COSMIC's calibration): primary stratum π = 0.2
  Δe-AURC −0.056 [−0.081, −0.033], ΔECE −0.029 [−0.073, −0.006]; conclusion unchanged.

### 02:15 Post-hoc results: EM for every baseline (`results/eval_baselines_em.json`) and lead by feature group (`results/eval_lead_attribution.json`)
- Re-running `scripts/19_new_baselines.py` (to persist its rows) reproduced `eval_new_baselines.json` up to
  floating-point noise (1,121 of 4,881 values differ in the 7th significant digit; multi-threaded random-forest
  prediction sums trees in a non-deterministic order). No conclusion depends on it.
- **EM on every direct baseline.** Log loss and ECE improve for all 7 direct baselines at π = 0.2 (ranking is
  unchanged by construction). Score baselines stay far behind (best: MS2Query-style score 0.138 vs 0.071), and at
  π = 0.1 their calibrated trusted tier nearly empties (logistic(s1, gap): ~1 query per simulated batch vs 71 for the proposed system).
  **Finding (negative for the decomposition):** the direct full-feature model + EM matches the proposed system
  (π = 0.2: e-AURC 0.0043 vs 0.0041, log loss 0.0701 vs 0.0706, ECE 0.0029 vs 0.0043; 100% of sufficient
  classes and crowding groups consistent; wet-lab simulation equally consistent). Prevalence robustness comes from
  the EM prior correction, not from the P(in)·P(correct|in) decomposition. Reported in the summary and §7; the
  method is not changed.
- **Lead by feature group** (logistic(s1, gap) decomposed → proposed, π = 0.2, AURC lead 0.0273): non-linear
  score model 27%, top-1 library support (sim_extra) 52%, analogue hits 2% (n.s.), metadata 19%, crowding 1%
  (n.s.); log-loss lead 0.0767: 3% / 55% / 0% / 38% / 4% (crowding step significant). Drop-one agrees:
  removing sim_extra or meta costs most; removing crowd_struct or analog costs nothing measurable.
- Operational: at 02:03 the reproduction run was paused (SIGSTOP, during its external-set search) so that the
  lead-attribution training could run without CPU contention (LightGBM slowed ~15× when sharing 2 cores); it
  was resumed at 02:14. This does not affect any result of either run.

### 02:25 Reproduction check, first findings (run from commit d1745d0 in a fresh clone)
- Every result file of the re-run so far is identical to the committed one in every number, except:
  - `dev_feature_variants.json`: the committed file was produced at commit 7188f13, before `fam_lib_size` was
    moved out of the `crowd_family` group (2a5d27a). The diagnostic script built "all family features" from
    that group, so on re-run the variants `base+family_all` and `full` silently lost `fam_lib_size`
    (P(in) val-A log loss 0.081 instead of 0.153). Fixed in the script ("all family features" now explicitly
    includes `fam_lib_size`, the original meaning) and re-run in the main repository (02:50); the report's
    §6.8 table reads from it. With the column now appended last, LightGBM's column subsampling draws different
    subsets, so the two affected values move slightly (P(in) val-A log loss `base+family_all` 0.153 → 0.144,
    `full` 0.132 → 0.125); every other variant is identical and the conclusion (the count feature is a
    shortcut) is unchanged.
  - `prepare_audit.json`: `frac_within_tol` was computed under the earlier 10-ppm setting (0.933) and is 0.986
    at the final 20 ppm; the value is not used in the report. `search_summary.json`: the committed file had
    extra `ext` keys from an earlier run; all shared numbers are identical.
  - Result files produced by the frozen commit are replaced by the re-run's copies (clean provenance: commit
    d1745d0, not dirty). Files changed deliberately after that commit keep their new versions.
- **Traceability gap found:** the 19:58 precursor-tolerance numbers (99.2% at 20 ppm) came from an ad hoc
  computation whose code was not kept; no reconstruction reproduced them. The saved re-computation
  (`scripts/dev/diag_precursor_tolerance.py`) gives 98.8% at 20 ppm over all same-molecule/adduct pairs (the
  stated rule would then have chosen 30 ppm) and 99.9% for each spectrum's nearest same-molecule spectrum. The
  engine tolerance is not changed after the fact; the report states the gap.

### 03:25 Final re-run of the new analyses from the committed code (commit aebe5f4)
- All result files written today by then-uncommitted code were regenerated from commit aebe5f4 (clean
  provenance): `eval_baselines_em`, `eval_cosmic_subset`, `eval_lead_attribution`, `eval_cosmic_customdb`,
  `cosmic_customdb_audit`, `dev_precursor_tolerance`, `dev_feature_variants`; then figures and checks.
  Every number is identical to the earlier outputs (checked file by file).
- `ingest_cosmic_customdb.py` now computes the tautomer sensitivity itself instead of reading the audit file
  (removes a circular dependency; `make cosmic` runs the import, then the audit). Output unchanged.
- The lead-attribution ablation models were trained at 01:32–02:14 with the same script content and reused from
  `data/derived/models/entropy_lead/`; the final JSON therefore has no per-model training log (a fresh
  `make posthoc` retrains them).
- `results/repro_check.json` was produced by `scripts/dev/compare_runs.py`; the per-file notes in it were
  added by hand from the analysis in the 02:25 entry.
