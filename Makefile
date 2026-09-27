# One command: `make all` (public data download -> every table and figure -> report -> audits).
# CPU only. Approximate time on 2 cores: ~2.5 h (training ~45 min; external-source variants ~30 min).
PY ?= python3
EXT = data/derived/ext

.PHONY: all data search features train dev evaluate shift external newbaselines cosmic figures report check clean-derived

all: data search features train dev evaluate shift external newbaselines cosmic figures report check

data:
	$(PY) scripts/01_download.py
	$(PY) scripts/02_prepare.py
	$(PY) scripts/03_classyfire.py
	$(PY) scripts/04_struct_neighbors.py
	$(PY) scripts/06_families.py

search:
	$(PY) scripts/05_search.py

features:
	$(PY) scripts/07_features.py

train:
	$(PY) scripts/08_train.py entropy cosine
	$(PY) scripts/08b_recalibrate.py entropy cosine

dev:   # validation-only diagnostics referenced in LOG.md
	$(PY) scripts/dev/diag_feature_variants.py
	$(PY) scripts/dev/diag_calibrators.py

evaluate:   # the single final test evaluation of the frozen systems
	$(PY) scripts/09_evaluate.py

shift:
	$(PY) scripts/16_scenario_cross_instrument.py
	$(PY) scripts/21_wetlab_ax.py
	$(PY) scripts/10_shift_instrument.py

external:
	$(PY) scripts/11_external_massbank.py
	$(PY) scripts/03_classyfire.py $(EXT)/molecules.parquet
	MSCONF_DERIVED=$(EXT) $(PY) scripts/04_struct_neighbors.py
	MSCONF_DERIVED=$(EXT) $(PY) scripts/06_families.py
	MSCONF_DERIVED=$(EXT) MSCONF_QUERY_FOLDS=ext $(PY) scripts/05_search.py
	MSCONF_DERIVED=$(EXT) $(PY) scripts/07_features.py
	MSCONF_EXT_VARIANT=v1 $(PY) scripts/12_external_eval.py
	$(PY) scripts/20_tautomer_keys.py
	for V in v2 v3; do \
	  MSCONF_EXT_VARIANT=$$V $(PY) scripts/11_external_massbank.py && \
	  MSCONF_DERIVED=data/derived/ext_$$V $(PY) scripts/04_struct_neighbors.py && \
	  MSCONF_DERIVED=data/derived/ext_$$V $(PY) scripts/06_families.py && \
	  MSCONF_DERIVED=data/derived/ext_$$V MSCONF_QUERY_FOLDS=ext $(PY) scripts/05_search.py && \
	  MSCONF_DERIVED=data/derived/ext_$$V $(PY) scripts/07_features.py && \
	  MSCONF_EXT_VARIANT=$$V $(PY) scripts/12_external_eval.py || exit 1; \
	done

newbaselines:   # added after the main test evaluation (see LOG.md); the proposed method is unchanged
	$(PY) scripts/17_ms2query_features.py
	$(PY) scripts/18_target_decoy.py
	$(PY) scripts/19_new_baselines.py
	$(PY) scripts/22_neighbourhood_comparison.py

cosmic:   # builds the SIRIUS input package; COSMIC itself runs on the project owner's machine (cosmic/README.md)
	$(PY) scripts/cosmic/csi_training_keys.py
	$(PY) scripts/cosmic/make_subset.py
	if ls cosmic/results/summaries_*/structure_identifications.* >/dev/null 2>&1; then $(PY) scripts/cosmic/ingest_cosmic.py cosmic/results; fi

figures:
	$(PY) scripts/13_figures.py

report:
	$(PY) scripts/14_render_report.py

check:
	$(PY) scripts/15_checks.py

clean-derived:
	rm -rf data/derived
