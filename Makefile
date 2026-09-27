# One command: `make all` (public data download -> every table and figure).
# Approximate CPU time on 2 cores: search 7 min, features 2 min, training ~45 min, evaluation ~20 min.
PY ?= python3
EXT = data/derived/ext

.PHONY: all data search features train evaluate shift external figures report check clean-derived

all: data search features train evaluate shift external figures report check

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

evaluate:
	$(PY) scripts/09_evaluate.py

shift:
	$(PY) scripts/10_shift_instrument.py

external:
	$(PY) scripts/11_external_massbank.py
	$(PY) scripts/03_classyfire.py $(EXT)/molecules.parquet
	MSCONF_DERIVED=$(EXT) $(PY) scripts/04_struct_neighbors.py
	MSCONF_DERIVED=$(EXT) $(PY) scripts/06_families.py
	MSCONF_DERIVED=$(EXT) MSCONF_QUERY_FOLDS=ext $(PY) scripts/05_search.py
	MSCONF_DERIVED=$(EXT) $(PY) scripts/07_features.py
	$(PY) scripts/12_external_eval.py

figures:
	$(PY) scripts/13_figures.py

report:
	$(PY) scripts/14_render_report.py

check:
	$(PY) scripts/15_checks.py

clean-derived:
	rm -rf data/derived
