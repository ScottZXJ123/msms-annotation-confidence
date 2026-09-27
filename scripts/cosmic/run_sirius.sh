#!/usr/bin/env bash
# Run SIRIUS 6 (formula -> CSI:FingerID -> COSMIC) on the PUBLIC subset only.
# Requires a SIRIUS account with a license (academic licenses are granted by institutional e-mail):
#   sirius login --user-env SIRIUS_USER --password-env SIRIUS_PASS
# Never run this on files from data/lab/ (lab data must not be sent to any web service).
set -euo pipefail
IN=${1:-data/derived/cosmic/subset.mgf}
OUT=${2:-data/derived/cosmic/sirius_project.sirius}
case "$IN" in *data/lab*) echo "refusing: lab data"; exit 1;; esac
sirius --input "$IN" --project "$OUT" --cores 2 \
  formulas --profile qtof \
  fingerprints \
  structures \
  write-summaries --output data/derived/cosmic/summaries
