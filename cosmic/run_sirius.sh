#!/usr/bin/env bash
# COSMIC baseline: run SIRIUS 6 (formulas -> fingerprints -> classes -> structures [CSI:FingerID + COSMIC])
# on the PUBLIC MassSpecGym subset in ./input. Runs locally; uses the SIRIUS login already stored on this Mac.
# Never point this script at laboratory data (data/lab/): lab data must not be sent to any web service.
set -uo pipefail
SIRIUS="${SIRIUS:-/Applications/sirius.app/Contents/MacOS/sirius}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IN="$HERE/input"
OUT="$HERE/results"
DB="${COSMIC_DB:-pubchem}"
mkdir -p "$OUT"
case "$IN" in *data/lab*) echo "refusing: laboratory data"; exit 1;; esac
[ -x "$SIRIUS" ] || { echo "SIRIUS not found at $SIRIUS"; exit 1; }
{ echo "== $(date)"; "$SIRIUS" --version; } > "$OUT/environment.txt" 2>&1
# option name for the structure database differs between releases (--database / --databases)
DBOPT=$("$SIRIUS" structures --help 2>&1 | grep -oE -- '--databases?' | head -1)
DBOPT=${DBOPT:---database}
echo "structure DB option: $DBOPT $DB" >> "$OUT/environment.txt"

# Order: test subset first (the evaluation), then the validation subset (used only to calibrate COSMIC scores).
for SPLIT in test val; do
  for INST in orbitrap qtof; do
    F="$IN/${SPLIT}_${INST}.ms"
    [ -f "$F" ] || continue
    PROJ="$OUT/${SPLIT}_${INST}.sirius"
    LOG="$OUT/log_${SPLIT}_${INST}.txt"
    echo "[$(date +%H:%M:%S)] ${SPLIT}/${INST}: $(grep -c '^>compound' "$F") spectra -> $PROJ"
    # compute (resumable: re-running skips compounds that already have results)
    "$SIRIUS" --input "$F" --project "$PROJ" \
      formulas -p "$INST" \
      fingerprints \
      classes \
      structures "$DBOPT" "$DB" > "$LOG" 2>&1
    echo "    exit code $? (details in $(basename "$LOG"))"
    # summaries (command name differs between SIRIUS 6 releases; try the documented ones in turn)
    SUM="$OUT/summaries_${SPLIT}_${INST}"
    for CMD in write-summaries summaries summary; do
      if "$SIRIUS" --project "$PROJ" $CMD --output "$SUM" >> "$LOG" 2>&1; then echo "    summaries: $CMD -> $(basename "$SUM")"; break; fi
    done
  done
done
echo "[$(date +%H:%M:%S)] done. Hand back: $OUT/summaries_* (the .sirius projects can stay local)."
