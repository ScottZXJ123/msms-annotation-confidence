#!/usr/bin/env bash
# Head-to-head COSMIC comparison on the SAME candidate universe as our spectral library.
# CSI:FingerID/COSMIC searches custom structure databases built from the molecules of our reference library:
#   msg_test_a : test-phase library molecules (contains every test query molecule)      -> scenario (a)
#   msg_test_b : msg_test_a without the 300 test query molecules                        -> scenario (b)
#   msg_val_a / msg_val_b : the same for the validation subset (calibration of COSMIC scores only)
# Runs locally with the SIRIUS login already stored on this Mac (no password or token is asked for).
# Only public MassSpecGym spectra and structures are used. Never point this script at laboratory data.
set -uo pipefail
SIRIUS="${SIRIUS:-/Applications/sirius.app/Contents/MacOS/sirius}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IN="$HERE/../input"
OUT="$HERE/results"
DBDIR="$HERE/dbs"
mkdir -p "$OUT" "$DBDIR"
case "$HERE$IN" in *data/lab*) echo "refusing: laboratory data"; exit 1;; esac
[ -x "$SIRIUS" ] || { echo "SIRIUS not found at $SIRIUS"; exit 1; }
{ echo "== $(date)"; "$SIRIUS" --version 2>/dev/null | grep -E "^(SIRIUS|CSI)"; } > "$OUT/environment.txt"
ts() { date +%H:%M:%S; }

# ---------------------------------------------------------------- 1. custom structure databases
n_structures() { "$SIRIUS" custom-db show --db "$1" 2>/dev/null | grep -oE "Number of Structures: [0-9]+" | grep -oE "[0-9]+" | tail -1; }
for DB in test_a test_b val_a val_b; do
  NAME="msg_${DB}"
  TSV="$HERE/db_${DB}.tsv"
  EXPECTED=$(wc -l < "$TSV" | tr -d ' ')
  MIN=$((EXPECTED * 95 / 100))
  N=$(n_structures "$NAME"); N=${N:-0}
  if [ "$N" -ge "$MIN" ]; then echo "[$(ts)] database $NAME already built ($N structures)"; continue; fi
  echo "[$(ts)] building database $NAME from $(basename "$TSV") ($EXPECTED structures; takes a few minutes)"
  LOG="$OUT/log_db_${NAME}.txt"
  "$SIRIUS" custom-db remove --db "$NAME" --delete > "$LOG" 2>&1   # clean slate if an earlier attempt exists
  rm -f "$DBDIR/${NAME}.siriusdb"
  "$SIRIUS" custom-db create --location "$DBDIR/${NAME}.siriusdb" --name "$NAME" >> "$LOG" 2>&1
  "$SIRIUS" custom-db import --db "$NAME" "$TSV" >> "$LOG" 2>&1
  "$SIRIUS" custom-db show --db "$NAME" > "$OUT/dbinfo_${NAME}.txt" 2>&1
  N=$(grep -oE "Number of Structures: [0-9]+" "$OUT/dbinfo_${NAME}.txt" | grep -oE "[0-9]+" | tail -1); N=${N:-0}
  echo "    $NAME: $N structures (expected $EXPECTED)"
  if [ "$N" -lt "$MIN" ]; then
    echo "STOP: database $NAME has too few structures. Please send back cosmic/customdb/results/log_db_${NAME}.txt"; exit 1
  fi
done

# ---------------------------------------------------------------- 2. formulas + fingerprints once, then structure search per database
# Expansive search is switched OFF (--exp OFF): otherwise SIRIUS falls back to PubChem for low-confidence
# compounds and would return structures outside our library.
for SPLIT in test val; do
  for INST in orbitrap qtof; do
    F="$IN/${SPLIT}_${INST}.ms"
    [ -f "$F" ] || continue
    BASE="$OUT/base_${SPLIT}_${INST}.sirius"
    if [ ! -f "$OUT/.done_base_${SPLIT}_${INST}" ]; then
      rm -f "$BASE"
      echo "[$(ts)] ${SPLIT}/${INST}: formulas + fingerprints + classes for $(grep -c '^>compound' "$F") spectra"
      "$SIRIUS" --input "$F" --project "$BASE" formulas -p "$INST" fingerprints classes > "$OUT/log_base_${SPLIT}_${INST}.txt" 2>&1 \
        && touch "$OUT/.done_base_${SPLIT}_${INST}"
    fi
    for SC in a b; do
      P="$OUT/${SPLIT}_${INST}_${SC}.sirius"
      LOG="$OUT/log_${SPLIT}_${INST}_${SC}.txt"
      if [ ! -f "$OUT/.done_${SPLIT}_${INST}_${SC}" ]; then
        rm -f "$P"; cp "$BASE" "$P"
        echo "[$(ts)] ${SPLIT}/${INST}/(${SC}): structure search in msg_${SPLIT}_${SC}"
        "$SIRIUS" --project "$P" structures --database "msg_${SPLIT}_${SC}" --exp OFF > "$LOG" 2>&1 \
          && "$SIRIUS" --project "$P" write-summaries --output "$OUT/summaries_${SPLIT}_${INST}_${SC}" >> "$LOG" 2>&1 \
          && touch "$OUT/.done_${SPLIT}_${INST}_${SC}"
        [ -f "$OUT/.done_${SPLIT}_${INST}_${SC}" ] && echo "    ok" || echo "    FAILED (see $(basename "$LOG"))"
      fi
    done
  done
done
echo "[$(ts)] done. Results for import: cosmic/customdb/results/summaries_* and dbinfo_* (the .sirius projects can stay local)."
