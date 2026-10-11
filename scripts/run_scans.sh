#!/bin/sh
# Production scans for the demonstration figures (~20 min on 16 cores).
set -e
BIN=./target/release/examples/relic_scan
mkdir -p runs/traces
SS_MASSES=45,47.5,50,52,54,55,56,57,57.5,58,58.5,59,59.5,60,60.5,61,61.5,62,62.25,62.4,62.5,62.55,62.6,62.7,62.85,63,63.5,64
MED_MASSES=0.25,0.3,0.35,0.4,0.42,0.44,0.45,0.46,0.47,0.475,0.48,0.485,0.49,0.4925,0.495,0.4975,0.499,0.4995,0.5005,0.501,0.5025,0.505,0.51,0.52,0.53,0.55,0.575,0.6
$BIN ss 0 $SS_MASSES runs/scan_ss.tsv --find-fbe --trace runs/traces 2> runs/scan_ss.log
for med in pseudoscalar vector axial scalar; do
  $BIN $med 1.0 $MED_MASSES runs/scan_$med.tsv --find-fbe --trace runs/traces 2> runs/scan_$med.log
done
echo done > runs/DONE
