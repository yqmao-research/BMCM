#!/bin/bash
set -e
cd "$(dirname "$0")/.."
for c in caltech3v caltech4v caltech5v caltech6v youtube coco voc2007; do
  if [ -f data/$c.mat ]; then
    python -m bmcm.train --config configs/$c.json --out runs/main/$c
  else
    echo "skip $c: data/$c.mat not found"
  fi
done
