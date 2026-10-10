#!/usr/bin/env bash
# Robustness steps that come after run_all.sh's main results (they were added while run_all.sh was running):
#   - wide prior for the ABC (native dichroism from 0.2 1/ns), because the primary posterior piles up at the lower prior edge
#   - re-conditioning on the literature under a changed process assumption (isotropic / more anisotropic oxidation, thermal drift)
#   - sensitivity table with all of these, and the best rotation under every alternative way of conditioning
#   - the headline batch score on the further accepted worlds, and the optimiser benchmark at the level of wafers
# Safe to re-run: finished world files are skipped, sensitivity scenarios are cached.
#   bash scripts/run_extra.sh [--after-pid PID]
set -u
cd "$(dirname "$0")/.."
AFTER=""
while [ $# -gt 0 ]; do case "$1" in --after-pid) AFTER=$2; shift 2;; *) shift;; esac; done
[ -n "$AFTER" ] && while kill -0 "$AFTER" 2>/dev/null; do sleep 10; done
run() {
  name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name" | tee -a results/pipeline.log
  if "$@" > "results/$name.log" 2>&1; then echo "[$(date +%H:%M:%S)] done  $name" | tee -a results/pipeline.log; else echo "[$(date +%H:%M:%S)] FAILED $name" | tee -a results/pipeline.log; fi
}
[ -f results/anchored_worlds_wide.json ]   || run anchored_wide   python scripts/anchored_worlds.py --draws 4000 --seed 5 --prior wide --marginal 0 --out results/anchored_worlds_wide.json
[ -f results/anchored_worlds_ox10.json ]   || run anchored_ox10   python scripts/anchored_worlds.py --draws 4000 --seed 6 --ox-ratio 1.0 --marginal 0 --out results/anchored_worlds_ox10.json
[ -f results/anchored_worlds_ox13.json ]   || run anchored_ox13   python scripts/anchored_worlds.py --draws 4000 --seed 7 --ox-ratio 1.3 --marginal 0 --out results/anchored_worlds_ox13.json
[ -f results/anchored_worlds_chim02.json ] || run anchored_chim02 python scripts/anchored_worlds.py --draws 4000 --seed 8 --chi-th -0.2 --marginal 0 --out results/anchored_worlds_chim02.json
[ -f results/anchored_worlds_chip02.json ] || run anchored_chip02 python scripts/anchored_worlds.py --draws 4000 --seed 9 --chi-th 0.2 --marginal 0 --out results/anchored_worlds_chip02.json
run sensitivity_extra     python scripts/sensitivity.py
run rotation_by_worldset  python scripts/rotation_by_worldset.py
run other_worlds_check    python scripts/other_worlds_check.py      # headline batch score on the accepted worlds that were not used for selection or test
run benchmark_clusters    python scripts/benchmark_clusters.py      # optimiser benchmark at the level of wafers; P3 selection-vs-unseen gap
run fresh_population_anchors python scripts/fresh_population_anchor_check.py   # do the accepted worlds still pass the anchors on fresh device populations?
run mask_family_a_rounded python scripts/mask_family_a_rounded.py    # mask predictions if family A has rounded arm tips instead of sharp corners
echo "[$(date +%H:%M:%S)] extra steps finished" | tee -a results/pipeline.log
