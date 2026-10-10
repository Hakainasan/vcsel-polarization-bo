#!/usr/bin/env bash
# Main pipeline (roughly 1.5 hours on 2 cores; scripts/run_extra.sh adds about 30 minutes of robustness checks). Every step writes its own log in results/.
# Order: the steps that the design conclusions rest on come first, the more expensive robustness checks last.
# The primary literature-consistent set (results/anchored_worlds.json) is made before this script (about 8 minutes):
#   python scripts/anchored_worlds.py --draws 8000 --seed 1 --thresholds primary --marginal 300
# Options:  --from STEP        resume at STEP (skip the steps before it)
#           --after-pid PID    wait until process PID has finished before starting
set -u
cd "$(dirname "$0")/.."
FROM=""; AFTER=""
while [ $# -gt 0 ]; do
  case "$1" in --from) FROM=$2; shift 2;; --after-pid) AFTER=$2; shift 2;; *) shift;; esac
done
[ -n "$AFTER" ] && while kill -0 "$AFTER" 2>/dev/null; do sleep 15; done
STARTED=1; [ -n "$FROM" ] && STARTED=0
run() {
  name=$1; shift
  if [ "$STARTED" = 0 ] && [ "$name" = "$FROM" ]; then STARTED=1; fi
  [ "$STARTED" = 1 ] || return 0
  echo "[$(date +%H:%M:%S)] start $name"
  if "$@" > "results/$name.log" 2>&1; then echo "[$(date +%H:%M:%S)] done  $name"; else echo "[$(date +%H:%M:%S)] FAILED $name"; fi
}
run literature_designs python scripts/literature_designs.py
run design_batch       python scripts/design_batch.py --q 16 --sobol 128 --train 32 --test 32
run robustness_extras  python scripts/robustness_extras.py --jobs 2
run mask_predictions   python scripts/mask_predictions.py
run benchmark_p2       python scripts/run_benchmark.py p2 --seeds 8 --budget 48 --jobs 2
run benchmark_p3       python scripts/run_benchmark.py p3 --seeds 8 --budget 48 --jobs 2
run anchored_loose     python scripts/anchored_worlds.py --draws 3000 --seed 2 --thresholds loose --marginal 0 --out results/anchored_worlds_loose.json
run anchored_strict    python scripts/anchored_worlds.py --draws 4000 --seed 3 --thresholds strict --marginal 0 --out results/anchored_worlds_strict.json
run anchored_chen      python scripts/anchored_worlds.py --draws 4000 --seed 4 --sfm chen --marginal 0 --out results/anchored_worlds_chen.json
run sensitivity        python scripts/sensitivity.py
run benchmark_p1       python scripts/run_benchmark.py p1 --seeds 4 --budget 48 --jobs 2
run sfm_validation     python scripts/validate_sfm_sweep.py
echo "[$(date +%H:%M:%S)] pipeline finished (next: bash scripts/run_extra.sh, then make_figures / make_tables / make_readme)"
