#!/bin/sh
# Check the TLA+ model with the statechart tool and TLC, outside the repository.
#
# Requires:
#   STATECHART  path to the statechart skill's scripts/statechart launcher
#   TLA2TOOLS   path to tla2tools.jar
#   JAVA        path to a Java 11+ binary
# Optional:
#   SCRATCH     scratch directory (default: a new directory under /tmp)
#
# Writes report.md, tlc.out and chart.mmd for each configuration to
# checked/<config>/, and TLC state-graph dumps to $SCRATCH/dump/ for
# verify/parity.mjs.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
: "${STATECHART:?set STATECHART to the statechart launcher}"
: "${TLA2TOOLS:?set TLA2TOOLS to tla2tools.jar}"
: "${JAVA:?set JAVA to a java binary}"
export TLA2TOOLS JAVA
scratch=${SCRATCH:-$(mktemp -d /tmp/learning-lab-check.XXXXXX)}
mkdir -p "$scratch/lab/model" "$scratch/out" "$scratch/dump"
cp "$here/lab-process.js" "$here/package.json" "$scratch/lab/"
cp "$here/model/LearningLab.tla" "$here/model/view.ts" "$here/model/"*.cfg "$scratch/lab/model/"
for cfg in fixed-scenarios free-outcomes; do
  rm -rf "$scratch/out/$cfg"
  (cd "$scratch/lab/model" && "$STATECHART" LearningLab.tla --config "$cfg.cfg" --view view.ts --out "$scratch/out/$cfg")
  mkdir -p "$here/checked/$cfg"
  rm -f "$here/checked/$cfg/"*.out "$here/checked/$cfg/report.md" "$here/checked/$cfg/chart.mmd"
  cp "$scratch/out/$cfg/report.md" "$scratch/out/$cfg/"*.out "$here/checked/$cfg/"
  [ -f "$scratch/out/$cfg/chart.mmd" ] && cp "$scratch/out/$cfg/chart.mmd" "$here/checked/$cfg/"
  # Full state graph with action names, for the JavaScript parity check.
  (cd "$scratch/lab/model" && "$JAVA" -XX:+UseParallelGC -cp "$TLA2TOOLS" tlc2.TLC -workers 1 -deadlock \
    -config "$cfg.cfg" -dump dot,actionlabels "$scratch/dump/$cfg" -metadir "$scratch/meta-$cfg" LearningLab \
    > "$scratch/dump/$cfg.tlc.out" 2>&1)
done
echo "scratch: $scratch"
