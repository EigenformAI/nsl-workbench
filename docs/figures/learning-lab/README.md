# Learning-lab process model

This folder holds the checked process behind the learning-lab figure. In the
lab, two episode agents work at the same time on one generation floor. Their
commits feed two separate destinations: an optional shared knowledge graph and
the floor's training-row tray. After a publication barrier, training reloads
the fixed base, and a validated handoff opens the next floor. This folder
contains no scene or page. A figure consumes `lab-process.js` through the
contract in [`CONTRACT.md`](CONTRACT.md).

## Files

| path | what it is |
|---|---|
| `lab-process.js` | Pure ES module: scenarios, snapshots, enabled actions, transitions, rules, demos. |
| `CONTRACT.md` | Snapshot schema, API, action ids, scenarios, bounds, chart highlighting. |
| `model/LearningLab.tla` | TLA+ model, action for action the same as the module. |
| `model/fixed-scenarios.cfg` | The four playback scenarios together. This is the drawn configuration. |
| `model/free-outcomes.cfg` | Every verification, row-count, admission, and training outcome varies. TLC only. |
| `model/view.ts` | Chart labels; imports the module's label tables. |
| `checked/fixed-scenarios/` | `report.md`, complete `tlc.out`, canonical `chart.mmd`, and the Pele `chart.svg`. |
| `checked/free-outcomes/` | `report.md` and complete `tlc.out` (no chart: above the 500-state budget). |
| `checked/demos.txt` | Every model-derived demo as numbered reader-word steps. |
| `checked/*.out` | Output of the parity, jitter, behaviour-test, and mutant runs below. |
| `scripts/` | `check-model.sh` (TLC), `render-chart.mjs` (Pele), `replay.mjs` (demos). |
| `verify/` | Behaviour tests, mutants, JavaScript–TLC parity, randomised interleavings. |

## The process and its rules

Each episode runs Explore → Hypothesise → Code → Verify → Rewrite in order,
then makes one durable commit. The two agents' episodes interleave freely.
An agent reads the shared facts once, when its episode starts.
A signal-passing rewrite adds one or more training rows. If the task has a
shared archive and the archive admits the result, the commit also adds a
shared fact. Archive admission does not decide row eligibility. The floor
publishes when both agents have settled. Training follows on floors 1 and 2
only. It reloads the fixed configured base, not the serving adapter, and uses
the floor's published rows. A candidate that fails format, identity, or
configuration checks never serves. The same is true when the trainer errors.

TLC checks these invariants. Their reader sentences are `RULES` in the module:

`TypeOK`, `RowsOnlyFromEligibleRewrite`, `EligibleRowsKept`,
`FactsOnlyAfterCommit`, `NoDoubleContribution`, `NoGraphWithoutArchive`,
`ContextFromCommittedFacts`, `NoPublishWhileActive`, `NoTrainingBeforePublish`,
`FixedTrainingBase`, `TrainsOnPublishedRows`, `ServeOnlyValidated`,
`NoFinalTraining`, plus TLC's deadlock check.

## Results

From `checked/*/report.md`:

- **fixed-scenarios** (drawn): "TLC reached 477 states and 783 steps. The chart
  draws 20 states and 21 arrows." All 13 invariants and `NoDeadlock` hold.
  None is broken or unchecked.
- **free-outcomes**: "TLC reached 218,636 states". All 13 invariants and
  `NoDeadlock` hold. The tool writes no chart for it because it exceeds the
  500-state chart budget. TLC itself has no such limit and explored everything
  ("0 states left on queue").

Neither configuration uses a VIEW, SYMMETRY, or state constraint. The chart
hides only the four terminal stutter steps (`Finished`). It merges concrete
states by region label: one node such as "Exploring" stands for many snapshots.

From `checked/parity.out`, the JavaScript module and TLC produce identical
graphs: "states: JS 477, TLC 477; steps: JS 779, TLC 779" (fixed) and "states:
JS 218636, TLC 218636; steps: JS 345742, TLC 345742" (free). Every one of the
4 (fixed) and 39,090 (free) terminal stutters is a `Finished` self-loop on a
terminal state.

## What is and is not claimed

- **Bounded scope.** Two agents, three floors, one episode per slot per floor.
  Slot A always runs episode 0, slot B episode 1. The real runner keeps handing
  episodes to free workers. This model shows one episode per slot per floor so
  the whole graph can be drawn.
- **Fixed outcomes.** Playback scenarios fix every verdict, row count,
  admission, and training result in advance. They are illustrations, not an
  exhaustive search. The `free-outcomes` check varies those outcomes within the
  same bounds, with 1 or 2 rows per eligible episode.
- **Termination, not liveness.** On TLC's own graph, parity finds no cycle and
  no dead end outside a terminal phase. So every schedule of this bounded
  model ends after at most 46 steps (fixed) or 47 steps (free). No temporal
  property, fairness, or runtime liveness is checked or claimed.
- **Logical concurrency.** Interleaving is logical, not simultaneous
  execution. `verify/jitter.mjs` gives each agent random, changing speeds. Over
  2,000 runs per scenario, every run satisfies the rules and ends, and the
  per-floor totals are identical. Only the order and what each episode read
  differ: "distinct per-floor totals 1; distinct interleavings/reading patterns
  78" for `materials`.
- **Graceful stops only.** The model covers graceful completion, a trainer
  error, and an invalid candidate. It omits deadlines, cancellation, crashes,
  recovery, circuit breakers, saturation, and endpoint scheduling.
- **No quality claims.** Validation means format, identity, and configuration
  checks. More floors do not imply a better model. No private scores appear in
  snapshots.
- **Graph abstraction.** The shared knowledge graph is a conceptual view of
  admitted archive records and their parent links. A fact's parents are all
  facts the agent read at episode start.
- **Demos are not traces.** `makeDemo()` applies the transition function under
  a fixed schedule. It is a model-derived demo, not a recorded TLC trace.
  Parity shows that every demo step is a TLC step, because every reachable
  JavaScript step is one.

## Source anchors

The behaviour follows these repository files:

- concurrent episode slots: `src/execution/parallel.py` (scheduling, and the
  join of started workers before publication at lines 1039–1125);
- episode stages, eligibility, and admission:
  `tasks/materials_adsorption/task.py` (stages at 775–887; signal-pass
  eligibility versus marginal-pass admission at 1496–1555; rewrite refused
  without eligibility at 1056–1061; finalization and commit at 1693–1798);
- rewrite inference rows: `tasks/materials_adsorption/training.py:87–144`;
- publication order: `src/execution/generation.py:811–863`;
- fixed-base training, validation before serving, no final training:
  `scripts/run_train_loop.py` (830–870, 932–1025).

The materials training configuration in this repository uses one worker, two
generations, and a one-generation row window. This model deliberately shows
two workers and three floors to illustrate supported concurrency and
repetition. Its training window is also one floor.

## Reproduce

```sh
# TLC and the chart. Needs Java 11+, tla2tools.jar, and the statechart tool.
STATECHART=/path/to/statechart/scripts/statechart TLA2TOOLS=/path/to/tla2tools.jar \
  JAVA=/path/to/java SCRATCH=/tmp/learning-lab-check scripts/check-model.sh
npm install && node scripts/render-chart.mjs     # or PELE=/path/to/pele/dist/pele.js

# Checks (Node 22+).
node --max-old-space-size=6000 verify/parity.mjs /tmp/learning-lab-check/dump
node --test verify/lab-process.test.mjs
node verify/mutants.mjs
node verify/jitter.mjs
node scripts/replay.mjs --check
```

`verify/mutants.mjs` copies the module with one rule removed per mutant, for
example "rows only for admitted rewrites". The behaviour tests must fail for
each of the 12 mutants. `checked/mutants.out` records each one as rejected.

## Credits

The model and chart use the statechart skill by Lauren Tan. Charts are
rendered with [Pele](https://pele.run). The figure takes
[Tinycharts](https://april-zhu.com/tinycharts) as its inspiration.
