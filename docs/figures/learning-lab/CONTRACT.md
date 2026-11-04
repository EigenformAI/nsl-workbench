# Learning-lab process contract

`lab-process.js` is the only owner of process state for the learning-lab
figure. A scene or page controller reads snapshots and asks the module for
enabled actions and transitions. It never computes a process change itself,
and it never infers transitions from chart arrows.

The module is a pure ES module with no DOM, Node, timer, or random APIs. Use it
with `import * as Lab from "./lab-process.js"` in Node or in a browser module
script. To inline it in an offline page, put the file text and the controller
code in one `<script type="module">` element. `export` declarations are valid
there.

## Process in reader words

Each generation is a floor of the lab. On a floor, two episode slots (agents A
and B) work at the same time. Each episode moves in order through **Explore →
Hypothesise → Code → Verify → Rewrite** and ends with one durable commit. The
two agents' steps interleave freely.

- When an episode starts, the agent snapshots the shared facts committed so
  far. It does not see facts committed later.
- Verify reveals the signal-check verdict. A failing episode ends with no rows
  and no fact.
- A passing episode rewrites, then commits. The commit adds its successful
  rewrite inference rows (one or more) to this floor's row tray. If the task
  has a shared archive and the archive admits the result, the commit also adds
  one persistent shared fact. Its parent links are the facts the agent read at
  start. An eligible rewrite that the archive does not admit still adds rows.
- The floor publishes only when both slots have run their episode and both
  agents are idle.
- After publication on floors 1 and 2, training reloads the fixed configured
  base on this floor's published rows. The trainer either writes a candidate or
  errors. A candidate passes or fails format, identity, and configuration
  checks. Only a validated candidate opens the next floor and serves there.
- The final floor (floor 3) publishes and stops. No training follows it.

## Bounds and fixed outcomes

- Two agents, three floors, and one episode per slot per floor. Slot A always
  runs episode 0 and slot B runs episode 1. The real runner keeps giving new
  episodes to free workers, so a page must state this bound.
- Outcomes are exogenous and fixed per scenario. They are not chosen by the
  scheduler, and they are not exhaustive. The separate `free-outcomes` check
  varies them; see `README.md`.
- The model has no time, deadlines, cancellation, crashes, recovery, circuit
  breakers, saturation, endpoint scheduling, or model quality. A trainer error
  and a rejected candidate are the only abnormal endings. Both are graceful,
  terminal stops.

## Scenarios

| id | archive | episode outcomes per floor (A, B) | training after floor 1, floor 2 | ends |
|---|---|---|---|---|
| `materials` | yes | F1: 2 rows + fact, 1 row not admitted (duplicate) · F2: signal fails, 1 row + fact · F3: 1 row + fact, 2 rows not admitted (marginal) | valid, valid | `complete`, 3 floors |
| `no-archive` | no | same verdicts and row counts; never a fact | valid, valid | `complete`, 3 floors |
| `trainer-failure` | yes | same as `materials` | trainer error | `trainer-failed`, 1 floor |
| `invalid-artifact` | yes | same as `materials` | valid, invalid candidate | `artifact-rejected`, 2 floors |
| `free-outcomes`, `free-outcomes-no-archive` | yes / no | every verdict, row count (1 or 2), and admission varies | every trainer and validation result varies | checking only, no demo |

`scenario(id)` returns `{ id, title, summary, graph, generations, episodes,
training, demoSchedule }`. Its `episodes` table holds future outcomes, so
do not show that table to readers ahead of the snapshot.

## Snapshot

A snapshot is plain JSON. Every snapshot is complete: Back and Reset only need
to select a stored snapshot. All floor history comes from it, and nothing needs
to accumulate on the scene side. Records are bounded by 2 agents, 3 floors, 2
episodes per floor, 2 rows per episode, and 6 facts.

```js
{
  scenario: "materials",
  generation: 0,              // current floor index; display number is generation + 1
  phase: "collecting",        // see PHASES
  trainingBase: "configured-base",
  agents: {
    A: { stage: "explore", episode: 0, context: ["F0.0"] },  // context: fact ids read at episode start
    B: { stage: "idle", episode: null, context: [] },
  },
  knowledge: [                // persistent across floors; sorted by id
    { id: "F0.0", floor: 0, episode: 0, agent: "A", parents: [] },
  ],
  floors: [                   // one entry per opened floor; length === generation + 1
    {
      floor: 0,
      serving: "configured-base",       // or "adapter-from-floor-N"
      published: false,
      episodes: [                       // issued episodes, sorted by episode index
        { episode: 0, agent: "A",
          status: "running",            // running | no-rows | rows | rows+fact
          eligible: null,               // null until Verify, then true or false
          rows: 0,                      // rows added at commit
          fact: null,                   // fact id when admitted
          reason: null },               // reader words for no-rows or not admitted
      ],
      rowCount: 0,                      // this floor's tray
      training: {
        status: "none",                 // none | training | candidate | valid | invalid | trainer-error
        base: null,                     // "configured-base" once a candidate exists
        rowsFrom: [],                   // episodes whose rows the candidate used
      },
    },
  ],
}
```

Notes:

- Shared facts (`knowledge`) and training rows (`floors[g].episodes[].rows`,
  `floors[g].rowCount`) are separate destinations. An admitted episode also
  has rows, but rows never imply a fact.
- `status: "rows"` with a `reason` such as `"duplicate of an admitted fact"`
  means an eligible rewrite that the archive did not admit. It is not a failed
  rewrite.
- Abstract identifiers: facts are `F<floor>.<episode>`. `rowIds(floor,
  episode, n)` gives row ids `R<floor>.<episode>.<j>`. No transcripts or private
  scores are present.
- Current serving model: `floors[generation].serving`. A rejected or failed
  candidate never changes it, and never adds a floor.

## Functions

| export | purpose |
|---|---|
| `initialState(id)` | Floor 1, collecting, serving the configured base, no facts. Reset uses this. |
| `enabledActions(state)` | Enabled actions, in a stable order. Empty in terminal phases. |
| `transition(state, actionOrId)` | Pure: returns a new snapshot. Throws `action not enabled: …` with the reason. Does not mutate its input. |
| `whyDisabled(state, id)` | Reader-word reason an action is unavailable, or `null` if enabled. |
| `isTerminal(state)` | `phase` is `complete`, `trainer-failed`, or `artifact-rejected`. |
| `makeDemo(id, { schedule }?)` | `{ scenario, kind: "model-derived demo", note, steps: [{ index, action, label, state }] }`. Deterministic. Step 0 is `initialState(id)`. Each later step equals `transition(previous, action)`. |
| `agentView(state, agent)` | Roster entry: stage, episode, context, `eligible` (only while rewriting), and the agent's last settled episode on this floor. |
| `summary(state)` | Counts for status text: floor, floors open, phase, serving, training base, facts, rows this floor and all floors, published, candidate status, active agents. |
| `checkRules(state)` | Names of broken rules (normally `[]`). Mirrors the TLA+ invariants. |
| `chartRegions(state)` | `{ "Agent A", "Agent B", Floor }` labels. Each label is a node label in `checked/fixed-scenarios/chart.mmd`. |
| `scenarioIds()`, `scenario(id)` | Scenario list and metadata. |
| `toModelVars(state)`, `chartRegionsFromModel(vars)` | TLA+ projection, used by verification and `model/view.ts`. |

Constants: `AGENTS`, `STAGES`, `WORK_STAGES`, `PHASES`, `TERMINAL_PHASES`,
`TRAINING_BASE`, `EPISODES_PER_FLOOR`, `STAGE_LABELS`, `PHASE_LABELS`, and
`RULES` (rule name → reader sentence). Helpers: `factId`, `rowIds`,
`servingName`, `episodeOf`.

### Actions

An action is `{ id, type, tla, label, agent?, pass?, rows?, admit?, result? }`.
`label` is a reader-word sentence for status text. `tla` is the TLA+ action
name.

| id | when enabled |
|---|---|
| `start:A` / `start:B` | collecting, agent idle, and the agent has not run its episode on this floor |
| `advance:X` | agent at explore, hypothesise, or code |
| `verify-pass:X` / `verify-fail:X` | agent at verify; fixed scenarios offer exactly one |
| `commit:X` | agent at rewrite. Free scenarios use `commit:X:<rows>:<admit\|reject>` |
| `publish` | collecting, both agents idle, both episodes run |
| `start-training` | published, not the final floor |
| `trainer-produces` / `trainer-fails` | training |
| `validate` / `reject` | candidate |
| `serve-next` | validated |

## Chart highlighting

`checked/fixed-scenarios/chart.svg` is the canonical Pele rendering of
`chart.mmd`. Each region (`Agent A`, `Agent B`, `Floor`) is a cluster. Each
state node is a `<g class="pele-node …" data-id="sN">`. To highlight the
current snapshot, build a map from (region, label) to `sN` by reading
`chart.mmd` (`state "<region>" as rK {` … `sN: <label>`). Then add the `active`
class to the three nodes for `chartRegions(snapshot)`. The SVG style already
defines `.pele .active`. Chart nodes merge many concrete states, and region
arrows do not define independent machines. Use `enabledActions`, not
the arrows, to decide what can happen next.

## Changing this contract

Additive view-only helpers can be added without new checks. Any change to
snapshot fields that the model covers, to actions, guards, or outcomes needs
new TLC checks (`scripts/check-model.sh`), parity (`verify/parity.mjs`), and
review.
