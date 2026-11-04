// Compares the JavaScript process model with TLC's reachable state graph.
//
//   node verify/parity.mjs [DUMP_DIR]
//
// DUMP_DIR holds <config>.dot files from `scripts/check-model.sh` (default
// $LAB_DUMP). For each configuration, the JavaScript breadth-first graph must
// equal TLC's graph: the same projected states, the same initial states, and
// the same (state, action name, state) steps. TLC's terminal stutter steps
// (`Finished`) have no JavaScript action and are checked separately.
//
// It also checks, on TLC's graph, that no non-terminal state is a dead end and
// that the graph has no cycle, so every schedule of the bounded model ends in a
// terminal phase. For the drawn configuration it checks that every reachable
// state maps to chart nodes in chart.mmd.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AGENTS, TERMINAL_PHASES, WORK_STAGES, chartRegions, enabledActions } from "../lab-process.js";
import { explore, keyOf } from "./graph.mjs";
import { readDot } from "./tla-value.mjs";

const here = join(dirname(fileURLToPath(import.meta.url)), "..");
const dumpDir = process.argv[2] ?? process.env.LAB_DUMP;
if (!dumpDir) {
  console.error("usage: node verify/parity.mjs DUMP_DIR (or set LAB_DUMP)");
  process.exit(2);
}

const scenariosOf = (cfg) => {
  const text = readFileSync(join(here, "model", `${cfg}.cfg`), "utf8");
  return [.../CONSTANT Scenarios = \{([^}]*)\}/.exec(text)[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
};

const sample = (set, n = 3) => [...set].slice(0, n);
let failed = false;
const fail = (msg) => {
  failed = true;
  console.log(`  FAIL ${msg}`);
};

async function tlcGraph(file) {
  const fpKey = new Map();
  const phase = new Map();
  const initial = new Set();
  const rawEdges = [];
  await readDot(file, {
    onState(fp, vars, isInitial) {
      const k = keyOf(vars);
      fpKey.set(fp, k);
      phase.set(k, vars.phase);
      if (isInitial) initial.add(k);
    },
    onEdge(from, to, action) {
      rawEdges.push([from, to, action]);
    },
  });
  const edges = new Set();
  const stutters = [];
  for (const [from, to, action] of rawEdges) {
    const f = fpKey.get(from);
    const t = fpKey.get(to);
    if (action === "Finished") stutters.push([f, t]);
    else edges.add(`${f}|${action}|${t}`);
  }
  return { states: new Set(fpKey.values()), phase, initial, edges, stutters };
}

function checkTermination(tlc) {
  const out = new Map();
  for (const e of tlc.edges) {
    const [f, , t] = e.split("|");
    if (!out.has(f)) out.set(f, []);
    out.get(f).push(t);
  }
  let deadEnds = 0;
  for (const k of tlc.states) if (!out.has(k) && !TERMINAL_PHASES.includes(tlc.phase.get(k))) deadEnds++;
  const badStutters = tlc.stutters.filter(([f, t]) => f !== t || !TERMINAL_PHASES.includes(tlc.phase.get(f))).length;
  // Iterative DFS cycle check, plus the longest path length.
  const color = new Map();
  const depth = new Map();
  let cycle = false;
  for (const root of tlc.states) {
    if (color.has(root)) continue;
    const stack = [[root, 0]];
    color.set(root, 1);
    while (stack.length) {
      const top = stack[stack.length - 1];
      const succ = out.get(top[0]) ?? [];
      if (top[1] < succ.length) {
        const n = succ[top[1]++];
        if (color.get(n) === 1) cycle = true;
        else if (!color.has(n)) {
          color.set(n, 1);
          stack.push([n, 0]);
        }
      } else {
        color.set(top[0], 2);
        depth.set(top[0], Math.max(0, ...succ.map((n) => (depth.get(n) ?? 0) + 1)));
        stack.pop();
      }
    }
  }
  const longest = Math.max(...[...tlc.initial].map((k) => depth.get(k)));
  return { deadEnds, badStutters, cycle, longest };
}

function chartNodes() {
  const mmd = readFileSync(join(here, "checked", "fixed-scenarios", "chart.mmd"), "utf8");
  const regions = {};
  let current = null;
  for (const line of mmd.split("\n")) {
    const r = /^\s*state "([^"]+)" as r\d+ \{/.exec(line);
    if (r) {
      current = r[1];
      regions[current] = new Set();
      continue;
    }
    const n = /^\s*s\d+: (.+)$/.exec(line);
    if (n && current) regions[current].add(n[1]);
  }
  return regions;
}

for (const cfg of ["fixed-scenarios", "free-outcomes"]) {
  const file = join(dumpDir, `${cfg}.dot`);
  console.log(`${cfg}:`);
  if (!existsSync(file)) {
    fail(`missing ${file}; run scripts/check-model.sh first`);
    continue;
  }
  const scenarios = scenariosOf(cfg);
  const drawn = cfg === "fixed-scenarios";
  const js = explore(scenarios, { keepStates: drawn });
  const tlc = await tlcGraph(file);
  const jsKeys = new Set(js.states.keys());
  const missingInJs = [...tlc.states].filter((k) => !jsKeys.has(k));
  const missingInTlc = [...jsKeys].filter((k) => !tlc.states.has(k));
  const edgesOnlyTlc = [...tlc.edges].filter((e) => !js.edges.has(e));
  const edgesOnlyJs = [...js.edges].filter((e) => !tlc.edges.has(e));
  const initialSame = js.initial.size === tlc.initial.size && [...js.initial].every((k) => tlc.initial.has(k));
  console.log(`  scenarios: ${scenarios.join(", ")}`);
  console.log(`  states: JS ${jsKeys.size}, TLC ${tlc.states.size}; steps: JS ${js.edges.size}, TLC ${tlc.edges.size} (+${tlc.stutters.length} terminal stutters)`);
  if (missingInJs.length || missingInTlc.length) fail(`state sets differ: ${missingInJs.length} only in TLC, ${missingInTlc.length} only in JS`);
  if (edgesOnlyTlc.length || edgesOnlyJs.length)
    fail(`step sets differ: ${edgesOnlyTlc.length} only in TLC ${sample(edgesOnlyTlc)}, ${edgesOnlyJs.length} only in JS ${sample(edgesOnlyJs)}`);
  if (!initialSame) fail("initial states differ");
  const term = checkTermination(tlc);
  console.log(`  TLC graph: dead ends outside terminal phases ${term.deadEnds}; cycles ${term.cycle ? "yes" : "none"}; longest schedule ${term.longest} steps; misplaced stutters ${term.badStutters}`);
  if (term.deadEnds || term.cycle || term.badStutters) fail("termination check");

  if (drawn) {
    const nodes = chartNodes();
    let unmapped = 0;
    let overlap = 0;
    let bothCanMove = 0;
    for (const s of js.states.values()) {
      for (const [region, label] of Object.entries(chartRegions(s))) if (!nodes[region]?.has(label)) unmapped++;
      const st = AGENTS.map((a) => s.agents[a].stage);
      if (st.every((x) => WORK_STAGES.includes(x)) && st[0] !== st[1]) overlap++;
      const movers = new Set(enabledActions(s).filter((x) => x.agent).map((x) => x.agent));
      if (movers.size === AGENTS.length) bothCanMove++;
    }
    console.log(`  chart: ${unmapped} reachable state regions missing from chart.mmd`);
    console.log(`  concurrency: ${overlap} states with both agents mid-episode at different stations; ${bothCanMove} states where both agents can act`);
    if (unmapped) fail("chart mapping");
    if (!overlap || !bothCanMove) fail("no concurrent states");
  }
  if (!failed) console.log("  OK: JavaScript and TLC graphs are identical");
}
process.exit(failed ? 1 : 0);
