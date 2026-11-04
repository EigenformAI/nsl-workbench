// Randomised interleaving check for the logical-concurrency claims.
//
//   node verify/jitter.mjs [runs-per-scenario] [seed]
//
// Each run gives the two agents random, changing speeds (jitter) and picks
// the next enabled action by those weights. Floor actions compete with agent
// actions on equal terms. For every fixed scenario it checks that:
//   - every visited state satisfies the rules;
//   - every run ends in a terminal phase within the longest schedule TLC found;
//   - per-floor totals (rows, facts, publication, training result, floors
//     opened) are the same for every schedule;
//   - schedules do change interleaving details: which agent finished first,
//     and which facts each episode read at its start.
import { AGENTS, WORK_STAGES, checkRules, enabledActions, initialState, isTerminal, transition } from "../lab-process.js";

const runs = Number(process.argv[2] ?? 2000);
let seed = Number(process.argv[3] ?? 20261104);
const LONGEST = 46; // longest schedule in TLC's fixed-scenarios graph (verify/parity.mjs)

// Mulberry32: small, seedable, deterministic.
const rand = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const totals = (s) =>
  JSON.stringify({
    phase: s.phase,
    floors: s.floors.map((f) => ({
      rows: f.rowCount,
      published: f.published,
      training: f.training.status,
      results: f.episodes.map((e) => [e.episode, e.status, e.rows]),
    })),
    facts: s.knowledge.map((f) => f.id),
  });

let failed = false;
for (const id of ["materials", "no-archive", "trainer-failure", "invalid-artifact"]) {
  const outcomes = new Set();
  const details = new Set();
  let violations = 0;
  let tooLong = 0;
  let overlapRuns = 0;
  let maxSteps = 0;
  for (let r = 0; r < runs; r++) {
    let s = initialState(id);
    const speed = Object.fromEntries(AGENTS.map((a) => [a, 0.2 + rand()]));
    let steps = 0;
    let overlapped = false;
    const order = [];
    while (!isTerminal(s)) {
      for (const a of AGENTS) speed[a] = Math.min(3, Math.max(0.05, speed[a] * (0.6 + 0.8 * rand()))); // jitter
      const enabled = enabledActions(s);
      const weights = enabled.map((x) => (x.agent ? speed[x.agent] : 1));
      let pick = rand() * weights.reduce((x, y) => x + y, 0);
      let i = 0;
      while (pick > weights[i] && i < enabled.length - 1) pick -= weights[i++];
      const act = enabled[i];
      if (act.type === "commit" || act.type === "verify") order.push(`${s.generation}${act.agent}`);
      s = transition(s, act);
      steps++;
      if (checkRules(s).length) violations++;
      const st = AGENTS.map((a) => s.agents[a].stage);
      if (st.every((x) => WORK_STAGES.includes(x)) && st[0] !== st[1]) overlapped = true;
      if (steps > LONGEST) break;
    }
    if (!isTerminal(s)) tooLong++;
    if (overlapped) overlapRuns++;
    maxSteps = Math.max(maxSteps, steps);
    outcomes.add(totals(s));
    details.add(JSON.stringify([order, s.knowledge.map((f) => [f.id, f.parents])]));
  }
  const ok = violations === 0 && tooLong === 0 && outcomes.size === 1 && details.size > 1;
  if (!ok) failed = true;
  console.log(
    `${id}: ${runs} runs; rule violations ${violations}; unfinished ${tooLong}; longest ${maxSteps} steps; ` +
      `distinct per-floor totals ${outcomes.size}; distinct interleavings/reading patterns ${details.size}; ` +
      `runs with both agents at different stations ${overlapRuns} — ${ok ? "OK" : "FAIL"}`,
  );
}
process.exit(failed ? 1 : 0);
