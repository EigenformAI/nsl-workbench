// Behaviour checks for the learning-lab process module.
// LAB_PROCESS may point at another copy of the module (verify/mutants.mjs uses
// this to confirm each check rejects a credible wrong implementation).
import assert from "node:assert/strict";
import { test } from "node:test";
import { pathToFileURL } from "node:url";

const L = await import(process.env.LAB_PROCESS ? pathToFileURL(process.env.LAB_PROCESS).href : "../lab-process.js");

const AGENTS_ = L.AGENTS;
const run = (state, ...ids) => ids.reduce((s, id) => L.transition(s, id), state);
const ids = (state) => L.enabledActions(state).map((a) => a.id);
const fullEpisode = (a) => [`start:${a}`, `advance:${a}`, `advance:${a}`, `advance:${a}`];
// Run floor 1 of the primary scenario: both slots pass verification.
const floorOneSettled = () =>
  run(L.initialState("materials"), ...fullEpisode("A"), "verify-pass:A", "commit:A", ...fullEpisode("B"), "verify-pass:B", "commit:B");
const toPhase = (state, phase) => {
  let s = state;
  for (let n = 0; s.phase !== phase; n++) {
    assert.ok(n < 100, `did not reach ${phase}`);
    s = L.transition(s, L.enabledActions(s).find((a) => !a.agent) ?? L.enabledActions(s)[0]);
  }
  return s;
};
const reachable = (scenarioId) => {
  const seen = new Map();
  const queue = [L.initialState(scenarioId)];
  while (queue.length) {
    const s = queue.pop();
    const k = JSON.stringify(s);
    if (seen.has(k)) continue;
    seen.set(k, s);
    assert.ok(seen.size <= 1000, `${scenarioId}: more reachable states than the bounded model has`);
    for (const a of L.enabledActions(s)) queue.push(L.transition(s, a));
  }
  return [...seen.values()];
};

test("an episode contributes once: no second commit, no second episode for the slot on this floor", () => {
  const s = run(L.initialState("materials"), ...fullEpisode("A"), "verify-pass:A", "commit:A");
  assert.equal(s.floors[0].rowCount, 2);
  assert.ok(!ids(s).includes("commit:A"));
  assert.ok(!ids(s).includes("start:A"));
  assert.throws(() => L.transition(s, "commit:A"), /not enabled/);
  assert.equal(s.floors[0].episodes.filter((e) => e.agent === "A").length, 1);
});

test("the floor cannot publish while either agent is mid-episode", () => {
  let s = run(L.initialState("materials"), ...fullEpisode("A"), "verify-pass:A", "commit:A", "start:B");
  for (const step of ["advance:B", "advance:B", "advance:B", "verify-pass:B"]) {
    assert.ok(!ids(s).includes("publish"));
    assert.match(L.whyDisabled(s, "publish"), /Agent B/);
    s = L.transition(s, step);
  }
  assert.ok(!ids(s).includes("publish"), "still rewriting");
  s = L.transition(s, "commit:B");
  assert.ok(ids(s).includes("publish"));
});

test("the floor cannot publish before every slot has run its episode", () => {
  const s = run(L.initialState("materials"), ...fullEpisode("A"), "verify-pass:A", "commit:A");
  assert.ok(!ids(s).includes("publish"));
});

test("training starts only after publication", () => {
  const settled = floorOneSettled();
  assert.ok(!ids(settled).includes("start-training"));
  assert.throws(() => L.transition(settled, "start-training"), /not enabled/);
  const published = L.transition(settled, "publish");
  assert.equal(published.phase, "published");
  assert.ok(published.floors[0].published);
  assert.deepEqual(ids(published), ["start-training"]);
});

test("only a validated candidate is served, and only it opens a floor", () => {
  const candidate = toPhase(floorOneSettled(), "candidate");
  assert.ok(!ids(candidate).includes("serve-next"));
  assert.equal(candidate.floors.length, 1);
  const validated = L.transition(candidate, "validate");
  const next = L.transition(validated, "serve-next");
  assert.equal(next.floors.length, 2);
  assert.equal(next.floors[1].serving, "adapter-from-floor-0");
  assert.equal(next.floors[0].serving, "configured-base");
});

test("a rejected candidate or a trainer error leaves serving and the floor count unchanged", () => {
  const rejected = L.makeDemo("invalid-artifact").steps.at(-1).state;
  assert.equal(rejected.phase, "artifact-rejected");
  assert.equal(rejected.floors.length, 2);
  assert.equal(rejected.floors[1].serving, "adapter-from-floor-0");
  assert.equal(rejected.floors[1].training.status, "invalid");
  assert.deepEqual(ids(rejected), []);

  const failed = L.makeDemo("trainer-failure").steps.at(-1).state;
  assert.equal(failed.phase, "trainer-failed");
  assert.equal(failed.floors.length, 1);
  assert.equal(failed.floors[0].training.status, "trainer-error");
  assert.equal(failed.floors[0].training.base, null);
});

test("the final floor publishes without training", () => {
  const steps = L.makeDemo("materials").steps;
  const last = steps.at(-1).state;
  assert.equal(last.phase, "complete");
  assert.equal(last.floors.length, 3);
  assert.ok(last.floors[2].published);
  assert.equal(last.floors[2].training.status, "none");
  assert.deepEqual(ids(last), []);
  assert.match(L.whyDisabled(last, "start-training"), /ended|final/);
  assert.equal(steps.filter((x) => x.action === "start-training").length, 2);
});

test("training always reloads the fixed configured base on this floor's published rows", () => {
  const steps = L.makeDemo("materials").steps;
  const candidates = steps.filter((x) => x.action === "trainer-produces").map((x) => x.state);
  assert.equal(candidates.length, 2);
  const second = candidates[1];
  assert.equal(second.floors[1].serving, "adapter-from-floor-0");
  assert.equal(second.floors[1].training.base, L.TRAINING_BASE);
  assert.deepEqual(second.floors[1].training.rowsFrom, [1]); // episode 0 failed its signal check
});

test("a task without a shared archive never gains facts, but still gains rows", () => {
  for (const s of reachable("no-archive")) assert.deepEqual(s.knowledge, []);
  const last = L.makeDemo("no-archive").steps.at(-1).state;
  assert.deepEqual(
    last.floors.map((f) => f.rowCount),
    [3, 1, 3],
  );
});

test("row eligibility and archive admission are separate outcomes", () => {
  const s = floorOneSettled();
  const [a, b] = s.floors[0].episodes;
  assert.deepEqual([a.status, a.rows, a.fact], ["rows+fact", 2, "F0.0"]);
  assert.deepEqual([b.status, b.rows, b.fact], ["rows", 1, null]); // eligible, not admitted
  assert.equal(b.eligible, true);
  assert.equal(b.reason, "duplicate of an admitted fact");
  assert.equal(s.floors[0].rowCount, 3);
  assert.deepEqual(s.knowledge.map((f) => f.id), ["F0.0"]);
  // Floor 2, slot A fails its signal check: no rows and no fact.
  const floor2 = L.makeDemo("materials").steps.map((x) => x.state).find((x) => x.floors[1]?.episodes[0]?.status === "no-rows");
  assert.equal(floor2.floors[1].episodes[0].rows, 0);
  assert.equal(floor2.floors[1].episodes[0].eligible, false);
});

test("a fact appears only at the durable commit, not at verification", () => {
  const verified = run(L.initialState("materials"), ...fullEpisode("A"), "verify-pass:A");
  assert.deepEqual(verified.knowledge, []);
  assert.equal(verified.floors[0].rowCount, 0);
  assert.equal(L.agentView(verified, "A").eligible, true);
  const committed = L.transition(verified, "commit:A");
  assert.deepEqual(committed.knowledge.map((f) => f.id), ["F0.0"]);
});

test("agents read shared knowledge at episode start, not continuously", () => {
  // B starts before A commits: B's context and its fact's parents exclude A's fact.
  const early = run(L.initialState("materials"), "start:A", "start:B", "advance:A", "advance:A", "advance:A", "verify-pass:A", "commit:A");
  assert.deepEqual(early.agents.B.context, []);
  // Both admitted on one floor (free outcomes): B's fact keeps the parents B
  // read at its start, not the facts present at its commit.
  const both = run(
    L.initialState("free-outcomes"),
    "start:A", "start:B", "advance:A", "advance:A", "advance:A", "verify-pass:A", "commit:A:1:admit",
    "advance:B", "advance:B", "advance:B", "verify-pass:B", "commit:B:1:admit",
  );
  assert.deepEqual(both.knowledge.map((f) => [f.id, f.parents]), [["F0.0", []], ["F0.1", []]]);
  // B starts after A commits: it reads A's fact.
  const late = run(L.initialState("materials"), ...fullEpisode("A"), "verify-pass:A", "commit:A", "start:B");
  assert.deepEqual(late.agents.B.context, ["F0.0"]);
  // Knowledge persists to the next floor, and a new episode reads it there.
  const floor2 = L.makeDemo("materials").steps.map((x) => x.state).find((x) => x.generation === 1 && x.agents.A.stage === "explore");
  assert.ok(floor2.agents.A.context.includes("F0.0"));
  const lastFact = L.makeDemo("materials").steps.at(-1).state.knowledge.find((f) => f.id === "F1.1");
  assert.ok(lastFact.parents.includes("F0.0"));
});

test("outcome booleans stay hidden until the step that decides them", () => {
  let s = L.initialState("materials");
  for (const step of fullEpisode("A")) {
    s = L.transition(s, step);
    assert.equal(L.agentView(s, "A").eligible, null);
    assert.equal(s.floors[0].episodes[0].eligible, null);
  }
});

test("both agents can be active at different stations and advance independently", () => {
  const s = run(L.initialState("materials"), "start:A", "advance:A", "start:B");
  assert.deepEqual([s.agents.A.stage, s.agents.B.stage], ["hypothesise", "explore"]);
  const a = ids(s);
  assert.ok(a.includes("advance:A") && a.includes("advance:B"));
  assert.deepEqual(L.transition(L.transition(s, "advance:A"), "advance:B"), L.transition(L.transition(s, "advance:B"), "advance:A"));
});

test("every reachable state of every fixed scenario satisfies the rules", () => {
  for (const id of ["materials", "no-archive", "trainer-failure", "invalid-artifact"])
    for (const s of reachable(id)) assert.deepEqual(L.checkRules(s), [], `${id}: ${JSON.stringify(s)}`);
});

test("the rule checker rejects corrupted snapshots", () => {
  const s = floorOneSettled();
  const fakeFact = structuredClone(s);
  fakeFact.knowledge.push({ id: "F0.1", floor: 0, episode: 1, agent: "B", parents: [] });
  assert.ok(L.checkRules(fakeFact).includes("FactsOnlyAfterCommit"));
  const doubled = structuredClone(s);
  doubled.floors[0].rowCount += 1;
  assert.ok(L.checkRules(doubled).includes("NoDoubleContribution"));
  const continued = toPhase(floorOneSettled(), "candidate");
  continued.floors[0].training.base = "adapter-from-floor-0";
  assert.ok(L.checkRules(continued).includes("FixedTrainingBase"));
});

test("demos are deterministic, follow the transition function, and keep full floor history", () => {
  for (const id of ["materials", "no-archive", "trainer-failure", "invalid-artifact"]) {
    const demo = L.makeDemo(id);
    assert.deepEqual(demo, L.makeDemo(id));
    assert.equal(demo.kind, "model-derived demo");
    assert.deepEqual(demo.steps[0].state, L.initialState(id));
    for (let i = 1; i < demo.steps.length; i++) {
      const { state, action } = demo.steps[i];
      assert.deepEqual(state, L.transition(demo.steps[i - 1].state, action));
      assert.equal(state.floors.length, state.generation + 1);
      // Earlier floors are carried forward unchanged once the next floor opens.
      for (let g = 0; g < state.generation; g++) assert.deepEqual(state.floors[g], demo.steps[i - 1].state.floors[g]);
      assert.deepEqual(JSON.parse(JSON.stringify(state)), state);
    }
    assert.ok(L.isTerminal(demo.steps.at(-1).state));
  }
  const overlap = L.makeDemo("materials").steps.some(({ state }) => {
    const [a, b] = [state.agents.A.stage, state.agents.B.stage];
    return a !== "idle" && b !== "idle" && a !== b;
  });
  assert.ok(overlap, "the staggered demo shows both agents at different stations");
  const sameFloorRead = L.makeDemo("materials").steps.some(({ state }) =>
    AGENTS_.some((a) => state.agents[a].context.some((id) => id.startsWith(`F${state.generation}.`))),
  );
  assert.ok(sameFloorRead, "the demo shows an episode reading a fact committed earlier on its floor");
});

test("transition does not mutate its input", () => {
  const s = L.initialState("materials");
  const before = JSON.stringify(s);
  L.transition(s, "start:A");
  assert.equal(JSON.stringify(s), before);
});
