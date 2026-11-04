// Learning-lab process model: concurrent episode agents within a generation,
// a publication barrier, fixed-base training, and a validated handoff.
//
// This file is the only owner of process transitions for the learning-lab figure.
// It is a pure ES module with no DOM or Node APIs, so it runs unchanged in a
// browser (`<script type="module">`) and in Node. It mirrors
// `model/LearningLab.tla` action for action; `verify/parity.mjs` compares the
// two state graphs exactly over the fields both share (see toModelVars).
// JS-only fields (reasons, fact ids on episodes, agent names on records) are
// covered by checkRules() and the behaviour tests.
//
// The model is bounded: two episode slots, three generation floors, one
// episode per slot per floor, and fixed (scenario-chosen) verification,
// archive and training outcomes. It does not model time, crashes, deadlines,
// circuit breakers, recovery, or model quality.

export const AGENTS = Object.freeze(["A", "B"]);
export const STAGES = Object.freeze(["idle", "explore", "hypothesise", "code", "verify", "rewrite"]);
export const WORK_STAGES = Object.freeze(["explore", "hypothesise", "code", "verify", "rewrite"]);
export const PHASES = Object.freeze([
  "collecting",
  "published",
  "training",
  "candidate",
  "validated",
  "complete",
  "trainer-failed",
  "artifact-rejected",
]);
export const TERMINAL_PHASES = Object.freeze(["complete", "trainer-failed", "artifact-rejected"]);
export const TRAINING_BASE = "configured-base";

// Episode budget: each slot runs exactly one episode per floor, and slot A's
// episode is episode 0, slot B's is episode 1. The real runner keeps handing
// new episodes to free workers; this bound keeps the whole state graph small
// enough to draw.
export const EPISODES_PER_FLOOR = AGENTS.length;
export const episodeOf = (agent) => AGENTS.indexOf(agent);

// Episode outcomes are exogenous and fixed per scenario. `signal` is the
// signal-pass training-eligibility verdict revealed at Verify. `rows` is the
// number of successful rewrite inference rows the episode contributes. `admit`
// is the archive admission verdict at durable commit (ignored when the task has
// no shared archive). `reason` explains a non-admission in reader words.
const ok = (rows) => ({ signal: true, rows, admit: true });
const notAdmitted = (rows, reason) => ({ signal: true, rows, admit: false, reason });
const ineligible = { signal: false, rows: 0, admit: false, reason: "signal check failed" };

const PRIMARY_EPISODES = [
  [ok(2), notAdmitted(1, "duplicate of an admitted fact")],
  [ineligible, ok(1)],
  [ok(1), notAdmitted(2, "marginal gain too small")],
];

// Demo schedules: on each step the preferred agent (one letter per step) acts
// if it can; otherwise the other agent acts; floor actions run when no agent
// can act. Characters past the end of a pattern alternate A/B.
// Floor 1 overlaps the agents; on floor 2, B commits before A starts, so A
// reads a fact from its own floor; floor 3 overlaps them with B leading.
const STAGGERED = ["AAABABABABABABAB", "BBBBBBAAAAAA", "BBBABABABABABAB"];

const SCENARIOS = {
  materials: {
    id: "materials",
    title: "Materials task with a shared archive",
    summary:
      "Two episode slots work concurrently. Signal-pass rewrites add training rows; " +
      "archive-admitted ones also add a shared fact. Both handoffs validate, so three floors open.",
    graph: true,
    generations: 3,
    episodes: PRIMARY_EPISODES,
    training: ["valid", "valid"],
    demoSchedule: STAGGERED,
  },
  "no-archive": {
    id: "no-archive",
    title: "Generic task without a shared archive",
    summary:
      "Same episode outcomes, but the task has no shared archive. Rewrites still add training rows; " +
      "no shared facts or graph links ever appear.",
    graph: false,
    generations: 3,
    episodes: PRIMARY_EPISODES,
    training: ["valid", "valid"],
    demoSchedule: STAGGERED,
  },
  "trainer-failure": {
    id: "trainer-failure",
    title: "Trainer fails after the first floor",
    summary:
      "The first floor publishes, but the trainer errors before writing a candidate. " +
      "Serving does not change and no second floor opens.",
    graph: true,
    generations: 3,
    episodes: PRIMARY_EPISODES,
    training: ["trainer-error", "valid"],
    demoSchedule: STAGGERED,
  },
  "invalid-artifact": {
    id: "invalid-artifact",
    title: "Second candidate fails validation",
    summary:
      "The first handoff validates and opens floor 2. The second trainer run writes a candidate " +
      "that fails format and identity checks, so floor 3 never opens.",
    graph: true,
    generations: 3,
    episodes: PRIMARY_EPISODES,
    training: ["valid", "invalid"],
    demoSchedule: STAGGERED,
  },
  // Every verification, row-count, admission and training outcome is a free
  // choice. Used for exhaustive rule checking; it has no demo.
  "free-outcomes": {
    id: "free-outcomes",
    title: "All outcome combinations (checking only)",
    summary:
      "Verification, row count (1 or 2), archive admission, trainer result and validation " +
      "are chosen freely at each step. Used to check the rules, not for playback.",
    graph: true,
    generations: 3,
    episodes: null,
    training: null,
    demoSchedule: null,
  },
  "free-outcomes-no-archive": {
    id: "free-outcomes-no-archive",
    title: "All outcome combinations without an archive (checking only)",
    summary: "As free-outcomes, for a task without a shared archive.",
    graph: false,
    generations: 3,
    episodes: null,
    training: null,
    demoSchedule: null,
  },
};

const deepFreeze = (o) => {
  if (o && typeof o === "object" && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
};
deepFreeze(SCENARIOS);

export const scenarioIds = () => Object.keys(SCENARIOS);

export function scenario(id) {
  const s = SCENARIOS[id];
  if (!s) throw new Error(`unknown scenario: ${id}`);
  return s;
}

// ---------- identifiers ----------

export const factId = (floor, episode) => `F${floor}.${episode}`;
export const rowIds = (floor, episode, rows) =>
  Array.from({ length: rows }, (_, j) => `R${floor}.${episode}.${j + 1}`);

// ---------- state ----------

const idleAgent = () => ({ stage: "idle", episode: null, context: [] });

const newFloor = (floor, serving) => ({
  floor,
  serving, // "configured-base" on floor 0, else "adapter-from-floor-N"
  published: false,
  episodes: [], // issued episodes, sorted by episode index
  rowCount: 0,
  training: { status: "none", base: null, rowsFrom: [] },
});

export const servingName = (fromFloor) => (fromFloor < 0 ? "configured-base" : `adapter-from-floor-${fromFloor}`);

export function initialState(scenarioId) {
  scenario(scenarioId);
  return {
    scenario: scenarioId,
    generation: 0,
    phase: "collecting",
    trainingBase: TRAINING_BASE, // every candidate reloads this; it never changes
    agents: Object.fromEntries(AGENTS.map((a) => [a, idleAgent()])),
    knowledge: [],
    floors: [newFloor(0, servingName(-1))],
  };
}

const clone = (s) => JSON.parse(JSON.stringify(s));
const byEpisode = (x, y) => x.episode - y.episode;
const compareId = (x, y) => (x < y ? -1 : x > y ? 1 : 0);

// ---------- outcome choices ----------

const episodeOutcome = (scn, floor, episode) => scn.episodes?.[floor]?.[episode] ?? null;

function verifyChoices(scn, floor, episode) {
  const o = episodeOutcome(scn, floor, episode);
  return o ? [o.signal] : [true, false];
}

function commitChoices(scn, floor, episode) {
  const o = episodeOutcome(scn, floor, episode);
  const admits = (a) => (scn.graph ? a : false);
  if (o) return [{ rows: o.rows, admit: admits(o.admit) }];
  const out = [];
  for (const rows of [1, 2]) for (const admit of scn.graph ? [true, false] : [false]) out.push({ rows, admit });
  return out;
}

function trainingChoices(scn, floor) {
  const t = scn.training?.[floor];
  if (!t) return { trainer: ["candidate", "trainer-error"], validation: ["valid", "invalid"] };
  return {
    trainer: [t === "trainer-error" ? "trainer-error" : "candidate"],
    validation: [t === "invalid" ? "invalid" : "valid"],
  };
}

// ---------- actions ----------
//
// An action is a plain object. `id` is unique among the enabled actions of a
// state, `tla` is the matching TLA+ action name, and `label` is reader words.

const STAGE_DONE = {
  explore: "finishes exploring public evidence",
  hypothesise: "writes a hypothesis",
  code: "submits descriptor code",
};

const agentLabel = (a) => `Agent ${a}`;

export function enabledActions(state) {
  const scn = scenario(state.scenario);
  const floor = state.floors[state.generation];
  const out = [];
  if (state.phase === "collecting") {
    for (const a of AGENTS) {
      const ag = state.agents[a];
      if (ag.stage === "idle" && !floor.episodes.some((e) => e.episode === episodeOf(a))) {
        out.push({
          id: `start:${a}`,
          type: "start",
          agent: a,
          tla: "StartEpisode",
          label: `${agentLabel(a)} starts its episode and reads the committed facts`,
        });
      } else if (STAGE_DONE[ag.stage]) {
        out.push({
          id: `advance:${a}`,
          type: "advance",
          agent: a,
          tla: "Advance",
          label: `${agentLabel(a)} ${STAGE_DONE[ag.stage]}`,
        });
      } else if (ag.stage === "verify") {
        for (const pass of verifyChoices(scn, state.generation, ag.episode)) {
          out.push(
            pass
              ? { id: `verify-pass:${a}`, type: "verify", agent: a, pass, tla: "VerifyPasses",
                  label: `${agentLabel(a)}'s signal check passes; rewrite is training-eligible` }
              : { id: `verify-fail:${a}`, type: "verify", agent: a, pass, tla: "VerifyFails",
                  label: `${agentLabel(a)}'s signal check fails; the episode ends without rows` },
          );
        }
      } else if (ag.stage === "rewrite") {
        const choices = commitChoices(scn, state.generation, ag.episode);
        for (const c of choices) {
          const suffix = choices.length > 1 ? `:${c.rows}:${c.admit ? "admit" : "reject"}` : "";
          const reason = episodeOutcome(scn, state.generation, ag.episode)?.reason;
          const fact = !scn.graph
            ? "no shared archive"
            : c.admit
              ? "archive admits a fact"
              : `archive does not admit${reason ? ` (${reason})` : ""}`;
          out.push({
            id: `commit:${a}${suffix}`,
            type: "commit",
            agent: a,
            rows: c.rows,
            admit: c.admit,
            tla: "CommitRewrite",
            label: `${agentLabel(a)} commits its rewrite: ${c.rows} row${c.rows > 1 ? "s" : ""}, ${fact}`,
          });
        }
      }
    }
    const settled = AGENTS.every((a) => state.agents[a].stage === "idle");
    if (settled && floor.episodes.length === AGENTS.length) {
      const final = state.generation === scn.generations - 1;
      out.push({
        id: "publish",
        type: "publish",
        tla: "Publish",
        label: final ? "all workers settled; the final floor publishes (no training follows)" : "all workers settled; the floor publishes",
      });
    }
  } else if (state.phase === "published") {
    out.push({ id: "start-training", type: "start-training", tla: "StartTraining",
      label: "training reloads the configured base with this floor's published rows" });
  } else if (state.phase === "training") {
    for (const r of trainingChoices(scn, state.generation).trainer) {
      out.push(
        r === "candidate"
          ? { id: "trainer-produces", type: "trainer", result: r, tla: "TrainerProduces", label: "the trainer writes a candidate artifact" }
          : { id: "trainer-fails", type: "trainer", result: r, tla: "TrainerFails", label: "the trainer errors; no candidate" },
      );
    }
  } else if (state.phase === "candidate") {
    for (const r of trainingChoices(scn, state.generation).validation) {
      out.push(
        r === "valid"
          ? { id: "validate", type: "validate", result: r, tla: "ArtifactValid", label: "the candidate passes format, identity and configuration checks" }
          : { id: "reject", type: "validate", result: r, tla: "ArtifactInvalid", label: "the candidate fails validation; serving is unchanged" },
      );
    }
  } else if (state.phase === "validated") {
    out.push({ id: "serve-next", type: "serve-next", tla: "ServeNextFloor",
      label: `floor ${state.generation + 2} opens, serving the validated candidate` });
  }
  return out;
}

// Why an action is unavailable, in reader words. Returns null when enabled.
export function whyDisabled(state, actionId) {
  if (enabledActions(state).some((x) => x.id === actionId)) return null;
  const scn = scenario(state.scenario);
  const busy = AGENTS.filter((a) => state.agents[a].stage !== "idle");
  if (TERMINAL_PHASES.includes(state.phase)) return "the run has ended";
  if (actionId === "publish") {
    if (state.phase !== "collecting") return "this floor is already published";
    if (busy.length) return `waiting for ${busy.map(agentLabel).join(" and ")} to settle`;
    if (state.floors[state.generation].episodes.length < AGENTS.length) return "an agent has not run its episode on this floor";
  }
  if (actionId === "start-training") {
    if (state.phase === "collecting") return "training waits for publication";
    if (state.phase === "complete") return "no training follows the final floor";
  }
  if (actionId === "serve-next" && state.phase !== "validated") return "only a validated candidate can be served";
  const m = /^(start|advance|verify-pass|verify-fail|commit):([AB])/.exec(actionId);
  if (m) {
    if (state.phase !== "collecting") return "episodes run only while the floor is collecting";
    const ag = state.agents[m[2]];
    if (m[1] === "start" && ag.stage !== "idle") return `${agentLabel(m[2])} is still running an episode`;
    if (m[1] === "start") return `${agentLabel(m[2])} already ran its episode on this floor`;
    return `${agentLabel(m[2])} is at ${ag.stage}`;
  }
  return "not available in this state";
}

const NEXT_STAGE = { explore: "hypothesise", hypothesise: "code", code: "verify" };

// Pure transition. Throws if the action is not enabled in `state`.
// `action` may be an action object from enabledActions() or its id.
export function transition(state, action) {
  const id = typeof action === "string" ? action : action?.id;
  const act = enabledActions(state).find((x) => x.id === id);
  if (!act) throw new Error(`action not enabled: ${id} (${whyDisabled(state, id)})`);
  const s = clone(state);
  const scn = scenario(s.scenario);
  const floor = s.floors[s.generation];
  const ag = act.agent ? s.agents[act.agent] : null;
  switch (act.type) {
    case "start": {
      ag.stage = "explore";
      ag.episode = episodeOf(act.agent);
      ag.context = s.knowledge.map((f) => f.id); // episode-start snapshot only
      floor.episodes.push({ episode: ag.episode, agent: act.agent, status: "running", eligible: null, rows: 0, fact: null, reason: null });
      floor.episodes.sort(byEpisode);
      break;
    }
    case "advance":
      ag.stage = NEXT_STAGE[ag.stage];
      break;
    case "verify": {
      const rec = floor.episodes.find((e) => e.episode === ag.episode);
      rec.eligible = act.pass;
      if (act.pass) {
        ag.stage = "rewrite";
      } else {
        rec.status = "no-rows";
        rec.reason = "signal check failed";
        s.agents[act.agent] = idleAgent();
      }
      break;
    }
    case "commit": {
      // Durable commit: rows enter this floor's tray; an admitted fact enters
      // the persistent shared knowledge. Both happen only here.
      const rec = floor.episodes.find((e) => e.episode === ag.episode);
      rec.rows = act.rows;
      floor.rowCount += act.rows;
      if (act.admit) {
        const id = factId(s.generation, ag.episode);
        rec.status = "rows+fact";
        rec.fact = id;
        s.knowledge.push({ id, floor: s.generation, episode: ag.episode, agent: act.agent, parents: [...ag.context] });
        s.knowledge.sort((x, y) => compareId(x.id, y.id));
      } else {
        rec.status = "rows";
        // Not admitted is not a failed rewrite: the rows above still count.
        rec.reason = !scn.graph
          ? "task has no shared archive"
          : (episodeOutcome(scn, s.generation, ag.episode)?.reason ?? "archive did not admit");
      }
      s.agents[act.agent] = idleAgent();
      break;
    }
    case "publish":
      floor.published = true;
      s.phase = s.generation === scn.generations - 1 ? "complete" : "published";
      break;
    case "start-training":
      s.phase = "training";
      floor.training.status = "training";
      break;
    case "trainer": {
      if (act.result === "candidate") {
        s.phase = "candidate";
        floor.training = {
          status: "candidate",
          base: TRAINING_BASE, // always the fixed configured base, never the serving adapter
          rowsFrom: floor.episodes.filter((e) => e.rows > 0).map((e) => e.episode),
        };
      } else {
        s.phase = "trainer-failed";
        floor.training.status = "trainer-error";
      }
      break;
    }
    case "validate":
      if (act.result === "valid") {
        s.phase = "validated";
        floor.training.status = "valid";
      } else {
        s.phase = "artifact-rejected";
        floor.training.status = "invalid";
      }
      break;
    case "serve-next": {
      const from = s.generation;
      s.generation += 1;
      s.phase = "collecting";
      s.floors.push(newFloor(s.generation, servingName(from)));
      break;
    }
    default:
      throw new Error(`unhandled action type: ${act.type}`);
  }
  return s;
}

export const isTerminal = (state) => TERMINAL_PHASES.includes(state.phase);

// ---------- derived reader views ----------

// What the roster shows for one agent. Outcome booleans are revealed only
// after the step that decides them: `eligible` after Verify, `admitted` after
// the durable commit.
export function agentView(state, agent) {
  const ag = state.agents[agent];
  const floor = state.floors[state.generation];
  const mine = floor.episodes.filter((e) => e.agent === agent);
  const last = mine.filter((e) => e.status !== "running").at(-1) ?? null;
  return {
    agent,
    stage: ag.stage,
    episode: ag.episode,
    context: ag.context,
    eligible: ag.stage === "rewrite" ? true : null,
    lastSettled: last && {
      episode: last.episode,
      status: last.status,
      eligible: last.eligible,
      rows: last.rows,
      admitted: last.status === "rows+fact",
      fact: last.fact,
      reason: last.reason,
    },
  };
}

export function summary(state) {
  const floor = state.floors[state.generation];
  return {
    generation: state.generation,
    floorsOpen: state.floors.length,
    phase: state.phase,
    serving: floor.serving,
    trainingBase: TRAINING_BASE,
    facts: state.knowledge.length,
    rowsThisFloor: floor.rowCount,
    rowsAllFloors: state.floors.reduce((n, f) => n + f.rowCount, 0),
    published: floor.published,
    candidate: floor.training.status,
    activeAgents: AGENTS.filter((a) => state.agents[a].stage !== "idle"),
  };
}

// ---------- rules (mirror of the TLA+ invariants) ----------

export const RULES = Object.freeze({
  RowsOnlyFromEligibleRewrite: "Rows come only from committed, signal-passing rewrites",
  EligibleRowsKept: "Every committed eligible rewrite keeps its rows, whether or not the archive admits it",
  FactsOnlyAfterCommit: "A shared fact exists only for a committed, admitted episode",
  NoDoubleContribution: "No episode contributes twice; the tray count equals the sum of episode rows",
  NoGraphWithoutArchive: "A task without a shared archive never gains shared facts",
  ContextFromCommittedFacts: "An agent's context and a fact's parents hold only committed facts, never the fact itself",
  NoPublishWhileActive: "A floor publishes only after every slot has run its episode and every agent is idle",
  NoTrainingBeforePublish: "Training and candidates exist only for published floors",
  FixedTrainingBase: "Every candidate is trained from the fixed configured base",
  TrainsOnPublishedRows: "A candidate uses exactly the published rows of its floor",
  ServeOnlyValidated: "A floor above the first serves only a candidate that passed validation",
  NoFinalTraining: "No training happens after the final floor",
});

// Returns the names of broken rules (empty when all hold).
export function checkRules(state) {
  const scn = scenario(state.scenario);
  const broken = new Set();
  const knowledgeIds = new Set(state.knowledge.map((f) => f.id));
  const lastFloor = scn.generations - 1;
  state.floors.forEach((floor, g) => {
    let rows = 0;
    const seen = new Set();
    for (const e of floor.episodes) {
      if (seen.has(e.episode) || e.agent !== AGENTS[e.episode]) broken.add("NoDoubleContribution");
      seen.add(e.episode);
      rows += e.rows;
      const o = episodeOutcome(scn, g, e.episode);
      const committed = e.status === "rows" || e.status === "rows+fact";
      if (e.rows > 0 && (!committed || e.eligible !== true)) broken.add("RowsOnlyFromEligibleRewrite");
      if (committed && o && !o.signal) broken.add("RowsOnlyFromEligibleRewrite");
      if (e.eligible === true && e.status !== "running" && !committed) broken.add("EligibleRowsKept");
      if (committed && e.rows === 0) broken.add("EligibleRowsKept");
      if (committed && o && e.rows !== o.rows) broken.add("EligibleRowsKept");
      const fid = factId(g, e.episode);
      if (knowledgeIds.has(fid) && e.status !== "rows+fact") broken.add("FactsOnlyAfterCommit");
      if (e.status === "rows+fact" && !knowledgeIds.has(fid)) broken.add("FactsOnlyAfterCommit");
    }
    if (rows !== floor.rowCount) broken.add("NoDoubleContribution");
    if (floor.episodes.length > EPISODES_PER_FLOOR) broken.add("NoDoubleContribution");
    const current = g === state.generation;
    if (floor.published && current && (floor.episodes.length < EPISODES_PER_FLOOR || AGENTS.some((a) => state.agents[a].stage !== "idle")))
      broken.add("NoPublishWhileActive");
    if (floor.published && floor.episodes.some((e) => e.status === "running")) broken.add("NoPublishWhileActive");
    if (floor.training.status !== "none" && !floor.published) broken.add("NoTrainingBeforePublish");
    if (floor.training.base !== null && floor.training.base !== TRAINING_BASE) broken.add("FixedTrainingBase");
    if (floor.training.base !== null) {
      const published = floor.episodes.filter((e) => e.rows > 0).map((e) => e.episode);
      if (!floor.published || JSON.stringify(published) !== JSON.stringify(floor.training.rowsFrom))
        broken.add("TrainsOnPublishedRows");
    }
    if (g === 0 && floor.serving !== servingName(-1)) broken.add("ServeOnlyValidated");
    if (g > 0 && (floor.serving !== servingName(g - 1) || state.floors[g - 1].training.status !== "valid"))
      broken.add("ServeOnlyValidated");
    if (g === lastFloor && floor.training.status !== "none") broken.add("NoFinalTraining");
  });
  if (state.floors.length !== state.generation + 1) broken.add("ServeOnlyValidated");
  if (knowledgeIds.size !== state.knowledge.length) broken.add("NoDoubleContribution");
  if (!scn.graph && state.knowledge.length) broken.add("NoGraphWithoutArchive");
  for (const f of state.knowledge) {
    const e = state.floors[f.floor]?.episodes.find((x) => x.episode === f.episode);
    if (!e || e.status !== "rows+fact" || e.agent !== f.agent) broken.add("FactsOnlyAfterCommit");
    if (f.parents.some((p) => !knowledgeIds.has(p) || compareId(p, f.id) === 0)) broken.add("ContextFromCommittedFacts");
  }
  for (const a of AGENTS) {
    const ag = state.agents[a];
    if (ag.stage === "idle") {
      if (ag.episode !== null || ag.context.length) broken.add("ContextFromCommittedFacts");
      continue;
    }
    if (ag.context.some((p) => !knowledgeIds.has(p))) broken.add("ContextFromCommittedFacts");
    if (knowledgeIds.has(factId(state.generation, ag.episode))) broken.add("FactsOnlyAfterCommit");
    const e = state.floors[state.generation].episodes.find((x) => x.episode === ag.episode);
    if (!e || e.status !== "running" || e.agent !== a) broken.add("NoDoubleContribution");
  }
  return [...broken].sort();
}

// ---------- TLA+ projection and chart labels ----------

const unFact = (id) => {
  const m = /^F(\d+)\.(\d+)$/.exec(id);
  return [Number(m[1]), Number(m[2])];
};
const servingIndex = (name) => (name === "configured-base" ? -1 : Number(name.replace("adapter-from-floor-", "")));

// Projects a JS state onto the TLA+ variables, as plain JSON in the same shape
// that `verify/tla-value.mjs` produces from TLC output. Sets become sorted
// arrays; functions on 0..N become arrays indexed by floor or episode.
export function toModelVars(state) {
  const scn = scenario(state.scenario);
  const floors = Array.from({ length: scn.generations }, (_, g) => state.floors[g] ?? null);
  const sortJson = (xs) => xs.map((x) => JSON.stringify(x)).sort().map((x) => JSON.parse(x));
  return {
    scn: state.scenario,
    gen: state.generation,
    phase: state.phase,
    stage: Object.fromEntries(AGENTS.map((a) => [a, state.agents[a].stage])),
    ep: Object.fromEntries(AGENTS.map((a) => [a, state.agents[a].episode ?? -1])),
    ctx: Object.fromEntries(AGENTS.map((a) => [a, sortJson(state.agents[a].context.map(unFact))])),
    facts: sortJson(state.knowledge.map((f) => ({ id: unFact(f.id), parents: sortJson(f.parents.map(unFact)) }))),
    episodes: floors.map((f) =>
      Array.from({ length: EPISODES_PER_FLOOR }, (_, k) => {
        const e = f?.episodes.find((x) => x.episode === k);
        const verdict = e?.eligible == null ? "pending" : e.eligible ? "pass" : "fail";
        return e ? { rows: e.rows, status: e.status, verdict } : { rows: 0, status: "unissued", verdict };
      }),
    ),
    rowCount: floors.map((f) => f?.rowCount ?? 0),
    published: floors.map((f) => f?.published ?? false),
    serving: floors.map((f) => (f ? servingIndex(f.serving) : -2)),
    training: floors.map((f) => f?.training.status ?? "none"),
    trainBase: floors.map((f) => (f?.training.base ? servingIndex(f.training.base) : -2)),
    trainRows: floors.map((f) => [...(f?.training.rowsFrom ?? [])].sort((x, y) => x - y)),
  };
}

export const STAGE_LABELS = Object.freeze({
  idle: "Idle",
  explore: "Exploring",
  hypothesise: "Hypothesising",
  code: "Coding",
  verify: "Verifying",
  rewrite: "Rewriting",
});

export const PHASE_LABELS = Object.freeze({
  collecting: "Collecting episodes",
  published: "Floor published",
  training: "Training from the base",
  candidate: "Checking the candidate",
  validated: "Candidate validated",
  complete: "Final floor published",
  "trainer-failed": "Trainer failed",
  "artifact-rejected": "Candidate rejected",
});

// Chart regions for TLA+ variables (used by model/view.ts) and for JS states.
export const chartRegionsFromModel = (v) => ({
  "Agent A": STAGE_LABELS[v.stage.A],
  "Agent B": STAGE_LABELS[v.stage.B],
  Floor: PHASE_LABELS[v.phase],
});
export const chartRegions = (state) => chartRegionsFromModel(toModelVars(state));

// ---------- model-derived demos ----------

// Deterministic playback built by applying transition() under a fixed agent
// preference schedule. This is a model-derived demo, not a TLC trace.
export function makeDemo(scenarioId, { schedule } = {}) {
  const scn = scenario(scenarioId);
  const patterns = schedule ?? scn.demoSchedule;
  if (!patterns) throw new Error(`scenario ${scenarioId} has no demo schedule`);
  let state = initialState(scenarioId);
  const steps = [{ index: 0, action: null, label: "Start: floor 1 is serving the configured base", state }];
  let cursor = 0;
  let floorSeen = 0;
  for (let guard = 0; guard < 1000 && !isTerminal(state); guard++) {
    if (state.generation !== floorSeen) {
      floorSeen = state.generation;
      cursor = 0;
    }
    const enabled = enabledActions(state);
    const pattern = patterns[state.generation] ?? "";
    const agentActs = enabled.filter((x) => x.agent);
    let act;
    if (agentActs.length) {
      const pref = pattern[cursor] ?? (cursor % 2 ? "B" : "A");
      cursor += 1;
      act = agentActs.find((x) => x.agent === pref) ?? agentActs[0];
    } else {
      act = enabled[0];
    }
    if (!act) throw new Error(`demo stuck in non-terminal state at step ${steps.length}`);
    state = transition(state, act);
    steps.push({ index: steps.length, action: act.id, label: act.label, state });
  }
  if (!isTerminal(state)) throw new Error("demo did not terminate");
  return {
    scenario: scenarioId,
    kind: "model-derived demo",
    note: "Built by applying the model's transition function under a fixed schedule; not a recorded TLC trace.",
    steps,
  };
}
