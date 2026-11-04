// Confirms the behaviour checks reject credible wrong implementations.
// Each mutant edits one rule out of a copy of lab-process.js; the test suite
// must fail against every mutant and pass against the original.
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = readFileSync(join(here, "lab-process.js"), "utf8");
const testFile = join(here, "verify", "lab-process.test.mjs");

const MUTANTS = [
  ["double contribution: commit leaves the agent rewriting",
    `      s.agents[act.agent] = idleAgent();\n      break;\n    }\n    case "publish":`,
    `      break;\n    }\n    case "publish":`],
  ["publication with active agents",
    "if (settled && floor.episodes.length === AGENTS.length) {",
    "if (floor.episodes.length === AGENTS.length) {"],
  ["training before publication",
    `} else if (state.phase === "published") {`,
    `}\n  if (state.phase === "published" || (state.phase === "collecting" && state.generation < scenario(state.scenario).generations - 1)) {`],
  ["serving an unvalidated candidate",
    `} else if (state.phase === "validated") {`,
    `}\n  if (state.phase === "validated" || state.phase === "candidate") {`],
  ["rejected candidate still validated",
    `        s.phase = "artifact-rejected";`,
    `        s.phase = "validated";`],
  ["training after the final floor",
    `s.phase = s.generation === scn.generations - 1 ? "complete" : "published";`,
    `s.phase = "published";`],
  ["facts without a shared archive",
    "const admits = (a) => (scn.graph ? a : false);",
    "const admits = (a) => a;"],
  ["rows only for admitted rewrites (graph/row conflation)",
    "      rec.rows = act.rows;\n      floor.rowCount += act.rows;",
    "      rec.rows = act.admit ? act.rows : 0;\n      floor.rowCount += rec.rows;"],
  ["fact visible at verification, before commit",
    `        ag.stage = "rewrite";`,
    `        ag.stage = "rewrite";\n        s.knowledge.push({ id: factId(s.generation, ag.episode), floor: s.generation, episode: ag.episode, agent: act.agent, parents: [] });`],
  ["agents read knowledge continuously",
    `        s.knowledge.sort((x, y) => compareId(x.id, y.id));`,
    `        s.knowledge.sort((x, y) => compareId(x.id, y.id));\n        for (const b of AGENTS) if (b !== act.agent && s.agents[b].stage !== "idle") s.agents[b].context = s.knowledge.map((f) => f.id);`],
  ["training continues from the serving adapter",
    "          base: TRAINING_BASE, // always the fixed configured base, never the serving adapter",
    "          base: floor.serving,"],
  ["eligibility revealed before verification",
    `    eligible: ag.stage === "rewrite" ? true : null,`,
    `    eligible: ag.stage === "idle" ? null : episodeOutcome(scenario(state.scenario), state.generation, ag.episode)?.signal ?? null,`],
];

const runTests = (modulePath) =>
  spawnSync(process.execPath, ["--test", testFile], {
    env: { ...process.env, LAB_PROCESS: modulePath },
    encoding: "utf8",
    timeout: 60_000,
  });

const dir = mkdtempSync(join(tmpdir(), "lab-mutants-"));
let ok = true;
try {
  const base = runTests(join(here, "lab-process.js"));
  console.log(`original: ${base.status === 0 ? "tests pass" : "TESTS FAIL"}`);
  if (base.status !== 0) ok = false;
  MUTANTS.forEach(([name, from, to], i) => {
    if (source.split(from).length !== 2) {
      console.log(`mutant ${i + 1} (${name}): target text not found exactly once`);
      ok = false;
      return;
    }
    const file = join(dir, `mutant-${i + 1}.js`);
    writeFileSync(file, source.replace(from, to));
    const r = runTests(file);
    const failing = [...new Set([...r.stdout.matchAll(/^✖ (.+?) \(\d/gm)].map((m) => m[1]).filter((t) => t !== "failing tests:"))];
    const rejected = r.status !== 0 && failing.length > 0;
    console.log(`mutant ${i + 1} (${name}): ${rejected ? `rejected by ${failing.length} check(s), e.g. "${failing[0]}"` : "NOT REJECTED"}`);
    if (!rejected) ok = false;
  });
} finally {
  rmSync(dir, { recursive: true, force: true });
}
process.exit(ok ? 0 : 1);
