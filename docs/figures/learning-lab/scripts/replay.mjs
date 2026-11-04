// Prints each scenario's model-derived demo as numbered reader-word steps.
//
//   node scripts/replay.mjs            print every demo
//   node scripts/replay.mjs materials  print one demo
//   node scripts/replay.mjs --check    compare with checked/demos.txt
//
// The demos come from makeDemo() in lab-process.js. They are built from the
// model's transition function under a fixed schedule, not recorded TLC traces.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AGENTS, STAGE_LABELS, makeDemo, scenario, scenarioIds, summary } from "../lab-process.js";

const here = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const check = args.includes("--check");
const wanted = args.filter((a) => a !== "--check");
const ids = wanted.length ? wanted : scenarioIds().filter((id) => scenario(id).demoSchedule);

const lines = [];
for (const id of ids) {
  const demo = makeDemo(id);
  lines.push(`# ${id}: ${scenario(id).title} (${demo.kind}, ${demo.steps.length - 1} steps)`);
  for (const { index, label, state } of demo.steps) {
    const s = summary(state);
    const agents = AGENTS.map((a) => `${a}:${STAGE_LABELS[state.agents[a].stage]}`).join(" ");
    lines.push(
      `${String(index).padStart(2)}. ${label}\n    floor ${s.generation + 1}/${s.floorsOpen} ${s.phase} | ${agents} | ` +
        `rows ${s.rowsThisFloor} (all ${s.rowsAllFloors}) | facts ${s.facts} | serving ${s.serving} | candidate ${s.candidate}`,
    );
  }
  lines.push("");
}
const text = lines.join("\n");
if (check) {
  const saved = readFileSync(join(here, "checked", "demos.txt"), "utf8");
  if (saved !== text) {
    console.error("checked/demos.txt is out of date; run: node scripts/replay.mjs > checked/demos.txt");
    process.exit(1);
  }
  console.log("checked/demos.txt matches makeDemo()");
} else {
  process.stdout.write(text);
}
