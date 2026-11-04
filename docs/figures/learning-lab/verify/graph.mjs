// Breadth-first exploration of the JavaScript process model.
import { createHash } from "node:crypto";
import { enabledActions, initialState, toModelVars, transition } from "../lab-process.js";

export const stable = (v) =>
  Array.isArray(v)
    ? `[${v.map(stable).join(",")}]`
    : v && typeof v === "object"
      ? `{${Object.keys(v).sort().map((k) => `${JSON.stringify(k)}:${stable(v[k])}`).join(",")}}`
      : JSON.stringify(v);

export const keyOf = (vars) => createHash("sha1").update(stable(vars)).digest("hex");

// Returns { states: Map(key -> state), edges: Set("from|action|to"), initial: Set(key) }.
export function explore(scenarioIds, { keepStates = true } = {}) {
  const states = new Map();
  const edges = new Set();
  const initial = new Set();
  const queue = [];
  for (const id of scenarioIds) {
    const s = initialState(id);
    const k = keyOf(toModelVars(s));
    initial.add(k);
    if (!states.has(k)) {
      states.set(k, keepStates ? s : null);
      queue.push([k, s]);
    }
  }
  while (queue.length) {
    const [k, s] = queue.pop();
    for (const a of enabledActions(s)) {
      const t = transition(s, a);
      const tk = keyOf(toModelVars(t));
      edges.add(`${k}|${a.tla}|${tk}`);
      if (!states.has(tk)) {
        states.set(tk, keepStates ? t : null);
        queue.push([tk, t]);
      }
    }
  }
  return { states, edges, initial };
}
