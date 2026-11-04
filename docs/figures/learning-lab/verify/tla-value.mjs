// Reads TLC's printed values and `-dump dot,actionlabels` state graphs into
// plain JSON with the same shape as lab-process.js `toModelVars()`:
// sets become arrays sorted by their JSON text, records become objects with
// sorted keys, and functions on 0..N-1 become arrays.
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

const sortKeys = (o) => Object.fromEntries(Object.keys(o).sort().map((k) => [k, o[k]]));

export function parseValue(text) {
  let i = 0;
  const ws = () => {
    while (i < text.length && /\s/.test(text[i])) i++;
  };
  const eat = (tok) => {
    ws();
    if (!text.startsWith(tok, i)) throw new Error(`expected ${tok} at ${i}: ${text.slice(i, i + 40)}`);
    i += tok.length;
  };
  const peek = (tok) => {
    ws();
    return text.startsWith(tok, i);
  };
  const list = (close, item) => {
    const out = [];
    if (peek(close)) {
      eat(close);
      return out;
    }
    for (;;) {
      out.push(item());
      if (peek(",")) eat(",");
      else if (peek("@@")) eat("@@");
      else break;
    }
    eat(close);
    return out;
  };
  const value = () => {
    ws();
    if (peek("<<")) {
      eat("<<");
      return list(">>", value);
    }
    if (peek("{")) {
      eat("{");
      const xs = list("}", value);
      return xs.map((x) => JSON.stringify(x)).sort().map((x) => JSON.parse(x));
    }
    if (peek("[")) {
      eat("[");
      const pairs = list("]", () => {
        ws();
        const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i));
        if (!m) throw new Error(`expected field at ${i}`);
        i += m[0].length;
        eat("|->");
        return [m[0], value()];
      });
      return sortKeys(Object.fromEntries(pairs));
    }
    if (peek("(")) {
      eat("(");
      const pairs = list(")", () => {
        const k = value();
        eat(":>");
        return [k, value()];
      });
      const keys = pairs.map(([k]) => k);
      if (keys.every((k, n) => k === n)) return pairs.map(([, v]) => v);
      return sortKeys(Object.fromEntries(pairs.map(([k, v]) => [String(k), v])));
    }
    if (peek('"')) {
      const m = /^"((?:[^"\\]|\\.)*)"/.exec(text.slice(i));
      i += m[0].length;
      return JSON.parse(`"${m[1]}"`);
    }
    if (peek("TRUE")) {
      eat("TRUE");
      return true;
    }
    if (peek("FALSE")) {
      eat("FALSE");
      return false;
    }
    const m = /^-?\d+/.exec(text.slice(i));
    if (!m) throw new Error(`unexpected value at ${i}: ${text.slice(i, i + 40)}`);
    i += m[0].length;
    return Number(m[0]);
  };
  const v = value();
  ws();
  if (i !== text.length) throw new Error(`trailing text at ${i}: ${text.slice(i, i + 40)}`);
  return v;
}

// Parses a TLC state ("/\ x = 1\n/\ y = ...") into an object of variables.
export function parseState(text) {
  const vars = {};
  for (const part of text.split(/^\/\\ /m)) {
    if (!part.trim()) continue;
    const eq = part.indexOf(" = ");
    vars[part.slice(0, eq).trim()] = parseValue(part.slice(eq + 3));
  }
  return sortKeys(vars);
}

const unescapeDot = (s) => s.replace(/\\(.)/g, (_, c) => (c === "n" ? "\n" : c));

// Streams a TLC dot dump. Calls onState(fingerprint, vars, isInitial) and
// onEdge(fromFingerprint, toFingerprint, actionName).
export async function readDot(file, { onState, onEdge }) {
  const lines = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
  for await (const line of lines) {
    let m = /^(-?\d+) -> (-?\d+) \[label="([^"]*)"/.exec(line);
    if (m) {
      onEdge(m[1], m[2], m[3]);
      continue;
    }
    m = /^(-?\d+) \[label="((?:[^"\\]|\\.)*)"(.*)\];?$/.exec(line);
    if (m) onState(m[1], parseState(unescapeDot(m[2])), /style\s*=\s*filled/.test(m[3]));
  }
}
