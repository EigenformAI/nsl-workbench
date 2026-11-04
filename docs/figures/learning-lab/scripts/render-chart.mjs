// Renders checked/fixed-scenarios/chart.mmd to chart.svg with Pele 1.9.0.
// Pele resolves from node_modules (`npm install` here) or from $PELE, a path
// to the installed package's entry module.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = join(dirname(fileURLToPath(import.meta.url)), "..");
const { render } = await import(process.env.PELE ? pathToFileURL(process.env.PELE).href : "pele");
const dir = join(here, process.argv[2] ?? "checked/fixed-scenarios");
const source = readFileSync(join(dir, "chart.mmd"), "utf8");

// Colours follow the statechart skill's suggested Pele theme. `.active` is for
// the figure to highlight the current snapshot's nodes.
const style = `<style>
.pele{--pele-bg:#ffffff;--pele-fg:#1c1c1c;--pele-muted:#656565;--pele-line:#808080;--pele-surface:#fcfcfc;--pele-surface-alt:#f1f1f1;--pele-border:#c9c9c9}
.pele .blue rect{fill:#c3e0ff}.pele .yellow rect{fill:#f0dcbf}.pele .teal rect{fill:#bfe3df}.pele .pink rect{fill:#ffcbe6}
.pele .active rect{stroke:#e36b35!important;stroke-width:4px!important}
.pele .breaks rect{fill:#ffecee!important;stroke:#e72e3b!important;stroke-width:2.5px!important}.pele .breaks text{fill:#cd2834!important}
.pele .mayBreak rect{fill:#fff3e6!important;stroke:#e8730c!important;stroke-width:2.5px!important}.pele .mayBreak text{fill:#b85a06!important}
.pele .bugPath rect{stroke:#1680f3!important;stroke-width:2.5px!important}
.pele .repeats rect{stroke:#9b35d6!important;stroke-width:2.5px!important;stroke-dasharray:6 3}
@media (prefers-color-scheme: dark){.pele{--pele-bg:#1f1f1f;--pele-fg:#f5f5f5;--pele-muted:#acacac;--pele-line:#939393;--pele-surface:#121212;--pele-surface-alt:#2c2c2c;--pele-border:#4e4e4e}
.pele .blue rect{fill:#1b3857}.pele .yellow rect{fill:#573d17}.pele .teal rect{fill:#174641}.pele .pink rect{fill:#57233e}}
</style>`;

let svg;
try {
  ({ svg } = render(source, { limit: 1_000_000, outputLimit: 8_000_000, maxEdges: 10_000 }));
} catch (e) {
  console.error(`Pele could not render ${dir}/chart.mmd: ${e?.name}: ${e?.message}`);
  process.exit(1);
}
svg = svg.replace(/<svg([^>]*)>/, `<svg$1>${style.replace(/\n/g, "")}`);
writeFileSync(join(dir, "chart.svg"), svg);
console.log(`wrote ${join(dir, "chart.svg")} (${svg.length} bytes)`);
