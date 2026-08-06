// Generates public/llms.txt — a machine-readable index of every docs page
// (llmstxt.org convention) — from content/docs/**/meta.json + mdx frontmatter.
// Runs as part of `npm run build`.
import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CONTENT = join(ROOT, "content", "docs");
const SITE = process.env.DOCS_SITE_URL || "https://lego-rl.pages.dev";

function frontmatter(file) {
  const src = readFileSync(file, "utf8");
  const m = src.match(/^---\n([\s\S]*?)\n---/);
  const fm = {};
  if (m) {
    for (const line of m[1].split("\n")) {
      const kv = line.match(/^(\w+):\s*(.*)$/);
      if (kv) fm[kv[1]] = kv[2].trim();
    }
  }
  return fm;
}

function pageLine(dir, slug, urlBase) {
  const file = join(dir, `${slug}.mdx`);
  if (!existsSync(file)) return null;
  const fm = frontmatter(file);
  const url = slug === "index" ? urlBase : `${urlBase}/${slug}`;
  const desc = fm.description ? `: ${fm.description}` : "";
  return `- [${fm.title || slug}](${url})${desc}`;
}

const rootMeta = JSON.parse(readFileSync(join(CONTENT, "meta.json"), "utf8"));
const lines = [
  "# Lego-RL Documentation",
  "",
  "> Lego-RL trains real coding agents with online reinforcement learning on real repositories: the agent drives the same model it is training, a verifier grades the result, and the trainer updates the weights.",
  "",
];

for (const entry of rootMeta.pages) {
  const dir = join(CONTENT, entry);
  if (existsSync(dir) && statSync(dir).isDirectory()) {
    const meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
    lines.push(`## ${meta.title || entry}`, "");
    for (const slug of meta.pages) {
      if (slug.startsWith("---")) continue; // sidebar separators
      const line = pageLine(dir, slug, `${SITE}/docs/${entry}`);
      if (line) lines.push(line);
    }
    lines.push("");
  } else {
    const line = pageLine(CONTENT, entry, `${SITE}/docs`);
    if (line) lines.push(line, "");
  }
}

const out = join(ROOT, "public", "llms.txt");
writeFileSync(out, lines.join("\n"));
console.log(`[gen-llms] wrote ${out}`);
