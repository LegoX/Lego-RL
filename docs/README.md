# SWE-Lego-RL docs

User-facing documentation for [SWE-Lego-RL](../README.md), built as a
[fumadocs](https://fumadocs.dev/) (Next.js) site and deployed to Cloudflare
Pages as a static export.

Live site: **<https://swe-lego-rl.pages.dev>**

The prose lives in `content/docs/`; everything else is the minimal app shell
needed to render and deploy it.

## Requirements

Node **>= 20** (Next 16 + fumadocs 16). This host's system Node may be 18, so a
newer Node is installed via [nvm](https://github.com/nvm-sh/nvm). Activate it
before any `npm` command here:

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 22
```

## Develop

```bash
npm install        # first time (also generates the fumadocs .source/)
npm run dev        # dev server at http://localhost:3000 (redirects / -> /docs)
```

## Build & preview the static export

```bash
npm run build      # static export to out/ (next.config.mjs sets output: 'export')
npx serve out      # preview the exported site
```

## Deploy to Cloudflare Pages

```bash
bash deploy_cloudflare_pages.sh
```

Builds and deploys `out/` to the `swe-lego-rl` Cloudflare Pages project
(published at <https://swe-lego-rl.pages.dev>, separate from the training
dashboard's `swe-lego-rl-dashboard`). It activates Node 22 via nvm, asserts
Node >= 20, and reuses the dashboard's Cloudflare credentials
(`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` from `.env.cf` or
`~/.config/rl_dashboard_cloudflare.env`). Override the project with
`PROJECT_NAME=...` (or `DOCS_PROJECT_NAME=...` in the env file).

Two things to know before deploying:

- **`BRANCH_NAME` (default `main`) is the Cloudflare Pages production branch,
  not a git branch.** It only labels the deployment on Cloudflare's side; you do
  not need to check out a matching branch locally, and the script never reads
  your git state. Override it with `BRANCH_NAME=...` (or `DOCS_BRANCH_NAME=...`
  in the env file).
- **The script deploys the working tree, not a commit** (it passes
  `--commit-dirty=true`). Uncommitted edits under `content/docs/` go live like
  any other change, so commit first if you want the published site to correspond
  to something in git.

## Structure

```text
docs/
├── content/docs/          # the documentation (MDX + meta.json page order)
│   ├── meta.json          #   top-level page order
│   ├── index.mdx          #   motivation / landing
│   ├── getting-started.mdx
│   ├── architecture/      #   environment → agent-loop workers → in-process proxy →
│   │                      #   global load balancer → session scheduler → rollouter → trainer
│   ├── core-concepts.mdx
│   ├── run-training/      #   preflight, inference stack, backends, results, scaling
│   ├── dashboard.mdx
│   ├── troubleshooting/   #   startup · training runtime · val & eval scores · cluster & sandbox
│   └── reference/         #   configuration, status
├── src/                   # app shell (docs route, layouts, source loader)
├── public/_redirects      # Cloudflare Pages root redirect (/ -> /docs)
├── next.config.mjs        # createMDX() + output: 'export'
├── source.config.ts       # fumadocs content source (frontmatter/meta schema)
└── deploy_cloudflare_pages.sh
```

## Add or edit a page

1. Add an `.mdx` file under `content/docs/` (or a subfolder) with `title` and
   `description` frontmatter.
2. Add its slug to the folder's `meta.json` `pages` array to place it in the
   sidebar order.
3. Link to other pages by their route, e.g. `/docs/run-training/backends`.

Do **not** list `index` in a folder's `pages` array. A folder's `index.mdx`
becomes the folder's own sidebar link; listing it as well drops that link and
renders the title twice (`Example Usages > Example Usages`).

## Writing conventions

- **Titles** — the shortest noun phrase that names the thing. `Configuration`,
  not `Config Variants`; `2. Site config`, not `2. Describe your cluster once`.
- **Sentences** — reference register: state the fact, the command, or the
  constraint. No narration ("you almost certainly do not need"), no hedges
  ("worth a glance"), no filler ("actually", "simply", "just"). A sentence
  introducing a code block names what the block does and stops.
- **Math** — `$…$` / `$$…$$`, rendered by KaTeX (`remark-math` +
  `rehype-katex`, wired in `source.config.ts`; CSS imported in
  `src/app/layout.tsx`). Use a formula only where it states the rule more
  precisely than a sentence would. This is a manual, not a paper — as of the
  current revision the whole site has two.
- **Terminology follows the technical report** — *run validation*, *agent
  plugin*, *trial* (execution-side) vs *rollout* (training-side), *routing
  replay (R3)*. `Preflight` / `PREFLIGHT_ONLY` are kept because they name the
  actual script surface.
- **American English** (behavior, not behaviour).
- **Cite sources** for concepts from the literature; the reference list lives at
  the bottom of `core-concepts.mdx`.

## llms.txt

`npm run build` regenerates `public/llms.txt` — a machine-readable index of
every page and its description — via `scripts/gen-llms.mjs`, served at
`/llms.txt`. Override the base URL with `DOCS_SITE_URL`.

Build outputs (`node_modules/`, `.next/`, `.source/`, `out/`) are gitignored.
