/**
 * Cloudflare Pages Function — proxies wandb API requests.
 *
 * Environment variables (set via wrangler pages secret):
 *   WANDB_API_KEY    — wandb API key
 *   WANDB_ENTITY     — wandb team/user
 *   WANDB_PROJECT    — wandb project name
 */

interface Env {
  WANDB_API_KEY: string;
  WANDB_ENTITY: string;
  WANDB_PROJECT: string;
}

export const onRequestGet: PagesFunction<Env> = async (context) => {
  const { WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT } = context.env;
  const path = (context.params.path as string[]).join("/");

  if (path === "config") {
    return Response.json({
      log_dir: "",
      wandb_entity: WANDB_ENTITY,
      wandb_project: WANDB_PROJECT,
      data_source: "wandb",
    });
  }

  if (path === "runs") {
    const resp = await fetch(
      `https://api.wandb.ai/api/v1/${WANDB_ENTITY}/${WANDB_PROJECT}/runs?per_page=50`,
      { headers: { Authorization: `Bearer ${WANDB_API_KEY}` } },
    );
    const data = await resp.json() as any;
    const runs = (Array.isArray(data) ? data : data.runs || data.data || []).map(
      (r: any) => ({
        id: r.id || r.name,
        name: r.displayName || r.name,
        state: r.state || "unknown",
        created_at: r.createdAt || "",
        source: "wandb",
      }),
    );
    return Response.json(runs);
  }

  const historyMatch = path.match(/^runs\/([^/]+)\/metrics$/);
  if (historyMatch) {
    const runId = historyMatch[1];
    const resp = await fetch(
      `https://api.wandb.ai/api/v1/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}/history?samples=1500`,
      { headers: { Authorization: `Bearer ${WANDB_API_KEY}` } },
    );
    const raw = await resp.json() as any;
    const rows = Array.isArray(raw) ? raw : raw.history || raw.data || [];
    const metrics: Record<string, any>[] = [];
    const allKeys = new Set<string>();
    for (const row of rows) {
      const point: Record<string, any> = {};
      for (const [k, v] of Object.entries(row)) {
        if (k.startsWith("_") && k !== "_step") continue;
        if (typeof v === "number" && v === v) {
          const key = k === "_step" ? "step" : k;
          point[key] = v;
          if (key !== "step") allKeys.add(key);
        }
      }
      if (Object.keys(point).length > 0) metrics.push(point);
    }
    return Response.json({
      run: { id: runId, name: runId, state: "unknown", created_at: "", source: "wandb" },
      metrics,
      available_keys: [...allKeys].sort(),
    });
  }

  const latestMatch = path.match(/^runs\/([^/]+)\/latest$/);
  if (latestMatch) {
    const runId = latestMatch[1];
    const resp = await fetch(
      `https://api.wandb.ai/api/v1/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}/history?samples=1`,
      { headers: { Authorization: `Bearer ${WANDB_API_KEY}` } },
    );
    const raw = await resp.json() as any;
    const rows = Array.isArray(raw) ? raw : raw.history || raw.data || [];
    return Response.json(rows.length > 0 ? rows[rows.length - 1] : null);
  }

  const keysMatch = path.match(/^runs\/([^/]+)\/keys$/);
  if (keysMatch) {
    const runId = keysMatch[1];
    const resp = await fetch(
      `https://api.wandb.ai/api/v1/${WANDB_ENTITY}/${WANDB_PROJECT}/runs/${runId}/history?samples=500`,
      { headers: { Authorization: `Bearer ${WANDB_API_KEY}` } },
    );
    const raw = await resp.json() as any;
    const rows = Array.isArray(raw) ? raw : raw.history || raw.data || [];
    const allKeys = new Set<string>();
    for (const row of rows) {
      for (const k of Object.keys(row)) {
        if (!k.startsWith("_")) allKeys.add(k);
      }
    }
    return Response.json([...allKeys].sort());
  }

  const logsMatch = path.match(/^runs\/([^/]+)\/logs$/);
  if (logsMatch) {
    return Response.json(
      { lines: ["Log streaming is only available when using the local server backend."], total_lines: 1, offset: 0 },
    );
  }

  return new Response("Not found", { status: 404 });
};
