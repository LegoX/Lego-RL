/**
 * Static mode — serve the dashboard from a pre-exported bundle, with no backend.
 *
 * `export_static.py` walks a run's read-only GET endpoints against a live
 * server.py and writes each response to `data/<slug>.json`, plus a manifest.
 * Here we install a `window.fetch` shim that maps the same `/api/...` URLs onto
 * those files, so every panel keeps calling `fetch("/api/runs/<id>/metrics")`
 * unchanged and none of the ~15 call sites need to know where the data lives.
 *
 * The shim is a no-op when `data/manifest.json` is absent, which is the case for
 * a normal `bash webui/start_dashboard.sh` — one build serves both modes.
 */

const API = "/api/";

/** FNV-1a, 32-bit. Kept byte-identical to `_slug_hash` in export_static.py. */
function fnv1a(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/**
 * `/api/runs/foo/metrics?keys=b,a` -> `runs_foo_metrics__<hash>`.
 *
 * Query keys are sorted so the slug does not depend on the order a caller
 * happened to build its URLSearchParams in; the hash carries the values, which
 * would otherwise make filenames unbounded.
 */
export function apiSlug(pathWithQuery: string): string {
  const [rawPath, rawQuery = ""] = pathWithQuery.split("?");
  const path = rawPath.replace(/^\/api\//, "").replace(/\/+$/, "");
  const params = [...new URLSearchParams(rawQuery).entries()].sort(([a], [b]) =>
    a < b ? -1 : a > b ? 1 : 0,
  );
  const canonical = path + (params.length ? "?" + params.map(([k, v]) => `${k}=${v}`).join("&") : "");
  const readable = path.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80);
  return `${readable}__${fnv1a(canonical)}`;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Probe for the export bundle and, if present, route /api through it.
 * Returns whether static mode was installed.
 */
export async function installStaticMode(): Promise<boolean> {
  const base = import.meta.env.BASE_URL || "/";
  const dataRoot = `${base.replace(/\/+$/, "")}/data/`;

  // server.py answers unknown paths with index.html (SPA fallback), so a live
  // deployment returns 200 + HTML here rather than 404. Content-type and shape
  // are both checked so that never reads as a manifest.
  let manifest: { run_id?: string; files?: string[] } | null = null;
  try {
    const res = await fetch(`${dataRoot}manifest.json`, { cache: "no-store" });
    if (res.ok && (res.headers.get("content-type") ?? "").includes("json")) {
      const parsed = await res.json();
      if (parsed && Array.isArray(parsed.files)) manifest = parsed;
    }
  } catch {
    // Absent, or HTML that failed to parse: live deployment, leave fetch alone.
  }
  if (!manifest) return false;

  const available = new Set(manifest.files ?? []);
  const realFetch = window.fetch.bind(window);

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const idx = url.indexOf(API);
    if (idx === -1) return realFetch(input as RequestInfo, init);

    const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
    if (method !== "GET") {
      // Ingest, delete and AI-analysis generation all mutate server state.
      return jsonResponse(
        { error: "This is a static export of one run — that action needs the live dashboard." },
        405,
      );
    }

    const slug = apiSlug(url.slice(idx));
    if (!available.has(slug)) {
      // Panels already render an empty state on a non-ok response; returning 404
      // here is what makes an unexported endpoint degrade instead of hang.
      return jsonResponse({ error: "not included in this static export" }, 404);
    }
    return realFetch(`${dataRoot}${slug}.json`, { cache: "force-cache" });
  };

  return true;
}
