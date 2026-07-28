import { source } from "@/lib/source";
import { createFromSource } from "fumadocs-core/search/server";

// Static export: the index is generated at build time and served as a file,
// so search works without a server (output: 'export').
export const revalidate = false;

export const { staticGET: GET } = createFromSource(source);
