import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

// Resolve `virtual:g2g-solver` to a module that embeds the wasm caps-solver blob
// (built by build-wasm.sh into src/wasm, gitignored) as base64 and instantiates
// it from those bytes. Instantiating from bytes rather than fetching a sibling
// .wasm is what lets the real solver run in the single-file bundle and the
// published artifact under a strict CSP. When the pkg has not been built the
// module resolves to a null loader and the builder falls back to the family
// heuristic. Bundled (not fetched) so it works identically in dev, build, and
// artifact; keeping it a virtual module means an unbuilt pkg never breaks the
// build (unlike a static import of a maybe-absent file).
function g2gSolver() {
  const virtualId = "virtual:g2g-solver";
  const resolvedId = "\0" + virtualId;
  const glue = resolve(here, "src/wasm/g2g_validate.js");
  const wasm = resolve(here, "src/wasm/g2g_validate_bg.wasm");
  return {
    name: "g2g-solver",
    resolveId(id) {
      if (id === virtualId) return resolvedId;
    },
    load(id) {
      if (id !== resolvedId) return undefined;
      if (!existsSync(glue) || !existsSync(wasm)) {
        return "export async function loadSolver() { return null; }";
      }
      const b64 = readFileSync(wasm).toString("base64");
      return `
import init, { validate_pipeline } from ${JSON.stringify(glue)};
const B64 = ${JSON.stringify(b64)};
function toBytes(s) {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) u8[i] = bin.charCodeAt(i);
  return u8;
}
let ready;
export async function loadSolver() {
  try {
    if (!ready) ready = init({ module_or_path: toBytes(B64) }).then(() => validate_pipeline);
    return await ready;
  } catch {
    return null;
  }
}
`;
    },
  };
}

// `base: "./"` keeps asset + registry.json references relative, so the built page
// works served from any directory. `viteSingleFile` inlines JS/CSS (including the
// embedded solver base64) into one index.html; registry.json stays a sibling
// fetched at runtime, so `dist` is a self-contained bundle plus that snapshot.
export default defineConfig({
  base: "./",
  plugins: [g2gSolver(), react(), viteSingleFile()],
});
