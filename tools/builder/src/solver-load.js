// Browser-only entry point for loading the wasm caps solver. The virtual module
// is provided by the g2gSolver Vite plugin (vite.config.js): it embeds the blob
// as base64 and instantiates from bytes, or resolves to a null loader when the
// pkg has not been built. Kept separate from solve.js so the node test can import
// solve.js without resolving this Vite-only virtual module.
export { loadSolver } from "virtual:g2g-solver";
