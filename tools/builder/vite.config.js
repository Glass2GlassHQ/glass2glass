import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// `base: "./"` keeps asset + registry.json references relative, so the built
// page works served from any directory. `viteSingleFile` inlines JS/CSS into one
// index.html (registry.json stays a sibling file, fetched at runtime), so `dist`
// is a self-contained bundle plus the registry snapshot.
export default defineConfig({
  base: "./",
  plugins: [react(), viteSingleFile()],
});
