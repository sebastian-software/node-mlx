import { defineConfig } from "tsup"

export default defineConfig([
  // Library
  {
    entry: { index: "src/index.ts" },
    format: ["esm", "cjs"],
    dts: true,
    clean: true,
    sourcemap: true,
    shims: true,
    external: ["*.node"],
    esbuildOptions(options) {
      options.logOverride = {
        "empty-import-meta": "silent"
      }
    }
  },
  // CLI
  {
    entry: { cli: "src/cli.ts" },
    format: ["esm"],
    sourcemap: true,
    external: ["*.node"],
    banner: {
      js: "#!/usr/bin/env node"
    },
    esbuildOptions(options) {
      options.logOverride = {
        "empty-import-meta": "silent"
      }
    }
  }
])
