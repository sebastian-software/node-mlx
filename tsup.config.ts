import { defineConfig } from "tsup"

export default defineConfig({
  entry: {
    index: "packages/node-mlx/src/index.ts"
  },
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
})
