import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    globals: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov"],
      include: ["packages/node-mlx/src/**/*.ts"],
      exclude: ["packages/node-mlx/src/**/*.d.ts"]
    },
    include: ["packages/node-mlx/test/**/*.test.ts"],
    testTimeout: 60000
  }
})
