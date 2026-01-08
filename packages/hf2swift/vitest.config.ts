import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    globals: true,
    // Co-located tests next to source files
    include: ["src/**/*.test.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
      exclude: ["src/cli.ts", "src/**/*.test.ts"]
    }
  }
})
