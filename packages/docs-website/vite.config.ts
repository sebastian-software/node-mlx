import { reactRouter } from "@react-router/dev/vite"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"
import tsconfigPaths from "vite-tsconfig-paths"
import mdx from "fumadocs-mdx/vite"
import * as MdxConfig from "./source.config"

export default defineConfig({
  base: process.env.BASE_PATH || "/",
  plugins: [
    tsconfigPaths({
      projects: ["./tsconfig.json"]
    }),
    mdx(MdxConfig),
    tailwindcss(),
    reactRouter()
  ]
})
