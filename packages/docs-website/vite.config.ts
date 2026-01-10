import { reactRouter } from "@react-router/dev/vite"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"
import tsconfigPaths from "vite-tsconfig-paths"
import mdx from "fumadocs-mdx/vite"
import * as MdxConfig from "./source.config"

// Ensure BASE_PATH ends with a slash for correct asset URL construction
const basePath = process.env.BASE_PATH
  ? process.env.BASE_PATH.endsWith("/")
    ? process.env.BASE_PATH
    : `${process.env.BASE_PATH}/`
  : "/"

export default defineConfig({
  base: basePath,
  plugins: [
    tsconfigPaths({
      projects: ["./tsconfig.json"]
    }),
    mdx(MdxConfig),
    tailwindcss(),
    reactRouter()
  ]
})
