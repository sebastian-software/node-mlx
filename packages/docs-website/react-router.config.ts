import type { Config } from "@react-router/dev/config"
import { glob } from "node:fs/promises"
import { createGetUrl, getSlugs } from "fumadocs-core/source"

const getUrl = createGetUrl("/docs")

export default {
  ssr: false,
  // Note: Don't use basename - it puts HTML files in a subdirectory
  // which breaks asset paths. Use Vite's base config instead.
  async prerender({ getStaticPaths }) {
    const paths: string[] = []

    for (const path of getStaticPaths()) {
      paths.push(path)
    }

    for await (const entry of glob("**/*.mdx", { cwd: "content/docs" })) {
      // Normalize Windows paths (backslashes) to forward slashes
      const normalizedEntry = entry.replace(/\\/g, "/")
      paths.push(getUrl(getSlugs(normalizedEntry)))
    }

    return paths
  }
} satisfies Config
