import type { Config } from "@react-router/dev/config"
import { glob } from "node:fs/promises"
import { createGetUrl, getSlugs } from "fumadocs-core/source"

const getUrl = createGetUrl("/docs")
const basePath = process.env.BASE_PATH?.replace(/\/$/, "") || ""

export default {
  ssr: false,
  // Only set basename if BASE_PATH is provided (for production builds)
  ...(basePath ? { basename: basePath } : {}),
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
