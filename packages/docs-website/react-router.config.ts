import type { Config } from "@react-router/dev/config"

// Strip trailing slash from BASE_PATH if present
const basePath = process.env.BASE_PATH?.replace(/\/$/, "")

export default {
  ssr: false,
  // Set basename for GitHub Pages subpath deployment
  ...(basePath ? { basename: basePath } : {})
} satisfies Config
