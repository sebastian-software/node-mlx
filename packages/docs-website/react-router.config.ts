import type { Config } from "@react-router/dev/config"

export default {
  ssr: false
  // SPA mode - no prerendering needed
  // GitHub Pages will serve index.html for all routes via 404.html redirect
} satisfies Config
