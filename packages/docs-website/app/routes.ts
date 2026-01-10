import { index, type RouteConfig } from "@react-router/dev/routes"

// Docs route temporarily disabled due to fumadocs SSR requirements
// TODO: Re-enable when SPA-compatible docs solution is implemented
export default [index("routes/home.tsx")] satisfies RouteConfig
