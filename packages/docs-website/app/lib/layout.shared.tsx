import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared"

import logoSvg from "../assets/logo.svg"

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-2">
          <img src={logoSvg} alt="node-mlx" className="w-6 h-6" />
          <span className="font-bold">node-mlx</span>
        </div>
      )
    },
    githubUrl: "https://github.com/sebastian-software/node-mlx",
    links: [
      {
        text: "Documentation",
        url: "https://github.com/sebastian-software/node-mlx#readme",
        external: true
      },
      {
        text: "npm",
        url: "https://www.npmjs.com/package/node-mlx",
        external: true
      }
    ]
  }
}
