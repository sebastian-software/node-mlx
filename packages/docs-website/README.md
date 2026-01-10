# @node-mlx/docs

Documentation website for node-mlx, built with [Fumadocs](https://fumadocs.dev) + React Router.

## Development

```bash
# Install dependencies
pnpm install

# Start dev server
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173) to view the documentation.

## Build

```bash
# Build for production
pnpm build

# Preview the build locally
pnpm start
```

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to `main`.

- **URL:** https://sebastian-software.github.io/node-mlx/
- **Workflow:** `.github/workflows/docs.yml`

## Stack

- **Framework:** [Fumadocs](https://fumadocs.dev) with React Router
- **Bundler:** Vite 7
- **Styling:** Tailwind CSS 4
- **API Docs:** TypeDoc
