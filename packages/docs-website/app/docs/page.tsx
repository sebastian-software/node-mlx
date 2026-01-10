import { useParams } from "react-router"
import browserCollections from "fumadocs-mdx:collections/browser"
import { DocsLayout } from "fumadocs-ui/layouts/docs"
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from "fumadocs-ui/layouts/docs/page"
import defaultMdxComponents from "fumadocs-ui/mdx"

import { baseOptions } from "../lib/layout.shared"

// Static page tree for navigation (generated from content structure)
const pageTree = {
  name: "Documentation",
  children: [
    {
      type: "page" as const,
      name: "Getting Started",
      url: "/docs"
    },
    {
      type: "folder" as const,
      name: "Models",
      children: [
        {
          type: "page" as const,
          name: "Supported Models",
          url: "/docs/models"
        }
      ]
    }
  ]
}

const clientLoader = browserCollections.docs.createClientLoader({
  component({ toc, default: Mdx, frontmatter }) {
    return (
      <DocsPage toc={toc}>
        <title>{frontmatter.title}</title>
        <meta name="description" content={frontmatter.description} />
        <DocsTitle>{frontmatter.title}</DocsTitle>
        <DocsDescription>{frontmatter.description}</DocsDescription>
        <DocsBody>
          <Mdx components={{ ...defaultMdxComponents }} />
        </DocsBody>
      </DocsPage>
    )
  }
})

// Map URL paths to content paths
function getContentPath(slugs: string[]): string {
  if (slugs.length === 0) return "/docs"
  return `/docs/${slugs.join("/")}`
}

export default function Page() {
  const params = useParams()
  const slugs = (params["*"] || "").split("/").filter((v) => v.length > 0)
  const contentPath = getContentPath(slugs)

  const Content = clientLoader.getComponent(contentPath)

  return (
    <DocsLayout {...baseOptions()} tree={pageTree}>
      <Content />
    </DocsLayout>
  )
}
