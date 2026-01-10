import { useMemo } from "react"
import { useParams } from "react-router"
import browserCollections from "fumadocs-mdx:collections/browser"
import { DocsLayout } from "fumadocs-ui/layouts/docs"
import { DocsBody, DocsDescription, DocsPage, DocsTitle } from "fumadocs-ui/layouts/docs/page"
import defaultMdxComponents from "fumadocs-ui/mdx"

import { baseOptions } from "../lib/layout.shared"
import { source } from "../lib/source"

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

export default function Page() {
  const params = useParams()
  const slugs = (params["*"] || "").split("/").filter((v) => v.length > 0)

  const page = useMemo(() => source.getPage(slugs), [slugs.join("/")])

  if (!page) {
    return (
      <DocsLayout {...baseOptions()} tree={source.pageTree}>
        <DocsPage>
          <DocsTitle>Page Not Found</DocsTitle>
          <DocsBody>
            <p>The requested documentation page could not be found.</p>
          </DocsBody>
        </DocsPage>
      </DocsLayout>
    )
  }

  const Content = clientLoader.getComponent(page.path)

  return (
    <DocsLayout {...baseOptions()} tree={source.pageTree}>
      <Content />
    </DocsLayout>
  )
}
