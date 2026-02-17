import Link from "next/link";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import { getAllPostIds, getPostData } from "@/lib/posts";
import { Header } from "@/components/header";
import { CodeBlock } from "@/components/code-block";

export async function generateStaticParams() {
  const paths = getAllPostIds();
  return paths;
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const post = await getPostData(id);

  if (!post) {
    return {
      title: "Post not found",
    };
  }

  return {
    title: `${post.title} | Lei Blog`,
    description: post.excerpt,
  };
}

export default async function PostPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const post = await getPostData(id);

  if (!post) {
    notFound();
  }

  return (
    <div className="min-h-screen">
      <Header currentPath={`/posts/${id}`} />

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className="mb-8 sm:mb-12">
          <Link
            href="/"
            className="text-xs sm:text-sm text-foreground-muted hover:text-accent inline-flex items-center gap-2 font-mono transition-colors"
          >
            <span className="text-accent">←</span> back_to_home
          </Link>
        </div>

        <article>
          {/* Post Header */}
          <header className="mb-8 sm:mb-12">
            <h1 className="text-2xl sm:text-4xl font-bold mb-4 text-heading">{post.title}</h1>
            <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-xs sm:text-sm text-foreground-muted font-mono">
              <span>{post.date}</span>
              <span className="text-border">·</span>
              <span>{post.readTime} read</span>
              <span className="text-border hidden sm:inline">·</span>
              <div className="flex flex-wrap gap-2">
                {post.tags.map((tag) => (
                  <span
                    key={tag}
                    className="tag text-xs"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          </header>

          {/* Post Content */}
          <div className="prose prose-sm sm:prose-lg max-w-none prose-headings:font-semibold prose-headings:text-heading prose-h1:text-2xl sm:prose-h1:text-3xl prose-h1:mb-4 prose-h2:text-xl sm:prose-h2:text-2xl prose-h2:mt-6 sm:prose-h2:mt-8 prose-h2:mb-3 prose-h3:text-lg sm:prose-h3:text-xl prose-h3:mt-4 sm:prose-h3:mt-6 prose-h3:mb-2 prose-p:text-foreground-muted prose-p:leading-relaxed prose-p:mb-4 prose-a:text-link prose-a:no-underline hover:prose-a:underline prose-strong:text-foreground prose-strong:font-semibold prose-code:text-accent prose-code:bg-code-bg prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none prose-blockquote:border-l-accent prose-blockquote:bg-card-bg prose-blockquote:py-3 sm:prose-blockquote:py-4 prose-blockquote:px-4 sm:prose-blockquote:px-6 prose-blockquote:rounded-r-lg prose-blockquote:not-italic prose-table:w-full prose-table:border-collapse prose-table:text-sm prose-th:bg-card-bg prose-th:p-2 prose-th:text-left prose-th:font-semibold prose-th:text-foreground prose-th:border prose-th:border-border prose-td:p-2 prose-td:border prose-td:border-border prose-td:text-foreground-muted">
            <MDXRemote
              source={post.content}
              options={{
                mdxOptions: {
                  remarkPlugins: [remarkGfm],
                },
              }}
              components={{
                pre: CodeBlock,
              }}
            />
          </div>
        </article>

        {/* Post Footer */}
        <footer className="mt-12 sm:mt-16 pt-6 sm:pt-8 border-t border-border">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-sm">
            <Link
              href="/"
              className="text-foreground-muted hover:text-accent font-mono transition-colors"
            >
              ← all_posts
            </Link>
          </div>
        </footer>
      </main>
    </div>
  );
}
