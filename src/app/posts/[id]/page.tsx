import Link from "next/link";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllPostIds, getPostData, getAllPosts } from "@/lib/posts";
import fs from "fs";
import path from "path";
import { ThemeToggle } from "@/components/theme-toggle";

const postsDirectory = path.join(process.cwd(), "src/content/posts");

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
    title: `${post.title} | Geek Blog`,
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

  const fullPath = path.join(postsDirectory, `${id}.mdx`);
  const source = fs.readFileSync(fullPath, "utf8");

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div>
              <Link href="/">
                <h1 className="text-2xl font-bold font-mono tracking-tight">
                  <span className="text-accent">&lt;</span>
                  {" "}
                  <span className="text-heading">geek_blog</span>
                  {" "}
                  <span className="text-accent">/&gt;</span>
                </h1>
              </Link>
            </div>
            <nav className="flex gap-6 text-sm items-center">
              <Link href="/" className="text-foreground-muted hover:text-accent transition-colors">
                home
              </Link>
              <Link href="/tags" className="text-foreground-muted hover:text-accent transition-colors">
                tags
              </Link>
              <Link href="/about" className="text-foreground-muted hover:text-accent transition-colors">
                about
              </Link>
              <a
                href="https://github.com/charles-lei"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground-muted hover:text-accent transition-colors"
              >
                github
              </a>
              <ThemeToggle />
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-3xl mx-auto px-6 py-12">
        <div className="mb-12">
          <Link
            href="/"
            className="text-sm text-foreground-muted hover:text-accent inline-flex items-center gap-2 font-mono transition-colors"
          >
            <span className="text-accent">←</span> back_to_home
          </Link>
        </div>

        <article>
          {/* Post Header */}
          <header className="mb-12">
            <h1 className="text-4xl font-bold mb-4 text-heading">{post.title}</h1>
            <div className="flex items-center gap-4 text-sm text-foreground-muted font-mono">
              <span>{post.date}</span>
              <span className="text-border">·</span>
              <span>{post.readTime} read</span>
              <span className="text-border">·</span>
              <div className="flex gap-2">
                {post.tags.map((tag) => (
                  <span
                    key={tag}
                    className="tag"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          </header>

          {/* Post Content */}
          <div className="prose prose-lg max-w-none prose-headings:font-semibold prose-headings:text-heading prose-h1:text-3xl prose-h1:mb-4 prose-h2:text-2xl prose-h2:mt-8 prose-h2:mb-3 prose-h3:text-xl prose-h3:mt-6 prose-h3:mb-2 prose-p:text-foreground-muted prose-p:leading-relaxed prose-p:mb-4 prose-a:text-link prose-a:no-underline hover:prose-a:underline prose-strong:text-foreground prose-strong:font-semibold prose-code:text-accent prose-code:bg-code-bg prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none prose-pre:bg-code-bg prose-pre:border prose-pre:border-border-subtle prose-pre:rounded-lg prose-pre:px-4 prose-pre:py-3 prose-blockquote:border-l-accent prose-blockquote:bg-card-bg prose-blockquote:py-4 prose-blockquote:px-6 prose-blockquote:rounded-r-lg prose-blockquote:not-italic">
            <MDXRemote source={source} />
          </div>
        </article>

        {/* Post Footer */}
        <footer className="mt-16 pt-8 border-t border-border">
          <div className="flex items-center justify-between text-sm">
            <Link
              href="/"
              className="text-foreground-muted hover:text-accent font-mono transition-colors"
            >
              ← all_posts
            </Link>
            <div className="flex gap-4">
              <a
                href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(
                  post.title
                )}&url=${encodeURIComponent(
                  `https://yourdomain.com/posts/${post.id}`
                )}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground-muted hover:text-accent font-mono transition-colors"
              >
                share
              </a>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
