import Link from "next/link";
import { notFound } from "next/navigation";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllPostIds, getPostData, getAllPosts } from "@/lib/posts";
import fs from "fs";
import path from "path";
import { ThemeToggle } from "@/components/theme-toggle";
import { CodeBlock } from "@/components/code-block";

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
                  <span className="text-heading">lei_blog</span>
                  {" "}
                  <span className="text-accent">/&gt;</span>
                </h1>
              </Link>
            </div>
            <nav className="flex gap-6 text-sm items-center">
              <Link href="/" className="text-foreground-muted hover:text-accent transition-colors">
                Home
              </Link>
              <Link href="/tags" className="text-foreground-muted hover:text-accent transition-colors">
                Tags
              </Link>
              <Link href="/about" className="text-foreground-muted hover:text-accent transition-colors">
                About
              </Link>
              <a
                href="https://github.com/charles-lei"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground-muted hover:text-accent transition-colors"
                aria-label="GitHub"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
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
          <div className="prose prose-lg max-w-none prose-headings:font-semibold prose-headings:text-heading prose-h1:text-3xl prose-h1:mb-4 prose-h2:text-2xl prose-h2:mt-8 prose-h2:mb-3 prose-h3:text-xl prose-h3:mt-6 prose-h3:mb-2 prose-p:text-foreground-muted prose-p:leading-relaxed prose-p:mb-4 prose-a:text-link prose-a:no-underline hover:prose-a:underline prose-strong:text-foreground prose-strong:font-semibold prose-code:text-accent prose-code:bg-code-bg prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none prose-blockquote:border-l-accent prose-blockquote:bg-card-bg prose-blockquote:py-4 prose-blockquote:px-6 prose-blockquote:rounded-r-lg prose-blockquote:not-italic">
            <MDXRemote
              source={source}
              components={{
                pre: CodeBlock,
              }}
            />
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
