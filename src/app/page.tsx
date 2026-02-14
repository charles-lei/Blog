import Link from "next/link";
import { getAllPosts } from "@/lib/posts";
import { ThemeToggle } from "@/components/theme-toggle";

export default async function HomePage() {
  const posts = getAllPosts();

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
              <p className="text-foreground-muted mt-1 text-sm font-mono">
                thoughts.on(&quot;technology&quot;);
              </p>
            </div>
            <nav className="flex gap-6 text-sm items-center">
              <Link href="/" className="text-accent font-medium">
                Home
              </Link>
              <Link href="/tags" className="text-foreground-muted hover:text-accent transition-colors">
                Tags
              </Link>
              <Link href="/about" className="text-foreground-muted hover:text-accent transition-colors">
                About
              </Link>
              <a
                href="https://github.com"
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
      <main className="max-w-4xl mx-auto px-6 py-12 flex-1">
        {/* Hero Section */}
        <section className="mb-16">
          <div className="accent-border-left">
            <p className="text-foreground-muted font-mono text-sm mb-2">
              <span className="text-accent">const</span>{" "}
              <span className="text-accent-secondary">introduction</span> = {"{"}
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              role: <span className="text-accent-tertiary">&quot;developer&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              passion: <span className="text-accent-tertiary">&quot;building things&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              coffee: <span className="text-accent-tertiary">true</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm">{"};"}</p>
            <p className="text-foreground-muted mt-6 text-lg leading-relaxed">
              Welcome to my corner of the internet. I write about code,
              systems, and the craft of software engineering.
            </p>
          </div>
        </section>

        {/* Blog Posts */}
        <section>
          <h2 className="text-sm font-mono text-foreground-muted mb-8 flex items-center gap-2">
            <span className="text-accent">{"//"}</span> latest_posts
          </h2>

          {posts.length === 0 ? (
            <div className="border border-border-subtle rounded-lg p-6 bg-card-bg">
              <p className="text-foreground-muted text-center font-mono text-sm">
                <span className="text-accent">{"/*"}</span> No posts yet. Add some
                MDX files to src/content/posts/
                <span className="text-accent">{" */"}</span>
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {posts.map((post, index) => (
                <article
                  key={post.id}
                  className="group border border-border-subtle bg-card-bg rounded-lg p-6 card-hover glow-subtle"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <Link href={`/posts/${post.id}`}>
                        <h3 className="text-xl font-semibold text-heading group-hover:text-accent transition-colors mb-2">
                          {post.title}
                        </h3>
                      </Link>
                      <p className="text-foreground-muted mb-4 line-clamp-2">
                        {post.excerpt}
                      </p>
                      <div className="flex items-center gap-4 text-xs text-foreground-muted font-mono">
                        <span>{post.date}</span>
                        <span className="text-border">·</span>
                        <span>{post.readTime} read</span>
                        <span className="text-border">·</span>
                        <div className="flex gap-2">
                          {post.tags.slice(0, 2).map((tag) => (
                            <span
                              key={tag}
                              className="tag"
                            >
                              #{tag}
                            </span>
                          ))}
                          {post.tags.length > 2 && (
                            <span className="text-foreground-muted">
                              +{post.tags.length - 2}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <span className="text-accent font-mono text-sm">
                      0{index + 1}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between text-sm text-foreground-muted font-mono">
            <p>
              © 2025 <span className="text-foreground-muted/60">built_with</span>{" "}
              <span className="text-accent">Next.js</span>
            </p>
            <p>
              {"<"}no_ads tracked={<span className="text-accent-tertiary">false</span>} /{">"}
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
