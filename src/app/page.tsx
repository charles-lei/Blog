import Link from "next/link";
import { getAllPosts } from "@/lib/posts";
import { Header } from "@/components/header";

export default async function HomePage() {
  const posts = getAllPosts();

  return (
    <div className="min-h-screen">
      <Header currentPath="/" />

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12 flex-1">
        {/* Hero Section */}
        <section className="mb-12 sm:mb-16">
          <div className="accent-border-left">
            <p className="text-foreground-muted font-mono text-xs sm:text-sm mb-2">
              <span className="text-accent">const</span>{" "}
              <span className="text-accent-secondary">introduction</span> = {"{"}
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm ml-2 sm:ml-4">
              role: <span className="text-accent-tertiary">&quot;developer&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm ml-2 sm:ml-4">
              passion: <span className="text-accent-tertiary">&quot;building things&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm ml-2 sm:ml-4">
              coffee: <span className="text-accent-tertiary">true</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm">{"};"}</p>
            <p className="text-foreground-muted mt-4 sm:mt-6 text-base sm:text-lg leading-relaxed">
              Welcome to my corner of the internet. I write about code,
              systems, and the craft of software engineering.
            </p>
          </div>
        </section>

        {/* Blog Posts */}
        <section>
          <h2 className="text-xs sm:text-sm font-mono text-foreground-muted mb-6 sm:mb-8 flex items-center gap-2">
            <span className="text-accent">{"//"}</span> latest_posts
          </h2>

          {posts.length === 0 ? (
            <div className="border border-border-subtle rounded-lg p-4 sm:p-6 bg-card-bg">
              <p className="text-foreground-muted text-center font-mono text-xs sm:text-sm">
                <span className="text-accent">{"/*"}</span> No posts yet. Add some
                MDX files to src/content/posts/
                <span className="text-accent">{" */"}</span>
              </p>
            </div>
          ) : (
            <div className="space-y-4 sm:space-y-6">
              {posts.map((post, index) => (
                <article
                  key={post.id}
                  className="group border border-border-subtle bg-card-bg rounded-lg p-4 sm:p-6 card-hover glow-subtle"
                >
                  <div className="flex items-start justify-between gap-3 sm:gap-4">
                    <div className="flex-1 min-w-0">
                      <Link href={`/posts/${post.id}`}>
                        <h3 className="text-lg sm:text-xl font-semibold text-heading group-hover:text-accent transition-colors mb-2">
                          {post.title}
                        </h3>
                      </Link>
                      <p className="text-foreground-muted mb-3 sm:mb-4 line-clamp-2 text-sm sm:text-base">
                        {post.excerpt}
                      </p>
                      <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-xs text-foreground-muted font-mono">
                        <span>{post.date}</span>
                        <span className="text-border hidden sm:inline">·</span>
                        <span>{post.readTime} read</span>
                        <span className="text-border hidden sm:inline">·</span>
                        <div className="flex flex-wrap gap-1 sm:gap-2">
                          {post.tags.slice(0, 2).map((tag) => (
                            <span
                              key={tag}
                              className="tag text-xs"
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
                    <span className="text-accent font-mono text-sm flex-shrink-0">
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
      <footer className="border-t border-border mt-12 sm:mt-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 sm:gap-0 text-xs sm:text-sm text-foreground-muted font-mono">
            <p>
              © 2025 <span className="text-foreground-muted/60">built_with</span>{" "}
              <span className="text-accent">Next.js</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
