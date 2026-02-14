import Link from "next/link";
import { getAllTags, getAllPosts } from "@/lib/posts";
import { Header } from "@/components/header";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tags | Lei Blog",
  description: "Browse all blog posts by tag",
};

export default async function TagsPage() {
  const tags = getAllTags();
  const posts = getAllPosts();

  const getPostsByTag = (tag: string) => {
    return posts.filter((post) => post.tags.includes(tag));
  };

  return (
    <div className="min-h-screen">
      <Header currentPath="/tags" />

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className="mb-8 sm:mb-12">
          <Link
            href="/"
            className="text-xs sm:text-sm text-foreground-muted hover:text-accent inline-flex items-center gap-2 font-mono transition-colors"
          >
            <span className="text-accent">←</span> back_to_home
          </Link>
        </div>

        <section className="mb-12 sm:mb-16">
          <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-heading">Tags</h1>
          <p className="text-foreground-muted font-mono text-xs sm:text-sm mb-6">
            <span className="text-accent">const</span>{" "}
            <span className="text-accent-secondary">available_tags</span> = {"{"}
          </p>
        </section>

        {tags.length === 0 ? (
          <div className="border border-border-subtle rounded-lg p-4 sm:p-6 bg-card-bg">
            <p className="text-foreground-muted text-center font-mono text-xs sm:text-sm">
              <span className="text-accent">{"/*"}</span> No tags found yet. Add
              some posts!
              <span className="text-accent">{" */"}</span>
            </p>
          </div>
        ) : (
          <div className="space-y-6 sm:space-y-8">
            {tags.map((tagData, index) => (
              <div key={tagData.tag}>
                <h2 className="text-xs sm:text-sm font-mono text-foreground-muted mb-3 sm:mb-4 flex items-center gap-2">
                  <span className="text-accent">#{index + 1}</span>
                  <span className="text-heading">{tagData.tag}</span>
                  <span className="text-foreground-muted/60">
                    ({tagData.count} {tagData.count === 1 ? "post" : "posts"})
                  </span>
                </h2>
                <div className="ml-4 sm:ml-6 space-y-2 sm:space-y-3">
                  {getPostsByTag(tagData.tag).map((post) => (
                    <Link
                      key={post.id}
                      href={`/posts/${post.id}`}
                      className="block text-foreground-muted hover:text-accent transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3 sm:gap-4 group">
                        <div className="flex-1 min-w-0">
                          <h3 className="text-heading group-hover:text-accent transition-colors text-sm sm:text-base">
                            {post.title}
                          </h3>
                          <p className="text-xs sm:text-sm text-foreground-muted/60 mt-1">
                            {post.date} · {post.readTime} read
                          </p>
                        </div>
                        <span className="text-accent font-mono text-sm flex-shrink-0">
                          →
                        </span>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
