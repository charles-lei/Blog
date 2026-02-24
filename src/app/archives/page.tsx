import Link from "next/link";
import { getAllPosts } from "@/lib/posts";
import { Header } from "@/components/header";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Archives | Lei Blog",
  description: "Browse all blog posts by date",
};

export default async function ArchivesPage() {
  const posts = getAllPosts();

  // 按年月分组
  const groupedPosts = posts.reduce((acc, post) => {
    const date = new Date(post.date);
    const year = date.getFullYear();
    const month = date.getMonth() + 1;

    if (!acc[year]) {
      acc[year] = {};
    }
    if (!acc[year][month]) {
      acc[year][month] = [];
    }
    acc[year][month].push(post);

    return acc;
  }, {} as Record<number, Record<number, typeof posts>>);

  // 按年倒序排列
  const years = Object.keys(groupedPosts)
    .map(Number)
    .sort((a, b) => b - a);

  const monthNames = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ];

  return (
    <div className="min-h-screen">
      <Header currentPath="/archives" />

      <main className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className="mb-8 sm:mb-12">
          <Link
            href="/"
            className="text-xs sm:text-sm text-foreground-muted hover:text-accent inline-flex items-center gap-2 font-mono transition-colors"
          >
            <span className="text-accent">←</span> back_to_home
          </Link>
        </div>

        <section className="mb-8 sm:mb-12">
          <h1 className="text-2xl sm:text-3xl font-bold mb-2 text-heading">Archives</h1>
          <p className="text-foreground-muted font-mono text-xs sm:text-sm">
            <span className="text-accent">const</span>{" "}
            <span className="text-accent-secondary">total_posts</span> ={" "}
            <span className="text-accent-tertiary">{posts.length}</span>;
          </p>
        </section>

        {/* Timeline */}
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-3 sm:left-4 top-0 bottom-0 w-0.5 bg-border" />

          {years.map((year) => (
            <div key={year} className="mb-8 sm:mb-12">
              {/* Year marker */}
              <div className="flex items-center gap-3 sm:gap-4 mb-4 sm:mb-6">
                <div className="relative z-10 w-6 sm:w-8 h-6 sm:h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-bold text-background">{year.toString().slice(-2)}</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-heading">{year}</h2>
                <span className="text-xs sm:text-sm text-foreground-muted font-mono">
                  ({Object.values(groupedPosts[year]).flat().length} posts)
                </span>
              </div>

              {/* Months */}
              <div className="ml-9 sm:ml-12 space-y-6">
                {Object.keys(groupedPosts[year])
                  .map(Number)
                  .sort((a, b) => b - a)
                  .map((month) => (
                    <div key={`${year}-${month}`}>
                      {/* Month header */}
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-2 h-2 rounded-full bg-accent-secondary" />
                        <h3 className="text-sm sm:text-base font-semibold text-foreground-muted">
                          {monthNames[month - 1]}
                        </h3>
                        <span className="text-xs text-foreground-muted/60">
                          ({groupedPosts[year][month].length})
                        </span>
                      </div>

                      {/* Posts */}
                      <div className="ml-4 space-y-2">
                        {groupedPosts[year][month].map((post) => (
                          <Link
                            key={post.id}
                            href={`/posts/${post.id}`}
                            className="group flex items-start gap-3 sm:gap-4 py-2 px-3 -ml-3 rounded-lg hover:bg-card-bg transition-colors"
                          >
                            <span className="text-xs text-foreground-muted/60 font-mono w-12 sm:w-16 flex-shrink-0 pt-0.5">
                              {new Date(post.date).getDate().toString().padStart(2, "0")}
                            </span>
                            <div className="flex-1 min-w-0">
                              <h4 className="text-sm sm:text-base text-heading group-hover:text-accent transition-colors line-clamp-2">
                                {post.title}
                              </h4>
                              <div className="flex flex-wrap items-center gap-2 mt-1">
                                <span className="text-xs text-foreground-muted">{post.readTime} read</span>
                                <span className="text-border">·</span>
                                <div className="flex flex-wrap gap-1">
                                  {post.tags.slice(0, 2).map((tag) => (
                                    <span key={tag} className="tag text-xs">
                                      #{tag}
                                    </span>
                                  ))}
                                  {post.tags.length > 2 && (
                                    <span className="text-xs text-foreground-muted">+{post.tags.length - 2}</span>
                                  )}
                                </div>
                              </div>
                            </div>
                            <span className="text-accent font-mono text-sm flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                              →
                            </span>
                          </Link>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
