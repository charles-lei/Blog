import type { MetadataRoute } from "next";
import { getAllPosts } from "@/lib/posts";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const posts = getAllPosts();
  const domain = "https://yourdomain.com";

  return [
    {
      url: domain,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 1,
    },
    {
      url: `${domain}/about`,
      lastModified: new Date(),
      changeFrequency: "monthly",
      priority: 0.8,
    },
    {
      url: `${domain}/tags`,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 0.7,
    },
    ...posts.map((post) => ({
      url: `${domain}/posts/${post.id}`,
      lastModified: new Date(post.date),
      changeFrequency: "monthly" as const,
      priority: 0.6,
    })),
  ];
}
