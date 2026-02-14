import fs from "fs";
import path from "path";
import matter from "gray-matter";

const postsDirectory = path.join(process.cwd(), "src/content/posts");

export interface Post {
  id: string;
  title: string;
  date: string;
  excerpt: string;
  tags: string[];
  readTime: string;
  content: string;
}

export function getAllPosts(): Post[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const fileNames = fs.readdirSync(postsDirectory);
  const allPostsData = fileNames
    .filter((fileName) => fileName.endsWith(".mdx"))
    .map((fileName) => {
      const id = fileName.replace(/\.mdx$/, "");
      const fullPath = path.join(postsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, "utf8");
      const matterResult = matter(fileContents);

      return {
        id,
        title: matterResult.data.title as string,
        date: matterResult.data.date as string,
        excerpt: matterResult.data.excerpt as string,
        tags: matterResult.data.tags as string[],
        readTime: matterResult.data.readTime as string,
        content: matterResult.content,
      };
    });

  return allPostsData.sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getAllPostIds() {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const fileNames = fs.readdirSync(postsDirectory);
  return fileNames
    .filter((fileName) => fileName.endsWith(".mdx"))
    .map((fileName) => ({
      id: fileName.replace(/\.mdx$/, ""),
    }));
}

export async function getPostData(id: string): Promise<Post | null> {
  const fullPath = path.join(postsDirectory, `${id}.mdx`);

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(fullPath, "utf8");
  const matterResult = matter(fileContents);

  return {
    id,
    title: matterResult.data.title as string,
    date: matterResult.data.date as string,
    excerpt: matterResult.data.excerpt as string,
    tags: matterResult.data.tags as string[],
    readTime: matterResult.data.readTime as string,
    content: matterResult.content,
  };
}

export function getAllTags(): { tag: string; count: number }[] {
  const posts = getAllPosts();
  const tagCounts = new Map<string, number>();

  posts.forEach((post) => {
    post.tags.forEach((tag) => {
      tagCounts.set(tag, (tagCounts.get(tag) || 0) + 1);
    });
  });

  return Array.from(tagCounts.entries())
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count);
}
