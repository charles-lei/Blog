import fs from "fs";
import path from "path";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllPostIds } from "./posts";

const postsDirectory = path.join(process.cwd(), "src/content/posts");

export async function getPostSource(id: string) {
  const fullPath = path.join(postsDirectory, `${id}.mdx`);

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  return fs.readFileSync(fullPath, "utf8");
}

export function MDXContent({ source }: { source: string }) {
  return <MDXRemote source={source} />;
}
