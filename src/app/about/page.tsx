import Link from "next/link";
import { Header } from "@/components/header";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About | Lei Blog",
  description: "About me and this blog",
};

export default function AboutPage() {
  const skillBadges = {
    languages: [
      { name: "Python", src: "https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" },
      { name: "Ruby", src: "https://img.shields.io/badge/ruby-%23CC342D.svg?style=for-the-badge&logo=ruby&logoColor=white" },
      { name: "Rust", src: "https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" },
      { name: "C#", src: "https://img.shields.io/badge/c%23-%23239120.svg?style=for-the-badge&logo=csharp&logoColor=white" },
      { name: "Bash", src: "https://img.shields.io/badge/bash_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white" },
      { name: "Solidity", src: "https://img.shields.io/badge/Solidity-%23363636.svg?style=for-the-badge&logo=solidity&logoColor=white" },
      { name: "TypeScript", src: "https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white" },
      { name: "JavaScript", src: "https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E" },
    ],
    frontend: [
      { name: "React", src: "https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB" },
      { name: "Vue", src: "https://img.shields.io/badge/vuejs-%2335495e.svg?style=for-the-badge&logo=vuedotjs&logoColor=%234FC08D" },
      { name: "Next.js", src: "https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white" },
      { name: "TailwindCSS", src: "https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white" },
      { name: "HTML5", src: "https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white" },
      { name: "Vite", src: "https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white" },

    ],
    backend: [
      { name: "Node.js", src: "https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white" },
      { name: "Express.js", src: "https://img.shields.io/badge/express.js-%23404d59.svg?style=for-the-badge&logo=express&logoColor=%2361DAFB" },
      { name: "FastAPI", src: "https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" },
      { name: "Flask", src: "https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white" },
      { name: "Bun", src: "https://img.shields.io/badge/Bun-%23000000.svg?style=for-the-badge&logo=bun&logoColor=white" },
    ],
    database: [
      { name: "PostgreSQL", src: "https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white" },
      { name: "MongoDB", src: "https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white" },
      { name: "Redis", src: "https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white" },
      { name: "SQLite", src: "https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white" },
    ],
    devops: [
      { name: "Docker", src: "https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" },
      { name: "Kubernetes", src: "https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white" },
      { name: "AWS", src: "https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white" },
      { name: "Vercel", src: "https://img.shields.io/badge/vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white" },
      { name: "Nginx", src: "https://img.shields.io/badge/nginx-%23009639.svg?style=for-the-badge&logo=nginx&logoColor=white" },
      { name: "GitHub Actions", src: "https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white" },
      { name: "Prometheus", src: "https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=white" },
      { name: "Grafana", src: "https://img.shields.io/badge/grafana-%23F46800.svg?style=for-the-badge&logo=grafana&logoColor=white" },
      { name: "Alibaba Cloud", src: "https://img.shields.io/badge/Alibaba%20Cloud-FF6A00?style=for-the-badge&logo=alibabacloud&logoColor=white" },
    ],
    tools: [
      { name: "Git", src: "https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white" },
      { name: "GitHub", src: "https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white" },
      { name: "NPM", src: "https://img.shields.io/badge/NPM-%23CB3837.svg?style=for-the-badge&logo=npm&logoColor=white" },
      { name: "PNPM", src: "https://img.shields.io/badge/pnpm-%234a4a4a.svg?style=for-the-badge&logo=pnpm&logoColor=f69220" },
      { name: "Yarn", src: "https://img.shields.io/badge/yarn-%232C8EBB.svg?style=for-the-badge&logo=yarn&logoColor=white" },
      { name: "Postman", src: "https://img.shields.io/badge/Postman-FF6C37?style=for-the-badge&logo=postman&logoColor=white" },
      { name: "ESLint", src: "https://img.shields.io/badge/ESLint-4B3263?style=for-the-badge&logo=eslint&logoColor=white" },
    ],
    aiData: [
      { name: "PyTorch", src: "https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" },
      { name: "scikit-learn", src: "https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" },
      { name: "Pandas", src: "https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" },
      { name: "NumPy", src: "https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" },
      { name: "OpenCV", src: "https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" },
    ],
  };

  return (
    <div className="min-h-screen">
      <Header currentPath="/about" />

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

        <section className="accent-border-left mb-8 sm:mb-12">
          <h1 className="text-2xl sm:text-3xl font-bold mb-4 text-heading">About Me</h1>
          <p className="text-foreground-muted font-mono text-xs sm:text-sm mb-6">
            <span className="text-accent">interface</span>{" "}
            <span className="text-accent-secondary">Developer</span> {"{"}
          </p>
        </section>

        <div className="space-y-6">
          <p className="text-foreground-muted leading-relaxed text-sm sm:text-base">
            Hi, I&apos;m a fullstack developer passionate about building elegant solutions to complex
            problems. I believe in writing clean, maintainable code and sharing
            knowledge with the community. I&apos;m currently learning Agentic AI, Keep on exploring Latest Technologies
          </p>

          <div className="mt-8 sm:mt-12 space-y-6">
            <h2 className="text-xs sm:text-sm font-mono text-foreground-muted mb-4 sm:mb-6 flex items-center gap-2">
              <span className="text-accent">{"//"}</span> tech_stack
            </h2>

            {[
              { title: "Languages", badges: skillBadges.languages },
              { title: "Frontend", badges: skillBadges.frontend },
              { title: "Backend", badges: skillBadges.backend },
              { title: "Database", badges: skillBadges.database },
              { title: "DevOps & Cloud", badges: skillBadges.devops },
              { title: "AI & Data", badges: skillBadges.aiData },
              { title: "Tools", badges: skillBadges.tools },
            ].map((category) => (
              <div key={category.title}>
                <h3 className="text-xs font-mono text-accent-secondary mb-2">{category.title}</h3>
                <div className="flex flex-wrap gap-2">
                  {category.badges.map((badge) => (
                    <img
                      key={badge.name}
                      src={badge.src}
                      alt={badge.name}
                      className="h-5 sm:h-6"
                      loading="lazy"
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 sm:mt-12 p-4 sm:p-6 border border-border-subtle rounded-lg bg-card-bg">
            <p className="text-foreground-muted font-mono text-xs sm:text-sm mb-2">
              <span className="text-accent">const</span>{" "}
              <span className="text-accent-secondary">contact</span> = {"{"}
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm ml-2 sm:ml-4">
              email: <span className="text-accent-tertiary">&quot;leilis@qq.com&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm ml-2 sm:ml-4">
              github:{" "}
              <span className="text-accent-tertiary">&quot;github.com/charles-lei&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm ml-2 sm:ml-4">
              Wechat:{" "}
              <span className="text-accent-tertiary">&quot;@ccppllxx&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm">{"};"}</p>
          </div>
        </div>
      </main>
    </div>
  );
}
