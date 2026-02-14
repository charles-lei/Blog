import Link from "next/link";
import { Header } from "@/components/header";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About | Lei Blog",
  description: "About me and this blog",
};

export default function AboutPage() {
  const skills = [
    { name: "TypeScript", level: 90 },
    { name: "React / Next.js", level: 85 },
    { name: "Node.js", level: 80 },
    { name: "Python", level: 75 },
    { name: "Go", level: 60 },
  ];

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
            Hi, I&apos;m a software developer passionate about building elegant solutions to complex
            problems. I believe in writing clean, maintainable code and sharing
            knowledge with the community.
          </p>

          <p className="text-foreground-muted leading-relaxed text-sm sm:text-base">
            When I&apos;m not coding, you can find me reading about distributed systems,
            tinkering with side projects, or enjoying a good cup of coffee.
          </p>

          <div className="mt-8 sm:mt-12">
            <h2 className="text-xs sm:text-sm font-mono text-foreground-muted mb-4 sm:mb-6 flex items-center gap-2">
              <span className="text-accent">{"//"}</span> tech_stack
            </h2>
            <div className="space-y-3 sm:space-y-4">
              {skills.map((skill) => (
                <div key={skill.name}>
                  <div className="flex justify-between text-xs sm:text-sm mb-1">
                    <span className="text-foreground-muted font-mono">{skill.name}</span>
                    <span className="text-foreground-muted font-mono">{skill.level}%</span>
                  </div>
                  <div className="h-2 bg-card-bg rounded-full overflow-hidden border border-border-subtle">
                    <div
                      className="h-full bg-gradient-to-r from-accent to-accent-secondary rounded-full transition-all duration-500"
                      style={{ width: `${skill.level}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
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
              Discord:{" "}
              <span className="text-accent-tertiary">&quot;@discord&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-xs sm:text-sm">{"};"}</p>
          </div>
        </div>
      </main>
    </div>
  );
}
