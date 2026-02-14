import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";
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
              <Link href="/about" className="text-accent font-medium">
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
      <main className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-12">
          <Link
            href="/"
            className="text-sm text-foreground-muted hover:text-accent inline-flex items-center gap-2 font-mono transition-colors"
          >
            <span className="text-accent">←</span> back_to_home
          </Link>
        </div>

        <section className="accent-border-left mb-12">
          <h1 className="text-3xl font-bold mb-4 text-heading">About Me</h1>
          <p className="text-foreground-muted font-mono text-sm mb-6">
            <span className="text-accent">interface</span>{" "}
            <span className="text-accent-secondary">Developer</span> {"{"}
          </p>
        </section>

        <div className="space-y-6">
          <p className="text-foreground-muted leading-relaxed">
            Hi, I&apos;m a software developer passionate about building elegant solutions to complex
            problems. I believe in writing clean, maintainable code and sharing
            knowledge with the community.
          </p>

          <p className="text-foreground-muted leading-relaxed">
            When I&apos;m not coding, you can find me reading about distributed systems,
            tinkering with side projects, or enjoying a good cup of coffee.
          </p>

          <div className="mt-12">
            <h2 className="text-sm font-mono text-foreground-muted mb-6 flex items-center gap-2">
              <span className="text-accent">{"//"}</span> tech_stack
            </h2>
            <div className="space-y-4">
              {skills.map((skill) => (
                <div key={skill.name}>
                  <div className="flex justify-between text-sm mb-1">
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

          <div className="mt-12 p-6 border border-border-subtle rounded-lg bg-card-bg">
            <p className="text-foreground-muted font-mono text-sm mb-2">
              <span className="text-accent">const</span>{" "}
              <span className="text-accent-secondary">contact</span> = {"{"}
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              email: <span className="text-accent-tertiary">&quot;leilis@qq.com&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              github:{" "}
              <span className="text-accent-tertiary">&quot;github.com/charles-lei&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              Discord:{" "}
              <span className="text-accent-tertiary">&quot;@discord&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm">{"};"}</p>
          </div>
        </div>
      </main>
    </div>
  );
}
