import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About | Geek Blog",
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
                  <span className="text-heading">geek_blog</span>
                  {" "}
                  <span className="text-accent">/&gt;</span>
                </h1>
              </Link>
            </div>
            <nav className="flex gap-6 text-sm items-center">
              <Link href="/" className="text-foreground-muted hover:text-accent transition-colors">
                home
              </Link>
              <Link href="/tags" className="text-foreground-muted hover:text-accent transition-colors">
                tags
              </Link>
              <Link href="/about" className="text-accent font-medium">
                about
              </Link>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-foreground-muted hover:text-accent transition-colors"
              >
                github
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
              email: <span className="text-accent-tertiary">&quot;hello@example.com&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              github:{" "}
              <span className="text-accent-tertiary">&quot;github.com/yourusername&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm ml-4">
              twitter:{" "}
              <span className="text-accent-tertiary">&quot;@yourusername&quot;</span>,
            </p>
            <p className="text-foreground-muted font-mono text-sm">{"};"}</p>
          </div>
        </div>
      </main>
    </div>
  );
}
