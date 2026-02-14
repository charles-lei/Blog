import { codeToHtml } from "shiki";
import { CopyButton } from "./copy-button";

interface CodeBlockProps {
  children: React.ReactNode;
  className?: string;
}

export async function CodeBlock({ children, className }: CodeBlockProps) {
  let codeText = "";
  let language = "text";

  // Extract language from className (e.g., "language-typescript")
  if (className?.startsWith("language-")) {
    language = className.replace("language-", "");
  }

  // Extract code text from children
  if (typeof children === "string") {
    codeText = children;
  } else if (children && typeof children === "object" && "props" in children) {
    const child = children as { props?: { children?: string; className?: string } };
    if (child.props?.children) {
      codeText = child.props.children;
    }
    // Also check for language in code element
    if (child.props?.className?.startsWith("language-")) {
      language = child.props.className.replace("language-", "");
    }
  }

  let highlightedHtml = "";

  try {
    highlightedHtml = await codeToHtml(codeText, {
      lang: language,
      theme: "github-dark",
    });
  } catch {
    // Fallback to plain text if language is not supported
    highlightedHtml = await codeToHtml(codeText, {
      lang: "text",
      theme: "github-dark",
    });
  }

  return (
    <div className="relative group rounded-xl overflow-hidden border border-zinc-700 bg-[#0d1117] my-6">
      {/* Header with language label */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-800/80 border-b border-zinc-700">
        <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
          {language}
        </span>
        <CopyButton text={codeText} />
      </div>
      {/* Code content */}
      <div
        className="shiki-code overflow-x-auto"
        dangerouslySetInnerHTML={{ __html: highlightedHtml }}
      />
    </div>
  );
}
