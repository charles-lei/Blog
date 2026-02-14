"use client";

import { useState, useEffect } from "react";

interface TerminalProps {
  text: string;
  typingSpeed?: number;
  className?: string;
}

export function Terminal({ text, typingSpeed = 100, className = "" }: TerminalProps) {
  const [displayText, setDisplayText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(text.substring(0, currentIndex + 1));
      }, typingSpeed);
      return () => clearTimeout(timeout);
    }
  }, [currentIndex, text, typingSpeed]);

  return (
    <span className={className}>
      {displayText}
      {currentIndex < text.length && (
        <span className="inline-block w-2 h-4 bg-cyan-400 ml-0.5 animate-pulse" />
      )}
    </span>
  );
}

interface CodeBlockProps {
  children: React.ReactNode;
  className?: string;
}

export function CodeBlock({ children, className = "" }: CodeBlockProps) {
  return (
    <div className={`relative ${className}`}>
      <div className="absolute top-0 left-0 right-0 h-6 bg-zinc-800 rounded-t-lg flex items-center px-3 gap-2">
        <div className="w-3 h-3 rounded-full bg-red-500" />
        <div className="w-3 h-3 rounded-full bg-yellow-500" />
        <div className="w-3 h-3 rounded-full bg-green-500" />
        <span className="ml-2 text-xs text-zinc-500 font-mono">terminal</span>
      </div>
      <pre className="pt-8 pb-4 px-4 bg-zinc-900 rounded-lg border border-zinc-800 overflow-x-auto">
        <code className="text-sm font-mono text-zinc-300">{children}</code>
      </pre>
    </div>
  );
}
