"use client";

import { useState, useEffect } from "react";

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null;
    const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";

    const initialTheme = savedTheme || systemTheme;
    setTheme(initialTheme);
    document.documentElement.classList.toggle("light", initialTheme === "light");
    document.documentElement.classList.toggle("dark", initialTheme === "dark");
    setMounted(true);
  }, []);

  const toggleTheme = (newTheme: "light" | "dark") => {
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    document.documentElement.classList.toggle("light", newTheme === "light");
    document.documentElement.classList.toggle("dark", newTheme === "dark");
  };

  if (!mounted) {
    return null;
  }

  return (
    <div className="flex items-center gap-1 bg-zinc-800/50 dark:bg-zinc-800/50 light:bg-zinc-300/50 rounded-md p-1">
      <button
        onClick={() => toggleTheme("dark")}
        className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
          theme === "dark"
            ? "bg-orange-500 text-white shadow-sm"
            : "text-zinc-500 dark:text-zinc-500 light:text-zinc-700 hover:text-zinc-300 dark:hover:text-zinc-300 light:hover:text-zinc-900"
        }`}
        aria-label="Switch to dark mode"
      >
        Dark
      </button>
      <button
        onClick={() => toggleTheme("light")}
        className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
          theme === "light"
            ? "bg-zinc-300 text-zinc-900 dark:bg-zinc-300 dark:text-zinc-900 light:bg-zinc-300 light:text-zinc-900 shadow-sm"
            : "text-zinc-500 dark:text-zinc-500 light:text-zinc-700 hover:text-zinc-300 dark:hover:text-zinc-300 light:hover:text-zinc-900"
        }`}
        aria-label="Switch to light mode"
      >
        Light
      </button>
    </div>
  );
}
