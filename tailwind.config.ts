import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        "background-secondary": "var(--background-secondary)",
        foreground: "var(--foreground)",
        "foreground-muted": "var(--foreground-muted)",
        border: "var(--border)",
        "border-subtle": "var(--border-subtle)",
        accent: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
          secondary: "var(--accent-secondary)",
          tertiary: "var(--accent-tertiary)",
        },
        "card-bg": "var(--card-bg)",
        "card-hover": "var(--card-hover)",
        "code-bg": "var(--code-bg)",
        heading: "var(--heading)",
        link: {
          DEFAULT: "var(--link)",
          hover: "var(--link-hover)",
        },
        tag: {
          bg: "var(--tag-bg)",
          text: "var(--tag-text)",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        lg: "12px",
        xl: "16px",
      },
      animation: {
        "fade-in": "fadeIn 0.4s ease-out",
        pulse: "pulse 2s ease-in-out infinite",
      },
    },
  },
  plugins: [typography],
};
export default config;
