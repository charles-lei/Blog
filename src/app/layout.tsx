import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Geek Blog",
  description: "A minimalist tech-focused blog",
  authors: [{ name: "Your Name" }],
  keywords: ["blog", "tech", "programming", "geek"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="antialiased min-h-screen flex flex-col">
        {children}
      </body>
    </html>
  );
}
