# Lei Blog

A minimalist, tech-focused personal blog built with Next.js, TypeScript, and Tailwind CSS. Features a clean dark theme with monospace fonts and a terminal-inspired aesthetic.

## Features

- 🚀 Fast static site generation with Next.js App Router
- 📝 MDX support for rich blog post content
- 🎨 Minimalist tech/geek aesthetic with dark theme
- 📱 Fully responsive design
- ⚛️ Built with React and TypeScript
- 🎯 Monospace fonts (JetBrains Mono) for code elements
- 🏷️ Tag system for organizing posts
- 📡 RSS feed for subscribers
- ♿ Accessible navigation
- 🔍 SEO optimized with sitemap and robots.txt

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Content**: MDX for blog posts
- **Fonts**: Inter & JetBrains Mono

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

Open [http://localhost:3000](http://localhost:3000) to view the blog.

## Project Structure

```
src/
├── app/
│   ├── about/
│   │   └── page.tsx       # About page
│   ├── posts/
│   │   └── [id]/
│   │       └── page.tsx   # Dynamic blog post pages (MDX)
│   ├── tags/
│   │   └── page.tsx       # Tags listing page
│   ├── rss.xml/
│   │   └── route.ts       # RSS feed
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Homepage
├── components/
│   └── terminal.tsx       # Terminal components
├── content/
│   └── posts/
│       ├── *.mdx          # Blog post content
├── lib/
│   ├── posts.ts           # Post utilities
│   └── mdx.tsx           # MDX utilities
```

## Creating New Posts

1. Create a new `.mdx` file in `src/content/posts/`
2. Add frontmatter with metadata:

```mdx
---
title: "Your Post Title"
date: "2025-02-14"
excerpt: "A brief description of the post."
tags: ["tag1", "tag2"]
readTime: "5 min"
---

Your post content here in **Markdown**!
```

3. Write your content using Markdown syntax
4. The post will automatically appear on the homepage

## Customization

### Colors

Edit the CSS variables in `src/app/globals.css`:

```css
:root {
  --background: #0a0a0a;
  --foreground: #e4e4e7;
  --accent: #22d3ee;
  --accent-secondary: #a78bfa;
  /* ... */
}
```

### Domain

Update `yourdomain.com` in:
- `src/app/rss.xml/route.ts`
- `src/app/sitemap.ts`
- `src/app/robots.ts`

### Personal Info

Edit:
- `src/app/about/page.tsx` - About page content
- `src/app/page.tsx` - Homepage hero section
- `src/app/layout.tsx` - Site metadata

## Deployment

This blog can be deployed to any static hosting service:

### Vercel (Recommended)

```bash
vercel deploy
```

### Netlify

Connect your Git repository to Netlify

### GitHub Pages

1. Run `npm run build`
2. Push `out/` directory to `gh-pages` branch

### Static Export

```bash
npm run build
# Static files will be in .next/static
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## License

MIT
