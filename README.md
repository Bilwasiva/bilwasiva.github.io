# Bilwasiva Basu Mallick — AI/ML Portfolio and Technical Blog

**Live site:** https://bilwasiva.github.io/

A professional portfolio and long-form technical writing hub built on Jekyll and GitHub Pages. The redesign emphasizes readable research writing, clear professional positioning, accessible navigation, and crawlable article discovery.

## Sections

- [Home](https://bilwasiva.github.io/) — professional introduction, research areas, and featured writing.
- [Writing](https://bilwasiva.github.io/writing/) — searchable, topic-filterable archive of published technical articles.
- [Projects](https://bilwasiva.github.io/projects/) — selected engineering areas, open-source references, and educational examples.
- [About](https://bilwasiva.github.io/aboutme/) — background, interests, and ways to connect.
- [RSS](https://bilwasiva.github.io/feed.xml) — subscribe to new articles.

## Development

This repo retains the existing `Beautiful Jekyll` foundation with a progressively enhanced custom CSS/JS design.

- Site settings: `_config.yml`
- Search/share/JSON-LD metadata: `_includes/head.html`
- Design and breakpoints: `css/portfolio.css`
- Interaction (theme, article filtering, table of contents, reading progress): `js/portfolio.js`
- Blog articles: `_posts/YYYY-MM-DD-slug.md`
- Indexable public URLs: `sitemap.xml` and `robots.txt`
- Homepage: `index.html`; writing: `writing/index.html`; work: `projects/index.html`

Posts are rendered by Jekyll using `_layouts/post.html`. The original article sources remain unchanged and linked. Interactive enhancements are optional; core content and navigation work without JavaScript.

## Publishing checklist

1. Confirm the GitHub Pages workflow completed successfully after each update.
2. Test homepage, `/writing/`, `/projects/`, `/aboutme/`, a recent long-form article, `/sitemap.xml`, and `/robots.txt` on mobile and desktop.
3. Add `https://bilwasiva.github.io/sitemap.xml` to Google Search Console after verifying ownership. Indexing is never guaranteed.
4. Supply a real Google Analytics or privacy-friendly analytics configuration **only** when account ownership, consent needs, and the actual measurement ID are known. Do not use placeholder IDs.
5. For new posts, add a substantive subtitle/description, tags, a descriptive cover image and `share-img` when available, and citations to primary research.
6. Periodically update the About, Projects, and README pages for factual accuracy. Avoid unsupported numerical impact claims.

## Attribution

Originally based on [Beautiful Jekyll](https://deanattali.com/beautiful-jekyll/), credited in the site footer. Existing third-party code retains its license terms.