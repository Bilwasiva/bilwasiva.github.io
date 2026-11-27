# Bilwasiva personal website — architecture and editorial system

## Deployment

- GitHub Pages / Jekyll static site.
- Responsive, self-contained CSS design system: `/css/site.css`.
- Progressive-enhancement client script: `/js/site.js`.
- No database or middleware is required for publicly readable articles, topic filtering, reading progress, theme preference, or navigation.
- All posts in `_posts` and their existing permalink patterns are preserved; no post content is rewritten by this redesign.
- Old routes `/blog/` and `/aboutme.html` remain readable and link users to the primary canonical section.

## Key paths

- `/`: evidence-led homepage; strong personal identity; 4 featured essays; specialty map; career snapshot; selected code; latest notes.
- `/writing/`: static, crawlable full archive with client-side search, filters, and sorting.
- `/projects/`: factual industry areas and actual public code/demonstrations. Avoid claims of production deployments for simulation-only code.
- `/aboutme/`: detailed biography and education; no fabricated awards, paper counts, or impact metrics.
- `/start-here/`: curated reading paths to increase article discovery and internal linking.
- `/contact/`: direct mail/LinkedIn contact; no unverified web form storing personal data.

## User experience and SEO

- Semantic sections, single main H1 per custom page, readable type scale, skip link, keyboard/focus styles, mobile menu, reduced-motion support.
- Structured data for Person/ProfilePage and BlogPosting, canonical URLs, OpenGraph/Twitter meta, image metadata, sitemap, robots.txt, RSS.
- Real hero image is generated for this site: `assets/img/ai-hero-laboratory.webp` (178 KB). Social sharing image: `ai-hero-social.jpg` (~235 KB).
- Homepage artwork is a conceptual visualization, not a photo of a real deployed system.
- Uses lazy-loading for offscreen images and a preload for homepage hero image.

## Why not add a backend, database, React, and middleware by default?

A static professional portfolio is read-heavy and low-interaction. Adding services that do not answer a user need increases security, deployment, costs, maintenance, and page weight. The chosen architecture uses build-time Jekyll templating and standards-based browser APIs for fast editorial browsing. A backend is justified when adding authenticated dashboards, private collections, newsletter opt-in with consent, or complex community workflows, and should be deployed with explicit security design and a real data provider. Avoid anonymous comment forms storing PII without anti-abuse.

## Next steps for brand growth

- Connect a real Search Console property and verify sitemap indexing; no account verification was performed automatically.
- Add verified case studies with evidence and authorization (architecture, role, measurable result, methods, team, links).
- Review accessible navigation, actual color contrast, browser performance and real user field Core Web Vitals.
- Update résumé and professional social profiles consistently.
- Publish original research-minded articles with internal links to related work.
- Track organic clicks, newsletter opt-ins if implemented later, GitHub stars, reading engagement and qualified contact inquiries without inventing numbers.