# Local QA report — visual rebuild

## What was tested

- Desktop 1440 × 900, tablet 1024 × 768, mobile 390 × 844 rendered screenshots using headless Chromium `set_content` with full-page images. This is a **local source render**, not direct access to the live public URL; network access to the live domain was blocked in the rendering container.
- Homepage: no horizontal overflow, computed hero H1 72px / 52px / 42.9px, mobile toggle opens menu and updates `aria-expanded` correctly.
- Writing, Projects, Start Here, Contact pages: no horizontal overflow at 1440px or 390px.
- Writing search: input `Physical AI` narrowed 6 mocked preview articles to 1; topic filter also ran.
- Images in the desktop visual preview loaded. One mobile lazy image was still pending when inspected, which is expected under offscreen lazy loading; a deployed browser-level check is recommended.
- `node --check js/site.js`: passed.
- YAML content front matter parsed for all authored content pages.
- New article links target existing Jekyll permalink patterns; all existing posts remain untouched.

## Important limitations

- Screenshots were rendered from local preview HTML and actual CSS/JS; Jekyll's complete rendering must be verified in the GitHub Pages workflow.
- Real Lighthouse scores, Search Console indexing, third-party embeds and cross-browser behavior cannot be certified by an offline local render.
- Screenshots use a local typographic portrait placeholder; production uses the existing `img/bio-photo.jpg` from the repository.
- Any 'impact' outcomes are described qualitatively; published quantified claims require supporting evidence.