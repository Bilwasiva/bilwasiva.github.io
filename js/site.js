/* Progressive enhancement: no external JS framework, no user tracking, works on GitHub Pages. */
(() => {
 'use strict';
 const doc = document;
 const root = doc.documentElement;
 const select = (s, base=doc) => base.querySelector(s);
 const selectAll = (s, base=doc) => Array.from(base.querySelectorAll(s));
 const safeStore = {
   get(k) { try { return window.localStorage.getItem(k); } catch (_) { return null; } },
   set(k,v) { try { window.localStorage.setItem(k,v); } catch (_) {} }
 };
 // Appearance: accessible button, persists optional user preference.
 const appearanceButton = select('#theme-toggle');
 const pref = safeStore.get('bilwasiva-theme');
 const systemDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
 const theme = pref === 'light' || pref === 'dark' ? pref : (systemDark ? 'dark' : 'light');
 root.dataset.theme = theme;
 const updateAppearance = () => {
   if (!appearanceButton) return;
   const dark = root.dataset.theme === 'dark';
   appearanceButton.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} appearance`);
   appearanceButton.title = `Switch to ${dark ? 'light' : 'dark'} appearance`;
   appearanceButton.innerHTML = `<span aria-hidden="true">${dark ? '☀' : '☾'}</span>`;
 };
 updateAppearance();
 appearanceButton?.addEventListener('click', () => {
   root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
   safeStore.set('bilwasiva-theme', root.dataset.theme);
   updateAppearance();
 });
 // Responsive menu: keyboard, escape, outside clicks.
 const menuToggle = select('#menu-toggle');
 const menu = select('#primary-navigation');
 const closeMenu = () => { menu?.classList.remove('is-open'); menuToggle?.setAttribute('aria-expanded','false'); menuToggle?.setAttribute('aria-label','Open menu'); };
 menuToggle?.addEventListener('click', () => { const open = !menu?.classList.contains('is-open'); menu?.classList.toggle('is-open',open); menuToggle.setAttribute('aria-expanded',String(open)); menuToggle.setAttribute('aria-label', open?'Close menu':'Open menu'); });
 menu?.addEventListener('click', (e) => { if (e.target.closest('a')) closeMenu(); });
 doc.addEventListener('keydown', (e) => { if(e.key==='Escape') closeMenu(); });
 doc.addEventListener('click', (e) => { if (menu?.classList.contains('is-open') && !e.target.closest('.site-header')) closeMenu(); });
 // Archive: client-side title+tag search, topic filtering, sort; markup stays indexed by crawlers.
 const archive = select('#archive-articles');
 if (archive) {
   const cards = selectAll('.article-card[data-search]', archive);
   const queryInput = select('#article-query');
   const filterButtons = selectAll('.filter-button[data-topic]');
   const sortSelect = select('#article-sort');
   const count = select('#article-count');
   const empty = select('#article-empty');
   const params = new URLSearchParams(window.location.search);
   const initialQuery = params.get('q') || '';
   const initialTopic = params.get('topic') || 'all';
   if (queryInput) queryInput.value=initialQuery;
   let activeTopic = initialTopic;
   const apply = () => {
     const q = (queryInput?.value || '').trim().toLowerCase();
     const topic = activeTopic.toLowerCase();
     let visible = 0;
     cards.forEach(card=>{
       const text=(card.dataset.search||'').toLowerCase();
       const tags=(card.dataset.topics||'').toLowerCase();
       const matchesQuery=!q||text.includes(q);
       const matchesTopic=topic==='all'||tags.includes(topic);
       const show=matchesQuery&&matchesTopic;
       card.hidden=!show;if(show)visible++;
     });
     if(count) count.textContent=`Showing ${visible} of ${cards.length} articles`;
     if(empty) empty.hidden=visible!==0;
     filterButtons.forEach(button=>{const on=button.dataset.topic===activeTopic;button.classList.toggle('active',on);button.setAttribute('aria-pressed',String(on));});
   };
   filterButtons.forEach(b=>b.addEventListener('click',()=>{activeTopic=b.dataset.topic||'all';apply();}));
   queryInput?.addEventListener('input',apply);
   sortSelect?.addEventListener('change',()=>{
     const sorted=[...cards].sort((a,b)=>{
       const v=a.dataset.date||''; const w=b.dataset.date||'';
       if (sortSelect.value==='az') return (a.dataset.title||'').localeCompare(b.dataset.title||'');
       return sortSelect.value==='oldest' ? v.localeCompare(w) : w.localeCompare(v);
     });
     sorted.forEach(card=>archive.appendChild(card));apply();
   });
   if (initialTopic!=='all' && !filterButtons.some(b=>b.dataset.topic===initialTopic)) activeTopic='all';
   apply();
 }
 // Reading progress & TOC: only on article pages. Remove duplicate first H1 when it equals header title.
 const article = select('#article-body');
 if (article) {
   const articleTitle = select('.article-header h1')?.textContent?.trim().replace(/\s+/g,' ');
   const firstHeading = select('h1', article);
   if(firstHeading && firstHeading.textContent.trim().replace(/\s+/g,' ')===articleTitle) firstHeading.remove();
   const wordCount = (article.innerText || '').trim().split(/\s+/).filter(Boolean).length;
   const readingTime = select('#reading-time');
   if(readingTime) readingTime.textContent=`${Math.max(1,Math.ceil(wordCount/220))} min read`;
   const toc = select('#article-toc');
   const tocList = select('#article-toc-list');
   const headings = selectAll('h2',article).filter(h=>!h.closest('blockquote') && h.textContent.trim());
   if (tocList && headings.length) {
     const seen = new Set();
     headings.forEach((h,i)=>{
       let id=h.id || h.textContent.trim().toLowerCase().normalize('NFKD').replace(/[^a-z0-9\s-]/g,'').replace(/\s+/g,'-').slice(0,60) || `section-${i+1}`;
       let proposed=id,n=2;while(seen.has(proposed) || (doc.getElementById(proposed)&&doc.getElementById(proposed)!==h)){proposed=`${id}-${n++}`;}
       h.id=proposed;seen.add(proposed);
       const li=doc.createElement('li');const a=doc.createElement('a');a.href=`#${h.id}`;a.textContent=h.textContent.trim().replace(/\s+/g,' ').slice(0,86);li.appendChild(a);tocList.appendChild(li);
     });
     if('IntersectionObserver' in window){const obs=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){selectAll('a',tocList).forEach(link=>link.classList.toggle('is-active',link.hash===`#${entry.target.id}`));}});},{rootMargin:'-120px 0px -70% 0px'});headings.forEach(h=>obs.observe(h));}
   } else if (toc) toc.hidden=true;
   const progress = select('#reading-progress');
   let scheduled=false;
   const refresh = () => {scheduled=false;if(!progress)return;const start=article.getBoundingClientRect().top+window.scrollY;const end=article.getBoundingClientRect().bottom+window.scrollY-window.innerHeight;const current=window.scrollY;const share=end<=start?(current>start?1:0):Math.max(0,Math.min(1,(current-start)/(end-start)));progress.style.transform=`scaleX(${share.toFixed(4)})`;};
   window.addEventListener('scroll',()=>{if(!scheduled){scheduled=true;window.requestAnimationFrame(refresh);}},{passive:true});window.addEventListener('resize',refresh);refresh();
 }
 // Clipboard: clear user feedback without hiding failures.
 select('#copy-article-link')?.addEventListener('click', async e=>{
   const button=e.currentTarget;
   try { await navigator.clipboard.writeText(window.location.href);const previous=button.textContent;button.textContent='Copied ✓';setTimeout(()=>button.textContent=previous,1700); }
   catch(_){button.textContent='Copy unavailable';}
 });
})();