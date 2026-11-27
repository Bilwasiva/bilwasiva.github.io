/* Progressive enhancement: the site remains browseable without JavaScript. */
(function () {
  'use strict';
  var root = document.documentElement;
  var themeButton = document.getElementById('bm-theme-toggle');
  function setTheme(value) {
    var theme = value === 'dark' ? 'dark' : 'light';
    root.setAttribute('data-theme', theme);
    if (themeButton) {
      themeButton.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      themeButton.setAttribute('title', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
      themeButton.querySelector('span').textContent = theme === 'dark' ? '☀' : '◐';
    }
  }
  try {
    setTheme(localStorage.getItem('bm-theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  } catch (_) { setTheme('light'); }
  if (themeButton) themeButton.addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    setTheme(next);
    try { localStorage.setItem('bm-theme', next); } catch (_) {}
  });

  var menuButton = document.getElementById('bm-menu-toggle');
  var menu = document.getElementById('bm-nav-links');
  if (menuButton && menu) {
    menuButton.addEventListener('click', function () {
      var open = menuButton.getAttribute('aria-expanded') !== 'true';
      menuButton.setAttribute('aria-expanded', String(open));
      menuButton.setAttribute('aria-label', open ? 'Close navigation menu' : 'Open navigation menu');
      menu.classList.toggle('bm-menu-open', open);
    });
    menu.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () { menu.classList.remove('bm-menu-open'); menuButton.setAttribute('aria-expanded', 'false'); });
    });
    document.addEventListener('keydown', function (evt) {
      if (evt.key === 'Escape' && menu.classList.contains('bm-menu-open')) {
        menu.classList.remove('bm-menu-open'); menuButton.setAttribute('aria-expanded', 'false'); menuButton.focus();
      }
    });
  }

  var search = document.getElementById('bm-post-search');
  var filters = Array.prototype.slice.call(document.querySelectorAll('.bm-filter[data-topic]'));
  var cards = Array.prototype.slice.call(document.querySelectorAll('.bm-searchable'));
  var counter = document.getElementById('bm-result-count');
  var noResults = document.getElementById('bm-no-results');
  if (search && cards.length) {
    var params = new URLSearchParams(window.location.search);
    var activeTopic = params.get('topic') || 'all';
    var query = params.get('q') || '';
    if (!filters.some(function (item) {return item.getAttribute('data-topic') === activeTopic;})) activeTopic = 'all';
    search.value = query;
    function matchesTopic(content, topic) {
      if (topic === 'all') return true;
      if (topic === 'robotics') return /(robot|physical ai|world model|vision.language.action|vla|embodied|sim.to.real)/.test(content);
      if (topic === 'llm') return /(llm|language model|inference|transformer|rag|reasoning|genai|generative ai|foundation model|kv.cache)/.test(content);
      if (topic === 'agentic') return /(agent|multi.agent|orchestration|autonomous)/.test(content);
      if (topic === 'recommendation') return /(recommend|rec.?sys|personaliz|ranking|retrieval)/.test(content);
      return content.indexOf(topic) >= 0;
    }
    function applyFilters(updateUrl) {
      var terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
      var count = 0;
      cards.forEach(function (card) {
        var hay = ((card.getAttribute('data-search') || '') + ' ' + (card.getAttribute('data-topics') || '')).toLowerCase();
        var visible = terms.every(function (term) {return hay.indexOf(term) !== -1;}) && matchesTopic(hay, activeTopic);
        card.hidden = !visible;
        if (visible) count++;
      });
      filters.forEach(function (button) {
        var active = button.getAttribute('data-topic') === activeTopic;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      if (counter) counter.textContent = 'Showing ' + count + ' of ' + cards.length + ' articles';
      if (noResults) noResults.hidden = count !== 0;
      if (updateUrl && window.history && history.replaceState) {
        var url = new URL(window.location.href);
        if (search.value.trim()) url.searchParams.set('q', search.value.trim()); else url.searchParams.delete('q');
        if (activeTopic !== 'all') url.searchParams.set('topic', activeTopic); else url.searchParams.delete('topic');
        history.replaceState(null, '', url.toString());
      }
    }
    search.addEventListener('input', function () { applyFilters(true); });
    filters.forEach(function (button) {
      button.addEventListener('click', function () { activeTopic = button.getAttribute('data-topic'); applyFilters(true); });
    });
    applyFilters(false);
  }

  var article = document.querySelector('.blog-post');
  var progress = document.getElementById('bm-reading-progress');
  var readTime = document.getElementById('bm-read-time');
  if (article) {
    if (readTime) {
      var words = (article.textContent || '').trim().split(/\s+/).length;
      readTime.textContent = Math.max(1, Math.round(words / 220)) + ' min read (est.)';
    }
    var toc = document.getElementById('bm-toc');
    var tocList = document.getElementById('bm-toc-list');
    if (toc && tocList) {
      var headings = Array.prototype.slice.call(article.querySelectorAll('h2')).filter(function (h) {return h.textContent.trim().length > 4;});
      if (headings.length < 3) toc.hidden = true;
      else headings.forEach(function (h, index) {
        if (!h.id) h.id = 'article-section-' + (index + 1);
        var li = document.createElement('li');
        var a = document.createElement('a');
        a.textContent = h.textContent.trim();
        a.href = '#' + encodeURIComponent(h.id);
        li.appendChild(a); tocList.appendChild(li);
      });
    }
  }
  var copy = document.getElementById('bm-copy-link');
  if (copy) copy.addEventListener('click', function () {
    var original = copy.textContent;
    var reset = function () { setTimeout(function () {copy.textContent = original;}, 1900); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(window.location.href).then(function () {copy.textContent = 'Link copied ✓';reset();}).catch(function () {copy.textContent = 'Use your browser to copy the URL';reset();});
    } else {copy.textContent = 'Copy the URL from your browser';reset();}
  });
  var goTop = document.createElement('button');
  goTop.type = 'button'; goTop.className = 'bm-go-top'; goTop.textContent = '↑';
  goTop.setAttribute('aria-label', 'Back to top');
  goTop.addEventListener('click', function () {window.scrollTo({top:0,behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});});
  document.body.appendChild(goTop);
  var ticking = false;
  function updateOnScroll() {
    ticking = false;
    goTop.classList.toggle('bm-show', window.scrollY > 650);
    if (progress && article) {
      var rect = article.getBoundingClientRect();
      var length = Math.max(1, article.offsetHeight - window.innerHeight * .65);
      var passed = Math.max(0, -rect.top + window.innerHeight * .25);
      progress.style.transform = 'scaleX(' + Math.min(1, Math.max(0, passed / length)) + ')';
    }
  }
  window.addEventListener('scroll', function () {if(!ticking){ticking=true;window.requestAnimationFrame(updateOnScroll);}}, {passive:true});
  updateOnScroll();
})();