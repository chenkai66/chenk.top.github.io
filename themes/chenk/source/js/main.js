/* ============================================
   CHENK THEME — Main JavaScript
   ============================================ */

(function() {
  'use strict';

  // === Dark Mode ===
  var themeToggle = document.getElementById('theme-toggle');
  var mobileThemeToggle = document.getElementById('mobile-theme-toggle');
  var html = document.documentElement;

  function getPreferredTheme() {
    var saved = localStorage.getItem('theme');
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function setTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }

  // Already set in <head> script to prevent FOUC, but ensure consistency
  setTheme(getPreferredTheme());

  function toggleTheme() {
    var current = html.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
  }

  if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
  if (mobileThemeToggle) mobileThemeToggle.addEventListener('click', toggleTheme);

  // === Reading Progress Bar ===
  var progressBar = document.getElementById('reading-progress');
  if (progressBar) {
    window.addEventListener('scroll', function() {
      var winHeight = document.documentElement.scrollHeight - window.innerHeight;
      if (winHeight > 0) {
        var scrolled = (window.scrollY / winHeight) * 100;
        progressBar.style.width = Math.min(scrolled, 100) + '%';
      }
    }, { passive: true });
  }

  // === Header scrolled state (subtle bg + shadow when not at top) ===
  var siteHeader = document.querySelector('.site-header');
  if (siteHeader) {
    var headerScrollHandler = function() {
      if (window.scrollY > 8) {
        siteHeader.classList.add('scrolled');
      } else {
        siteHeader.classList.remove('scrolled');
      }
    };
    window.addEventListener('scroll', headerScrollHandler, { passive: true });
    headerScrollHandler();
  }

  // === Back to Top (smooth animation) ===
  var backToTop = document.getElementById('back-to-top');
  if (backToTop) {
    window.addEventListener('scroll', function() {
      if (window.scrollY > 400) {
        backToTop.classList.add('visible');
      } else {
        backToTop.classList.remove('visible');
      }
    }, { passive: true });

    backToTop.addEventListener('click', function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // === Code Blocks: Copy Button + Language Label ===
  // Handle BOTH plain <pre><code> and Hexo's <figure class="highlight {lang}"><table>...
  document.querySelectorAll('.markdown-body figure.highlight').forEach(function(fig) {
    // Extract language from class names: "highlight python" -> "python"
    var langs = Array.from(fig.classList).filter(function(c){ return c !== 'highlight'; });
    var lang = langs[0] || '';
    if (lang) {
      var tag = document.createElement('span');
      tag.className = 'code-lang-tag';
      tag.textContent = lang;
      fig.appendChild(tag);
    }
    var codeCell = fig.querySelector('td.code');
    var codeText = codeCell ? codeCell.innerText.replace(/\n+$/,'') : fig.innerText;
    var btn = document.createElement('button');
    btn.className = 'code-copy-btn';
    btn.setAttribute('aria-label', 'Copy code');
    btn.innerHTML = '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    btn.addEventListener('click', function(){
      navigator.clipboard.writeText(codeText).then(function(){
        btn.classList.add('copied');
        setTimeout(function(){ btn.classList.remove('copied'); }, 2000);
      }).catch(function(){
        var ta = document.createElement('textarea'); ta.value = codeText;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch(e) {}
        document.body.removeChild(ta);
        btn.classList.add('copied');
        setTimeout(function(){ btn.classList.remove('copied'); }, 2000);
      });
    });
    fig.appendChild(btn);
  });

  document.querySelectorAll('.markdown-body pre').forEach(function(pre) {
    // Skip pre inside hexo figure.highlight (handled above)
    if (pre.closest('figure.highlight')) return;
    var code = pre.querySelector('code');
    if (!code) return;

    // Extract language from class (e.g. "language-python" or "hljs language-python")
    var langClass = code.className.match(/language-(\w+)/);
    var lang = langClass ? langClass[1] : null;

    // Add language label
    if (lang) {
      pre.setAttribute('data-lang', lang);
      var langLabel = document.createElement('span');
      langLabel.className = 'code-lang-label';
      langLabel.setAttribute('data-lang', lang);
      langLabel.textContent = lang;
      pre.appendChild(langLabel);
    }

    // Add copy button with SVG icon
    var btn = document.createElement('button');
    btn.className = 'code-copy-btn';
    btn.setAttribute('aria-label', 'Copy code');
    btn.innerHTML = '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

    btn.addEventListener('click', function() {
      var text = code.textContent;
      navigator.clipboard.writeText(text).then(function() {
        btn.classList.add('copied');
        setTimeout(function() {
          btn.classList.remove('copied');
        }, 2000);
      }).catch(function() {
        // Fallback for older browsers
        var textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        btn.classList.add('copied');
        setTimeout(function() { btn.classList.remove('copied'); }, 2000);
      });
    });

    pre.style.position = 'relative';
    pre.appendChild(btn);
  });

  // === Callout / Admonition Detection ===
  // Transform blockquotes starting with **Note:**, **Warning:**, **Tip:**, **Danger:**, **Important:**
  document.querySelectorAll('.markdown-body blockquote').forEach(function(bq) {
    var firstP = bq.querySelector('p:first-child');
    if (!firstP) return;

    var firstStrong = firstP.querySelector('strong:first-child');
    if (!firstStrong) return;

    var text = firstStrong.textContent.trim().toLowerCase().replace(/:$/, '');
    var calloutType = null;
    var icon = '';

    if (text === 'note' || text === 'info') {
      calloutType = 'note';
      icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>';
    } else if (text === 'warning' || text === 'caution') {
      calloutType = 'warning';
      icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/></svg>';
    } else if (text === 'tip' || text === 'hint') {
      calloutType = 'tip';
      icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/></svg>';
    } else if (text === 'danger' || text === 'important') {
      calloutType = 'danger';
      icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    }

    if (calloutType) {
      bq.classList.add('callout-' + calloutType);

      // Restructure content
      var titleDiv = document.createElement('div');
      titleDiv.className = 'callout-title';
      titleDiv.innerHTML = icon + '<span>' + firstStrong.textContent + '</span>';

      var contentDiv = document.createElement('div');
      contentDiv.className = 'callout-content';

      // Move the rest of the first paragraph content (after the strong tag)
      var remainingHTML = firstP.innerHTML.replace(firstStrong.outerHTML, '').replace(/^[\s:]+/, '');
      if (remainingHTML.trim()) {
        var restP = document.createElement('p');
        restP.innerHTML = remainingHTML;
        contentDiv.appendChild(restP);
      }

      // Move remaining children
      var children = Array.from(bq.children);
      children.forEach(function(child) {
        if (child !== firstP) {
          contentDiv.appendChild(child);
        }
      });

      bq.innerHTML = '';
      bq.appendChild(titleDiv);
      bq.appendChild(contentDiv);
    }
  });

  // === Table Wrapper for horizontal scroll ===
  document.querySelectorAll('.markdown-body table').forEach(function(table) {
    if (table.parentElement.classList.contains('table-wrapper')) return;
    var wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
    table.style.display = 'table';
  });

  // === Lazy loading for images ===
  document.querySelectorAll('.markdown-body img').forEach(function(img) {
    if (!img.getAttribute('loading')) {
      img.setAttribute('loading', 'lazy');
    }
  });

  // === TOC Scroll-Spy with Smooth Scroll ===
  var tocLinks = document.querySelectorAll('.toc-sidebar a');
  if (tocLinks.length > 0) {
    var headings = [];
    tocLinks.forEach(function(link) {
      var href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        var heading = document.getElementById(decodeURIComponent(href.slice(1)));
        if (heading) headings.push({ el: heading, link: link });
      }
    });

    // Smooth scroll on TOC click
    tocLinks.forEach(function(link) {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        var href = link.getAttribute('href');
        if (href && href.startsWith('#')) {
          var target = document.getElementById(decodeURIComponent(href.slice(1)));
          if (target) {
            var headerOffset = 80;
            var elementPosition = target.getBoundingClientRect().top;
            var offsetPosition = elementPosition + window.scrollY - headerOffset;
            window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
            // Update URL hash without jumping
            history.pushState(null, null, href);
          }
        }
      });
    });

    // Scroll spy
    if (headings.length > 0) {
      var scrollSpyTimer;
      window.addEventListener('scroll', function() {
        if (scrollSpyTimer) return;
        scrollSpyTimer = requestAnimationFrame(function() {
          scrollSpyTimer = null;
          var scrollY = window.scrollY + 100;
          var active = headings[0];

          for (var i = 0; i < headings.length; i++) {
            if (headings[i].el.offsetTop <= scrollY) {
              active = headings[i];
            }
          }

          tocLinks.forEach(function(l) { l.classList.remove('active'); });
          if (active) {
            active.link.classList.add('active');
            // Scroll the TOC sidebar to keep active link visible
            var tocSidebar = document.getElementById('toc-sidebar');
            if (tocSidebar) {
              var linkRect = active.link.getBoundingClientRect();
              var sidebarRect = tocSidebar.getBoundingClientRect();
              if (linkRect.top < sidebarRect.top || linkRect.bottom > sidebarRect.bottom) {
                active.link.scrollIntoView({ block: 'center', behavior: 'smooth' });
              }
            }
          }
        });
      }, { passive: true });
    }
  }

  // === Image Lightbox ===
  var lightboxOverlay = null;

  function createLightbox() {
    lightboxOverlay = document.createElement('div');
    lightboxOverlay.className = 'lightbox-overlay';
    lightboxOverlay.addEventListener('click', closeLightbox);
    document.body.appendChild(lightboxOverlay);
  }

  function openLightbox(src, alt) {
    if (!lightboxOverlay) createLightbox();
    lightboxOverlay.innerHTML = '<img src="' + src + '" alt="' + (alt || '') + '">';
    // Force reflow before adding active class for animation
    lightboxOverlay.offsetHeight;
    lightboxOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    if (lightboxOverlay) {
      lightboxOverlay.classList.remove('active');
      document.body.style.overflow = '';
      setTimeout(function() {
        lightboxOverlay.innerHTML = '';
      }, 300);
    }
  }

  document.querySelectorAll('.markdown-body img').forEach(function(img) {
    // Skip small images and icons
    if (img.naturalWidth && img.naturalWidth < 100) return;
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', function(e) {
      e.preventDefault();
      openLightbox(img.src, img.alt);
    });
  });

  // === Search Toggle ===
  var searchToggle = document.getElementById('search-toggle');
  var searchOverlay = document.getElementById('search-overlay');
  var searchInput = document.getElementById('search-input');

  function openSearch() {
    if (searchOverlay) {
      searchOverlay.classList.add('active');
      if (searchInput) {
        setTimeout(function() { searchInput.focus(); }, 100);
      }
    }
  }

  function closeSearch() {
    if (searchOverlay) {
      searchOverlay.classList.remove('active');
      if (searchInput) searchInput.value = '';
    }
  }

  if (searchToggle && searchOverlay) {
    searchToggle.addEventListener('click', openSearch);

    searchOverlay.addEventListener('click', function(e) {
      if (e.target === searchOverlay) closeSearch();
    });
  }

  // === Keyboard Shortcuts ===
  document.addEventListener('keydown', function(e) {
    // Escape: close overlays
    if (e.key === 'Escape') {
      closeSearch();
      closeLightbox();
      closeMobileDrawer();
    }

    // Cmd/Ctrl+K: toggle search
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      if (searchOverlay && searchOverlay.classList.contains('active')) {
        closeSearch();
      } else {
        openSearch();
      }
    }

    // / key: open search (only when not typing in input)
    if (e.key === '/' && !isTyping(e)) {
      e.preventDefault();
      openSearch();
    }
  });

  function isTyping(e) {
    var el = e.target || e.srcElement;
    return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable;
  }

  // === Mobile Navigation Drawer ===
  var mobileToggle = document.getElementById('mobile-menu-toggle');
  var mobileDrawer = document.getElementById('mobile-drawer');
  var mobileDrawerOverlay = document.getElementById('mobile-drawer-overlay');
  var mobileDrawerClose = document.getElementById('mobile-drawer-close');

  function openMobileDrawer() {
    if (mobileDrawer) mobileDrawer.classList.add('active');
    if (mobileDrawerOverlay) mobileDrawerOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeMobileDrawer() {
    if (mobileDrawer) mobileDrawer.classList.remove('active');
    if (mobileDrawerOverlay) mobileDrawerOverlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (mobileToggle) mobileToggle.addEventListener('click', openMobileDrawer);
  if (mobileDrawerClose) mobileDrawerClose.addEventListener('click', closeMobileDrawer);
  if (mobileDrawerOverlay) mobileDrawerOverlay.addEventListener('click', closeMobileDrawer);

  // === Language Switcher ===
  var langSwitcher = document.getElementById('lang-switcher');
  var langLabel = document.getElementById('lang-label');
  var mobileLangSwitcher = document.getElementById('mobile-lang-switcher');
  var mobileLangLabel = document.getElementById('mobile-lang-label');

  function getPageLang() {
    // Hard language of the current URL (for paired-URL switching).
    var pl = document.documentElement.getAttribute('data-page-lang') || '';
    if (pl) return pl.indexOf('zh') === 0 ? 'zh' : 'en';
    var path = window.location.pathname;
    if (path.indexOf('/zh/') === 0) return 'zh';
    if (path.indexOf('/en/') === 0) return 'en';
    return null;
  }

  function getSiteLang() {
    // The user's *preferred* site language. Used for filtering listing pages
    // and for deciding what the lang switcher button should display next.
    var saved = localStorage.getItem('siteLang');
    if (saved === 'zh' || saved === 'en') return saved;
    // Fall back to page-lang (for first-time visitors landing on a translated post)
    return getPageLang() || 'en';
  }

  function getCurrentLang() { return getSiteLang(); }

  function switchLanguage() {
    var path = window.location.pathname;
    var current = getSiteLang();
    var next = current === 'en' ? 'zh' : 'en';

    // Persist the user's choice immediately.
    localStorage.setItem('siteLang', next);

    // 1) Paired translation URL set by Hexo helper — preferred for article pages.
    var paired = document.documentElement.getAttribute('data-translation');
    if (paired && paired !== '') {
      window.location.href = paired;
      return;
    }

    // 2) /en/... ↔ /zh/... path swap fallback for article pages without a pair.
    if (path.indexOf('/zh/') === 0) {
      window.location.href = path.replace('/zh/', '/en/');
      return;
    }
    if (path.indexOf('/en/') === 0) {
      window.location.href = path.replace('/en/', '/zh/');
      return;
    }

    // 3) Non-localized page (home, archive, me, projects, series, about, …):
    //    just reload — the JS will re-apply the filter & update the label.
    window.location.reload();
  }

  if (langSwitcher) langSwitcher.addEventListener('click', switchLanguage);
  if (mobileLangSwitcher) mobileLangSwitcher.addEventListener('click', switchLanguage);

  // Update language labels — show the language the user will switch TO,
  // so the button reads as an action ("EN" = click to go English).
  function updateLangLabels() {
    var current = getCurrentLang();
    var next = current === 'en' ? 'CN' : 'EN';
    if (langLabel) langLabel.textContent = next;
    if (mobileLangLabel) mobileLangLabel.textContent = next;
  }
  updateLangLabels();

  // === Per-language content filtering on listing pages ===
  // Hides post cards / archive items that don't match the user's siteLang.
  // ONLY applied on home + archive pages (not category/tag pages, since those
  // already cluster posts by category/tag name and should show all matches).
  // If the filter would empty the page, fall back to showing all.
  function applyLangFilter() {
    var path = window.location.pathname;
    // Only filter on root home + archives + tags index + categories index — never on
    // a specific category/tag page where posts are already grouped.
    var isFilterable = path === '/' || path === '/index.html'
      || /^\/(en|zh)\/?$/.test(path)
      || /^\/page\/\d+\/?$/.test(path)
      || /^\/archives\/?$/.test(path);
    if (!isFilterable) return;

    var siteLang = localStorage.getItem('siteLang') || getCurrentLang();
    var wantZh = siteLang === 'zh';
    var items = document.querySelectorAll('[data-post-lang]');
    if (items.length === 0) return;

    var visible = 0;
    items.forEach(function(el) {
      var pl = (el.getAttribute('data-post-lang') || '').toLowerCase();
      var isZh = pl.indexOf('zh') === 0;
      var match = (isZh === wantZh);
      el.style.display = match ? '' : 'none';
      if (match) visible++;
    });

    // If we hid everything, restore — better to show all than empty
    if (visible === 0) {
      items.forEach(function(el) { el.style.display = ''; });
    }
  }
  applyLangFilter();


})();
