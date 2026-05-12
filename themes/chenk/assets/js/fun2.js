// More fun: keyboard shortcuts (g/G/r/?), help panel, image lightbox.
(function () {
  function isLangZh() { return (document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0; }

  // ---------- Helper: are we typing in an input? ----------
  function isTyping(e) {
    var t = e.target;
    var tag = (t && t.tagName) || '';
    return tag === 'INPUT' || tag === 'TEXTAREA' || (t && t.isContentEditable);
  }

  // ---------- Keyboard shortcuts ----------
  var lastG = 0;
  document.addEventListener('keydown', function (e) {
    if (isTyping(e)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    // h → home
    if (e.key === 'h') {
      var lang = (document.documentElement.lang || 'en').toLowerCase().slice(0, 2);
      window.location.href = '/' + lang + '/';
      return;
    }
    // t → toggle theme
    if (e.key === 't') {
      var btn = document.querySelector('[data-theme-toggle]');
      if (btn) btn.click();
      return;
    }
    // [ / ] → previous / next in series
    if (e.key === '[' || e.key === ']') {
      var sel = e.key === '['
        ? '.series-nav-prev-next .prev-item a, .series-nav .prev a'
        : '.series-nav-prev-next .next-item a, .series-nav .next a';
      var link = document.querySelector(sel);
      if (link) {
        flashToast(e.key === '[' ? (isLangZh() ? '← 上一篇' : '← Previous') : (isLangZh() ? '下一篇 →' : 'Next →'));
        setTimeout(function () { window.location.href = link.href; }, 250);
      } else {
        flashToast(isLangZh() ? '没有更多了' : 'End of series');
      }
      return;
    }
    // gg → top
    if (e.key === 'g') {
      var now = Date.now();
      if (now - lastG < 600) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        lastG = 0;
      } else {
        lastG = now;
      }
      return;
    }
    // G → bottom
    if (e.key === 'G') {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      return;
    }
    // r → random article
    if (e.key === 'r') {
      jumpToRandomArticle();
      return;
    }
    // ? → help panel
    if (e.key === '?') {
      toggleHelp();
      return;
    }
    // Esc → close help if open
    if (e.key === 'Escape') {
      var p = document.getElementById('shortcut-help');
      if (p && p.classList.contains('open')) p.classList.remove('open');
    }
  });

  // ---------- Random article ----------
  function jumpToRandomArticle() {
    // Try to fetch the language-specific search index (already exists for /search)
    var lang = (document.documentElement.lang || 'en').toLowerCase().slice(0, 2);
    var idxUrl = '/' + lang + '/index.json';
    fetch(idxUrl, { cache: 'force-cache' }).then(function (r) {
      if (!r.ok) throw new Error('no index');
      return r.json();
    }).then(function (data) {
      var posts = (data && data.posts) || data || [];
      if (!Array.isArray(posts) || !posts.length) return;
      var pick = posts[Math.floor(Math.random() * posts.length)];
      var url = pick.permalink || pick.url || pick.href;
      if (url) {
        flashToast(lang === 'zh' ? '🎲 随机跳转中…' : '🎲 Roll the dice…');
        setTimeout(function () { window.location.href = url; }, 350);
      }
    }).catch(function () {
      flashToast(lang === 'zh' ? '随机跳转暂不可用' : 'Random jump unavailable');
    });
  }

  // ---------- Help panel ----------
  function toggleHelp() {
    var panel = document.getElementById('shortcut-help');
    if (!panel) {
      panel = buildHelpPanel();
      document.body.appendChild(panel);
    }
    panel.classList.toggle('open');
  }

  function buildHelpPanel() {
    var isZh = (document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0;
    var p = document.createElement('div');
    p.id = 'shortcut-help';
    p.innerHTML =
      '<div class="sh-card">' +
        '<div class="sh-head"><span>' + (isZh ? '键盘快捷键' : 'Keyboard shortcuts') + '</span><button class="sh-close" aria-label="Close">×</button></div>' +
        '<dl class="sh-list">' +
          row('g g', isZh ? '回到顶部' : 'Go to top') +
          row('Shift + G', isZh ? '跳到底部' : 'Go to bottom') +
          row('r', isZh ? '随机一篇文章' : 'Random article') +
          row('h', isZh ? '回到首页' : 'Back to home') +
          row('[', isZh ? '上一篇（同系列）' : 'Previous in series') +
          row(']', isZh ? '下一篇（同系列）' : 'Next in series') +
          row('t', isZh ? '切换主题' : 'Toggle theme') +
          row('⌘ K  /  Ctrl K', isZh ? '搜索' : 'Search') +
          row('Shift + ?', isZh ? '显示这个面板' : 'Show this panel') +
          row('Esc', isZh ? '关闭面板' : 'Close panel') +
          row('↑ ↑ ↓ ↓', isZh ? '彩蛋 ✨' : 'Easter egg ✨') +
        '</dl>' +
        '<div class="sh-foot">' + (isZh ? '随时按 ? 调出' : 'Press ? anytime') + '</div>' +
      '</div>';
    p.addEventListener('click', function (e) {
      if (e.target === p || e.target.classList.contains('sh-close')) p.classList.remove('open');
    });
    return p;
  }
  function row(k, label) {
    return '<dt><kbd>' + k.replace(/ /g, '</kbd> <kbd>') + '</kbd></dt><dd>' + label + '</dd>';
  }

  function flashToast(msg) {
    var t = document.createElement('div');
    t.className = 'tiny-toast';
    t.textContent = msg;
    document.body.appendChild(t);
    requestAnimationFrame(function () { t.classList.add('show'); });
    setTimeout(function () { t.classList.remove('show'); setTimeout(function () { t.remove(); }, 300); }, 1100);
  }

  // ---------- Image lightbox ----------
  var lb = null;
  function openLightbox(src, alt) {
    if (!lb) {
      lb = document.createElement('div');
      lb.id = 'lightbox';
      lb.innerHTML = '<img alt=""><div class="lb-cap"></div>';
      lb.addEventListener('click', function () { lb.classList.remove('open'); });
      document.body.appendChild(lb);
    }
    lb.querySelector('img').src = src;
    lb.querySelector('img').alt = alt || '';
    lb.querySelector('.lb-cap').textContent = alt || '';
    lb.classList.add('open');
  }

  document.addEventListener('click', function (e) {
    var img = e.target.closest('article .prose img, .article-main img');
    if (!img) return;
    if (img.closest('a')) return; // image is wrapped in a link, leave it
    e.preventDefault();
    openLightbox(img.currentSrc || img.src, img.alt);
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && lb && lb.classList.contains('open')) {
      lb.classList.remove('open');
    }
  });

  // ---------- Floating ? button (always visible, bottom-right) ----------
  function createHelpButton() {
    if (document.getElementById('kbd-fab')) return;
    var b = document.createElement('button');
    b.id = 'kbd-fab';
    b.type = 'button';
    b.setAttribute('aria-label', isLangZh() ? '键盘快捷键' : 'Keyboard shortcuts');
    b.title = (isLangZh() ? '按 Shift + ? 打开' : 'Press Shift + ? to open');
    b.textContent = '?';
    b.addEventListener('click', toggleHelp);
    document.body.appendChild(b);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createHelpButton);
  } else {
    createHelpButton();
  }
})();
