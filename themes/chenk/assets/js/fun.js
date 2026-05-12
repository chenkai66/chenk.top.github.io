// Fun stuff: typewriter hero, Konami code, days-ago, dark-mode transition.
(function () {
  // ===== Days-ago label on article-end =====
  document.querySelectorAll('[data-days-since]').forEach(function (el) {
    var since = new Date(el.dataset.daysSince);
    var now = new Date();
    var days = Math.max(0, Math.floor((now - since) / (1000 * 60 * 60 * 24)));
    var slot = el.querySelector('.days');
    if (slot) slot.textContent = days;
  });

  // ===== Smooth dark-mode transition =====
  // Apply a temporary CSS transition class so theme toggle fades nicely.
  var style = document.createElement('style');
  style.textContent = '.theme-fading, .theme-fading *, .theme-fading *::before, .theme-fading *::after { transition: background-color .35s ease, color .35s ease, border-color .35s ease, box-shadow .35s ease !important; }';
  document.head.appendChild(style);
  // Hook into existing theme.js: it uses [data-theme-toggle] click handler that calls setTheme().
  // We add our own listener on the same element to add/remove the .theme-fading class.
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-theme-toggle]')) {
      document.body.classList.add('theme-fading');
      setTimeout(function () { document.body.classList.remove('theme-fading'); }, 500);
    }
  });

  // ===== Typewriter effect on home hero (only on first visit per session) =====
  var hero = document.querySelector('.home-page .hero h1');
  if (hero && !sessionStorage.getItem('typewriter-played')) {
    var html = hero.innerHTML;
    // Skip if already animated or no <span class="accent">
    if (html.indexOf('typewriter-played') === -1) {
      // Convert HTML to a sequence of node operations
      var segments = parseHero(hero);
      hero.innerHTML = '';
      hero.classList.add('typewriter');
      typeSegments(hero, segments, 0, function () {
        hero.classList.remove('typewriter');
        sessionStorage.setItem('typewriter-played', '1');
      });
    }
  }

  function parseHero(el) {
    var out = [];
    el.childNodes.forEach(function (n) {
      if (n.nodeType === Node.TEXT_NODE) {
        out.push({ type: 'text', value: n.textContent });
      } else if (n.nodeType === Node.ELEMENT_NODE) {
        out.push({ type: 'tag', tag: n.tagName, className: n.className, value: n.textContent });
      }
    });
    return out;
  }

  function typeSegments(target, segments, idx, done) {
    if (idx >= segments.length) { done(); return; }
    var seg = segments[idx];
    var container;
    if (seg.type === 'text') container = target;
    else {
      container = document.createElement(seg.tag);
      if (seg.className) container.className = seg.className;
      target.appendChild(container);
    }
    var i = 0;
    var text = seg.value;
    function step() {
      if (i < text.length) {
        container.appendChild(document.createTextNode(text[i]));
        i++;
        setTimeout(step, text[i - 1] === ' ' ? 18 : (Math.random() * 24 + 14));
      } else {
        typeSegments(target, segments, idx + 1, done);
      }
    }
    step();
  }

  // ===== Konami code easter egg =====
  // ↑ ↑ ↓ ↓ ← → ← → B A → triggers a celebration + secret message
  var SEQ = ['ArrowUp','ArrowUp','ArrowDown','ArrowDown'];
  var pos = 0;
  document.addEventListener('keydown', function (e) {
    var key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (key === SEQ[pos]) {
      pos++;
      if (pos === SEQ.length) {
        pos = 0;
        triggerKonami();
      }
    } else {
      pos = key === SEQ[0] ? 1 : 0;
    }
  });

  function triggerKonami() {
    // 1) Confetti emoji burst
    var emojis = ['📐','🧮','🪐','🎲','🧠','📚','💡','🚀','✨','🎯'];
    for (var i = 0; i < 36; i++) {
      var span = document.createElement('span');
      span.textContent = emojis[Math.floor(Math.random() * emojis.length)];
      span.style.cssText =
        'position:fixed;top:-40px;left:' + (Math.random() * 100) + 'vw;' +
        'font-size:' + (16 + Math.random() * 24) + 'px;' +
        'pointer-events:none;z-index:99999;' +
        'transform:translateY(0) rotate(0deg);' +
        'transition:transform ' + (1.6 + Math.random() * 1.4) + 's ease-in,opacity ' + (1.4 + Math.random()) + 's ease-in;';
      document.body.appendChild(span);
      requestAnimationFrame(function (s) {
        return function () {
          s.style.transform = 'translateY(110vh) rotate(' + (Math.random() * 720 - 360) + 'deg)';
          s.style.opacity = '0';
        };
      }(span));
      setTimeout((function (s) { return function () { s.remove(); }; })(span), 3500);
    }
    // 2) Friendly toast
    var toast = document.createElement('div');
    toast.textContent = '↑↑↓↓ — you found it. 🎉';
    toast.style.cssText =
      'position:fixed;left:50%;bottom:32px;transform:translateX(-50%);' +
      'background:var(--ink,#1f2933);color:var(--bg,#fdfcf9);' +
      'padding:14px 22px;border-radius:999px;font-family:var(--font-sans),system-ui;' +
      'font-size:14px;font-weight:600;box-shadow:0 8px 24px rgba(0,0,0,.18);' +
      'z-index:99999;opacity:0;transition:opacity .3s ease,transform .3s ease;';
    document.body.appendChild(toast);
    requestAnimationFrame(function () { toast.style.opacity = '1'; toast.style.transform = 'translateX(-50%) translateY(-6px)'; });
    setTimeout(function () { toast.style.opacity = '0'; setTimeout(function () { toast.remove(); }, 400); }, 4200);
  }

  // ===== Console banner for the curious =====
  if (window.console && console.log) {
    var bannerStyle = 'background:#1f6feb;color:#fff;padding:8px 12px;border-radius:4px;font-size:13px;font-family:monospace;';
    console.log('%cHello, fellow developer 👋', bannerStyle);
    console.log('%cBlog source: github.com/chenkai66', 'color:#999;font-family:monospace;');
    console.log('%cTry ↑↑↓↓ anywhere on the site for a surprise.', 'color:#999;font-family:monospace;font-style:italic;');
  }
})();
