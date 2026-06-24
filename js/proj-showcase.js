// Project showcase — CircularTestimonials adapted to dependency-free vanilla.
(function () {
  function calcGap(w) {
    var minW = 1024, maxW = 1456, minG = 40, maxG = 70;
    if (w <= minW) return minG;
    if (w >= maxW) return maxG;
    return minG + (maxG - minG) * ((w - minW) / (maxW - minW));
  }
  function init(root) {
    var wrap = root.querySelector('.proj-images');
    var imgs = Array.prototype.slice.call(root.querySelectorAll('.proj-images img'));
    var n = imgs.length;
    if (!n) return;
    var nameEl = root.querySelector('.proj-name');
    var desEl = root.querySelector('.proj-designation');
    var quoteEl = root.querySelector('.proj-quote');
    var visitEl = root.querySelector('.proj-visit');
    var dotsWrap = root.querySelector('.proj-dots');
    var active = 0, timer = null, hovered = false;
    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var autoplay = root.dataset.autoplay === 'true' && !reduce;

    var dots = [];
    if (dotsWrap) {
      for (var d = 0; d < n; d++) {
        (function (idx) {
          var b = document.createElement('button');
          b.type = 'button'; b.className = 'proj-dot';
          b.setAttribute('aria-label', 'Show project ' + (idx + 1));
          b.addEventListener('click', function () { go(idx); restart(); });
          dotsWrap.appendChild(b); dots.push(b);
        })(d);
      }
    }

    function place(i) {
      var w = wrap.clientWidth || 480, gap = calcGap(w), up = gap * 0.6;
      var left = (active - 1 + n) % n, right = (active + 1) % n, s = imgs[i].style;
      if (i === active) { s.zIndex = 3; s.opacity = 1; s.pointerEvents = 'auto'; s.transform = 'translateX(0) translateY(0) scale(1) rotateY(0deg)'; }
      else if (i === left) { s.zIndex = 2; s.opacity = 1; s.pointerEvents = 'auto'; s.transform = 'translateX(-' + gap + 'px) translateY(-' + up + 'px) scale(0.85) rotateY(15deg)'; }
      else if (i === right) { s.zIndex = 2; s.opacity = 1; s.pointerEvents = 'auto'; s.transform = 'translateX(' + gap + 'px) translateY(-' + up + 'px) scale(0.85) rotateY(-15deg)'; }
      else { s.zIndex = 1; s.opacity = 0; s.pointerEvents = 'none'; }
    }
    function render() {
      for (var i = 0; i < n; i++) place(i);
      var a = imgs[active];
      if (nameEl) nameEl.textContent = a.dataset.name || '';
      if (desEl) desEl.textContent = a.dataset.designation || '';
      if (visitEl) visitEl.href = a.dataset.href || '#';
      for (var k = 0; k < dots.length; k++) dots[k].classList.toggle('is-active', k === active);
      if (quoteEl) {
        quoteEl.style.opacity = 0; quoteEl.style.transform = 'translateY(8px)';
        setTimeout(function () { quoteEl.textContent = a.dataset.quote || ''; quoteEl.style.opacity = 1; quoteEl.style.transform = 'none'; }, 170);
      }
    }
    function go(i) { active = (i + n) % n; render(); }
    function tick() { if (!hovered) go(active + 1); }
    function restart() { if (timer) clearInterval(timer); if (autoplay) timer = setInterval(tick, 5200); }

    root.querySelectorAll('.proj-arrow').forEach(function (btn) {
      btn.addEventListener('click', function () { go(active + (btn.dataset.dir === 'next' ? 1 : -1)); restart(); });
    });

    // swipe (touch / pointer) on the image stack
    var sx = 0, sy = 0, down = false, moved = false;
    wrap.addEventListener('pointerdown', function (e) { down = true; moved = false; sx = e.clientX; sy = e.clientY; });
    wrap.addEventListener('pointermove', function (e) { if (down && Math.abs(e.clientX - sx) > 8) moved = true; });
    wrap.addEventListener('pointerup', function (e) {
      if (!down) return; down = false;
      var dx = e.clientX - sx, dy = e.clientY - sy;
      if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy)) { go(active + (dx < 0 ? 1 : -1)); restart(); }
    });
    wrap.addEventListener('pointercancel', function () { down = false; });

    imgs.forEach(function (img, i) {
      img.addEventListener('click', function () {
        if (moved) { moved = false; return; }      // ignore clicks that were swipes
        if (i === active) { if (img.dataset.href) window.open(img.dataset.href, '_blank', 'noopener'); }
        else { go(i); restart(); }
      });
    });
    root.addEventListener('mouseenter', function () { hovered = true; });
    root.addEventListener('mouseleave', function () { hovered = false; });
    window.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') { go(active - 1); restart(); }
      else if (e.key === 'ArrowRight') { go(active + 1); restart(); }
    });
    var rt; window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(render, 150); });
    render(); restart();
  }
  function boot() { document.querySelectorAll('.proj-show').forEach(init); }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
