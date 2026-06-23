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
    var active = 0, timer = null;
    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var autoplay = root.dataset.autoplay === 'true' && !reduce;

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
      if (quoteEl) {
        quoteEl.style.opacity = 0; quoteEl.style.transform = 'translateY(8px)';
        setTimeout(function () { quoteEl.textContent = a.dataset.quote || ''; quoteEl.style.opacity = 1; quoteEl.style.transform = 'none'; }, 170);
      }
    }
    function go(i) { active = (i + n) % n; render(); }
    function restart() { if (timer) clearInterval(timer); if (autoplay) timer = setInterval(function () { go(active + 1); }, 5200); }

    root.querySelectorAll('.proj-arrow').forEach(function (btn) {
      btn.addEventListener('click', function () { go(active + (btn.dataset.dir === 'next' ? 1 : -1)); restart(); });
    });
    imgs.forEach(function (img, i) {
      img.addEventListener('click', function () {
        if (i === active) { if (img.dataset.href) window.open(img.dataset.href, '_blank', 'noopener'); }
        else { go(i); restart(); }
      });
    });
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
