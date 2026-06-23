// ContainerScroll — Aceternity scroll-reveal adapted to dependency-free vanilla.
// The card starts tilted (rotateX) and flattens as it scrolls to center.
(function () {
  function init(section) {
    var card = section.querySelector('.cs-card');
    var header = section.querySelector('.cs-header');
    if (!card) return;
    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { card.style.transform = 'none'; return; }
    var ticking = false;
    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
    function update() {
      ticking = false;
      var vh = window.innerHeight;
      var r = card.getBoundingClientRect();
      var p = clamp((vh - r.top) / (vh * 0.85), 0, 1);
      var mobile = window.innerWidth <= 768;
      var rot = (20 * (1 - p));
      var scale = mobile ? (0.88 + 0.12 * p) : (1.05 - 0.05 * p);
      card.style.transform = 'rotateX(' + rot.toFixed(2) + 'deg) scale(' + scale.toFixed(3) + ')';
      if (header) header.style.transform = 'translateY(' + (-50 * p).toFixed(1) + 'px)';
    }
    function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(update); } }
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    update();
  }
  function boot() { document.querySelectorAll('.cs-section').forEach(init); }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
