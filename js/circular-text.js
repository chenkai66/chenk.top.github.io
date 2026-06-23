// CircularText — adapted from React Bits to dependency-free vanilla.
// Lays out letters around a circle; CSS handles the spin + hover speed-up.
(function () {
  function build(el) {
    if (el.dataset.built) return;
    el.dataset.built = '1';
    var chars = Array.from(el.getAttribute('data-text') || '');
    var n = chars.length;
    if (!n) return;
    var r = (el.clientWidth / 2) - 16;
    if (r < 40) r = 40;
    var frag = document.createDocumentFragment();
    for (var i = 0; i < n; i++) {
      var s = document.createElement('span');
      var deg = (360 / n) * i;
      s.style.transform = 'translate(-50%,-50%) rotate(' + deg + 'deg) translateY(-' + r + 'px)';
      s.textContent = chars[i] === ' ' ? ' ' : chars[i];
      frag.appendChild(s);
    }
    el.appendChild(frag);
  }
  function boot() { document.querySelectorAll('.circular-ring[data-text]').forEach(build); }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
