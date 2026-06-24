// VariableProximity — React Bits effect adapted to dependency-free vanilla.
// Letters interpolate font-variation-settings (weight/opsz) by cursor proximity.
(function () {
  function parse(s) {
    return s.split(',').map(function (x) { return x.trim(); }).filter(Boolean).map(function (x) {
      var p = x.split(/\s+/); return { axis: p[0].replace(/['"]/g, ''), val: parseFloat(p[1]) };
    });
  }
  function init(el) {
    if (el.dataset.vpReady) return; el.dataset.vpReady = '1';
    var from = el.getAttribute('data-from') || "'wght' 400, 'opsz' 12";
    var to = el.getAttribute('data-to') || "'wght' 1000, 'opsz' 40";
    var radius = parseFloat(el.getAttribute('data-radius') || '120');
    var falloff = el.getAttribute('data-falloff') || 'linear';
    var fromA = parse(from), toA = parse(to);
    var text = el.textContent;
    el.textContent = '';
    var letters = [], frag = document.createDocumentFragment();
    text.split('').forEach(function (ch) {
      if (ch === ' ') { frag.appendChild(document.createTextNode(' ')); return; }
      var sp = document.createElement('span'); sp.className = 'vp-l'; sp.textContent = ch;
      sp.style.fontVariationSettings = from; sp.setAttribute('aria-hidden', 'true');
      frag.appendChild(sp); letters.push(sp);
    });
    el.appendChild(frag);
    var sr = document.createElement('span'); sr.className = 'sr-only'; sr.textContent = text; el.appendChild(sr);
    el.setAttribute('aria-label', text);

    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) return;
    var mx = -9999, my = -9999, active = false;
    window.addEventListener('mousemove', function (e) { mx = e.clientX; my = e.clientY; active = true; }, { passive: true });
    window.addEventListener('touchmove', function (e) { var t = e.touches[0]; if (t) { mx = t.clientX; my = t.clientY; active = true; } }, { passive: true });

    function loop() {
      if (active) {
        for (var i = 0; i < letters.length; i++) {
          var r = letters[i].getBoundingClientRect();
          var cx = r.left + r.width / 2, cy = r.top + r.height / 2;
          var d = Math.sqrt((mx - cx) * (mx - cx) + (my - cy) * (my - cy));
          if (d >= radius) { letters[i].style.fontVariationSettings = from; continue; }
          var n = Math.min(Math.max(1 - d / radius, 0), 1);
          var f = falloff === 'exponential' ? n * n : falloff === 'gaussian' ? Math.exp(-Math.pow(d / (radius / 2), 2) / 2) : n;
          var s = fromA.map(function (a, idx) {
            var tv = (toA[idx] && toA[idx].axis === a.axis) ? toA[idx].val : a.val;
            return "'" + a.axis + "' " + (a.val + (tv - a.val) * f).toFixed(1);
          }).join(', ');
          letters[i].style.fontVariationSettings = s;
        }
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }
  function boot() { document.querySelectorAll('.vproximity').forEach(init); }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
