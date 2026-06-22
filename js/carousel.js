// Carousel — adapted from React Bits (motion) to dependency-free vanilla.
(function () {
  var GAP = 16, PAD = 16;

  function initCarousel(root) {
    var track = root.querySelector('.carousel-track');
    if (!track) return;
    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

    var loop = root.dataset.loop === 'true';
    var autoplay = root.dataset.autoplay === 'true' && !reduce;
    var delay = parseInt(root.dataset.delay || '3500', 10);
    var pauseOnHover = root.dataset.pause === 'true';

    var baseWidth = root.clientWidth || 340;
    var itemWidth = baseWidth - PAD * 2;
    var off = itemWidth + GAP;
    track.style.gap = GAP + 'px';
    track.style.perspective = '1000px';

    var items = Array.prototype.slice.call(track.querySelectorAll('.carousel-item'));
    var realCount = items.length;
    items.forEach(function (it) { it.style.width = itemWidth + 'px'; });

    if (loop && realCount > 1) {
      var firstC = items[0].cloneNode(true);
      var lastC = items[realCount - 1].cloneNode(true);
      lastC.setAttribute('aria-hidden', 'true');
      firstC.setAttribute('aria-hidden', 'true');
      track.insertBefore(lastC, items[0]);
      track.appendChild(firstC);
      items = Array.prototype.slice.call(track.querySelectorAll('.carousel-item'));
      items.forEach(function (it) { it.style.width = itemWidth + 'px'; });
    }

    var pos = loop ? 1 : 0;
    var x = -pos * off;
    var dragging = false, startX = 0, startTX = 0;

    var dotsWrap = root.querySelector('.carousel-indicators');
    var dots = [];
    if (dotsWrap) {
      for (var i = 0; i < realCount; i++) {
        (function (idx) {
          var d = document.createElement('button');
          d.type = 'button'; d.className = 'carousel-indicator';
          d.setAttribute('aria-label', 'Go to slide ' + (idx + 1));
          d.addEventListener('click', function () { goTo(loop ? idx + 1 : idx); });
          dotsWrap.appendChild(d); dots.push(d);
        })(i);
      }
    }
    function activeIndex() { return loop ? (((pos - 1) % realCount) + realCount) % realCount : Math.min(pos, realCount - 1); }
    function updateDots() { var a = activeIndex(); dots.forEach(function (d, i) { d.classList.toggle('active', i === a); d.classList.toggle('inactive', i !== a); }); }

    function updateRotate() {
      items.forEach(function (it, i) {
        var ry = -90 * (x / off + i);
        if (ry > 90) ry = 90; if (ry < -90) ry = -90;
        it.style.transform = 'rotateY(' + ry.toFixed(2) + 'deg)';
      });
    }
    function applyX(animate) {
      track.style.transition = animate && !reduce ? 'transform 0.5s cubic-bezier(0.25,0.8,0.3,1)' : 'none';
      track.style.transform = 'translateX(' + x + 'px)';
      updateRotate();
    }
    function settle(animate) { x = -pos * off; applyX(animate); updateDots(); }
    function goTo(p) { pos = p; settle(true); }

    track.addEventListener('transitionend', function (e) {
      if (e.propertyName !== 'transform') return;
      if (!loop) return;
      if (pos === items.length - 1) { pos = 1; x = -pos * off; applyX(false); }
      else if (pos === 0) { pos = realCount; x = -pos * off; applyX(false); }
      updateDots();
    });

    // autoplay
    var timer = null, hovered = false;
    function tick() {
      if (pauseOnHover && hovered) return;
      pos = loop ? pos + 1 : (pos + 1 > items.length - 1 ? 0 : pos + 1);
      settle(true);
    }
    function play() { if (!autoplay || items.length <= 1) return; stop(); timer = setInterval(tick, delay); }
    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    root.addEventListener('mouseenter', function () { hovered = true; });
    root.addEventListener('mouseleave', function () { hovered = false; });

    // drag
    track.addEventListener('pointerdown', function (e) {
      dragging = true; startX = e.clientX; startTX = x;
      track.style.transition = 'none';
      try { track.setPointerCapture(e.pointerId); } catch (err) {}
    });
    track.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      x = startTX + (e.clientX - startX);
      track.style.transform = 'translateX(' + x + 'px)';
      updateRotate();
    });
    function endDrag(e) {
      if (!dragging) return; dragging = false;
      var dx = (e.clientX || startX) - startX;
      var dir = dx < -40 ? 1 : dx > 40 ? -1 : 0;
      pos = Math.max(0, Math.min(pos + dir, items.length - 1));
      settle(true);
    }
    track.addEventListener('pointerup', endDrag);
    track.addEventListener('pointercancel', endDrag);

    settle(false);
    play();
  }

  function boot() { document.querySelectorAll('.carousel-container').forEach(initCarousel); }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
