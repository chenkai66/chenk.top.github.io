// Hero stats: count-up animation when visible
(function() {
  function easeOutQuad(t) { return t * (2 - t); }
  function animateNum(el) {
    var target = parseInt(el.dataset.target, 10);
    var suffix = el.dataset.suffix || "";
    var duration = 1200;
    var startTime = performance.now();
    function tick(now) {
      var elapsed = now - startTime;
      var progress = Math.min(elapsed / duration, 1);
      var eased = easeOutQuad(progress);
      var value = Math.floor(target * eased);
      el.textContent = value + suffix;
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target + suffix;
    }
    requestAnimationFrame(tick);
  }
  function animateDays(el) {
    var since = new Date(el.dataset.daysSince);
    var now = new Date();
    var days = Math.floor((now - since) / (1000 * 60 * 60 * 24));
    el.dataset.target = days;
    animateNum(el);
  }
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        var el = entry.target;
        if (el.dataset.daysSince) animateDays(el);
        else if (el.dataset.target) animateNum(el);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.4 });
  document.querySelectorAll(".hero-stats .num").forEach(function(el) {
    observer.observe(el);
  });
})();
