/* projects filter pills */
(function () {
  function init() {
    var pills = document.querySelectorAll('.pp-pill');
    var cards = document.querySelectorAll('.pp-card[data-status]');
    var empty = document.querySelector('.pp-empty');
    if (!pills.length || !cards.length) return;

    pills.forEach(function (pill) {
      pill.addEventListener('click', function () {
        var filter = pill.getAttribute('data-filter');

        pills.forEach(function (p) {
          p.classList.remove('is-active');
          p.setAttribute('aria-selected', 'false');
        });
        pill.classList.add('is-active');
        pill.setAttribute('aria-selected', 'true');

        var visible = 0;
        cards.forEach(function (card) {
          var status = card.getAttribute('data-status');
          var show = filter === 'all' || status === filter;
          if (show) {
            card.removeAttribute('hidden');
            visible++;
          } else {
            card.setAttribute('hidden', '');
          }
        });

        if (empty) {
          if (visible === 0) empty.removeAttribute('hidden');
          else empty.setAttribute('hidden', '');
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
