// More fun: cite-on-copy, "you finished" celebration, first-visit hint.
(function () {
  // ===== 1. Cite-on-copy: when user copies prose, append a source link =====
  document.addEventListener('copy', function (e) {
    var sel = window.getSelection && window.getSelection();
    if (!sel || sel.isCollapsed) return;
    var text = sel.toString();
    if (text.length < 30) return; // short snippets get left alone
    var article = document.querySelector('article.prose, .article-main');
    if (!article || !article.contains(sel.anchorNode)) return;

    var isZh = (document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0;
    var url = window.location.origin + window.location.pathname;
    var title = document.title.split('·')[0].trim();
    var citation = isZh
      ? '\n\n— 摘自《' + title + '》\n' + url
      : '\n\n— from "' + title + '"\n' + url;

    if (e.clipboardData) {
      e.clipboardData.setData('text/plain', text + citation);
      e.preventDefault();
    }
  });

  // ===== 2. "You finished" celebration when reaching article end =====
  var finishShown = false;
  function checkFinish() {
    if (finishShown) return;
    var article = document.querySelector('article.prose, .article-main');
    if (!article) return;
    var rect = article.getBoundingClientRect();
    // Trigger when the article's bottom is within 200px of viewport bottom
    if (rect.bottom <= window.innerHeight + 200 && rect.top < 0) {
      finishShown = true;
      showFinishToast();
    }
  }
  function showFinishToast() {
    var isZh = (document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0;
    var msg = isZh ? '🎉 读完啦，辛苦了' : '🎉 You made it to the end';
    var t = document.createElement('div');
    t.className = 'finish-toast';
    t.textContent = msg;
    document.body.appendChild(t);
    requestAnimationFrame(function () { t.classList.add('show'); });
    setTimeout(function () {
      t.classList.remove('show');
      setTimeout(function () { t.remove(); }, 400);
    }, 2400);
  }
  if (document.querySelector('article.prose, .article-main')) {
    window.addEventListener('scroll', checkFinish, { passive: true });
  }

  // ===== 3. First-visit hint bubble (lifetime: localStorage) =====
  var FV_KEY = 'first-visit-hint-shown';
  function maybeShowHint() {
    try { if (localStorage.getItem(FV_KEY)) return; } catch (e) { return; }
    setTimeout(function () {
      var fab = document.getElementById('kbd-fab');
      if (!fab) return;
      var isZh = (document.documentElement.lang || '').toLowerCase().indexOf('zh') === 0;
      var b = document.createElement('div');
      b.className = 'first-hint';
      b.innerHTML =
        '<span>' + (isZh ? '👋 按 ? 看键盘快捷键' : '👋 Press ? for keyboard shortcuts') + '</span>' +
        '<button aria-label="Got it" type="button">×</button>';
      document.body.appendChild(b);
      requestAnimationFrame(function () { b.classList.add('show'); });

      function dismiss() {
        b.classList.remove('show');
        setTimeout(function () { b.remove(); }, 300);
        try { localStorage.setItem(FV_KEY, '1'); } catch (e) {}
      }
      b.querySelector('button').addEventListener('click', dismiss);
      // auto-dismiss after 8 seconds
      setTimeout(dismiss, 8000);
    }, 4500);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', maybeShowHint);
  } else {
    maybeShowHint();
  }
})();
