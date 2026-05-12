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

  // ===== 2. "You finished" celebration =====
  // Trigger as soon as the user reaches the end of the main body — typically right
  // before the References / Further reading / Summary section.
  var finishShown = false;
  function findFinishMarker() {
    var headings = document.querySelectorAll('article.prose h2, .article-main h2');
    if (!headings.length) return null;
    // Patterns that mean "the wrap-up section" — toast fires when this enters viewport
    var patterns = [
      /^summary/i, /^further reading/i, /^references?$/i, /^see also/i,
      /^where the story continues/i, /^conclusion/i, /^takeaways?\b/i, /^key takeaways/i,
      /^what to take away/i, /^wrap[\s-]?up/i, /^recap/i, /^final thoughts/i,
      /^closing thoughts/i, /^closing/i, /^tl;?\s?dr/i, /^next steps/i, /^bibliography/i,
      /^acknowledgements?/i, /^appendix/i,
      /^总结/, /^小结/, /^参考/, /^延伸阅读/, /^扩展阅读/, /^结语/, /^结论/, /^结尾/,
      /^写在最后/, /^要点/, /^收尾/, /^下一步/, /^致谢/, /^附录/,
    ];
    for (var i = 0; i < headings.length; i++) {
      var t = (headings[i].textContent || '').trim();
      for (var j = 0; j < patterns.length; j++) {
        if (patterns[j].test(t)) return headings[i];
      }
    }
    // Fallback: use the LAST H2 in the article — articles without explicit wrap-up
    // sections still get a sensible finish trigger right when their final chapter shows up.
    return headings[headings.length - 1];
  }
  function checkFinish() {
    if (finishShown) return;
    var marker = findFinishMarker();
    if (marker) {
      // Trigger when marker is just about to enter viewport
      var rect = marker.getBoundingClientRect();
      if (rect.top <= window.innerHeight * 0.85) {
        finishShown = true;
        showFinishToast();
      }
      return;
    }
    // Fallback: no marker found — use bottom-of-article heuristic
    var article = document.querySelector('article.prose, .article-main');
    if (!article) return;
    var rect2 = article.getBoundingClientRect();
    if (rect2.bottom <= window.innerHeight + 200 && rect2.top < 0) {
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

// ===== Heading anchor copy-to-clipboard =====
(function () {
  document.addEventListener("click", function (e) {
    var link = e.target.closest(".heading-link");
    if (!link) return;
    e.preventDefault();
    var url = window.location.origin + window.location.pathname + link.getAttribute("href");
    if (navigator.clipboard) {
      navigator.clipboard.writeText(url).then(function () {
        var isZh = (document.documentElement.lang || "").toLowerCase().indexOf("zh") === 0;
        var t = document.createElement("div");
        t.className = "tiny-toast show";
        t.textContent = isZh ? "✓ 已复制章节链接" : "✓ Section link copied";
        document.body.appendChild(t);
        setTimeout(function () { t.classList.remove("show"); setTimeout(function () { t.remove(); }, 300); }, 1400);
      }).catch(function () {});
    }
    // Also update URL hash
    history.replaceState(null, "", link.getAttribute("href"));
  });
})();
