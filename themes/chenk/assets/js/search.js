// Cmd+K / Ctrl+K search overlay — fetches /index.json (Hugo-emitted)
(function () {
  const overlay = document.querySelector(".search-overlay");
  if (!overlay) return;
  const input = overlay.querySelector("input");
  const results = overlay.querySelector(".search-results");
  let docs = null;
  let loading = false;
  let activeIdx = -1;

  function open() {
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
    setTimeout(() => input.focus(), 30);
    if (!docs && !loading) load();
  }
  function close() {
    overlay.classList.remove("open");
    document.body.style.overflow = "";
  }

  async function load() {
    loading = true;
    try {
      const lang = document.documentElement.getAttribute("lang") || "en";
      const base = lang.startsWith("zh") ? "/zh" : "/en";
      let r = await fetch(base + "/index.json");
      if (!r.ok) r = await fetch("/index.json");
      docs = await r.json();
    } catch (e) { docs = []; }
    loading = false;
    runSearch();
  }

  function score(q, doc) {
    q = q.toLowerCase();
    const t = (doc.title || "").toLowerCase();
    const s = (doc.summary || "").toLowerCase();
    const tags = (doc.tags || []).join(" ").toLowerCase();
    let sc = 0;
    if (t.includes(q)) sc += 10;
    if (t.startsWith(q)) sc += 6;
    if (tags.includes(q)) sc += 4;
    if (s.includes(q)) sc += 2;
    const words = q.split(/\s+/);
    for (const w of words) {
      if (w.length < 2) continue;
      if (t.includes(w)) sc += 1;
      if (s.includes(w)) sc += 0.5;
    }
    return sc;
  }

  function runSearch() {
    const q = input.value.trim();
    if (!q) {
      results.innerHTML = '<div class="search-empty">Type to search the archive&hellip;</div>';
      activeIdx = -1;
      return;
    }
    if (!docs) {
      results.innerHTML = '<div class="search-empty">Loading index&hellip;</div>';
      return;
    }
    const ranked = docs
      .map((d) => ({ d, s: score(q, d) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, 12);
    if (!ranked.length) {
      results.innerHTML = '<div class="search-empty">No matches.</div>';
      return;
    }
    results.innerHTML = ranked
      .map(function (r, i) {
        var d = r.d;
        var seriesTag = d.series ? `<span class="r-series">${escape(d.series.replace(/-/g, ' '))}</span>` : '';
        var date = d.date ? `<span class="r-date">${escape(d.date.slice(0,10))}</span>` : '';
        var titleHL = highlight(escape(d.title), q);
        var summaryHL = highlight(escape((d.summary || '').slice(0, 140)), q);
        return `<a class="result${i === 0 ? ' active' : ''}" href="${d.url}">
          <div class="r-meta">${seriesTag}${date}</div>
          <span class="title">${titleHL}</span>
          <span class="excerpt">${summaryHL}</span>
        </a>`;
      })
      .join("");
    activeIdx = 0;
  }

  function highlight(text, q) {
    if (!q) return text;
    var words = q.split(/\s+/).filter(function (w) { return w.length >= 2; });
    if (!words.length) return text;
    var pattern = new RegExp('(' + words.map(function (w) { return w.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'); }).join('|') + ')', 'gi');
    return text.replace(pattern, '<mark>$1</mark>');
  }
  function escape(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  document.addEventListener("keydown", function (e) {
    const ck = (e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey);
    if (ck) {
      e.preventDefault();
      overlay.classList.contains("open") ? close() : open();
    } else if (e.key === "Escape" && overlay.classList.contains("open")) {
      close();
    } else if (overlay.classList.contains("open")) {
      const items = results.querySelectorAll(".result");
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (activeIdx < items.length - 1) activeIdx++;
        items.forEach((it, i) => it.classList.toggle("active", i === activeIdx));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (activeIdx > 0) activeIdx--;
        items.forEach((it, i) => it.classList.toggle("active", i === activeIdx));
      } else if (e.key === "Enter") {
        if (items[activeIdx]) { e.preventDefault(); window.location.href = items[activeIdx].href; }
      }
    }
  });

  document.addEventListener("click", function (e) {
    if (e.target.closest("[data-search-open]")) {
      e.preventDefault();
      open();
    } else if (e.target.closest("[data-search-close]") || e.target === overlay) {
      close();
    }
  });

  if (input) {
    let t;
    input.addEventListener("input", function () {
      clearTimeout(t);
      t = setTimeout(runSearch, 100);
    });
  }
})();
