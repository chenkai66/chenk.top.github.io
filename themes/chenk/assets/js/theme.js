// Theme toggle (dark mode) + scroll header + reading progress
(function () {
  const root = document.documentElement;
  const KEY = "chenk-theme";

  // Init: stored > prefers-color-scheme > light
  const stored = localStorage.getItem(KEY);
  if (stored === "dark" || stored === "light") {
    root.setAttribute("data-theme", stored);
  } else if (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches) {
    root.setAttribute("data-theme", "dark");
  }

  function setTheme(t) {
    root.setAttribute("data-theme", t);
    localStorage.setItem(KEY, t);
  }

  document.addEventListener("click", function (e) {
    const t = e.target.closest("[data-theme-toggle]");
    if (!t) return;
    const cur = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    setTheme(cur === "dark" ? "light" : "dark");
  });

  // Header shadow on scroll
  const header = document.querySelector(".site-header");
  if (header) {
    let prev = -1;
    const onScroll = () => {
      const y = window.scrollY;
      const scrolled = y > 4;
      if (scrolled !== prev) {
        header.classList.toggle("scrolled", scrolled);
        prev = scrolled;
      }
    };
    onScroll();
    document.addEventListener("scroll", onScroll, { passive: true });
  }

  // Reading progress (only on article pages)
  const article = document.querySelector(".article-main .prose");
  const bar = document.querySelector(".read-progress");
  if (article && bar) {
    const update = () => {
      const top = article.offsetTop;
      const height = article.offsetHeight - window.innerHeight + 200;
      const y = window.scrollY - top + 200;
      const pct = Math.max(0, Math.min(100, (y / Math.max(1, height)) * 100));
      bar.style.setProperty("--p", pct.toFixed(2) + "%");
    };
    update();
    document.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
  }
})();
