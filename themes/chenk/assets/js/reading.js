// Scroll-spy TOC + mobile drawer
(function () {
  // ===== TOC scroll-spy =====
  const tocLinks = document.querySelectorAll(".toc a[href^='#']");
  if (tocLinks.length) {
    const map = new Map();
    tocLinks.forEach(function (link) {
      const id = decodeURIComponent(link.getAttribute("href").slice(1));
      const target = document.getElementById(id);
      if (target) map.set(target, link);
    });
    if (map.size) {
      const observer = new IntersectionObserver(
        function (entries) {
          // Pick the entry highest in viewport that is intersecting
          const visible = entries.filter((e) => e.isIntersecting);
          if (visible.length === 0) return;
          visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
          const link = map.get(visible[0].target);
          if (!link) return;
          tocLinks.forEach((l) => l.classList.remove("active"));
          link.classList.add("active");
        },
        { rootMargin: "-20% 0px -65% 0px", threshold: [0, 1] }
      );
      map.forEach((_, target) => observer.observe(target));
    }
  }

  // ===== Mobile drawer =====
  const drawer = document.querySelector(".drawer");
  if (drawer) {
    document.addEventListener("click", function (e) {
      if (e.target.closest("[data-drawer-open]")) {
        drawer.classList.add("open");
        document.body.style.overflow = "hidden";
      } else if (e.target.closest("[data-drawer-close]")) {
        drawer.classList.remove("open");
        document.body.style.overflow = "";
      }
    });
  }
})();
