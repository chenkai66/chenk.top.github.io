// Lang switcher: prefer translation pair via data-translation; fallback to root lang switch
(function () {
  document.addEventListener("click", function (e) {
    const el = e.target.closest("[data-lang-switch]");
    if (!el) return;
    e.preventDefault();
    const root = document.documentElement;
    const pair = root.getAttribute("data-translation");
    const cur = root.getAttribute("lang") || "en";
    if (pair && pair.length > 1) {
      window.location.href = pair;
      return;
    }
    // No translation — go to other lang's home
    const target = cur.startsWith("zh") ? "/en/" : "/zh/";
    window.location.href = target;
  });
})();
