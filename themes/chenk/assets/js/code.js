// Code block enhancements: copy button + language label
(function () {
  // Hugo wraps code in .highlight > pre[class*="language-"]
  const blocks = document.querySelectorAll(".prose .highlight, .prose pre:not(.no-enhance)");
  blocks.forEach(function (block) {
    // Skip <pre> children of .highlight — already handled by the outer .highlight pass
    if (block.tagName === "PRE" && block.closest(".highlight")) return;
    const isHighlight = block.classList.contains("highlight");
    const pre = isHighlight ? block.querySelector("pre") : block;
    if (!pre) return;
    if (block.querySelector(".code-copy")) return;
    const code = pre.querySelector("code") || pre;

    // Lang label from class (chroma puts language-* on <code>; raw <pre> may have it directly)
    let lang = "";
    const cls = (code.className || pre.className || "");
    const m = cls.match(/language-([a-z0-9+#-]+)/i);
    if (m) lang = m[1];
    // Strip useless lang values
    if (lang && /^(fallback|text|plain|plaintext|none)$/i.test(lang)) lang = "";

    if (lang) {
      const tag = document.createElement("span");
      tag.className = "code-lang";
      tag.textContent = lang;
      block.style.position = "relative";
      block.prepend(tag);
    }

    const btn = document.createElement("button");
    btn.className = "code-copy";
    btn.type = "button";
    btn.textContent = "copy";
    btn.addEventListener("click", function () {
      const text = code.innerText;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = "copied";
        btn.classList.add("done");
        setTimeout(function () {
          btn.textContent = "copy";
          btn.classList.remove("done");
        }, 1600);
      });
    });
    block.style.position = "relative";
    block.prepend(btn);
  });
})();
