/* chenk.top private knowledge base — client for the approved-members API */
(function () {
  "use strict";

  var LOCAL = /^(127\.0\.0\.1|localhost)$/.test(location.hostname);
  var API = LOCAL ? "http://127.0.0.1:8080" : "https://learn.chenk.top";
  var TOKEN_KEY = "kb_token";
  var LANG_KEY = "kb_lang";
  var token = null;
  var user = null;
  var lang = "zh";
  var pendingRoute = "";
  var homeCache = {};
  var main = document.getElementById("main");
  var mermaidPromise = null;

  try { token = localStorage.getItem(TOKEN_KEY); } catch (e) { token = null; }
  try {
    var saved = localStorage.getItem(LANG_KEY);
    var q = new URLSearchParams(location.search);
    lang = q.get("lang") || saved || "zh";
  } catch (e) { lang = "zh"; }
  if (lang !== "en") lang = "zh";
  try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}

  var TXT = {
    zh: {
      nav: { home: "首页", series: "系列", projects: "项目", archive: "归档", data: "数据", kb: "知识库", about: "关于" },
      docTitle: "知识库",
      eyebrow: "私有知识库",
      title: "外投开发教程",
      lede: "从 Python 工程师视角讲清广告外投的业务、数据链路、投放系统与中间件。共 9 章 57 节，中英双语，仅授权成员可读。",
      chapters: "章",
      sections: "节",
      span: "更新跨度",
      private: "仅授权成员",
      login: "登录", register: "申请访问", email: "邮箱", password: "密码", name: "称呼",
      loginTitle: "登录知识库", loginLede: "这里是私有知识库，只有被授权的成员可以阅读。",
      registerTitle: "申请访问", registerLede: "注册后需要管理员授权，通过后即可阅读全部章节。",
      noAccount: "还没有账号？", hasAccount: "已有账号？", submitLogin: "登录", submitRegister: "提交申请",
      signingIn: "正在登录…", registering: "正在提交…",
      pendingTitle: "等待授权", pendingLede: "账号已注册，管理员授权后即可阅读。授权状态会实时更新。",
      refresh: "刷新状态", signout: "退出登录", signedOut: "已退出。",
      toc: "本节目录", prev: "上一节", next: "下一节", minutes: "分钟",
      backHome: "全部章节", chapter: "第 {n} 章",
      searchPlaceholder: "搜索章节、主题…", searchHint: "输入关键词，搜索 57 个章节", searchNoResult: "没有匹配的章节", searchLogin: "登录后才能搜索全部章节",
      account: "账号", accountLede: "修改密码后，其他设备会自动退出。",
      currentPw: "当前密码", newPw: "新密码（至少 10 位）", changePw: "修改密码",
      pwChanged: "密码已更新。", signoutAll: "退出登录",
      admin: "成员管理", adminLede: "新注册账号默认待授权，批准后才能阅读。",
      members: "成员", pendingCount: "待授权", approvedCount: "已授权",
      colMember: "成员", colStatus: "状态", colJoined: "加入", colActions: "操作",
      stPending: "待授权", stApproved: "已授权", stDisabled: "已停用",
      approve: "批准", disable: "停用", enable: "恢复", makeAdmin: "设为管理员", revokeAdmin: "撤销管理员",
      adminOnly: "仅管理员可访问。", loading: "加载中…",
      errNetwork: "网络异常，请稍后再试。", errGeneric: "操作失败，请重试。",
      errCred: "邮箱或密码不正确。", errDisabled: "该账号已被停用。",
      errRate: "尝试次数过多，请 15 分钟后再试。", errExists: "该邮箱已注册，请直接登录。",
      errBadEmail: "请填写有效邮箱。", errBadName: "请填写称呼（不超过 40 字）。",
      errShortPw: "密码至少 10 位。", errBadPw: "当前密码不正确。",
      notFound: "没有找到这一节。", empty: "暂无内容。", langSwitch: "EN"
    },
    en: {
      nav: { home: "Home", series: "Series", projects: "Projects", archive: "Archive", data: "Data", kb: "Library", about: "About" },
      docTitle: "Library",
      eyebrow: "Private library",
      title: "External Ads Engineering Guide",
      lede: "Business, data pipelines, delivery systems and middleware for external advertising, written for Python engineers. 9 chapters, 57 sections, bilingual, authorized members only.",
      chapters: "chapters",
      sections: "sections",
      span: "Date range",
      private: "Authorized members",
      login: "Sign in", register: "Request access", email: "Email", password: "Password", name: "Name",
      loginTitle: "Sign in", loginLede: "This is a private library. Only approved members can read it.",
      registerTitle: "Request access", registerLede: "Registration requires admin approval before you can read.",
      noAccount: "No account yet?", hasAccount: "Already registered?", submitLogin: "Sign in", submitRegister: "Submit request",
      signingIn: "Signing in…", registering: "Submitting…",
      pendingTitle: "Awaiting approval", pendingLede: "Your account is registered. You can read once an admin approves it. This page checks the status on demand.",
      refresh: "Check status", signout: "Sign out", signedOut: "Signed out.",
      toc: "On this page", prev: "Previous", next: "Next", minutes: "min",
      backHome: "All chapters", chapter: "Chapter {n}",
      searchPlaceholder: "Search chapters and topics…", searchHint: "Type to search across 57 sections", searchNoResult: "No matching sections", searchLogin: "Sign in to search the library",
      account: "Account", accountLede: "Changing your password signs out your other devices.",
      currentPw: "Current password", newPw: "New password (min 10 chars)", changePw: "Change password",
      pwChanged: "Password updated.", signoutAll: "Sign out",
      admin: "Members", adminLede: "New accounts are pending until an admin approves them.",
      members: "members", pendingCount: "pending", approvedCount: "approved",
      colMember: "Member", colStatus: "Status", colJoined: "Joined", colActions: "Actions",
      stPending: "Pending", stApproved: "Approved", stDisabled: "Disabled",
      approve: "Approve", disable: "Disable", enable: "Enable", makeAdmin: "Make admin", revokeAdmin: "Revoke admin",
      adminOnly: "Admins only.", loading: "Loading…",
      errNetwork: "Network error, try again.", errGeneric: "Something went wrong.",
      errCred: "Email or password is incorrect.", errDisabled: "This account has been disabled.",
      errRate: "Too many attempts. Try again in 15 minutes.", errExists: "This email is registered. Sign in instead.",
      errBadEmail: "Enter a valid email address.", errBadName: "Enter a name (max 40 characters).",
      errShortPw: "Password must be at least 10 characters.", errBadPw: "Current password is incorrect.",
      notFound: "This section was not found.", empty: "Nothing here yet.", langSwitch: "中文"
    }
  };
  function T() { return TXT[lang]; }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function routeHash(path) {
    return "#" + path + (/[?]/.test(path) ? "&" : "?") + "lang=" + lang;
  }
  function fmtDate(iso) {
    if (!iso) return "";
    var p = iso.split("-");
    var d = new Date(Date.UTC(+p[0], +p[1] - 1, +p[2]));
    try {
      return new Intl.DateTimeFormat(lang === "en" ? "en-US" : "zh-CN", { year: "numeric", month: lang === "en" ? "short" : "long", day: "numeric", timeZone: "UTC" }).format(d);
    } catch (e) { return iso; }
  }

  /* ---------- chrome ---------- */
  function navMarkup() {
    var t = T();
    var items = [
      ["home", "/" + lang + "/"],
      ["series", "/" + lang + "/series/"],
      ["projects", "/" + lang + "/projects/"],
            ["archive", "/" + lang + "/archives/"],
      ["data", "/" + lang + "/data/"],
      ["kb", "/kb/" + (lang === "en" ? "?lang=en" : "")],
      ["about", "/" + lang + "/about/"]
    ];
    return items.map(function (it) {
      var cls = it[0] === "kb" ? ' class="active" aria-current="page"' : "";
      return '<a href="' + it[1] + '"' + cls + ">" + esc(t.nav[it[0]]) + "</a>";
    }).join("");
  }
  function footerMarkup() {
    var en = lang === "en";
    var p = "/" + lang + "/";
    var series = [
      ["产品思维", "Product Thinking", "https://www.chenk.top" + p + "series/product-thinking/"],
      ["个人理财", "Personal Finance", "https://www.chenk.top" + p + "series/personal-finance/"],
      ["阿里云全栈实战", "Alibaba Cloud Full Stack", "https://www.chenk.top" + p + "series/aliyun-fullstack/"],
      ["Claude Code 实战", "Claude Code Hands-On", "https://www.chenk.top" + p + "series/claude-code-learn/"],
      [en ? "All series →" : "全部系列 →", "", p + "series/"]
    ];
    var explore = [
      [en ? "All Posts" : "全部文章", "", p + "archives/"],
      [en ? "Tags" : "标签", "", p + "tags/"],
      [en ? "Categories" : "分类", "", p + "categories/"],
      [en ? "Library" : "知识库", "", "/kb/" + (en ? "?lang=en" : "")],
      [en ? "About" : "关于", "", p + "about/"]
    ];
    var projects = [
      ["AI4Marketing", "", "https://llm4marketing.com"],
      ["AI4Science", "", "https://llm4science.top"],
      ["DaaS", "", "https://company-skill.com"],
      ["MiniGameForge", "", "https://forge.llm4marketing.asia"],
      ["llm-elevator", "", "https://llm4marketing.asia"],
      ["NodeMind", "", "https://askgraph.online"],
      ["Meeting Copilot", "", "https://meeting.company-skill.com"],
      ["agent-pet", "", "https://pet.llm4marketing.asia"],
      [en ? "All projects →" : "全部项目 →", "", p + "projects/"]
    ];
    var elsewhere = [
      ["GitHub", "", "https://github.com/chenkai66"],
      ["Email", "", "mailto:chenkai.nb.666@gmail.com"],
      ["WeChat", "wechat", ""]
    ];
    function item(it) {
      if (it[1] === "wechat") {
        return '<li><button type="button" class="footer-wechat" data-wechat="ck1640234528" aria-label="' + (en ? "Copy WeChat ID" : "复制微信号") + '">WeChat</button></li>';
      }
      var external = /^https?:/.test(it[2]);
      return '<li><a href="' + it[2] + '"' + (external ? ' target="_blank" rel="noopener"' : "") + ">" + esc(it[0]) + "</a></li>";
    }
    function col(title, items) {
      return '<div class="footer-col"><h3>' + esc(title) + '</h3><ul>' + items.map(item).join("") + "</ul></div>";
    }
    return col(en ? "Series" : "系列", series) + col(en ? "Explore" : "探索", explore) + col(en ? "Projects" : "项目", projects) + col(en ? "Elsewhere" : "其他", elsewhere);
  }
  function paintChrome() {
    document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    var brand = document.getElementById("kb-brand");
    if (brand) brand.href = "/" + lang + "/";
    var fGrid = document.getElementById("kb-footer-grid");
    if (fGrid) fGrid.innerHTML = footerMarkup();
    var fCopy = document.getElementById("kb-footer-copy");
    if (fCopy) fCopy.textContent = lang === "en" ? "© 2026 Chen Kai" : "© 2026 陈锴";
    var fSign = document.getElementById("kb-footer-sign");
    if (fSign) fSign.textContent = lang === "en" ? "Hangzhou." : "杭州";
    var wxHint = document.getElementById("kb-wx-hint");
    if (wxHint) wxHint.textContent = lang === "en" ? "Search this ID inside WeChat to connect" : "在微信里搜索此 ID 即可添加我";
    var wxCopy = document.getElementById("kb-wx-copy");
    if (wxCopy && !wxCopy.dataset.copied) wxCopy.textContent = lang === "en" ? "Copy ID" : "复制微信号";
    var wxClose = document.querySelector("#kb-wx-modal .wx-modal-close");
    if (wxClose) wxClose.setAttribute("aria-label", lang === "en" ? "Close" : "关闭");
    document.getElementById("kb-nav").innerHTML = navMarkup();
    document.getElementById("kb-drawer-nav").innerHTML = navMarkup();
    var langBtn = document.getElementById("kb-lang");
    if (langBtn) {
      langBtn.textContent = T().langSwitch;
      langBtn.href = "#" + currentPath() + "?lang=" + (lang === "en" ? "zh" : "en");
    }
    if (searchInput) searchInput.placeholder = T().searchPlaceholder;
    document.title = T().docTitle + " · chenk.top";
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("chenk-theme", theme); } catch (e) {}
  }
  document.getElementById("kb-theme").addEventListener("click", function () {
    setTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light");
  });
  var drawer = document.getElementById("kb-drawer");
  var backdrop = document.getElementById("kb-backdrop");
  function closeDrawer() {
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
    backdrop.hidden = true;
  }
  document.getElementById("kb-menu").addEventListener("click", function () {
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    backdrop.hidden = false;
  });
  document.getElementById("kb-drawer-close").addEventListener("click", closeDrawer);
  backdrop.addEventListener("click", closeDrawer);

  /* ---------- wechat ---------- */
  var wxModal = document.getElementById("kb-wx-modal");
  function openWx() {
    wxModal.hidden = false;
    requestAnimationFrame(function () { wxModal.classList.add("is-open"); });
  }
  function closeWx() {
    wxModal.classList.remove("is-open");
    setTimeout(function () { wxModal.hidden = true; }, 220);
  }
  function copyWx(btn) {
    var id = btn.getAttribute("data-wx-copy") || "ck1640234528";
    function done() {
      btn.dataset.copied = "1";
      btn.classList.add("is-done");
      btn.textContent = lang === "en" ? "Copied" : "已复制";
      setTimeout(function () {
        delete btn.dataset.copied;
        btn.classList.remove("is-done");
        btn.textContent = lang === "en" ? "Copy ID" : "复制微信号";
      }, 1800);
    }
    function fallback() {
      var ta = document.createElement("textarea");
      ta.value = id;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch (e) {}
      document.body.removeChild(ta);
      done();
    }
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(id).then(done, fallback);
    else fallback();
  }
  document.addEventListener("click", function (e) {
    if (e.target.closest(".footer-wechat")) { openWx(); return; }
    if (e.target.closest("[data-wx-close]")) { closeWx(); return; }
    var copyBtn = e.target.closest("[data-wx-copy]");
    if (copyBtn) copyWx(copyBtn);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !wxModal.hidden) closeWx();
  });

  /* ---------- search ---------- */
  var searchOverlay = document.getElementById("kb-search-overlay");
  var searchInput = document.getElementById("kb-search-input");
  var searchResults = document.getElementById("kb-search-results");

  function markText(text, q) {
    var safe = esc(text);
    if (!q) return safe;
    var needle = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return safe.replace(new RegExp("(" + needle + ")", "ig"), "<mark>$1</mark>");
  }
  function allSections(d) {
    var out = [];
    (d.chapters || []).forEach(function (ch) {
      (ch.sections || []).forEach(function (sec) { out.push({ ch: ch, sec: sec }); });
    });
    return out;
  }
  function paintResults(d, q) {
    var t = T();
    if (!d) {
      searchResults.innerHTML = '<div class="search-empty">' + esc(t.searchLogin) + "</div>";
      return;
    }
    var ql = q.toLowerCase();
    var hits = allSections(d).filter(function (row) {
      var s = row.sec;
      var hay = (s.title + " " + (s.tags || []).join(" ") + " " + row.ch.title + " " + s.slug).toLowerCase();
      return hay.indexOf(ql) >= 0;
    });
    if (!hits.length) {
      searchResults.innerHTML = '<div class="search-empty">' + esc(t.searchNoResult) + "</div>";
      return;
    }
    searchResults.innerHTML = hits.slice(0, 30).map(function (row) {
      var s = row.sec;
      return '<a class="result" href="' + routeHash("/a/" + s.slug) + '">' +
        '<span class="r-meta"><span class="r-series">' + esc(row.ch.title) + '</span><span class="r-date">' + esc(fmtDate(s.date)) + "</span></span>" +
        '<span class="title">' + markText(s.title, q) + "</span>" +
        (s.tags && s.tags.length ? '<span class="excerpt">' + esc(s.tags.join(" · ")) + "</span>" : "") +
      "</a>";
    }).join("");
  }
  function renderSearchResults(q) {
    var t = T();
    if (!q) {
      searchResults.innerHTML = '<div class="search-empty">' + esc(t.searchHint) + "</div>";
      return;
    }
    if (homeCache[lang]) return paintResults(homeCache[lang], q);
    searchResults.innerHTML = '<div class="search-empty">' + esc(t.loading) + "</div>";
    api("/api/home?lang=" + lang).then(function (res) {
      if (res.ok) homeCache[lang] = res.data;
      paintResults(res.ok ? res.data : null, q);
    });
  }
  function openSearch() {
    searchOverlay.classList.add("open");
    searchOverlay.setAttribute("aria-hidden", "false");
    searchInput.placeholder = T().searchPlaceholder;
    searchInput.value = "";
    renderSearchResults("");
    setTimeout(function () { searchInput.focus(); }, 30);
  }
  function closeSearch() {
    searchOverlay.classList.remove("open");
    searchOverlay.setAttribute("aria-hidden", "true");
  }
  document.getElementById("kb-search").addEventListener("click", openSearch);
  searchInput.addEventListener("input", function () { renderSearchResults(searchInput.value.trim()); });
  searchResults.addEventListener("click", function (ev) { if (ev.target.closest("a.result")) closeSearch(); });
  searchOverlay.addEventListener("click", function (ev) { if (ev.target === searchOverlay) closeSearch(); });
  document.addEventListener("keydown", function (ev) {
    var open = searchOverlay.classList.contains("open");
    if ((ev.metaKey || ev.ctrlKey) && (ev.key === "k" || ev.key === "K")) {
      ev.preventDefault();
      if (open) closeSearch(); else openSearch();
      return;
    }
    if (!open) return;
    if (ev.key === "Escape") { ev.preventDefault(); closeSearch(); }
    if (ev.key === "Enter") {
      var first = searchResults.querySelector("a.result");
      if (first) { ev.preventDefault(); first.click(); }
    }
  });

  /* ---------- api ---------- */
  function api(path, opts) {
    opts = opts || {};
    var headers = {};
    if (opts.body) headers["Content-Type"] = "application/json";
    if (token) headers["Authorization"] = "Bearer " + token;
    return fetch(API + path, {
      method: opts.method || (opts.body ? "POST" : "GET"),
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
      cache: "no-store"
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (res.status === 401) { clearToken(); }
        return { status: res.status, ok: res.ok, data: data };
      });
    }, function () {
      return { status: 0, ok: false, data: { error: "network" } };
    });
  }

  function clearToken() {
    token = null;
    user = null;
    try { localStorage.removeItem(TOKEN_KEY); } catch (e) {}
  }
  function setSession(data) {
    if (data && data.token) {
      token = data.token;
      try { localStorage.setItem(TOKEN_KEY, token); } catch (e) {}
    }
    if (data && data.user) user = data.user;
  }

  function errText(code) {
    var t = T();
    var map = {
      invalid_credentials: t.errCred, disabled: t.errDisabled, rate_limited: t.errRate,
      exists: t.errExists, bad_email: t.errBadEmail, bad_name: t.errBadName,
      short_password: t.errShortPw, bad_password: t.errBadPw, network: t.errNetwork
    };
    return map[code] || t.errGeneric;
  }

  /* ---------- routing ---------- */
  function currentPath() {
    var raw = (location.hash || "").replace(/^#/, "");
    var q = raw.indexOf("?");
    return (q >= 0 ? raw.slice(0, q) : raw) || "/";
  }
  function parseHash() {
    var raw = (location.hash || "").replace(/^#/, "");
    var q = raw.indexOf("?");
    var path = q >= 0 ? raw.slice(0, q) : raw;
    var params = new URLSearchParams(q >= 0 ? raw.slice(q + 1) : "");
    var qLang = params.get("lang");
    if (qLang === "zh" || qLang === "en") {
      if (qLang !== lang) {
        lang = qLang;
        try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
      }
    }
    var parts = (path || "/").split("/").filter(Boolean);
    return { path: "/" + parts.join("/"), parts: parts };
  }
  function go(path) {
    var target = routeHash(path);
    if (location.hash === target) route();
    else location.hash = target;
  }

  function route() {
    var r = parseHash();
    paintChrome();
    closeDrawer();
    closeSearch();
    window.scrollTo(0, 0);
    var p = r.parts;
    if (p[0] === "a" && p[1]) return requireAuth(function () { return renderArticle(decodeURIComponent(p[1])); }, r.path);
    if (p[0] === "login") return renderLogin();
    if (p[0] === "register") return renderRegister();
    if (p[0] === "pending") return requireAuth(renderPending, r.path);
    if (p[0] === "account") return requireAuth(renderAccount, r.path);
    if (p[0] === "admin") return requireAuth(renderAdmin, r.path);
    return requireAuth(renderHome, "/");
  }

  function requireAuth(fn, wanted) {
    if (!token) {
      pendingRoute = wanted && wanted !== "/" ? wanted : "";
      return renderLogin();
    }
    if (!user) {
      return api("/api/me").then(function (res) {
        if (res.status === 401) {
          pendingRoute = wanted && wanted !== "/" ? wanted : "";
          return renderLogin();
        }
        if (res.ok) {
          user = res.data.user;
          if (user.status !== "approved" && wanted !== "/pending") return renderPending();
          return fn();
        }
        return renderError(errText(res.data.error));
      });
    }
    if (user.status !== "approved" && wanted !== "/pending") return renderPending();
    return fn();
  }

  /* ---------- views ---------- */
  function loading() {
    main.innerHTML = '<div class="shell"><p class="muted">' + esc(T().loading) + "</p></div>";
  }

  function renderLogin(message) {
    var t = T();
    main.innerHTML =
      '<div class="shell auth-wrap">' +
        '<section class="auth-card">' +
          '<p class="eyebrow">' + esc(t.eyebrow) + "</p>" +
          "<h1>" + esc(t.loginTitle) + "</h1>" +
          '<p class="lede">' + esc(t.loginLede) + "</p>" +
          '<div id="kb-flash">' + (message ? flash(message, "error") : "") + "</div>" +
          '<form class="form" id="kb-login-form" novalidate>' +
            '<div class="field"><label for="li-email">' + esc(t.email) + '</label><input id="li-email" name="email" type="email" autocomplete="username" required></div>' +
            '<div class="field"><label for="li-pw">' + esc(t.password) + '</label><input id="li-pw" name="password" type="password" autocomplete="current-password" required></div>' +
            '<button class="btn primary" type="submit">' + esc(t.submitLogin) + "</button>" +
          "</form>" +
          '<p class="auth-foot">' + esc(t.noAccount) + ' <a href="' + routeHash("/register") + '">' + esc(t.register) + "</a></p>" +
        "</section>" +
        '<aside class="auth-aside">' +
          '<p class="eyebrow">' + esc(t.eyebrow) + "</p>" +
          "<h2>" + esc(t.title) + "</h2>" +
          "<p>" + esc(t.lede) + "</p>" +
          "<ul><li>9 / 57</li><li>" + esc(t.private) + "</li><li>中文 · English</li></ul>" +
        "</aside>" +
      "</div>";

    document.getElementById("kb-login-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      var form = ev.target;
      var btn = form.querySelector("button");
      btn.disabled = true;
      btn.textContent = t.signingIn;
      api("/api/login", { body: { email: form.email.value.trim(), password: form.password.value } }).then(function (res) {
        btn.disabled = false;
        btn.textContent = t.submitLogin;
        if (!res.ok) {
          document.getElementById("kb-flash").innerHTML = flash(errText(res.data.error), "error");
          return;
        }
        setSession(res.data);
        if (res.data.user.status !== "approved") return go("/pending");
        return go(pendingRoute || "/");
      });
    });
  }

  function renderRegister(message) {
    var t = T();
    main.innerHTML =
      '<div class="shell auth-wrap">' +
        '<section class="auth-card">' +
          '<p class="eyebrow">' + esc(t.eyebrow) + "</p>" +
          "<h1>" + esc(t.registerTitle) + "</h1>" +
          '<p class="lede">' + esc(t.registerLede) + "</p>" +
          '<div id="kb-flash">' + (message ? flash(message, "error") : "") + "</div>" +
          '<form class="form" id="kb-reg-form" novalidate>' +
            '<div class="field"><label for="rg-name">' + esc(t.name) + '</label><input id="rg-name" name="name" maxlength="40" autocomplete="nickname" required></div>' +
            '<div class="field"><label for="rg-email">' + esc(t.email) + '</label><input id="rg-email" name="email" type="email" autocomplete="username" required></div>' +
            '<div class="field"><label for="rg-pw">' + esc(t.password) + '</label><input id="rg-pw" name="password" type="password" minlength="10" autocomplete="new-password" required></div>' +
            '<button class="btn primary" type="submit">' + esc(t.submitRegister) + "</button>" +
          "</form>" +
          '<p class="auth-foot">' + esc(t.hasAccount) + ' <a href="' + routeHash("/login") + '">' + esc(t.login) + "</a></p>" +
        "</section>" +
        '<aside class="auth-aside">' +
          "<h2>" + esc(t.title) + "</h2>" +
          "<p>" + esc(t.lede) + "</p>" +
        "</aside>" +
      "</div>";

    document.getElementById("kb-reg-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      var form = ev.target;
      var btn = form.querySelector("button");
      btn.disabled = true;
      btn.textContent = t.registering;
      api("/api/register", { body: { name: form.name.value.trim(), email: form.email.value.trim(), password: form.password.value } }).then(function (res) {
        btn.disabled = false;
        btn.textContent = t.submitRegister;
        if (!res.ok) {
          document.getElementById("kb-flash").innerHTML = flash(errText(res.data.error), "error");
          return;
        }
        setSession(res.data);
        return go("/pending");
      });
    });
  }

  function renderPending(message) {
    var t = T();
    main.innerHTML =
      '<div class="shell" style="max-width:720px">' +
        '<p class="eyebrow">' + esc(t.eyebrow) + "</p>" +
        '<h1 class="kb-title">' + esc(t.pendingTitle) + "</h1>" +
        '<p class="lede">' + esc(t.pendingLede) + "</p>" +
        '<div id="kb-flash">' + (message ? flash(message, "ok") : "") + "</div>" +
        '<div class="panel">' +
          "<p><strong>" + esc(user ? user.name : "") + "</strong> · " + esc(user ? user.email : "") + "</p>" +
          '<p style="margin-top:14px"><span class="badge warn">' + esc(t.stPending) + "</span></p>" +
          '<div style="display:flex;gap:10px;margin-top:18px;flex-wrap:wrap">' +
            '<button class="btn" type="button" id="kb-refresh">' + esc(t.refresh) + "</button>" +
            '<button class="btn ghost" type="button" id="kb-signout">' + esc(t.signout) + "</button>" +
          "</div>" +
        "</div>" +
      "</div>";
    document.getElementById("kb-refresh").addEventListener("click", function () {
      var btn = this;
      btn.disabled = true;
      api("/api/me").then(function (res) {
        btn.disabled = false;
        if (res.ok && res.data.user.status === "approved") {
          user = res.data.user;
          return go(pendingRoute || "/");
        }
        document.getElementById("kb-flash").innerHTML = flash(res.ok ? t.stPending : errText(res.data.error), res.ok ? "ok" : "error");
      });
    });
    document.getElementById("kb-signout").addEventListener("click", signOut);
  }

  function signOut() {
    api("/api/logout", { method: "POST" }).then(function () {
      clearToken();
      pendingRoute = "";
      go("/login");
    });
  }

  function renderHome() {
    loading();
    api("/api/home?lang=" + lang).then(function (res) {
      if (res.status === 403) return renderPending();
      if (res.status === 401) { clearToken(); return renderLogin(); }
      if (!res.ok) return renderError(errText(res.data.error));
      var d = res.data;
      homeCache[lang] = d;
      var sections = d.chapters.reduce(function (n, ch) { return n + ch.sections.length; }, 0);
      var first = d.chapters[0] && d.chapters[0].sections[0] ? d.chapters[0].sections[0].date : "";
      var t = T();
      var html = '<div class="shell">' +
        '<p class="eyebrow">' + esc(t.eyebrow) + ' · <span class="badge ok">' + esc(t.private) + "</span></p>" +
        '<h1 class="kb-title">' + esc(d.title || t.title) + "</h1>" +
        '<p class="lede">' + esc(d.description || d.subtitle || t.lede) + "</p>" +
        '<div class="stats">' +
          '<div class="stat"><b>' + d.chapters.length + "</b><span>" + esc(t.chapters) + "</span></div>" +
          '<div class="stat"><b>' + sections + "</b><span>" + esc(t.sections) + "</span></div>" +
          '<div class="stat"><b>' + esc(fmtDate(first)) + "</b><span>" + esc(t.span) + "</span></div>" +
        "</div>";

      d.chapters.forEach(function (ch) {
        html += '<section class="chapter">' +
          '<div class="chapter-head"><span class="chapter-num">' + esc(ch.id.split("-")[0]) + '</span><h2>' + esc(ch.title) + "</h2></div>" +
          (ch.description ? '<p class="chapter-desc">' + esc(ch.description) + "</p>" : "") +
          '<ul class="sec-list">';
        ch.sections.forEach(function (sec) {
          html += '<li class="sec-row">' +
            '<span class="sec-idx">' + esc(sec.slug) + "</span>" +
            '<a href="' + routeHash("/a/" + sec.slug) + '">' + esc(sec.title) + "</a>" +
            '<span class="sec-meta">' + esc(fmtDate(sec.date)) + " · " + sec.minutes + " " + esc(t.minutes) + "</span>" +
          "</li>";
        });
        html += "</ul></section>";
      });
      html += '<p class="muted">' + esc(t.private) + "</p></div>";
      main.innerHTML = html;
    });
  }

  function flattenToc(items, depth) {
    var out = [];
    (items || []).forEach(function (it) {
      out.push({ id: it.id, name: it.name || it.html || "", level: it.level || depth });
      out = out.concat(flattenToc(it.children, (it.level || depth) + 1));
    });
    return out;
  }

  function renderArticle(slug) {
    loading();
    api("/api/doc/" + encodeURIComponent(slug) + "?lang=" + lang).then(function (res) {
      if (res.status === 403) return renderPending();
      if (res.status === 401) { clearToken(); return renderLogin(); }
      if (res.status === 404) return renderError(T().notFound);
      if (!res.ok) return renderError(errText(res.data.error));
      var d = res.data;
      var t = T();
      var flat = flattenToc(d.toc, 1);
      var tocHtml = flat.length
        ? '<aside class="toc"><p class="toc-title">' + esc(t.toc) + "</p><ul>" +
          flat.map(function (it) {
            return '<li class="lv' + Math.min(4, Math.max(2, it.level)) + '"><a href="#' + esc(it.id) + '">' + esc(it.name) + "</a></li>";
          }).join("") + "</ul></aside>"
        : "";
      var tags = (d.tags || []).map(function (tag) { return '<span class="tag">' + esc(tag) + "</span>"; }).join("");
      main.innerHTML =
        '<div class="shell article-shell">' +
          "<article>" +
            '<div class="article-top"><a href="' + routeHash("/") + '">' + esc(t.backHome) + "</a><span>/</span><span>" + esc(d.chapter.title) + "</span></div>" +
            '<h1 class="article-title">' + esc(d.title) + "</h1>" +
            '<div class="article-meta"><span>' + esc(fmtDate(d.date)) + "</span><span>" + d.minutes + " " + esc(t.minutes) + "</span>" + tags + "</div>" +
            '<div class="prose" id="kb-prose">' + d.html + "</div>" +
            '<nav class="article-nav">' +
              (d.prev ? '<a href="' + routeHash("/a/" + d.prev.slug) + '"><span class="dir">' + esc(t.prev) + "</span>" + esc(d.prev.title) + "</a>" : "<span></span>") +
              (d.next ? '<a class="next" href="' + routeHash("/a/" + d.next.slug) + '"><span class="dir">' + esc(t.next) + "</span>" + esc(d.next.title) + "</a>" : "<span></span>") +
            "</nav>" +
          "</article>" +
          tocHtml +
        "</div>";

      document.title = d.title + " · chenk.top";
      decorateArticle(document.getElementById("kb-prose"));
      renderMermaid(document.getElementById("kb-prose"));
      observeToc();
    });
  }

  function decorateArticle(root) {
    root.querySelectorAll("a[href]").forEach(function (a) {
      var href = a.getAttribute("href");
      if (href && href.indexOf("/a/") === 0) {
        var slug = href.slice(3).split(/[?#]/)[0];
        a.setAttribute("href", "#/a/" + slug + "?lang=" + lang);
      } else if (href && href.charAt(0) === "#") {
        a.addEventListener("click", function (ev) {
          var el = document.getElementById(href.slice(1));
          if (el) { ev.preventDefault(); el.scrollIntoView(); }
        });
      } else if (href && /^https?:/.test(href) && href.indexOf(location.origin) !== 0) {
        a.setAttribute("target", "_blank");
        a.setAttribute("rel", "noopener noreferrer");
      }
    });
    root.querySelectorAll("table").forEach(function (table) {
      if (!table.parentElement.classList.contains("table-scroll")) {
        var wrap = document.createElement("div");
        wrap.className = "table-scroll";
        table.parentNode.insertBefore(wrap, table);
        wrap.appendChild(table);
      }
    });
  }

  function renderMermaid(root) {
    var blocks = root.querySelectorAll(".mermaid");
    if (!blocks.length) return;
    loadMermaid().then(function (mermaid) {
      if (!mermaid) return;
      var dark = document.documentElement.getAttribute("data-theme") === "dark";
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "base",
        themeVariables: {
          background: dark ? "#17171f" : "#fffefb",
          primaryColor: dark ? "#1c2440" : "#eceafd",
          primaryTextColor: dark ? "#e9e7e2" : "#181821",
          primaryBorderColor: dark ? "#82b6ff" : "#1f6feb",
          lineColor: dark ? "#8b8b9a" : "#5a5a68",
          fontFamily: "Inter, PingFang SC, sans-serif"
        },
        flowchart: { curve: "basis", htmlLabels: true },
        sequence: { useMaxWidth: true },
        gantt: { useMaxWidth: true }
      });
      mermaid.run({ nodes: Array.prototype.slice.call(blocks) }).catch(function () {});
    });
  }

  function loadMermaid() {
    if (window.mermaid) return Promise.resolve(window.mermaid);
    if (mermaidPromise) return mermaidPromise;
    mermaidPromise = new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = API + "/static/mermaid.min.js";
      s.async = true;
      s.onload = function () { resolve(window.mermaid || null); };
      s.onerror = function () { resolve(null); };
      document.head.appendChild(s);
    });
    return mermaidPromise;
  }

  function observeToc() {
    var headings = main.querySelectorAll("#kb-prose h2, #kb-prose h3");
    var links = main.querySelectorAll(".toc a");
    if (!headings.length || !links.length || !("IntersectionObserver" in window)) return;
    var byId = {};
    links.forEach(function (l) { byId[l.getAttribute("href").slice(1)] = l; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var link = byId[en.target.id];
        if (!link) return;
        links.forEach(function (l) { l.style.color = ""; });
        if (en.isIntersecting) link.style.color = "var(--ink)";
      });
    }, { rootMargin: "-70px 0px -72% 0px" });
    headings.forEach(function (h) { if (h.id) obs.observe(h); });
  }

  function renderAccount() {
    var t = T();
    main.innerHTML =
      '<div class="shell" style="max-width:720px">' +
        '<p class="eyebrow">' + esc(t.account) + "</p>" +
        '<h1 class="kb-title">' + esc(user ? user.name : "") + "</h1>" +
        '<p class="lede">' + esc(user ? user.email : "") + " · " + esc(t.private) + "</p>" +
        '<div id="kb-flash"></div>' +
        '<div class="panel">' +
          "<h2>" + esc(t.changePw) + "</h2>" +
          '<form class="form" id="kb-pw-form" novalidate>' +
            '<div class="field"><label for="pw-cur">' + esc(t.currentPw) + '</label><input id="pw-cur" name="current" type="password" autocomplete="current-password" required></div>' +
            '<div class="field"><label for="pw-new">' + esc(t.newPw) + '</label><input id="pw-new" name="next" type="password" minlength="10" autocomplete="new-password" required></div>' +
            '<button class="btn" type="submit">' + esc(t.changePw) + "</button>" +
          "</form>" +
          '<p style="margin-top:16px"><button class="btn ghost" type="button" id="kb-signout">' + esc(t.signoutAll) + "</button></p>" +
        "</div>" +
      "</div>";
    document.getElementById("kb-signout").addEventListener("click", signOut);
    document.getElementById("kb-pw-form").addEventListener("submit", function (ev) {
      ev.preventDefault();
      var form = ev.target;
      var btn = form.querySelector("button");
      btn.disabled = true;
      api("/api/password", { body: { current: form.current.value, new: form.next.value } }).then(function (res) {
        btn.disabled = false;
        document.getElementById("kb-flash").innerHTML = res.ok
          ? flash(t.pwChanged, "ok")
          : flash(errText(res.data.error), "error");
        if (res.ok) form.reset();
      });
    });
  }

  function renderAdmin() {
    if (!user || !user.admin) return renderError(T().adminOnly);
    loading();
    api("/api/admin/users").then(function (res) {
      if (!res.ok) return renderError(errText(res.data.error));
      var users = res.data.users;
      var t = T();
      var pending = users.filter(function (u) { return u.status === "pending"; }).length;
      var approved = users.filter(function (u) { return u.status === "approved"; }).length;
      var rows = users.map(function (u) {
        var actions = [];
        if (u.status === "pending") actions.push(["approve", t.approve, "primary"]);
        if (u.status === "approved" && u.id !== user.id) actions.push(["disable", t.disable, "ghost"]);
        if (u.status === "disabled") actions.push(["enable", t.enable, "ghost"]);
        if (u.id !== user.id) actions.push([u.admin ? "revoke_admin" : "make_admin", u.admin ? t.revokeAdmin : t.makeAdmin, "ghost"]);
        return '<tr data-id="' + u.id + '">' +
          '<td class="who"><strong>' + esc(u.name) + '</strong><span class="muted">' + esc(u.email) + "</span></td>" +
          '<td><span class="badge ' + (u.status === "approved" ? "ok" : u.status === "pending" ? "warn" : "") + '">' + esc(u.status === "approved" ? t.stApproved : u.status === "pending" ? t.stPending : t.stDisabled) + "</span>" + (u.admin ? ' <span class="tag">admin</span>' : "") + "</td>" +
          '<td class="muted">' + esc(fmtDate(new Date(u.created_at * 1000).toISOString().slice(0, 10))) + "</td>" +
          '<td><div class="actions">' + actions.map(function (a) {
            return '<button class="btn small ' + a[2] + '" type="button" data-act="' + a[0] + '">' + esc(a[1]) + "</button>";
          }).join("") + "</div></td>" +
        "</tr>";
      }).join("");
      main.innerHTML =
        '<div class="shell">' +
          '<div class="admin-head"><div><p class="eyebrow">' + esc(t.admin) + '</p><h1 class="kb-title">' + esc(t.admin) + "</h1>" +
            '<p class="lede">' + esc(t.adminLede) + "</p></div>" +
            '<div class="stats" style="border:none;margin:0;padding:0">' +
              '<div class="stat"><b>' + users.length + "</b><span>" + esc(t.members) + "</span></div>" +
              '<div class="stat"><b>' + pending + "</b><span>" + esc(t.pendingCount) + "</span></div>" +
              '<div class="stat"><b>' + approved + "</b><span>" + esc(t.approvedCount) + "</span></div>" +
            "</div></div>" +
          '<div id="kb-flash"></div>' +
          '<div class="panel" style="overflow-x:auto"><table class="admin-table"><thead><tr>' +
            "<th>" + esc(t.colMember) + "</th><th>" + esc(t.colStatus) + "</th><th>" + esc(t.colJoined) + '</th><th style="text-align:right">' + esc(t.colActions) + "</th>" +
          "</tr></thead><tbody>" + rows + "</tbody></table></div>" +
        "</div>";
      main.querySelectorAll("button[data-act]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var tr = btn.closest("tr");
          var id = tr.getAttribute("data-id");
          btn.disabled = true;
          api("/api/admin/action", { body: { user_id: Number(id), action: btn.getAttribute("data-act") } }).then(function (res) {
            if (res.ok) return renderAdmin();
            btn.disabled = false;
            document.getElementById("kb-flash").innerHTML = flash(errText(res.data.error), "error");
          });
        });
      });
    });
  }

  function flash(message, kind) {
    return '<div class="flash ' + (kind || "error") + '">' + esc(message) + "</div>";
  }
  function renderError(message) {
    main.innerHTML = '<div class="shell" style="max-width:720px"><div class="panel"><p>' + esc(message) + "</p>" +
      '<p style="margin-top:12px"><a href="' + routeHash("/") + '">' + esc(T().backHome) + "</a></p></div></div>";
  }

  /* ---------- boot ---------- */
  window.addEventListener("hashchange", route);
  document.getElementById("kb-lang").addEventListener("click", function (ev) {
    ev.preventDefault();
    lang = lang === "en" ? "zh" : "en";
    try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
    location.hash = "#" + currentPath() + "?lang=" + lang;
    route();
  });
  if (!location.hash) location.hash = "#/?lang=" + lang;
  if (token) {
    api("/api/me").then(function (res) {
      if (res.ok) user = res.data.user;
      else if (res.status !== 0) clearToken();
      route();
    });
  } else {
    route();
  }
})();
