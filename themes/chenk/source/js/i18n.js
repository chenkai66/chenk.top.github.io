/**
 * Site-wide i18n: translates static text marked with [data-i18n] attributes.
 * The current language is read from localStorage('siteLang'), which the
 * language switcher in main.js writes ('en' or 'zh').
 */
(function () {
  window.I18N = {
    en: {
      'nav.home': 'Home',
      'nav.series': 'Series',
      'nav.projects': 'Projects',
      'nav.archives': 'Archives',
      'nav.about': 'About',

      'home.eyebrow': 'Tech · Mathematics · Curiosity',
      'home.hero.title': 'Machine Learning, Mathematics & Computer Science',
      'home.hero.subtitle': 'Deep technical content explained for humans. From first principles to production systems.',
      'home.stats.articles': 'Articles',
      'home.stats.series': 'Series',
      'home.stats.languages': 'Languages',
      'home.featured': 'Featured Series',
      'home.recent': 'Recent Articles',

      'series.title': 'All Series',
      'series.subtitle.suffix': 'ongoing series covering ML, math, and CS',

      'projects.title': 'Projects',
      'projects.subtitle': 'Systems I run, not just things I built. Each one lives on a real domain and carries real users.',
      'projects.footer': 'All three live on a single ECS box. The dashboard at llm4marketing.com is the same control plane that serves both — different mounts, different domains.',

      'archives.title': 'Archives',
      'archives.subtitle.suffix': 'articles and counting',

      'about.title': 'About',

      'me.tagline': 'Engineer · Researcher · Builder of agent systems',
      'me.cta.work': 'See my work',
      'me.cta.blog': 'Read the blog',
      'me.about.eyebrow': 'About',
      'me.bynumbers': 'By the numbers',
      'me.stats.articles': 'Articles',
      'me.stats.series': 'Series',
      'me.stats.languages': 'Languages',
      'me.stats.systems': 'Live Systems',
      'me.stats.note': 'Numbers update automatically with each generate.',
      'me.projects.eyebrow': 'Projects in production',
      'me.projects.title': 'Things I run, not just things I built',
      'me.recent.eyebrow': 'Recent writing',
      'me.recent.title': 'From the blog',
      'me.recent.cta': 'All articles →',
      'me.contact.title': 'Want to talk?',
      'me.contact.text': "I'm always up for a chat about ML systems, agent infrastructure, or long-form technical writing. Drop a line — I read everything.",

      'footer.about': 'About',
      'footer.series': 'Series',
      'footer.site': 'Site',
      'footer.allseries': 'All series →',
      'footer.rss': 'RSS Feed',
      'footer.copyright.suffix': 'Built with knowledge and curiosity.',
      'footer.powered': 'Powered by',
      'footer.theme': 'Theme',

      'post.previous': '← Previous',
      'post.next': 'Next →',
      'post.tags': 'Tags',
      'post.toc': 'On This Page',
      'post.updated': 'Updated',
      'post.minRead': 'min read',
      'post.words': 'words',

      'search.placeholder': 'Search articles...',

      'pagination.newer': '← Newer',
      'pagination.older': 'Older →',
      'pagination.page': 'Page',
      'pagination.of': 'of'
    },

    'zh-CN': {
      'nav.home': '首页',
      'nav.series': '系列',
      'nav.projects': '项目',
      'nav.archives': '归档',
      'nav.about': '关于',

      'home.eyebrow': '技术 · 数学 · 一些不太正经的好奇心',
      'home.hero.title': '机器学习、数学、计算机系统',
      'home.hero.subtitle': '把硬核技术写得人能读懂。从底层原理一路推到生产部署。',
      'home.stats.articles': '文章',
      'home.stats.series': '系列',
      'home.stats.languages': '语种',
      'home.featured': '精选系列',
      'home.recent': '最近更新',

      'series.title': '全部系列',
      'series.subtitle.suffix': '个连载中的系列，覆盖机器学习、数学、计算机',

      'projects.title': '在做的项目',
      'projects.subtitle': '不是写过 demo 那种"做过"，是真的在线上跑、在真实域名上服务真实用户的系统。',
      'projects.footer': '三个项目跑在同一台 ECS 上。llm4marketing.com 的控制台就是这套统一的 agent 平面，不同挂载点、不同域名。',

      'archives.title': '归档',
      'archives.subtitle.suffix': '篇，仍在增长',

      'about.title': '关于',

      'me.tagline': '工程师 · 研究者 · 在做 Agent 系统',
      'me.cta.work': '看看我的项目',
      'me.cta.blog': '去看博客',
      'me.about.eyebrow': '关于我',
      'me.bynumbers': '一些数字',
      'me.stats.articles': '篇文章',
      'me.stats.series': '个系列',
      'me.stats.languages': '种语言',
      'me.stats.systems': '个线上系统',
      'me.stats.note': '数字在每次构建时自动更新。',
      'me.projects.eyebrow': '生产中的项目',
      'me.projects.title': '在线上跑的，不只是写过的',
      'me.recent.eyebrow': '最近写的',
      'me.recent.title': '博客里的',
      'me.recent.cta': '全部文章 →',
      'me.contact.title': '聊一聊？',
      'me.contact.text': '随时欢迎聊 ML 系统、Agent 基础设施、或者长文写作。每一封邮件我都会回。',

      'footer.about': '关于',
      'footer.series': '系列',
      'footer.site': '站内',
      'footer.allseries': '全部系列 →',
      'footer.rss': 'RSS 订阅',
      'footer.copyright.suffix': '以知识和好奇心为燃料。',
      'footer.powered': '驱动',
      'footer.theme': '主题',

      'post.previous': '← 上一篇',
      'post.next': '下一篇 →',
      'post.tags': '标签',
      'post.toc': '本页目录',
      'post.updated': '更新于',
      'post.minRead': '分钟读完',
      'post.words': '字',

      'search.placeholder': '搜索文章……',

      'pagination.newer': '← 较新',
      'pagination.older': '较早 →',
      'pagination.page': '第',
      'pagination.of': '页 / 共'
    }
  };

  function resolveLang() {
    var saved = localStorage.getItem('siteLang') || '';
    if (saved.indexOf('zh') === 0) return 'zh-CN';
    if (saved === 'en') return 'en';
    // Fall back to the page's declared language
    var pageLang = document.documentElement.getAttribute('data-page-lang') || '';
    if (pageLang.indexOf('zh') === 0) return 'zh-CN';
    return 'en';
  }

  function applyI18n() {
    var lang = resolveLang();
    var dict = window.I18N[lang] || window.I18N.en;

    // Translate text content for any element with [data-i18n="key"]
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      var key = el.getAttribute('data-i18n');
      if (dict[key] != null) el.textContent = dict[key];
    });

    // Translate attributes via [data-i18n-attr="attrName:key;attrName2:key2"]
    document.querySelectorAll('[data-i18n-attr]').forEach(function (el) {
      var spec = el.getAttribute('data-i18n-attr') || '';
      spec.split(';').forEach(function (pair) {
        var parts = pair.split(':');
        if (parts.length !== 2) return;
        var attr = parts[0].trim();
        var key = parts[1].trim();
        if (attr && dict[key] != null) el.setAttribute(attr, dict[key]);
      });
    });

    document.documentElement.setAttribute('data-site-lang', lang);
  }

  // Run immediately so paint reflects translated text
  applyI18n();

  // Re-run on DOMContentLoaded in case script was placed before content
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyI18n);
  }

  // Expose for re-application after dynamic content
  window.applyI18n = applyI18n;
})();
