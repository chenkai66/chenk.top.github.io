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

      'home.eyebrow': '技术 · 数学 · 好奇心',
      'home.hero.title': '机器学习、数学与计算机科学',
      'home.hero.subtitle': '面向人类的深度技术内容。从第一性原理到生产系统。',
      'home.stats.articles': '文章',
      'home.stats.series': '系列',
      'home.stats.languages': '语言',
      'home.featured': '精选系列',
      'home.recent': '最近文章',

      'series.title': '所有系列',
      'series.subtitle.suffix': '个进行中的系列，涵盖 ML、数学和 CS',

      'projects.title': '项目',
      'projects.subtitle': '我维护的系统，不只是构建过的东西。每一个都跑在真实域名上，承载真实用户。',
      'projects.footer': '三个都跑在同一台 ECS 上。llm4marketing.com 的仪表盘就是同一个控制平面 — 不同挂载、不同域名。',

      'archives.title': '归档',
      'archives.subtitle.suffix': '篇文章，持续增加中',

      'about.title': '关于',

      'me.tagline': '工程师 · 研究者 · Agent 系统的建造者',
      'me.cta.work': '看我的工作',
      'me.cta.blog': '阅读博客',
      'me.about.eyebrow': '关于',
      'me.bynumbers': '数字背后',
      'me.stats.articles': '文章',
      'me.stats.series': '系列',
      'me.stats.languages': '语言',
      'me.stats.systems': '在线系统',
      'me.stats.note': '数字随每次生成自动更新。',
      'me.projects.eyebrow': '生产中的项目',
      'me.projects.title': '我运行的，而不只是构建过的',
      'me.recent.eyebrow': '最近写作',
      'me.recent.title': '来自博客',
      'me.recent.cta': '全部文章 →',
      'me.contact.title': '想聊聊？',
      'me.contact.text': '我总是乐意聊聊 ML 系统、Agent 基础设施或长篇技术写作。给我留言——每一封我都会读。',

      'footer.about': '关于',
      'footer.series': '系列',
      'footer.site': '站点',
      'footer.allseries': '全部系列 →',
      'footer.rss': 'RSS 订阅',
      'footer.copyright.suffix': '以知识与好奇心构建。',
      'footer.powered': '驱动于',
      'footer.theme': '主题',

      'post.previous': '← 上一篇',
      'post.next': '下一篇 →',
      'post.tags': '标签',
      'post.toc': '本页内容',
      'post.updated': '更新于',
      'post.minRead': '分钟阅读',
      'post.words': '字',

      'search.placeholder': '搜索文章...',

      'pagination.newer': '← 更新',
      'pagination.older': '更早 →',
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
