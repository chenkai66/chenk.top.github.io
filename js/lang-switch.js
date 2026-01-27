/* global KEEP */

KEEP.initLangSwitch = () => {
  const triggers = document.querySelectorAll('.lang-switch-trigger');
  if (!triggers || !triggers.length) return;

  const getTargetPath = () => {
    const { pathname, search, hash } = window.location;

    // Only implement zh-CN (default, no prefix) <-> en (/en) toggle for now.
    const enPrefix = '/en';
    const isEn = pathname === enPrefix || pathname.startsWith(enPrefix + '/');

    let nextPathname;
    if (isEn) {
      nextPathname = pathname.replace(/^\/en(?=\/|$)/, '') || '/';
    } else {
      nextPathname = enPrefix + (pathname === '/' ? '/' : pathname);
    }

    return nextPathname + search + hash;
  };

  triggers.forEach(dom => {
    dom.addEventListener('click', () => {
      window.location.href = getTargetPath();
    });
  });
};

