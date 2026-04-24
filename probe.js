const Hexo = require('hexo');
const path = require('path');
const hexo = new Hexo(process.cwd(), {silent: false, debug: true});
hexo.init().then(() => {
  return hexo.load();
}).then(() => {
  const posts = hexo.locals.get('posts');
  console.log('Total posts:', posts.length);
  const p = hexo.model('Post').find({}).limit(3).data;
  p.forEach(x => console.log('  ', x.source, '->', x.title));
  console.log('isRenderable test for source/_posts/Personal.md:');
  console.log('  ', hexo.render.isRenderable('source/_posts/Personal.md'));
  console.log('  ', hexo.render.isRenderable('Personal.md'));
}).catch(e => { console.error('ERR', e); process.exit(1); });
