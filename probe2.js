const Hexo = require('hexo');
const hexo = new Hexo(process.cwd(), {silent: true});
hexo.init().then(() => {
  return hexo.source.process();
}).then(() => {
  return hexo.source.list();
}).then(list => {
  const posts = Object.keys(list).filter(p => p.includes('_posts'));
  console.log('Total source files:', Object.keys(list).length);
  console.log('Files with _posts:', posts.length);
  console.log('Sample posts:', posts.slice(0, 5));
}).catch(e => { console.error('ERR', e); process.exit(1); });
