const Hexo = require('hexo');
const hexo = new Hexo(process.cwd(), {silent: true});
hexo.init().then(() => {
  console.log('source dir:', hexo.source_dir);
  console.log('Box source:', hexo.source.base);
  return hexo.source._readDir(hexo.source.base);
}).then(files => {
  console.log('files found:', files ? files.length : 'undefined');
  if (files) console.log('sample:', files.slice(0,3));
}).catch(e => { console.error('ERR', e.message); });
