const { listDir } = require('hexo-fs');
listDir('source').then(list => {
  const posts = list.filter(p => p.includes('_posts'));
  console.log('Total:', list.length, 'with _posts:', posts.length);
  console.log('Sample:', posts.slice(0, 5));
});
