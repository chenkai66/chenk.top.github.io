/* Hexo helper: translation_url(page)
 *
 * Pairs EN ↔ ZH posts by (series_key, series_order). Series in the en posts
 * is a string like "recommendation-systems"; in zh posts it's an object
 * { name, part, total }. We normalize to a series_key and an order number,
 * then look up the counterpart post.
 *
 * Returns a leading-slash path string, or null if no pair exists.
 */

function getSeriesKey(post) {
  if (!post.series) return null;
  if (typeof post.series === 'string') return post.series;
  if (typeof post.series === 'object') {
    if (post.series.slug) return post.series.slug;
    if (post.series.key) return post.series.key;
    // Map zh series.name back to a slug via theme.series
    return null;
  }
  return null;
}

function getSeriesOrder(post) {
  if (post.series_order != null) return Number(post.series_order);
  if (post.series && typeof post.series === 'object' && post.series.part != null) {
    return Number(post.series.part);
  }
  // Fall back to extracting from filename like "01-foo"
  if (post.source) {
    var m = String(post.source).match(/\/(\d+)[-_]/);
    if (m) return Number(m[1]);
  }
  return null;
}

// Map zh post's series.name (e.g. "推荐系统") to the en series key by part order.
// Build once after Hexo finishes generating the locals (lazy, per call).
function buildZhNameToKey(hexo) {
  var map = {};
  var theme = hexo.theme && hexo.theme.config;
  if (!theme || !theme.series) return map;
  // We don't have direct ZH→EN name mapping in theme config.
  // Heuristic: zh posts that share series_order with an en post in the same series can be paired.
  // So instead of name mapping, we use folder-based key inference below.
  return map;
}

function inferSeriesKeyFromPath(post) {
  if (!post.source) return null;
  // source like "_posts/en/recommendation-systems/01-fundamentals.md"
  var m = String(post.source).match(/_posts\/(?:en|zh)\/([^/]+)\//);
  return m ? m[1] : null;
}

hexo.extend.helper.register('translation_url', function(page) {
  if (!page) return null;
  var lang = page.lang;
  if (!lang) return null;

  var thisKey = getSeriesKey(page) || inferSeriesKeyFromPath(page);
  var thisOrder = getSeriesOrder(page);
  if (!thisKey || thisOrder == null) return null;

  var otherLang = lang === 'en' ? 'zh-CN' : 'en';
  var posts = hexo.locals.get('posts');
  if (!posts) return null;

  var match = null;
  posts.each(function(p) {
    if (match) return;
    if (p.lang !== otherLang) return;
    var pKey = getSeriesKey(p) || inferSeriesKeyFromPath(p);
    var pOrder = getSeriesOrder(p);
    if (pKey === thisKey && pOrder === thisOrder) {
      match = p;
    }
  });

  if (!match) return null;
  var path = match.path || '';
  return '/' + path.replace(/^\/+/, '');
});
