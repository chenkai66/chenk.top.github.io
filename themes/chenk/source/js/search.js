/* ============================================
   Local Search — Enhanced
   ============================================ */

(function() {
  'use strict';

  var searchInput = document.getElementById('search-input');
  var searchResults = document.getElementById('search-results');
  var searchData = null;

  if (!searchInput || !searchResults) return;

  // Load search data
  function loadSearchData() {
    if (searchData) return Promise.resolve(searchData);
    return fetch('/search.json')
      .then(function(res) {
        if (!res.ok) throw new Error('Search data not found');
        return res.json();
      })
      .then(function(data) { searchData = data; return data; })
      .catch(function() {
        searchResults.innerHTML = '<p class="search-empty">Search index not available. Run <code>hexo generate</code> first.</p>';
        return [];
      });
  }

  // Highlight matching text
  function highlightMatch(text, query) {
    if (!text || !query) return text;
    var escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    var regex = new RegExp('(' + escaped + ')', 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  }

  // Search function
  function search(query) {
    if (!query || query.length < 2) {
      searchResults.innerHTML = '<p class="search-empty">Type at least 2 characters to search...</p>';
      return;
    }

    loadSearchData().then(function(data) {
      if (!data || data.length === 0) return;

      var results = [];
      var q = query.toLowerCase();

      data.forEach(function(item) {
        var title = (item.title || '').toLowerCase();
        var content = (item.content || '').toLowerCase();
        var titleMatch = title.indexOf(q);
        var contentMatch = content.indexOf(q);

        if (titleMatch !== -1 || contentMatch !== -1) {
          var score = 0;
          if (titleMatch !== -1) score += 10;
          if (titleMatch === 0) score += 5; // boost exact start
          if (contentMatch !== -1) score += 1;
          results.push({ item: item, score: score, contentMatch: contentMatch });
        }
      });

      results.sort(function(a, b) { return b.score - a.score; });

      if (results.length === 0) {
        searchResults.innerHTML = '<p class="search-empty">No results found for "' + escapeHtml(query) + '"</p>';
        return;
      }

      var html = '<div class="search-count" style="padding:0.5rem 1.5rem;font-size:0.78rem;color:var(--color-text-muted);border-bottom:1px solid var(--color-border-light);">' +
        results.length + ' result' + (results.length !== 1 ? 's' : '') + '</div>';

      html += results.slice(0, 20).map(function(r) {
        var excerpt = '';
        if (r.contentMatch !== -1) {
          var start = Math.max(0, r.contentMatch - 50);
          var end = Math.min(r.item.content.length, r.contentMatch + 100);
          excerpt = (start > 0 ? '...' : '') +
            r.item.content.substring(start, end) +
            (end < r.item.content.length ? '...' : '');
          excerpt = highlightMatch(escapeHtml(excerpt), query);
        }

        var title = highlightMatch(escapeHtml(r.item.title), query);

        return '<a href="' + r.item.url + '" class="search-result-item">' +
          '<span class="search-result-title">' + title + '</span>' +
          (excerpt ? '<span class="search-result-excerpt">' + excerpt + '</span>' : '') +
          '</a>';
      }).join('');

      searchResults.innerHTML = html;
    });
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Debounced input
  var timer;
  searchInput.addEventListener('input', function() {
    clearTimeout(timer);
    timer = setTimeout(function() {
      search(searchInput.value.trim());
    }, 200);
  });

  // Handle Enter to navigate to first result
  searchInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      var firstResult = searchResults.querySelector('.search-result-item');
      if (firstResult) {
        window.location.href = firstResult.getAttribute('href');
      }
    }
    // Arrow key navigation in results
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      var items = searchResults.querySelectorAll('.search-result-item');
      if (items.length === 0) return;

      var current = searchResults.querySelector('.search-result-item.focused');
      var idx = -1;
      if (current) {
        idx = Array.from(items).indexOf(current);
        current.classList.remove('focused');
      }

      if (e.key === 'ArrowDown') {
        idx = Math.min(idx + 1, items.length - 1);
      } else {
        idx = Math.max(idx - 1, 0);
      }

      items[idx].classList.add('focused');
      items[idx].scrollIntoView({ block: 'nearest' });
    }
  });
})();
