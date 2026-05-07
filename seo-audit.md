# chenk.top SEO + RSS + UX Audit (2026-04-25)

Deploy SHA: 05f466ac

## SEO findings + fixes

### Before
| Issue | Status |
|---|---|
| robots.txt | **Missing** (404) |
| sitemap.xml | OK (sitemap index → /en/ + /zh/), hreflang already included |
| canonical | OK |
| hreflang | OK (en, zh, x-default) |
| meta description | OK (per-page) |
| og:type / og:title / og:description / og:url / og:locale | OK |
| og:image | **Missing** |
| og:site_name | **Missing** |
| twitter:card | OK (summary_large_image) |
| twitter:title / twitter:description / twitter:image | **Missing** |
| article:published_time / article:modified_time / article:author | **Missing** |
| article:tag | **Missing** |
| JSON-LD structured data | **Missing** |

### Fixes
1. Created `static/robots.txt` (allows all, disallows 404, points at sitemap).
2. Rewrote `themes/chenk/layouts/partials/head.html`:
   - Resolves `$ogImage` from `.Params.cover` → series cover at OSS → site default
   - Added `og:image`, `og:image:alt`, `og:site_name`
   - Added `article:published_time`, `article:modified_time` (when present), `article:author`, `article:tag` (one per tag)
   - Added Twitter Card `twitter:title`, `twitter:description`, `twitter:image`
   - Added JSON-LD `Article` schema (headline, description, image, author, publisher, datePublished, dateModified, mainEntityOfPage, inLanguage)
   - Added JSON-LD `WebSite` schema with `SearchAction` for the home page

## RSS findings + fixes

### Before
| Feed | Status |
|---|---|
| /en/index.xml | OK |
| /zh/index.xml | OK |
| /en/series/<x>/index.xml | **404** — `term` output didn't include RSS |

### Fixes
- `hugo.toml`: changed `term = ["HTML"]` → `term = ["HTML", "RSS"]`
- Per-series RSS now generated, e.g. `/en/series/aliyun-pai/index.xml` returns 200 and lists all chapters with full descriptions.

## Sitemap findings + fixes

### Before
- Sitemap index at `/sitemap.xml` → `/en/sitemap.xml` + `/zh/sitemap.xml`
- Each language sitemap lists all articles, taxonomy pages, hreflang alternate links
- 404.html not in sitemap (correct)

### Fixes
- None needed; sitemap is correct.

## UX dead-ends + fixes

### Before
- `series-nav.html` only showed prev/next chapter — **no "back to series index" link** if reader hit the last chapter (or the only chapter).
- That meant readers at the end of a series had no in-page path back to the series listing.

### Fixes
- Updated `series-nav.html`:
  - Last chapter (`prev` exists, `next` missing) now shows "← Previous" + "Next: Back to <Series> index →" linking to `/en|zh/series/<slug>/`
  - Single-chapter series now shows a standalone "More in this series → Back to <Series> index" block
  - i18n strings hardcoded (en/zh) since `i18n/` directory is empty in this repo

### Other UX checks
- Mobile drawer: has explicit close button (`data-drawer-close`) — OK
- 404 page: includes JS auto-lowercase redirect for legacy uppercase URLs — OK
- Series listing pages render correctly with hue/cover/progress
- Tag and category pages reachable (HTTP 200)

## Internal SEO health

Could not exhaustively crawl in 30-min budget but spot checks:
- No duplicate titles observed in sample
- All articles have descriptions in front matter (visible in RSS feed)
- Tag/category pages exist and link from articles via `<a class="tag">` in single.html

## Verification (post-deploy)

```
robots.txt           HTTP 200
sitemap.xml          HTTP 200
/en/series/aliyun-pai/index.xml  HTTP 200 (5 items, full descriptions)
Article meta tags:   og:image, og:site_name, twitter:title, twitter:image,
                     article:published, application/ld+json   all present
"Back to" link present on chapter article
```

Deploy SHA: 05f466ac
