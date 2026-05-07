# QA Report — chenk.top (2026-04-25)

## Part A: Dark mode

**Status: WORKING CORRECTLY** (no fix needed)

Playwright verified end-to-end:

| Scenario | data-theme | body bg | --paper |
|---|---|---|---|
| `prefers-color-scheme: dark` (no LS) | `dark` (set by inline FOUC script) | `rgb(15, 17, 23)` | `#0f1117` |
| `prefers-color-scheme: light` (no LS) | (unset) | `rgb(253, 252, 249)` | `#fdfcf9` |
| After clicking `[data-theme-toggle]` from light | `dark` | `rgb(15, 17, 23)` | `#0f1117` |
| After reload (LS = `dark`) | `dark` | `rgb(15, 17, 23)` | persisted |

`dark.css` is bundled last in `head.html`, the `:root[data-theme="dark"] body` selector overrides the body radial-gradient correctly, and `theme.js` sets the attribute + writes `localStorage["chenk-theme"]`.

The user-reported issue ("data-theme=dark but bg stays light") could not be reproduced. Most likely a stale browser cache of the old `site.min.<hash>.css` from before dark.css was wired up. New CSS hash deployed today is `0baf21daefd7e3...` — a hard reload (Cmd+Shift+R) should pick it up.

Screenshots: `/root/qa/home_dark_pref.png`, `/root/qa/home_dark_toggle.png`.

## Part B: QA findings

### Fixed and deployed

1. **Mobile horizontal overflow on PAI article** (375x812 viewport)
   - Root cause: `.prose table` had no overflow handling; PAI article tables (3-column comparison) measured up to 636px wide and pushed body to 412px on a 375px screen.
   - Fix: `themes/chenk/assets/css/typography.css` — set `.prose table` to `display: block; overflow-x: auto` on viewports `< 721px`, restore `display: table` above. Tables now scroll internally; body width returns to 375.

2. **Broken footer link `/en/me/` and `/zh/me/`**
   - Root cause: `themes/chenk/layouts/partials/footer.html` had `<li><a href="{{ $base }}/me/">{{ site.Params.author }}</a></li>` but no content file at `content/{en,zh}/me.md` (only the layout `themes/chenk/layouts/me/single.html` exists, with no source content).
   - Fix: removed the redundant `/me/` `<li>` (the same column already has an `/about/` link to "About" which renders the author profile).

Both fixes built (`hugo --gc --minify`) and pushed to `chenk.top.github.io@master` (commit `db5c779`). Verified live: mobile docW = winW = 375, table `overflowX: auto`, footer `/me/` count = 0.

### Verified OK (no issues)

- **Code blocks** on `/en/aliyun-pai/01-platform-overview/`: 4 `<pre>`, 8 `.highlight/.chroma`, 6 copy buttons present.
- **Language switcher** (EN -> ZH) on `/en/aliyun-pai/01-platform-overview/` -> `/zh/aliyun-pai/01-platform-overview/` returned 200.
- **Internal link sample** (30 links from `/en/series/`): all returned 200 except the `/me/` link, which is now removed.
- **Series index page**: design intentionally links each series card to the first chapter; chapter count badge ("N chapters") is computed from data — not a list of links. Counts match `len(filtered RegularPages)`. Working as designed.
- **Image 404s**: none observed on home, projects, series, or first 3 PAI articles. Some series cards reference `https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/<key>.jpg` and use `onerror` to remove themselves cleanly when missing — graceful.
- **Search input**: present on home.

### Logged but not fixed (overlap or ambiguous)

- **Math articles `/en/ml-math-derivations/01/`, `/en/linear-algebra/01/`, `/en/ode/01/` returned 404 in the test.** Reason: actual slugs are `01-introduction-and-mathematical-foundations`, `01-the-essence-of-vectors`, `01-origins-and-intuition`. These are NOT broken links on the site — they were just wrong test URLs in the QA script. The math rendering pipeline (KaTeX in `head.html`) was not exercised; recommend a follow-up that picks real article slugs.
- **`lang-switch-missing` for EN-only test URLs**: same root cause — those URLs 404 in EN, so no hreflang ZH counterpart is in the DOM. Not a real bug.

### Not fixed (out of scope per coordination notes)

- Cover images on series cards / homepage: another agent owns the OSS upload. None failed visibly today (graceful onerror).
- Matplotlib figures: another agent owns these.
- Projects page: skipped per instructions.

## Files changed

- `/root/chenk-hugo/themes/chenk/layouts/partials/footer.html`
- `/root/chenk-hugo/themes/chenk/assets/css/typography.css`

Backups of pre-edit files at `/tmp/footer.html.bak`, `/tmp/typography.css.bak`.
