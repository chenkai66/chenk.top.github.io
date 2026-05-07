# Comments Setup (Giscus)

This site uses [Giscus](https://giscus.app) for article comments — backed by GitHub Discussions on the site repo (`chenkai66/chenk.top.github.io`). Free, no tracking, GitHub login.

The Hugo scaffold (partial + CSS + config slot + single.html wiring) is already in place. Comments are gated behind `params.giscus.enabled` in `hugo.toml`. Until the steps below are completed, nothing renders on article pages.

## One-time setup

### 1. Enable GitHub Discussions on the site repo

1. Open https://github.com/chenkai66/chenk.top.github.io/settings
2. Scroll to **Features** → tick **Discussions**
3. Open https://github.com/chenkai66/chenk.top.github.io/discussions/categories
4. Create a new category:
   - Name: `Comments`
   - Description: `Article comments via Giscus`
   - Discussion format: **Announcement** (recommended — only maintainers can start a thread; Giscus does it for you per article)
   - (Q&A also works if you prefer thread voting.)

### 2. Install the Giscus GitHub App

Visit https://github.com/apps/giscus and install it. Restrict access to the single repo `chenkai66/chenk.top.github.io`.

### 3. Get the IDs from giscus.app

1. Go to https://giscus.app
2. **Repository**: `chenkai66/chenk.top.github.io`
   - Page should confirm: \"Success! This repository meets all of the above criteria.\"
3. **Page ↔ Discussions Mapping**: select **`pathname`** (matches the partial).
4. **Discussion Category**: select **`Comments`** (the category created in step 1).
5. **Features**: leave defaults (reactions on, metadata off, input position top — already matches the partial).
6. **Theme**: `preferred_color_scheme` (already in the partial).

Scroll to the **Enable giscus** block at the bottom. You will see a `<script>` snippet with two values you need:

```html
data-repo-id=\"R_kgDO...\"
data-category-id=\"DIC_kwDO...\"
```

### 4. Paste the IDs into `hugo.toml`

Edit `/root/chenk-hugo/hugo.toml`. Find the `[params.giscus]` block and update:

```toml
[params.giscus]
enabled = true                               # flip to true
repo = \"chenkai66/chenk.top.github.io\"
repoId = \"R_kgDO...\"                         # paste from giscus.app
category = \"Comments\"
categoryId = \"DIC_kwDO...\"                   # paste from giscus.app
```

### 5. Rebuild and deploy

```bash
cd /root/chenk-hugo && hugo --gc --minify
cd /tmp/chenk-deploy && git pull --rebase
rm -rf en zh covers css js *.html *.xml
cp -r /root/chenk-hugo/public/* .
echo www.chenk.top > CNAME
git add -A
git -c user.name=chenkai66 -c user.email=chenkai.nb.666@gmail.com commit -m \"Enable Giscus comments\"
git push origin master
```

Visit any article (e.g. https://www.chenk.top/en/aliyun-pai/01-platform-overview/) and scroll to the bottom — the Giscus widget should load.

## Files involved

- `themes/chenk/layouts/partials/comments.html` — the Giscus script tag, gated by `Site.Params.giscus.enabled`
- `themes/chenk/assets/css/article.css` — `.comments-section` styles (appended at end)
- `themes/chenk/layouts/_default/single.html` — calls `{{ partial \"comments.html\" . }}` after `.article-end`
- `hugo.toml` — `[params.giscus]` block

## Disabling

Set `enabled = false` in `hugo.toml` and rebuild. The partial renders nothing when disabled — no script tag, no third-party request.

## Notes

- `data-mapping=\"pathname\"` means each article URL gets its own discussion thread, auto-created on first comment. The thread title equals the article path.
- Bilingual: the partial picks Giscus locale (`zh-CN` vs `en`) from the page language. Both EN and ZH versions of an article get **separate** discussions because their pathnames differ (`/en/...` vs `/zh/...`).
- If you ever rename a slug, the old discussion will be orphaned. Either rename the GitHub Discussion title to the new path, or accept losing those comments.
