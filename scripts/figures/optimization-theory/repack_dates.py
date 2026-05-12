#!/usr/bin/env python3
"""Pack optim 01-12 into 2022-09-14..2022-09-30 dense block.
Move llmgr/gcsan standalones to 2023-01-22/29 so the block stays clean.
"""
import os, glob, re

BASE = '/root/chenk-hugo/content'

# Article filename → new date
NEW_DATES = {
    # optim series — assigned to dense block, 1-2 days apart
    '01-convex-analysis-foundations.md': '2022-09-14 09:00:00',
    '02-smoothness-strong-convexity-nesterov.md': '2022-09-15 09:00:00',
    '03-gradient-descent-family.md': '2022-09-16 09:00:00',
    '04-learning-rate-schedules.md': '2022-09-18 09:00:00',
    '05-acceleration-beyond-nesterov.md': '2022-09-20 09:00:00',
    '06-composite-proximal-methods.md': '2022-09-21 09:00:00',
    '07-second-order-methods.md': '2022-09-22 09:00:00',
    '08-lagrangian-duality-kkt.md': '2022-09-24 09:00:00',
    '09-interior-point-barrier.md': '2022-09-26 09:00:00',
    '10-stochastic-variance-reduction.md': '2022-09-27 09:00:00',
    '11-nonconvex-saddle-escape.md': '2022-09-29 09:00:00',
    '12-discrete-global-optimization.md': '2022-09-30 09:00:00',
}

# Standalones to move out of the way (2022-09-27, 2022-09-30 → 2023-01-22, 2023-01-29)
STANDALONE_MOVES = {
    'llmgr.md': '2023-01-22 09:00:00',
    'gcsan.md': '2023-01-29 09:00:00',
}

def update_date(filepath, new_date):
    with open(filepath) as f:
        content = f.read()
    new_content = re.sub(
        r'^date:\s*\S.*$',
        f'date: {new_date}',
        content,
        count=1,
        flags=re.MULTILINE
    )
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

# Update optim series articles (both EN and ZH)
print("== Optimization series dates ==")
for fname, date in NEW_DATES.items():
    for lang in ['en', 'zh']:
        path = os.path.join(BASE, lang, 'optimization-theory', fname)
        if os.path.exists(path):
            if update_date(path, date):
                print(f"  {lang}/optimization-theory/{fname}: {date}")
        else:
            print(f"  SKIP (not yet created): {lang}/optimization-theory/{fname}")

print("\n== Standalone moves ==")
for fname, date in STANDALONE_MOVES.items():
    for lang in ['en', 'zh']:
        # Find the file (zh might have different name)
        candidates = glob.glob(os.path.join(BASE, lang, 'standalone', '*.md'))
        for c in candidates:
            with open(c) as f:
                content = f.read()
            tk_match = re.search(r'^translationKey:\s*"?([^"\n]+)"?', content, re.MULTILINE)
            tk = tk_match.group(1) if tk_match else None
            base = os.path.basename(c)
            # Match by basename for EN, by translationKey for ZH
            if lang == 'en' and base == fname:
                if update_date(c, date):
                    print(f"  {lang}/standalone/{base}: {date}")
            elif lang == 'zh':
                # Find ZH counterpart by translationKey pointing to same article
                # llmgr's tk likely "llmgr" or similar
                fname_stem = fname[:-3]
                if tk and (fname_stem in tk or tk in fname_stem):
                    if update_date(c, date):
                        print(f"  {lang}/standalone/{base} (tk={tk}): {date}")

print("\nDone.")
