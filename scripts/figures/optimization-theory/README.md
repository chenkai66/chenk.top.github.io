# Optimization-theory diagram generation scripts

These matplotlib scripts generate the figures embedded in the optimization-theory series.
Most were written by per-article subagents on 2026-05-12 and rescued from /tmp.

## Index

- `gen_convex_figs.py` — article 01 (convex analysis foundations)
- `make_figs.py` / `insert.py` / `patch.py` / `patch_zh.py` — article 02 fig6, fig7
- `make_figs_10.py` / `insert_figs_10.py` — article 10 (stochastic + variance reduction) figs 1–5
- `insert_figs.py` / `insert_images.py` — assorted helpers
- `fix_fig2_motion.py`, `fix_fig5_workflows.py`, `fix_fig7_template.py` — Linux 9 (Vim) figure regeneration

## Conventions

- Width 10–14in, dpi 180, white background, brand colors: `#2E5BFF` blue, `#D97706` copper, `#059669` green, `#7C3AED` purple, `#475569` gray
- Save to `/tmp/figN_*.png`, upload to `posts/en/optimization-theory/{slug}/figN_*.png` on OSS
- All labels in English even for ZH article (math is universal)

To regenerate, run on the server: `python3 scripts/figures/optimization-theory/{name}.py`
