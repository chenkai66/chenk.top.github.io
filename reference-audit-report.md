# arXiv Citation Audit Report (chenk.top)

Generated: 2026-04-25
Source: `/root/chenk-hugo/content/en/**/*.md`
Method: regex extract `arXiv:NNNN.NNNNN` -> arxiv.org/api/query (3s throttle) -> compare title/author/year

## Summary

- **Total citations:** 250
- **Unique arXiv IDs:** 219
- **MALFORMED_ID:** 0 (all matched YYYY.NNNNN)
- **NOT_FOUND (hallucinated arXiv ID):** 0 - every ID resolves to a real paper on arXiv
- **REAL_DRIFT (author/title mismatch after manual review):** 4
- **REAL_MATCH:** 215

**Headline finding:** No fully hallucinated arXiv IDs. Every cited ID points to a real paper. However, **3 IDs are mis-attributed** (wrong paper or fabricated authors), which is a different but equally serious form of LLM hallucination - the citation looks authoritative but the paper described is not the paper at that arXiv URL. One additional case has a paraphrased title.

## Confirmed problematic citations (REAL_DRIFT)

### `arXiv:2104.10013`
- **Issue:** WRONG_PAPER: Article cites this ID for 'Extended PINNs (XPINNs) by Jagtap & Karniadakis (2020)' but arXiv:2104.10013 is actually 'Parallel Physics-Informed Neural Networks via Domain Decomposition' by Khemraj Shukla, Ameya D. Jagtap, George Em Karniadakis (2021). The XPINNs paper does have an arXiv preprint (2009.04525) but a different ID was used.
- **Actual paper:** *Parallel Physics-Informed Neural Networks via Domain Decomposition* - Khemraj Shukla (2021)
- **Cited in:**
  - `content/en/pde-ml/01-Physics-Informed-Neural-Networks.md:333`
    Claim: `[^xpinn]: A. D. Jagtap, G. E. Karniadakis. *Extended Physics-Informed Neural Networks (XPINNs).* Commun. Comput. Phys., 28(5):2002–2041, 2020.`

### `arXiv:2205.10573`
- **Issue:** WRONG_AUTHORS: Article attributes 'Spectral Neural Operators' to 'Tran, A., Mathews, A., Xie, L., & Ong, C. S.' but arXiv:2205.10573 is actually authored by V. Fanaskov & I. Oseledets. (The cited title matches; the author list appears fabricated.)
- **Actual paper:** *Spectral Neural Operators* - V. Fanaskov (2022)
- **Cited in:**
  - `content/en/pde-ml/02-Neural-Operator-Theory.md:339`
    Claim: `- Tran, A., Mathews, A., Xie, L., & Ong, C. S. (2022). [Spectral Neural Operators](https://arxiv.org/abs/2205.10573).`

### `arXiv:2410.13228`
- **Issue:** WRONG_AUTHORS_AND_TITLE: Article cites 'Z. Liu et al. From PINNs to PIKANs: Physics-Informed Kolmogorov-Arnold Networks' but arXiv:2410.13228 is by Juan Diego Toscano et al, titled 'From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning'. (Likely confusion with the KAN paper by Liu et al, arXiv:2404.19756.)
- **Actual paper:** *From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning* - Juan Diego Toscano (2024)
- **Cited in:**
  - `content/en/pde-ml/01-Physics-Informed-Neural-Networks.md:336`
    Claim: `[^pikan]: Z. Liu et al. *From PINNs to PIKANs: Physics-Informed Kolmogorov-Arnold Networks.*`
  - `content/en/pde-ml/01-Physics-Informed-Neural-Networks.md:336`
    Claim: `[^pikan]: Z. Liu et al. *From PINNs to PIKANs: Physics-Informed Kolmogorov-Arnold Networks.* [arXiv:2410.13228](https://arxiv.org/abs/2410.13228), 2024.`

### `arXiv:2411.07279`
- **Issue:** TITLE_DRIFT: Article cites title as '...test-time training for abstract reasoning' but arXiv title is 'The Surprising Effectiveness of Test-Time Training for Few-Shot Learning'. Authors and year match.
- **Actual paper:** *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning* - Ekin Akyürek (2024)
- **Cited in:**
  - `content/en/llm-engineering/04-post-training.md:411`
    Claim: `- Akyürek, E., Damani, M., Qiu, L., et al. (2024). The surprising effectiveness of test-time training for abstract reasoning.`

## Per-series breakdown

| Series | Unique IDs | Status |
|---|---:|---|
| linear-algebra | 1 | OK (all citations check out) |
| llm-engineering | 43 | PROBLEMS: 2411.07279 |
| ml-math-derivations | 2 | OK (all citations check out) |
| nlp | 13 | OK (all citations check out) |
| pde-ml | 44 | PROBLEMS: 2104.10013, 2205.10573, 2410.13228 |
| recommendation-systems | 8 | OK (all citations check out) |
| reinforcement-learning | 38 | OK (all citations check out) |
| standalone | 39 | OK (all citations check out) |
| time-series | 3 | OK (all citations check out) |
| transfer-learning | 46 | OK (all citations check out) |

## Methodology notes

- The first heuristic pass flagged 25 'AUTHOR drift' cases where the surname extracted from the citation differed from arXiv's first-author surname. **22 of those were false positives** caused by the regex picking up the second author when the first author was at the start of the line (e.g. 'Hu, E., Shen, Y., ...' regex matched 'Shen' instead of 'Hu'). After manual review only the cases above are genuine drift.
- Several apparent-mismatches were not flagged as errors because both the article wording and arXiv metadata are correct in their own right:
  - `2407.21783` Llama 3 first author was renamed from 'Dubey' to 'Grattafiori' in arXiv revision; both names appear in the paper.
  - `2211.05100` BLOOM lists 'BigScience Workshop' as first author on arXiv but Scao et al is the conventional citation form.
  - `1205.2618` BPR - UAI 2009 paper, arXiv preprint posted 2012.
- arXiv API rate-limited at 3.1s between requests. Total run time ~11 min for 219 IDs.

## Recommendation

The blog has a **very high citation accuracy rate (215/219 unique IDs correctly attributed, 98.2%)**. Three of the four issues are in the `pde-ml` series, suggesting one generation pass produced multiple subtly-wrong PDE-ML citations. Recommended fixes:

1. `pde-ml/01-Physics-Informed-Neural-Networks.md:333` - Replace `arXiv:2104.10013` with `arXiv:2009.04525` (the actual XPINNs paper) OR keep the ID and rewrite the citation as 'Shukla, K., Jagtap, A. D., & Karniadakis, G. E. (2021). *Parallel Physics-Informed Neural Networks via Domain Decomposition*'.
2. `pde-ml/02-Neural-Operator-Theory.md:339` - Replace authors 'Tran, A., Mathews, A., Xie, L., & Ong, C. S.' with 'Fanaskov, V. & Oseledets, I.' for the Spectral Neural Operators paper.
3. `pde-ml/01-Physics-Informed-Neural-Networks.md:336` - Replace 'Z. Liu et al.' with 'Toscano, J. D. et al.' and align the title to '*From PINNs to PIKANs: Recent Advances in Physics-Informed Machine Learning*' (or, if the intent was to cite the original KAN paper, replace the arXiv ID with `2404.19756`).
4. `llm-engineering/04-post-training.md:411` - Update title to '*The Surprising Effectiveness of Test-Time Training for Few-Shot Learning*' (current text says 'for abstract reasoning').