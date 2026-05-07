#!/usr/bin/env python3
"""Generate Wanxiang editorial illustrations for ml-math-derivations chapters.

One image used for both EN and ZH. Hero before first ## heading; mid after a chosen
narrative ## heading. Idempotent against OSS HEAD-check and md insertion markers.
"""
import os
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote

import dashscope
from dashscope import ImageSynthesis

DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"
N_IMAGES = 1

CONTENT_ROOT = Path("/root/chenk-hugo/content")
SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
LOG_FILE = SCRIPTS_DIR / "wanxiang_mlmath.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_mlmath_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_mlmath")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OSSUTIL = "/usr/local/bin/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}"

WORKERS = 3
SUBMIT_MIN_INTERVAL_S = 1.5
MAX_SUBMIT_RETRIES = 6
POLL_TIMEOUT_S = 180

NEG = (
    "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
    "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered, frame, border"
)

dashscope.api_key = DASHSCOPE_API_KEY

SERIES = "ml-math-derivations"

# (en_stem, zh_stem, hero_prompt, mid_prompt, en_mid_heading, zh_mid_heading)
ARTICLES = [
    (
        "01-Introduction-and-Mathematical-Foundations",
        "01-绪论与数学基础",
        "abstract editorial illustration of a learning machine emerging from a sea of finite data points, distilling a universal pattern as a luminous filament, muted indigo and parchment palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of VC dimension as concentric capability rings around a hypothesis class, points being shattered by curved decision surfaces, soft amber and slate palette, editorial style, no text, 16:9",
        "3. VC dimension: complexity for infinite classes",
        "3. VC 维：把复杂度度量推广到无限假设类",
    ),
    (
        "02-Linear-Algebra-and-Matrix-Theory",
        "02-线性代数与矩阵论",
        "abstract editorial illustration of vectors transformed by a luminous matrix — rotation, stretching, projection — onto a translucent subspace, muted teal and brass palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of singular value decomposition as a unit sphere being morphed into an ellipsoid by stretching along orthogonal axes, dusty rose and ivory palette, editorial style, no text, 16:9",
        "4. Singular Value Decomposition -- the universal factorisation",
        "4. 奇异值分解 (SVD)——通用分解",
    ),
    (
        "03-Probability-Theory-and-Statistical-Inference",
        "03-概率论与统计推断",
        "abstract editorial illustration of probability mass flowing across a sample space as luminous rivulets converging on a posterior peak, muted plum and parchment palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of the central limit theorem as many irregular distributions blurring together into a single luminous bell curve, soft teal and warm sand palette, editorial style, no text, 16:9",
        "4. Limit Theorems: Why ML Works at Scale",
        "4. 极限定理：大样本下 ML 为何有效",
    ),
    (
        "04-Convex-Optimization-Theory",
        "04-凸优化理论",
        "abstract editorial illustration of a convex bowl-shaped landscape with a luminous gradient trajectory descending toward the global minimum, muted slate and copper palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of constrained optimization as a feasible polygon glowing inside a level-set landscape, with KKT tangency lines lit from within, dusty indigo and brass palette, editorial style, no text, 16:9",
        "5. Constrained Optimization and Duality",
        "5. 约束优化与对偶理论",
    ),
    (
        "05-Linear-Regression",
        "05-线性回归",
        "abstract editorial illustration of scattered data points snapping into a luminous regression line through a translucent feature plane, soft pastel palette of dusty rose and ivory, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of a projection matrix casting a vector onto a column-space plane, the residual glowing perpendicularly, muted teal and warm sand palette, editorial style, no text, 16:9",
        "The Projection Matrix",
        "投影矩阵",
    ),
    (
        "06-Logistic-Regression-and-Classification",
        "06-逻辑回归与分类",
        "abstract editorial illustration of a smooth sigmoid curve as a glass ribbon separating two colored regions of probability, muted indigo and warm coral palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of softmax as several competing probability streams normalized into a single luminous simplex, dusty teal and brass palette, editorial style, no text, 16:9",
        "4. Multi-Class Extension: Softmax Regression",
        "4. 多分类推广：Softmax 回归",
    ),
    (
        "07-Decision-Trees",
        "07-决策树",
        "abstract editorial illustration of a branching tree of glowing decision nodes with flowing paths splitting feature space into rectangular regions, muted forest green and parchment palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of information entropy as mixed colored particles being separated by a splitting axis into purer clusters, soft amber and slate palette, editorial style, no text, 16:9",
        "Splitting Criteria",
        "分裂准则",
    ),
    (
        "08-Support-Vector-Machines",
        "08-支持向量机",
        "abstract editorial illustration of two color regions separated by a wide margin street, with floating support vector dots glowing on the margin walls, muted slate and copper palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of the kernel trick: points unliftable by a line in 2D rising into a 3D space where a luminous plane cleanly separates them, dusty plum and brass palette, editorial style, no text, 16:9",
        "3. Kernels",
        "3. 核函数",
    ),
    (
        "09-Naive-Bayes",
        "09-朴素贝叶斯",
        "abstract editorial illustration of Bayes rule as a translucent prior cloud being updated by a luminous likelihood beam into a sharper posterior, muted teal and ivory palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of three likelihood variants — gaussian, multinomial, bernoulli — as three distinct glowing distribution shapes feeding into a single decision rule, soft amber and slate palette, editorial style, no text, 16:9",
        "3. Three Variants — Same Bayes Rule, Different Likelihood",
        "3. 三种变体——同一个贝叶斯规则，不同的似然",
    ),
    (
        "10-Semi-Naive-Bayes-and-Bayesian-Networks",
        "10-半朴素贝叶斯与贝叶斯网络",
        "abstract editorial illustration of a directed acyclic graph of glowing variable nodes connected by translucent dependency arrows, muted indigo and parchment palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of a tree-augmented Naive Bayes structure as a central root spawning a single tree of dependent feature nodes, dusty teal and brass palette, editorial style, no text, 16:9",
        "3. TAN: tree-augmented Naive Bayes",
        "3. TAN：树增广朴素贝叶斯",
    ),
    (
        "11-Ensemble-Learning",
        "11-集成学习",
        "abstract editorial illustration of many weak learners — small glowing trees — averaging into one strong, sharply lit collective prediction, muted slate and copper palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of bagging vs boosting: parallel independent trees vs a sequential chain that each focuses on residuals, dusty rose and forest green palette, editorial style, no text, 16:9",
        "3. Bagging vs Boosting --- the bias--variance picture",
        "3. Bagging vs Boosting：偏差-方差视角",
    ),
    (
        "12-XGBoost-and-LightGBM",
        "12-XGBoost与LightGBM",
        "abstract editorial illustration of gradient boosted trees stacked into a luminous additive ensemble, residuals shrinking with each layer, muted teal and brass palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of a histogram-binned feature being split with leaf-wise growth, candidate split points glowing along a discretized axis, dusty plum and ivory palette, editorial style, no text, 16:9",
        "LightGBM: efficient gradient boosting",
        "LightGBM：把工程做到极致",
    ),
    (
        "13-EM-Algorithm-and-GMM",
        "13-EM算法与GMM",
        "abstract editorial illustration of two overlapping gaussian clouds being teased apart by golden assignment lines, muted indigo and warm amber palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of EM as alternating coordinate ascent on a luminous ELBO surface, two arrows tracing E-step and M-step, dusty teal and copper palette, editorial style, no text, 16:9",
        "3. EM as coordinate ascent on the ELBO",
        "3. EM 是 ELBO 上的坐标上升",
    ),
    (
        "14-Variational-Inference-and-Variational-EM",
        "14-变分推断与变分EM",
        "abstract editorial illustration of an intractable posterior being approximated by a tractable family glowing within a manifold of distributions, muted plum and ivory palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of mean-field factorization as a complex joint distribution decomposing into independent glowing marginal slabs, soft slate and brass palette, editorial style, no text, 16:9",
        "3. Mean-Field Approximation",
        "3. 平均场近似",
    ),
    (
        "15-Hidden-Markov-Models",
        "15-隐马尔可夫模型",
        "abstract editorial illustration of hidden states linked by translucent transition arrows along a chain, observation symbols floating above each state, muted teal and warm sand palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of the Viterbi path as a single luminous best route lighting up through a trellis of state-time cells, dusty indigo and copper palette, editorial style, no text, 16:9",
        "4. Viterbi: From Sum to Max",
        "4. Viterbi：把求和换成求最大",
    ),
    (
        "16-Conditional-Random-Fields",
        "16-条件随机场",
        "abstract editorial illustration of a linear-chain CRF as a sequence of conditioned nodes globally normalized by a partition function glowing across the whole chain, muted slate and brass palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of BiLSTM-CRF as bidirectional luminous streams feeding into a CRF decoding layer that smooths label transitions, dusty teal and copper palette, editorial style, no text, 16:9",
        "6. CRF in the Deep Learning Era: BiLSTM-CRF",
        "6. 深度学习时代：BiLSTM-CRF",
    ),
    (
        "17-Dimensionality-Reduction-and-PCA",
        "17-降维与主成分分析",
        "abstract editorial illustration of a high-dimensional cloud being projected onto a low-dimensional plane by a luminous principal axis, muted indigo and parchment palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of t-SNE preserving local neighborhoods as nearby points clustering into glowing islands across a low-dimensional sea, dusty rose and slate palette, editorial style, no text, 16:9",
        "6. t-SNE: Neighbour-Preserving Visualisation",
        "6. t-SNE：用概率分布保留邻域结构",
    ),
    (
        "18-Clustering-Algorithms",
        "18-聚类算法",
        "abstract editorial illustration of unlabeled data points self-organizing into distinct luminous clusters around glowing centroids, muted teal and warm coral palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of DBSCAN density-based clustering as dense regions glowing brighter and being connected through reachable neighbors while sparse outliers fade, dusty slate and amber palette, editorial style, no text, 16:9",
        "4. DBSCAN: Density-Based Clustering",
        "4. DBSCAN：密度驱动的聚类",
    ),
    (
        "19-Neural-Networks-and-Backpropagation",
        "19-神经网络与反向传播",
        "abstract editorial illustration of a multilayer neural network with translucent activation flows forward and luminous gradient signals streaming backward through the layers, muted indigo and brass palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of backpropagation as the chain rule unfurling through nested computation nodes, gradients accumulating along glowing edges, dusty teal and copper palette, editorial style, no text, 16:9",
        "3. Backpropagation: The Chain Rule at Scale",
        "3. 反向传播：把链式法则用到极致",
    ),
    (
        "20-Regularization-and-Model-Selection",
        "20-正则化与模型选择",
        "abstract editorial illustration of a constrained ball-shaped regularization region intersecting a loss landscape, the optimum gently pulled toward the origin, muted slate and parchment palette, Stripe Press / Quanta Magazine aesthetic, no text, 16:9",
        "abstract illustration of L1 vs L2 geometry: a diamond vs a circle constraint touching elliptical loss contours, with sparse axis-aligned solutions glowing where the diamond meets, dusty plum and brass palette, editorial style, no text, 16:9",
        "3. L1 Regularisation (Lasso) and Sparsity",
        "3. L1 正则化（Lasso）与稀疏性",
    ),
]

SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("wx-mlmath")


def oss_key(lang: str, stem: str, n: int) -> str:
    return f"posts/{lang}/{SERIES}/{stem}/illustration_{n}.jpg"


def oss_url(lang: str, stem: str, n: int) -> str:
    return f"{OSS_PUBLIC_BASE}/{oss_key(lang, stem, n)}"


def oss_exists(lang: str, stem: str, n: int) -> bool:
    raw_url = oss_url(lang, stem, n)
    # URL-encode the path portion to support non-ASCII (Chinese) stems
    proto, rest = raw_url.split("://", 1)
    host, path = rest.split("/", 1)
    encoded_url = f"{proto}://{host}/{quote(path, safe='/')}"
    req = Request(encoded_url, method="HEAD", headers={"User-Agent": "illu/1.0"})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status == 200
    except HTTPError as e:
        return e.code != 404 and False
    except URLError:
        return False


submit_lock = Lock()
_last_submit_ts = [0.0]


def _throttled_submit(prompt: str, tag: str):
    backoff = 4.0
    for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
        with submit_lock:
            now = time.time()
            wait = SUBMIT_MIN_INTERVAL_S - (now - _last_submit_ts[0])
            if wait > 0:
                time.sleep(wait)
            _last_submit_ts[0] = time.time()
        try:
            rsp = ImageSynthesis.async_call(
                model=MODEL, prompt=prompt, negative_prompt=NEG,
                n=N_IMAGES, size=SIZE,
            )
        except Exception as e:
            log.error("[%s] async_call exception attempt %d: %s", tag, attempt, e)
            time.sleep(backoff); backoff = min(backoff * 1.7, 30); continue
        if rsp.status_code == 200 and getattr(rsp, "output", None):
            return rsp
        code = getattr(rsp, "code", "") or ""
        msg = getattr(rsp, "message", "") or ""
        if rsp.status_code == 429 or "Throttl" in str(code) or "rate" in str(msg).lower():
            log.warning("[%s] 429 attempt %d, sleeping %.1f", tag, attempt, backoff)
            time.sleep(backoff); backoff = min(backoff * 1.7, 30); continue
        log.error("[%s] async_call failed: status=%s code=%s msg=%s", tag, rsp.status_code, code, msg)
        return None
    return None


def generate_image(prompt: str, tag: str) -> bytes | None:
    log.info("[%s] submit (%d chars)", tag, len(prompt))
    rsp = _throttled_submit(prompt, tag)
    if rsp is None:
        return None
    task_id = rsp.output.task_id
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        try:
            st = ImageSynthesis.fetch(task=task_id)
        except Exception as e:
            log.warning("[%s] fetch exception: %s", tag, e); continue
        status = st.output.task_status
        if status == "SUCCEEDED":
            results = st.output.results or []
            if not results:
                log.error("[%s] SUCCEEDED no results", tag); return None
            url = results[0].url
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=60) as r:
                    return r.read()
            except Exception as e:
                log.error("[%s] download failed: %s", tag, e); return None
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            log.error("[%s] %s: %s", tag, status, st); return None
    log.error("[%s] poll timeout", tag); return None


def upload_oss(local: Path, lang: str, stem: str, n: int) -> bool:
    target = f"oss://{OSS_BUCKET}/{oss_key(lang, stem, n)}"
    cmd = [OSSUTIL, "cp", "-f", \"--cache-control\", \"public, max-age=300, must-revalidate\", "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
           "--region", "cn-beijing", str(local), target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s/%d] ossutil rc=%d %s", stem, n, r.returncode, r.stderr[-200:])
        return False
    return True


def process_image(en_stem: str, zh_stem: str, n: int, prompt: str) -> dict:
    """Generate one image and upload to BOTH en (en_stem) and zh (zh_stem) OSS paths."""
    tag = f"{en_stem}/illu{n}"
    result = {"en_stem": en_stem, "zh_stem": zh_stem, "n": n, "ok": False, "skipped": False,
              "url_en": None, "url_zh": None}

    if oss_exists("en", en_stem, n) and oss_exists("zh", zh_stem, n):
        log.info("[%s] both langs exist, skipping", tag)
        result.update(ok=True, skipped=True,
                      url_en=oss_url("en", en_stem, n),
                      url_zh=oss_url("zh", zh_stem, n))
        return result

    data = generate_image(prompt, tag)
    if data is None:
        return result

    out_dir = TMP_DIR / en_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / f"illustration_{n}.jpg"
    local.write_bytes(data)
    log.info("[%s] saved %d bytes -> %s", tag, len(data), local)

    ok_en = upload_oss(local, "en", en_stem, n)
    ok_zh = upload_oss(local, "zh", zh_stem, n)
    if ok_en and ok_zh:
        result.update(ok=True, url_en=oss_url("en", en_stem, n),
                      url_zh=oss_url("zh", zh_stem, n))
    return result


def insert_into_markdown(md_path: Path, lang: str, stem: str,
                         hero_url: str, mid_url: str, mid_heading: str, title: str) -> bool:
    if not md_path.exists():
        log.warning("missing md: %s", md_path)
        return False
    text = md_path.read_text(encoding="utf-8")

    hero_marker = "illustration_1.jpg"
    mid_marker = "illustration_2.jpg"
    has_hero = hero_marker in text
    has_mid = mid_marker in text

    if has_hero and has_mid:
        log.info("[%s/%s] already has both, skipping insert", lang, stem)
        return True

    lines = text.split("\n")
    new_lines: list[str] = []

    hero_line = f"![{title} — visual]({hero_url})"
    mid_line = f"![{title} — visual]({mid_url})"

    inserted_hero = has_hero
    inserted_mid = has_mid

    for i, line in enumerate(lines):
        if not inserted_hero and line.startswith("## "):
            new_lines.append(hero_line)
            new_lines.append("")
            inserted_hero = True
        new_lines.append(line)
        if not inserted_mid and line.strip() == f"## {mid_heading}":
            new_lines.append("")
            new_lines.append(mid_line)
            inserted_mid = True

    if not inserted_hero:
        log.warning("[%s/%s] no ## heading found for hero", lang, stem)
    if not inserted_mid:
        log.warning("[%s/%s] mid heading '%s' not found", lang, stem, mid_heading)

    new_text = "\n".join(new_lines)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
        log.info("[%s/%s] wrote %d bytes (hero=%s mid=%s)", lang, stem,
                 len(new_text), inserted_hero and not has_hero, inserted_mid and not has_mid)
    return inserted_hero and inserted_mid


def get_title(md_path: Path) -> str:
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return md_path.stem
    m = re.search(r'^title:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else md_path.stem


def main():
    image_jobs = []
    for en_stem, zh_stem, hero_p, mid_p, en_mid, zh_mid in ARTICLES:
        image_jobs.append((en_stem, zh_stem, 1, hero_p))
        image_jobs.append((en_stem, zh_stem, 2, mid_p))

    log.info("Total image jobs: %d (across %d articles)", len(image_jobs), len(ARTICLES))

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="wx") as ex:
        futs = {ex.submit(process_image, en_s, zh_s, n, p): (en_s, n)
                for en_s, zh_s, n, p in image_jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
                results[(r["en_stem"], r["n"])] = r
            except Exception as e:
                en_s, n = futs[f]
                log.error("[%s/%d] worker exception: %s", en_s, n, e)
                results[(en_s, n)] = {"ok": False, "en_stem": en_s, "n": n}

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "series": SERIES,
        "total_jobs": len(image_jobs),
        "succeeded": sum(1 for r in results.values() if r.get("ok")),
        "skipped": sum(1 for r in results.values() if r.get("skipped")),
        "results": list(results.values()),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest written: %s", MANIFEST_FILE)

    insert_summary = {"en_articles": 0, "zh_articles": 0, "failed": []}
    for en_stem, zh_stem, hero_p, mid_p, en_mid, zh_mid in ARTICLES:
        r1 = results.get((en_stem, 1), {})
        r2 = results.get((en_stem, 2), {})
        if not (r1.get("ok") and r2.get("ok")):
            insert_summary["failed"].append(en_stem)
            log.warning("[%s] missing image, skip insert", en_stem)
            continue
        # EN
        en_md = CONTENT_ROOT / "en" / SERIES / f"{en_stem}.md"
        en_title = get_title(en_md)
        if insert_into_markdown(en_md, "en", en_stem,
                                oss_url("en", en_stem, 1),
                                oss_url("en", en_stem, 2),
                                en_mid, en_title):
            insert_summary["en_articles"] += 1
        # ZH
        zh_md = CONTENT_ROOT / "zh" / SERIES / f"{zh_stem}.md"
        zh_title = get_title(zh_md)
        if insert_into_markdown(zh_md, "zh", zh_stem,
                                oss_url("zh", zh_stem, 1),
                                oss_url("zh", zh_stem, 2),
                                zh_mid, zh_title):
            insert_summary["zh_articles"] += 1

    log.info("Insert summary: %s", insert_summary)
    print(json.dumps({"manifest": str(MANIFEST_FILE), **insert_summary,
                      "total_succeeded": manifest["succeeded"],
                      "total_jobs": manifest["total_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
