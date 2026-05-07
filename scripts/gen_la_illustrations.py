#!/usr/bin/env python3
"""Generate Wanxiang illustrations for linear-algebra series (18 chapters).
Handles distinct EN/ZH stems. Each chapter gets 2 images.
Idempotent. Inserts into both EN and ZH markdown.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import dashscope
from dashscope import ImageSynthesis

DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
MODEL = "wanx2.1-t2i-plus"
SIZE = "1024*576"
N_IMAGES = 1

CONTENT_ROOT = Path("/root/chenk-hugo/content")
SCRIPTS_DIR = Path("/root/chenk-hugo/scripts")
LOG_FILE = SCRIPTS_DIR / "wanxiang_la_illustrations.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_la_illustrations_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_la_illustrations")
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

SERIES = "linear-algebra"

# (en_stem, zh_stem, hero_prompt, mid_prompt, en_mid_heading, zh_mid_heading)
CHAPTERS = [
    (
        "01-the-essence-of-vectors", "01-向量的本质",
        "abstract editorial illustration of arrow-vectors hovering in a luminous void, varying lengths and directions converging toward an unseen origin, muted indigo and warm parchment palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of an inner product handshake: two arrow-vectors meeting at an angle, a translucent cosine glow between them, dusty rose and slate palette, editorial style, no text, 16:9",
        "3. The Inner Product: Where Geometry Meets Algebra",
        "三、内积：几何与代数在这里握手",
    ),
    (
        "02-linear-combinations-and-vector-spaces", "02-线性组合与向量空间",
        "abstract editorial illustration of a span: two seed vectors stretching and combining to flood a translucent plane with reachable points, muted teal and copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of a basis as the smallest complete toolbox, three glowing primitive arrows from which all of space is constructed, slate and warm brass palette, editorial style, no text, 16:9",
        "4. Basis — The Smallest Complete Toolbox",
        "4. 基——最小而完整的工具箱",
    ),
    (
        "03-matrices-as-linear-transformations", "03-矩阵作为线性变换",
        "abstract editorial illustration of a geometric grid being rotated and sheared by a translucent transformation, gridlines bending coherently, muted indigo and amber palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of matrix multiplication as composition: two transformations stacked, the basis vectors of one landing as columns inside the other, copper and slate palette, editorial style, no text, 16:9",
        "4. Matrix Multiplication = Composition of Transformations",
        "4. 矩阵乘法 = 变换的复合",
    ),
    (
        "04-the-secrets-of-determinants", "04-行列式的秘密",
        "abstract editorial illustration of a parallelogram with shifting area glow, the same shape flexing larger and smaller as a hidden scaling factor changes, muted plum and gold palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of a 3D parallelepiped as a luminous volume scaling factor, its sides glowing with orientation and chirality, dusty teal and copper palette, editorial style, no text, 16:9",
        "3D Determinants: A Volume Scaling Factor",
        "三维行列式：体积的缩放因子",
    ),
    (
        "05-linear-systems-and-column-space", "05-线性方程组与列空间",
        "abstract editorial illustration of a vector b being decomposed into the column space of a matrix, columns rising as luminous pillars spanning a translucent plane, muted slate and warm copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of the null space as a hidden direction in which the matrix crushes everything to zero, a dark axis through a glowing column space, dusty indigo and ash palette, editorial style, no text, 16:9",
        "Null Space: What Gets Crushed",
        "零空间：被压扁的方向",
    ),
    (
        "06-eigenvalues-and-eigenvectors", "06-特征值与特征向量",
        "abstract editorial illustration of vector arrows pointing through a transformation, surviving as scaled stable directions while others bend, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of diagonalization: a tangled matrix unwinding along its eigen-axes into a clean diagonal, copper and slate palette, editorial style, no text, 16:9",
        "Diagonalization: Making a Matrix Trivial",
        "对角化：把矩阵化简到极致",
    ),
    (
        "07-orthogonality-and-projections", "07-正交性与投影",
        "abstract editorial illustration of a vector casting a perpendicular shadow onto a plane, the projection lit from above with translucent right-angle glow, muted teal and parchment palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of Gram-Schmidt as a sequential straightening of tilted arrows into a perfectly orthogonal frame, slate and warm gold palette, editorial style, no text, 16:9",
        "Gram-Schmidt: Manufacturing Orthogonal Bases",
        "Gram-Schmidt：手工制造正交基",
    ),
    (
        "08-symmetric-matrices-and-quadratic-forms", "08-对称矩阵与二次型",
        "abstract editorial illustration of a symmetric matrix as a perfectly mirrored grid with eigen-axes glowing as orthogonal principal directions, muted plum and ivory palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of a positive-definite quadratic form as a glowing convex bowl in 3D, level curves nested as concentric ellipses, copper and slate palette, editorial style, no text, 16:9",
        "Positive Definite Matrices",
        "正定矩阵",
    ),
    (
        "09-singular-value-decomposition", "09-奇异值分解SVD",
        "abstract editorial illustration of a circle morphing into an ellipse via three sequential operations: rotate, stretch along principal axes, rotate again, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of low-rank approximation: a mosaic image being reconstructed from only a few dominant singular components, soft amber and slate palette, editorial style, no text, 16:9",
        "Low-Rank Approximation: the Theorem Behind Compression",
        "五、低秩逼近：压缩背后的定理",
    ),
    (
        "10-matrix-norms-and-condition-numbers", "10-矩阵范数与条件数",
        "abstract editorial illustration of a unit ball being stretched into an ellipsoid, the longest axis glowing as the operator norm, muted teal and warm brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of an ill-conditioned matrix amplifying a tiny input perturbation into a wildly different output, two nearly-parallel beams diverging dramatically, dusty crimson and slate palette, editorial style, no text, 16:9",
        "Ill-Conditioned Matrices: The Nightmare",
        "病态矩阵：数值计算的噩梦",
    ),
    (
        "11-matrix-calculus-and-optimization", "11-矩阵微积分与优化",
        "abstract editorial illustration of a gradient field over a multidimensional landscape, arrows pointing uphill as glowing slope vectors, muted ochre and slate palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of backpropagation as luminous gradient signals flowing backward through a stacked chain of matrix layers, copper and indigo palette, editorial style, no text, 16:9",
        "The Chain Rule and Backpropagation",
        "链式法则与反向传播",
    ),
    (
        "12-sparse-matrices-and-compressed-sensing", "12-稀疏矩阵与压缩感知",
        "abstract editorial illustration of a sparse matrix as a vast mostly-empty grid with only a few luminous nonzero entries lighting up like stars, muted indigo and gold palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of compressed sensing: a faint signal being reconstructed from only a handful of random measurements via L1 minimisation, dusty teal and copper palette, editorial style, no text, 16:9",
        "Compressed Sensing",
        "压缩感知",
    ),
    (
        "13-tensors-and-multilinear-algebra", "13-张量与多线性代数",
        "abstract editorial illustration of a tensor as a stacked 3D lattice of glowing cells, fibers and slices visible through translucent layers, muted plum and slate palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of CP decomposition: a complex tensor unwinding into a sum of simple rank-one outer-product components, copper and parchment palette, editorial style, no text, 16:9",
        "CP Decomposition: Tensor as a Sum of Simple Pieces",
        "CP 分解：把张量拆成简单成分之和",
    ),
    (
        "14-random-matrix-theory", "14-随机矩阵理论",
        "abstract editorial illustration of an eigenvalue spectrum forming a luminous semicircle from a swarm of random matrix entries, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of the Marchenko-Pastur distribution: a bulk of singular values forming a translucent crescent over a horizon line, slate and warm amber palette, editorial style, no text, 16:9",
        "4. The Marchenko-Pastur Law",
        "4. Marchenko-Pastur 律",
    ),
    (
        "15-linear-algebra-in-machine-learning", "15-机器学习中的线性代数",
        "abstract editorial illustration of PCA: a high-dimensional point cloud collapsing onto its principal axes, the dominant direction glowing as the variance arrow, muted teal and brass palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of a kernel SVM: data being lifted into a higher-dimensional feature space where a translucent hyperplane separates classes, dusty rose and slate palette, editorial style, no text, 16:9",
        "4. Support Vector Machines and the Kernel Trick",
        "4. 支持向量机与核技巧",
    ),
    (
        "16-linear-algebra-in-deep-learning", "16-深度学习中的线性代数",
        "abstract editorial illustration of a neural network as a chain of matrix multiplications, glowing weight grids stacked into a luminous pipeline, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of attention as a soft lookup: query, key, value matrices interlocking into a glowing triadic gear, slate and warm amber palette, editorial style, no text, 16:9",
        "4. Attention Is a Soft Lookup -- Done with Three Matmuls",
        "4. 注意力 = 用三次矩阵乘法做软查表",
    ),
    (
        "17-linear-algebra-in-computer-vision", "17-计算机视觉中的线性代数",
        "abstract editorial illustration of a pinhole camera projecting a 3D scene onto a 2D image plane through a luminous projection matrix, muted slate and copper palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of two-view epipolar geometry: two camera centers and an epipolar line stitching corresponding image points across views, dusty teal and warm brass palette, editorial style, no text, 16:9",
        "6 Two Views: Epipolar Geometry",
        "6 两视图：对极几何",
    ),
    (
        "18-frontiers-and-summary", "18-前沿应用与总结",
        "abstract editorial illustration of an 18-node dependency graph of linear algebra topics, glowing edges threading concepts into a unified knowledge web, muted indigo and parchment palette, magazine-cover aesthetic, no text, 16:9",
        "abstract illustration of attention in large language models as matrix multiplication wearing a hat, three Q K V matrices interlocking into a luminous gear, copper and slate palette, editorial style, no text, 16:9",
        "Large language models: attention is matrix multiplication wearing a hat",
        "大模型时代的线性代数：注意力是戴了帽子的矩阵乘法",
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
log = logging.getLogger("wx-la")


def oss_key(lang: str, stem: str, n: int) -> str:
    return f"posts/{lang}/{SERIES}/{stem}/illustration_{n}.jpg"


def oss_url(lang: str, stem: str, n: int) -> str:
    return f"{OSS_PUBLIC_BASE}/{quote(oss_key(lang, stem, n), safe='/')}"


def oss_exists(lang: str, stem: str, n: int) -> bool:
    url = oss_url(lang, stem, n)
    req = Request(url, method="HEAD", headers={"User-Agent": "illu/1.0"})
    try:
        with urlopen(req, timeout=10) as r:
            return r.status == 200
    except HTTPError:
        return False
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
    log.info("[%s] submit", tag)
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
                return None
            url = results[0].url
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=60) as r:
                    return r.read()
            except Exception as e:
                log.error("[%s] download failed: %s", tag, e); return None
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            log.error("[%s] %s", tag, status); return None
    log.error("[%s] poll timeout", tag); return None


def upload_oss(local: Path, lang: str, stem: str, n: int) -> bool:
    target = f"oss://{OSS_BUCKET}/{oss_key(lang, stem, n)}"
    cmd = [OSSUTIL, "cp", "-f", "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
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

    ok_en = upload_oss(local, "en", en_stem, n)
    ok_zh = upload_oss(local, "zh", zh_stem, n)
    if ok_en and ok_zh:
        result.update(ok=True, url_en=oss_url("en", en_stem, n),
                      url_zh=oss_url("zh", zh_stem, n))
    return result


def insert_into_markdown(md_path: Path, hero_url: str, mid_url: str, mid_heading: str, title: str) -> bool:
    if not md_path.exists():
        log.warning("missing md: %s", md_path)
        return False
    text = md_path.read_text(encoding="utf-8")

    has_hero = "illustration_1.jpg" in text
    has_mid = "illustration_2.jpg" in text

    if has_hero and has_mid:
        log.info("[%s] already has both, skipping insert", md_path.name)
        return True

    lines = text.split("\n")
    new_lines: list[str] = []

    hero_line = f"![{title} — visual]({hero_url})"
    mid_line = f"![{title} — visual]({mid_url})"

    inserted_hero = has_hero
    inserted_mid = has_mid

    for line in lines:
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
        log.warning("[%s] no ## heading found for hero", md_path.name)
    if not inserted_mid:
        log.warning("[%s] mid heading '%s' not found", md_path.name, mid_heading)

    new_text = "\n".join(new_lines)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
        log.info("[%s] wrote %d bytes", md_path.name, len(new_text))
    return inserted_hero and inserted_mid


def get_title(md_path: Path) -> str:
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return md_path.stem
    m = re.search(r'^title:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else md_path.stem


def main():
    image_jobs = []  # (en_stem, zh_stem, n, prompt)
    for en_stem, zh_stem, hero_p, mid_p, en_h, zh_h in CHAPTERS:
        image_jobs.append((en_stem, zh_stem, 1, hero_p))
        image_jobs.append((en_stem, zh_stem, 2, mid_p))

    log.info("Total image jobs: %d (across %d chapters)", len(image_jobs), len(CHAPTERS))

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="wx") as ex:
        futs = {ex.submit(process_image, en, zh, n, p): (en, zh, n) for en, zh, n, p in image_jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
                results[(r["en_stem"], r["n"])] = r
            except Exception as e:
                en, zh, n = futs[f]
                log.error("[%s/%d] worker exception: %s", en, n, e)
                results[(en, n)] = {"ok": False, "en_stem": en, "zh_stem": zh, "n": n}

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "series": SERIES,
        "total_jobs": len(image_jobs),
        "succeeded": sum(1 for r in results.values() if r.get("ok")),
        "skipped": sum(1 for r in results.values() if r.get("skipped")),
        "results": [r for r in results.values()],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    insert_summary = {"en_articles": 0, "zh_articles": 0, "failed": []}
    for en_stem, zh_stem, hero_p, mid_p, en_h, zh_h in CHAPTERS:
        r1 = results.get((en_stem, 1), {})
        r2 = results.get((en_stem, 2), {})
        if not (r1.get("ok") and r2.get("ok")):
            insert_summary["failed"].append(en_stem)
            continue
        # EN
        en_md = CONTENT_ROOT / "en" / SERIES / f"{en_stem}.md"
        title = get_title(en_md)
        if insert_into_markdown(en_md,
                                oss_url("en", en_stem, 1),
                                oss_url("en", en_stem, 2),
                                en_h, title):
            insert_summary["en_articles"] += 1
        # ZH
        zh_md = CONTENT_ROOT / "zh" / SERIES / f"{zh_stem}.md"
        title = get_title(zh_md)
        if insert_into_markdown(zh_md,
                                oss_url("zh", zh_stem, 1),
                                oss_url("zh", zh_stem, 2),
                                zh_h, title):
            insert_summary["zh_articles"] += 1

    log.info("Insert summary: %s", insert_summary)
    print(json.dumps({"manifest": str(MANIFEST_FILE), **insert_summary,
                      "succeeded": manifest["succeeded"],
                      "total_jobs": manifest["total_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
