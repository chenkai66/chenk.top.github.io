#!/usr/bin/env python3
"""Generate Wanxiang illustrations for ode + pde-ml series.

Per-article: hero (before first ## heading) + mid (after a chosen heading).
Same image for EN and ZH. Idempotent. Different stem per language supported.
"""
import os
from __future__ import annotations
import json, logging, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
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
LOG_FILE = SCRIPTS_DIR / "wanxiang_ode_pde.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_ode_pde_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_ode_pde")
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
POLL_TIMEOUT_S = 240

NEG = ("text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
       "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered, frame, border")

dashscope.api_key = DASHSCOPE_API_KEY

# Each entry: (en_stem, zh_stem, hero_prompt, mid_prompt, en_mid_heading, zh_mid_heading)
# OSS path uses en_stem (canonical) so EN and ZH share image but different markdown locations.
ARTICLES = {
    "ode": [
        ("01-origins-and-intuition", "01-微分方程的起源与直觉",
         "abstract editorial illustration of a coffee cup cooling, with translucent isothermal curves spiraling outward into a soft phase space, muted ivory and warm brown palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a direction field as a gentle vector grid revealing hidden integral curves before any equation is solved, dusty teal and parchment palette, editorial style, no text, 16:9",
         "2. Direction fields: see the solution before you compute it",
         "2. 方向场：解方程之前就先「看见」解"),
        ("02-first-order-methods", "02-一阶微分方程",
         "abstract editorial illustration of an integrating factor multiplying both sides of an equation into perfect harmony, threads weaving into a tidy spiral, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of brine flowing into and out of a translucent tank, salt concentration evolving as a glowing curve, soft teal and copper palette, editorial style, no text, 16:9",
         "5. A worked application: salt in a tank",
         "5. 综合应用：水箱里的盐"),
        ("03-linear-theory", "03-高阶线性微分方程",
         "abstract editorial illustration of a damped oscillator in three regimes — under, critical, over — as three braided trajectories settling toward equilibrium, muted plum and slate palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of three damping regimes drawn as overlapping decaying sinusoids in graphite, copper, and rose, editorial style, no text, 16:9",
         "4. Damped oscillation: the trichotomy in pictures",
         "4. 阻尼振动：用图把三分法看清"),
        ("04-constant-coefficients", "04-拉普拉斯变换",
         "abstract editorial illustration of a time-domain wave morphing through a glowing transform into frequency-domain spectrum bars, muted violet and amber palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a transfer function's poles and zeros mapped onto a complex plane with a glowing stability boundary, slate and copper palette, editorial style, no text, 16:9",
         "6. Transfer functions and the geometry of stability",
         "6. 传递函数与稳定性的几何"),
        ("05-laplace-transform", "05-级数解法与特殊函数",
         "abstract editorial illustration of an unknown solution unfurling as a luminous power series, term by term, around an ordinary point, muted teal and ivory palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of Bessel functions as concentric ripples on a circular drumhead, soft amber and indigo palette, editorial style, no text, 16:9",
         "4. Bessel Functions",
         "4. Bessel 函数"),
        ("06-power-series", "06-线性微分方程组",
         "abstract editorial illustration of two coupled equations merging into one luminous vector equation, threads twisting into a single braid, muted cobalt and brass palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a 2D phase portrait showing nodes, spirals, and saddles as glowing trajectories, dusty teal and copper palette, editorial style, no text, 16:9",
         "4. Phase Portraits in 2D",
         "4. 二维相图"),
        ("07-systems-and-phase-plane", "07-稳定性理论",
         "abstract editorial illustration of perturbation arrows converging on a glowing fixed point at the center of a soft gradient phase space, muted slate and warm brass palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a Lyapunov function as a luminous bowl with trajectories sliding down toward the equilibrium at its base, dusty plum and copper palette, editorial style, no text, 16:9",
         "Lyapunov's Direct Method",
         "Lyapunov 直接方法"),
        ("08-nonlinear-stability", "08-非线性系统与相图",
         "abstract editorial illustration of a nonlinear phase portrait with nullclines crossing and trajectories curving through a soft gradient field, muted teal and copper palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of Lotka-Volterra predator-prey dynamics as two oscillating curves locked in a luminous closed orbit, dusty olive and rose palette, editorial style, no text, 16:9",
         "Lotka-Volterra Predator-Prey",
         "Lotka-Volterra 捕食模型"),
        ("09-bifurcation-chaos", "09-混沌理论与洛伦兹系统",
         "abstract editorial illustration of a Lorenz butterfly attractor in muted indigo and copper, two glowing wings woven from infinite trajectories, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of the butterfly effect: two nearly identical trajectories diverging exponentially across a soft phase space, slate and warm amber palette, editorial style, no text, 16:9",
         "The Butterfly Effect, Visualised",
         "蝴蝶效应可视化"),
        ("10-bifurcation-theory", "10-分岔理论",
         "abstract editorial illustration of a bifurcation diagram branching gracefully as a parameter sweeps across a critical value, muted plum and gold palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a Hopf bifurcation: a stable focus blooming outward into a luminous limit cycle, dusty teal and copper palette, editorial style, no text, 16:9",
         "3. The Hopf bifurcation: a focus gives birth to a cycle",
         "3. Hopf 分岔：焦点孕育出环"),
        ("11-numerical-methods", "11-数值方法",
         "abstract editorial illustration of forward Euler steps marching along a smooth flow field, accumulating subtle drift, muted indigo and parchment palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of stiffness as an explicit method oscillating wildly while an implicit one glides smoothly along the same trajectory, slate and copper palette, editorial style, no text, 16:9",
         "6. Stiffness: the failure mode of explicit methods",
         "6. 刚性问题：显式方法的失败模式"),
        ("12-boundary-value-problems", "12-边值问题",
         "abstract editorial illustration of a shooting method as multiple trajectories launched from one boundary, fanning out toward a target boundary point, muted teal and ochre palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a finite-difference grid discretizing a continuous domain into a luminous lattice of nodes, dusty cobalt and parchment palette, editorial style, no text, 16:9",
         "3. The finite-difference method",
         "3. 有限差分法"),
        ("13-pde-introduction", "13-偏微分方程引论",
         "abstract editorial illustration of three classical PDEs — heat, wave, Laplace — visualized as three distinct flow patterns blooming from a common origin, muted teal, plum, and ivory palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a wave equation: a propagating ripple traveling outward across a smooth surface, soft indigo and copper palette, editorial style, no text, 16:9",
         "The Wave Equation",
         "波动方程"),
        ("14-epidemiology", "14-传染病模型与流行病学",
         "abstract editorial illustration of an SIR model as three populations flowing between glowing compartments, muted rose, slate, and ivory palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of herd immunity as a translucent shield emerging from vaccinated nodes blocking disease spread through a population network, soft teal and amber palette, editorial style, no text, 16:9",
         "Vaccination and Herd Immunity",
         "疫苗接种与群体免疫"),
        ("15-population-dynamics", "15-种群动力学",
         "abstract editorial illustration of predator-prey oscillation as two interlocking curves rising and falling in counter-rhythm, muted forest green and warm copper palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of Fisher-KPP wave fronts spreading across a continuous landscape, soft teal and ochre palette, editorial style, no text, 16:9",
         "Spatial Spread: Fisher-KPP",
         "空间扩散：Fisher-KPP"),
        ("16-control-theory", "16-控制理论基础",
         "abstract editorial illustration of a closed-loop control system as a luminous feedback ring connecting plant, sensor, and controller, muted slate and copper palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a Bode plot as two stacked frequency-response curves with stability margin highlighted, dusty indigo and amber palette, editorial style, no text, 16:9",
         "5. Bode Plots and Stability Margins",
         "5. Bode 图与稳定裕度"),
        ("17-physics-engineering-applications", "17-物理与工程应用",
         "abstract editorial illustration of a nonlinear pendulum tracing a phase-plane separatrix between libration and rotation, muted plum and brass palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of an RLC circuit as a glowing loop with capacitor, inductor, and resistor producing a damped oscillation, slate and copper palette, editorial style, no text, 16:9",
         "2. RLC Circuits -- the same equation, in copper",
         "2. RLC 电路 ── 同一个方程，换成铜线"),
        ("18-advanced-topics-summary", "18-前沿专题与总结",
         "abstract editorial illustration of a continuous flowing trajectory passing through translucent neural network layers, depth becoming time, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a stochastic differential equation as a noisy trajectory diffusing within a guiding drift field, soft teal and parchment palette, editorial style, no text, 16:9",
         "4. Stochastic Differential Equations -- when noise has agency",
         "4. 随机微分方程 ── 当噪声成为主角"),
    ],
    "pde-ml": [
        ("01-Physics-Informed-Neural-Networks", "01-物理信息神经网络",
         "abstract editorial illustration of physics equations woven through translucent neural network layers, partial differential operators glowing as they flow into a loss landscape, muted teal and copper palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of PINN training pathologies — gradient imbalance and stiff loss landscapes — as turbulent fields around a struggling network, slate and warm rust palette, editorial style, no text, 16:9",
         "4 Training pathologies: the part that's actually hard",
         "4 训练病理：PINN 真正难的地方"),
        ("02-Neural-Operator-Theory", "02-神经算子理论",
         "abstract editorial illustration of a neural operator as a luminous mapping between two function spaces, threads connecting input fields to output fields, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a Fourier Neural Operator: a function transformed into spectral modes, modulated, and transformed back, dusty plum and copper palette, editorial style, no text, 16:9",
         "5. The Fourier Neural Operator",
         "5. Fourier Neural Operator"),
        ("03-Variational-Principles", "03-变分原理与优化",
         "abstract editorial illustration of a gradient flow descending through a Wasserstein landscape of probability densities, muted teal and ivory palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of mean-field neural network training as infinitely many neurons flowing through a continuous parameter manifold, soft indigo and copper palette, editorial style, no text, 16:9",
         "3. Mean-Field Theory of Neural-Network Training",
         "3. 神经网络训练的 Mean-Field 理论"),
        ("04-Variational-Inference", "04-变分推断与Fokker-Planck方程",
         "abstract editorial illustration of a particle cloud diffusing under an invisible drift force, evolving according to a Fokker-Planck equation, muted plum and parchment palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of Langevin dynamics as a noisy trajectory descending an energy landscape, soft teal and warm copper palette, editorial style, no text, 16:9",
         "3. Langevin Dynamics: Sampling as a PDE",
         "3. Langevin 动力学：把采样变成 PDE"),
        ("05-Symplectic-Geometry", "05-辛几何与保结构网络",
         "abstract editorial illustration of phase space as a symplectic manifold with trajectories preserving area as they flow, muted indigo and gold palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a Hamiltonian neural network learning energy-conserving dynamics from data, slate and copper palette, editorial style, no text, 16:9",
         "4. Hamiltonian Neural Networks (HNN)",
         "4. 哈密顿神经网络（HNN）"),
        ("06-Continuous-Normalizing-Flows", "06-连续归一化流与Neural-ODE",
         "abstract editorial illustration of a continuous flowing trajectory passing through translucent neural ODE layers, depth becoming time, muted teal and brass palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of optimal transport flow matching: a source density morphing smoothly into a target density along straight-line trajectories, dusty rose and indigo palette, editorial style, no text, 16:9",
         "4. Optimal Transport and Flow Matching",
         "4. 最优传输与 Flow Matching"),
        ("07-Diffusion-Models", "07-扩散模型与Score-Matching",
         "abstract editorial illustration of a particle cloud diffusing into noise then reverse-flowing back into structured form, muted indigo and amber palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a score field as a vector field pointing toward higher data density, guiding samples back from noise, soft slate and copper palette, editorial style, no text, 16:9",
         "3. Score-Based Generative Models",
         "3. 基于 Score 的生成模型"),
        ("08-Reaction-Diffusion-Systems", "08-反应扩散系统与GNN",
         "abstract editorial illustration of Turing patterns emerging from a uniform field as reaction and diffusion compete, soft jade and rose palette, magazine-cover aesthetic, no text, 16:9",
         "abstract illustration of a graph neural network as nodes diffusing information across edges, mirroring a heat equation on graphs, dusty teal and copper palette, editorial style, no text, 16:9",
         "4. GCN Is the Heat Equation",
         "4. GCN 就是热方程"),
    ],
}

SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wx-ode-pde")


def oss_key(lang, series, stem, n):
    return f"posts/{lang}/{series}/{stem}/illustration_{n}.jpg"


def oss_url(lang, series, stem, n):
    return f"{OSS_PUBLIC_BASE}/{oss_key(lang, series, stem, n)}"


def oss_exists(lang, series, stem, n):
    url = oss_url(lang, series, stem, n)
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


def _throttled_submit(prompt, tag):
    backoff = 4.0
    for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
        with submit_lock:
            now = time.time()
            wait = SUBMIT_MIN_INTERVAL_S - (now - _last_submit_ts[0])
            if wait > 0:
                time.sleep(wait)
            _last_submit_ts[0] = time.time()
        try:
            rsp = ImageSynthesis.async_call(model=MODEL, prompt=prompt, negative_prompt=NEG, n=N_IMAGES, size=SIZE)
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


def generate_image(prompt, tag):
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


def upload_oss(local, lang, series, stem, n):
    target = f"oss://{OSS_BUCKET}/{oss_key(lang, series, stem, n)}"
    cmd = [OSSUTIL, "cp", "-f", \"--cache-control\", \"public, max-age=300, must-revalidate\", "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
           "--region", "cn-beijing", str(local), target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s/%s/%d] ossutil rc=%d %s", series, stem, n, r.returncode, r.stderr[-200:])
        return False
    return True


def process_image(series, stem, n, prompt):
    """Generate one image and upload to BOTH en and zh OSS paths under en_stem (canonical)."""
    tag = f"{series}/{stem}/illu{n}"
    result = {"series": series, "stem": stem, "n": n, "ok": False, "skipped": False, "url": None}

    if oss_exists("en", series, stem, n) and oss_exists("zh", series, stem, n):
        log.info("[%s] both langs exist, skipping", tag)
        result.update(ok=True, skipped=True, url=oss_url("en", series, stem, n))
        return result

    data = generate_image(prompt, tag)
    if data is None:
        return result

    out_dir = TMP_DIR / series / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / f"illustration_{n}.jpg"
    local.write_bytes(data)
    log.info("[%s] saved %d bytes -> %s", tag, len(data), local)

    ok_en = upload_oss(local, "en", series, stem, n)
    ok_zh = upload_oss(local, "zh", series, stem, n)
    if ok_en and ok_zh:
        result.update(ok=True, url=oss_url("en", series, stem, n))
    return result


def insert_into_markdown(md_path, lang, series, stem, hero_url, mid_url, mid_heading, title):
    if not md_path.exists():
        log.warning("missing md: %s", md_path)
        return False
    text = md_path.read_text(encoding="utf-8")

    has_hero = "illustration_1.jpg" in text
    has_mid = "illustration_2.jpg" in text

    if has_hero and has_mid:
        log.info("[%s/%s/%s] already has both, skip insert", lang, series, stem)
        return True

    lines = text.split("\n")
    new_lines = []
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
        log.warning("[%s/%s/%s] no ## heading found for hero", lang, series, stem)
    if not inserted_mid:
        log.warning("[%s/%s/%s] mid heading '%s' not found", lang, series, stem, mid_heading)

    new_text = "\n".join(new_lines)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf-8")
        log.info("[%s/%s/%s] wrote %d bytes", lang, series, stem, len(new_text))
    return inserted_hero and inserted_mid


def get_title(md_path):
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return md_path.stem
    m = re.search(r'^title:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else md_path.stem


def main():
    image_jobs = []
    article_meta = []
    for series, articles in ARTICLES.items():
        for en_stem, zh_stem, hero_p, mid_p, en_mid, zh_mid in articles:
            image_jobs.append((series, en_stem, 1, hero_p))
            image_jobs.append((series, en_stem, 2, mid_p))
            article_meta.append((series, en_stem, zh_stem, en_mid, zh_mid))

    log.info("Total image jobs: %d (across %d articles)", len(image_jobs), len(article_meta))

    results = {}
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="wx") as ex:
        futs = {ex.submit(process_image, s, st, n, p): (s, st, n) for s, st, n, p in image_jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
                results[(r["series"], r["stem"], r["n"])] = r
            except Exception as e:
                s, st, n = futs[f]
                log.error("[%s/%s/%d] worker exception: %s", s, st, n, e)
                results[(s, st, n)] = {"ok": False, "series": s, "stem": st, "n": n}

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_jobs": len(image_jobs),
        "succeeded": sum(1 for r in results.values() if r.get("ok")),
        "skipped": sum(1 for r in results.values() if r.get("skipped")),
        "results": list(results.values()),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifest written: %s", MANIFEST_FILE)

    insert_summary = {"en_articles": 0, "zh_articles": 0, "failed": []}
    for series, en_stem, zh_stem, en_mid, zh_mid in article_meta:
        r1 = results.get((series, en_stem, 1), {})
        r2 = results.get((series, en_stem, 2), {})
        if not (r1.get("ok") and r2.get("ok")):
            insert_summary["failed"].append(f"{series}/{en_stem}")
            log.warning("[%s/%s] missing image, skip insert", series, en_stem)
            continue
        for lang, stem_lang, mid_h in [("en", en_stem, en_mid), ("zh", zh_stem, zh_mid)]:
            md_path = CONTENT_ROOT / lang / series / f"{stem_lang}.md"
            title = get_title(md_path)
            hero_url = oss_url(lang, series, en_stem, 1)
            mid_url = oss_url(lang, series, en_stem, 2)
            ok = insert_into_markdown(md_path, lang, series, en_stem, hero_url, mid_url, mid_h, title)
            if ok:
                insert_summary[f"{lang}_articles"] += 1

    log.info("Insert summary: %s", insert_summary)
    print(json.dumps({"manifest": str(MANIFEST_FILE), **insert_summary,
                      "total_succeeded": manifest["succeeded"],
                      "total_jobs": manifest["total_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
