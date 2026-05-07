#!/usr/bin/env python3
"""Generate Wanxiang illustrations for RL + TL chapters."""
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
LOG_FILE = SCRIPTS_DIR / "wanxiang_rl_tl.log"
MANIFEST_FILE = SCRIPTS_DIR / "wanxiang_rl_tl_manifest.json"
TMP_DIR = Path("/tmp/wanxiang_rl_tl")
TMP_DIR.mkdir(parents=True, exist_ok=True)

OSSUTIL = "/usr/local/bin/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"
OSS_PUBLIC_BASE = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}"

WORKERS = 4
SUBMIT_MIN_INTERVAL_S = 1.5
MAX_SUBMIT_RETRIES = 5
POLL_TIMEOUT_S = 180

NEG = (
    "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, "
    "blurry, distorted, photorealistic faces, stock photo aesthetic, cluttered, frame, border"
)

dashscope.api_key = DASHSCOPE_API_KEY

# Each entry: stem -> [hero_prompt, mid_prompt, en_mid_heading, zh_mid_heading,
#                       en_filename_stem, zh_filename_stem]
# (Filenames differ between EN and ZH directories, so we record both.)

ARTICLES: dict[str, dict[str, list]] = {
    "reinforcement-learning": {
        "01-fundamentals-and-core-concepts": [
            "abstract editorial illustration of an agent-environment loop with state and action arrows pulsing between a small luminous agent and a translucent world, reward symbols glowing along the return path, muted teal and copper palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a Markov decision process as nested luminous rings of states transitioning under actions, with bellman backups flowing inward, muted indigo and brass palette, editorial style, no text, 16:9",
            "Markov Decision Process: The Mathematical Foundation",
            "马尔可夫决策过程：数学基础",
            "01-fundamentals-and-core-concepts",
            "01-基础与核心概念",
        ],
        "02-q-learning-and-dqn": [
            "abstract editorial illustration of a Q-table grid morphing into a deep neural network, cells dissolving into glowing neurons and weighted edges, muted cobalt and warm amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of DQN's two innovations — experience replay buffer as a pool of frozen memories, and a target network as a delayed mirror — slate and copper palette, editorial style, no text, 16:9",
            "DQN's Core Innovations",
            "DQN 的两大创新",
            "02-q-learning-and-dqn",
            "02-Q-Learning与深度Q网络",
        ],
        "03-policy-gradient-and-actor-critic": [
            "abstract editorial illustration of a policy distribution as a soft cloud of probabilities shifting toward higher reward peaks in parameter space, muted plum and gold palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of an actor and a critic as two cooperating networks — one proposing actions, the other evaluating them — connected by a luminous gradient stream, dusty teal and copper palette, editorial style, no text, 16:9",
            "3. Actor-Critic: Replacing Returns with TD Estimates",
            "3. Actor-Critic：用 TD 估计替代回报",
            "03-policy-gradient-and-actor-critic",
            "03-Policy-Gradient与Actor-Critic方法",
        ],
        "04-exploration-and-curiosity-driven-learning": [
            "abstract editorial illustration of an agent venturing into an unknown gridworld lit by a curiosity glow at the frontier of unseen states, muted indigo and warm amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of intrinsic curiosity as a prediction-error halo around novel states, with familiar regions dimming, slate and copper palette, editorial style, no text, 16:9",
            "2. The curiosity blueprint: intrinsic rewards",
            "2. 好奇心的统一框架：内在奖励",
            "04-exploration-and-curiosity-driven-learning",
            "04-探索策略与好奇心驱动学习",
        ],
        "05-model-based-rl-and-world-models": [
            "abstract editorial illustration of a learned world model as a translucent simulation orb floating beside the real environment, the agent rehearsing futures inside it, muted teal and parchment palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of a latent world model dreaming sequences in a compressed inner space, glowing trajectories curling through abstract dimensions, muted plum and brass palette, editorial style, no text, 16:9",
            "5. World Models: Dreaming in a Latent Space",
            "五、World Models：在潜空间里做梦",
            "05-model-based-rl-and-world-models",
            "05-Model-Based强化学习与世界模型",
        ],
        "06-ppo-and-trpo": [
            "abstract editorial illustration of a policy update constrained inside a glowing trust region — a soft luminous bubble preventing destructive jumps in parameter space, muted indigo and copper palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of PPO's clipped objective as a gentle ratio bracket holding policy ratios within bounds, dusty teal and warm brass palette, editorial style, no text, 16:9",
            "PPO: keeping 90% of the benefit at 20% of the complexity",
            "PPO：花两成代价拿走九成收益",
            "06-ppo-and-trpo",
            "06-PPO与TRPO-信任域策略优化",
        ],
        "07-imitation-learning": [
            "abstract editorial illustration of a demonstrator showing a luminous path while an apprentice agent learns to mimic the trajectory, soft amber and slate palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of inverse reinforcement learning as a hidden reward landscape being inferred from observed expert footprints, muted plum and copper palette, editorial style, no text, 16:9",
            "4. Inverse reinforcement learning",
            "4. 逆强化学习（IRL）",
            "07-imitation-learning",
            "07-模仿学习与逆强化学习",
        ],
        "08-alphago-and-mcts": [
            "abstract editorial illustration of a Monte Carlo search tree with branches glowing brighter as rollout values back-propagate up to the root, muted indigo and gold palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of AlphaGo as a fusion of policy and value networks guiding a luminous search tree across an abstract Go board, slate and warm copper palette, editorial style, no text, 16:9",
            "2. AlphaGo (2016): Networks Meet Search",
            "2. AlphaGo（2016）：网络遇上搜索",
            "08-alphago-and-mcts",
            "08-AlphaGo与蒙特卡洛树搜索",
        ],
        "09-multi-agent-rl": [
            "abstract editorial illustration of multiple agents interacting in a shared environment, their action arrows weaving and intersecting in a luminous social fabric, muted teal and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of centralised training with decentralised execution — a global critic illuminated above many local actors operating with partial information, dusty indigo and copper palette, editorial style, no text, 16:9",
            "2. CTDE: train with everything, execute with almost nothing",
            "2. CTDE：训练时全信息，执行时几乎零信息",
            "09-multi-agent-rl",
            "09-多智能体强化学习",
        ],
        "10-offline-reinforcement-learning": [
            "abstract editorial illustration of an agent learning entirely from a frozen replay buffer — a crystalline archive of past trajectories — without ever touching the live environment, muted slate and warm amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of conservative Q-learning as a value function gently pushed downward on out-of-distribution actions, copper and indigo palette, editorial style, no text, 16:9",
            "2. Conservative Q-Learning (CQL)",
            "2. CQL：保守 Q-Learning",
            "10-offline-reinforcement-learning",
            "10-离线强化学习",
        ],
        "11-hierarchical-and-meta-rl": [
            "abstract editorial illustration of a high-level manager network decomposing a goal into subtasks, with low-level worker policies executing primitive actions below, muted plum and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of meta-RL as an outer learning loop that adapts an inner agent across a distribution of tasks, concentric luminous rings, dusty teal and copper palette, editorial style, no text, 16:9",
            "4. Meta-RL: learning to learn",
            "4. 元学习：学会学习",
            "11-hierarchical-and-meta-rl",
            "11-层次化强化学习与元学习",
        ],
        "12-rlhf-and-llm-applications": [
            "abstract editorial illustration of human preferences shaping a reward signal that guides a language model — paired comparisons distilling into a luminous reward landscape, muted rose and slate palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of DPO as a direct preference optimization that bypasses the reward model — chosen and rejected responses pulling the policy through a single gradient, dusty teal and copper palette, editorial style, no text, 16:9",
            "5. DPO: Skipping the Reward Model and the RL",
            "5. DPO：跳过奖励模型与 RL",
            "12-rlhf-and-llm-applications",
            "12-RLHF与大语言模型应用",
        ],
    },
    "transfer-learning": {
        "01-fundamentals-and-core-concepts": [
            "abstract editorial illustration of knowledge transferring from a luminous source domain to a fainter target domain across a glowing bridge of shared features, muted teal and warm copper palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of negative transfer as misaligned distributions colliding and dissipating instead of reinforcing, muted slate and burgundy palette, editorial style, no text, 16:9",
            "5. Negative Transfer",
            "5. 负迁移",
            "01-fundamentals-and-core-concepts",
            "01-基础与核心概念",
        ],
        "02-pre-training-and-fine-tuning": [
            "abstract editorial illustration of a model trained on a vast general corpus — oceans of text condensing into a dense luminous backbone, muted indigo and parchment palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of fine-tuning as a small targeted update on a frozen pre-trained backbone, glowing adaptation arrows flowing into the head, dusty copper and teal palette, editorial style, no text, 16:9",
            "Fine-tuning: why it converges so fast",
            "微调：为什么收敛得这么快",
            "02-pre-training-and-fine-tuning",
            "02-预训练与微调技术",
        ],
        "03-domain-adaptation": [
            "abstract editorial illustration of feature alignment between source and target distributions — two cloud-like distributions being gently pulled toward a shared embedding manifold, muted plum and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of adversarial domain alignment via a gradient reversal layer — a luminous discriminator failing to tell domains apart, dusty teal and copper palette, editorial style, no text, 16:9",
            "3. DANN — Adversarial Alignment in One Backward Pass",
            "3. DANN：一次反向传播完成的对抗对齐",
            "03-domain-adaptation",
            "03-域适应方法",
        ],
        "04-few-shot-learning": [
            "abstract editorial illustration of a small handful of labeled examples training a model to recognise a new class — N tiny luminous prototypes shaping a decision boundary, muted rose and slate palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of metric learning as a learned distance space where same-class samples cluster tightly and different-class samples push apart, dusty indigo and copper palette, editorial style, no text, 16:9",
            "Metric Learning: Classification by Distance",
            "度量学习：用距离来分类",
            "04-few-shot-learning",
            "04-Few-Shot-Learning",
        ],
        "05-knowledge-distillation": [
            "abstract editorial illustration of a large teacher model distilling its knowledge into a small student model — soft probability flows pouring from a wide luminous backbone into a narrower one, muted teal and warm amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of feature-based distillation as intermediate hidden states being aligned between teacher and student, dusty plum and copper palette, editorial style, no text, 16:9",
            "Feature-based distillation",
            "特征蒸馏：让中间层也对齐",
            "05-knowledge-distillation",
            "05-知识蒸馏",
        ],
        "06-multi-task-learning": [
            "abstract editorial illustration of a shared backbone with multiple task-specific heads branching outward like translucent petals, muted indigo and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of gradient conflicts between tasks being resolved by a balancing mechanism — opposing arrows redirected to a constructive resultant, dusty teal and copper palette, editorial style, no text, 16:9",
            "Gradient Conflicts and Task Balancing",
            "梯度冲突与任务平衡",
            "06-multi-task-learning",
            "06-多任务学习",
        ],
        "07-zero-shot-learning": [
            "abstract editorial illustration of a model generalising to unseen tasks via natural-language instruction — semantic descriptors lighting up class regions never directly trained on, muted plum and gold palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of CLIP as a dual-tower model bridging vision and language into a shared embedding sphere, dusty teal and warm copper palette, editorial style, no text, 16:9",
            "6. CLIP and the vision-language pretraining era",
            "6. CLIP 与视觉-语言预训练时代",
            "07-zero-shot-learning",
            "07-零样本学习",
        ],
        "08-multimodal-transfer": [
            "abstract editorial illustration of vision and language streams fusing into a single multimodal representation — a glowing image and a glowing sentence converging into a shared embedding, muted indigo and amber palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of contrastive learning as paired image-text samples pulling together while mismatched pairs push apart in a luminous embedding space, dusty rose and slate palette, editorial style, no text, 16:9",
            "2. Contrastive learning: the math behind alignment",
            "二、对比学习：对齐背后的数学",
            "08-multimodal-transfer",
            "08-多模态迁移",
        ],
        "09-parameter-efficient-fine-tuning": [
            "abstract editorial illustration of small LoRA adapters as luminous low-rank inserts inside a frozen transformer backbone — a few bright threads weaving through dimmed pre-trained weights, muted teal and brass palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of LoRA's low-rank decomposition as two narrow matrices factoring a wide update, glowing rank-r bottleneck, dusty copper and indigo palette, editorial style, no text, 16:9",
            "LoRA: Low-Rank Adaptation",
            "LoRA：低秩适配",
            "09-parameter-efficient-fine-tuning",
            "09-参数高效微调",
        ],
        "10-continual-learning": [
            "abstract editorial illustration of a model learning new tasks while preserving old ones — a luminous timeline of knowledge layers stacking without erasing previous ones, muted plum and parchment palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of catastrophic forgetting as faded older task representations being protected by elastic weight regularisation anchors, dusty teal and copper palette, editorial style, no text, 16:9",
            "Why Forgetting Happens",
            "遗忘是怎么发生的",
            "10-continual-learning",
            "10-持续学习",
        ],
        "11-cross-lingual-transfer": [
            "abstract editorial illustration of cross-lingual transfer as multilingual word embeddings aligning into a shared semantic space — luminous threads from different scripts meeting at common meaning, muted indigo and gold palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of multilingual pretraining as a unified backbone consuming many languages and emitting language-agnostic representations, dusty teal and copper palette, editorial style, no text, 16:9",
            "Multilingual Pretraining",
            "多语言预训练",
            "11-cross-lingual-transfer",
            "11-跨语言迁移",
        ],
        "12-industrial-applications-and-best-practices": [
            "abstract editorial illustration of a production transfer-learning pipeline as a luminous factory of pre-trained backbones being adapted, evaluated, and deployed into downstream services, muted slate and warm copper palette, magazine-cover aesthetic, no text, 16:9",
            "abstract illustration of monitoring distribution shift as drift indicators glowing on a production dashboard, with retraining loops gated by alerts, dusty teal and amber palette, editorial style, no text, 16:9",
            "6. Monitoring distribution shift",
            "6. 监控分布漂移",
            "12-industrial-applications-and-best-practices",
            "12-工业应用与最佳实践",
        ],
    },
}

SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wx-rl-tl")


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
    except HTTPError as e:
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
    cmd = [OSSUTIL, "cp", "-f", "--meta", "Cache-Control:public, max-age=300, must-revalidate", "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
           "--region", "cn-beijing", str(local), target]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        log.error("[%s/%s/%d] ossutil rc=%d %s", series, stem, n, r.returncode, r.stderr[-200:])
        return False
    return True


def process_image(series, stem, n, prompt):
    tag = f"{series}/{stem}/illu{n}"
    result = {"series": series, "stem": stem, "n": n, "ok": False, "skipped": False, "url_en": None, "url_zh": None}
    if oss_exists("en", series, stem, n) and oss_exists("zh", series, stem, n):
        log.info("[%s] both langs exist, skipping", tag)
        result.update(ok=True, skipped=True, url_en=oss_url("en", series, stem, n), url_zh=oss_url("zh", series, stem, n))
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
        result.update(ok=True, url_en=oss_url("en", series, stem, n), url_zh=oss_url("zh", series, stem, n))
    return result


def insert_into_markdown(md_path, lang, series, stem, hero_url, mid_url, mid_heading, title):
    if not md_path.exists():
        log.warning("missing md: %s", md_path); return False
    text = md_path.read_text(encoding="utf-8")
    if "illustration_1.jpg" in text and "illustration_2.jpg" in text:
        log.info("[%s/%s/%s] already has both, skipping insert", lang, series, stem); return True
    has_hero = "illustration_1.jpg" in text
    has_mid = "illustration_2.jpg" in text
    lines = text.split("\n")
    new_lines = []
    hero_line = f"![{title} — visual]({hero_url})"
    mid_line = f"![{title} — visual]({mid_url})"
    inserted_hero = has_hero
    inserted_mid = has_mid
    for line in lines:
        if not inserted_hero and line.startswith("## "):
            new_lines.append(hero_line); new_lines.append(""); inserted_hero = True
        new_lines.append(line)
        if not inserted_mid and line.strip() == f"## {mid_heading}":
            new_lines.append(""); new_lines.append(mid_line); inserted_mid = True
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
        for stem, spec in articles.items():
            hero_prompt, mid_prompt, en_mid, zh_mid, en_stem, zh_stem = spec
            image_jobs.append((series, stem, 1, hero_prompt))
            image_jobs.append((series, stem, 2, mid_prompt))
            article_meta.append((series, stem, hero_prompt, mid_prompt, en_mid, zh_mid, en_stem, zh_stem))

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
    for series, stem, hero_prompt, mid_prompt, en_mid, zh_mid, en_stem, zh_stem in article_meta:
        r1 = results.get((series, stem, 1), {})
        r2 = results.get((series, stem, 2), {})
        if not (r1.get("ok") and r2.get("ok")):
            insert_summary["failed"].append(f"{series}/{stem}")
            log.warning("[%s/%s] missing image, skip insert", series, stem); continue
        for lang, mid_heading, fname_stem in [("en", en_mid, en_stem), ("zh", zh_mid, zh_stem)]:
            md_path = CONTENT_ROOT / lang / series / f"{fname_stem}.md"
            title = get_title(md_path)
            hero_url = oss_url(lang, series, stem, 1)
            mid_url = oss_url(lang, series, stem, 2)
            ok = insert_into_markdown(md_path, lang, series, stem, hero_url, mid_url, mid_heading, title)
            if ok:
                insert_summary[f"{lang}_articles"] += 1

    log.info("Insert summary: %s", insert_summary)
    print(json.dumps({"manifest": str(MANIFEST_FILE), **insert_summary,
                      "total_succeeded": manifest["succeeded"],
                      "total_jobs": manifest["total_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
