---
title: "NLP (11): Multimodal Large Language Models"
date: 2024-06-11 09:00:00
tags:
  - NLP
  - Multimodal
  - LLM
  - CLIP
  - BLIP-2
  - LLaVA
  - Whisper
  - Vision-Language
categories: Natural Language Processing
series: NLP
part: 11
total_parts: 12
lang: en
mathjax: true
description: "A deep dive into multimodal LLMs: contrastive vision-language pre-training with CLIP, parameter-efficient bridging with BLIP-2's Q-Former, visual instruction tuning with LLaVA, robust speech recognition with Whisper, GPT-4V capabilities, and the MMBench/MME/MMMU benchmark landscape — with reproducible code."
---
Humans never perceive the world in one channel at a time. We watch a chart while reading the caption, hear a tone of voice while reading a face, glance at a screenshot while debating a bug. Pure-text language models are deaf and blind to all of that. **Multimodal Large Language Models (MLLMs)** close the gap by aligning images, audio, and video into the same representation space the language model already speaks.

The story of the last four years can be told in three steps. **CLIP** (2021) showed that 400 M noisy image–text pairs from the web are enough to learn a single embedding space where text and image search each other by cosine similarity. **BLIP-2** (2023) made the alignment cheap: freeze a powerful vision encoder, freeze a powerful LLM, and learn a tiny ~188 M-parameter "Q-Former" between them. **LLaVA / GPT-4V** (2023–2024) replaced the bridge with a one-line projector and **visual instruction tuning** — feed the LLM image embeddings as if they were word embeddings, then fine-tune on GPT-4-generated visual conversations. The result is a single model that can read a chart, debug a screenshot, and describe a photograph with the fluency of a chat assistant.

This article walks through that arc end-to-end: the math behind contrastive alignment, the architectures that scale, the audio side of the story (Whisper), what GPT-4V can and cannot do, and how to read the MMBench / MME / MMMU benchmark numbers — all with working code.

## What You Will Learn

- **CLIP** — InfoNCE contrastive loss, dual encoders, zero-shot classification, and how the temperature $\tau$ shapes training
- **BLIP-2** — why Q-Former works, its 32 learnable queries, and the two-stage pre-training recipe
- **LLaVA** — visual instruction tuning, why a 4 M-parameter linear projector can rival heavier connectors, and the GPT-4-distilled instruction data trick
- **Whisper** — multilingual encoder–decoder ASR over log-mel spectrograms, special-token control, and model-size trade-offs
- **GPT-4V and frontier MLLMs** — capability matrix, cross-domain reasoning, and known failure modes (counting, fine-grained grounding, hallucinations)
- **Evaluation** — MMBench, MME, MMMU, SEED-Bench, POPE — what each measures and how to compare scores honestly
- **Practice** — building a CLIP-based multimodal retrieval index, image captioning, VQA, and Whisper transcription pipelines

## Prerequisites

- Transformer architecture (see [Part 4](/en/nlp-attention-transformer/))
- Pre-training and instruction fine-tuning (see [Parts 5](/en/nlp-bert-pretrained-models/) and [8](/en/nlp-fine-tuning-peft/))
- Comfort with PyTorch and the Hugging Face ecosystem; some computer-vision intuition (patches, ViTs) is useful but not required

---

## Vision–Language Models

### CLIP: Contrastive Vision–Language Alignment

![CLIP dual encoder and in-batch contrastive matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig1_clip_architecture.png)

CLIP (Radford et al., OpenAI 2021) trains two encoders side by side. An **image encoder** $f_I$ — a Vision Transformer (ViT-L/14) or a ResNet — turns an image $x$ into a vector $I = f_I(x) \in \mathbb{R}^{512}$. A **text encoder** $f_T$ — a 12-layer Transformer — turns a caption $t$ into the same-dimensional vector $T = f_T(t)$. Both are L2-normalised so that $I \cdot T$ is a cosine similarity. Training maximises this similarity for matching pairs and pushes it down for everything else.

Concretely, with $N$ image–text pairs in a batch and similarity matrix $s_{ij} = (I_i \cdot T_j)/\tau$, CLIP minimises a symmetric InfoNCE loss:

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\Bigg[\log\frac{e^{s_{ii}}}{\sum_{j} e^{s_{ij}}} + \log\frac{e^{s_{ii}}}{\sum_{j} e^{s_{ji}}}\Bigg].$$

The two terms are the image-to-text and text-to-image classification losses — softmax over the $N$ candidates in the batch, with the diagonal as the correct class. The temperature $\tau \approx 0.07$ is a *learnable* scalar; it is initialised to $\log(1/\tau) = \log 100$ and clipped to prevent collapse. Larger batches give a harder negative pool: CLIP's 400 M-pair training used batches of 32 768 across hundreds of GPUs.

Three properties fall out of this objective:

1. **Zero-shot classification** — to classify an image into one of $K$ labels, encode each label as `"a photo of a {label}"` and pick the text with highest cosine similarity. No labelled training examples needed.
2. **Cross-modal retrieval** — image-to-text and text-to-image search reduce to a dot-product index (FAISS, ScaNN, etc.).
3. **Conditioning generative models** — CLIP's image embeddings are the supervisory signal in DALL·E 2, Stable Diffusion's text encoder, and many image-to-image systems.

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("cat.jpg")
labels = ["a photo of a cat", "a photo of a dog",
          "a photo of a car", "a photo of a bird"]

with torch.no_grad():
    inputs = processor(text=labels, images=image,
                       return_tensors="pt", padding=True)
    out = model(**inputs)

# logits_per_image is already scaled by 1/τ
probs = out.logits_per_image.softmax(dim=-1)[0]
for label, p in zip(labels, probs):
    print(f"{p.item():>6.2%}  {label}")
```

A typical run on a clean cat photo yields >95% on `"a photo of a cat"` — without any training on `"cat"` examples.

**When CLIP fails.** CLIP is bag-of-words-ish: it confuses *"a red cube on a blue sphere"* with *"a blue cube on a red sphere"*. It struggles with counting (two vs. three), fine-grained categories (Yorkshire vs. Norwich terrier), and text inside images (use a model with explicit OCR). For Chinese, use Chinese-CLIP or Qwen-VL — the original WIT corpus is overwhelmingly English.

### BLIP-2: Bridging Frozen Vision and Frozen Language

CLIP aligns; it does not generate text. **BLIP-2** (Li et al., ICML 2023) addresses generation while keeping pre-training cheap. Both the image encoder (ViT-g/14, 1 B parameters) and the LLM (OPT-2.7B / 6.7B or FlanT5) stay **frozen**. Only a small bridge — the **Q-Former** — is trained.

![BLIP-2 architecture with Q-Former bridging frozen ViT and frozen LLM](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig2_blip2_qformer.png)

The Q-Former is a 12-layer Transformer with two crucial twists:

- **32 learnable query embeddings** that act as the input — they have nothing to do with image tokens, they are just trainable vectors.
- A **cross-attention layer** in every block where the queries attend over the patch features coming out of the frozen ViT. The queries thus *pull* the most language-relevant visual information out of the frozen image features.

After 12 layers the 32 queries become 32 "soft visual tokens" that are linearly projected to the LLM's embedding dimension and prepended to the text. To the LLM they look exactly like ordinary word embeddings.

BLIP-2 trains in two stages:

1. **Vision–language representation learning.** The Q-Former is trained against the frozen ViT only, with three losses combined: image–text contrastive (ITC, à la CLIP), image–text matching (ITM, a binary classifier on hard negatives), and image-grounded text generation (ITG, a captioning loss). This stage teaches the queries what visual content is.
2. **Vision-to-language generation.** The Q-Former's output is plugged into the frozen LLM and trained with standard language-modelling loss on image-conditioned text. This stage teaches the queries to *speak the LLM's language*.

The total trainable parameter count is **~188 M** — about 1.4% of OPT-6.7B. Yet BLIP-2 zero-shots VQA-v2 and image captioning competitively with much larger end-to-end-trained models.

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16, device_map="auto",
)

image = Image.open("street.jpg")
prompt = "Question: What is happening in this photo? Answer:"
inputs = processor(image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=60)
print(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
```

### LLaVA: Visual Instruction Tuning

BLIP-2 pre-trains a clever bridge. LLaVA (Liu et al., NeurIPS 2023) does something more radical: replace the bridge with a **single linear layer** (or 2-layer MLP) and instead invest in **visual instruction-following data**.

![LLaVA — vision encoder, projector, LLM, and connector comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig3_llava_architecture.png)

The pipeline is almost embarrassingly simple. Take a frozen CLIP ViT-L/14, dump its 256 patch features $Z_v \in \mathbb{R}^{256 \times 1024}$, push them through a learned projector $W$ to land in the LLM's 4096-dim embedding space:

$$H_v = W \cdot Z_v.$$

Concatenate $H_v$ with the user's text token embeddings, feed everything to Vicuna, and predict the assistant's response autoregressively. Training proceeds in two stages:

1. **Feature alignment.** Train only $W$ on 558 K image–caption pairs (LAION/CC/SBU). The LLM learns nothing new — only the projector learns to map CLIP features into a space the frozen LLM can interpret.
2. **End-to-end visual instruction tuning.** Unfreeze the LLM and fine-tune $W$ + LLM together on **158 K image–instruction pairs generated by GPT-4** from COCO captions and bounding boxes. GPT-4 was prompted to produce three types of conversation: detailed descriptions, conversational Q&A, and complex reasoning chains.

The single linear projector is responsible for at most ~17 M parameters yet, on MMBench, LLaVA-1.5 with the MLP-2 connector reaches 80.0 — beating heavier Q-Former-based designs in their own benchmarks. The lesson: **for instruction-following MLLMs, data quality and end-to-end fine-tuning matter more than the connector**.

```python
# LLaVA-1.5 inference via Transformers
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto",
)

image = Image.open("ui_screenshot.png")
prompt = ("USER: <image>\nWhat does this UI do, and is there a bug? "
          "Be specific. ASSISTANT:")

inputs = processor(text=prompt, images=image,
                   return_tensors="pt").to("cuda", torch.float16)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

---

## Vision–Language Tasks: VQA, Captioning, Retrieval

![VQA, image captioning, and cross-modal retrieval pipelines](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig4_vl_tasks.png)

The three classical task families dominate evaluation and most practical use:

- **Visual Question Answering (VQA).** Given an image and a question, produce an answer. Datasets: VQA-v2, OK-VQA (open-knowledge), GQA (compositional), TextVQA (requires OCR). Modern MLLMs handle all of these with a single instruction-tuned model.
- **Image Captioning.** Generate a natural-language description. Datasets: COCO Captions, NoCaps, Flickr30K. Beam search with `num_beams=3` and `length_penalty>1.0` typically gives a good detail/fluency trade-off.
- **Cross-modal Retrieval.** Given a text query, return the most relevant images (or vice-versa). Reduces to nearest-neighbour search over normalised CLIP embeddings — production systems use FAISS-IVF or ScaNN with billions of items.

### Image Captioning

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-large"):
        self.proc = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).eval()

    @torch.no_grad()
    def caption(self, image_path, prefix=None,
                max_length=50, num_beams=3):
        img = Image.open(image_path).convert("RGB")
        # `prefix` enables conditional captioning, e.g. "A photo of"
        inputs = self.proc(img, text=prefix, return_tensors="pt") \
            if prefix else self.proc(img, return_tensors="pt")
        out = self.model.generate(
            **inputs, max_length=max_length,
            num_beams=num_beams, length_penalty=1.0, early_stopping=True,
        )
        return self.proc.decode(out[0], skip_special_tokens=True)

cap = ImageCaptioner()
print(cap.caption("dog.jpg"))                        # unconditional
print(cap.caption("dog.jpg", prefix="A photo of"))   # conditional
```

### Visual Question Answering

```python
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

class VQA:
    def __init__(self, model_id="Salesforce/blip-vqa-base"):
        self.proc = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id).eval()

    @torch.no_grad()
    def answer(self, image_path, question, max_length=30):
        img = Image.open(image_path).convert("RGB")
        inputs = self.proc(img, question, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=max_length)
        return self.proc.decode(out[0], skip_special_tokens=True)

vqa = VQA()
print(vqa.answer("market.jpg", "How many apples are on the table?"))
print(vqa.answer("market.jpg", "What color is the basket?"))
```

### Production-Ready CLIP Retrieval

```python
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPRetrieval:
    """Tiny CLIP-based image index. For >1 M items use FAISS-IVF."""

    def __init__(self, model_id="openai/clip-vit-base-patch32",
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.embeds = []   # list of np.ndarray of shape (512,)
        self.paths = []

    @torch.no_grad()
    def add_images(self, paths, batch_size=32):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            self.embeds.extend(emb.cpu().numpy())
            self.paths.extend(batch_paths)

    @torch.no_grad()
    def search_text(self, query, top_k=5):
        inputs = self.proc(text=[query], return_tensors="pt",
                           padding=True).to(self.device)
        q = self.model.get_text_features(**inputs)
        q = q / q.norm(dim=-1, keepdim=True)
        sims = np.vstack(self.embeds) @ q.cpu().numpy().T
        sims = sims.flatten()
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.paths[i], float(sims[i])) for i in idx]
```

For an index of >1 M images, switch to FAISS with an IVF-PQ index — embeddings are 512-dim float32, so 1 M items occupy ~2 GB in raw form but compress to <100 MB with PQ.

---

## Audio: Whisper

Speech is the second great non-text modality, and **Whisper** (Radford et al., ICML 2023) is the open-source workhorse. It is a vanilla encoder–decoder Transformer trained on **680 K hours** of weakly-supervised multilingual audio scraped from the internet, with a single objective and a single model that handles transcription, translation, language identification, and voice-activity detection.

![Log-mel spectrogram input and Whisper encoder–decoder](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig5_whisper_audio.png)

The pipeline:

1. Audio is resampled to **16 kHz mono**, then split into 30-second chunks. Short clips are padded with silence; long clips are processed with a sliding window.
2. The chunk is converted to an **80-channel log-mel spectrogram** with a 25 ms window and 10 ms hop — 3000 frames per 30-second chunk.
3. Two **stride-2 Conv1D + GELU** layers downsample to 1500 audio tokens, which the **encoder** (12–32 self-attention layers depending on size) processes.
4. The **decoder** is autoregressive and prepended with control tokens: `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`. Switching `<|transcribe|>` for `<|translate|>` makes the same model produce English translations of foreign-language audio.

Model sizes: tiny (39 M), base (74 M), small (244 M), medium (769 M), large (1.55 B). For most production work, **`small` or `medium` on a GPU give 0.1–0.3× real-time latency with WER under 10% on common English audio**. Use `large-v3` when accents, code-switching, or noise dominate.

```python
import whisper

class WhisperASR:
    def __init__(self, size="base", device="cuda"):
        self.model = whisper.load_model(size, device=device)

    def transcribe(self, audio_path, language=None, task="transcribe"):
        # task in {"transcribe", "translate"}
        return self.model.transcribe(
            audio_path,
            language=language,           # None -> auto-detect
            task=task,
            word_timestamps=True,        # adds per-word timing
            condition_on_previous_text=False,  # safer on long audio
            no_speech_threshold=0.6,
            verbose=False,
        )

asr = WhisperASR(size="small")
result = asr.transcribe("meeting.mp3")
print(f"Detected language: {result['language']}")
for seg in result["segments"]:
    print(f"[{seg['start']:6.2f} - {seg['end']:6.2f}] {seg['text'].strip()}")
```

**Practical notes.** Set `condition_on_previous_text=False` on long recordings — otherwise an early hallucination cascades. For batched throughput, use [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend, 4–8× faster). For diarisation (who-spoke-when), pair Whisper with `pyannote.audio`.

---

## GPT-4V and the Frontier MLLM Landscape

![GPT-4V capability matrix and example interactions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig6_gpt4v_examples.png)

**GPT-4V** (and its successors GPT-4o and GPT-4.1) treat images as a first-class input modality alongside text. The model handles a broad mix of tasks out of the box:

- **OCR and document understanding** — read receipts, forms, tables, screenshots.
- **Charts and graphs** — extract values, summarise trends, answer comparative questions.
- **Diagrams and flowcharts** — explain UML, system diagrams, hand-drawn whiteboards.
- **Code and UI** — debug screenshots of code, suggest CSS fixes from a rendered page.
- **Visual reasoning** — multi-image comparison, step-by-step math from a photo of an exam, scene safety reasoning.

```python
from openai import OpenAI
import base64, mimetypes

client = OpenAI()

def analyse_image(path: str, prompt: str, model: str = "gpt-4o") -> str:
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}",
                               "detail": "high"}},
            ],
        }],
        max_tokens=500,
        temperature=0.2,
    )
    return resp.choices[0].message.content

print(analyse_image("dashboard.png",
                    "Summarise the key metrics and any anomaly."))
```

The `detail` parameter (`"low" | "high" | "auto"`) trades cost for resolution — `"high"` re-tiles the image into 512-px patches and uses ~10× more vision tokens.

**Open-weight alternatives** worth knowing: LLaVA-1.6 (good baseline, easy to fine-tune), Qwen-VL-Max (strong on Chinese and OCR), InternVL (strong on grounding), CogVLM (strong on perception). All can be deployed locally for cost control or on-prem privacy.

**Where MLLMs still struggle** (consistently, across vendors):

- **Counting** beyond ~5 objects.
- **Fine-grained spatial reasoning** ("which pen is closer to the cup?").
- **Reading dense text** at low resolution — small fonts in screenshots.
- **Hallucinations of objects** that are not in the image, especially when the prompt is leading ("describe the dog" when there is no dog).

---

## Video Understanding

Video is a 4D tensor (time, height, width, channels) and naive frame-by-frame processing explodes the token budget. The dominant pattern in 2024 is:

1. **Uniform frame sampling** — pick $N \in \{8, 16, 32\}$ frames evenly across the clip.
2. **Per-frame vision encoding** with a frozen ViT, optionally with a temporal-pooling adapter (e.g., VideoLLaMA, Video-LLaVA) that mixes neighbouring frames before projection.
3. **Concatenation** of frame tokens (or pooled frame tokens) into the LLM's context, often with a temporal positional embedding so the model knows the order.

For very long videos (hours), use a coarse-to-fine strategy: summarise short windows with a captioner, then run an LLM over the captions to retrieve the few seconds worth analysing in detail.

```python
import cv2

def sample_frames(video_path, num_frames=8):
    """Uniformly sample frames as PIL.Image objects."""
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read {video_path}")
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames
```

Models like Gemini 1.5 Pro accept native video input and handle hour-long clips by streaming tokens through their long-context attention.

---

## Evaluating MLLMs: MMBench, MME, MMMU, SEED, POPE

![Multimodal benchmark comparison and capability radar](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/multimodal-nlp/fig7_mllm_benchmarks.png)

Benchmark choice matters more than people admit. Each one stresses a different facet, and a single-number leaderboard is misleading.

| Benchmark | What it measures | Format | Notes |
|-----------|------------------|--------|-------|
| **MMBench** | 20 ability dimensions: perception, reasoning, OCR, fine-grained recognition | Multiple choice (4 options) | "Circular evaluation" rotates option order to defeat letter bias. |
| **MME** | 14 subtasks split into Perception (10) + Cognition (4) | Yes/No questions | Score is sum of accuracies (max 2800). Easy to game with "always say yes". |
| **MMMU** | College-level questions across 30 subjects (medicine, engineering, art) | Open-ended + MCQ | The hardest of the lot in 2024 — frontier models still under 60%. |
| **SEED-Bench** | Spatial relations, instance counting, scene understanding, temporal | MCQ | Includes a video subset (SEED-Bench v2). |
| **POPE** | Object-hallucination probing | Yes/No about object presence | Pairs real and *fictional* objects to expose hallucination. |
| **TextVQA / DocVQA** | OCR-heavy reading comprehension | Open-ended | Tests whether the vision tower can read. |
| **RefCOCO** | Phrase grounding (return a bounding box) | Region prediction | Diagnostic for spatial precision. |

A good evaluation report includes at least one **perception** benchmark (MME, SEED), one **reasoning** benchmark (MMMU), one **hallucination** benchmark (POPE), and a domain-specific test for the actual deployment.

---

## Practical Tips for Building Multimodal Applications

A few hard-won lessons from shipping MLLM systems:

- **Pre-process aggressively.** Resize to the model's native resolution (CLIP: 224, LLaVA-1.5: 336, GPT-4V high-detail: 768). Larger inputs do not help and waste vision tokens.
- **Cache image embeddings.** A CLIP encode is far cheaper than re-encoding the same image for every query — store the 512-d vector and reuse it.
- **Quantise the LLM half.** 4-bit AWQ / GPTQ on the language model half of LLaVA-1.5 cuts VRAM from 14 GB to ~5 GB with <1 MMBench point lost.
- **Keep prompts grounded.** Phrasing matters enormously. *"What color is the car?"* is fine; *"What color is the car if there is one?"* drops hallucination on no-car images by ~40%. Always allow the model to say *"I don't know"* explicitly.
- **Watch for OCR drift.** Many MLLMs OCR text inside images and then "trust" it — adversarial text in the corner of a photo can hijack the response. Sanitise inputs in security-sensitive deployments.

---

## Frequently Asked Questions

**Q: CLIP vs BLIP vs LLaVA — when do I pick each?**
Use **CLIP** for retrieval, zero-shot classification, and any system that needs only a fixed embedding. Use **BLIP-2** when you want VQA / captioning with a frozen LLM and tight compute. Use **LLaVA-style** instruction-tuned MLLMs when you need conversational, instruction-following multimodal interaction.

**Q: Why does BLIP-2 train so few parameters?**
The expensive parts — a 1 B-parameter ViT and a 7 B-parameter LLM — are frozen. Only the 188 M-parameter Q-Former bridges them. The two-stage curriculum keeps the bridge focused: stage 1 grounds it in vision, stage 2 aligns its output to the LLM's vocabulary.

**Q: How much data do I need to fine-tune?**
For domain adaptation of a pre-trained MLLM (LLaVA, Qwen-VL), 5 K – 50 K high-quality image–instruction pairs are typically enough with LoRA on the LLM and full training of the projector. For pre-training from scratch, think 10⁸–10⁹ pairs.

**Q: How do I keep VRAM under 24 GB?**
Combine three things: 4-bit quantisation of the LLM (`bitsandbytes` or AWQ), LoRA on attention projections, and gradient checkpointing. A 7B LLaVA fine-tunes comfortably on a single RTX 4090.

**Q: How do I evaluate honestly?**
Pick a *primary* benchmark aligned with your use case (e.g., DocVQA for documents), a *hallucination* benchmark (POPE), and a held-out internal test set. Always report at least three benchmarks — single-number rankings hide trade-offs.

**Q: Can I use CLIP on Chinese text?**
Original OpenAI CLIP is 95% English. For Chinese, use **Chinese-CLIP** (OFA-Sys), **Qwen-VL**, or **Wukong-CLIP**. They are drop-in replacements with the same API.

---

## Series Navigation

- **Previous**: [Part 10 — RAG and Knowledge Enhancement](/en/nlp-rag-knowledge-enhancement/)
- **Next**: [Part 12 — Frontiers and Practical Applications](/en/nlp-frontiers-applications/)
- [View all 12 parts in the NLP series](/tags/NLP/)
