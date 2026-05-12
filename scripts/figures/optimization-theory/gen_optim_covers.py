#!/usr/bin/env python3
"""Generate Wanxiang covers for 7 new optimization-theory articles."""
import os
import time
import requests
import dashscope
from dashscope import ImageSynthesis
import os
import oss2

dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY', 'sk-6407a4292fd94f24aecd2fcfdaaa7567')

OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
auth = oss2.Auth(OSS_AK, OSS_SK)
bucket = oss2.Bucket(auth, "https://oss-cn-beijing.aliyuncs.com", "blog-pic-ck")

PALETTE = "deep indigo and aubergine with copper and ivory highlights, mathematical aesthetic"
NEG = "text, letters, words, numbers, watermark, logo, signature, ugly, low quality, blurry, distorted, photorealistic faces, frame, border"

ARTICLES = [
    ("01", "01-convex-analysis-foundations", "Abstract geometric visualization of convex sets and convex functions, smooth curved surfaces, supporting hyperplanes, epigraph as a convex region above a curve, mathematical elegance"),
    ("05", "05-acceleration-beyond-nesterov", "Abstract visualization of accelerated trajectories curving through a contour landscape, momentum vectors, ball rolling down a valley with curving path, dynamic motion"),
    ("07", "07-second-order-methods", "Abstract visualization of curvature and tangent planes, parabolic surfaces, quadratic approximation bowls, Newton step jumping to minimum, geometric precision"),
    ("08", "08-lagrangian-duality-kkt", "Abstract visualization of duality, mirrored surfaces, supporting hyperplane between two convex regions, primal-dual symmetry, geometric balance"),
    ("09", "09-interior-point-barrier", "Abstract visualization of interior path inside a polytope, glowing central path, logarithmic barrier walls, Newton steps along curved trajectory inside feasible region"),
    ("10", "10-stochastic-variance-reduction", "Abstract visualization of stochastic noise being reduced, scattered gradient arrows converging to clean trajectory, variance contraction, statistical convergence"),
    ("11", "11-nonconvex-saddle-escape", "Abstract visualization of saddle points and escape trajectories, mountain pass topology, perturbation arrows, non-convex landscape with multiple basins, dynamical escape"),
]


def generate_cover(slug, prompt_subject):
    target_key = f"posts/covers/articles/optimization-theory/{slug}.jpg"
    # Skip if exists
    try:
        bucket.head_object(target_key)
        print(f"  EXISTS: {slug}")
        return
    except oss2.exceptions.NoSuchKey:
        pass

    full_prompt = f"{prompt_subject}, {PALETTE}, abstract conceptual art, no text or letters, clean composition, high quality digital painting"

    print(f"  Generating: {slug}...", flush=True)

    rsp = ImageSynthesis.async_call(
        model="wanx2.1-t2i-plus",
        prompt=full_prompt,
        negative_prompt=NEG,
        n=1,
        size="1024*576",
    )
    if rsp.status_code != 200:
        print(f"    FAIL submit: {rsp}")
        return
    task_id = rsp.output.task_id

    # Poll
    for _ in range(60):
        time.sleep(3)
        st = ImageSynthesis.fetch(rsp)
        if st.status_code != 200:
            continue
        status = st.output.task_status
        if status == "SUCCEEDED":
            url = st.output.results[0].url
            # Download and upload to OSS
            data = requests.get(url, timeout=60).content
            tmp = f"/tmp/cover_{slug}.jpg"
            with open(tmp, "wb") as f:
                f.write(data)
            bucket.put_object_from_file(target_key, tmp)
            print(f"    OK -> {target_key}")
            return
        elif status == "FAILED":
            print(f"    FAILED: {st}")
            return
    print(f"    TIMEOUT")


for num, slug, prompt in ARTICLES:
    generate_cover(slug, prompt)

print("\nAll done.")
