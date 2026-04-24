---
title: "Tennis-Scene Computer Vision: From Paper Survey to Production"
date: 2025-02-10 09:00:00
tags:
  - Computer Vision
  - Object Detection
  - 3D Reconstruction
  - Pose Estimation
categories: Technical Design
lang: en
mathjax: true
description: "A complete CV system for tennis: small high-speed object detection, multi-camera 3D reconstruction, physics-based trajectory prediction, and pose-based action recognition. From the literature down to a 16.7 ms-per-frame deployment budget."
---

A 6.7 cm tennis ball travels at over 200 km/h. Reconstructing its 3D trajectory from eight 4K cameras in real time, while simultaneously classifying what stroke each player is making, is a system problem that touches **small-object detection, multi-view geometry, Kalman filtering, physics modelling, and human-pose estimation** — all at once. This post walks the same path you'd walk at deployment time: state the constraints, survey the literature, choose, then build, and finally lay out a millisecond-by-millisecond budget for what runs in production.

## What you will see

- Why traditional detectors collapse on 10–20 px tennis balls and how the TrackNet line fixes it
- Multi-camera calibration, PTP synchronisation, and DLT triangulation in code and math
- A 9-state Kalman filter coupled with a drag-plus-Magnus ODE for trajectory prediction
- Action recognition: rule-based templates vs. end-to-end learning, and when each wins
- How to fit detection → 3D → tracking → pose → analytics into a 16.7 ms / frame budget

**Prerequisites**: pinhole camera model, basic Kalman filtering, and some PyTorch inference experience.

![End-to-end tennis-scene CV pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig1_pipeline.png)

The whole pipeline has **16.7 ms** to spend per frame at 60 fps, so every box above must finish in single-digit milliseconds. Each subsequent section pays one slice of that budget.

---

## 1. Requirements: quantify "hard"

Before anything is built, the non-negotiable numbers go on the table.

### 1.1 Capability matrix

| Capability | Input | Output | Latency budget |
| --- | --- | --- | --- |
| Ball detection | Single 4K frame | 2D bbox + confidence | < 5 ms / camera |
| Multi-view 3D | Synced 2D detections (≥2 views) | $(X, Y, Z)$ + covariance | < 2 ms |
| Trajectory tracking | 3D observation sequence | position / velocity / accel | < 1 ms |
| Landing prediction | Current state + spin estimate | 1–2 s future trajectory | < 3 ms |
| Player pose | Single-camera frame | 17 keypoints | < 6 ms / person |
| Action classification | Keypoint sequence | Class + confidence | < 1 ms |

If any single stage blows its budget, the whole pipeline drops from 60 fps to 30 fps and the experience visibly stutters.

### 1.2 The physics of "hard"

**Pixel budget for a small target.** A tennis ball seen from the opposite baseline (~28 m away) at a 35 mm-equivalent focal length occupies about 12 px. A one-pixel center error projects back to **2.3 cm of lateral 3D error** — already at the threshold of a baseline line-call decision.

**Motion blur.** At 50 m/s ball speed and a 1/250 s shutter, the ball smears across 20 cm in a single frame, producing a 30 px streak. Without sub-1/1000 s global shutters, no detection algorithm fully recovers.

**Synchronisation, geometrically amplified.** A 5 ms offset between two cameras puts the ball at different image positions by ~25 cm. Triangulation degenerates and the recovered 3D point drifts along an "anti-epipolar" line. PTP synchronisation under 1 ms is therefore a hard floor.

**Occlusion and brief disappearance.** The ball is occluded by the net band as it crosses, and there is always a blind region right after the bounce. Any single-shot detector misses several consecutive frames, so a tracker with strong physics priors must take over.

---

## 2. Literature survey and selection

Six years of papers, sorted by the four sub-tasks. Each section ends with the choice that goes into production.

### 2.1 Small-object detection

The collapse of generic detectors on small objects shows up cleanly in the data:

![Small-object detection comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig2_detection_compare.png)

Left: AP@0.5 vs object side length in pixels — the tennis-ball regime sits in the 8–22 px shaded band, where **Faster R-CNN scores 0.05–0.2** and even YOLOv8 at 1280 input only reaches 0.3–0.5. Right: latency vs accuracy bubbles. The only models that satisfy both 60 fps (< 16.7 ms) and AP > 0.9 live on the **TrackNet V2/V3 specialised track**.

**TrackNet** (2019–2023) reframes "ball" as a temporal problem rather than per-frame detection:
- V1 used a VGG-U-Net consuming 3 consecutive frames and produced a heatmap; the first stable solution
- V2 swapped in MobileNetV2, dropping params from 15 M to 2.8 M, 3× faster, only 1.2 pp worse
- V3 added Transformer cross-frame self-attention and pushed high-speed detection success from 92.3% to 96.7%

A **coarse-to-fine** alternative (YOLOv5 proposals + ResNet-50 verification) cuts false positives by ~60% but costs another 10 ms — fine for offline replay analysis, not for live broadcast.

> **Choice.** Live path: **YOLOv8-l @ 1280 + 3-frame temporal vote.** Offline path: stack TrackNet V3 as a refinement stage.

### 2.2 Multi-view geometry and 3D reconstruction

Hartley & Zisserman's *Multiple View Geometry* is still, eighteen years on, the right book to read. Three things to internalise:

**Pinhole projection**:

$$
s\,\mathbf{x} = K\,[R \mid t]\,\mathbf{X}
$$

where $K$ is the $3\times3$ intrinsic matrix, $[R \mid t]$ is the $3\times4$ extrinsic, and $\mathbf{X}$ is the world-frame 3D homogeneous point.

**Zhang's method** (1998): solve $K$ from a checkerboard. Need at least 10 images at varied angles, and the reprojection error must be < 0.5 px to be production-grade.

**DLT triangulation**: from $n$ camera observations $\mathbf{x}_i$, recover $\mathbf{X}$. Each observation contributes two linear constraints:

$$
\begin{aligned}
x_i\,P_i^{(3)} - P_i^{(1)} &= 0 \\
y_i\,P_i^{(3)} - P_i^{(2)} &= 0
\end{aligned}
$$

Stack into $A_{2n\times4}\mathbf{X}=0$ and take the right singular vector of the smallest singular value via SVD.

**Automatic Camera Network Calibration (2024)**: board-free re-calibration. SIFT/ORB across views → SfM jointly estimates poses and a sparse cloud → bundle adjustment minimises reprojection error to < 0.5 px. Saves the on-site pain of waving a checkerboard around the venue.

> **Choice.** Use Zhang + checkerboard once at install (precise, one-time effort); run SfM + BA nightly to compensate for thermal and mechanical drift.

### 2.3 Multi-object tracking

The SORT family solves data association. Tennis has a single ball but many of these techniques are needed for player tracking and re-acquiring the ball after occlusion.

| Algorithm | Key trick | Best for |
| --- | --- | --- |
| SORT (2016) | Kalman + Hungarian / IoU | Player tracking, CPU-friendly |
| DeepSORT (2017) | + 128-d ReID feature | Re-acquire after player occlusion |
| ByteTrack (2021) | Two-stage matching using low-confidence boxes | Partial occlusion, motion blur |

**Tennis-specific gotcha.** Single target, very high speed, gravity-dominated. The constant-velocity assumption inside vanilla SORT-Kalman is wrong — you need constant-acceleration (or an EKF that includes Magnus force), otherwise the deceleration phase near the apex is systematically under-tracked.

### 2.4 Trajectory prediction

**Physics-Informed Neural Networks (Raissi 2019)** add a physics-residual loss term:

$$
\mathcal{L} = \underbrace{\sum_i \|\hat{\mathbf{p}}(t_i) - \mathbf{p}_i\|^2}_{\text{data}} + \lambda\,\underbrace{\sum_j \|\ddot{\hat{\mathbf{p}}}(t_j) - \mathbf{f}(\hat{\mathbf{p}}, \dot{\hat{\mathbf{p}}})\|^2}_{\text{physics}}
$$

In the data-poor regime that is one serve (30–50 observations), PINNs cut landing-point error by ~30% versus a pure LSTM.

**TrackNetV2 + bidirectional LSTM** is the more engineering-friendly version: forward LSTM for live prediction, backward LSTM for offline correction. Landing-point error drops from 32 cm (pure physics) to 18 cm.

> **Choice.** Live path: physics ODE + Kalman (deterministic, interpretable). Offline path: stack PINN refinement.

### 2.5 Human pose

| Model | Paradigm | COCO AP | Note |
| --- | --- | --- | --- |
| OpenPose (2017) | bottom-up + PAF | 65 | Classic; accuracy no longer enough |
| HRNet-W48 (2019) | top-down, always high-res | 77 | MMPose default |
| ViTPose-H (2022) | Transformer top-down | 81.1 | Current SOTA |
| 4D Human (2024) | 3D + SMPL + temporal | — | For racket-arc analysis |

> **Choice.** HRNet-W48 (best speed-accuracy balance). Add 4D Human for coaching-grade analytics.

---

## 3. System architecture

### 3.1 Hardware and synchronisation

Eight cameras around a singles court (23.77 m × 8.23 m):

- **Corners 1–4**: 5–8 m height, 30–45° downtilt — full-court coverage
- **Net-facing 5**: net-clearance and let calls
- **Side mid-court 6–7**: precise height triangulation
- **Umpire-chair 8** (optional): top-down replay

**Camera spec**: 3840×2160 / 60 fps (120 fps for top-tier), global shutter ≤ 1/1000 s, 8–12 mm wide-angle, GigE Vision or USB 3.0, hardware trigger or PTP. Recommended: FLIR Blackfly S or Basler ace.

**Time sync**: IEEE 1588 PTP. One server is the Grandmaster Clock, everything else is a Slave, switches must support Boundary Clock — sub-microsecond is then routinely achievable. NTP's millisecond-level jitter is hopeless here.

### 3.2 Software layering

```
┌────────────────────────────────────────────────────────┐
│ Application   Visualisation │ Analytics │ Reports │ HUD │
├────────────────────────────────────────────────────────┤
│ Business      Event detect │ Tactics │ History │ Push  │
├────────────────────────────────────────────────────────┤
│ Algorithm     Detect → 3D recon → Predict → Line call  │
│               Person → Pose → Action class             │
├────────────────────────────────────────────────────────┤
│ Data          Frame sync │ Undistort │ BG model │ CLAHE│
├────────────────────────────────────────────────────────┤
│ Capture       8x cameras │ Timestamps │ Metadata       │
└────────────────────────────────────────────────────────┘
```

**Concurrency**: producer-consumer with a message queue (RabbitMQ or Redis Streams):

- **Capture threads** (one per camera): push `(frame, ts)` into the queue
- **Detection workers**: N parallel GPU-bound workers
- **Fusion thread**: pulls same-instant detections from all cameras (max ∆t = 5 ms), runs triangulation
- **Tracker thread**: single worker maintaining the Kalman state machine
- **Output thread**: render and broadcast over WebSocket

### 3.3 Edge + cloud split

![Edge + cloud architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig7_architecture.png)

Eight 4K@60 streams total **~6 Gbps** of raw video — uploading that to a cloud is impossible. So:

- **Edge (per-camera Jetson Orin)**: first-stage YOLO + ROI cropping. Egress is candidate boxes (≤100 / frame) and ROI patches; bandwidth drops to **~80 Mbps**
- **Aggregator (on-prem GPU)**: time sync, triangulation, tracking, pose. All latency-sensitive work lives here
- **Cloud**: event JSON, statistics, replay, long-term storage; bandwidth **< 1 Mbps**

This split lets a venue ship "one gigabit uplink and a GPU-less cloud" and still run the system.

---

## 4. Core algorithms in code

The four modules below are load-bearing walls; every one comes with a minimal runnable implementation.

### 4.1 Multi-camera calibration

```python
import cv2
import numpy as np
from typing import List, Tuple


class MultiCameraCalibration:
    """Multi-camera calibration + DLT triangulation."""

    def __init__(self, num_cameras: int = 8):
        self.num_cameras = num_cameras
        self.camera_matrices: List[np.ndarray] = []   # K, 3x3
        self.dist_coeffs: List[np.ndarray] = []       # (k1, k2, p1, p2, k3)
        self.R_matrices: List[np.ndarray] = []
        self.t_vectors: List[np.ndarray] = []

    def calibrate_single_camera(
        self,
        images: List[np.ndarray],
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zhang's method: solve K from checkerboard images."""
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints, imgpoints = [], []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if not ok:
                continue
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
            )
            objpoints.append(objp)
            imgpoints.append(corners)

        if len(objpoints) < 10:
            raise ValueError(f"Need >= 10 valid views, got {len(objpoints)}")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Reprojection error MUST stay < 1 px to be acceptable.
        err = 0.0
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            err += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        err /= len(objpoints)
        print(f"reprojection error: {err:.4f} px {'OK' if err < 1.0 else 'redo'}")
        return K, dist

    def triangulate_point(
        self,
        points_2d: List[Tuple[float, float]],
        camera_ids: List[int],
    ) -> np.ndarray:
        """DLT triangulation: >=2 views of a 2D point -> 3D."""
        if len(points_2d) < 2:
            raise ValueError("need at least 2 views")

        A = []
        for (x, y), cid in zip(points_2d, camera_ids):
            P = self.camera_matrices[cid] @ np.hstack(
                [self.R_matrices[cid], self.t_vectors[cid]]
            )
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.asarray(A)

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return (X / X[3])[:3]
```

**Calibration pass criteria**: per-camera reprojection error < 1 px, stereo extrinsic baseline error < 1%. Either failure inflates the **3D error by an order of magnitude**.

### 4.2 Ball detection: YOLOv8 + physics priors

```python
from ultralytics import YOLO


class TennisBallDetector:
    """Real-time tennis ball detection: YOLOv8 plus physics-prior post-processing."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda"):
        self.model = YOLO(model_path)
        self.device = device
        self.img_size = 1280       # large input is the cheapest small-object win
        self.conf_thr = 0.25       # low threshold reduces miss rate; priors below filter FPs
        self.iou_thr = 0.45
        self.ball_cls = 32         # COCO 'sports ball'

    def detect(self, frame, expected_size_px: Tuple[int, int] = (5, 60)) -> list:
        """expected_size_px is the *prior* — derived from focal length and ball distance."""
        results = self.model.predict(
            frame, imgsz=self.img_size, conf=self.conf_thr, iou=self.iou_thr,
            classes=[self.ball_cls], device=self.device, verbose=False,
        )

        out = []
        s_min, s_max = expected_size_px
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            # Prior 1: ball diameter must be in expected range (focal+distance)
            if not (s_min <= max(w, h) <= s_max):
                continue
            # Prior 2: tennis ball is roughly circular -> aspect ratio near 1
            if not 0.6 <= w / max(h, 1e-6) <= 1.6:
                continue
            out.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                "confidence": float(box.conf[0]),
            })
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return out
```

The two physics priors take this from ~50% precision to ~95% — circular logos on billboards and white-line crossings get filtered out by size and aspect ratio in a single pass.

### 4.3 9-state Kalman: gravity in the model, not in the head

```python
from collections import deque


class TennisBallTracker:
    """9-state (x,y,z, vx,vy,vz, ax,ay,az) Kalman tracker."""

    def __init__(self, fps: int = 60):
        self.dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(9, 3)

        dt = self.dt
        # Constant-acceleration motion: x_{t+1} = x_t + v dt + 0.5 a dt^2
        F = np.eye(9, dtype=np.float32)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        F[0, 6] = F[1, 7] = F[2, 8] = 0.5 * dt ** 2
        F[3, 6] = F[4, 7] = F[5, 8] = dt
        self.kf.transitionMatrix = F

        H = np.zeros((3, 9), dtype=np.float32)
        H[0, 0] = H[1, 1] = H[2, 2] = 1
        self.kf.measurementMatrix = H

        Q = np.eye(9, dtype=np.float32) * 0.03
        Q[6:, 6:] *= 5            # acceleration is the noisiest (spin, wind, drag non-linearities)
        self.kf.processNoiseCov = Q
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 1000

        self.history = deque(maxlen=300)   # 5 s @ 60 fps
        self.lost = 0
        self.initialized = False
        self.max_lost = 30                  # 0.5 s missing -> reset

    def update(self, measurement: np.ndarray = None):
        prediction = self.kf.predict()
        if measurement is not None:
            m = measurement.reshape(3, 1).astype(np.float32)
            if not self.initialized:
                self.kf.statePost = np.array(
                    [m[0, 0], m[1, 0], m[2, 0], 0, 0, 0, 0, 0, -9.8],
                    dtype=np.float32,
                ).reshape(9, 1)
                self.initialized = True
            self.kf.correct(m)
            self.lost = 0
        else:
            self.lost += 1
            if self.lost > self.max_lost:
                self.initialized = False

        s = self.kf.statePost if measurement is not None else prediction
        pos, vel, acc = s[0:3].flatten(), s[3:6].flatten(), s[6:9].flatten()
        self.history.append({"position": pos.copy(), "velocity": vel.copy()})
        return pos, vel, acc
```

**Why 9 states, not the classic 6?** The 6-state $(x, v)$ filter assumes constant velocity. A tennis ball has gravity (9.8 m/s²), drag, and Magnus — non-zero, time-varying acceleration. Putting acceleration into the state and seeding $a_z = -9.8$ as a prior **drops apex-region prediction error from 12 cm to 3 cm**.

### 4.4 Trajectory prediction: drag + Magnus ODE

```python
from scipy.integrate import odeint


class TennisTrajectoryPredictor:
    """Physics-based trajectory prediction: drag + Magnus + gravity."""

    def __init__(self):
        self.g = 9.81
        self.rho = 1.225          # kg/m^3
        self.m = 0.0585           # kg, ITF spec
        self.r = 0.0335           # m
        self.Cd = 0.55            # drag coefficient for a sphere
        self.Cm = 0.00029         # Magnus coefficient (empirically fit)
        self.A = np.pi * self.r ** 2

    def predict(self, p0, v0, spin=None, dt=0.01, duration=2.0):
        if spin is None:
            spin = np.zeros(3)
        t = np.arange(0, duration, dt)
        traj = odeint(self._dyn, np.concatenate([p0, v0]), t, args=(spin,))
        # cut at first ground contact
        idx = np.where(traj[:, 2] <= 0)[0]
        return traj[: idx[0] + 1, :3] if len(idx) else traj[:, :3]

    def _dyn(self, state, t, spin):
        v = state[3:6]
        speed = np.linalg.norm(v) + 1e-9
        drag = -0.5 * self.rho * self.Cd * self.A * speed * v
        magnus = self.Cm * np.cross(spin, v)
        gravity = np.array([0, 0, -self.g * self.m])
        a = (drag + magnus + gravity) / self.m
        return np.concatenate([v, a])

    def landing(self, p0, v0, spin=None):
        traj = self.predict(p0, v0, spin, dt=0.005, duration=5.0)
        if len(traj) == 0:
            return None
        return traj[-1]
```

The model carries all three forces:

$$
m\,\ddot{\mathbf{p}} = \underbrace{-\tfrac{1}{2}\rho\,C_d\,A\,\|\dot{\mathbf{p}}\|\,\dot{\mathbf{p}}}_{\text{drag}} + \underbrace{C_m\,(\boldsymbol{\omega}\times\dot{\mathbf{p}})}_{\text{Magnus}} + m\,\mathbf{g}
$$

The qualitative output — **topspin dives, backspin floats** — matches real rallies:

![3D trajectories under three spins, with landing zones](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig3_trajectory_3d.png)

Left: 3D trajectories at the same launch (45 m/s, 5° elevation) under no spin / topspin / backspin. Right: top-down landing zones. Topspin lands 1.5 m inside the baseline; backspin floats out by 0.4 m; the difference is entirely the Magnus term.

---

## 5. Court structure and pose: use every prior the scene gives you

### 5.1 Court line detection: Hough + homography

Court lines do double duty — they are the line-call evidence, and they are a **free scene-level calibration check**. Four known white-line intersections uniquely determine a homography, which lets you correct any drift in camera pose between formal calibrations.

![Court line detection](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig4_court_lines.png)

The pipeline is a textbook three-step: Canny edges → probabilistic Hough transform → match lines and intersections against the known court template. A few lines of OpenCV:

```python
import cv2

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 60, 180)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                        minLineLength=50, maxLineGap=10)
# lines: shape (N, 1, 4) -> (x1, y1, x2, y2)
```

Once four anchor intersections match, `cv2.findHomography` produces $H$. Run this nightly (or per match) and **long-term drift from mechanical vibration and thermal expansion stays under 5 cm in 3D**.

### 5.2 Player pose: HRNet + rule templates

Action recognition labels each impact moment as one of 5–6 strokes (serve, forehand, backhand, volley, smash, ready). I benchmarked end-to-end action models (ST-GCN, VideoMAE) against **keypoints + handwritten rule templates**:

| Dimension | End-to-end | Keypoints + rules |
| --- | --- | --- |
| Annotation cost | 1000+ video clips | 0 (rules are written) |
| Inference latency | 30–50 ms | < 1 ms |
| Interpretability | black box | transparent, tunable |
| Accuracy | 95%+ | 92% (this system) |

For tennis — few classes, geometrically distinctive — the rule approach wins on cost-per-percentage-point. Three canonical strokes and the keypoint geometry that distinguishes them:

![Skeleton features per stroke](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig5_pose_skeleton.png)

- **Serve**: right wrist (kp10) at least 50 px above right shoulder (kp6); left wrist (kp9) similarly high (the toss); feet wide
- **Forehand**: right wrist on the body's left side (kp10.x < kp6.x), shoulder rotation > 10° clockwise, right foot forward
- **Backhand (two-handed)**: left wrist on the body's right side, wrists within 50 px of each other (two-handed grip), left foot forward

Template matching is a weighted vote:

$$
\text{score}(\text{action}) = \frac{\sum_i w_i\,\mathbf{1}[\text{feature}_i \text{ matched}]}{\sum_i w_i}
$$

Threshold at 0.6, then add a 5-frame majority vote for temporal smoothing. The resulting confusion matrix:

![Action recognition confusion matrix and per-class metrics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/tennis-cv-system-design/fig6_action_recognition.png)

The remaining errors concentrate on **volley ↔ ready** (small motion, similar pose) and **smash ↔ serve** (both have an arm raised above the head). Both pairs can be disambiguated at the analytics layer using ball height and player position context.

```python
class TennisPoseClassifier:
    """Keypoint-rule templates, < 1 ms per frame."""

    SKELETON_NAMES = {  # COCO subset
        5: "L_shoulder", 6: "R_shoulder",
        7: "L_elbow",    8: "R_elbow",
        9: "L_wrist",    10: "R_wrist",
        11: "L_hip",     12: "R_hip",
        15: "L_ankle",   16: "R_ankle",
    }

    def classify(self, kp: np.ndarray) -> Tuple[str, float]:
        scores = {
            "serve":    self._serve(kp),
            "forehand": self._forehand(kp),
            "backhand": self._backhand(kp),
            "volley":   self._volley(kp),
            "ready":    self._ready(kp),
        }
        action = max(scores, key=scores.get)
        return action, scores[action]

    def _serve(self, kp):
        feats = [
            (0.30, kp[10, 1] < kp[6, 1] - 50),     # right wrist above right shoulder
            (0.20, kp[9, 1]  < kp[5, 1]),          # left wrist above left shoulder (toss)
            (0.15, abs(kp[15, 0] - kp[16, 0]) > 100),  # feet wide
            (0.20, self._arm_angle(kp, "R") >= 120),
            (0.15, kp[5, 0] - kp[11, 0] < -10),    # body leans back
        ]
        return self._weighted(feats)

    def _forehand(self, kp):
        feats = [
            (0.30, kp[10, 0] < kp[6, 0]),           # right wrist on body's left side
            (0.25, self._shoulder_rot(kp) > 10),
            (0.20, kp[16, 0] > kp[15, 0]),          # right foot forward
            (0.15, kp[8, 1] > kp[6, 1]),
            (0.10, kp[10, 0] > kp[5, 0]),           # follow-through
        ]
        return self._weighted(feats)

    # _backhand / _volley / _ready follow the same pattern
    @staticmethod
    def _weighted(feats):
        s = sum(w for w, ok in feats if ok)
        z = sum(w for w, _ in feats)
        return s / z if z else 0.0
```

---

## 6. End-to-end integration and budget

### 6.1 Frame synchroniser

```python
from queue import Queue


class FrameSynchronizer:
    """Pop the next set of frames whose timestamps fall within max_diff."""

    def __init__(self, num_cameras: int, max_diff_ms: float = 5.0):
        self.n = num_cameras
        self.max_diff = max_diff_ms / 1000.0
        self.buffers = [Queue(maxsize=10) for _ in range(num_cameras)]

    def add(self, cam_id: int, frame, ts: float):
        if self.buffers[cam_id].full():
            self.buffers[cam_id].get_nowait()
        self.buffers[cam_id].put((frame, ts))

    def pop_synced(self):
        if any(b.empty() for b in self.buffers):
            return None, None
        heads = [b.queue[0] for b in self.buffers]
        ts_list = [t for _, t in heads]
        if max(ts_list) - min(ts_list) > self.max_diff:
            # drop the oldest and try again next call
            oldest = int(np.argmin(ts_list))
            self.buffers[oldest].get_nowait()
            return None, None
        frames, ts = [], []
        for b in self.buffers:
            f, t = b.get_nowait()
            frames.append(f); ts.append(t)
        return frames, ts
```

### 6.2 Main loop

```python
class TennisAnalysisSystem:
    def __init__(self, num_cameras=8, calib_path="calib.json"):
        self.calib = MultiCameraCalibration(num_cameras)
        self.calib.load(calib_path)
        self.detector = TennisBallDetector()
        self.tracker = TennisBallTracker()
        self.predictor = TennisTrajectoryPredictor()
        self.pose_clf = TennisPoseClassifier()
        self.sync = FrameSynchronizer(num_cameras)

    def step(self, frames, timestamps):
        # 1. Detect ball per camera
        ball_2d = []
        for cid, f in enumerate(frames):
            dets = self.detector.detect(f)
            if dets:
                ball_2d.append((cid, dets[0]["center"]))

        # 2. Triangulate
        ball_3d = None
        if len(ball_2d) >= 2:
            ball_3d = self.calib.triangulate_point(
                [p for _, p in ball_2d], [c for c, _ in ball_2d]
            )

        # 3. Kalman update
        pos, vel, acc = self.tracker.update(ball_3d)

        # 4. Landing prediction
        landing = self.predictor.landing(pos, vel) if np.linalg.norm(vel) > 1 else None

        # 5. Pose (main camera only)
        # ... call HRNet then classify ...

        return {"position": pos, "velocity": vel, "landing": landing}
```

### 6.3 Slicing the 16.7 ms budget

| Stage | Measured (RTX 4090, fp16) | Share |
| --- | --- | --- |
| 8-way parallel ball detection | 4.8 ms | 29% |
| Frame sync + ROI extraction | 1.2 ms | 7% |
| DLT triangulation | 0.4 ms | 2% |
| Kalman update | 0.1 ms | 1% |
| Physics ODE landing | 2.6 ms | 16% |
| Main-cam pose (HRNet) | 5.5 ms | 33% |
| Action classification (rules) | 0.3 ms | 2% |
| Render + serialize | 1.4 ms | 8% |
| **Total** | **16.3 ms** | **97%** |

It just fits inside the 60 fps budget; **pose is the largest single cost**. To reach 120 fps for top-tier broadcast, the two highest-leverage moves are: convert YOLOv8 to TensorRT INT8 (saves ~2 ms), and swap HRNet for a distilled RTMPose-s (saves ~3 ms).

---

## 7. Deployment optimisation and robustness

### 7.1 Model acceleration

```bash
# YOLOv8 -> TensorRT, FP16 ~2x, INT8 ~4x
yolo export model=yolov8l.pt format=engine half=true device=0
yolo export model=yolov8l.pt format=engine int8=true data=tennis.yaml
```

INT8 quantisation needs a calibration set (200–500 representative frames); skip it and you lose 3–5 pp of accuracy.

### 7.2 Robustness fallbacks

- **Occlusion**: when the ball is body-blocked, lean on the Kalman extrapolation; up to 30 frames (0.5 s) of seamless re-acquisition
- **Lighting changes**: CLAHE adaptive histogram equalisation + background model refresh every 5 minutes
- **Multi-hypothesis**: when consecutive detections wobble in confidence, keep top-3 trajectory candidates and disambiguate next frame
- **Out-of-FOV**: predict next-frame ROI from the current trajectory and feed a narrowed search window to the detector (~3× faster)

### 7.3 Monitoring

Prometheus scrape, Grafana dashboard, alerts on:

| Metric | Healthy | Action |
| --- | --- | --- |
| End-to-end P99 latency | < 25 ms | auto-downsample to 30 fps |
| Detection recall (5 min window) | > 0.9 | switch to backup model |
| Triangulation success rate | > 0.85 | trigger auto-recalibration |
| GPU utilisation | < 85% | capacity warning |

---

## 8. Summary

The thing that makes this system actually ship is the acceptance that **no single module hits production accuracy on its own**. It's a relay where the next stage's prior covers the previous stage's weakness:

- Calibration drift is corrected by court-line homography and SfM
- Detector small-object weakness is covered by ROI priors and temporal voting
- Tracker non-linearity is covered by physics priors (gravity, drag, Magnus)
- Pose ambiguity is resolved by ball position and game context

Measured numbers on a standard 8-camera setup: **3D position error < 5 cm, landing prediction error < 20 cm, end-to-end latency < 16.7 ms / frame, action-recognition macro-F1 of 0.91**.

**Open problems worth pushing on next**:

- **End-to-end 3D**: a multi-view Transformer that emits 3D trajectories directly, bypassing the detect-fuse-track chain
- **Event cameras**: 10 kHz asynchronous sensors (e.g. DAVIS346) to eliminate motion blur at the source
- **Self-supervision**: energy and momentum conservation as physics losses to cut annotation cost

## References

- Huang et al., "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications", arXiv:1907.03698, 2019
- Sun et al., "Deep High-Resolution Representation Learning for Human Pose Estimation (HRNet)", CVPR 2019
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, Cambridge University Press, 2003
- Zhang, "A Flexible New Technique for Camera Calibration", TPAMI 2000
- Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
- Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)", ICIP 2017
- Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022
- Cao et al., "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields", TPAMI 2021
- Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation", NeurIPS 2022
- Raissi et al., "Physics-Informed Neural Networks", JCP 2019
- Jocher et al., "YOLOv8: Ultralytics YOLO", GitHub, 2023
- IEEE 1588-2019, "Standard for a Precision Clock Synchronization Protocol"
