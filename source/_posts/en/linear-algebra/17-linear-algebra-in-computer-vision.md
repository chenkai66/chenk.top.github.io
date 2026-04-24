---
title: "Essence of Linear Algebra (17): Linear Algebra in Computer Vision"
date: 2024-04-29 09:00:00
tags:
  - Linear Algebra
  - computer vision
  - 3D reconstruction
  - image processing
categories:
  - Linear Algebra
series:
  name: "Linear Algebra"
  part: 17
  total: 18
lang: en
mathjax: true
description: "Images are matrices, geometric transformations are matrix multiplications, camera imaging is a projective map, and 3D reconstruction is solving linear systems. Master the linear algebra that quietly powers every corner of computer vision."
---

Computer vision is the science of teaching machines to see. What is striking is how thoroughly the whole field reduces to linear algebra: an image is a matrix, a geometric transformation is a matrix product, a camera is a $3 \times 4$ projection matrix, two-view geometry is the equation $\mathbf{x}_2^\top \mathbf{F}\, \mathbf{x}_1 = 0$, and 3D reconstruction is a sparse linear least-squares problem. Once you see the field through that lens, what once looked like a zoo of algorithms turns out to be a small set of linear-algebraic ideas applied repeatedly.

> **What you will learn**
> - How images become matrices, tensors, and high-dimensional vectors
> - 2D rotation, scale, shear, and translation as matrix multiplication, unified by homogeneous coordinates
> - Why perspective requires the homography $\mathbf{H}$ and how to fit it from point correspondences
> - The pinhole camera as a $3 \times 4$ projection matrix $\mathbf{P} = \mathbf{K}[\mathbf{R}\,|\,\mathbf{t}]$
> - Epipolar geometry, the fundamental and essential matrices, triangulation
> - SVD-based image compression and the Eckart-Young theorem
> - Convolution as matrix multiplication; the Harris structure tensor and optical flow as $2 \times 2$ linear systems
>
> **Prerequisites:** linear transformations (Chapter 3), eigendecomposition (Chapter 6), SVD (Chapter 9).

---

## 1 Images as Matrices, Tensors, and Vectors

### From pixels to a matrix

A camera sensor is a grid of light buckets. The number stored at row $i$, column $j$ of a grayscale image is the number of photons that bucket collected, normalised to a fixed range. Mathematically, an $H \times W$ grayscale image is a matrix

$$
\mathbf{I} \in \mathbb{R}^{H \times W}, \qquad I_{ij} \in [0, 1].
$$

That is the entire story. Every operation we will run on the image is some linear-algebraic transformation of this matrix.

![A grayscale image is a matrix; an RGB image is an H x W x 3 tensor.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/17-linear-algebra-in-computer-vision/fig1_image_as_matrix.png)

### Color images are 3-channel tensors

Color cameras record three intensities per pixel through red, green, and blue filters, giving a 3D array

$$
\mathcal{I} \in \mathbb{R}^{H \times W \times 3}, \qquad \mathcal{I}_{ij\,c} \in [0, 1].
$$

In NumPy this is a `(H, W, 3)` array; in PyTorch the convention is `(3, H, W)`. The standard conversion to grayscale weights the channels by human luminance sensitivity:

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B.
$$

```python
import numpy as np, cv2
img = cv2.imread('photo.jpg')           # (H, W, 3), BGR order
B, G, R = cv2.split(img.astype(float) / 255)
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

### Images as high-dimensional vectors

Many machine learning algorithms — PCA, SVMs, fully connected layers — expect a single vector. We "flatten" an $H \times W$ image into an $HW$-dimensional vector by stacking columns (or rows):

$$
\mathrm{vec}(\mathbf{I}) \in \mathbb{R}^{HW}.
$$

A $640 \times 480$ image becomes a vector with 307,200 entries, sitting in a vector space whose axes are individual pixels. Two images can now be compared with **cosine similarity**

$$
\cos\theta = \frac{\mathbf{a}^\top \mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert},
$$

a tool used in everything from image retrieval to face recognition. The huge dimension is exactly why we use SVD/PCA to find a much lower-dimensional subspace where similarity actually means something visual.

---

## 2 Geometric Transformations as Matrix Multiplication

### Why matrices

A geometric transformation maps each pixel coordinate $(x, y)$ to a new location. Storing it as a matrix gains two superpowers:

- **Composition is multiplication.** The result of applying $\mathbf{A}$ then $\mathbf{B}$ is just $\mathbf{B}\mathbf{A}$.
- **Inversion is matrix inverse.** Undoing a transformation never requires re-deriving a formula.

### Rotation, scale, and shear

Counterclockwise rotation by $\theta$ around the origin:

$$
\mathbf{R}(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}.
$$

Three properties make rotation matrices unusually pleasant. They are **orthogonal** ($\mathbf{R}^\top\mathbf{R} = \mathbf{I}$), so $\mathbf{R}^{-1} = \mathbf{R}^\top$; they have $\det\mathbf{R} = 1$, so they preserve area and orientation; and they compose by adding angles, $\mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha + \beta)$.

Scaling along the axes and shear are equally simple:

$$
\mathbf{S} = \begin{bmatrix}s_x & 0 \\ 0 & s_y\end{bmatrix}, \qquad
\mathbf{H}_k = \begin{bmatrix}1 & k \\ 0 & 1\end{bmatrix}.
$$

### Order matters

Composition reads right-to-left: $\mathbf{T} = \mathbf{R}\,\mathbf{S}$ means "first scale, then rotate". Reverse the order and you generally get a different image (the only safe case is uniform scaling, which commutes with rotation).

```python
def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

S = np.diag([2.0, 2.0])
T = rotation_matrix(np.pi / 4) @ S        # first S, then R
```

---

## 3 Homogeneous Coordinates: Making Translation Linear

### The translation problem

Rotation, scaling, and shear all fix the origin and so are linear maps $\mathbf{y} = \mathbf{A}\mathbf{x}$. **Translation** $\mathbf{y} = \mathbf{x} + \mathbf{t}$ does not — it sends the origin somewhere else, breaking linearity. We would like every transformation in the same matrix-multiplication framework.

### The trick: lift to one extra dimension

Append a $1$ to every 2D point:

$$
\begin{bmatrix} x \\ y \end{bmatrix} \;\longrightarrow\; \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}.
$$

Now translation is linear in the larger space:

$$
\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix}
=
\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix}.
$$

Geometrically, we have embedded the 2D plane into 3D as the slice $z = 1$. What looked like a 2D translation is a 3D shear, and shears are linear. Every affine transformation of the plane now fits into a single $3 \times 3$ matrix:

$$
\mathbf{M} =
\begin{bmatrix}
a_{11} & a_{12} & t_x \\
a_{21} & a_{22} & t_y \\
0 & 0 & 1
\end{bmatrix}.
$$

The upper-left $2 \times 2$ block is the linear part; the right column is the translation; the bottom row identifies "this is still affine".

![Rotation, scaling, and translation as 3x3 homogeneous matrices.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/17-linear-algebra-in-computer-vision/fig3_affine_transforms.png)

### Rotating around an arbitrary point

A common pattern: rotate around the image center $(c_x, c_y)$ rather than the origin. The composition is "translate the centre to the origin, rotate, translate back":

$$
\mathbf{M} = \mathbf{T}(c_x, c_y)\,\mathbf{R}(\theta)\,\mathbf{T}(-c_x, -c_y).
$$

```python
def affine_matrix(rotation_deg, scale, translation):
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    sx, sy = scale; tx, ty = translation
    return np.array([[sx * c, -sy * s, tx],
                     [sx * s,  sy * c, ty],
                     [0,       0,      1]])

M = affine_matrix(30, (1.5, 1.5), (100, 50))
# OpenCV uses the 2x3 top rows:
# warped = cv2.warpAffine(img, M[:2, :], (W, H))
```

---

## 4 Perspective and the Homography

### Why affine is not enough

Affine maps preserve parallelism: parallel lines stay parallel. The real world disagrees — railway tracks **converge** to the horizon. Photographs of a flat object taken from an angle exhibit this convergence as well, and we need a more flexible map to undo it.

### The $3 \times 3$ homography

A **homography** is a general $3 \times 3$ invertible matrix $\mathbf{H}$ acting on homogeneous coordinates,

$$
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix}
= \mathbf{H}
\begin{bmatrix} x \\ y \\ 1 \end{bmatrix},
\qquad
u = x'/w', \quad v = y'/w'.
$$

The crucial difference from an affine map is the bottom row: in an affine matrix it is $(0,\,0,\,1)$, whereas a homography allows nonzero entries there. Those entries produce the **perspective division** $w'$ that bends parallel lines together.

A homography has $9 - 1 = 8$ degrees of freedom (matrices are equivalent up to scale). Each point correspondence $(x_i, y_i) \leftrightarrow (u_i, v_i)$ gives two equations, so **four pairs are enough** to determine $\mathbf{H}$ in principle. In practice we use many noisy pairs and solve

$$
\mathbf{A}\,\mathbf{h} = \mathbf{0}
$$

with SVD — taking the singular vector with the smallest singular value as $\mathbf{h}$. This is the **Direct Linear Transform (DLT)**, and it is the prototype of every projective estimation algorithm in this chapter.

![A skewed photograph of a planar surface rectified to a frontal view by H.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/17-linear-algebra-in-computer-vision/fig4_homography.png)

```python
src = np.float32([[100,100],[200,100],[200,200],[100,200]])
dst = np.float32([[120, 90],[220,110],[210,220],[ 90,210]])
H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
# corrected = cv2.warpPerspective(img, H, (W, H))
```

**Where this shows up.** Panorama stitching (one $\mathbf{H}$ per image pair when the camera only rotates), document scanners (rectifying skewed photos of paper), augmented reality (placing a virtual object on a tracked planar marker), top-down "bird's-eye" views in autonomous driving.

---

## 5 The Pinhole Camera and Its Projection Matrix

### Pinhole geometry

The simplest camera model imagines a tiny aperture at the origin and a virtual image plane at distance $f$ (the focal length). A 3D point $(X, Y, Z)$ in the camera frame projects to the image plane at

$$
u = f\,\frac{X}{Z}, \qquad v = f\,\frac{Y}{Z}.
$$

The division by depth $Z$ is the source of perspective: distant objects project to smaller image coordinates.

### Intrinsics: from millimetres to pixels

Real sensors do not have unit pixels and the principal point need not coincide with the optical axis. Stuffing those constants into a matrix gives the **camera intrinsic matrix**

$$
\mathbf{K} =
\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix},
$$

where $f_x, f_y$ are focal lengths in pixels, $(c_x, c_y)$ is the principal point, and $s$ is sensor skew (essentially zero on modern cameras).

### Extrinsics and the full projection

The world frame is generally different from the camera frame, related by a rigid motion: rotate by $\mathbf{R}$, then translate by $\mathbf{t}$. Stacking these together,

$$
\lambda
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
=
\mathbf{K}\,[\mathbf{R}\,|\,\mathbf{t}]
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
= \mathbf{P}\,\mathbf{X}_w,
$$

where $\mathbf{P} \in \mathbb{R}^{3 \times 4}$ is the **projection matrix** and $\lambda$ absorbs the scalar coming from perspective division. Eleven parameters live in $\mathbf{P}$: five intrinsics plus six pose parameters (three rotation angles, three translations).

![Pinhole camera projection: 3D world points become 2D pixels via P = K[R|t].](./17-linear-algebra-in-computer-vision/fig5_camera_projection.png)

```python
def project(X_world, K, R, t):
    Xc = X_world @ R.T + t                 # world -> camera
    x  = Xc[:, 0] / Xc[:, 2]                # perspective division
    y  = Xc[:, 1] / Xc[:, 2]
    u  = K[0, 0] * x + K[0, 2]
    v  = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)
```

### Calibration and Zhang's method

Estimating $\mathbf{K}$ and the lens distortion coefficients from images of a known checkerboard at several poses is **camera calibration**. Zhang's algorithm (1999) reduces it to estimating one homography per board image, extracting linear constraints on $\mathbf{K}^{-\top}\mathbf{K}^{-1}$ from the orthonormality of a rotation matrix, and refining with a small nonlinear optimisation. OpenCV's `calibrateCamera` does the whole pipeline in one call.

---

## 6 Two Views: Epipolar Geometry

### The epipolar constraint

Look at the same 3D point through two cameras. If it lands at $\mathbf{x}_1$ in image 1 and $\mathbf{x}_2$ in image 2, then the two pixels are not free — the four points (3D point, two camera centres, two images) are coplanar in space. Algebraically that coplanarity is

$$
\mathbf{x}_2^\top\,\mathbf{F}\,\mathbf{x}_1 = 0,
$$

where $\mathbf{F}$ is the $3 \times 3$ **fundamental matrix**. It depends only on the relative pose of the two cameras, not on the scene. From this constraint, given a point $\mathbf{x}_1$ in image 1, the corresponding point in image 2 must lie on the **epipolar line** $\boldsymbol{\ell}_2 = \mathbf{F}\,\mathbf{x}_1$. Stereo matching reduces from a 2D search to a 1D search.

### Essential vs fundamental

When the cameras are calibrated and we work in normalised coordinates $\hat{\mathbf{x}} = \mathbf{K}^{-1}\mathbf{x}$, the same constraint becomes the **essential matrix**:

$$
\hat{\mathbf{x}}_2^\top\,\mathbf{E}\,\hat{\mathbf{x}}_1 = 0, \qquad \mathbf{F} = \mathbf{K}_2^{-\top}\,\mathbf{E}\,\mathbf{K}_1^{-1}.
$$

The essential matrix has the special form $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$, where $[\mathbf{t}]_\times$ is the skew-symmetric matrix of the translation vector. Decomposing $\mathbf{E}$ via SVD recovers the **relative camera pose** $(\mathbf{R}, \mathbf{t})$ up to a sign and the unknowable absolute scale of monocular vision.

```python
F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
E, _ = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
```

---

## 7 3D Reconstruction: Triangulation, SfM, and Bundle Adjustment

### Triangulation as a linear system

Given two known projection matrices $\mathbf{P}_1, \mathbf{P}_2$ and a corresponding pair $\mathbf{x}_1 \leftrightarrow \mathbf{x}_2$, each image equation $\lambda_i \mathbf{x}_i = \mathbf{P}_i \mathbf{X}$ contributes two linear constraints on the 3D point $\mathbf{X}$ after eliminating the unknown depth $\lambda_i$. Stacking gives a $4 \times 4$ linear system $\mathbf{A}\mathbf{X} = \mathbf{0}$, solved by SVD: $\mathbf{X}$ is the right singular vector with the smallest singular value, and the 3D coordinates are recovered after the homogeneous division.

```python
X4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
X3 = (X4[:3] / X4[3]).T
```

### Structure from Motion

**SfM** scales triangulation up to many views. The standard incremental pipeline is:

1. **Detect and match features** (SIFT, ORB) between every pair of overlapping images.
2. **Initialise** with two views: estimate $\mathbf{E}$ via the 5-point algorithm + RANSAC, recover relative pose, triangulate an initial point cloud.
3. **Grow** by adding one image at a time: solve **PnP** (Perspective-n-Point) to register the new camera, then triangulate any newly observable 3D points.
4. **Refine globally** with bundle adjustment.

### Bundle adjustment

The flagship optimisation problem of multi-view geometry: jointly refine **all** camera parameters $\{\mathbf{P}_i\}$ and **all** 3D points $\{\mathbf{X}_j\}$ to minimise the total reprojection error,

$$
\min_{\{\mathbf{P}_i\},\,\{\mathbf{X}_j\}} \;\sum_{(i, j) \in \mathcal{V}} \rho\!\left(\lVert \mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j)\rVert^2\right),
$$

where $\pi$ is the projection function and $\rho$ is a robust kernel (Huber) for outliers. This is a giant nonlinear least-squares problem — millions of variables in modern reconstructions — but the Jacobian is **block-sparse**: every observation involves exactly one camera and one point. Levenberg-Marquardt with the Schur complement exploits that sparsity and makes the problem solvable on a laptop.

---

## 8 SLAM: Linear Algebra in Real-Time

### The SLAM problem

A robot moves through an unknown environment, reading sensors as it goes. **SLAM** asks it to simultaneously estimate its own trajectory and a map of landmarks. Modern SLAM systems are essentially online bundle adjustment, with two distinctive linear-algebraic ingredients.

### Lie groups for rotations

A 3D rigid-body pose lives in the matrix group

$$
\mathbf{T} = \begin{bmatrix}\mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1\end{bmatrix} \in SE(3),
$$

with $\mathbf{R} \in SO(3)$. Optimising $\mathbf{R}$ as nine numbers is awkward because we must enforce $\mathbf{R}^\top\mathbf{R} = \mathbf{I}$. The **Lie algebra** $\mathfrak{se}(3) \cong \mathbb{R}^6$ is an unconstrained vector space, and the exponential map $\mathbf{T} = \exp(\boldsymbol{\xi}^\wedge)$ moves between them. We do gradient steps in $\mathbb{R}^6$ and lift back to $SE(3)$ — clean, constraint-free optimisation.

```python
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def so3_exp(phi):                    # Rodrigues' formula
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3)
    a = phi / theta; A = skew(a)
    return np.eye(3) + np.sin(theta) * A + (1 - np.cos(theta)) * (A @ A)
```

### Pose-graph optimisation

Modern SLAM (g2o, GTSAM, Ceres) models the problem as a **factor graph**: nodes are poses and landmarks, edges are noisy relative measurements with covariance $\boldsymbol{\Omega}_k$. We minimise

$$
\min \;\sum_k \lVert \mathbf{e}_k\rVert^2_{\boldsymbol{\Omega}_k},
$$

via Gauss-Newton iterations, each of which solves the sparse linear system

$$
\mathbf{H}\,\Delta\mathbf{x} = -\mathbf{b},
$$

where $\mathbf{H} = \mathbf{J}^\top \boldsymbol{\Omega}\,\mathbf{J}$ inherits the graph's sparsity. Sparse Cholesky on $\mathbf{H}$ is the inner loop of essentially every modern SLAM stack.

---

## 9 Image Filtering as Matrix Multiplication

![Edge-detect, blur, and sharpen kernels with their input/output pairs.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/17-linear-algebra-in-computer-vision/fig2_convolution_kernels.png)

### Convolution is a (huge, sparse, structured) linear map

A 2D convolution $\mathbf{G} = \mathbf{I} * \mathbf{K}$ is, when written in vector form, a matrix multiplication $\mathbf{g} = \mathbf{T}\mathbf{i}$ where $\mathbf{T}$ is a **doubly block Toeplitz** matrix built from the kernel. We never form $\mathbf{T}$ explicitly — it would have $(HW)^2$ entries — but its existence justifies analysing convolutions with linear-algebraic tools (eigenvalues of $\mathbf{T}$ are precisely the discrete Fourier transform of the kernel).

### Three kernels you should know by heart

**Box / mean (low-pass, blur).** Averages a neighbourhood; the weights sum to one so brightness is preserved.

$$
\mathbf{K}_{\text{blur}} = \tfrac{1}{9}\begin{bmatrix}1 & 1 & 1\\ 1 & 1 & 1\\ 1 & 1 & 1\end{bmatrix}
$$

**Laplacian (band-pass, edges).** A discrete second derivative; weights sum to zero, so flat regions vanish and discontinuities light up.

$$
\mathbf{K}_{\text{edge}} = \begin{bmatrix}0 & -1 & 0\\ -1 & 4 & -1\\ 0 & -1 & 0\end{bmatrix}
$$

**Sharpen (high-pass + identity).** "Original plus its own edges":

$$
\mathbf{K}_{\text{sharp}} = \begin{bmatrix}0 & -1 & 0\\ -1 & 5 & -1\\ 0 & -1 & 0\end{bmatrix} = \mathbf{I} + \mathbf{K}_{\text{edge}}.
$$

### Convolution theorem

Every linear translation-invariant filter is diagonalised by the Fourier basis: spatial convolution becomes pointwise multiplication in frequency,

$$
\mathcal{F}[\mathbf{I} * \mathbf{K}] = \mathcal{F}[\mathbf{I}] \cdot \mathcal{F}[\mathbf{K}].
$$

Designing a filter becomes shaping its frequency response: low-pass for noise removal, high-pass for edges, band-pass for textures.

---

## 10 Two Vision Algorithms That Are Just $2 \times 2$ Eigenvalue Problems

### Harris corners and the structure tensor

Around a pixel, summarise local image gradients with the **structure tensor**

$$
\mathbf{M} = \sum_{(x, y) \in W} w(x, y)
\begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix},
$$

a $2 \times 2$ symmetric positive-semidefinite matrix. Its eigenvalues $\lambda_1 \ge \lambda_2 \ge 0$ describe how the image varies in the two principal directions of that window:

| eigenvalues | meaning |
| --- | --- |
| both small | flat region |
| one large, one small | edge |
| both large | corner |

Harris's elegant trick avoids computing the eigenvalues directly:

$$
R = \det(\mathbf{M}) - k\,\mathrm{trace}(\mathbf{M})^2 = \lambda_1\lambda_2 - k(\lambda_1 + \lambda_2)^2,
$$

a corner score that is large only when both eigenvalues are large.

```python
def harris(img, k=0.04, win=2, ksize=3):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    Sxx = cv2.GaussianBlur(Ix * Ix, (2 * win + 1,) * 2, 0)
    Syy = cv2.GaussianBlur(Iy * Iy, (2 * win + 1,) * 2, 0)
    Sxy = cv2.GaussianBlur(Ix * Iy, (2 * win + 1,) * 2, 0)
    detM, trM = Sxx * Syy - Sxy ** 2, Sxx + Syy
    return detM - k * trM ** 2
```

### Optical flow: the same matrix again

Optical flow estimates the per-pixel displacement $(u, v)$ between two consecutive frames. Assuming brightness is conserved, a first-order Taylor expansion gives the **brightness constancy equation**

$$
I_x\,u + I_y\,v + I_t = 0,
$$

one scalar equation in two unknowns — locally underdetermined (the **aperture problem**). The Lucas-Kanade fix: assume $(u, v)$ is constant in a small window, write one equation per pixel, and solve the resulting overdetermined system in least squares:

$$
\underbrace{\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}}_{\text{exactly the structure tensor } \mathbf{M}}
\begin{bmatrix} u \\ v \end{bmatrix}
= -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}.
$$

The system is well-posed precisely when $\mathbf{M}$ is well-conditioned — that is, **at corners**. The same eigenvalue analysis tells Harris where corners live and tells Lucas-Kanade where it can trust its flow estimate.

![Optical flow: per-pixel displacement vectors estimated by local least squares.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/17-linear-algebra-in-computer-vision/fig7_optical_flow.png)

---

## 11 SVD for Image Compression

An $H \times W$ grayscale image has SVD

$$
\mathbf{I} = \sum_{i=1}^{r} \sigma_i\,\mathbf{u}_i\,\mathbf{v}_i^\top, \qquad \sigma_1 \ge \sigma_2 \ge \cdots \ge 0.
$$

The **Eckart-Young theorem** (Chapter 9) says the truncation to the first $k$ terms is the best rank-$k$ approximation under both Frobenius and spectral norms. For natural images the singular values decay rapidly — most of the visual energy lives in the first few dozen components — so a tiny $k$ already produces a recognisable picture.

Storage drops from $HW$ to $k(H + W + 1)$ numbers, a compression ratio of roughly $k(H + W) / (HW)$. SVD is nowhere near JPEG efficiency on natural images (JPEG exploits perceptual redundancy in DCT coefficients), but it is conceptually clean and the gold standard for low-rank approximation in many other corners of vision: face recognition (Eigenfaces), background subtraction (RPCA), denoising.

![SVD low-rank reconstruction at three ranks plus the singular value spectrum.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/17-linear-algebra-in-computer-vision/fig6_svd_compression.png)

---

## 12 Linear Algebra in Modern Deep Vision

### Convolution as matrix multiplication via im2col

GPUs are extraordinarily good at dense matrix multiplication. The **im2col** trick rewrites every convolution as one giant `gemm`: extract every receptive-field patch as a column, stack the kernels as rows, multiply once. The result is reshaped back to a feature map. Every CNN is, mechanically, a sequence of matrix multiplications interleaved with elementwise nonlinearities.

### Self-attention is matrix multiplication

The Transformer's self-attention layer is

$$
\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V},
$$

where $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ are linear projections of the token sequence. Vision Transformers (ViT) tokenise an image into patches and feed them through this mechanism — the entire forward pass is a tower of matrix products.

### Batch normalisation is an affine map

At inference time, batch norm with stored running statistics and learned $(\gamma, \beta)$ is just an elementwise affine transform $y = \gamma\,\hat{x} + \beta$. It can be folded into the preceding convolution at deployment time, removing it from the compute graph entirely.

---

## Exercises

### Basics

1. Write matrix expressions for (a) vertical flip, (b) horizontal flip, (c) brightness $+50$, (d) contrast $\times 1.5$ on a grayscale image $\mathbf{I}$.
2. Prove for 2D rotation matrices: (a) $\det \mathbf{R} = 1$, (b) $\mathbf{R}^{-1} = \mathbf{R}^\top$, (c) $\mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha + \beta)$.
3. Express in $3 \times 3$ homogeneous form and combine into a single matrix: translate by $(2, 3)$, then rotate $45^\circ$ around the origin, then scale by $2$.

### Intermediate

4. Why do four point correspondences suffice (in principle) to estimate a homography? Set up the $\mathbf{A}\mathbf{h} = \mathbf{0}$ system that the DLT algorithm solves.
5. Show that the essential matrix factorises as $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$. Use the SVD $\mathbf{E} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top$ to recover $\mathbf{R}$ and $\mathbf{t}$ up to sign.
6. Given two projection matrices and a corresponding image-point pair, derive the linear system whose solution is the triangulated 3D point.

### Programming

7. Implement affine warping from scratch, building the $3 \times 3$ matrix and applying inverse-warping with bilinear interpolation.
8. Implement the **8-point algorithm** for estimating $\mathbf{F}$, including normalisation and rank-2 enforcement via SVD.
9. Implement Harris corner detection end-to-end with non-maximum suppression.

### Application

10. Design a panorama stitcher. Why is a homography sufficient when the camera only rotates? How would you handle accumulated drift over many images and exposure differences?
11. Implement a monocular visual odometry loop: extract features, match between frames, estimate $\mathbf{E}$, recover relative pose, accumulate trajectory. Discuss scale drift and how to detect tracking failure.

---

## Chapter Summary

Computer vision and linear algebra are inseparable.

**Representation.** Grayscale images are matrices; colour images are 3-channel tensors; flattened images are vectors in $\mathbb{R}^{HW}$.

**Geometry.** Rotation, scaling, and shear are linear maps; homogeneous coordinates promote translation to a linear operation; perspective requires a $3 \times 3$ homography with a non-trivial bottom row.

**Cameras and 3D.** A pinhole camera is the $3 \times 4$ projection matrix $\mathbf{P} = \mathbf{K}[\mathbf{R}\,|\,\mathbf{t}]$. Two-view geometry is encoded in the fundamental matrix $\mathbf{F}$. Triangulation, structure from motion, and bundle adjustment are linear (or sparse nonlinear) least squares.

**State estimation.** SLAM optimises poses on the manifold $SE(3)$ via Lie algebra; pose-graph optimisation is a sparse Cholesky in the inner loop.

**Filtering and features.** Convolution is a giant block-Toeplitz multiplication. Harris corners and Lucas-Kanade flow are the same $2 \times 2$ eigenvalue analysis applied to the structure tensor.

**Compression and learning.** SVD gives optimal low-rank approximations of images. Deep CNNs and Transformers are towers of matrix multiplications, with batch norm reducing to a deployment-time affine map.

Once you internalise this list, the field's papers stop looking like a heap of unrelated algorithms and start looking like creative re-uses of a small linear-algebraic toolkit.

---

## References

- Hartley, R., & Zisserman, A. *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press, 2004.
- Szeliski, R. *Computer Vision: Algorithms and Applications* (2nd ed.). Springer, 2022.
- Forsyth, D. A., & Ponce, J. *Computer Vision: A Modern Approach* (2nd ed.). Pearson, 2011.
- Strang, G. *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press, 2016.
- Barfoot, T. D. *State Estimation for Robotics*. Cambridge University Press, 2017.
- Zhang, Z. "A flexible new technique for camera calibration." *IEEE TPAMI* 22(11), 2000.
- Lucas, B., & Kanade, T. "An iterative image registration technique with an application to stereo vision." *IJCAI*, 1981.
- Harris, C., & Stephens, M. "A combined corner and edge detector." *Alvey Vision Conference*, 1988.

---

## Series Navigation

- **Previous:** [Chapter 16: Linear Algebra in Deep Learning](/en/chapter-16-linear-algebra-in-deep-learning/)
- **Next:** [Chapter 18: Frontiers and Summary](/en/chapter-18-frontiers-and-summary/)
- **Full Series:** Essence of Linear Algebra (1--18)
