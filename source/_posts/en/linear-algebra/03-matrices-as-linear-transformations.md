---
title: "Matrices as Linear Transformations"
date: 2025-01-23 09:00:00
tags:
  - Linear Algebra
  - Matrices
  - Linear Transformations
  - Rotation
  - Scaling
description: "Matrices are not tables of numbers -- they are machines that transform space. This chapter shows you how to see rotation, scaling, shearing, reflection, and projection as matrices, and why matrix multiplication means composing transformations."
categories: Linear Algebra
series:
  name: "Linear Algebra"
  part: 3
  total: 18
lang: en
mathjax: true
disableNunjucks: true
---

## The Big Idea

Open a traditional textbook and matrices show up as "rectangular arrays of numbers." You learn rules for adding and multiplying them, but no one explains *why* the multiplication rule looks the way it does, or why $AB \neq BA$ in general.

Here is the secret the symbol-pushing version hides: **a matrix is a function that transforms space.** Every $m \times n$ matrix is a machine that eats an $n$-dimensional vector and spits out an $m$-dimensional one. Once you can *see* that, the strange rules stop being strange. They are simply the bookkeeping for what happens to the basis vectors.

In this chapter we will:

- Define a linear transformation and develop the geometric "fingerprint" of one.
- Show that **a matrix is fully determined by where it sends the basis vectors.**
- Build a visual gallery of rotations, scalings, shears, reflections, and projections.
- Prove that **matrix multiplication is composition of transformations** -- which immediately explains non-commutativity.
- Talk about inverses (undoing a transformation), kernel and image, and what happens when a matrix is *singular*.

**Prerequisites.** Chapter 1 (vectors, addition, scalar multiplication) and Chapter 2 (linear independence, basis, span).

---

## 1. A Matrix Is a Function

Forget for a moment that a matrix is "an array of numbers." A matrix $A$ is a **function**

$$
A: \mathbb{R}^{n} \longrightarrow \mathbb{R}^{m}, \qquad \vec{v} \mapsto A\vec{v}.
$$

You feed it a vector, you get a vector back. The whole job of this chapter is to describe what kind of function this can be.

**Photocopier analogy.** A photocopier with a zoom dial at 150% maps every point of the original to a point 1.5 times farther from the centre. That is one specific transformation -- uniform scaling. Matrices represent a much larger zoo of similar "geometric machines": rotation, shear, reflection, projection, and any combination of them.

### What Counts as "Linear"?

Matrices are not arbitrary functions. They are exactly the functions that respect the two operations from Chapter 1: vector addition and scalar multiplication. Formally, a transformation $T$ is **linear** when

$$
T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v}) \qquad \text{(additivity)}
$$

$$
T(c\,\vec{v}) = c\,T(\vec{v}) \qquad \text{(homogeneity)}
$$

These two algebraic rules have a striking *geometric* fingerprint, easiest to see by looking at how the integer grid is allowed to deform.

![Standard basis and the unit grid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig1_unit_grid.png)

A linear transformation is allowed to do anything to the picture above provided that:

- **The origin stays put.** $T(\vec{0}) = \vec{0}$. (Plug $c = 0$ into homogeneity.)
- **Straight lines stay straight.** No bending, no curving.
- **Parallel lines stay parallel and evenly spaced.** The deformed grid is still a grid -- it may be tilted, stretched, or even collapsed, but every cell is still a parallelogram of the same shape.

**Rubber-sheet analogy.** Imagine the picture is drawn on a rubber sheet pinned at the origin. You can stretch it, rotate it, shear it, or even flatten it, but you cannot tear it or fold it. Anything you can do to that sheet is a linear transformation; anything that requires tearing or folding is not.

### What Is *Not* Linear

It is just as useful to know what falls outside this club:

- **Translation.** $T(\vec{v}) = \vec{v} + \vec{b}$ moves the origin, so it fails $T(\vec{0}) = \vec{0}$. Translation is *affine*, not linear. (We will rescue it later with homogeneous coordinates.)
- **Bending.** Anything that turns a straight line into a curve.
- **Squaring or multiplying components.** Functions like $T(x, y) = (x^{2}, y)$ or $T(x, y) = (xy, y)$ break additivity.

---

## 2. The Key Insight: Columns Are Where the Basis Vectors Land

This is the single most important sentence in the chapter. Read it twice:

> A matrix is fully determined by where it sends the basis vectors. The columns of the matrix are exactly those landing spots.

Let me show why. In $\mathbb{R}^{2}$ the standard basis is $\hat{\imath} = (1, 0)$ and $\hat{\jmath} = (0, 1)$. Every vector decomposes as

$$
\vec{v} = \begin{pmatrix} x \\ y \end{pmatrix} = x\,\hat{\imath} + y\,\hat{\jmath}.
$$

If a transformation $T$ is linear, additivity and homogeneity force

$$
T(\vec{v}) = T(x\,\hat{\imath} + y\,\hat{\jmath}) = x\,T(\hat{\imath}) + y\,T(\hat{\jmath}).
$$

So the moment we know **two vectors** -- where $\hat{\imath}$ lands and where $\hat{\jmath}$ lands -- we know what $T$ does to *every* vector in the plane. Stack those two landing spots as the columns of a matrix:

$$
A = \Big[\;T(\hat{\imath})\;\Big|\;T(\hat{\jmath})\;\Big] = \begin{pmatrix} a & b \\ c & d \end{pmatrix},
$$

and the formula $A\vec{v} = x\,T(\hat{\imath}) + y\,T(\hat{\jmath})$ is exactly the matrix-vector product:

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix}
= x\begin{pmatrix} a \\ c \end{pmatrix} + y\begin{pmatrix} b \\ d \end{pmatrix}
= \begin{pmatrix} ax + by \\ cx + dy \end{pmatrix}.
$$

Read the right-hand side as a *recipe*: matrix-vector multiplication is "linear combination of the columns of $A$, with weights coming from $\vec{v}$." Once you internalise this, you stop computing matrix products and start *reading* them.

### Worked Example

Suppose $T$ sends $\hat{\imath} \mapsto (2, 1)$ and $\hat{\jmath} \mapsto (-1, 3)$. Then

$$
A = \begin{pmatrix} 2 & -1 \\ 1 & 3 \end{pmatrix}, \qquad
A\begin{pmatrix} 3 \\ 2 \end{pmatrix}
= 3\begin{pmatrix} 2 \\ 1 \end{pmatrix} + 2\begin{pmatrix} -1 \\ 3 \end{pmatrix}
= \begin{pmatrix} 4 \\ 9 \end{pmatrix}.
$$

No memorised formula required -- just "three copies of column 1 plus two copies of column 2."

---

## 3. A Visual Gallery of 2D Transformations

To build intuition, we will look at five transformations the same way: keep the original unit grid on the left, draw the deformed grid on the right, and mark $\hat{\imath}$ in **blue**, $\hat{\jmath}$ in **purple**. The unit square (in green) shows what happens to area.

### 3.1 Rotation

Track $\hat{\imath}$ around the unit circle: at angle $\theta$ counter-clockwise it lands at $(\cos\theta, \sin\theta)$. Similarly $\hat{\jmath}$ goes from angle $90^{\circ}$ to $90^{\circ} + \theta$, landing at $(-\sin\theta, \cos\theta)$. Stack them as columns:

$$
R_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}.
$$

![Rotation by 30 degrees](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig2_rotation.png)

A few special angles worth memorising:

| Angle | Matrix | Effect |
|-------|--------|--------|
| $90^{\circ}$ | $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ | Quarter turn CCW |
| $180^{\circ}$ | $\begin{pmatrix}-1&0\\0&-1\end{pmatrix}$ | Half turn (= negation) |
| $-90^{\circ}$ | $\begin{pmatrix}0&1\\-1&0\end{pmatrix}$ | Quarter turn CW |

**Game programming aside.** Every "turn left" key press in a 2D game multiplies the player's facing vector by $R_{\Delta\theta}$. Rotations preserve length and angle, so the character does not stretch when it spins.

### 3.2 Scaling

To stretch by $s_{x}$ along $x$ and $s_{y}$ along $y$, just send $\hat{\imath} \mapsto (s_{x}, 0)$ and $\hat{\jmath} \mapsto (0, s_{y})$:

$$
S = \begin{pmatrix} s_{x} & 0 \\ 0 & s_{y} \end{pmatrix}.
$$

![Scaling: x by 2, y by 1.5](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig3_scaling.png)

The unit square turns into an $s_{x} \times s_{y}$ rectangle, so its area is multiplied by $s_{x} s_{y}$. That product -- the **determinant** -- is the headline of Chapter 4.

Resizing an image in Photoshop is exactly this: every pixel coordinate is multiplied by a diagonal $S$.

### 3.3 Shear

A horizontal shear sends $\hat{\imath} \mapsto (1, 0)$ unchanged and $\hat{\jmath} \mapsto (k, 1)$ -- it slides the top of the unit square sideways:

$$
H = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}.
$$

![Horizontal shear](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig4_shear.png)

The new $x$ coordinate is $x + ky$, so the higher up a point sits, the more it slides to the right. Italic text is the canonical example: bottoms of letters stay put, tops lean. Wind blowing tall grass is a vertical shear.

Notice that $\det H = 1$: shears never change area, even though they distort shape dramatically.

### 3.4 Reflection

Reflection flips one coordinate. The matrix is again "where do the basis vectors land?":

| Reflection | Matrix | What happens |
|-----------|--------|--------------|
| About $x$-axis | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ | Flip top--bottom |
| About $y$-axis | $\begin{pmatrix}-1&0\\0&1\end{pmatrix}$ | Flip left--right |
| About $y = x$ | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ | Swap $x$ and $y$ |
| About origin | $\begin{pmatrix}-1&0\\0&-1\end{pmatrix}$ | Equals $180^{\circ}$ rotation |

A reflection has $\det = -1$: area is preserved, but orientation is reversed (right-handed becomes left-handed). That sign is what distinguishes a reflection from a rotation.

### 3.5 Projection

Projection onto the $x$-axis sends $\hat{\imath} \mapsto (1, 0)$ and crushes $\hat{\jmath} \mapsto (0, 0)$:

$$
P_{x} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.
$$

The whole 2D plane gets flattened onto a 1D line. This is the cast-shadow transformation, with the "sun" directly overhead. More generally, projection onto the line through the origin in unit-vector direction $\vec{u}$ is $P = \vec{u}\vec{u}^{\!\top}$.

Projections are the first transformations that **lose information** -- both $(1, 2)$ and $(1, 99)$ project to $(1, 0)$. We will see in Section 6 that this is exactly what makes them non-invertible. The `fig7` figure further down shows the same phenomenon for any singular matrix.

### Summary Table

| Transformation | Matrix | Determinant | Invertible? |
|----------------|--------|-------------|-------------|
| Rotation by $\theta$ | $\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$ | $+1$ | Yes |
| Scaling $(s_{x}, s_{y})$ | $\operatorname{diag}(s_{x}, s_{y})$ | $s_{x}s_{y}$ | Yes if both nonzero |
| Horizontal shear | $\begin{pmatrix}1&k\\0&1\end{pmatrix}$ | $+1$ | Yes |
| Reflection (any axis) | various | $-1$ | Yes (it's its own inverse) |
| Projection onto a line | rank 1 | $0$ | **No** |

---

## 4. Matrix Multiplication = Composition of Transformations

We can finally explain the multiplication rule.

**Setup.** Apply $A$ first, then $B$:

$$
\vec{v} \;\xrightarrow{A}\; A\vec{v} \;\xrightarrow{B}\; B(A\vec{v}).
$$

We want a *single* matrix that does both steps in one go. Whatever it is, it must equal $B(A\vec{v})$ for every $\vec{v}$. The multiplication rule is *defined* so that this single matrix is exactly $BA$:

$$
B(A\vec{v}) \;\equiv\; (BA)\vec{v}.
$$

That identity is not a theorem you prove; it is the **design specification** of matrix multiplication. The strange "row times column" arithmetic is just what falls out when you work through "where does $\hat{\imath}$ go after $A$ and then $B$?"

**Right-to-left reading rule.** In an expression like $CBA\vec{v}$, the first matrix to act is the *rightmost* one. Read it as "first $A$, then $B$, then $C$." This is the opposite of how English flows, and it is the source of countless bugs in graphics code -- so it is worth saying out loud whenever you write it.

### Why Order Matters: A Picture

Take $A = R_{45^{\circ}}$ (rotate 45 degrees) and $B = \operatorname{diag}(2, 1)$ (stretch $x$ by 2). Apply them in the order "$A$ then $B$":

![Matrix multiplication as composition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig5_composition.png)

You can read the third panel three different ways and they all agree:

1. **Geometrically.** Rotate the unit square 45 degrees, then stretch the rotated picture horizontally.
2. **Algebraically.** $BA = \begin{pmatrix}2&0\\0&1\end{pmatrix}\begin{pmatrix}\frac{\sqrt{2}}{2}&-\frac{\sqrt{2}}{2}\\\frac{\sqrt{2}}{2}&\frac{\sqrt{2}}{2}\end{pmatrix} = \begin{pmatrix}\sqrt{2}&-\sqrt{2}\\\frac{\sqrt{2}}{2}&\frac{\sqrt{2}}{2}\end{pmatrix}$.
3. **Column reading.** The first column of $BA$ is $B$ applied to the first column of $A$, i.e. $B$ applied to $A\hat{\imath}$, i.e. where $\hat{\imath}$ ends up after both transformations. Same for the second column.

Now reverse the order: "$B$ first, then $A$."

$$
A B = \begin{pmatrix}\frac{\sqrt{2}}{2}&-\frac{\sqrt{2}}{2}\\\frac{\sqrt{2}}{2}&\frac{\sqrt{2}}{2}\end{pmatrix}\begin{pmatrix}2&0\\0&1\end{pmatrix} = \begin{pmatrix}\sqrt{2}&-\frac{\sqrt{2}}{2}\\\sqrt{2}&\frac{\sqrt{2}}{2}\end{pmatrix} \neq BA.
$$

Geometrically: stretching first turns the square into a wide rectangle, and rotating that rectangle gives a different parallelogram than rotating-then-stretching. **Matrix multiplication is not commutative because composition of transformations is not commutative.** That is the whole story.

### Associativity Saves Computation

Multiplication *is* associative: $(AB)C = A(BC)$. This is not just an algebraic curiosity -- it is what makes 3D graphics fast.

In a typical scene, every vertex is scaled, rotated, and translated:

$$
M = T \cdot R \cdot S.
$$

If you have a million vertices, you do **not** apply $S$, then $R$, then $T$ to each vertex individually (that would be $3$ million matrix-vector multiplies). You compute $M$ once, then do $M\vec{v}$ a million times -- a $3\times$ speedup, plus better numerical conditioning. GPU pipelines are built around this idea.

---

## 5. Identity and Inverse: Doing Nothing, and Undoing

### The Identity

The transformation that does nothing -- everything maps to itself -- corresponds to the **identity matrix**

$$
I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}.
$$

Geometrically, the after-grid is identical to the before-grid:

![Identity transformation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig6_identity.png)

It satisfies $IA = AI = A$ for every matrix $A$, the matrix analogue of "multiplying by 1."

### The Inverse

If $A$ represents a transformation, the **inverse** $A^{-1}$ undoes it:

$$
A^{-1} A \;=\; A A^{-1} \;=\; I.
$$

Examples that are obvious once you think geometrically:

- $R_{\theta}^{-1} = R_{-\theta}$ (rotate the other way).
- $\operatorname{diag}(s_{x}, s_{y})^{-1} = \operatorname{diag}(1/s_{x}, 1/s_{y})$, provided neither factor is zero.
- Every reflection is its own inverse: reflect twice and you are back where you started.

### When Does an Inverse Exist?

Not every transformation can be undone. Projection lost information by collapsing the $y$-direction; **any transformation that loses a dimension cannot be inverted**. Both $(1, 2)$ and $(1, 5)$ project to $(1, 0)$, so given only $(1, 0)$ you cannot recover the original $y$.

The clean criterion -- which we will earn properly in Chapter 4 -- is:

> $A$ is invertible $\iff$ $A$ does not collapse any dimension $\iff$ $\det(A) \neq 0$.

### The $2\times 2$ Inverse Formula

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ with $\det(A) = ad - bc \neq 0$:

$$
A^{-1} = \frac{1}{ad - bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}.
$$

(Confirm by multiplying it out -- you should get $I$.)

---

## 6. Singular Matrices: When the Plane Collapses

This is where the geometric viewpoint really earns its keep. Consider the matrix

$$
S = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}.
$$

The second row is twice the first; the second column is twice the first. Algebraically, $\det(S) = 1 \cdot 4 - 2 \cdot 2 = 0$. Geometrically:

![Singular matrix collapses 2D to 1D](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/03-matrices-as-linear-transformations/fig7_singular.png)

Every point in the plane is sent onto the single line $y = 2x$ -- the line spanned by the column $(1, 2)$. The unit square is crushed to a line segment of zero area. The orange "after" grid lines all coincide with that line because every input gets mapped onto it.

This is a **singular** (non-invertible) matrix, and there are now two interesting subspaces to give names to.

### Kernel: What Gets Crushed

The **kernel** (or **null space**) of $A$ is the set of vectors that get sent to zero:

$$
\ker(A) = \{\,\vec{v} : A\vec{v} = \vec{0}\,\}.
$$

For our $S$, vectors of the form $\vec{v} = t\,(2, -1)$ satisfy $S\vec{v} = \vec{0}$ -- this is the dashed grey "kernel direction" in the figure. The kernel is a 1D line.

Intuition: the kernel is the set of *directions the matrix annihilates*. A matrix is invertible iff its kernel is just $\{\vec{0}\}$.

### Image: What Can Be Reached

The **image** (or **range**) is the set of all possible outputs:

$$
\operatorname{Im}(A) = \{\,A\vec{v} : \vec{v} \in \mathbb{R}^{n}\,\}.
$$

For $S$, the image is the line $y = 2x$. This is also called the **column space** because it is exactly the span of the columns of $A$. The two columns of $S$ are $(1, 2)$ and $(2, 4)$ -- both lie on the same line, so they span only a line, not a plane.

### The Rank-Nullity Theorem

The two subspaces always satisfy a bookkeeping identity:

$$
\dim\ker(A) + \dim\operatorname{Im}(A) = n,
$$

where $n$ is the input dimension. Translation: every dimension the matrix collapses (kernel) is a dimension lost from the output (image). For our $S$: $1 + 1 = 2$. We will see this same equation again in Chapter 5 when we discuss linear systems.

---

## 7. Transformations in 3D and Beyond

Everything generalises cleanly. In $\mathbb{R}^{3}$ a linear transformation is a $3 \times 3$ matrix, and the three columns are where $\hat{\imath}$, $\hat{\jmath}$, $\hat{k}$ land.

**Rotation about the $z$-axis** (the $z$-axis is fixed; the $xy$ plane rotates):

$$
R_{z}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}.
$$

**Orthographic projection onto the $xy$ plane** (drop the $z$ component):

$$
P_{xy} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.
$$

This 3D projection is the higher-dimensional analogue of the singular matrix above: it collapses 3D space onto a 2D plane, so $\det = 0$ and it is non-invertible.

---

## 8. Python: Visualising Transformations

The script that produced every figure in this chapter lives at `scripts/figures/linear-algebra/03-matrices-as-linear-transformations.py`. Here is the minimal idea behind it -- a function that draws a matrix's effect on the unit square:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_transform(A, title="Transformation"):
    """Visualise how matrix A transforms the unit square."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T  # 2 x 5
    transformed = A @ square                                       # the magic line

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, shape, label, color in [
        (axes[0], square,      "Before",                     "#2563eb"),
        (axes[1], transformed, f"After (det={np.linalg.det(A):.2f})", "#10b981"),
    ]:
        ax.fill(shape[0], shape[1], alpha=0.3, color=color)
        ax.plot(shape[0], shape[1], color=color, linewidth=2)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_title(label)
    plt.suptitle(title); plt.tight_layout(); plt.show()

plot_transform(np.array([[2,   0],   [0,   1.5]]), "Scaling")
plot_transform(np.array([[1,   0.5], [0,   1  ]]), "Shear")
plot_transform(np.array([[-1,  0],   [0,   1  ]]), "Reflection about y-axis")
plot_transform(np.array([[1,   2],   [2,   4  ]]), "Singular: collapses to a line")
```

### A Tiny 2D Game Transform

Game engines combine scale, rotation, and translation using **homogeneous coordinates** (a 2D point $(x, y)$ is stored as $(x, y, 1)$, which lets a single $3 \times 3$ matrix handle translation too):

```python
import numpy as np

class Transform2D:
    """Combine scale, rotation, and translation as one 3x3 matrix."""

    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.rotation = 0.0           # radians
        self.scale    = np.array([1.0, 1.0])

    def matrix(self) -> np.ndarray:
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        S = np.diag([self.scale[0], self.scale[1], 1.0])
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
        T = np.array([[1, 0, self.position[0]],
                      [0, 1, self.position[1]],
                      [0, 0, 1]])
        # Read right-to-left: scale, then rotate, then translate.
        return T @ R @ S

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points)
        homo = np.ones((3, len(pts)))
        homo[:2, :] = pts.T
        return (self.matrix() @ homo)[:2, :].T

t = Transform2D()
t.position = np.array([100.0, 50.0])
t.rotation = np.pi / 4               # 45 degrees
t.scale    = np.array([2.0, 1.5])
print(t.apply([[-1, -1], [1, -1], [1, 1], [-1, 1]]))
```

This is exactly the "TRS" pattern (Translate * Rotate * Scale) used by Unity, Unreal, Godot, and every other 3D engine.

---

## Frequently Asked Questions

**Q: Is translation a linear transformation?**
A: No -- it moves the origin, so $T(\vec{0}) \neq \vec{0}$. To handle translation with matrices, computer graphics uses **homogeneous coordinates**: a 2D point $(x, y)$ becomes the 3D vector $(x, y, 1)$, and a translation by $(t_{x}, t_{y})$ becomes the $3 \times 3$ matrix that adds $t_{x}$ and $t_{y}$ to the first two coordinates. Translation is not linear in $\mathbb{R}^{2}$, but it *is* linear in $\mathbb{R}^{3}$ when we restrict attention to the plane $z = 1$.

**Q: Why is matrix multiplication defined this strange way?**
A: Because the rule was *engineered* to make "product of matrices = composition of transformations" true. You start by demanding $(BA)\vec{v} = B(A\vec{v})$, work out what each entry must be, and the row-times-column formula falls out. It is not arbitrary; it is forced.

**Q: Why are rotation matrices special?**
A: They preserve length and angle. Algebraically, $R^{\!\top} R = I$ (they are *orthogonal*) and $\det R = +1$ (they preserve orientation -- no flipping). Together these properties define the special orthogonal group $\mathrm{SO}(2)$, which is the mathematical object behind every smooth spinning animation you have ever seen.

**Q: How can I tell at a glance whether a matrix is a rotation, scaling, shear, or projection?**
A: Look at the determinant and the columns:
- $\det = +1$ and columns orthonormal $\Rightarrow$ rotation.
- $\det = -1$ and columns orthonormal $\Rightarrow$ reflection.
- Columns parallel to the axes $\Rightarrow$ axis-aligned scaling.
- $\det = 0$ $\Rightarrow$ singular: collapses dimension, no inverse exists.
- Otherwise probably a shear, a general scaling along non-axis directions, or a combination.

---

## Chapter Summary

The single sentence to remember:

> **A matrix is a linear transformation. Its columns tell you where the basis vectors land.**

Everything else cascades from there:

- Matrix-vector multiplication = linear combination of the columns of $A$.
- Matrix multiplication = composition of transformations (right to left).
- Non-commutativity comes from the fact that "rotate then stretch" is not the same as "stretch then rotate."
- The identity matrix does nothing; the inverse undoes whatever its partner did.
- A matrix is **singular** ($\det = 0$) exactly when it collapses a dimension. The crushed directions form the kernel; the surviving image is the column space.

We have done this all in 2D for visual clarity. In Chapter 4 we will quantify *how much* a transformation expands or shrinks space -- the determinant -- and finally see why $\det(AB) = \det(A)\det(B)$.

---

## What Comes Next

**Chapter 4: The Secrets of Determinants.** The determinant is a single number that captures the volume-scaling factor of a transformation. We will see why it multiplies under composition, why a sign flip means orientation reversal, and why $\det = 0$ is precisely the singular case we just met.

---

## Series Navigation

- **Previous:** [Chapter 2 -- Linear Combinations and Vector Spaces](/en/chapter-02-linear-combinations-and-vector-spaces/)
- **Next:** [Chapter 4 -- The Secrets of Determinants](/en/chapter-04-the-secrets-of-determinants/)
- **Series:** Essence of Linear Algebra (3 of 18)
