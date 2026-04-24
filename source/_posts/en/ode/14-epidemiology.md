---
title: "Ordinary Differential Equations (14): Epidemic Models and Epidemiology"
date: 2024-09-05 09:00:00
tags:
  - Ordinary Differential Equations
  - Epidemiology
  - SIR Model
  - SEIR Model
  - COVID-19
  - Infectious Disease Dynamics
  - Python
categories:
  - Ordinary Differential Equations
series:
  name: "Ordinary Differential Equations"
  part: 14
  total: 18
lang: en
mathjax: true
description: "Mathematical epidemiology from first principles. Build the SIR and SEIR models, derive R0 and the herd-immunity threshold, fit COVID-style scenarios with asymptomatic transmission and time-varying interventions."
disableNunjucks: true
series_order: 14
---

**In early 2020 the entire world watched a small system of ordinary differential equations decide policy.** "Flatten the curve" was not a slogan; it was the intuition of a specific equation. *Herd immunity* was not a guess; it was the threshold $1 - 1/R_0$ derived in a single line. The SIR model -- four lines of math, written down in 1927 by Kermack and McKendrick -- turned out to be precise enough to drive trillion-dollar decisions.

This chapter builds that machinery from scratch. We start with the basic SIR model, derive every threshold and final-size relation analytically, and then layer on the realism: incubation periods (SEIR), asymptomatic transmission, vaccinations, and time-varying interventions (a stylised COVID-style scenario). Throughout, the goal is not to *believe* a forecast but to **understand which mechanism a given parameter controls**.

## What You Will Learn

- The SIR model as a 3-equation system: compartments, parameters, and the **basic reproduction number** $R_0 = \beta/\gamma$
- The threshold theorem ($R_0 > 1 \Leftrightarrow$ outbreak) and the **final-size relation** $S_\infty = S_0 e^{-R_0(1 - S_\infty)}$
- Closed-form expressions for the **peak height** $I^* = 1 - 1/R_0 - \ln R_0 / R_0$ and the herd-immunity threshold $1 - 1/R_0$
- The SEIR variant, latent-period dependence, and why it slows the initial growth rate
- COVID-19 extensions: asymptomatic compartment, intervention-induced time-varying $R_e(t)$, reporting iceberg
- Network-level reproduction numbers and the role of super-spreaders
- Practical fitting: what $R_0$, doubling time, and serial interval actually measure

**Prerequisites**: phase-plane analysis from [Chapter 7](/en/ode-chapter-07-systems-and-phase-plane/), nonlinear stability from [Chapter 8](/en/ode-chapter-08-nonlinear-stability/), numerical methods from [Chapter 11](/en/ode-chapter-11-numerical-methods/).

---

## The SIR Model

Split the population into three compartments:

- $S$ -- **susceptible** (can catch the disease)
- $I$ -- **infectious** (currently transmitting)
- $R$ -- **removed** (recovered with immunity, isolated, or dead)

Mass-action transmission and exponential recovery give the **Kermack-McKendrick SIR system**:

$$\boxed{\;\dot S = -\frac{\beta\,S\,I}{N},\qquad \dot I = \frac{\beta\,S\,I}{N} - \gamma I,\qquad \dot R = \gamma I.\;}$$

Two parameters carry all the physics:

- $\beta$ -- transmission coefficient: average effective contacts per unit time, times probability of transmission per contact.
- $\gamma$ -- removal rate: $1/\gamma$ is the average duration of infectiousness.

The total $S + I + R \equiv N$ is conserved (there are no births or deaths in the closed model), so the system is genuinely two-dimensional.

### The basic reproduction number $R_0$

Linearise around the disease-free equilibrium $(S, I, R) = (N, 0, 0)$. The $I$-equation becomes
$$\dot I \approx (\beta - \gamma) I,$$
so the disease grows when $\beta > \gamma$, i.e. when
$$\boxed{\;R_0 \equiv \frac{\beta}{\gamma} > 1.\;}$$
**Threshold theorem.** The disease-free equilibrium is locally stable iff $R_0 < 1$. This number has a beautifully concrete meaning: it is the *expected number of secondary infections caused by one typical infectious individual placed into a fully susceptible population*. If each infected person infects fewer than one other, the chain dies out. If more than one, it explodes -- at first.

### What controls the peak?

Divide the $I$- and $S$-equations:
$$\frac{dI}{dS} = \frac{\gamma}{\beta S/N} - 1 = \frac{1}{R_0\,S/N} - 1.$$
Integrate from $(S_0, I_0) \approx (N, 0)$. With $s = S/N$ and $i = I/N$,
$$i(s) = i_0 + (s_0 - s) + \frac{1}{R_0}\ln\frac{s}{s_0}.$$
Setting $di/ds = 0$ gives the peak fraction infected at $s = 1/R_0$:
$$\boxed{\;I^* \;\approx\; 1 - \frac{1}{R_0} - \frac{\ln R_0}{R_0}.\;}$$
And the time of peak is when $S$ first crosses $1/R_0$. **Both** quantities are determined by $R_0$ alone (with $\gamma$ setting the timescale).

### Final size: who escapes?

Letting $t \to \infty$, $I \to 0$ and $S \to S_\infty$. Dividing $\dot S$ by $\dot R$ and integrating yields the transcendental
$$\boxed{\;S_\infty = S_0\,\exp\!\bigl[-R_0\,(1 - S_\infty/N)\bigr].\;}$$
For $R_0 = 2.5$ and $S_0 \approx N$, the equation gives $S_\infty / N \approx 0.107$ -- about 89% of the population is infected by the end. Surprising, but a one-line consequence of the math.

![SIR dynamics, phase portrait, cumulative incidence, and final-size relation.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/14-epidemiology/fig1_sir_model.png)

*Top-left: classic SIR time series for $R_0 = 2.5,\ 1/\gamma = 10$ d -- $S$ collapses, $I$ peaks then dies, $R$ saturates. Top-right: phase portrait in $(S, I)$ -- trajectories enter the upper half-plane, peak when crossing the vertical line $S = 1/R_0$, and spiral toward the $S$-axis. Bottom-left: cumulative incidence $1 - S$ asymptotes to the final-size value (red dashed line). Bottom-right: $S_\infty$ and $1 - S_\infty$ as functions of $R_0$ -- the curve is steep just above 1, meaning small changes in $R_0$ cause huge changes in eventual reach.*

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir(t, y, beta, gamma, N):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]

N, gamma = 1.0, 1/10
fig, ax = plt.subplots(figsize=(10, 5))
for R0 in [1.5, 2.0, 2.5, 3.0]:
    beta = R0 * gamma
    sol = solve_ivp(sir, [0, 200], [N - 1e-3, 1e-3, 0],
                    args=(beta, gamma, N),
                    t_eval=np.linspace(0, 200, 600))
    ax.plot(sol.t, sol.y[1], lw=2, label=f'R0 = {R0}')
ax.set_xlabel('Days'); ax.set_ylabel('Fraction infectious')
ax.legend(); plt.tight_layout(); plt.show()
```

---

## Sensitivity to $R_0$

$R_0$ is *the* lever that determines outcomes. A factor-of-two change in $R_0$ can multiply peak demand on hospitals five-fold and pull the peak forward by weeks. The sensitivity is also analytic, which is rare.

![Sensitivity to R0: family of I(t) curves, peak height and timing vs R0, herd-immunity threshold and final size vs R0.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/14-epidemiology/fig2_r0_sensitivity.png)

*Left: a family of $I(t)$ curves at $R_0 \in \{1.2, 1.6, 2.0, 2.5, 3.5, 5.0\}$ for fixed $1/\gamma = 10$ d. Higher $R_0$ -> higher and earlier peak. Middle: peak fraction infected (red) and time of peak (blue) versus $R_0$. The black dashed line is the analytical formula $1 - 1/R_0 - \ln R_0 / R_0$ -- spot on. Right: total fraction infected at the end (red) and the herd-immunity threshold $1 - 1/R_0$ (green dashed). Note that the **final size always overshoots HIT** -- the epidemic does not stop the instant immunity reaches the threshold; it stops when transmission can no longer sustain itself, by which time many "extra" infections have happened.*

---

## Vaccination and Herd Immunity

Move a fraction $v$ of the population from $S$ to $R$ at $t = 0$. The effective reproduction number becomes
$$R_e = R_0 \cdot \frac{S(0)}{N} = R_0\,(1 - v).$$
For $R_e \leq 1$ we need $v \geq 1 - 1/R_0$. This is the **herd-immunity threshold** (HIT): the minimum fraction that must be immune for an introduced case to die out, on average.

| Disease | Typical $R_0$ | HIT |
|---|---|---|
| Influenza | 1.5 | 33% |
| COVID-19 (Wuhan strain) | 2.5 | 60% |
| SARS | 3.0 | 67% |
| COVID-19 (Delta) | 5.0 | 80% |
| Smallpox | 6.0 | 83% |
| Mumps | 10 | 90% |
| Measles | 15 | 93% |

For an **imperfect vaccine** with efficacy $\mathrm{VE} < 1$, the requirement becomes $v\,\mathrm{VE} \geq 1 - 1/R_0$, i.e.
$$v \geq \frac{1 - 1/R_0}{\mathrm{VE}}.$$
For high-$R_0$ diseases combined with imperfect vaccines this can become *infeasible* ($v > 1$).

![Vaccination effect: I(t) for several coverages, peak vs coverage, HIT vs R0, and the VE-correction.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/14-epidemiology/fig3_vaccination_effect.png)

*Top-left: $I(t)$ for vaccination coverages $v = 0,\ 0.20,\ 0.40,\ HIT,\ 0.85$. At $v = HIT$ the curve is barely an outbreak; at $v = 0.85$ no outbreak occurs at all. Top-right: peak and additional infections versus coverage; the green region is post-HIT. Bottom-left: HIT versus $R_0$ with example diseases marked. Bottom-right: required coverage $v$ as a function of $R_0$ for vaccine efficacies $\mathrm{VE} \in \{0.5, 0.7, 0.9, 1.0\}$ -- the "infeasible" red shading shows where high $R_0$ + low VE makes herd immunity unattainable through vaccination alone.*

The picture also justifies the public-health emphasis on getting **the laggards** vaccinated: under a high-$R_0$ disease, the last 10% matters as much as the first 60%.

---

## The SEIR Model

Many diseases have an **incubation period** -- people are infected but not yet infectious. Add a latent compartment $E$ (exposed):

$$\dot S = -\frac{\beta SI}{N}, \quad \dot E = \frac{\beta SI}{N} - \sigma E, \quad \dot I = \sigma E - \gamma I, \quad \dot R = \gamma I.$$

The transition rate $\sigma$ has $1/\sigma$ = average latent duration. The basic reproduction number is unchanged: $R_0 = \beta/\gamma$. So, *do incubation periods matter?*

For the **final size**, no -- it is identical to SIR because the long-run dynamics is set by $R_0$ alone. For the **growth rate**, very much yes. Linearising around the disease-free equilibrium gives a 2x2 system with characteristic equation
$$r^2 + (\sigma + \gamma)\,r + \sigma(\gamma - \beta) = 0.$$
For $R_0 > 1$ the positive root is
$$r_{\text{SEIR}} = \frac{1}{2}\!\left[-(\sigma + \gamma) + \sqrt{(\sigma + \gamma)^2 + 4\sigma\gamma(R_0 - 1)}\right] < r_{\text{SIR}} = \beta - \gamma.$$
The latent stage **slows the early exponential growth**, even though $R_0$ is unchanged. So at fixed $R_0$, SEIR predicts a later, slightly lower peak than SIR. Equivalently: doubling time depends on the *generation interval* $T_g \approx 1/\sigma + 1/\gamma$, not just on $R_0$.

![SEIR all four compartments, comparison with SIR, varying latent period, and growth rate vs latent duration.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/14-epidemiology/fig4_seir_variant.png)

*Top-left: full SEIR trajectory at $R_0 = 3,\ 1/\sigma = 5\ \text{d},\ 1/\gamma = 7\ \text{d}$. Top-right: SEIR vs SIR at the same $R_0$ -- SEIR peaks later and is slightly lower. Bottom-left: family of SEIR $I(t)$ for $1/\sigma \in \{0.5, 2, 5, 10, 20\}$ d -- longer latent periods produce later, broader peaks. Bottom-right: initial growth rate $r$ as a function of latent period; doubling time on the right axis. As $\sigma \to \infty$ (instantaneous transition), SEIR recovers SIR.*

---

## A COVID-Style Extension

Real epidemics need more than SEIR. COVID-19 introduced four mathematical wrinkles:

1. **Asymptomatic transmission.** A fraction $p$ of infections never develop symptoms but still spread (with reduced infectiousness $\kappa$).
2. **Time-varying $\beta$.** Lockdowns, mask mandates, and behavioural changes alter transmission.
3. **Reporting iceberg.** Only a fraction of true cases gets detected; reported = $\rho \cdot I_s$ with $\rho < 1$.
4. **Variants.** New strains restart the dynamics with a fresh $R_0$.

A minimal model splits $I$ into $I_a$ (asymptomatic) and $I_s$ (symptomatic):
$$\dot S = -\frac{\beta(t)\,(I_s + \kappa I_a)\,S}{N}, \quad \dot E = \frac{\beta(t)\,(I_s + \kappa I_a)\,S}{N} - \sigma E,$$
$$\dot I_a = p\,\sigma E - \gamma I_a, \quad \dot I_s = (1 - p)\sigma E - \gamma I_s, \quad \dot R = \gamma(I_a + I_s).$$
The instantaneous **effective reproduction number** is
$$R_e(t) = \frac{\beta(t)\,(1 - p + \kappa p)}{\gamma}\,\frac{S(t)}{N}.$$
We want $R_e(t) < 1$. Two ways: shrink $\beta$ (interventions) or shrink $S/N$ (immunity).

![COVID-style scenario: compartments with intervention timeline, R_e(t), reporting iceberg, counterfactual cumulative incidence.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ode/14-epidemiology/fig5_covid_example.png)

*Top-left: a stylised four-phase scenario -- baseline $\beta = 0.6$ for 30 days, lockdown $\beta = 0.18$ for 30 days, partial relaxation $\beta = 0.30$, then variant + relaxation $\beta = 0.40$. Top-right: effective $R_e(t)$ -- the lockdown phase pushes $R_e$ below 1 (green region) and the third wave climbs above 1 again. Bottom-left: reporting iceberg -- the dashed black line is what surveillance would report ($\rho I_s$), far below the true infectious prevalence $I_s + I_a$ (purple). Bottom-right: counterfactual comparison of cumulative infected with vs without interventions. The gap is the **cases averted** -- the policy benefit, expressed as a fraction of the population.*

This is *not* a fit to real data -- it is a clean cartoon of the structure that real fits use. The same equations, fitted to actual reported case time series with Bayesian inference, drove official forecasts in 2020-2022.

---

## Network and Heterogeneous Models

A homogeneous mean-field SIR is wildly optimistic about who can be reached. Real contacts are highly heterogeneous: most people have a handful of regular contacts; a small minority have hundreds. On a contact network with degree distribution $P(k)$, the basic reproduction number generalises:
$$R_0 = \frac{\beta}{\gamma}\,\frac{\langle k^2 \rangle - \langle k \rangle}{\langle k \rangle}.$$
For **scale-free networks** ($P(k) \propto k^{-\alpha}$ with $2 < \alpha < 3$), $\langle k^2 \rangle$ diverges with system size. This means the epidemic threshold goes to *zero* -- on such networks, even very low transmissibility can sustain an outbreak. **Super-spreaders** dominate the early dynamics in real epidemics; targeting them with contact tracing is disproportionately effective.

Two conceptual lessons:

- *Average* contact rate underestimates risk; the *variance* matters as much.
- Equal-coverage interventions waste resources; *prioritise high-degree nodes*.

---

## Applications and Limits

### Where the math wins

- **Order-of-magnitude forecasting.** Will hospitals overflow in 4 weeks? SEIR with the right $R_0$ gives a usable answer.
- **Threshold reasoning.** "How much vaccination do we need?" -> $1 - 1/R_0$, divided by VE. No fit needed.
- **Counterfactual comparison.** "How many lives did the lockdown save?" -> Run with and without intervention; difference is the answer (subject to $R_0$ uncertainty).

### Where it loses

- **Exact predictions.** $R_0$ varies between settings, populations, weather, behavioural changes. Exact case counts beyond a few weeks are not reliable.
- **Heterogeneity ignored.** Age structure, geography, household clustering all matter; mean-field models give the *wrong* peak timing in detail.
- **Behavioural feedback.** People reduce contacts when they see the news. This is a closed-loop system, and ignoring the feedback inflates predicted peaks.

The right way to use the math is as a *structured language* for arguing about scenarios, not as a literal forecast.

---

## Summary

| Concept | Key formula |
|---|---|
| Basic reproduction number | $R_0 = \beta / \gamma$ |
| Outbreak threshold | $R_0 > 1$ |
| Peak fraction infectious | $I^* \approx 1 - 1/R_0 - \ln R_0 / R_0$ |
| Final size relation | $S_\infty = S_0\,e^{-R_0(1 - S_\infty/N)}$ |
| Herd-immunity threshold | $1 - 1/R_0$ |
| Imperfect vaccine | $v \geq (1 - 1/R_0)/\mathrm{VE}$ |
| SEIR initial growth | $r$ from $r^2 + (\sigma + \gamma)r + \sigma(\gamma - \beta) = 0$ |
| Effective $R$ | $R_e(t) = R_0(t)\,S(t)/N$ |
| Network $R_0$ | $\beta\,\langle k^2 - k\rangle / (\gamma\,\langle k \rangle)$ |

---

## Exercises

**Conceptual.**

1. Why does the SIR final size *overshoot* the herd-immunity threshold? Make the argument both intuitive and analytical.
2. The serial interval and the generation interval are both proxies for "time between successive infections". Define them, and explain why they differ.
3. Two countries report the same case-doubling time. Could they have very different $R_0$? Use SEIR to argue.

**Computational.**

4. Solve the SIR system for $R_0 = 1.05, 1.5, 3.0$ and check the final-size formula numerically against the transcendental equation.
5. For SEIR at $R_0 = 3$, plot the *generation interval* $1/\sigma + 1/\gamma$ versus the doubling time; verify that doubling time grows linearly with generation interval at fixed $R_0$.
6. Implement the asymptomatic COVID model, fit it to your country's first-wave reported time series with two free parameters ($\beta$ and start time), and compute $R_e(t)$.

**Programming.**

7. Animate the family of SIR phase portraits as $R_0$ slides from 0.5 to 5.0.
8. Build a stochastic SIR (Gillespie algorithm) and compare its mean trajectory with the deterministic ODE for population sizes $N = 10^2,\ 10^4,\ 10^6$. When does the deterministic limit kick in?
9. Implement an age-structured SIR with two age classes (children, adults) and a 2x2 contact matrix. Compute the next-generation matrix and find $R_0$ as its dominant eigenvalue.

---

## References

- Kermack & McKendrick, "A contribution to the mathematical theory of epidemics," *Proc. Roy. Soc. A* 115 (1927)
- Anderson & May, *Infectious Diseases of Humans*, Oxford University Press (1991)
- Diekmann, Heesterbeek & Britton, *Mathematical Tools for Understanding Infectious Disease Dynamics*, Princeton (2013)
- Keeling & Rohani, *Modeling Infectious Diseases in Humans and Animals*, Princeton (2008)
- Brauer, Castillo-Chavez & Feng, *Mathematical Models in Epidemiology*, Springer (2019)
- Ferguson et al., "Impact of non-pharmaceutical interventions to reduce COVID-19 mortality," *Imperial College Report 9* (2020)
- Pastor-Satorras & Vespignani, "Epidemic spreading in scale-free networks," *Phys. Rev. Lett.* 86 (2001)

---

**Series Navigation**


*This is Part 14 of the 18-part series on Ordinary Differential Equations.*
