# Teaching Notes: Neural Operators, DeepONet, PINNs, and Theory

Compiled 2026-09-05. Every claim below was checked against the arXiv abstract page and/or
arXiv HTML full text (`arxiv.org/html/<id>`) for the stated paper. Where a fetch failed or
a claim could not be verified, it is marked **UNVERIFIED**. Section/equation/theorem numbers
are as they appear in the fetched HTML version, dated as noted.

---

## 1. DeepONet — Lu, Jin, Karniadakis, arXiv:1910.03193

**Read:** arXiv abstract page (`arxiv.org/abs/1910.03193`) and HTML full text
(`arxiv.org/html/1910.03193`), current version v3 (submitted 2019-10-08, v2 2020-04-14,
v3 2020-04-15). Journal version: Nature Machine Intelligence 3 (2021), doi
10.1038/s42256-021-00302-5.

**Preprint-vs-journal author discrepancy (verified via arXiv citation metadata):** The
arXiv preprint (all versions, including current v3) lists only three authors — Lu Lu,
Pengzhan Jin, George Em Karniadakis — confirmed directly from arXiv's citation metadata
tags. The task brief for this note-taking assignment named five authors (Lu, Jin, Pang,
Zhang, Karniadakis), which matches the published Nature Machine Intelligence author list
(adding Guofei Pang and Zhongqiang Zhang). I could not fetch the Nature Machine
Intelligence article body itself — `doi.org/10.1038/s42256-021-00302-5` redirects to
`nature.com`, which itself redirects to an authentication/paywall page
(`idp.nature.com/authorize`) that WebFetch cannot pass. **So: the two extra authors on the
journal version are confirmed by the mismatch between arXiv metadata and the citation
the source list gave me, but the journal abstract/body text, received/accepted dates, and any
substantive changes to the text are UNVERIFIED — paywalled.**

### Plain-language summary
Neural networks are known to approximate any continuous *function*. A 1995 theorem by Chen
and Chen says something stronger: a network with one hidden layer can approximate any
continuous *operator* — a map from one function to another function (e.g., "initial
condition in, entire solution trajectory out"). DeepONet is the first practical
architecture built to exploit that theorem. It splits the job into two networks: a
**branch net** that reads in the input function (sampled at fixed points) and a **trunk
net** that reads in the location where you want the output evaluated. Their outputs are
combined by a dot product. Once trained, a DeepONet can evaluate the output function at
any point without re-solving anything — it has learned the whole solution *operator*, not
one solution.

### Key definitions and equations
- **Universal approximation theorem for operators** (paper's Theorem 1, restating Chen &
  Chen 1995), quoted from the fetched text: for continuous non-polynomial activation σ,
  Banach space X, compact sets K1⊂X, K2⊂ℝᵈ, compact V⊂C(K1), and nonlinear continuous
  operator G: V→C(K2), for any ε>0 there exist n,p,m and constants such that
  |G(u)(y) − Σₖ₌₁ᵖ [Σᵢ₌₁ⁿ cᵢᵏσ(Σⱼ₌₁ᵐ ξᵢⱼᵏu(xⱼ)+θᵢᵏ)]·σ(wₖ·y+ζₖ)| < ε.
  The bracketed inner sum is identified as the **branch** (encodes u at m sensors), the
  outer σ(wₖ·y+ζₖ) is the **trunk** (encodes location y).
- **DeepONet architecture** (Section 2.1, Fig. 1): input function u is sampled at m fixed
  "sensor" locations x1,…,xm, giving a vector [u(x1),…,u(xm)]. The output location y is a
  separate input.
  - **Stacked DeepONet** (Fig. 1C): one trunk net (input y, output [t1,…,tp]) plus p
    separate branch nets (each takes the sensor vector, outputs a scalar bk).
  - **Unstacked DeepONet** (Fig. 1D): one trunk net plus one branch net (outputs the full
    vector [b1,…,bp] at once). Combination in both cases (Eq. 2): G(u)(y) ≈ Σₖ₌₁ᵖ bk·tk +
    b0. Unstacked has far fewer parameters and is reported as generalizing better
    (Section 4.1).

### Experiments and exact numbers (Section 4)
- **Linear ODE** (antiderivative operator, ds/dx=u(x)): m=100 sensors, 10,000 training
  functions; best fully-connected-net test MSE ≈10⁻⁴; unstacked DeepONet's error is
  reported as 10–20× lower (Fig. 3B).
- **Nonlinear ODE** (ds/dx = −s²(x)+u(x)): 10,000 training / 100,000 test points; the
  paper reports the empirical relation MSE_test ≈ 10 × MSE_train − 10⁻⁴ (p.9).
- **Gravity pendulum with external forcing** (ds1/dt=s2, ds2/dt=−k sin s1 + u(t)),
  Section 4.2:
  - Sensor-count scaling: for k=1, T=1, l=0.2, error decays roughly as MSE ∝ 4.6^(−#sensors)
    until saturating around 10 sensors; longer horizons need more sensors (T=5 needs ~25).
  - Dataset-size scaling (width 100, T=3): below ~10⁴ training samples, both test and
    generalization error fall **exponentially** (∝ e^(−x/2000)); above 10⁴, they fall
    **algebraically** — test error ∝ x^(−0.5), generalization error ∝ x^(−1). The paper
    flags this x^(−1) rate as "surprising," beating the classical x^(−0.5) rate from
    learning theory (p.12).
- **Diffusion-reaction PDE** (∂s/∂t = D∂²s/∂x² + ks² + u(x), D=0.01, k=0.01, on
  [0,1]×[0,1]): with only 100 input-function samples (P=1000 points per solution, ~10
  points per location on average), test error reaches ~10⁻⁵ (Section 4.3, p.13).
- These are the source of the abstract's headline claim: "polynomial rates (from half
  order to fourth order) and even exponential convergence" with respect to training
  dataset size.

### Limitations / future work (Section 5)
- No theoretical result yet for required network *size* (width/depth) for operator
  approximation, analogous to known function-approximation bounds.
- No theoretical explanation for why DeepONets generalize as well as they empirically do.
- Suggested future work: replacing the fully-connected branch/trunk with CNNs or
  attention.

### Teachable moments / misconceptions
- **Misconception:** "DeepONet is just a bigger MLP." Correct framing: the branch/trunk
  split is what makes it an *operator* network — the trunk factorizes out the query
  location y so the network can be evaluated at any y after training, which a plain MLP
  mapping "sampled input → sampled output" cannot do without retraining for a new grid.
- **Misconception:** more sensors always help. The pendulum experiment shows sensor count
  needs scale with the desired prediction horizon T and the input's length scale l (roughly
  m∝√T, m∝l⁻¹, per the fetched text) — oversampling wastes capacity, undersampling loses
  accuracy.
- Relation to FNO: DeepONet is the ancestor of the encoder/approximator/decoder framing
  later formalized by Kovachki–Lanthaler–Stuart (paper 3 below); its trunk net is
  essentially a learned basis, and FNO can be viewed as replacing that learned basis with
  a fixed Fourier basis plus nonlinear layers (see paper 5).

---

## 2. PINNs Part I — Raissi, Perdikaris, Karniadakis, arXiv:1711.10561

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/1711.10561`), v1,
submitted 2017-11-28. Journal version: J. Comput. Phys. 378 (2019), doi
10.1016/j.jcp.2018.10.045. I did not separately fetch the journal PDF; assume the JCP
paper is the two-part treatise's typeset merge — **journal-vs-preprint differences are
UNVERIFIED** (not requested to be checked separately by the team lead for this pair, only
for DeepONet).

### Plain-language summary
A physics-informed neural network (PINN) is a neural network trained to satisfy a PDE
directly, rather than trained on many example solutions. You plug the network's own output
into the PDE using automatic differentiation, and penalize the residual (how far it is
from zero) as a loss term alongside the usual data-fitting loss. Part I is about the
"forward" problem: given a known PDE and some initial/boundary data, find the solution.

### Key definitions and equations
- PDE form assumed (Section 2): u_t + N[u] = 0 for x∈Ω, t∈[0,T], N a (possibly nonlinear)
  differential operator.
- Residual network: f := u_t + N[u], computed via autodiff on the same network that
  outputs u(t,x).
- **Loss (continuous time model, Eq. 4-ish, Section 2.1):** MSE = MSE_u + MSE_f, where
  MSE_u is squared error against known initial/boundary data (Nu points) and MSE_f is
  squared PDE residual at Nf collocation points (no labels needed there — it's an
  unsupervised physics penalty).
- **Discrete time models (Section 3):** embed an implicit Runge-Kutta scheme with q stages
  directly into the network so a single network forward pass predicts all RK stage values;
  this lets the model take one very large timestep Δt using a high-order (large q) implicit
  scheme "at effectively no extra cost" (paper's own phrase, Section 3.1 footnote) because
  the stages are just extra network outputs, not extra unknowns to solve for by hand.

### Worked examples and exact numbers
| Example | Model | Data pts | Collocation pts | Architecture | Relative L2 error |
|---|---|---|---|---|---|
| Burgers' eq. (u_t+uu_x−(0.01/π)u_xx=0) | continuous time | Nu=100 | Nf=10,000 | 9 layers × 20 neurons | 6.7×10⁻⁴ |
| Burgers' eq. | discrete time, q=500, Δt=0.8 (single step 0.1→0.9) | Nn=250 | — | 4×50 | 8.2×10⁻⁴ |
| Nonlinear Schrödinger (ih_t+0.5h_xx+|h|²h=0) | continuous time | N0=50, Nb=50 | Nf=20,000 | 5×100, tanh | 1.97×10⁻³ |
| Allen-Cahn (u_t−0.0001u_xx+5u³−5u=0) | discrete time, q=100, Δt=0.8 | Nn=200 | — | 4×200 | 6.99×10⁻³ |

- Burgers' continuous-time result is reported as "about two orders of magnitude lower"
  than the authors' own earlier Gaussian-process-based method.
- Optimizer: L-BFGS (full-batch) throughout; the paper notes mini-batch SGD variants are
  an option for larger datasets. Collocation/data points sampled via Latin Hypercube
  Sampling.
- Theoretical RK truncation-error bound quoted for the q=500 Burgers case:
  O(Δt^(2q)) = 0.8^1000 ≈ 10⁻⁹⁷ — i.e., the time-discretization error is astronomically
  smaller than the actual observed error, meaning the 8.2×10⁻⁴ error is dominated by
  network approximation/optimization error, not by the RK scheme.

### Limitations (Section 2 concluding remarks, Section 4)
- Collocation points scale exponentially with dimension — a curse-of-dimensionality
  bottleneck in high-dimensional PDEs.
- No uncertainty quantification (unlike the authors' earlier Gaussian-process approach).
- No convergence guarantee to a global optimum; empirical success only.
- Explicit disclaimer: PINNs are "not... replacements of classical numerical methods for
  solving partial differential equations (e.g., finite elements, spectral methods)" — those
  remain more robust and efficient where applicable. PINNs are pitched on simplicity of
  implementation and differentiability, not on beating classical solvers on cost.

### Teachable moments
- **PINN is a constraint-satisfaction method, not an operator-learning method.** It
  produces one function u(t,x) that solves one instance of a PDE (one set of boundary/
  initial conditions). Re-solving for a new initial condition means retraining from
  scratch. This is the central contrast with DeepONet/FNO, which learn a *map* from initial
  condition to solution and can be evaluated on new instances without retraining. PINO
  (paper 6) is explicitly built to close this gap.
- **Misconception:** "the PDE residual loss makes PINNs data-free." False — Part I's
  examples still need Nu initial/boundary data points; the PDE loss supplements, not
  replaces, data. (Full PDE-free discovery of the solution requires only IC/BC, not zero
  data — see Wang et al., paper 7, for a related architecture doing this with DeepONet.)

---

## 3. PINNs Part II — Raissi, Perdikaris, Karniadakis, arXiv:1711.10566

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/1711.10566`), v1,
submitted 2017-11-28. Same JCP 2019 journal reference as Part I.

### Plain-language summary
Part II flips the problem: instead of assuming you know the PDE and solving it, you assume
you know (noisy, scattered) measurements of the solution and want to discover unknown
coefficients in the governing PDE — e.g., what is the viscosity in Burgers' equation? The
unknown coefficients λ become extra trainable parameters of the same network, learned
jointly with the network weights by minimizing the same two-part loss.

### Key definitions and equations
- PDE with unknown parameters: u_t + N[u;λ] = 0. f := u_t + N[u;λ]; λ is now a learnable
  parameter vector, alongside the network's weights.
- **Continuous time algorithm:** MSE = MSE_u + MSE_f, collocation points taken at the same
  locations as the data (not extra points, since data is often sparse to begin with).
- **Discrete time algorithm:** same Runge-Kutta embedding as Part I, but now uses only two
  temporal snapshots (t^n, t^(n+1)) to recover λ. Stage count chosen via q =
  0.5·log(ε)/log(Δt) (ε = machine precision) so temporal discretization error is driven
  below machine precision.

### Worked examples and exact numbers
- **Burgers' equation, continuous time** (true λ1=1.0, λ2=0.01/π), N=2,000 points, 9×20
  network: noise-free parameter errors 0.096% (λ1), 0.469% (λ2). With 1% noise: 0.039% /
  0.008%. With 10% noise: 0.101% / 6.391%.
- **2D Navier–Stokes, cylinder wake, Re=100**, continuous time, only velocity data used
  (pressure inferred, not trained on): N=5,000 points (1% of the full dataset), 9×20
  network. Noise-free: λ1 error 0.078%, λ2 error 4.67%. With 1% noise: 0.17% / 5.70%.
  Pressure field is reconstructed only qualitatively, since it's never directly supervised.
- **Burgers', discrete time**, two snapshots Δt=0.8, Nn=199/201 points, 4×50 network:
  exact recovery noise-free; with 1% noise λ1 error 0.221%, λ2 error 3.215%; with 5% noise
  λ1 0.097%, λ2 13.479% (noticeably worse — noise sensitivity grows fast for the
  diffusion-like coefficient).
- **Korteweg–de Vries** (u_t+λ1·u·u_x+λ2·u_xxx=0), discrete time, two snapshots,
  4×50 network: noise-free λ1 error 0.023%, λ2 error 0.006%; 1% noise: 0.057% / 0.017%.

### Limitations
- The paper's own conclusion, quoted directly: "this two-part treatise creates more
  questions than it answers."
- The *form* of the PDE must be known in advance — only its coefficients are unknown; this
  is coefficient discovery, not equation discovery from scratch.
- Robustness degrades sharply at higher noise (10% noise cases show large scatter,
  especially for diffusion-type terms).
- Requires the observation window to contain "sufficient dynamics" for identifiability.

### Teachable moments
- The dramatic contrast between the KdV result (sub-0.1% coefficient error even at 1%
  noise) and the 5%-noise Burgers' result (13.5% error on the diffusion coefficient) is a
  good discussion point: diffusive/dissipative terms are harder to identify from noisy data
  than dispersive/advective terms, because noise looks locally like extra diffusion.

---

## 4. Kovachki, Lanthaler, Stuart — "Operator Learning: Algorithms and Analysis," arXiv:2402.15715

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/2402.15715`), v1,
submitted 2024-02-24 (only one version on arXiv). Per the task instructions, I read the
abstract, introduction, main theorem statements, and conclusion only (this is a ~32MB
review, not read cover-to-cover).

### Plain-language summary
This is a survey positioning "operator learning" as the field of using machine learning to
approximate maps between infinite-dimensional function spaces (typically the map from PDE
inputs — coefficients, initial/boundary data — to PDE outputs — the solution). It gives a
unifying mathematical lens for architectures like DeepONet and FNO and reviews what's
proven and what's still open.

### Taxonomy and unifying framework
- Two broad paradigms (their Section 3.5): (a) **encoder–decoder-net** methods (PCA-Net,
  DeepONet) — reduce the infinite-dimensional problem to encode → finite network →
  decode, echoing classical FEM/FVM/FDM structure; (b) **neural-operator-style**
  architectures (FNO and generalizations) that replace the affine layers of an ordinary
  neural network directly with integral operators, generalizing the whole network to
  function space rather than just its endpoints. Kernel methods and other approaches
  (reduced-order modeling, Koopman operator theory, closure modeling) are also mentioned.
- **General neural-operator layer** (Section 4.2): Lℓ(v)(x,θ) = σ(Wℓv(x) + bℓ +
  𝒦(v)(x,γℓ)) — a pointwise affine map plus a nonlinearity plus a parametrized integral
  (kernel) operator 𝒦. FNO is the special case where 𝒦 is Fourier multiplication.
- **DeepONet fits the encoder-decoder-net framework** (Section 3.2): written as
  ΨD(u,θ)(y) = Σⱼ αⱼ(Lu,θα)·ψⱼ(y,θψ) — coefficients αⱼ from the branch net, basis
  functions ψⱼ from the trunk net. Theorem 4.1 states any continuous operator between
  separable Banach spaces (with the approximation property) can be approximated by a
  ΨED = F𝒰∘α∘G𝒱 map, i.e., bounded linear encode/decode sandwiching a finite-dimensional
  map α — this is the abstract version of what DeepONet does concretely.
- **Universality of the "Averaging Neural Operator"** (Theorem 4.5): a minimal
  integral-kernel architecture with constant kernels is already a universal approximator
  on compact sets, establishing that FNO-style universality doesn't require anything more
  exotic than the general layer form above.

### Main open questions the authors flag
1. Quantitative complexity: how many parameters/samples are needed for error ε, for
   general (not special-structure) operators — largely open outside linear and
   holomorphic operator classes.
2. No optimization theory: results assume best-case (approximation-theoretic) networks;
   gradient-descent training dynamics are not analyzed, so there's a gap between what
   theory guarantees and what practitioners actually run.
3. Nonlinear vs. linear approximation: encoder-decoder approaches output within a fixed
   linear subspace (like PCA); FNO-style approaches can represent nonlinear
   solution manifolds; which is better and when is "still sparse" (their words, Section
   3.5).
4. How to design good random features for the operator-learning setting is "largely
   unresolved."

### Error-bound highlights referenced
- Linear operators (Theorem 5.1): sample complexity N ~ ε^(−(α+p)/(α′+min{p−1/2,s}))
  depending on smoothness/measure/prior decay parameters.
- Holomorphic operators (Section 5.2): best n-term approximation converges at rate
  n^(1−1/p) (sup norm) or n^(1/2−1/p) (Bochner L²) — dimension-independent, i.e., these
  escape the curse of dimensionality.
- General Lipschitz operators (Section 5.3): error decomposed into encoding/decoding/
  approximation pieces, with complexity depending on the latent dimensions and network
  width — this is the same three-way decomposition used concretely in paper 5 below
  (Lanthaler–Mishra–Karniadakis).

### Conclusions (their Section 6, paraphrased with direct quote elements)
Universality is established qualitatively for the major architectures (DeepONet, FNO,
Averaging Neural Operator), but quantitative theory — how big a network, how much data —
is only solved for restricted operator classes (linear, holomorphic); general Lipschitz
operators still lack tight rates. There remains a substantial gap between approximation
theory and the gradient-descent training practitioners actually use. The authors frame
operator learning as "a maturing field with solid foundational results but significant
open problems in bridging approximation theory to practical training dynamics."

### Teachable moments
- This paper is the right place to introduce students to the idea that DeepONet and FNO
  are not competitors so much as two points on a spectrum: encode/finite-map/decode
  (DeepONet) vs. generalize-the-whole-network-to-function-space (FNO). Both are proven
  universal; the open question is efficiency (how many parameters for a given accuracy),
  not existence.

---

## 5. Lanthaler, Mishra, Karniadakis — "Error estimates for DeepONets," arXiv:2102.09618

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/2102.09618`), current
version v3 (v1 2021-02-18, v2 2021-03-31, v3 2022-01-13).

### Plain-language summary
This paper puts DeepONet's empirical success on rigorous footing. It (a) extends the
universal-approximation guarantee to a broader, more realistic setting (measurable, not
just continuous, operators on possibly non-compact spaces); (b) breaks the total
approximation error into three named, separately-analyzable pieces; (c) shows that while
*some* operators are provably hard (need exponentially large networks), the specific
operators that arise from real PDEs are provably easy (only need polynomially/algebraically
growing networks).

### Key definitions, theorem statements, equations
- **Theorem 3.1** (extended universal approximation): for a Borel-measurable operator
  𝒢: C(D)→L²(D) with 𝒢∈L²(μ), for any ε>0 there is an operator network 𝒩 = ℛ∘𝒜∘ℰ
  (encoder-approximator-reconstructor) with ‖𝒢−𝒩‖_L²(μ) < ε. This drops the compactness
  and continuity assumptions Chen & Chen (1995)/the DeepONet paper's Theorem 1 needed, at
  the cost of using an L²(μ⊗dy) distance instead of sup-norm.
- **Error decomposition (Theorem 3.3):** total error ℰ̂ ≤ Lipα(𝒢)·Lip(ℛ∘𝒫)·(ℰ̂_ℰ)^α +
  Lip(ℛ)·ℰ̂_𝒜 + ℰ̂_ℛ, i.e., total error is controlled by (i) **encoding error** ℰ̂_ℰ (how well
  the m sensor values capture the input function), (ii) **approximation error** ℰ̂_𝒜 (how
  well the finite network approximates the induced finite-dimensional map), and (iii)
  **reconstruction error** ℰ̂_ℛ (how well the trunk-net basis reconstructs the output
  function from finitely many coefficients).
- **Link to spectral decay / PCA (Theorems 3.6, 3.8):** reconstruction error is
  lower-bounded by the tail sum of eigenvalues of the covariance operator of the pushed-
  forward output measure: √(Σ_{k>p} λk) ≤ ℰ̂_ℛ; similarly encoding error is lower-bounded by
  √(Σ_{k>m} λk) for the *input* measure's covariance eigenvalues. In plain terms: if the
  output functions don't have quickly-decaying "principal components," no amount of trunk
  net cleverness with only p basis functions can reconstruct them well — this ties DeepONet
  performance directly to the same spectral-decay ideas as PCA / Karhunen–Loève expansions.
- **Theorem 3.7:** for Lipschitz 𝒢: X→H^s(𝕋ⁿ) with bounded Sobolev norms, reconstruction
  error ℰ̂_ℛ ≤ C·p^(−s/n), with trunk-net size scaling as ≤ C·p·(1+log(p)²) — i.e.,
  algebraic decay in the number of trunk basis functions p, at a rate set by the Sobolev
  smoothness s and spatial dimension n.
- **Negative result (Remark 3.4, citing Yarotsky 2018 Theorem 1):** arbitrary Lipschitz
  functions f:ℝᵐ→ℝ require a ReLU network of size ≳ ε^(−m/2) — for DeepONet with m sensors
  this becomes "worse than algebraic" in 1/ε, i.e., a genuine curse of dimensionality for
  generic operators. Definition 3.5 formalizes "suffering the curse" as size(𝒩ε) ~
  O(ε^(−ϑε)) with ϑε→∞ as ε→0.
- **Four concrete PDE examples that break the curse (Sections 4.1–4.4):** forced nonlinear
  ODE (gravity pendulum), elliptic PDE (variable-coefficient diffusion), parabolic PDE
  (Allen-Cahn-type reaction-diffusion), and a hyperbolic/nonlinear conservation law. For
  all four, the paper proves algebraic (not exponential) convergence rate ε^(−ϑ) for a
  finite, problem-specific exponent ϑ — the exact exponent depends on the Sobolev
  regularity of the solution operator in each case (exact numeric exponents were not
  extracted cleanly from the HTML fetch for each of the four — **flagging this as a
  partial gap**: the fetch confirmed the qualitative "algebraic, finite ϑ" result and cited
  section numbers 4.1.5/4.2.5/4.3.5/4.4.2-3, but did not return the literal numeric
  exponent for each case; recommend pulling the PDF directly if exact exponents are needed
  for a slide).
- **Generalization error (Section 5):** O(N^(−1/2)) up to log factors, N = number of
  training samples — consistent with classical statistical learning theory rates, as
  distinct from the approximation-error rates above.

### Limitations (stated in Introduction)
Analysis covers only *best-approximation* error — it does not model errors from the actual
training algorithm (optimization error), noisy data, or train/test distribution mismatch;
these are named as future work.

### Teachable moments
- This is the natural "why does DeepONet from paper 1 actually work" companion. The
  three-way error decomposition (encode/approximate/reconstruct) is the single most
  reusable mental model in the whole reading list — it recurs in the Kovachki-Lanthaler-
  Stuart survey (Section 5.3) nearly verbatim.
- Good misconception to preempt: "operator learning escapes the curse of dimensionality in
  general." False — the curse is real for generic Lipschitz operators (Remark 3.4); PDE
  solution operators are *special* (smooth, structured) and that's specifically why they're
  learnable efficiently. Same theme repeats in FNO's paper (next entry).

---

## 6. Kovachki, Lanthaler, Mishra — "On universal approximation and error bounds for Fourier Neural Operators," arXiv:2107.07562

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/2107.07562`), v1,
submitted 2021-07-15 (only one version listed). Journal: Journal of Machine Learning
Research, Vol. 22 (2021), pp. 1–76.

### Plain-language summary
This is FNO's theoretical counterpart to paper 5's DeepONet analysis: it proves FNOs are
universal approximators of operators, and — more importantly for practice — proves that for
the specific PDE operators FNO is used on (Darcy flow, Navier–Stokes), the network doesn't
need to blow up in size as you demand more accuracy. It needs only "sub-log-linear" growth
in network size as a function of 1/ε, versus the "super-exponential" blow-up that's
possible for a generic operator.

### Architecture, as given in the paper
- **Spectral convolution** (Eq. 2.5): (K(θ)v)(x) = F⁻¹(Pθ(k)·F(v)(k))(x), where
  Pθ(k)∈ℂ^(dv×dv) are learnable Fourier multipliers at each wavenumber k∈ℤᵈ, constrained
  by Pθ(−k)=Pθ(k)† to keep outputs real.
- **Full FNO layer** (Eq. 2.7): Lℓ(v)(x) = σ(Wℓv(x) + bℓ(x) + F⁻¹(Pℓ(k)·F(v)(k))(x)) —
  pointwise linear term Wℓv(x), bias, plus the spectral (nonlocal) convolution term, then
  a nonlinearity σ.

### Main theorems
- **Theorem 2.5 (universal approximation):** for continuous operator G: H^s(𝕋ᵈ)→H^s'(𝕋ᵈ)
  on a compact K⊂H^s, for any ε>0 there is an FNO N with sup_{a∈K}‖G(a)−N(a)‖_{H^s'} ≤ ε.
  **Theorem 2.15** extends this to the computationally realizable "Ψ-FNO" (pseudo-spectral
  version), requiring input regularity s>d/2.
- **Darcy-flow elliptic PDE (Theorems 3.3, 3.4):** for the elliptic operator
  G: L^∞(𝕋ᵈ)→H¹(𝕋ᵈ) with coefficient a∈H^s(𝕋ᵈ), s>d/2+k, pseudo-spectral discretization
  error ‖u−u_N‖_{H¹} ≲ N^(−(k−1)); the Ψ-FNO achieves this with width O(N^d), depth
  O(log(N^k/ε)), giving overall network-size growth that is **poly-logarithmic in 1/ε**
  (to hit error ε requires N ≍ ε^(−1/(k−1))).
- **Incompressible Navier–Stokes (Theorem 3.5):** single pseudo-spectral timestep error
  ‖u^(n+1)−u_N^(n+1)‖_{L²} ≲ τ·N^(−k) + O(τ²); over T/τ steps, network size scales as width
  O(N^d·dv), depth O(T/τ·log(N^k/ε)) — still **sub-log-linear in 1/ε** given sufficient
  smoothness s>d/2+k.
- **Negative result / lower bound (Remark 3.1, Eq. 3.1):** for *generic* continuous
  Lipschitz operators, worst-case network width blows up as ε^(−ε^(−d/s)) —
  super-exponential. The point of Theorems 3.3–3.5 is that Darcy/Navier–Stokes escape this
  because they have special conservation-law/polynomial-nonlinearity structure that a
  pseudo-spectral method already exploits, and FNO's architecture mirrors pseudo-spectral
  methods exactly (Fourier layers do exact differentiation; σ-layers approximate the
  nonlinearity).

### Limitations
Regularity requirement s>d/2 (smoother s>d/2+k gives better rates); no efficiency guarantee
outside the PDE-structured setting (curse of dimensionality persists for generic
operators, matching paper 5's message); results proven on the periodic torus 𝕋ᵈ, with
extension to non-periodic Lipschitz domains needing a periodic extension (Theorem 2.9);
time-dependent analysis restricted to forward Euler in the main text (higher-order schemes
pushed to an appendix).

### Teachable moments
- Directly comparable to paper 5's DeepONet result: both papers make the *same* structural
  point (generic operators are cursed; PDE operators specifically are not) using
  architecture-specific machinery (spectral truncation for FNO vs. branch/trunk
  factorization for DeepONet). A good exam-style question: "why do both papers need to
  single out four/two specific PDE examples rather than proving a blanket efficiency
  result?" — because the escape from the curse is a property of the *operator* (its
  smoothness/structure), not the architecture; the architecture just needs to be expressive
  enough to exploit that structure once it exists.
- FNO's spectral convolution is a global, fixed (Fourier) basis; DeepONet's trunk net is a
  *learned*, generally non-orthogonal basis. This is the crux of the encoder-decoder-net
  vs. neural-operator taxonomy from paper 4.

---

## 7. Li et al. — Physics-Informed Neural Operator (PINO), arXiv:2111.03794

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/2111.03794`), current
version v4 (v1 2021-11-06, v2 2022-11-14, v3 2023-04-14, v4 2023-07-29).

### Plain-language summary
PINO is the explicit synthesis of operator learning (paper 1/6) and physics-informed
training (papers 2/3): it trains an FNO using *both* a data-fitting loss (on whatever
solution examples you have, possibly at low resolution) and a PDE-residual loss (evaluated
at higher resolution, using exact-in-Fourier-space differentiation). This lets it do things
neither parent method can do alone: generalize to resolutions never seen in training
("zero-shot super-resolution") and succeed on chaotic flows where plain PINNs fail to
optimize.

### Key formulation
- **Combined loss** (Section 3.2): 𝒥_combined = 𝒥_data(𝒢θ) + λ·𝒥_pde(𝒢θ), where the data
  loss can be computed on a coarse grid (e.g., 32×25 for the Burgers example) while the PDE
  loss is evaluated on a finer grid (e.g., 128×100) — the two loss terms don't need to share
  a resolution because the operator itself is resolution-agnostic (an FNO's Fourier layers
  can be evaluated at any grid). For instance-wise test-time fine-tuning, an additional
  anchor loss ℒ_op keeps the fine-tuned solution close to the pretrained operator's
  prediction: ℒ_total = ℒ_pde + α·ℒ_op.
- **Derivative computation (Section 3.3), three methods compared:** (i) numerical Fourier
  differentiation (O(n log n) via FFT, versus O(n) for finite differences, but exploits
  smoothness); (ii) pointwise autograd through the Fourier series representation; (iii)
  **function-wise/exact Fourier-space differentiation** (their preferred method, Eq. 15):
  u' = 𝒬'(v_L)·F⁻¹((i2πK/D)·F(v_L)) — computes the whole gradient field exactly in Fourier
  space, avoiding pointwise autograd's cost for higher-order derivatives.

### Zero-shot super-resolution — exact numbers
Kolmogorov-flow model trained at 64×64×32, tested at up to 256×256×65: relative L² error
stays essentially flat, reported (Table 1) as 6.04% / 6.02% / 6.01% at training / 2× / 4×
resolution — versus a plain FNO trained the same way degrading to 8.28–8.30% at the same
resolutions, i.e., PINO's PDE-constrained training generalizes across resolution while pure
data-driven FNO degrades slightly.

### Other experiment numbers (their Table 2 and Table 4)
- **Burgers':** PINO with data 1.22±0.03%; PINO without any data (PDE loss only) 1.50
  ±0.03%; DeepONet baseline 6.97±0.09%.
- **Darcy flow:** low-res (11×11) data + high-res (61×61) PDE constraint → 1.56±0.05%
  error at 4× resolution, versus 9.46±0.07% for data-only training at the same
  resolution (6× worse).
- **Kolmogorov flow (Navier–Stokes), where plain PINN fails (Section 4.2, Table 4):**
  PINN baseline 18.7% relative error, 4577 seconds; PINO (learned operator + per-instance
  fine-tuning) 0.9% error, 536 seconds — a 20× lower error and 8.5× wall-clock speedup. The
  paper attributes the PINN failure specifically to optimization-landscape difficulty on
  chaotic flow (directly echoing the Krishnapriyan et al. failure-mode paper, #8 below).
- **Long transient flow:** PINO operator inference alone 2.87% error (matches FNO); with
  instance-wise fine-tuning 1.84%; both variants keep a ~400× speedup over a GPU
  pseudo-spectral solver.

### Limitations (their Section 5)
Hard to extend to very high-dimensional problems because of the FFT backbone's scaling;
gradient-descent-based fine-tuning converges more slowly than simply refining the numerical
grid, and the paper explicitly says "further optimization techniques are to be developed";
training complexity and hyperparameter choice remain a practical burden.

### Teachable moments
- PINO is the cleanest concrete example of "PINN as a soft constraint" versus "operator
  learning as data-driven regression" not being mutually exclusive — they're combined as
  two loss terms on the same network. Worth contrasting directly with paper 3's PINN
  discovery setting: PINO doesn't discover unknown coefficients, it learns a *reusable*
  solution operator across many instances of a PDE family, with physics used only to reduce
  the data burden and stabilize training at high resolution.
- The Kolmogorov-flow numbers (18.7% PINN vs 0.9% PINO) are a strong, quotable teaching
  example for why "just add a physics loss to a plain feedforward network" (classic PINN)
  breaks down on chaotic/multiscale problems — this is exactly the failure mode paper 8
  formally diagnoses.

---

## 8. Wang, Wang, Perdikaris — Physics-informed DeepONets, arXiv:2103.10974

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/2103.10974`), v1,
submitted 2021-03-19 (only one version listed).

### Plain-language summary
This paper adds a PINN-style PDE-residual loss to DeepONet training (the reverse pairing
from PINO, which added an operator-learning backbone to PINN-style losses using FNO). The
headline capability: you can train a DeepONet to solve a *family* of PDE instances using
only initial/boundary condition data — no paired (input function → full solution) training
examples at all — because the PDE residual loss supplies the missing supervision.

### Key formulation
- **Loss (Eq. 3.1-3.2):** ℒ(θ) = ℒ_operator(θ) + ℒ_physics(θ). ℒ_operator is the standard
  supervised loss on any paired data available (Eq. 2.4); ℒ_physics(θ) =
  (1/NQm)·Σ|N(u^(i)(x_k), Gθ(u^(i))(y_j^(i)))|² is the PDE residual computed via automatic
  differentiation of the DeepONet's own output at collocation points {y_j}.
- **No-paired-data training** (Section 4.1, antiderivative example): training data
  consists only of the input function u(x) itself and the boundary value s(0)=0; the
  ODE constraint ds/dy=u is enforced purely through ℒ_physics at collocation points. The
  paper reports this recovers accuracy comparable to fully-supervised training despite
  using zero paired (u→s) examples.

### Benchmark numbers (their Section 4 / Table references as fetched)
| Problem | Architecture | Relative L² error | Notes |
|---|---|---|---|
| Antiderivative | tanh nets | 3.25×10⁻³ ± 3.19×10⁻³ | reported as far more data-efficient than the plain (non-physics) DeepONet |
| Diffusion-reaction | 5 layers × 50 units | 0.45% ± 0.16% | reported ~80% accuracy improvement plus 100% reduction in paired-data need vs. plain DeepONet |
| Burgers' | Modified MLP, loss weight λ=20 | 1.38% | reported "three orders of magnitude faster" than conventional solve, per instance |
| Eikonal equation (circular wavefronts) | 6 layers × 50 units | 4.22×10⁻³ | — |

- **Speedup claim:** a trained physics-informed DeepONet predicts solutions for O(10³)
  time-dependent PDE instances "in a fraction of a second" versus a conventional spectral
  solver run instance-by-instance; inference reported at roughly 10ms per PDE instance
  (Appendix C, Table 11b).

### Theoretical results
The fetch found **no new formal error bound** in this paper relating generalization error
to the size of the PDE residual — the analysis is empirical. It cites the Chen & Chen
(1995) universal approximation theorem and Lanthaler-Mishra-Karniadakis's error-estimate
paper (#5 above) as motivation but does not extend either.

### Limitations (their Section 5, "Summary and Discussion")
No principled method for choosing network architecture/feature embedding — e.g., Fourier
feature embeddings (their Eq. 4.5) are needed for high-frequency solutions but the choice
is heuristic; the Burgers' example needs manually-tuned loss weight λ swept over
{1,5,10,20,50,100} — "what are appropriate weights?" is posed as an open question rather
than answered; physics-informed training is slower than plain supervised DeepONet training
because of the automatic-differentiation overhead for the PDE residual; each PDE family
needs its own hand-tuned architecture/activation/depth (ReLU vs tanh vs ELU, varying
depths).

### Teachable moments
- Good direct comparison to PINO: same idea (combine data loss + PDE residual loss on an
  operator-learning backbone) but built on DeepONet's branch/trunk factorization instead of
  FNO's spectral layers — a natural "compare and contrast the two hybrid architectures"
  exercise, and a good segue into why loss weighting (the λ tuning problem raised here) is
  a recurring open problem across essentially all physics-informed training (also flagged,
  independently, in the PINN failure-modes paper below).

---

## 9. Krishnapriyan, Gholami, Zhe, Kirby, Mahoney — "Characterizing possible failure modes in physics-informed neural networks," arXiv:2109.01050 (NeurIPS 2021)

**Read:** arXiv abstract page and HTML full text (`arxiv.org/html/2109.01050`), v2
(v1 2021-09-02, v2 2021-11-11 — the NeurIPS camera-ready).

### Plain-language summary
This paper is the diagnostic counterpart to all the PINN-based methods above: it shows,
rigorously, that a standard PINN can fail badly — even on problems with a known closed-form
solution — once the PDE's coefficients push it into a "harder" regime (faster convection,
sharper reaction, etc.), and that this failure is *not* because the network lacks the
expressive capacity to represent the answer. It's an optimization failure: the PDE-residual
loss term creates an ill-conditioned loss landscape that gradient-based training simply
cannot escape.

### PDE test cases (with exact parameter sweeps, their Section 3 and Appendix A)
- **Convection** (Eq. 5): ∂u/∂t + β·∂u/∂x = 0, x∈[0,2π], u(x,0)=sin(x), periodic BCs.
  Tested β = 1, 10, 20, 30, 40, 50, 70. A standard PINN reaches roughly 100% relative
  error once β exceeds about 10.
- **Reaction** (Eq. 14, Appendix): ∂u/∂t − ρ·u·(1−u) = 0, u(x,0) a Gaussian bump centered
  at π. Tested ρ = 2–10; at ρ=5 the baseline PINN error is ~98%.
- **Reaction-diffusion** (Eq. 10, Section 3.2): ∂u/∂t − ν∂²u/∂x² − ρ·u(1−u) = 0, ρ fixed at
  5, ν swept 2–6. Baseline PINN error ranges 50–96%.

### The ill-conditioning analysis
- Method (Section 4.1): visualize the loss landscape by perturbing the trained model along
  the top two dominant Hessian eigenvectors. Result (their Figure 3): at β=1 the landscape
  is "rather smooth"; at higher β it becomes "complex and non-symmetric," and the optimizer
  gets stuck in high-loss local minima.
- Condition-number scaling (Appendix B.1): for convection, condition number κ ~ O(βN)²; for
  diffusion, κ ~ O(νN²)² — the diffusion term's condition number grows quadratically worse
  in the grid resolution N than convection's, which the paper uses to explain why error
  grows faster as ν increases than as β increases.
- **Core diagnosis, direct quote:** "The failure is due to optimization difficulties
  associated with the PINN's soft PDE constraint" — not insufficient network capacity.

### Two proposed fixes and exact improvement numbers
- **Curriculum regularization** (Section 5.1): start training at an easy coefficient (e.g.,
  β=1) and use the converged weights to warm-start training at progressively harder
  coefficients, "analogous to curriculum learning... but applied by progressively making
  the PDE/ODE harder to solve." Reported result at β=30 (their Table 1): standard PINN
  8.97×10⁻¹ relative error → curriculum-trained 2.02×10⁻² — a 44× improvement.
- **Sequence-to-sequence / time-marching** (Section 5.2, Fig. 5): instead of predicting the
  whole space-time solution at once, march forward in short time windows Δt, using u(t) to
  predict u(t+Δt), holding total collocation-point budget fixed for a fair comparison.
  Reported result for reaction-diffusion at ν=5, ρ=5 (their Table 2): whole-domain training
  9.35×10⁻¹ relative error → seq2seq with Δt=0.1 gives 2.39×10⁻² — a 39× improvement. These
  two results are the source of the abstract's "1–2 orders of magnitude lower error" claim.

### Practical implications (their Section 5 discussion)
Practitioners should not use vanilla whole-domain PINN training on convection-, reaction-,
or diffusion-dominated problems with non-trivial coefficients; curriculum regularization or
time-marching (seq2seq) training should be used instead. The authors frame this as
important precisely because these are simple, well-posed PDEs with known analytical
solutions — if PINNs fail here, "cut-and-paste" application of PINNs to harder,
real-world PDEs (their phrase) is unlikely to be reliable without similar fixes.

### Teachable moments / misconceptions this paper directly corrects
- **The single most important misconception in this whole reading list:** "PINNs fail
  because the network isn't big/expressive enough." This paper shows the opposite —
  capacity is not the bottleneck; the *loss landscape geometry induced by the soft PDE
  penalty* is. This reframes essentially every other PINN result in the list: e.g., it
  explains *why* PINO (paper 7) needed to add an FNO backbone plus its own fine-tuning
  scheme rather than just "a bigger PINN" to handle Kolmogorov flow, and it explains why
  Wang et al.'s physics-informed DeepONet (paper 8) needed manual loss-weight tuning (λ) to
  get Burgers' equation to train stably.
- Good numeric anchor for lecture: convection at β=1 trains fine; by β>10 it's essentially
  broken (~100% error) without a fix. This is a strong, simple demo of "PINNs are not a
  free lunch" that motivates the entire physics-informed-operator-learning literature
  (papers 6–8) as a response.

---

## Cross-cutting synthesis for teaching

1. **Two families, one theorem.** DeepONet (branch/trunk factorization of a learned basis)
   and FNO (fixed Fourier basis + nonlinear spectral layers) are both provably universal
   approximators of operators (Chen & Chen 1995 for DeepONet's ancestor result, extended to
   the measurable/non-compact setting by Lanthaler-Mishra-Karniadakis; Theorem 2.5 of
   Kovachki-Lanthaler-Mishra for FNO). Kovachki-Lanthaler-Stuart's survey shows both are
   instances of one general "neural operator layer" template — the real research frontier
   is not existence of good approximators but their *efficiency* (network size vs. error).

2. **The curse of dimensionality is real for generic operators, escaped only by structure.**
   Both the DeepONet error-estimate paper and the FNO universal-approximation paper prove
   the *same* two-sided result: arbitrary Lipschitz operators need exponentially (or worse)
   large networks, but the specific PDE solution operators used in practice (elliptic,
   parabolic, hyperbolic, Navier–Stokes) are smooth/structured enough to need only
   algebraic/poly-logarithmic network growth. This structure-dependence is the reusable
   "why neural operators work when they work" story for the course.

3. **PINN is a constraint method; DeepONet/FNO are operator-learning methods; PINO and
   physics-informed DeepONet are hybrids.** PINNs (Raissi et al.) solve one instance of a
   PDE by penalizing residuals; they don't amortize across instances. DeepONet/FNO learn a
   reusable map from problem instance to solution, purely from data. PINO and Wang et al.'s
   physics-informed DeepONet each bolt a PDE-residual loss onto an operator-learning
   backbone, gaining data efficiency and (for PINO specifically) resolution-invariance,
   while inheriting the operator-learning backbone's amortization benefit that plain PINNs
   lack.

4. **PINN failure modes (Krishnapriyan et al.) are load-bearing for the whole hybrid
   literature.** The optimization-landscape diagnosis explains, in one mechanism, why
   PINO needed more than "PINN + bigger network" to handle chaotic Kolmogorov flow, and
   why physics-informed DeepONet needed manual loss-weight tuning.

5. **Discretization matters, and can silently break "resolution invariance" claims.** The
   ReNO paper (Bartolucci et al.) shows that FNO — despite being provably universal in the
   continuum limit — is not perfectly "alias-free" in practice because pointwise
   nonlinearities (e.g., ReLU) can reintroduce high frequencies outside the truncated
   Fourier band; CNNs are not alias-free at all across resolutions; Convolutional Neural
   Operators (CNO) are constructed specifically to be alias-free by design (their
   Proposition/Section B.2); DeepONet is alias-free only under uniform-grid sensor
   placement with a matched frame basis, and loses this property under random sensor
   placement. This is the right paper to use when teaching students that "zero-shot
   super-resolution" (PINO's headline claim) and "discretization invariance" (a common FNO
   marketing point) are not automatic — they depend on specific architectural choices that
   the ReNO framework makes precise and testable (their empirical Figure 2 and Table 1).

## Verification notes / gaps to flag

- DeepONet: Nature Machine Intelligence journal body (doi 10.1038/s42256-021-00302-5) is
  paywalled behind `idp.nature.com` authentication; I could not fetch its text. The
  five-author list (adding Guofei Pang, Zhongqiang Zhang) is confirmed only indirectly, by
  the mismatch between arXiv's own citation metadata (3 authors, all versions) and the
  5-author list given in the task brief, which matches the standard published citation for
  this paper. **UNVERIFIED beyond that inference**: any textual differences between
  preprint and journal versions.
- Lanthaler-Mishra-Karniadakis (paper 5): the exact numeric convergence exponents ϑ for
  each of the four concrete examples (Sections 4.1.5, 4.2.5, 4.3.5, 4.4.2-3) were not
  cleanly returned by the HTML fetch — only the qualitative "finite algebraic exponent"
  result and section pointers were confirmed. If exact exponents are needed for teaching
  slides, pull them directly from the PDF (`arxiv.org/pdf/2102.09618`).
- All other numeric claims above (error percentages, convergence rates, table numbers) were
  read directly from each paper's arXiv HTML full text as cited per section; no numbers
  were invented.
