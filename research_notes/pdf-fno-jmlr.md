# Teaching Notes: Fourier Neural Operator (ICLR 2021) and Neural Operator (JMLR 2023)

Source PDFs:
- FNO: `2010.08895v3.pdf` — Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," ICLR 2021. 16 pages.
- JMLR: `21-1524.pdf` — Kovachki, Li, Liu, Azizzadenesheli, Bhattacharya, Stuart, Anandkumar, "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs," JMLR 24 (2023) 1-97.

---

## PAPER 1: Fourier Neural Operator (FNO), ICLR 2021

### Plain-language summary

The paper asks: instead of training a neural network to solve one instance of a partial differential
equation (PDE), can we train a network that learns the *solution operator* itself — the map from any
input coefficient/initial-condition function to the corresponding output solution function, for an
entire family of PDEs at once? Classical numerical solvers (finite element/finite difference) and
even "neural-FEM" methods like physics-informed neural networks (PINNs) must be re-run or
re-trained for every new instance of the input; that is slow when you need thousands of solves (e.g.
airfoil design inverse problems, p.1). The authors build on the "neural operator" framework (an
earlier iterative architecture of integral kernel operators) and propose parameterizing the
integral kernel directly in Fourier space, exploiting the Fast Fourier Transform (FFT) for speed.
The resulting Fourier Neural Operator (FNO) is trained once on data at one resolution and can then
be evaluated at other, even much higher, resolutions with no retraining and no accuracy loss — a
property called "zero-shot super-resolution." On benchmark PDEs (Burgers, Darcy Flow, Navier-Stokes)
FNO beats prior learning-based methods by wide margins and is up to three orders of magnitude
faster than traditional solvers, while being the first ML method to model turbulent Navier-Stokes
flow with resolution invariance (Abstract, p.1).

### Key definitions and equations (with page numbers)

- **Operator learning setup** (p.3): Let D ⊂ Rᵈ be bounded/open, A = A(D;Rᵈᵃ), U = U(D;Rᵈᵘ) separable
  Banach function spaces. G†: A → U is the (nonlinear) true solution operator. Given samples
  {aⱼ,uⱼ}, aⱼ ~ μ, uⱼ = G†(aⱼ), build a parametric approximation Gθ: A → U, θ ∈ Θ, minimizing
  Eₐ~μ[C(G(a,θ), G†(a))] (Eq. 1, p.3).
- **Discretization** (p.4): only point-wise evaluations of aⱼ, uⱼ are available on an n-point mesh
  Dⱼ = {x₁,...,xₙ} ⊂ D. Discretization-invariance means the operator can be evaluated at any x ∈ D,
  even x ∉ Dⱼ, so solutions transfer between different grids.
- **Definition 1 (Iterative updates)**, p.4: vₜ₊₁(x) := σ( W vₜ(x) + (K(a;φ)vₜ)(x) ), for all x ∈ D,
  where K: A × Θ_K → bounded linear operators on U(D;Rᵈᵛ), W: Rᵈᵛ→Rᵈᵛ linear, σ pointwise nonlinear
  activation.
- **Definition 2 (Kernel integral operator K)**, p.4: (K(a;φ)vₜ)(x) := ∫_D κ(x,y,a(x),a(y);φ) vₜ(y) dy,
  where κ_φ: R^{2(d+dₐ)} → R^{dᵥ×dᵥ} is a neural network kernel learned from data (this is the
  "graph neural operator," GNO — Li et al. 2020b, cited as the base neural-operator formulation).
  If we drop dependence on a and impose κ_φ(x,y)=κ_φ(x−y), (3) becomes a convolution.
- **Definition 3 (Fourier integral operator K)**, p.5: parameterize κ_φ directly in Fourier space.
  (K(φ)vₜ)(x) = F⁻¹( R_φ · (F vₜ) )(x), where R_φ is the Fourier transform of a periodic kernel
  κ: D̄ → R^{dᵥ×dᵥ}. Truncate the Fourier series at kmax modes: Z_kmax = {k ∈ Zᵈ : |kⱼ| ≤ kmax,j}.
  R is parameterized directly as a complex-valued (kmax × dᵥ × dᵥ)-tensor (conjugate symmetric since
  κ is real). Note: Z_kmax (a "box" on each axis) is NOT the canonical low-frequency set (which would
  bound the ℓ1-norm of k), but is chosen because it is efficient to implement (p.5).
- **Discrete FFT case** (p.5): with uniform resolution s₁×...×s_d = n, replace F with the standard
  FFT/inverse FFT formulas (explicit double sums given, p.5–6). kmax,j = 12 chosen empirically,
  giving kmax = 12ᵈ parameters per channel (p.6).
- **Parameterizations of R** tried: direct (ϕ_k per mode), linear (depends on Fa), and
  feed-forward-NN. Linear ≈ direct in performance but less efficient; NN parameterization performs
  *worse*, "likely due to the discrete structure of the space Zᵈ" (p.6). Direct parameterization used
  throughout.
- **Complexity** (p.6): weight tensor R has kmax < n modes, so the R-multiplication is O(kmax); the
  dominant cost is the FFT/inverse FFT, giving overall **O(n log n)** (quasi-linear), versus O(n²) for
  general (non-FFT) Fourier transforms and O(nkmax) if truncated but done densely. Uniform
  discretization is required for the FFT.
- **Why discretization-invariant** (p.6): parameters live in Fourier space; evaluating in physical
  space just means projecting onto basis functions e^{2πi⟨x,k⟩}, which are defined everywhere on Rᵈ —
  not tied to any particular mesh.

### Architecture details

- Full architecture (Figure 2a, p.4): input a → **lifting** P (shallow fully-connected net, local/
  pointwise) → v₀ → T=4 **Fourier layers** (integral operator + activation) → vT → **projection** Q
  (local net) → output u.
- **Fourier layer** (Figure 2b, p.4): input v → apply FFT F → linear transform R on the lower
  Fourier modes (filtering out higher modes) → inverse FFT F⁻¹ → this is added to a parallel local
  linear transform W applied directly on v (skip/bias path) → sum passed through nonlinearity σ.
  The W-branch (a pointwise/local linear "bias" term) is what lets FNO handle **non-periodic boundary
  conditions** despite using the (periodic) Fourier transform for the main branch — "the linear
  transform W (the bias term) keeps track of non-periodic boundary" (p.9). Both Darcy Flow (Dirichlet
  BC) and Navier-Stokes' time domain (non-periodic) are learned well despite this.
- Hyperparameters used across experiments (p.6, Section 5): 4 stacked Fourier layers, ReLU
  activation, batch normalization, N=1000 training / 200 testing instances (unless noted), Adam
  optimizer, 500 epochs, initial LR 0.001 halved every 100 epochs. kmax,j = 16, dᵥ=64 for the 1-D
  problem (Burgers); kmax,j = 12, dᵥ=32 for the 2-D problems (Darcy, Navier-Stokes). Lower-resolution
  data are downsampled from higher-resolution ground truth. All computation on a single Nvidia V100
  GPU (16GB).
- **FNO-2D vs FNO-3D** for time-dependent Navier-Stokes (p.8): FNO-2D does 2-D convolution in space
  with an RNN-like structure in time (maps prior 10 timesteps → next timestep, iterated); FNO-3D
  convolves directly in space-time (3-D functions to 3-D functions, mapping initial steps to the
  full trajectory at once). Authors "find the 3-d method to be more expressive and easier to train"
  (p.8) than its RNN-structured counterpart, though 2D-RNN can extrapolate to arbitrary T in fixed
  Δt increments while 3D is fixed to interval [0,T] but flexible in time-discretization within it.

### Experiments (all details, exact numbers)

**1. Burgers' equation** (1-D, p.7, Section 5.1): ∂ₜu + ∂ₓ(u²/2) = ν∂ₓₓu, x∈(0,1), t∈(0,1], periodic
BC, viscosity ν=0.1. Learn u₀ ↦ u(·,1). Data: initial condition u₀ ~ N(0, 625(−Δ+25I)⁻²), solved by
Fourier split-step method on an 8192-point (2¹³) mesh; other resolutions subsampled (Appendix A.3.1,
p.14). N=1000 train / 200 test.
- Table 3 results (relative L2 error) at resolutions s = 256, 512, 1024, 2048, 4096, 8192:
  - NN: ~0.45–0.48 (roughly flat, poor — no neighbor info)
  - GCN: ~0.40–0.42
  - FCN: 0.0958 → 0.3238 (error **grows** with resolution — not discretization invariant)
  - PCANN: ~0.039 (flat)
  - GNO: 0.0555 → 0.0699 (Nyström-sampled neural operator)
  - LNO (low-rank): 0.0212 → 0.0189 (their best non-FNO)
  - MGNO: 0.0243 → 0.0364
  - **FNO: 0.0149 → 0.0139** (lowest error, flat across resolutions)
  - Contribution bullet (p.3): FNO error is "30% lower" than best baseline on Burgers.

**2. Darcy Flow** (2-D steady state, p.7–8, Section 5.2): −∇·(a(x)∇u(x)) = f(x) on (0,1)², u=0 on
boundary, a ∈ L∞(D;R⁺) diffusion coefficient, f≡1 fixed. Learn a ↦ u (nonlinear operator despite
linear PDE). Data: a ~ ψ#N(0,(−Δ+9I)⁻²) with zero Neumann BC on Laplacian; ψ maps positive part of
reals to 12, negative to 3 (a two-phase-like coefficient field). Solved via 2nd-order finite
difference on a 421×421 grid; other resolutions downsampled (Appendix A.3.2, p.14). N=1000/200.
- Table 4 (relative L2 error) at s = 85, 141, 211, 421:
  - NN: 0.1716 (flat, mesh-free but weak)
  - FCN: 0.0253 → 0.1097 (grows with resolution)
  - PCANN: ~0.0298–0.0299
  - RBM: 0.0244 → 0.0259
  - GNO: 0.0346 → 0.0369
  - LNO: 0.0520 → 0.0445 (no data at s=421, marked "−")
  - MGNO: ~0.0416–0.0428
  - **FNO: 0.0108 → 0.0098** — "nearly one order of magnitude lower relative error" than benchmarks
    (p.8). Contribution bullet: "60% lower" error than best baseline (p.3).

**3. Navier-Stokes equation** (2-D, viscous, incompressible, vorticity form, unit torus, p.8):
∂ₜw + u·∇w = νΔw + f, ∇·u=0, w(x,0)=w₀(x). Learn the map from vorticity history up to t=10 to
vorticity at later times up to T>10 (Eq. 8). Data generated with pseudo-spectral method; w₀ ~
N(0, 7^{3/2}(−Δ+49I)^{−2.5}), fixed forcing f(x)=0.1(sin(2π(x₁+x₂))+cos(2π(x₁+x₂))). Data on
256×256 grid, downsampled to 64×64 for these experiments (fixed resolution 64×64 because baselines
are not resolution-invariant). Time step 1e−4 for data generation (2e−2 in MCMC use), solution
recorded every t=1 (Appendix A.3.3, p.15).
- Table 1 (p.8), exact numbers, benchmarked at fixed 64×64 resolution:

  | Method | Params | Time/epoch | ν=1e-3, T=50, N=1000 | ν=1e-4, T=30, N=1000 | ν=1e-4, T=30, N=10000 | ν=1e-5, T=20, N=1000 |
  |---|---|---|---|---|---|---|
  | FNO-3D | 6,558,537 | 38.99s | **0.0086** | 0.1918 | **0.0820** | 0.1893 |
  | FNO-2D | 414,517 | 127.80s | 0.0128 | **0.1559** | 0.0834 | **0.1556** |
  | U-Net | 24,950,491 | 48.67s | 0.0245 | 0.2051 | 0.1190 | 0.1982 |
  | TF-Net | 7,451,724 | 47.21s | 0.0225 | 0.2253 | 0.1168 | 0.2268 |
  | ResNet | 266,641 | 78.47s | 0.0701 | 0.2871 | 0.2311 | 0.2753 |

  Reading: FNO-3D wins when data is abundant (N=1000 at ν=1e-3, or N=10000 at ν=1e-4). When data is
  scarce relative to problem difficulty (ν=1e-4/N=1000, ν=1e-5/N=1000) **all methods exceed 15%
  error**, with FNO-2D lowest among them. Contribution bullet (p.3): "<1% error with viscosity
  ν=1e-3" and "8% error with viscosity ν=1e-4" when learning the entire time series — matches
  0.0086 and 0.0820 in the table above. Contribution bullet: 30% lower error than baselines on
  turbulent Navier-Stokes (ν=1e-4).
  - Baselines: ResNet = 18 layers of 2-D convolution with residual connections; U-Net = 4 blocks of
    2-D conv/deconv; TF-Net = combination of spatial+temporal convolutions designed for turbulence
    (Wang et al. 2020).

**4. Zero-shot super-resolution** (Section 5.4, p.9, Figure 1): train FNO-3D on 64×64×20-resolution
data (ν=1e-4, N=10000), evaluate directly on 256×256×80 resolution with **no retraining and no
higher-resolution data seen**. "Fourier neural operator is the only model among the benchmarks
(FNO-2D, U-Net, TF-Net, and ResNet) that can do zero-shot super-resolution. And surprisingly, it can
do super-resolution not only in the spatial domain but also in the temporal domain" (p.9). This is
presented as a headline result and shown pictorially in Figure 1 (p.2) with ground truth vs.
prediction panels.

**5. Bayesian inverse problem** (Section 5.5, p.9, Figure 6/Appendix A.5 p.16): use function-space
MCMC (preconditioned Crank–Nicolson, Cotter et al. 2013) to sample the posterior of the initial
vorticity in Navier-Stokes given sparse noisy observations at T=50, observed on a 7×7 grid.
25,000 posterior samples (5,000 burn-in), i.e. 30,000 forward-operator evaluations. Compare FNO as
surrogate vs. the traditional (pseudo-spectral) solver, both run on GPU.
- **Timing**: FNO takes 0.005s per evaluation on a 256×256 grid; the traditional solver takes 2.2s
  (even after optimizing its largest stable internal time step). "This amounts to 2.5 minutes for the
  MCMC using FNO and over 18 hours for the traditional solver." Even including 12 hours of offline
  data-generation + training time, FNO is still faster overall for repeated use. FNO and the
  traditional solver "recover almost the same posterior mean," and FNO is differentiable so it can be
  used directly in PDE-constrained optimization without an adjoint solve.
- Contribution bullet (p.3): "inference time of only 0.005s compared to the 2.2s of the pseudo-
  spectral method" on a 256×256 grid — "up to three orders of magnitude faster" than traditional
  solvers (Abstract).

### Spectral analysis (Appendix A.2, p.13)

- Navier-Stokes energy spectrum decays with slope k^{−5/3}, matching the expected turbulence
  spectrum, and this decay does not change appreciably over time (Figure 4, p.14).
- Truncation study (Figure 5, p.14): truncating a Navier-Stokes solution (ν=1e-3) at just 20 Fourier
  modes gives ~2% error, whereas FNO with only kmax,j=12 *parameterized* modes achieves ≤1% error —
  because the nonlinear activations between Fourier layers, plus the final decoder Q, recover high-
  frequency content that the truncated linear R operator alone discards (p.9). This directly answers
  the natural misconception "doesn't truncating Fourier modes throw away high-frequency information
  permanently?" — no, because FNO is not a linear filter; the nonlinearities reconstruct high
  frequencies from the low-frequency + local (W) information.

### Claims about resolution invariance / super-resolution — evidence given

- Figure 3 (p.7) shows FNO's error is flat across resolutions on Burgers and Darcy while FCN's error
  grows with resolution — the direct visual evidence for the discretization-invariance claim.
- Table 1's constraint that "we only present results for spatial resolution 64×64 since all
  benchmarks we compare against are designed for this resolution. Increasing it degrades their
  performance while FNO achieves the same errors" (p.8) — i.e., baselines were *not* even tested at
  higher resolution because they can't handle it; FNO's invariance is asserted qualitatively here
  plus demonstrated via the zero-shot super-resolution experiment.
- Zero-shot super-resolution (Section 5.4, described above) is the flagship empirical demonstration:
  train at 64×64×20, evaluate at 256×256×80, no new data, comparable accuracy (visualized in Figure 1
  ground-truth-vs-prediction panels).

### Stated limitations / open problems (Section 6, Discussion and Conclusion, p.9)

- **Data requirements**: "for more challenging PDEs, generating a few training samples can be
  already very expensive" — learning ν=1e-4 Navier-Stokes needed N=10000 training pairs from the
  numerical solver. Proposed future direction: combine neural operators with numerical solvers to
  reduce data requirements.
- **Recurrent structure**: the neural operator's iterative layers could share parameters as a
  recurrent network without hurting performance, but "we did not impose this restriction in the
  experiments" — an unexplored simplification.
- **Beyond PDEs**: operator learning could apply to computer vision (images/video as functions on
  2-D domains + time), an unexplored direction the authors flag as natural given discretization
  invariance.
- Implicit limitation revealed by Table 1: FNO (and everyone else) still exceeds 15% error in the
  data-scarce, highly turbulent regimes (ν=1e-4/N=1000 and ν=1e-5/N=1000) — the method's headline
  accuracy claims depend on having enough training data for the given viscosity/turbulence level.

### Teachable moments

- **Core conceptual leap**: moving from "learn a function that solves one PDE instance" (PINNs,
  Neural-FEM) to "learn the *operator* mapping the whole family of instances to their solutions."
  This reframes generalization: instead of retraining for every new coefficient/initial condition,
  you get a single trained network usable for any new instance via a forward pass (p.2 "Neural
  Operators" paragraph).
- **Convolution theorem is doing all the work**: parameterizing a translation-invariant integral
  kernel κ(x−y) is mathematically equivalent (by the convolution theorem) to pointwise multiplication
  in Fourier space — this is *why* FFT gives near-linear-time evaluation of what would otherwise be
  an O(n²) integral (Section 4, p.5). A common misconception to preempt: students may think "Fourier
  layer" means "do a Fourier transform of the whole network"; it's specifically that the *kernel* of
  a single integral-operator layer is convolutional and therefore diagonal in Fourier space.
  a Fourier layer is one *layer type* within an otherwise standard lift → iterate → project
  architecture, not a replacement for the whole network.
- **Truncating Fourier modes ≠ discarding high frequencies from the model's outputs.** As shown by
  the spectral analysis (Appendix A.2), the nonlinear activations between layers regenerate high-
  frequency content; only kmax,j=12 modes are directly parameterized yet ≤1% error is achieved on
  turbulent flow with substantial high-wavenumber energy. This is worth flagging explicitly as a
  common misconception.
- **Discretization invariance is a property of the *representation*, not a trick.** Because R is
  learned in Fourier space and Fourier basis functions e^{2πi⟨x,k⟩} are defined everywhere in Rᵈ (not
  tied to grid points), evaluating at new points/resolutions is a change of *basis evaluation*, not
  an approximation or interpolation step (p.6).
- **The role of the "bias" term W** is often missed: FFT-based methods normally assume periodicity,
  yet FNO handles non-periodic BCs (Darcy's Dirichlet BC, Navier-Stokes' non-periodic time axis) via
  the parallel local linear transform W acting directly in physical space, which "keeps track of"
  the non-periodic boundary (p.9). This is a good point to contrast against naive expectations that
  FFT-based architectures can only work on periodic domains.
- Good exercise/discussion point: why does FNO need MORE parameters at small parameter budgets than
  GNO/LNO/MGNO to hit low error (this shows up more explicitly in the JMLR paper's Table 8,
  "Refinability" — see below) — i.e. FNO lacks a "kernel sub-network," so its expressivity comes
  from the number of retained Fourier modes and channel width rather than a flexible kernel network.

### Direct quotes (short, with page numbers)

- "The Fourier neural operator is the first ML-based method to successfully model turbulent flows
  with zero-shot super-resolution." (Abstract, p.1)
- "It is up to three orders of magnitude faster compared to traditional PDE solvers." (Abstract, p.1)
- "By construction, the method shares the same learned network parameters irrespective of the
  discretization used on the input and output spaces." (p.3)
- "The Fourier neural operator is the only model among the benchmarks (FNO-2D, U-Net, TF-Net, and
  ResNet) that can do zero-shot super-resolution. And surprisingly, it can do super-resolution not
  only in the spatial domain but also in the temporal domain." (p.9)
- "This, however, does not mean that the Fourier neural operator can only approximate functions up
  to kmax,j modes." (p.9, on the spectral analysis / recovered high-frequency content)
- "This amounts to 2.5 minutes for the MCMC using FNO and over 18 hours for the traditional solver."
  (p.9)

---

## PAPER 2: Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs (JMLR 2023)

This is the extended, unifying journal version of the neural-operator research program: it gives the
general framework (of which FNO above is one instantiation), a formal definition of discretization
invariance with a proof that neural operators satisfy it, universal approximation theorems, and four
concrete parameterizations (GNO, LNO, MGNO, FNO) compared head-to-head. 97 pages; body runs to p.59,
followed by references and seven proof appendices (A–G) through p.97.

### Plain-language summary

Standard neural networks map finite vectors to finite vectors, so a network trained on one grid
resolution generally cannot be applied, unchanged, to data on a different grid — you would need a new
architecture or retraining. This paper formalizes what it *would* mean for a learning architecture to
be "discretization-invariant" (three explicit requirements: works on any set of input points, can be
queried at any output point, and converges to a genuine continuum operator as the mesh is refined —
Section 1.1, p.2) and proves that a broad class of "neural operator" architectures satisfies this
formal definition, while ordinary neural networks, GNNs, and transformers do not (Table 1, p.5).
Architecturally, a neural operator is a stack of layers, each of which sums a local linear operator, a
non-local integral kernel operator, and a bias function, followed by a fixed pointwise nonlinearity —
directly generalizing the "linear layer + nonlinearity" recipe of ordinary neural networks to
function space (Eq. 6, p.10). The paper offers four practical ways to compute the expensive integral
kernel term efficiently: a Nyström/graph-based approximation (GNO), a low-rank tensor-product kernel
(LNO), a multi-scale/multipole decomposition (MGNO), and a Fourier-space convolution (FNO, the same
architecture as the companion ICLR paper). It also proves a universal approximation theorem: neural
operators can approximate any continuous operator between quite general Banach spaces (Lebesgue,
Sobolev, or continuous-function spaces) to arbitrary accuracy, uniformly over compact sets, and also
in an average (Bochner-norm) sense for random inputs. Numerically, across the Poisson, Darcy,
Burgers, and Navier-Stokes equations, FNO is generally the most accurate and fastest of the four
proposed methods, though the paper is candid that FNO needs many more parameters than the kernel-
network-based methods (GNO/LNO/MGNO) to reach a given error level, and that a brute-force O(J²)
integration still beats FNO in raw expressiveness (by ~40%) at much higher cost.

### Key definitions and equations (with page numbers)

- **Discretization invariance, formal requirements** (Section 1.1, p.2): a fixed-parameter model
  must (1) accept any discretization/point-set of the input function, (2) be evaluable at any point
  of the output domain, (3) converge to a continuum operator as the discretization is refined.
  Transformers and GNNs satisfy (1)-(2) ("resolution invariant") but *fail* (3) — they don't
  converge to a continuum limit (p.2).
- **Table 1** (p.5) — comparison of model classes:

  | Property | NNs | DeepONets | Interpolation | Neural Operators |
  |---|---|---|---|---|
  | Discretization Invariance | ✗ | ✗ | ✓ | ✓ |
  | Output is a function? | ✗ | ✓ | ✓ | ✓ |
  | Query output at any point? | ✗ | ✓ | ✓ | ✓ |
  | Take input at any point? | ✗ | ✗ | ✓ | ✓ |
  | Universal Approximation | ✗ | ✓ | ✗ | ✓ |

  "Neural Operators are the only known class of models that guarantee both discretization-invariance
  and universal approximation." (p.3)
- **Definitions 1–2** (p.8): a *discrete refinement* of D⊂Rᵈ is a nested sequence D₁⊂D₂⊂...⊂D whose
  union of ε-balls eventually covers D for every ε>0; each Dₗ (|Dₗ|=L) is a *discretization* of D.
- **Definition 3** (p.8): discretized uniform risk R_K(G, Ĝ, D_L) = sup_{a∈K} ‖Ĝ(D_L, a|_{D_L}) −
  G(a)‖_U for compact K⊂A.
- **Definition 4 (Discretization invariance, formal)**, p.9: G: A×Θ→U is discretization-invariant if
  there's a sequence of finite-dim maps Ĝ_L: R^{Ld}×R^{Lm}×Θ→U such that for any θ and compact K⊂A,
  lim_{L→∞} R_K(G(·,θ), Ĝ_L(·,·,θ), D_L) = 0.
- **General architecture** (Eq. 6, p.10): Gθ := Q ∘ σ_T(W_{T-1}+K_{T-1}+b_{T-1}) ∘ ... ∘
  σ_1(W_0+K_0+b_0) ∘ P, where P,Q are local lifting/projection maps, W_t local linear operators
  (matrices), K_t non-local integral kernel operators (function-space analog of a weight matrix),
  b_t bias *functions* (not vectors), σ_t fixed pointwise activations. The paper explicitly frames
  bias-as-function and the extra local-linear-operator term as inspired by ResNet's residual
  structure (p.10).
- **Three versions of the integral kernel operator K_t** (Eqs. 7–9, p.10):
  1. (Eq. 7) plain kernel κ(x,y): (K_t v_t)(x) = ∫_{D_t} κ⁽ᵗ⁾(x,y) v_t(y) dν_t(y) — this is the
     version analyzed theoretically (Section 9) and used most in experiments.
  2. (Eq. 8) kernel additionally depends on the input function a at both points, κ(x,y,a(x),a(y)) —
     found empirically to outperform (7) on problems (like Darcy flow) with strong dependence of the
     solution on the parameter a, because otherwise a's influence, entering only via the "initial
     condition," fades with depth.
  3. (Eq. 9) kernel depends on the *current representation* v_t at both points, κ(x,y,v_t(x),v_t(y))
     — this makes the integral operator *nonlinear*, and the paper shows (Proposition 6) that with a
     specific choice of kernel and measure this reduces exactly to the transformer attention
     mechanism.
- **Single hidden-layer example** (Eq. 10–11, p.11): explicitly written out for T=2, showing how P,
  Q, κ⁽⁰⁾, κ⁽¹⁾, b₀, W₀ can each be parameterized (e.g. as small feed-forward nets) and concatenated
  into θ, then optimized by ordinary gradient-based methods after discretizing the loss.
- **Preprocessing trick** (p.12): augmenting the input with the identity coordinate map, i.e. using
  (x, a(x)) instead of just a(x), directly injects domain geometry into the network and eases
  learning; used in all numerical experiments. Derivative/smoothed-input features can also be
  concatenated.
- **Total error decomposition** (p.12): ‖Ĝ_θ(D_L, a|_{D_L}) − G†(a)‖_U ≤ discretization error +
  approximation error — discretization error → 0 as mesh refines (Theorem 8), approximation error →
  0 as network size grows (Theorems 11–14), for a *fixed* parameter set independent of the mesh.

### Four parameterizations (Section 4, pp.12–24) — the core "methods" content

General cost problem (p.13): computing the integral operator naively (Monte Carlo/Riemann sum, Eq.
14) over a J-point mesh costs O(J²) matrix-vector multiplications (treating channel dims as O(1)
constant), since every output point potentially needs contributions from every input point — this
motivates all four schemes below.

**4.1 Graph Neural Operator (GNO)**, pp.14–16
- *Nyström approximation*: subsample J′≪J points, approximate the kernel matrix as a rank-J′ block
  factorization K ≈ K_{JJ'} K_{J'J'} K_{J'J} (Eq. 15). Cost drops to O(J'²).
- *Truncation*: restrict the integral to a ball s(x)=B(x,r)∩D around each x (Eq. 16), giving cost
  O(c_s J²) with c_s≈r^d for D=[0,1]^d — a genuine locality/sparsity structure, not merely a speed
  trick; the paper proves (p.14-15) that composing L such truncated layers (with L chosen so that
  2^{L-1}r ≥ 1) recovers the same expressive power as the full (untruncated) kernel, at total cost
  O(Lr^d J²), which is beneficial (cheaper) whenever r^d(log₂(1/r)+1) < 1 (true for r<1/2 in 1-D, any
  r<1 for d≥2).
- Implemented via **message passing graph neural networks** (Gilmer et al. 2017 framework): treat the
  mesh as a weighted directed graph, node xⱼ holds v(xⱼ), edges connect xⱼ to its neighborhood
  N(xⱼ)=s(xⱼ)∩mesh; averaging aggregation gives u(xⱼ) = (1/|N(xⱼ)|) Σ_{y∈N(xⱼ)} κ(xⱼ,y) v(y) — exactly
  the Monte Carlo estimate of the truncated integral.
- Relation to CNNs (p.15-16): if κ(x,y)=κ(x−y) is compactly supported on B(0,r), Eq. 17 is a genuine
  convolution and, IF r is fixed independent of the mesh, this is a "convolution neural network layer
  consistent in function space." But a *standard* CNN instead fixes the *support in grid points*
  (filter size k) rather than physical radius r — so as J→∞ with fixed filter size k, the physical
  support shrinks to a point (the model becomes purely local in the continuum limit) unless k is also
  grown, in which case the parameter count (kmn) diverges. Conclusion: **standard CNNs are not
  discretization-invariant/consistent in function space** (explicitly demonstrated numerically in
  Section 7).

**4.2 Low-rank Neural Operator (LNO)**, pp.16–17
- Impose κ(x,y) = Σ_{j=1}^r φ⁽ʲ⁾(x)ψ⁽ʲ⁾(y) (tensor-product/finite-rank form). Then u(x) =
  Σⱼ ⟨ψ⁽ʲ⁾,v⟩ φ⁽ʲ⁾(x) — an inner product independent of x, giving **O(rJ) = O(J)** complexity (linear
  in discretization size). Interpretable as applying the truncated SVD of a rank-r operator; kernel
  matrix factorizes as K = K_{Jr}K_{rJ} (Eq. 18 rewritten). Similar in spirit to DeepONet but made
  function-space-consistent.

**4.3 Multipole Graph Neural Operator (MGNO)**, pp.17–20
- Motivated by the classical Fast Multipole Method (FMM): decompose the kernel matrix by interaction
  range, K = K₁+K₂+...+K_L (Eq. 19), short-range K₁ is sparse-but-full-rank, long-range K_L is
  dense-but-low-rank (Figure 3). Build a hierarchy of L discretizations with decreasing node counts
  J₁≥...≥J_L and increasing integration radii r₁≤...≤r_L.
- **Recursive low-rank decomposition** (Eq. 20): K ≈ K₁,₁ + K₁,₂K₂,₂K₂,₁ + K₁,₂K₂,₃K₃,₃K₃,₂K₂,₁ + ...,
  a multi-resolution matrix factorization (Kondor et al. 2014).
- **V-cycle algorithm** (Eqs. 21–25, Figure 4, p.19–20): a downward pass (fine→coarse, Eq. 24) and an
  upward pass (coarse→fine, Eq. 25) per layer, analogous to multigrid methods; a full V-cycle
  computes the decomposition (20).
- Two graph constructions: the *orthogonal multipole graph* (standard uniform-grid FMM construction,
  non-overlapping ranges) and the *generalized random graph* (overlapping ranges, works on arbitrary
  geometry/discretization, can combine with random sampling or active learning for very large J).
- Linear complexity achieved by choosing J_l, r_l so that ΣO(J_l² r_l^d) = O(J); worked example for
  d=2 given (p.20). Combined with Nyström sampling can reach O(J') for J'≪J.

**4.4 Fourier Neural Operator (FNO)**, pp.21–24
- Same construction as the companion ICLR paper: κ(x,y)=κ(x−y) on D=T^d (torus), convolution theorem
  gives u(x)=F⁻¹(F(κ)·F(v))(x); parameterize R_ϕ directly as truncated Fourier coefficients (kmax
  modes per axis), conjugate-symmetric for real outputs. Complexity **O(J log J)** via FFT (dominant
  cost is the FFT itself, since the R-multiplication is only O(kmax)).
- Same three R-parameterizations tested (direct / linear / feed-forward NN) with the same finding:
  direct ≈ linear in accuracy but direct is cheaper; NN parameterization is worse, "likely due to the
  discrete structure of the space Zᵈ; numerical evidence suggests neural networks are not adept at
  handling this structure" (p.23).
- Empirical rule of thumb given here (not in the ICLR paper): "we have found the choice kmax,j
  roughly around 1/3 to 2/3 of the maximum number of Fourier modes in the FFT of the grid valuation
  of the input function provides desirable performance" (p.22); kmax,j=12 used in experiments.
- **Non-uniform / non-periodic geometries** (p.23-24): the fast (FFT-based) FNO implementation
  strictly requires a uniform mesh on the torus (or a square with homogeneous Dirichlet/Neumann BC,
  via fast sine/cosine transform). For more general compact manifolds M, one can embed M into a
  periodic cube/torus via a Fourier continuation; in practice FNO does this simply by **zero-padding**
  the input and computing the loss only on the original (non-padded) region — "the Fourier neural
  operator will automatically generate a smooth extension to the padded domain in the output space"
  (p.24). This is a good concrete answer to "how does an FFT-based method handle non-periodic
  domains at all?"

**4.5 Summary table of the four methods' complexity** (p.24): GNO O(JJ'), LNO O(J), MGNO O(J), FNO
O(J log J).

### Neural operators vs. other deep learning models (Section 5, pp.24–29)

- **Proposition 5** (p.25): a neural operator with a *point-wise parameterized* first kernel, plus
  Monte-Carlo discretization of the integral, reduces exactly to a **DeepONet** (branch net + trunk
  net) — full derivation given (Eqs. 32). But this reduction is explicitly *not* discretization
  invariant, since the branch-net weights wⱼ are tied to specific input evaluation points x₁,...,x_q;
  as the mesh refines, the parameter count would need to grow. The paper's fix ("DeepONet-Operator,"
  Eq. 33) replaces the finite-dimensional inner product ⟨w̃ⱼ,ã⟩ with a genuine function-space inner
  product ⟨wⱼ,a⟩, restoring discretization invariance.
- **Linear vs. nonlinear approximation** (p.26): DeepONet-style constructions are *linear
  approximation* methods (approximating G†(A) by a fixed linear span of basis functions φ₁,...,φ_p);
  their quality is governed by the Kolmogorov n-width, which decays *slowly* for problems like
  advection-dominated flow maps, requiring large n (hence more parameters/data). General neural
  operators (Eq. 6/11) are (in general) *nonlinear* approximation methods, empirically shown (Section
  7) to outperform DeepONets and LNO at matched parameter counts — although the paper's own
  approximation theory (Section 9) is only proved via a linear-approximation reduction and so does
  not yet capture this empirical nonlinear-approximation benefit. Flagged explicitly as an open
  theoretical gap (p.26-27, and again in Section 10.1.3).
- **Function representation comparison** (p.27): GNO/MGNO finite-dimensionalize via pointwise graph
  values; FNO via a uniform-grid Fourier basis; LNO via a Barron-space product form; PCA-operator via
  PCA modes; DeepONet mixes pointwise input (branch) with Barron-space output (trunk); POD-DeepONet
  swaps the trunk for PCA modes. Quote: "'all models are wrong, but some are useful' (Box, 1976)" —
  each finite-dimensionalization trades flexibility for problem-specific bias; FNO is "the most
  specific" (grid + periodicity assumptions) but "usually works out of the box" on such problems,
  needing extra machinery (extension/interpolation/Fourier continuation) elsewhere.
- **Proposition 6** (p.27–29): the *attention mechanism in transformers is a special case of a neural
  operator layer.* Full derivation: choose the nonlinear kernel form (Eq. 9) with
  κ_v(v(x),v(y)) = g_v(v(x),v(y))·R, g_v built from softmax-normalized exponentials of ⟨Av(x),Bv(y)⟩/√m
  (query/key inner products), apply Monte-Carlo/Nyström discretization on k points, and re-parameterize
  R = R_out R_val — this yields exactly u_j = σ(v_j + R_out Σ_q S_j(z_q) R_val v_q), i.e. pre-
  normalization single-head self-attention with no layer norm (Eq. 36 and the display right after). A,
  B, R_val correspond to queries, keys, values. Multi-head attention = sum of several such κ_v terms.
  Caveat: standard attention is "memory and computation intensive... compared to neural operator
  architectures developed here," and many efficient vision transformers (e.g. ViT) are **not** special
  cases of neural operators because they tokenize via CNN patches, which are not discretization
  invariant (p.29).

### Discretization-invariance theorem (Section 9.2)

- **Theorem 8** (p.54, exact statement): Let D⊂Rᵈ, D′⊂Rᵈ′ be domains. Let A, U be real-valued Banach
  function spaces on D, D′ continuously embeddable in C(D̄), C(D̄′) respectively, and σ₁,σ₂,σ₃ ∈ C(R).
  Then for any n∈N, the set of neural operators NOₙ(σ₁,σ₂,σ₃;D,D′) (viewed as maps A→U) is
  discretization-invariant [in the sense of Definition 4]. Proof (Appendix E) constructs finite-
  dimensional Riemann-sum approximations and shows uniform convergence of the error over compact
  sets of A.

### Universal approximation theorems (Section 9.3, exact hypotheses and statements)

- **Function-class assumptions** used throughout (p.55):
  - **Assumption 9** (input space A on Lipschitz domain D⊂Rᵈ): A is one of (1) Lᵖ¹(D), 1≤p₁<∞; (2)
    W^{m1,p1}(D), 1≤p₁<∞, m₁∈N; (3) C(D̄).
  - **Assumption 10** (output space U on Lipschitz domain D′⊂Rᵈ′): U is one of (1) Lᵖ²(D′), m₂=0; (2)
    W^{m2,p2}(D′), m₂∈N; (3) C^{m2}(D̄′), m₂∈N₀.
- **Activation function classes** (p.52-54): A_m = activations making width-arbitrary networks dense
  in C^m(Rᵈ) on compacta (any non-polynomial C^m function qualifies per Pinkus 1999); A^L_m = those
  additionally linearly bounded (ReLU ∈ A^L₀, ELU ∈ A^L₁, tanh/sigmoid ∈ A^L_m for all m); BA =
  activations whose networks can approximate the identity map arbitrarily well on compacts while
  staying globally bounded (ReLU∈A^L₀∩BA with a 3-layer construction, citing Lanthaler et al. 2021).
- **Theorem 11** (p.56, exact statement): Under Assumptions 9–10, if G†:A→U is continuous, σ₁∈A^L₀,
  σ₂∈A₀, σ₃∈A_{m2}, then for any compact K⊂A and 0<ε≤1, there exists N∈N and a neural operator
  G∈NO_N(σ₁,σ₂,σ₃;D,D′) with sup_{a∈K} ‖G†(a)−G(a)‖_U ≤ ε. Moreover if U is a Hilbert space, σ₁∈BA,
  and ‖G†(a)‖_U≤M for all a, then G can be chosen with ‖G(a)‖_U≤4M for all a (a uniform boundedness
  guarantee, not just a local approximation guarantee).
- **Theorem 12** (p.56): extension of Theorem 11 to A=C^{m1}(D̄) (m₁∈N) using the "m1-th order neural
  operators" NO^{m1}_N (which explicitly take in derivatives up to order m₁ as extra input channels,
  since plain kernel integration alone cannot learn differentiation for this input class — see
  discussion just before Theorem 8, p.54).
- **Theorem 13** (p.56): a Bochner-norm (average-case, not just worst-case-over-compacta) version.
  Given a probability measure μ on A and G†∈L²_μ(A;H^{m2}(D)) (μ-measurable), σ₁∈A^L₀∩BA, σ₂∈A₀,
  σ₃∈A_{m2}, then for 0<ε≤1 there exists N and G∈NO_N with ‖G†−G‖_{L²_μ(A;H^{m2}(D))} ≤ ε.
- **Theorem 14** (p.57): the A=C^{m1}(D) analogue of Theorem 13, using m1-th order neural operators.
- **Proof strategy sketch** (p.55, Figure 16): three stages — (i) map input a to a finite vector via J
  linear functionals F:A→R^J (shown approximable by smooth-kernel integration, generalizing Chen &
  Chen 1995's Dirac-measure functionals); (ii) nonlinearly map this finite representation via a
  continuous ψ:R^J→R^{J'} (reduces to a standard finite-dim neural network universal approximation
  argument); (iii) expand the new representation as coefficients against a basis/representer set for
  U via a single IO (integral-operator) layer, using density of continuous functions. This three-step
  structure generalizes Bhattacharya et al. (2020)'s Hilbert-space argument to the broader Banach
  spaces in Assumptions 9–10; the authors state this is, to their knowledge, the first result of this
  generality (p.51-52).
- Explicit theoretical caveat repeated twice (p.51-52, p.58-59, Section 10.1.3): "our approximation
  theory uses the fact that neural operators can be reduced to a linear method of approximation... and
  does not capture any benefits of nonlinear approximation," even though nonlinear approximation
  benefits are exploited by trained networks in practice and observed numerically (Section 7). The
  question of how required parameter count scales with target accuracy, and whether the curse of
  dimensionality can be beaten for specific operator classes, is explicitly left open here (pointing
  to companion papers Lanthaler et al. 2021 and Kovachki et al. 2021 [FNO-specific approximation
  theory] for partial answers). Growth in the number of parameters needed for a given error tolerance
  "may be super-exponential" in general (p.50, "Sources of Error" paragraph).

### Experiments (Section 6 problem definitions + Section 7 results, exact numbers)

Four test problems defined in Section 6 (pp.29–35): 1-D Poisson (Green's function sanity check), 2-D
Darcy Flow, 1-D Burgers, 2-D Navier-Stokes (two solution-operator variants: fixed final time and
trajectory-to-trajectory). All reported errors are Monte-Carlo estimates of relative L2 error
E_{a~μ}[‖G†(a)−G_θ(a)‖_{L2(D)} / ‖G†(a)‖_{L2(D)}]. All computation on a single Nvidia V100 GPU
(16GB); code released at github.com/zongyi-li/graph-pde and github.com/zongyi-li/fourier_neural_operator
(p.35).

**Common setup for the four NO methods** (Section 7, p.35-36): 4 stacked integral-operator layers,
ReLU activation, no batch norm needed (unlike the ICLR FNO paper, which did use batch norm). N=1000
train/200 test, Adam, 500 epochs, LR 0.001 halved every 100 epochs. Channel dims d_{v0}=...=d_{v3}=64
for 1-D problems, =32 for 2-D problems. Kernel networks are 3-layer, width-256 feed-forward nets.
- GNO: truncation radius r=0.25, Nyström with J′=300 subsampled nodes.
- LNO: rank r=4.
- MGNO: on Darcy, random construction with 3 graph levels sampling J₁=400, J₂=100, J₃=25 nodes; on
  Burgers, orthogonal construction without sampling.
- FNO: kmax,j=16 (1-D problems), kmax,j=12 (2-D problems).

**6.1/7.1 Poisson equation** (1-D, Green's function, p.29-30, 36-37): −u''=f on (0,1), u(0)=u(1)=0.
Zero-hidden-layer neural operator (learn only κ_θ:R²→R directly). With N=1000 training examples,
relative test error **10⁻⁷** — "an almost perfect approximation to the true solution operator" (p.36).
Further check: does κ_θ actually approximate the analytic Green's function G(x,y)=½(x+y−|y−x|)−xy?
Figure 7 (p.37) shows learned vs. analytic kernel are visually near-identical, proof-of-concept that
training in the (average-case) Bochner norm also yields approximation in the much stronger *uniform*
topology over bounded sets — i.e. the model generalizes to inputs f well outside the training
distribution's support, even discontinuous f (p.36-37 discussion, with a short supporting
inequality argument).

**6.2/7.2.1 Darcy Flow** (2-D, p.31-32, 37-39): same PDE as in the ICLR paper. Solved via 2nd-order
finite difference on 421×421 grid; N=1000. Table 2 (p.39), relative L2 error at s=85,141,211,421:

  | Method | s=85 | s=141 | s=211 | s=421 |
  |---|---|---|---|---|
  | NN | 0.1716 | 0.1716 | 0.1716 | 0.1716 |
  | FCN | 0.0253 | 0.0493 | 0.0727 | 0.1097 |
  | PCANN | 0.0299 | 0.0298 | 0.0298 | 0.0299 |
  | RBM | 0.0244 | 0.0251 | 0.0255 | 0.0259 |
  | DeepONet | 0.0476 | 0.0479 | 0.0462 | 0.0487 |
  | GNO | 0.0346 | 0.0332 | 0.0342 | 0.0369 |
  | LNO | 0.0520 | 0.0461 | 0.0445 | — |
  | MGNO | 0.0416 | 0.0428 | 0.0428 | 0.0420 |
  | **FNO** | **0.0108** | **0.0109** | **0.0109** | **0.0098** |

  Note (p.38): a feature-engineered DeepONet variant (CNN branch + PCA trunk, from Lu et al. 2021b)
  can reach 0.0232 — about half FNO's error — but only "for a very coarse grid with s=29," i.e. not
  a like-for-like comparison at the resolutions tested here.

**6.3/7.2.2 Burgers' equation** (1-D, p.32, 39): same PDE/data-generation as ICLR paper (ν=0.1, 8192-
point spectral solve, N=1000). Table 3 (p.39), relative L2 error at s=256...8192:

  | Method | s=256 | s=512 | s=1024 | s=2048 | s=4096 | s=8192 |
  |---|---|---|---|---|---|---|
  | NN | 0.4714 | 0.4561 | 0.4803 | 0.4645 | 0.4779 | 0.4452 |
  | GCN | 0.3999 | 0.4138 | 0.4176 | 0.4157 | 0.4191 | 0.4198 |
  | FCN | 0.0958 | 0.1407 | 0.1877 | 0.2313 | 0.2855 | 0.3238 |
  | PCANN | 0.0398 | 0.0395 | 0.0391 | 0.0383 | 0.0392 | 0.0393 |
  | DeepONet | 0.0569 | 0.0617 | 0.0685 | 0.0702 | 0.0833 | 0.0857 |
  | GNO | 0.0555 | 0.0594 | 0.0651 | 0.0663 | 0.0666 | 0.0699 |
  | LNO | 0.0212 | 0.0221 | 0.0217 | 0.0219 | 0.0200 | 0.0189 |
  | MGNO | 0.0243 | 0.0355 | 0.0374 | 0.0360 | 0.0364 | 0.0364 |
  | **FNO** | **0.0018** | **0.0018** | **0.0018** | **0.0019** | **0.0020** | **0.0019** |

  FNO test error here (0.0018) is even lower than the equivalent ICLR-paper table (0.0149-0.0139) —
  note FNO's mean *training* error is 0.0012, std 0.0010; switching ReLU→GeLU further drops test
  error 0.0018→0.0007 (p.39). PCA-enhanced DeepONet (Lu et al. 2021b) reaches 0.0194 at s=128 (again
  a different/coarser grid, not directly comparable).

**7.2.3 Zero-shot super-resolution** (Darcy, p.39-40, Figure 9): GNO (not FNO here) trained on 16×16
resolution, evaluated on 241×241 — demonstrating that the graph-based method also generalizes across
resolution, with pointwise absolute-squared-error maps shown.

**6.4/7.3 Navier-Stokes** (p.33-35, 40-46): same PDE as ICLR paper, vorticity-streamfunction
formulation (Eqs. 44a-c), same forcing g(x)=0.1(sin(2π(x₁+x₂))+cos(2π(x₁+x₂))). Reynolds number
formula given explicitly: **Re = √0.1 / (ν·(2π)^{3/2})** (Chandler and Kerswell 2013, p.34) — this
lets students convert the paper's viscosities to Reynolds numbers (e.g. this underlies the paper's
"<1% error at Re=20, 8% error at Re=200" headline claim in the Introduction, p.3, which corresponds
to ν=1e-3 and ν=1e-4 respectively). Data: 256×256 pseudo-spectral grid, Crank–Nicolson (viscous
term) + Heun's method (nonlinear/forcing), 2/3 dealiasing rule.
- Table 4 (p.41) is numerically identical to Table 1 in the ICLR paper (FNO-3D/FNO-2D/U-Net/TF-Net/
  ResNet at 64×64, four (ν,T,N) configurations) — same headline numbers (FNO-3D: 0.0086 at ν=1e-3;
  FNO-2D: 0.1559 at ν=1e-4/N=1000, etc.)
- **Resolution study, Table 5 (p.42)** — NOT present in the ICLR paper — tests ν=1e-3, N=200, T=20 at
  s=64,128,256:

  | Method | s=64 | s=128 | s=256 |
  |---|---|---|---|
  | FNO-3D | 0.0098 | 0.0101 | 0.0106 |
  | FNO-2D | 0.0129 | 0.0128 | 0.0126 |
  | U-Net | 0.0253 | 0.0289 | 0.0344 |
  | TF-Net | 0.0277 | 0.0278 | 0.0301 |

  This is the JMLR paper's most direct quantitative evidence for resolution-invariance: FNO-2D/3D
  error stays essentially flat (0.0098→0.0106 for FNO-3D) while U-Net and TF-Net degrade measurably
  as resolution increases (0.0253→0.0344 and 0.0277→0.0301).
- **Zero-shot super-resolution** (7.3.1, p.42, Figure 11): identical experiment to the ICLR paper —
  FNO-3D trained on 64×64×20 (ν=1e-4≈Re200, N=10000), evaluated on 256×256×80, "the only model among
  the benchmarks... that can do zero-shot super-resolution," in both space and time.
- **Spectral analysis** (7.3.2, p.42-43, Figures 12-13): all methods capture the −5/3 spectral decay
  qualitatively, but FNO recovers high-wavenumber content beyond its truncated kmax,j=12 modes via
  nonlinearities (same finding/numbers as ICLR paper: 20-mode truncation alone gives ~2% error, full
  FNO gives ≤1%).
- **Non-periodic boundary condition** (7.3.3, p.43): same explanation as ICLR paper — the W (bias/
  local-linear) term lets FNO handle Darcy's Dirichlet BC and Navier-Stokes' non-periodic time axis.
- **Bayesian inverse problem** (7.3.4, p.44, Figure 14): same setup/numbers as the ICLR paper's
  Section 5.5 — pCN MCMC, 25,000 samples (5,000 burn-in), 30,000 forward evaluations, FNO 0.005s vs.
  solver 2.2s per evaluation, 2.5 minutes (FNO) vs. 18+ hours (solver) total MCMC time, 12 hours
  offline data-gen+training cost still leaves FNO net faster for repeated use.

**7.4 Discussion/comparison of the four methods** (pp.45-47) — unique to the JMLR paper, very useful
teaching material:
- **Table 6, Ingenuity** (p.45): GNO = Nyström approximation (graph-based: yes, kernel network: yes);
  LNO = low-rank approximation (graph-based: no, kernel network: yes); MGNO = multi-level graphs on
  GNO (graph-based: yes, kernel network: yes); FNO = convolution theorem/Fourier features
  (graph-based: no, kernel network: **no** — FNO has no learned kernel sub-network at all, which is
  why it's fast but needs many parameters, see Table 8 below).
- **Expressiveness** (7.4.2, p.45): "The full O(J²) integration always has the best results, but it
  is usually too expensive." Ranking of the four: GNO good but hurt by subsampling; LNO best on 1-D
  (Burgers) but struggles on 2-D since it has no built-in sampling speed-up; MGNO gets benefits of
  both via its multi-level structure; **FNO has overall the best performance and is the only method
  that captures the challenging (turbulent) Navier-Stokes equation** (explicit summary statement,
  p.45).
- **Table 7, Complexity/wall-clock** (p.46): GNO O(J'²r²), 4s/epoch; LNO O(J), 20s/epoch; MGNO
  ΣO(J_l²r_l²)~O(J), 8s/epoch; FNO O(J log J), 4s/epoch — measured on a single V100. Note LNO is
  algorithmically O(J) but is the *slowest in wall-clock* here; FNO is both asymptotically near-
  linear and fastest in practice "because it doesn't have the kernel network κ." MGNO is "relatively
  slower because of its multi-level graph structure" despite also being ~O(J).
- **Table 8, Refinability** (p.46) — relative error on Darcy Flow vs. number of parameters (rounded
  to nearest 0.05), at parameter budgets 10³,10⁴,10⁵,10⁶:

  | Method | 10³ params | 10⁴ | 10⁵ | 10⁶ |
  |---|---|---|---|---|
  | GNO | 0.075 | 0.065 | 0.060 | 0.035 |
  | LNO | 0.080 | 0.070 | 0.060 | 0.040 |
  | MGNO | 0.070 | 0.050 | 0.040 | 0.030 |
  | FNO | **0.200** | 0.035 | 0.020 | 0.015 |

  Key teaching point (explicit in text, p.46): "Because GNO, LNO, and MGNO have the kernel networks,
  the slope of their error rates are flat: they can work with a very small number of parameters. On
  the other hand, FNO does not have the sub-network. It needs a larger magnitude of parameters to
  obtain an acceptable error rate" — i.e. FNO is data/parameter-hungry at very small budgets (0.200
  error at 10³ params, far worse than the others) but overtakes them once given enough parameters
  (10⁴ and up). This directly nuances/complicates the simple "FNO is just better" reading of the
  headline result tables.
- **Robustness to noise** (7.4.5, p.46-48, Table 9, Figure 15): noise model a′(x)=a(x)+0.1·‖a‖_∞ ξ,
  ξ~N(0,1) i.i.d. per grid point. Four test problems: Burgers, an added **1-D advection equation**
  (new, not in the ICLR paper; input is a random square wave, following Lu et al. 2021b), Darcy,
  Navier-Stokes.

  | Problem | Train err | Test (clean) | Test (noisy) |
  |---|---|---|---|
  | Burgers | 0.002 | 0.002 | 0.018 |
  | Advection | 0.002 | 0.002 | 0.094 |
  | Darcy | 0.006 | 0.011 | 0.012 |
  | Navier-Stokes | 0.024 | 0.024 | 0.039 |
  | Burgers (train w/ noise) | 0.011 | 0.004 | 0.011 |
  | Advection (train w/ noise) | 0.020 | 0.010 | 0.019 |
  | Darcy (train w/ noise) | 0.007 | 0.012 | 0.012 |
  | Navier-Stokes (train w/ noise) | 0.026 | 0.026 | 0.025 |

  Finding: FNO is fairly robust to test-time noise on smoothing operators (Darcy, Navier-Stokes stay
  under 10% error even with 10% input noise) but noticeably less robust on advection (up to 9.4%
  error, since advection is non-smoothing and preserves discontinuities — "this is a hard problem for
  FNO since it has discontinuities; similar issues arise when using spectral methods for conservation
  laws," p.48) and somewhat on Burgers (steep fronts). Training with noise closes the clean/noisy gap
  almost entirely but slightly degrades clean-data performance (a explicit bias-variance style
  trade-off, p.47-48). A cited fix for the advection/discontinuity weakness: Wen et al. (2021) compose
  FNO with a CNN/U-Net branch, which helps on sharp-shock multiphase flow but "takes the method out of
  the realm of discretization-invariant methods" — the paper flags designing discretization-invariant
  discontinuity-handling tools as an open problem (p.48).

### Literature review section highlights (Section 8, pp.47–51) — good for situating the field

- Finite-dimensional CNN-type operator surrogates (Guo et al. 2016; Zhu & Zabaras 2018, etc.): not
  mesh-independent by construction.
- **DeepONet** (Lu et al. 2019, 2021a), building on Chen & Chen (1995)'s shallow universal
  approximation theorem for operators: branch net (input functions) + trunk net (query locations);
  Lanthaler et al. (2021) give a DeepONet error estimate and, notably, are the first to address and
  partially resolve the curse of dimensionality for specific operator-learning problems.
- PINNs/Deep Ritz/Deep Galerkin: parameterize the *solution* directly, not the operator — must
  retrain per input instance, closely analogous to classical finite-element style methods but with a
  neural-network basis.
- Reduced Basis Methods (RBM) are the closest classical relative; neural operators generalize RBM-
  style ideas to a purely data-driven, non-intrusive (no PDE knowledge required) supervised setting.
- Explicit statement that neural operators are "the first practical supervised learning methods
  designed to learn maps between infinite-dimensional spaces," alongside contemporaneous work
  (Bhattacharya et al. 2020, Nelsen & Stuart 2021, others) (p.49).
- **Sources of error, explicitly enumerated** (p.50-51): (1) operator-approximation error (addressed
  by Section 9's universal approximation theorems — driven to 0 by adding parameters); (2)
  discretization error (addressed by Theorem 8 — driven to 0 by refining the mesh); (3) empirical-
  risk/finite-training-sample error; (4) optimization error (failure to reach the global minimizer of
  the training loss). The paper only theoretically analyzes (1) and (2); (3) and (4) are explicitly
  left unaddressed — important to flag to students so they don't over-read the theory section as a
  full sample-complexity or optimization guarantee.

### Stated limitations and open problems (Section 10, Conclusions/Future Directions, pp.57–59)

- **10.1.1 New applications**: extending beyond Darcy/Burgers/Navier-Stokes to subgrid turbulence
  models for climate GCMs, high-contrast geological media, general physics simulation for games/VFX,
  and computer vision (images/video as functions on spatiotemporal domains) — all flagged as
  unexplored future directions, not results already shown.
- **10.1.2 New methodologies**: "the full O(J²) integration method still outperforms the FNO by about
  40%, albeit at greater cost" — an explicit, quantified admission that FNO/the fast approximations
  sacrifice some accuracy for speed relative to the "ideal" untruncated integral operator. Suggested
  directions: adaptive graphs/probability estimation in the Nyström step; alternative bases (PCA,
  Chebyshev) instead of Fourier; combining neural operators with classical solvers (hybrid
  solve-and-correct) or with explicit physics constraints (physics-informed operator learning).
- **10.1.3 Theory** (repeated emphasis): the universal approximation theorems here rely on *linear*
  approximation and do not explain the empirically-observed benefits of nonlinear approximation or of
  network depth (the experiments use 4-5 layers because it works better, but this isn't captured by
  the theory). Open questions explicitly listed: how should architecture/parameter scaling depend on
  which Banach space the true operator lives in, and which classes of PDEs can be approximated
  efficiently (i.e., without the curse of dimensionality) by which of the four parameterizations.

### Teachable moments (JMLR paper specifically, beyond what's covered in the FNO notes above)

- **Discretization invariance is a three-part formal definition, not a vibe.** Many learners
  conflate "works on any grid" (resolution invariance, which transformers/GNNs also have) with true
  discretization invariance, which additionally requires convergence to a fixed continuum operator as
  the mesh refines, with parameter count bounded independent of mesh size (Definition 4, p.9). This
  is the paper's central conceptual contribution and a good place to slow down and work through
  Table 1 (p.5) property by property.
- **All the tricks (Nyström, low-rank, multipole, Fourier) are answers to the same cost problem**:
  the naive integral-operator layer costs O(J²). Teaching this section as "four different ways to
  approximate a dense J×J kernel matrix cheaply," with explicit ties to classical numerical linear
  algebra (Nyström method, SVD/low-rank matrices, fast multipole method, FFT-diagonalized
  convolution), helps students see the architecture choices as principled rather than ad hoc.
- **Transformers-as-neural-operators (Proposition 6) is a genuinely surprising and pedagogically rich
  result**: it reframes self-attention as a *nonlinear*, Nyström-discretized integral kernel operator
  with a data-dependent (not translation-invariant) kernel. Good discussion question: why does this
  mean vanilla self-attention is NOT discretization-invariant in the strict sense the paper defines
  (hint: standard positional encodings and the softmax normalization over exactly k discrete tokens
  break convergence to a fixed continuum limit unless handled carefully) — the paper itself notes
  ViT-style tokenization via CNN patches breaks the correspondence (p.29).
- **DeepONet-as-neural-operator (Proposition 5) is the natural complementary result**: a specific,
  point-wise-parameterized, discretization-*dependent* instantiation of the general neural-operator
  template. Good exercise: have students identify exactly which architectural choice (point-wise
  parameterization of w_j tied to fixed grid points x₁,...,x_q) is the one that breaks discretization
  invariance, and how "DeepONet-Operator" (Eq. 33) fixes it by replacing a finite inner product with a
  function-space inner product.
- **FNO's parameter-hungriness at small budgets (Table 8) is a genuinely underappreciated caveat**
  relative to how FNO is usually presented (as simply "the best" method) — worth pairing directly with
  the headline error tables (2, 3, 4) so students don't walk away thinking FNO dominates GNO/LNO/MGNO
  unconditionally.
- **The advection-equation robustness failure (Table 9) is a clean, minimal counterexample** to a
  naive "FNO handles everything" takeaway — it isolates the mechanism (Fourier truncation +
  smoothing bias struggles with genuine discontinuities/non-smoothing dynamics) in a single simple
  1-D PDE, distinct from the higher-dimensional turbulence examples where FNO shines.
- **The four explicitly-listed sources of error** (p.50-51: approximation, discretization, finite-
  sample, optimization) is a useful general-purpose checklist for evaluating *any* claimed universal
  approximation result in operator learning — the theorems in this paper only control 2 of the 4.

### Direct quotes (short, with page numbers)

- "Neural Operators are the only known class of models that guarantee both discretization-invariance
  and universal approximation." (p.3)
- "families of graph neural networks... and transformer models... are resolution invariant... but
  they fail to converge to a continuum operator as discretization is refined." (p.2)
- "The attention mechanism in transformer models is a special case of a neural operator layer."
  (Proposition 6, p.27)
- "A neural operator with a point-wise parameterized first kernel and discretized integral operators
  yields a DeepONet." (Proposition 5, p.25)
- "Because GNO, LNO, and MGNO have the kernel networks, the slope of their error rates are flat: they
  can work with a very small number of parameters. On the other hand, FNO does not have the
  sub-network. It needs at a larger magnitude of parameters to obtain an acceptable error rate."
  (p.46, Table 8 discussion)
- "FNO has overall the best performance. It is also the only method that can capture the challenging
  Navier-Stokes equation." (p.45)
- "our approximation theory uses the fact that neural operators can be reduced to a linear method of
  approximation... and does not capture any benefits of nonlinear approximation." (p.52, repeated
  p.58)
- "the full O(J²) integration method still outperforms the FNO by about 40%, albeit at greater cost."
  (p.58)
- "The advection problem is a hard problem for the FNO since it has discontinuities; similar issues
  arise when using spectral methods for conservation laws." (p.48)

---

## Cross-paper notes for course design

- The ICLR 2021 FNO paper is best taught as the *motivating instance*: a single concrete, fast,
  empirically dominant architecture with a clear headline story (zero-shot super-resolution, 1000x
  speedup, turbulent Navier-Stokes). The JMLR 2023 paper is the *unifying theory and comparison*
  paper: it defines the general neural-operator template, proves the two central theorems
  (discretization invariance, Theorem 8; universal approximation, Theorems 11-14), and honestly
  reports FNO's weaknesses (parameter hunger at small budgets, poor robustness on non-smooth/
  discontinuous problems like advection) that the ICLR paper does not surface.
- Recommended teaching order: (1) motivate the operator-learning problem and discretization
  invariance requirement via Table 1 and Definition 4 of the JMLR paper; (2) introduce the general
  architecture (Eq. 6) and the O(J²) cost problem; (3) walk through GNO → LNO → MGNO → FNO as four
  answers to that cost problem, using the JMLR paper's Section 4 and Table 6/7; (4) use the ICLR
  paper's Figures 1-3 and zero-shot super-resolution result as the "why this matters" payoff; (5)
  close with the JMLR paper's Table 8 (refinability) and Table 9 (advection robustness) as correctives
  to an overly rosy picture of FNO, plus the universal approximation theorem statements (Theorems
  8, 11-14) and their explicit linear-approximation caveat as the theoretical capstone.
- Numbers that appear in BOTH papers (verified identical): Table 1 (ICLR) = Table 4 (JMLR) for
  Navier-Stokes fixed-resolution benchmarks; Table 3/4 (ICLR) = Table 3/2 (JMLR) for Burgers/Darcy;
  the Bayesian inverse problem timing numbers (0.005s vs 2.2s, 2.5 min vs 18 hours); the zero-shot
  super-resolution experiment (64x64x20 to 256x256x80). This shared numerical core confirms the JMLR
  paper is the "director's cut" of the ICLR paper's FNO results, extended with three additional
  methods (GNO, LNO, MGNO), the Poisson/Green's-function sanity check, the advection robustness test,
  the resolution study (Table 5), and the full universal approximation theory (Section 9).

### Correction added 2026-09-06 (lab session, verified against the PDFs on disk)
- Burgers domain: the ICLR text (2010.08895v3.pdf, Section 5.1) writes x in (0,1). The JMLR text
  (21-1524.pdf p.32, Section 6.3, eq. 41) writes x in (0, 2 pi) with the same nu = 0.1 and the same
  measure N(0, 625(-d2/dx2 + 25I)^-2). The two are not equivalent. On (0,1) with nu = 0.1 the k=1
  Fourier mode decays by exp(-0.1 (2 pi)^2) = 0.019 by t = 1, so u(., 1) is nearly constant and the
  Table 3 errors could not arise. On (0, 2 pi) the k=1 mode decays by exp(-0.1) = 0.905 and fronts
  form. Lab 01 run B (labs/01_burgers/run_b_paper_spec.py) uses (0, 2 pi). The line at :93 above
  should read (0, 2 pi).
- JMLR p.32 also states the solver: pseudo-spectral split step, heat part exact in Fourier space,
  nonlinear part forward Euler with a very small step, 2^13 = 8192 points, other resolutions
  subsampled, N = 1000 training examples.
