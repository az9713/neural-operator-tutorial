# Existing Teaching Material — Neural Operators / SciML (survey, 2026-09-05)

All URLs below were fetched (WebFetch or WebSearch-verified). Items I could not load directly are marked UNVERIFIED.

---

## 1. ETH Zurich — "AI in the Sciences and Engineering" (401-4656-21L)

**This is the closest existing course to the one being planned.**

- Course page: https://camlab-ethz.github.io/ai4s-course/ (fetched)
- Repo: https://github.com/camlab-ethz/AI_Science_Engineering (fetched)
- Prior-year page (2023, as "Deep Learning in Scientific Computing"): https://camlab.ethz.ch/teaching/deep-learning-in-scientific-computing-2023.html (fetched) — same instructor, earlier name of the course lineage.
- Lecture recordings, 2024 playlist: https://www.youtube.com/playlist?list=PLJkYEExhe7rYFkBIB2U5pf_RWzYnFLj7r
- Institution: ETH Zurich. Instructors: Prof. Siddhartha Mishra (lead), Dr. Ben Moseley (co-instructor in earlier editions), David Graber; TAs Bogdan Raonic, Shizheng Wen.
- Year: Fall 2025 (HS2025) is the current listed edition; runs annually.
- Format: weekly lecture (Thu) + tutorial (Mon), YouTube recordings, GitHub notebooks, Moodle for submissions.
- Prerequisites: not stated on the page I could reach (likely calculus/linear algebra/basic ML/PDEs given content — could not confirm exact wording).
- Topics stated on the course page: physics modeled by PDEs and limits of traditional simulators; neural PDE solvers (PINNs and variants); neural operators (FNO, CNO, Operator Transformers); graph neural networks and flexible transformer frameworks for complex geometries; generative AI (diffusion, flow) for multiscale problems and uncertainty quantification; physics foundation models.
- Notebook/tutorial list (from the GitHub repo, camlab-ethz/AI_Science_Engineering):
  1. Function Approximation with PyTorch
  2. Cross Validation and CNN Introduction
  3. PINN Training
  4. PINNs for Inverse Problems
  5. Fourier Neural Operator
  6. Convolutional Neural Operator
  7. Graph Neural Networks
  8. Vision Transformers for PDEs
  10. Automatic Differentiation Implementation
  11. JAX, Neural Differential Equations, Diffusion Models
  - New in Fall 2024: Time-Dependent CNO, Transfer Learning for PDEs
- Coverage of target topics: FNO — yes (dedicated notebook). DeepONet — NOT explicitly listed (CNO and Operator Transformers instead; DeepONet coverage unconfirmed). PINN — yes, forward and inverse. GNN — yes. Foundation models — mentioned in lecture description, not clear if a hands-on notebook exists. Theory — course is applied/notebook-driven; deep theoretical treatment (e.g., universal approximation proofs) is not the emphasis.
- Exercises: yes, tutorial notebooks are graded/practiced weekly; a course project is required.
- License/openness: fully public (lectures on YouTube, notebooks on GitHub); no explicit software license stated on the repo.
- **Assessment**: Does well — the single closest existing analogue: same three-pillar structure (PINN / neural operators / GNN) the new course likely wants, taught by the people who literally coined CNO and co-authored FNO-adjacent work, with runnable notebooks and free video. Lacks — DeepONet is not a first-class topic; foundation-model coverage is thin/uncertain; no coverage of evaluation/rollout metrics, aliasing, or industrial applications (semiconductors etc.); assumes strong applied-math background typical of an ETH graduate course, so may not suit a more industry-facing audience.

---

## 2. Ben Moseley — PINN tutorials and blog

- Blog post: https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/ (fetched). Author: Ben Moseley. Posted Aug 28, 2021, updated Nov 3, 2024.
- Code: https://github.com/benmoseley/harmonic-oscillator-pinn (linked from the post)
- Related workshop notebook: https://github.com/benmoseley/harmonic-oscillator-pinn-workshop (an "introductory crash-course" notebook, 2022) — UNVERIFIED content in detail, found via search only.
- Level: introductory, self-contained. Content: contrasts a purely data-driven network with a physics-informed one on the damped harmonic oscillator; shows how the governing ODE is added to the loss via autodifferentiation; shows improved extrapolation outside the training window.
- Format: blog post + one worked Jupyter/Python example. No exercises beyond the example notebook itself.
- License: code repo appears openly hosted on GitHub (no explicit license text confirmed); blog content has no stated restrictive license.
- **Assessment**: Does well — the best available single-hour "aha" introduction to the PINN loss-function idea, minimal and dependency-light, good as an assigned first reading. Lacks — no neural-operator content at all (single ODE example only), no discussion of failure modes (e.g., PINN training pathologies, spectral bias) beyond the one system, no assessment/quiz.

---

## 3. Brown University CRUNCH Group (Karniadakis)

- Seminar listings: https://sites.brown.edu/crunch-group/seminars/machine-learning-x-seminars/ (2019/2022/2023 editions found via search, not directly fetched in full).
- YouTube channel: https://www.youtube.com/channel/UC2ZZB80udkRvWQ4N3a8DOKQ (weekly SciML seminar recordings) — UNVERIFIED in detail (found via search, not fetched).
- DeepONet origin talk: Lu Lu, Oct 2019, "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators" (per search results).
- Karniadakis talk example: "George Karniadakis - From PINNs to DeepOnets" — YouTube (https://www.youtube.com/watch?v=QV1fVttZ6YE), UNVERIFIED content detail.
- Format: this is a **seminar series**, not a structured course — no syllabus, no assignments, no prerequisites stated, no certificate/progression. Content is individual research talks, valuable for staying current but not curriculum-shaped.
- Related structured tutorial library (third-party, built from CRUNCH-adjacent methods): https://github.com/jdtoscano94/Learning-Scientific_Machine_Learning_Residual_Based_Attention_PINNs_PIKANs_DeepONets — "Physics Informed Machine Learning Tutorials (PyTorch and JAX)" covering PINNs, PIKANs, DeepONets — found via search, not fetched in detail; flag as a candidate open tutorial repo worth checking directly before reuse.
- **DeepXDE** (the CRUNCH group's own software, doubles as its teaching material) — see item 4a below, since it's the more citable/structured resource from this group.
- **Assessment**: Does well — deepest bench of DeepONet-originating research talks and the most authoritative source for DeepONet theory/history. Lacks — no structured syllabus or assignments; a newcomer has to hunt across years of seminar recordings rather than follow a curriculum.

### 3a. DeepXDE documentation (Lu Lu / CRUNCH group software, doubles as tutorial)

- Docs: https://deepxde.readthedocs.io/en/latest/ (fetched)
- Covers: PINN (forward/inverse ODE, PDE, IDE, fractional PDE, stochastic PDE), DeepONet (multiple variants), multifidelity neural networks (MFNN).
- Structure: demos organized by problem type — function approximation, forward problems, inverse problems, operator learning — plus slides/video links and paper citations for some demos.
- Prerequisites: Python; supports TensorFlow/PyTorch/JAX/PaddlePaddle backends.
- License: not explicitly stated in the fetched page (DeepXDE itself is open-source on GitHub under an OSI license, LGPL-2.1 per its repo — unverified here, worth confirming before citing to students).
- **Assessment**: Does well — the single best structured, runnable operator-learning tutorial library with graded demo complexity, essentially a lab-manual for DeepONet + PINN. Lacks — no FNO implementation (FNO is a separate PyTorch codebase from a different group), no GNN, no foundation models, no lecture framing (it's a docs site, not a course).

---

## 4. Caltech (Anandkumar / Azizzadenesheli / Zongyi Li)

- No dedicated Caltech course on neural operators was found. CS159 "Advanced Topics in Machine Learning" (https://sites.google.com/view/cs159/lectures) exists but the most recent version findable was **Spring 2022**, and its lecture list is not neural-operator-specific from what the page shows — mark UNVERIFIED / likely not current or on-topic. Do not treat CS159 as an existing neural-operator course without further direct confirmation.
- Zongyi Li's blog post introducing FNO: https://zongyi-li.github.io/blog/2020/fourier-pde/ (found via search, page content not independently fetched in full — treat topic summary as search-derived).
  - Content (per search): introduces the Fourier Neural Operator that solves a family of PDEs "from scratch," explains resolution-invariance and the >1000x speed-up vs. traditional solvers on Navier-Stokes.
  - Original code: `zongyi-li/fourier_neural_operator` (now superseded by the maintained library below).
- **neuraloperator library docs** (Kossaifi, Kovachki, Li, Anandkumar, Pitt et al.): https://neuraloperator.github.io/dev/index.html (fetched)
  - Sections: Installation, Theory Guide, User Guide, API reference, Examples gallery, Developer's Guide.
  - Quickstart covers the full workflow: data loading, model creation (`FNO`, `TFNO`), training via a `Trainer` object, checkpointing.
  - Architectures explicitly documented on the page fetched: FNO and TFNO (Tucker-factorized FNO, ~90% parameter reduction); page notes "other architectures exist" without detailing them there (GNO and others are in the library but not covered on this particular page).
  - Example datasets: Darcy flow.
  - License: copyright held by Jean Kossaifi, David Pitt, Nikola Kovachki, Zongyi Li, Anima Anandkumar (2026) — exact license text not confirmed from the fetched page; the code repo (https://github.com/neuraloperator/neuraloperator) is the primary source and should be checked directly for the license file before reuse.
- Related theory paper often used as a textbook substitute: Kovachki, Li, Liu, Azizzadenesheli, Bhattacharya, Stuart, Anandkumar, "Neural Operator: Learning Maps Between Function Spaces," JMLR 2023 (arXiv:2108.08481), and the handbook chapter Kovachki, Lanthaler, Stuart, "Operator learning: Algorithms and analysis," Handbook of Numerical Analysis vol. 25 (2024) — found via search, not independently fetched; this is a paper/chapter, not course material, but is the standard citation for rigorous operator-learning theory.
- **Assessment**: Does well — the `neuraloperator` docs are the canonical, actively maintained hands-on reference for FNO/TFNO with runnable examples; Zongyi Li's blog is still the most-cited plain-language FNO explainer. Lacks — no structured course wraps this material (no syllabus, no exercises beyond docs examples, no assessment); DeepONet, PINN, and GNN are outside this library's scope entirely; documentation-page coverage of non-FNO architectures (GNO, etc.) was not detailed on the page fetched and should be checked separately in the API reference.

---

## 5. NVIDIA Deep Learning Institute / PhysicsNeMo (formerly Modulus)

- PhysicsNeMo product page: https://developer.nvidia.com/physicsnemo (fetched)
- DLI course, per NVIDIA's training catalog (found via search, catalog PDF not independently opened): **"Introduction to Physics-Informed Machine Learning With NVIDIA Modulus"** — self-paced, ~2 hours, $30 USD, hands-on lab via NVIDIA LaunchPad (no local GPU needed).
- Forum announcement of a PhysicsNeMo-branded successor course: https://forums.developer.nvidia.com/t/new-course-introduction-to-physics-informed-machine-learning-with-physicsnemo/229406 (title found via search, not independently fetched — likely the renamed/updated version of the Modulus course above).
- Topics (per PhysicsNeMo product page): PINNs, neural operators including FNO, GNNs, generative/diffusion models, point-cloud models.
- Teaching Kit for educators: modular lecture materials + hands-on exercises + GPU cloud access, aimed at university adoption (https://developer.nvidia.com/blog/nvidia-deep-learning-institute-launches-science-and-engineering-teaching-kit/, title/content via search).
- License: PhysicsNeMo itself is Apache-2.0, open-source, built on PyTorch.
- **Assessment**: Does well — only entry here that is explicitly industrial/production-oriented (built for engineers deploying surrogate models, not just researchers), and the sole paid/short-format option with a lab environment provided. Lacks — 2 hours is far too shallow to be a "course" in the sense of the others; it is vendor-specific to the PhysicsNeMo API rather than teaching the underlying math; no evidence of covering theory, evaluation metrics, or non-NVIDIA tooling.

---

## 6. MIT 18.337J / 6.338J — "Parallel Computing and Scientific Machine Learning" (Chris Rackauckas)

- Course site / book: https://book.sciml.ai/ and https://book.sciml.ai/course/ (fetched)
- Repos: https://github.com/mitmath/18337, https://github.com/SciML/SciMLBook
- Prerequisites: calculus, linear algebra, programming; graduate mathematical maturity; Julia used for problem sets (little/no prior Julia experience required).
- Topics: ODEs/PDEs, automatic differentiation (forward and reverse mode), neural ODEs as memory-efficient RNNs, PINNs and neural differential equations, "automatic discovery of differential equations," relationship between CNNs and PDEs, inverse problems and differentiable programming, probabilistic programming/Bayesian methods, global sensitivity analysis, uncertainty quantification, parallelization of large-scale simulation.
- Homework: 3 major assignments (parallelized ODE integrators; parameter estimation/bandwidth optimization; neural-ODE adjoints on GPU).
- Format: pre-recorded video lectures, 19 sections of lecture notes, no required textbook (points to Hairer & Wanner, Strang).
- License/openness: fully open, built with open-source tooling (Julia, Franklin.jl), code on GitHub under the SciML org.
- I did not find and could not confirm a "6.S898" cross-listing — the confirmed numbers are **18.337J/6.338J** (also appears as 6.7320J in the CSAIL listing). Treat "6.S898" as unconfirmed/possibly outdated numbering.
- **Assessment**: Does well — the strongest existing course on the *computational infrastructure* side (autodiff, differentiable programming, HPC) that any neural-operator course needs as a prerequisite layer, and it's rigorous and fully open. Lacks — essentially no dedicated neural-operator content (no FNO, DeepONet, or CNO); PINNs and neural ODEs are covered but operator learning as a distinct paradigm is not a focus; Julia-centric, which may not match a PyTorch-oriented audience.

---

## 7. Books

### 7a. Brunton & Kutz, "Data-Driven Science and Engineering" (2nd ed.)
UNVERIFIED in this pass — not independently fetched; known from general knowledge to cover SVD, DMD, sparse regression (SINDy), and some ML-for-dynamics content, but neural operators specifically are not a major focus of that book (it predates most FNO/DeepONet-era operator learning). Recommend a follow-up fetch of the book's official site/TOC before relying on this characterization.

### 7b. Karniadakis et al., "Physics-informed machine learning," Nature Reviews Physics (2021)
UNVERIFIED in this pass (not fetched directly) — this is the standard review citation for PINNs and neural operators as of 2021; useful as an assigned reading but is a journal review article, not teaching material with exercises.

### 7c. Thuerey et al., "Physics-based Deep Learning" (physicsbaseddeeplearning.org)
- Site: https://www.physicsbaseddeeplearning.org/intro.html (fetched)
- Repo: https://github.com/tum-pbs/pbdl-book (fetched) — titled "Physics-based Deep Learning Book v0.3 - the GenAI Edition"
- Authors: N. Thuerey, B. Holzschuh, P. Holl, G. Kohl, M. Lino, Q. Liu, P. Schnell, F. Trost (TUM Physics-based Simulation Group).
- Table of contents (7 major sections, from the intro page):
  1. Introduction (teaser, overview, equations, simulations)
  2. Neural Surrogates and Operators (supervised training, architectures, RANS flows)
  3. Physical Losses (PINNs, Helmholtz-Hodge decomposition)
  4. Differentiable Physics (gradient-based optimization, inverse problems)
  5. Probabilistic Learning (diffusion models, flow matching, score matching, graph-based methods)
  6. Reinforcement Learning (control problems)
  7. Improved Gradients and Fast-Forward Topics
  - README-derived list (from the GitHub repo, slightly different framing): Overview, Supervised Learning, Physical Loss Constraints, Differentiable Physics, Physical Gradients, Probabilistic Models (diffusion/flow matching), Reinforcement Learning, Bayesian Methods, Advanced Topics (GANs, Lagrangian methods, time-series, metrics), References & Notation.
- Format: Jupyter Book, all code runnable in-browser (Colab-style), free online.
- License: a LICENSE file exists in the repo but I could not confirm the exact license text from the fetch (likely CC-BY or similar for text, separate code license — **needs direct confirmation**, do not assert a specific license without checking the LICENSE file itself).
- Coverage of target topics: neural operators are covered under "Neural Surrogates and Operators" (general operator-learning framing; FNO/DeepONet by name not confirmed in the section list I could extract — needs a deeper fetch of that specific chapter page to confirm architecture-level coverage). PINN — yes (dedicated section). GNN — yes ("graph-based methods" under probabilistic learning). Foundation models — mentioned in passing per one search snippet ("scientific foundation models" as an emerging direction) but not a dedicated chapter.
- **Assessment**: Does well — the most complete free "book" spanning physical losses, differentiable simulators, and generative/probabilistic methods for physics, with everything runnable; strong on the differentiable-physics angle that most operator-learning courses skip. Lacks — FNO/DeepONet are not confirmed as named, dedicated sections (operator learning is present but the specific architectures used in industry — FNO, DeepONet, CNO — need direct confirmation by fetching the relevant chapter); no foundation-model chapter; not a "course" (no lectures/assignments/exercises, self-study only).

### 7d. Kovachki–Lanthaler–Stuart as textbook substitute
Covered in section 4 above (JMLR paper + Handbook of Numerical Analysis chapter). Rigorous but is a paper/monograph chapter, not structured teaching material — no exercises, no course wrapper.

### 7e. Trefethen, "Spectral Methods in MATLAB" (spectral-methods prerequisite)
- Author's page: https://people.maths.ox.ac.uk/trefethen/spectral.html (found via search)
- Content: 40 short MATLAB programs covering ODE/PDE boundary value problems, eigenvalues/pseudospectra, linear and nonlinear waves, numerical quadrature.
- Prerequisites stated: linear algebra, PDEs at a practical (not theoretical) level, and MATLAB familiarity.
- Audience: advanced undergrad/graduate students in numerical PDE methods, numerical analysts, computational engineers/scientists.
- **Assessment**: This is the standard prerequisite text for spectral-methods background (relevant because FNO is built on spectral/Fourier ideas) — does well as a compact, code-first primer; lacks any connection to neural networks at all (it's a pre-deep-learning numerical methods text) — a course would need to bridge it explicitly to FNO's spectral-convolution layer.

---

## 8. Secondary video explainers

- Steve Brunton (Eigensteve, University of Washington): an ~18-minute lecture on Fourier Neural Operators for physics-informed ML, covering "operators as images," Fourier-as-convolution, zero-shot super-resolution, mesh invariance, Green's-function/Laplace-operator applications. Found via Class Central listing (https://www.classcentral.com/course/youtube-fourier-neural-operator-fno-physics-informed-machine-learning-310478) and search; I could not independently fetch the actual YouTube page content (fetch attempts returned only YouTube boilerplate / 403s), so treat the exact video URL and full description as **UNVERIFIED** pending a direct successful fetch — locate via Brunton's channel https://www.youtube.com/@Eigensteve.
- Yannic Kilcher or other FNO explainers: not independently searched/fetched in this pass — flagged as a gap in this survey; a follow-up search on "Yannic Kilcher Fourier Neural Operator" is recommended if secondary explainer coverage matters to the course design.
- **Assessment**: useful as a single-sitting conceptual primer/flipped-classroom video, secondary priority as instructed; not independently verifiable at the URL level in this pass.

---

## 9. Conference tutorials and summer schools

### 9a. ICML 2024 Tutorial — "Neural Operator Learning" / "Machine Learning on Function Spaces #NeuralOperators"
- Page: https://icml.cc/virtual/2024/tutorial/35235 (fetched)
- Presenter: Kamyar Azizzadenesheli (per fetch; likely co-presented with others such as Kovachki/Anandkumar but only one presenter was confirmed on the fetched page — check the ICML page directly for the full presenter list before citing).
- Topics: theoretical foundations of neural operators, universal approximation, discretization-invariance, PDE and scientific-computing applications.
- Materials: slides available (PDF); video listed as not available on the page fetched.
- Related YouTube capture: https://www.youtube.com/watch?v=_j7bceE9AyA (found via search, title "ICML 2024 Tutorial 'Machine Learning on Function spaces #NeuralOperators'") — UNVERIFIED content, not independently fetched.

### 9b. KTH PhD Summer School on Physics-Informed Neural Networks and Applications (2025)
- Site: https://pinns.se/ (fetched)
- Dates: June 15–30, 2025 (one listing also shows a 19–30 June variant — dates across pages are inconsistent; verify exact dates before citing).
- Location: KTH Royal Institute of Technology, Stockholm.
- Instructors: KTH faculty (Jennifer Ryan, Stefano Markidis, Mathieu Barreau), SISSA's Gianluigi Rozza, Brown University's Khemraj Shukla (CRUNCH-group-adjacent — a direct link between this school and the Brown group).
- Syllabus: deep-learning fundamentals and optimization; PDEs and reduced-order models; PINNs, DeepONets, and frameworks (DeepXDE, PINA); data filtering and HPC; multiscale modeling with GPU acceleration; project areas spanning fluid/structural mechanics, medical imaging, transport, energy, bio/chemical engineering, manufacturing.
- Published output: a Springer Nature collection, "Advances in Physics-Informed Machine Learning," drawn from the school (https://link.springer.com/collections/ccdhhebjga).
- Format/materials availability: NOT confirmed whether slides/code/video are public after the fact — the fetch did not surface this; treat public-access status as UNVERIFIED.
- **Assessment**: Does well — explicitly covers both PINNs and DeepONets together with real HPC/GPU content and named frameworks (DeepXDE, PINA), and is one of the only items in this survey to explicitly combine the two major paradigms in one syllabus. Lacks — appears to be an invite/application-based PhD summer school rather than an open course; public availability of materials afterward is unconfirmed; FNO/CNO not mentioned in the topic list found.

### 9c. Oxford — "Physics Informed Neural Networks" (2025–2026)
- Page: https://www.cs.ox.ac.uk/teaching/courses/2025-2026/pinn/ (fetched)
- Instructor: David Kay. 16 lectures, Hilary Term 2026.
- Prerequisites: discrete math, linear algebra, continuous math, machine learning, programming competency; scientific computing recommended but optional.
- Topics: differential equations and physical systems, classical numerical approximation, NN optimization math, forming NNs to approximate PDE solutions with boundary/initial conditions, extensions to nonlinear/multivalued-output equations, applications to fluid flow and phase-field models.
- Format: lecture + student-chosen-language code development.
- **Assessment**: Does well — a proper university-credit course specifically on PINNs with real assessed coding work. Lacks — no neural-operator content at all per the syllabus found (PINN-only, no FNO/DeepONet/GNN); license/openness and whether recordings are public were not stated on the page.

---

## Topic → best existing resource → gap

| Topic | Best existing resource | Gap |
|---|---|---|
| FNO theory + code | `neuraloperator` docs (https://neuraloperator.github.io/dev/) + Zongyi Li's blog + ETH camlab notebook 5 | No single source combines rigorous theory (spectral convolution, universal approximation) with production-grade code and graded exercises in one place |
| DeepONet | DeepXDE docs (https://deepxde.readthedocs.io) + Brown CRUNCH seminars | Not integrated with FNO/CNO material anywhere found; no course teaches FNO and DeepONet side-by-side with a direct comparison |
| PINN fundamentals | Ben Moseley's blog + Oxford's PINN course + DeepXDE | Good coverage exists, but Oxford's course has zero neural-operator content, so PINN and operator learning remain taught in separate silos |
| Convolutional Neural Operator (CNO) | ETH camlab course/notebook 6 | Essentially ETH-exclusive; no other course or book in this survey mentions CNO by name |
| GNN for PDE simulation | ETH camlab notebook 7; PhysicsNeMo (GNN mentioned) | Thin — no dedicated GNN-for-simulation course or book chapter found with the depth of, e.g., MeshGraphNets-style content |
| Differentiable physics / hybrid solvers | Thuerey et al. `pbdl-book` | Strong, but not integrated with FNO/DeepONet material |
| Theory (universal approximation, error bounds) | Kovachki–Li et al. JMLR paper; Kovachki–Lanthaler–Stuart handbook chapter | Paper-only, no course wraps this in lectures with exercises |
| HPC / autodiff / differentiable programming infrastructure | MIT 18.337/6.338 (Rackauckas) | Julia-centric; no PyTorch-native equivalent course found with comparable rigor |
| Spectral-methods prerequisite | Trefethen, *Spectral Methods in MATLAB* | Pre-deep-learning text; needs explicit bridging to FNO's Fourier layer, which no source in this survey provides |
| Foundation models for physics/multi-physics | Mentioned only in passing (ETH course description, `pbdl-book` intro) | **No existing course or book has a dedicated foundation-model module** — confirmed gap |
| Industrial applications (e.g., semiconductors) | PhysicsNeMo (industrial framing generally) | No course ties operator learning to a specific industrial vertical like semiconductor simulation |
| Evaluation/rollout metrics, resolution invariance in practice, aliasing | Not found anywhere in this survey | Confirmed gap — see below |

## Topics no existing course covers well

Based on this survey, the following are genuine open gaps (not found as a dedicated module in any resource above):

1. **Multi-physics foundation models** — only passing mentions (ETH course description, one `pbdl-book` sentence); no dedicated teaching module found anywhere.
2. **Resolution invariance in practice** (as opposed to the theoretical claim) — no source walks through empirically verifying/breaking resolution invariance, super-resolution failure modes, or discretization mismatches between train/test.
3. **Aliasing** in spectral/Fourier operator layers — not covered as a topic in any resource surveyed, despite being a known FNO failure mode.
4. **Evaluation and rollout metrics** for autoregressive/time-stepping neural operators (drift, long-horizon stability, spectral-energy metrics) — not found as a taught topic anywhere in this survey.
5. **Industrial applications** such as semiconductor simulation — PhysicsNeMo gestures at industrial use generally but no course or book ties operator learning to a specific vertical like this.
6. **Physics-based self-improvement** (e.g., active learning / self-distillation loops using physics residuals to bootstrap training) — no resource in this survey addresses this as a topic.
7. **A unified FNO-vs-DeepONet-vs-CNO-vs-GNN comparison module** — every resource picks one or two of these; none puts them side by side with shared benchmarks/exercises.

---

## Notes on verification gaps for follow-up

- CS159 (Caltech) — could not confirm a current, neural-operator-focused edition; do not cite without further direct checking.
- `pbdl-book` license — LICENSE file exists but exact terms not confirmed; check before reusing content.
- DeepXDE's own license — likely LGPL-2.1 per general knowledge of the project, not independently confirmed here.
- Steve Brunton's FNO video — exact YouTube URL not independently confirmed (fetch attempts failed); locate via https://www.youtube.com/@Eigensteve before citing.
- KTH PINN summer school exact dates — two different date ranges appeared across pages (15–30 June vs. 19–30 June); confirm before citing.
- Yannic Kilcher FNO explainer — not searched in this pass; follow-up recommended if secondary video explainers matter.
- Brunton & Kutz book and the Karniadakis Nature Reviews Physics review — not independently fetched in this pass; characterizations are from general knowledge, not verified against the primary source.
