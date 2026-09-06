# Neural Operator Research Lineage: Beyond FNO, and Anandkumar-Group Applications

Compiled 2026-09-05. Every arXiv ID below was verified by fetching the arXiv abstract page directly. Where a fetch could not confirm a claim, it is marked UNVERIFIED. No arXiv ID or number in this document was invented.

## Timeline table

| Year | Paper | Key idea (one sentence) |
|---|---|---|
| 2020 | Graph Neural Operator (GNO), arXiv 2003.03485 | Approximate the integral kernel operator underlying a PDE solution operator with graph message passing, so one network works across discretizations. |
| 2020 | Multipole Graph Neural Operator (MGNO), arXiv 2006.09535 | Add a multi-level, multipole-inspired graph hierarchy to GNO so long-range interactions are captured in linear time instead of being truncated. |
| 2021 | Galerkin/Fourier Transformer (Cao), arXiv 2105.14995 | Drop softmax from self-attention and re-derive it as a Petrov-Galerkin projection, giving a cheaper, more accurate attention layer for operator learning. |
| 2021 | Seismic wave propagation and inversion with Neural Operators (Yang et al.), arXiv 2108.05421 | Train a neural operator once on random velocity models so it solves the wave equation and does full-waveform inversion for any new velocity structure almost instantly. |
| 2022 | FourCastNet (Pathak et al.), arXiv 2202.11214 | Chain adaptive Fourier neural operator layers to forecast global, 0.25-degree weather in under 2 seconds. |
| 2022 | U-shaped Neural Operator (U-NO), arXiv 2204.11127 | Reintroduce a U-Net-style encoder-decoder hierarchy into the neural operator so it can go deep without exploding memory use. |
| 2022 | Wavelet Neural Operator (WNO), arXiv 2205.02191 | Replace FNO's global Fourier kernel with a wavelet-domain kernel to get both spatial and frequency localization. |
| 2022 | OFormer, arXiv 2205.13671 | Build an attention-based encoder-decoder operator that makes no assumption about how input or query points are sampled. |
| 2022 | CO2 storage: Nested FNO (Wen et al.), arXiv 2210.17051 | Nest a hierarchy of FNOs at different refinement levels to get real-time, high-resolution basin-scale CO2 plume and pressure forecasts. |
| 2022 | Incremental FNO, arXiv 2211.15188 | Grow both the number of Fourier modes and the training-data resolution progressively during training to cut cost without hurting accuracy. |
| 2023 | Laplace Neural Operator (LNO), arXiv 2303.10528 | Swap the Fourier transform for a Laplace transform so the operator can handle non-periodic, transient signals with a pole-residue structure. |
| 2023 | Catheter design with neural operators (Zhou et al.), arXiv 2304.14554 | Use an FNO-based surrogate to search catheter wall geometries for one that hydrodynamically blocks upstream bacterial migration. |
| 2023 | Convolutional Neural Operator (CNO), arXiv 2302.01178 | Adapt classical CNNs into a discretization-invariant operator by carefully band-limiting all convolution and non-linearity operations. |
| 2023 | GNOT, arXiv 2302.14376 | Give a transformer a heterogeneous normalized attention layer and a geometric gating mechanism so one model handles irregular meshes and multiple input functions. |
| 2023 | Spherical FNO (SFNO, Bonev et al.), arXiv 2306.03838 | Generalize FNO's flat-geometry Fourier transform to a spherical-harmonic transform so weather forecasting on the globe stays stable over long rollouts. |
| 2023 | Plasma modelling with FNO (Gopakumar et al.), arXiv 2302.06542 | Show FNO predicts tokamak plasma evolution six orders of magnitude faster than a numerical MHD solver, on both simulated and real MAST camera data. |
| 2023 | Geometry-Informed Neural Operator (GINO), arXiv 2309.00583 | Combine a signed-distance-function/point-cloud encoding with graph and Fourier operator layers so one model handles arbitrary 3D geometries at industrial scale. |
| 2023 | Neural Operators for Accelerating Scientific Simulations and Design (survey), arXiv 2309.15325 (Nature Reviews Physics 6, 320-328, 2024) | Survey neural operators as a discretization-convergent, 4-5-orders-of-magnitude-faster alternative to classical numerical PDE solvers across science and engineering. |
| 2023 | Tensorized/Factorized FNO (MG-TFNO), arXiv 2310.00120 | Factorize FNO's weights in a high-order tensor form and add multi-grid domain decomposition, cutting memory more than 150x with less error. |
| 2023 | Plasma surrogate modelling with FNO, journal-of-record version (Gopakumar et al.), arXiv 2311.05967 (Nuclear Fusion 64, 056025, 2024) | Extended, journal version of the plasma-FNO surrogate work with a larger author/collaboration list (JOREK Team, MAST Team). |
| 2024 | Neural operators with localized integral and differential kernels (Liu-Schiaffini et al.), arXiv 2402.16845 (ICML 2024) | Add locally supported convolution-style kernels to FNO so it stops over-smoothing and captures local detail while staying resolution-independent. |
| 2024 | Transolver, arXiv 2402.02366 (ICML 2024 Spotlight) | Introduce "Physics-Attention" that adaptively groups mesh points into learnable slices by physical state, giving linear-complexity attention on arbitrary geometries. |
| 2024 | DPOT, arXiv 2403.03542 | Pretrain a Fourier-attention PDE foundation model, up to 0.5B parameters, on 10+ PDE datasets and 100k+ trajectories with an auto-regressive denoising objective. |
| 2024 | CoDA-NO, arXiv 2403.12553 | Tokenize functions along the channel/codomain axis so a single operator can be pretrained across different multiphysics PDE systems and fine-tuned with little data. |
| 2024 | FourCastNet 3, arXiv 2507.12144 | Move to a fully geometric, probabilistic ensemble forecasting formulation, beating conventional ensembles and matching diffusion models while running 8-60x faster. |
| 2025 | Transolver++, arXiv 2502.02414 | Re-engineer Transolver's parallelism and add a local adaptive mechanism so it scales to million-mesh-point industrial geometries on a single GPU. |

## Per-paper sections

### Graph Neural Operator (GNO)
- **Title:** Neural Operator: Graph Kernel Network for Partial Differential Equations
- **Authors:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2020 (v1: 7 Mar 2020)
- **arXiv ID:** 2003.03485 (verified)
- **Summary:** This is the paper that first generalizes a neural network to learn a mapping between two infinite-dimensional function spaces rather than between two finite vectors. A single set of network parameters describes an operator that can be evaluated on different finite-dimensional approximations (different meshes, different resolutions), because the core building block is an integral kernel operator, approximated here by message passing on a graph built over the domain's discretization points. It sits directly upstream of the Fourier Neural Operator: FNO is the same integral-kernel-operator idea, just with the kernel evaluated in Fourier space instead of graph space.
- **Key idea in one sentence:** Approximate the integral kernel operator underlying a PDE's solution operator using graph neural network message passing, so the same parameters generalize across discretizations.
- **Main quantitative result:** The abstract text available did not give a single headline error/speed number; its contribution is architectural (discretization-generalizing solution operators), demonstrated on PDE benchmarks including Darcy flow and Burgers-type problems.
- **Code repo:** Not listed on the abstract page; implementations of GNO live in the community `neuraloperator/neuraloperator` PyTorch library (github.com/neuraloperator/neuraloperator), which documents itself as containing the official FNO and related-architecture code including GNO.
- **Why a course would include it:** It is the direct conceptual ancestor of FNO and of the whole "operator learning via kernel integral" framing; understanding GNO is what makes clear why FNO's Fourier convolution is just one choice of kernel parameterization among several (graph, wavelet, Laplace, local-conv), all of which recur later in this lineage.

### Multipole Graph Neural Operator (MGNO)
- **Title:** Multipole Graph Neural Operator for Parametric Partial Differential Equations
- **Authors:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2020 (v1: 16 Jun 2020, v2: 19 Oct 2020)
- **arXiv ID:** 2006.09535 (verified)
- **Summary:** GNO's graphs only connect nearby points, so they miss long-range interactions unless the graph is fully connected, which is computationally infeasible at scale. This paper borrows the classical multipole method idea from numerical analysis (used for N-body simulation and fast summation): build a multi-level hierarchy of graphs so far-away interactions get bundled through coarser levels while nearby interactions stay fine-grained. The authors show this is mathematically equivalent to a multi-resolution matrix factorization of the operator's kernel matrix, unifying GNNs with classical fast-kernel-methods theory.
- **Key idea in one sentence:** Add a multi-level, multipole-style graph hierarchy on top of GNO so an operator captures interactions at all length scales with linear, not quadratic, complexity.
- **Main quantitative result:** The abstract states the multi-graph network "learns discretization-invariant solution operators to PDEs and can be evaluated in linear time"; no single headline percentage/error number is given in the abstract itself.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** It shows the fix for GNO's core scalability problem and previews the "hierarchical/multi-grid" idea that resurfaces later in Tensorized/Multi-Grid FNO (2310.00120) and Nested FNO for CO2 storage (2210.17051).

### Geometry-Informed Neural Operator (GINO)
- **Title:** Geometry-Informed Neural Operator for Large-Scale 3D PDEs
- **Authors:** Zongyi Li, Nikola Borislavov Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Prakash Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2023 (submitted 1 Sep 2023); this is the paper underlying NVIDIA's industrial CFD surrogate work
- **arXiv ID:** 2309.00583 (verified)
- **Summary:** FNO needs a regular grid; real industrial 3D geometries (car bodies, aircraft) are irregular meshes with varying shapes across a design space. GINO encodes an arbitrary input shape using a signed distance function plus a point cloud, then uses a GNO layer to move from that point cloud onto a regular latent grid where an FNO can operate efficiently, then a GNO layer again to decode back onto the original irregular mesh. This lets one trained model handle a full space of different car geometries.
- **Key idea in one sentence:** Combine a signed-distance-function/point-cloud shape encoding with graph-neural-operator "in/out" layers around a core FNO so one operator generalizes across different 3D geometries at industrial mesh scale.
- **Main quantitative result:** On a 3D vehicle aerodynamics dataset (Reynolds numbers up to five million), GINO achieves a 26,000x speed-up over optimized GPU CFD simulators for drag-coefficient computation, and a one-fourth reduction in error versus other deep-learning approaches on unseen geometry/condition combinations.
- **Code repo:** Not listed on the abstract page; the `neuraloperator/neuraloperator` GitHub repo is reported (via search, not a primary-source page fetch) to include a GINO implementation, e.g. an example script `scripts/train_gino_carcfd.py` — treat this specific file path as UNVERIFIED since it was not confirmed by direct WebFetch of that file.
- **Why a course would include it:** It is the natural synthesis of GNO (geometry flexibility) and FNO (fast global convolution), and is the direct basis for real automotive/aerospace CFD surrogate deployments discussed in the Anandkumar-group survey.

### U-shaped Neural Operator (U-NO)
- **Title:** U-NO: U-shaped Neural Operators
- **Authors:** Md Ashiqur Rahman, Zachary E. Ross, Kamyar Azizzadenesheli
- **Venue / year:** arXiv preprint, 2022 (v1: 23 Apr 2022; v3: 5 May 2023)
- **arXiv ID:** 2204.11127 (verified)
- **Summary:** Because neural operator layers behave much like fully-connected layers in terms of memory footprint, naively stacking many of them (to get a "deep" operator) blows up memory and is hard to train. U-NO borrows the U-Net idea from image segmentation: progressively downsample resolution/co-domain dimension going into the network, and progressively upsample coming back out, with skip connections. This memory-efficient hierarchy lets the operator go much deeper.
- **Key idea in one sentence:** Wrap FNO-style operator layers in a U-Net encoder-decoder hierarchy to enable much deeper neural operators without exploding memory use.
- **Main quantitative result:** On Darcy flow, U-NO gives an average 26% prediction improvement over the state of the art; on turbulent Navier-Stokes, 44% improvement; on the 3D spatiotemporal Navier-Stokes operator-learning task, 37% improvement.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** It is the standard answer to "how do you make a neural operator deep," a natural companion topic alongside Tensorized/Incremental FNO which solve the same depth/memory problem from a different angle (compression, not hierarchy).

### Spherical Fourier Neural Operator (SFNO)
- **Title:** Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
- **Authors:** Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2023 (submitted 6 Jun 2023)
- **arXiv ID:** 2306.03838 (verified)
- **Summary:** FNO's discrete Fourier transform implicitly assumes a flat, periodic domain. Earth's atmosphere lives on a sphere, so applying a flat DFT to global weather data introduces visual/spectral artifacts and unphysical dissipation, and long autoregressive rollouts (used to simulate weather many days or weeks ahead) blow up. SFNO replaces the DFT with a spherical harmonic transform, which respects the sphere's actual geometry, and shows this yields stable year-long autoregressive rollouts.
- **Key idea in one sentence:** Replace FNO's flat discrete Fourier transform with a spherical harmonic transform so operators trained on spherical data (the atmosphere) remain stable over very long autoregressive rollouts.
- **Main quantitative result:** Demonstrates stable autoregressive rollouts for a full simulated year (1,460 six-hourly steps) while retaining physically plausible dynamics.
- **Code repo:** Not listed on the abstract page; SFNO is the architecture underlying NVIDIA's FourCastNet v2/SFNO models, distributed via the `NVIDIA/makani` GitHub repo (found via search of NVIDIA NGC/GitHub listings, not independently WebFetched — treat the makani repo attribution as UNVERIFIED as a primary-source claim, though it is consistent with the FourCastNet 3 lineage below).
- **Why a course would include it:** It's the clearest case study of "the architecture must match the manifold" and is the direct bridge between FNO-family methods and FourCastNet's evolution (FourCastNet 1 uses AFNO/flat FNO; FourCastNet v2 switches to SFNO; FourCastNet 3 builds further on the geometric approach).

### Tensorized / Factorized FNO (Multi-Grid Tensorized FNO, MG-TFNO)
- **Title:** Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs
- **Authors:** Jean Kossaifi, Nikola Kovachki, Kamyar Azizzadenesheli, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2023 (submitted 29 Sep 2023)
- **arXiv ID:** 2310.00120 (verified; note this is the "Tensorized FNO" paper referenced in the task, titled MG-TFNO on arXiv)
- **Summary:** Learning solution operators at high resolution is limited by memory and by scarce training data at that resolution. This paper attacks both: (1) multi-grid domain decomposition splits the domain across devices so training parallelizes; (2) the FNO's weights are represented in a high-order latent Fourier subspace using tensor factorization, which compresses parameters drastically; (3) additional architectural tweaks to the FNO backbone improve accuracy further. Net effect: a data-efficient, far more compressed FNO that still scales to high resolution.
- **Key idea in one sentence:** Factorize FNO's weight tensors and add multi-grid domain decomposition so operators train efficiently on high-resolution PDE data with far less memory.
- **Main quantitative result:** On turbulent Navier-Stokes equations, achieves "less than half the error with over 150x compression" relative to standard FNO.
- **Code repo:** Not listed on the abstract page (implemented as part of `neuraloperator/neuraloperator`, per that repo's stated TFNO support, though the specific PR/commit was not independently WebFetched).
- **Why a course would include it:** It is the standard reference for "how do you actually deploy an FNO at scale without running out of GPU memory," directly relevant to any course section on compute cost.

### Incremental FNO (iFNO)
- **Title:** Incremental Spatial and Spectral Learning of Neural Operators for Solving Large-Scale PDEs
- **Authors:** Robert Joseph George, Jiawei Zhao, Jean Kossaifi, Zongyi Li, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2022 (v1: 28 Nov 2022; latest v4: 5 Mar 2024)
- **arXiv ID:** 2211.15188 (verified)
- **Summary:** Rather than training an FNO at full frequency-mode count and full data resolution from the start, iFNO grows both the number of Fourier modes it uses and the resolution of the training data progressively over the course of training, similar in spirit to curriculum learning or progressive-resizing in vision models. This reduces wasted compute early in training when a coarse model suffices.
- **Key idea in one sentence:** Progressively increase both the number of Fourier modes and the training-data resolution during training instead of fixing them upfront.
- **Main quantitative result:** 10% lower testing error, using 20% fewer frequency modes than a standard FNO, with 30% faster training.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** A concrete, low-complexity training-efficiency trick that is complementary to (not competing with) the tensorization/compression approach of MG-TFNO — useful for contrasting "compress the model" vs. "schedule the training" approaches to the same cost problem.

### CoDA-NO (Codomain Attention Neural Operator)
- **Title:** Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs
- **Authors:** Md Ashiqur Rahman, Robert Joseph George, Mogab Elleithy, Daniel Leibovici, Zongyi Li, Boris Bonev, Colin White, Julius Berner, Raymond A. Yeh, Jean Kossaifi, Kamyar Azizzadenesheli, Anima Anandkumar
- **Venue / year:** arXiv preprint, 2024 (v1: 19 Mar 2024)
- **arXiv ID:** 2403.12553 (verified)
- **Summary:** Multiphysics problems (e.g., fluid-structure interaction, Rayleigh-Bénard convection) couple multiple PDEs with different numbers/types of physical fields, so a single fixed-channel-count architecture doesn't transfer across systems. CoDA-NO tokenizes along the "codomain" (the channel/variable axis, e.g., pressure, velocity components, temperature) rather than only along space, then extends positional encoding, self-attention, and normalization to operate over function spaces. This lets a single model be pretrained self-supervised across multiple different PDE systems and later fine-tuned on a new one with few examples.
- **Key idea in one sentence:** Tokenize along the channel/codomain axis so self-attention over functions can pretrain across multiple different multiphysics PDE systems and transfer to new ones with limited data.
- **Main quantitative result:** Performance improvements exceeding 36% on downstream tasks (fluid flow, fluid-structure interaction, Rayleigh-Bénard convection) under limited data, versus prior approaches.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** It's the clearest example of a "foundation model for PDEs" strategy applied to the multiphysics setting, pairing naturally with DPOT (below) as two different takes on PDE pretraining.

### Transolver
- **Title:** Transolver: A Fast Transformer Solver for PDEs on General Geometries
- **Authors:** Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, Mingsheng Long
- **Venue / year:** ICML 2024 Spotlight; arXiv preprint 2024 (v1: 4 Feb 2024, v2: 1 Jun 2024)
- **arXiv ID:** 2402.02366 (verified)
- **Summary:** Standard transformer attention over raw mesh points scales quadratically and ignores the underlying physics. Transolver introduces "Physics-Attention," which adaptively partitions the discretized domain into a flexible number of learnable "slices," grouping mesh points that are in similar physical states regardless of spatial proximity, then applies attention over those slices. This captures physical correlations with linear complexity and works on arbitrary, irregular geometries.
- **Key idea in one sentence:** Adaptively group mesh points into physics-state-based "slices" and attend over the slices instead of the raw mesh, giving linear-complexity attention on arbitrary geometries.
- **Main quantitative result:** 22% relative gain averaged across six standard PDE benchmarks; also demonstrated on industrial car and airfoil design.
- **Code repo:** https://github.com/thuml/Transolver (confirmed via arXiv abstract page)
- **Why a course would include it:** Represents the "transformer, not spectral" branch of the operator-learning family tree, alongside Galerkin Transformer, OFormer, and GNOT; Transolver++ (2025) is its direct scale-up successor.

### Transolver++
- **Title:** Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries
- **Authors:** Huakun Luo, Haixu Wu, Hang Zhou, Lanxiang Xing, Yichen Di, Jianmin Wang, Mingsheng Long
- **Venue / year:** arXiv preprint, 2025 (v1: 4 Feb 2025, v2: 7 Feb 2025)
- **arXiv ID:** 2502.02414 (verified — this confirms Transolver++ exists, as the task asked to check)
- **Summary:** Prior neural PDE solvers, including the original Transolver, were limited to tens of thousands of mesh points, far short of the millions of points in real industrial CFD meshes. Transolver++ re-engineers Transolver with an "extremely optimized parallelism framework" and a "local adaptive mechanism" so a single GPU can process million-scale point clouds, and the model scales further, in linear complexity, by adding more GPUs.
- **Key idea in one sentence:** Re-engineer Transolver's parallelism and add a local adaptive mechanism to scale neural PDE solving to million-mesh-point industrial geometries.
- **Main quantitative result:** 13% relative improvement across the same six standard PDE benchmarks used for Transolver, and over 20% performance gain on million-scale industrial simulations (car and 3D aircraft designs) that are 100x larger than the previous benchmarks.
- **Code repo:** Not listed on the arXiv abstract page (note: the original Transolver's repo, github.com/thuml/Transolver, describes itself as the code release covering the Transolver lineage from the same lab).
- **Why a course would include it:** Shows the field's current frontier problem is no longer accuracy on toy PDEs but engineering neural solvers to industrial mesh scale — a useful "where is this going" closing example.

### Laplace Neural Operator (LNO)
- **Title:** LNO: Laplace Neural Operator for Solving Differential Equations
- **Authors:** Qianying Cao, Somdatta Goswami, George Em Karniadakis
- **Venue / year:** arXiv preprint, 2023 (v1: 19 Mar 2023, v2: 30 May 2023); later published in Nature Machine Intelligence (per search results, not independently WebFetched — cite as reported)
- **arXiv ID:** 2303.10528 (verified)
- **Summary:** FNO's Fourier basis assumes periodicity and struggles with transient (non-periodic, decaying/growing) responses. LNO instead decomposes the operator using the Laplace transform, which naturally represents transient dynamics via poles and residues. This gives better interpretability (poles have physical meaning, e.g. system resonances/decay rates) and, the authors report, exponential convergence.
- **Key idea in one sentence:** Replace the Fourier transform in FNO-style operators with a Laplace transform and its pole-residue structure to handle non-periodic and transient signals.
- **Main quantitative result:** A single Laplace layer in LNO outperforms four Fourier modules of FNO in approximating solutions to three ODEs (Duffing oscillator, driven gravity pendulum, Lorenz system) and three PDEs (Euler-Bernoulli beam, diffusion equation, reaction-diffusion system) — a qualitative "beats 4 layers with 1" result rather than a single percentage figure in the abstract itself.
- **Code repo:** https://github.com/qianyingcao/Laplace-Neural-Operator (found via search of the Nature Machine Intelligence companion listing; not independently WebFetched, so treat the exact URL as UNVERIFIED though highly likely correct given the consistent author match).
- **Why a course would include it:** It's the cleanest illustration that "which transform you put in the kernel" is a design choice with real consequences (periodic vs. transient signals), directly parallel to Wavelet Neural Operator and Spherical FNO doing the same substitution for different reasons (localization, spherical geometry).

### Wavelet Neural Operator (WNO)
- **Title:** Wavelet neural operator: a neural operator for parametric partial differential equations
- **Authors:** Tapas Tripura, Souvik Chakraborty
- **Venue / year:** arXiv preprint, 2022 (submitted 4 May 2022); later published in Computer Methods in Applied Mechanics and Engineering, 2023 (per search results, not independently WebFetched)
- **arXiv ID:** 2205.02191 (verified)
- **Summary:** Like LNO, WNO swaps FNO's Fourier kernel for a different transform — here, the wavelet transform — to get simultaneous time/frequency (or space/frequency) localization, which plain Fourier lacks. This lets the operator track both fine spatial patterns and their frequency content, which is useful for PDEs with localized features (shocks, fronts) that a purely global Fourier kernel smooths over.
- **Key idea in one sentence:** Use wavelet-domain integral kernels instead of Fourier-domain ones to get both spatial and frequency localization in a neural operator.
- **Main quantitative result:** Demonstrated across Burgers' equation, Darcy flow, Navier-Stokes, Allen-Cahn, and wave advection equations, plus a digital-twin application forecasting Earth's air temperature from historical data; no single headline percentage number was returned by the abstract fetch.
- **Code repo:** https://github.com/TapasTripura/Wavelet-Neural-Operator-for-pdes (found via search; not independently WebFetched, treat exact URL as reported rather than primary-source-confirmed).
- **Why a course would include it:** Completes the "swap the transform" trio (Fourier → wavelet → Laplace) and is a natural lead-in to the localized-kernel paper (2402.16845) which achieves local sensitivity a different way, via CNN-style stencils rather than a wavelet basis.

### Neural operators with localized integral and differential kernels
- **Title:** Neural Operators with Localized Integral and Differential Kernels
- **Authors:** Miguel Liu-Schiaffini, Julius Berner, Boris Bonev, Thorsten Kurth, Kamyar Azizzadenesheli, Anima Anandkumar
- **Venue / year:** ICML 2024; arXiv preprint (v1: 26 Feb 2024, v2: 8 Jun 2024)
- **arXiv ID:** 2402.16845 (verified)
- **Summary:** FNO's global Fourier convolutions can over-smooth solutions and miss local detail, while standard CNNs capture local detail but are tied to one training resolution. This paper shows how to get local sensitivity while keeping the "works at any resolution" property of an operator: (1) under an appropriate scaling of CNN kernel values, a CNN layer becomes a differential operator (inspired by classical finite-difference stencil methods); (2) local integral operators are built using discrete-continuous convolution bases. Adding these layers to an existing FNO significantly boosts performance without giving up resolution independence.
- **Key idea in one sentence:** Derive locally-supported differential and integral kernel layers (via scaled CNN stencils and discrete-continuous convolution bases) that can be added to FNO to fix its over-smoothing while preserving resolution independence.
- **Main quantitative result:** Adding these layers to FNOs reduces relative L2 error by 34-72% across experiments including turbulent 2D Navier-Stokes and the spherical shallow water equations.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** It directly addresses FNO's most commonly cited weakness (over-smoothing / loss of local detail) and is co-authored by the Anandkumar group, making it a natural "here's how the original team fixed the known flaw" case study.

### Convolutional Neural Operator (CNO)
- **Title:** Convolutional Neural Operators for robust and accurate learning of PDEs
- **Authors:** Bogdan Raonić, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, Emmanuel de Bézenac
- **Venue / year:** arXiv preprint, 2023 (v1: 2 Feb 2023; v3: 1 Dec 2023)
- **arXiv ID:** 2302.01178 (verified)
- **Summary:** Standard CNN architectures had been largely excluded from serious operator-learning work because naive convolutions on function inputs/outputs are not consistent across resolutions (aliasing, non-continuity). CNO carefully re-derives a convolutional architecture, controlling band-limiting at every convolution and non-linearity step, so the resulting network is a genuine, well-defined operator between function spaces, with a proven universality result, while still getting the local inductive bias and efficiency that made CNNs popular in vision.
- **Key idea in one sentence:** Re-engineer convolutional layers with careful band-limiting so a CNN becomes a provably discretization-invariant neural operator rather than a resolution-tied network.
- **Main quantitative result:** Reports superior performance compared to competing operator architectures across a diverse set of PDE benchmarks (the abstract fetch did not surface one single headline percentage figure).
- **Code repo:** https://github.com/bogdanraonic3/ConvolutionalNeuralOperator (confirmed via arXiv abstract page)
- **Why a course would include it:** An independent (non-Anandkumar-group) branch of the lineage that shows operator-learning theory rigorously reclaiming a classical, efficient architecture (CNNs) that had seemed disqualified by the resolution-invariance requirement.

### Galerkin / Fourier Transformer
- **Title:** Choose a Transformer: Fourier or Galerkin
- **Author:** Shuhao Cao
- **Venue / year:** NeurIPS 2021; arXiv preprint (v1: 31 May 2021, v4: 1 Nov 2021)
- **arXiv ID:** 2105.14995 (verified)
- **Summary:** This paper is the origin of using transformer-style self-attention for operator learning. It shows the softmax normalization typically used in scaled dot-product attention is sufficient but not necessary, and proposes a "Galerkin Transformer" variant that replaces softmax attention with a linear-attention scheme motivated by Petrov-Galerkin projection (a classical numerical-analysis technique for projecting a PDE onto a finite-dimensional trial/test-function space). The result is both cheaper to train and more accurate on the tested operator-learning tasks.
- **Key idea in one sentence:** Reinterpret linear (non-softmax) self-attention as a Petrov-Galerkin projection, giving an attention layer purpose-built for operator learning.
- **Main quantitative result:** Reports "significant improvements in both training cost and evaluation accuracy" over softmax-normalized counterparts on three tasks: viscid Burgers' equation, interface Darcy flow, and inverse interface coefficient identification (no single numeric headline figure surfaced in the fetched abstract).
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** It is the founding paper of the "transformer for PDEs" sub-lineage that leads to OFormer, GNOT, Transolver, and DPOT — essential for tracing why attention entered operator learning at all, and on what theoretical basis (not just "transformers work everywhere").

### OFormer
- **Title:** Transformer for Partial Differential Equations' Operator Learning
- **Authors:** Zijie Li, Kazem Meidani, Amir Barati Farimani
- **Venue / year:** arXiv preprint, 2022 (v1: 26 May 2022; v3: 27 Apr 2023)
- **arXiv ID:** 2205.13671 (verified)
- **Summary:** Introduces "OFormer" (Operator Transformer), an encoder-decoder attention architecture: the encoder uses self-attention plus point-wise MLPs to process the input function (with random Fourier features for encoding query coordinates), and the decoder uses cross-attention (rather than the masked self-attention typical of language transformers) so the model makes no assumption about how input points or query points are sampled — arbitrary, irregular, or randomly-sampled point clouds are handled natively.
- **Key idea in one sentence:** Build an encoder-decoder attention operator using self-attention and cross-attention with point-wise MLPs so it makes no assumptions about the sampling pattern of inputs or query points.
- **Main quantitative result:** Reports being competitive on standard operator-learning benchmarks and flexibly adaptable to randomly sampled inputs; no single numeric headline figure surfaced in the fetched abstract.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** A second, independent "transformer for operators" design (distinct from the Galerkin Transformer's linear-attention approach), useful for contrasting design choices before GNOT and Transolver push the idea further with geometric gating and physics-aware slicing.

### GNOT (General Neural Operator Transformer)
- **Title:** GNOT: A General Neural Operator Transformer for Operator Learning
- **Authors:** Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, Jun Zhu
- **Venue / year:** arXiv preprint, 2023 (v1: 28 Feb 2023; v3: 14 Jun 2023)
- **arXiv ID:** 2302.14376 (verified)
- **Summary:** Real-world PDE problems often have irregular meshes and multiple, heterogeneous input functions (e.g., boundary conditions plus material fields plus source terms), which earlier transformer operators didn't cleanly handle together. GNOT introduces a "heterogeneous normalized attention layer" for flexibly ingesting multiple input functions on irregular meshes, plus a "geometric gating mechanism" that acts like a soft, learned domain decomposition to handle problems with multiple physical scales.
- **Key idea in one sentence:** Add heterogeneous normalized attention and a geometric gating mechanism (a soft domain decomposition) so one transformer handles multiple input functions and multiscale physics on irregular meshes.
- **Main quantitative result:** Not surfaced as a single headline number in the fetched abstract; the paper is architectural/general-purpose rather than reporting one flagship benchmark figure in the abstract text.
- **Code repo:** https://github.com/thu-ml/GNOT (confirmed via arXiv abstract page)
- **Why a course would include it:** Bridges OFormer-style flexible attention with the domain-decomposition ideas that later resurface (in a physics-aware form) in Transolver's slice-based Physics-Attention.

### DPOT (Denoising Pre-training Operator Transformer)
- **Title:** DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training
- **Authors:** Zhongkai Hao, Chang Su, Songming Liu, Julius Berner, Chengyang Ying, Hang Su, Anima Anandkumar, Jian Song, Jun Zhu
- **Venue / year:** arXiv preprint, 2024 (v1: 6 Mar 2024; v4: 7 May 2024)
- **arXiv ID:** 2403.03542 (verified)
- **Summary:** PDE datasets are diverse and often scarce relative to what deep pretraining needs — trajectories differ in length, resolution, dimensionality, and PDE type, unlike the relatively uniform corpora used to pretrain large language models. DPOT proposes an auto-regressive denoising pretraining objective for stability, plus a scalable Fourier-attention architecture, and trains a "PDE foundation model" up to 0.5B parameters on more than ten different PDE datasets and over 100,000 trajectories.
- **Key idea in one sentence:** Pretrain a large Fourier-attention transformer across 10+ diverse PDE datasets using an auto-regressive denoising objective to build a general-purpose, fine-tunable PDE foundation model.
- **Main quantitative result:** Reports state-of-the-art results on these benchmarks and strong generalization to diverse downstream PDE tasks including 3D data, at up to 0.5B parameters trained on 100k+ trajectories (specific downstream accuracy percentages were not surfaced in the fetched abstract text).
- **Code repo:** https://github.com/thu-ml/DPOT (confirmed via arXiv abstract page)
- **Why a course would include it:** The clearest "foundation model for PDEs" case study alongside CoDA-NO, and it includes Anandkumar as a co-author, connecting the transformer-operator lineage back to the Anandkumar-group thread.

## Survey: Neural Operators for Accelerating Scientific Simulations and Design
- **Title:** Neural Operators for Accelerating Scientific Simulations and Design
- **Authors:** Kamyar Azizzadenesheli, Nikola Kovachki, Zongyi Li, Miguel Liu-Schiaffini, Jean Kossaifi, Anima Anandkumar
- **Venue / year:** Nature Reviews Physics, volume 6, pages 320-328, May 2024 (DOI 10.1038/s42254-024-00712-5, confirmed via search); arXiv preprint version submitted 27 Sep 2023, final v5 4 Jan 2024
- **arXiv ID:** 2309.15325 (verified)
- **Summary:** This is the group's own retrospective survey of the neural operator paradigm. Its central framing: neural operators learn mappings between infinite-dimensional function spaces (not fixed-size vector spaces), which is why a single trained model can be evaluated at resolutions it never saw during training — a property the survey calls discretization convergence. It positions neural operators as a data-driven surrogate that can augment or fully replace classical numerical simulators (CFD, weather, materials) while running roughly 4-5 orders of magnitude faster, and it discusses combining neural operators with physics constraints at inference time to sharpen fidelity and generalization.
- **What it says about discretization convergence:** The survey's core claim is that because the operator is learned as a mapping between function spaces rather than between fixed grids, it converges to a well-defined, discretization-independent limit as resolution increases — meaning the same trained weights can be evaluated at a coarser or finer resolution than they were trained on (zero-shot super-resolution), rather than being brittle to input-size mismatch the way a fixed-input-size neural network is.
- **What it says about the cost of FNO at scale:** The survey frames the 4-5-order-of-magnitude speed-up as the central practical payoff versus classical solvers, but the field's follow-on work it surveys (MG-TFNO's memory/compression fixes, iFNO's progressive training, the localized-kernel paper's fix for over-smoothing) shows that vanilla FNO at high resolution and large scale still runs into real memory and accuracy costs that motivated most of the architectural literature catalogued above.
- **Open problems the survey highlights:** generalizing operator learning further across geometries and multiphysics settings (the GINO/CoDA-NO direction), improving theoretical understanding and error guarantees, integrating physics constraints more tightly without sacrificing the speed advantage, and scaling training data and models toward foundation-model-style pretraining (the DPOT/CoDA-NO direction) — read directly off the survey's own abstract and its position as the framing document for essentially every 2023-2024 paper in this lineage; a full point-by-point open-problems list would require reading the full paper body, which was not fetched beyond the abstract page.
- **Code repo:** None listed on the abstract page.
- **Why a course would include it:** It's the field's own state-of-the-art summary written by the core inventors (Azizzadenesheli, Kovachki, Li, Liu-Schiaffini, Kossaifi, Anandkumar), making it the natural "assign this to tie everything together" reading.

## Anandkumar-group applications

### FourCastNet
- **Title:** FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators
- **Authors:** Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, Animashree Anandkumar
- **Venue / year:** arXiv preprint, 2022 (submitted 22 Feb 2022)
- **arXiv ID:** 2202.11214 (verified)
- **Summary:** FourCastNet chains Adaptive Fourier Neural Operator layers to build a global data-driven weather model that predicts short-to-medium-range forecasts at 0.25-degree resolution, matching the resolution of the leading physics-based forecast system (IFS) but running vastly faster because it's a learned surrogate rather than a numerical integration of the atmosphere's governing equations.
- **Key idea in one sentence:** Chain Adaptive Fourier Neural Operator layers to build a global, 0.25-degree-resolution, data-driven weather forecasting model.
- **Main quantitative result:** Generates a week-long forecast in under 2 seconds, orders of magnitude faster than the IFS numerical weather model, while accurately forecasting fast-timescale variables like surface wind speed, precipitation, and atmospheric water vapor.
- **Code repo:** https://github.com/NVlabs/FourCastNet (identified via search as the official NVIDIA repo, described as "Initial public release of code, data, and model weights for FourCastNet"; not independently WebFetched, so treat as reported rather than primary-source-confirmed).
- **Successor architecture:** FourCastNet v2/SFNO swaps in the Spherical FNO architecture (arXiv 2306.03838, above) to fix flat-DFT artifacts on the sphere; FourCastNet 3 (arXiv 2507.12144, below) is the next full successor paper.
- **Why a course would include it:** The flagship real-world deployment of FNO-family methods, and the video the team is presumably referencing — a natural anchor for the "applications" half of the course.

### FourCastNet 3
- **Title:** FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale
- **Authors:** Boris Bonev, Thorsten Kurth, Ankur Mahesh, Mauro Bisson, Jean Kossaifi, Karthik Kashinath, Anima Anandkumar, William D. Collins, Michael S. Pritchard, Alexander Keller
- **Venue / year:** arXiv preprint, 2025 (v1: 16 Jul 2025, v2: 18 Jul 2025)
- **arXiv ID:** 2507.12144 (verified)
- **Summary:** The direct successor to FourCastNet v2/SFNO. It reformulates weather forecasting as a fully geometric, probabilistic ensemble problem (respecting spherical geometry and modeling spatially correlated uncertainty), rather than a single deterministic forecast.
- **Key idea in one sentence:** Reformulate global weather forecasting as a geometric, probabilistic ensemble problem, extending the spherical-FNO approach to produce calibrated uncertainty at scale.
- **Main quantitative result:** Surpasses leading conventional ensemble forecast models and rivals the best diffusion-based methods, while running 8-60x faster; produces a 60-day global forecast at 0.25-degree, 6-hourly resolution in under 4 minutes on a single GPU, with realistic spectra retained out to 60-day lead times.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** Shows the field's current state of the art in operational-scale probabilistic weather forecasting, and closes the loop from FNO (2020) through SFNO (2023) to a production-grade ensemble system (2025).

### Catheter design with neural operators
- **Title:** AI-aided Geometric Design of Anti-infection Catheters
- **Authors:** Tingtao Zhou, Xuan Wan, Daniel Zhengyu Huang, Zongyi Li, Zhiwei Peng, Anima Anandkumar, John F. Brady, Paul W. Sternberg, Chiara Daraio
- **Venue / year:** arXiv preprint, physics.med-ph category, submitted 27 Apr 2023. **Verification note:** the task description named this as a Science Advances 2023 paper; the arXiv page itself does not list a journal venue, and search results (Caltech/Merkin Institute news coverage) describe the same Zhou et al. team and result without confirming the exact journal name in a source I fetched directly — treat "Science Advances" as UNVERIFIED pending a direct fetch of the journal page.
- **arXiv ID:** 2304.14554 (verified as matching this title/author list)
- **Summary:** Catheter-associated urinary tract infections, caused partly by bacteria migrating upstream along the catheter tube wall, cost roughly $300 million per year in the US (per Caltech's own news coverage of this project). The team used a Fourier-neural-operator surrogate to rapidly evaluate how different catheter wall geometries (fin-like triangular protrusions lining the interior) affect the flow-induced hydrodynamics that drive bacterial migration, replacing what would otherwise be days of computational fluid dynamics with a design search that runs in minutes. The optimized geometry was then validated with E. coli experiments in microfluidic devices and 3D-printed catheter prototypes.
- **Key idea in one sentence:** Use an FNO-based surrogate to rapidly search catheter interior wall geometries for a shape that hydrodynamically blocks upstream bacterial migration.
- **Main quantitative result:** The optimized geometry achieves "1-2 orders of magnitude improved suppression of bacterial contamination" at the catheter's upstream end compared to a standard smooth-walled catheter.
- **Code repo:** None listed on the abstract page.
- **Why a course would include it:** A concrete, human-health-relevant example of neural-operator-accelerated design optimization outside of physics/climate — good for showing the method's generality beyond PDEs traditionally associated with scientific computing.

### Nuclear fusion plasma modelling / control
- **Two related papers, both from the same Gopakumar-led collaboration with Anandkumar:**

**(a) Fourier Neural Operator for Plasma Modelling**
- **Authors:** Vignesh Gopakumar, Stanislas Pamela, Lorenzo Zanisi, Zongyi Li, Anima Anandkumar, MAST Team
- **Venue / year:** arXiv preprint, 2023 (submitted 13 Feb 2023)
- **arXiv ID:** 2302.06542 (verified)
- **Summary:** Predicting how plasma evolves inside a tokamak is central to designing a working fusion reactor, but numerical MHD (magnetohydrodynamic) solvers are extremely expensive. This paper trains an FNO surrogate on both simulated MHD data and real experimental camera data from the MAST spherical tokamak, and shows it can forecast plasma filament formation and heat deposition ahead of time.
- **Key idea in one sentence:** Train an FNO surrogate on both simulated and real MAST tokamak data to forecast plasma evolution far faster than a numerical MHD solver.
- **Main quantitative result:** Six orders of magnitude faster than the traditional numerical solver, maintaining accuracy around 10⁻⁵ normalized mean squared error; outperforms Conv-LSTM and U-Net baselines with fewer parameters and faster training; forecasts plasma filament formation and heat deposits at half the shot duration time.
- **Code repo:** Not listed on the abstract page.

**(b) Plasma Surrogate Modelling using Fourier Neural Operators (extended/journal version)**
- **Authors:** Vignesh Gopakumar, Stanislas Pamela, Lorenzo Zanisi, Zongyi Li, Ander Gray, Daniel Brennand, Nitesh Bhatia, Gregory Stathopoulos, Matt Kusner, Marc Peter Deisenroth, Anima Anandkumar, JOREK Team, MAST Team
- **Venue / year:** Nuclear Fusion, volume 64, article 056025 (2024); arXiv preprint v1: 10 Nov 2023, v2: 18 Jun 2024
- **arXiv ID:** 2311.05967 (verified)
- **Summary:** An extended, larger-collaboration version of the same FNO-for-plasma-surrogate result, now formally published in the journal Nuclear Fusion, expanding the team (adding the JOREK simulation-code team as collaborators) and demonstrating the surrogate in both simulation and experimental domains.
- **Key idea in one sentence:** Journal-published extension demonstrating FNO-based plasma-evolution surrogates across both JOREK simulation data and MAST experimental data.
- **Main quantitative result:** Same headline six-orders-of-magnitude speed-up over traditional solvers reported in the earlier preprint, now in peer-reviewed form.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include either/both:** Direct evidence for the survey's claim of "4-5+ orders of magnitude" speed-ups, in a domain (fusion) where simulation cost is a major bottleneck to reactor design iteration; useful case study for control-relevant, real-time surrogate modeling. Note: the task description also asked about "nuclear fusion plasma control" specifically — these two papers are about plasma *evolution forecasting/surrogate modelling*, which is a prerequisite for real-time control but is not itself a closed-loop control paper; no separate Anandkumar-group closed-loop tokamak control paper was found in this search, so that stronger "control" framing should be treated as UNVERIFIED / not found.

### CO2 storage: Nested FNO
- **Title:** Real-time high-resolution CO2 geological storage prediction using nested Fourier neural operators
- **Authors:** Gege Wen, Zongyi Li, Qirui Long, Kamyar Azizzadenesheli, Anima Anandkumar, Sally M. Benson
- **Venue / year:** Energy & Environmental Science, volume 16, issue 4, pages 1732-1741 (2023); arXiv preprint v1: 31 Oct 2022, v2: 1 Jun 2023
- **arXiv ID:** 2210.17051 (verified)
- **Summary:** Scaling up carbon capture and storage requires modeling how injected CO2 pressure and plume migration will evolve across a whole geological storage basin, at high spatial resolution, in something close to real time — a task existing numerical reservoir simulators can't do fast enough for the scale needed. The paper nests a hierarchy of FNOs at different levels of spatial refinement (echoing the earlier Multipole GNO hierarchy idea) to produce fast, high-resolution, basin-scale 3D/4D forecasts.
- **Key idea in one sentence:** Nest a hierarchy of FNOs at different spatial-refinement levels to forecast basin-scale CO2 storage pressure and plume migration in real time and at high resolution.
- **Main quantitative result:** Nearly 700,000x speed-up in flow prediction compared to existing numerical methods.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** One of the largest reported speed-up numbers in the entire lineage (nearly six orders of magnitude), and a directly climate-relevant application (carbon capture and storage scale-up) that a course on ML/AI applications would want to highlight.

### Seismic / inverse problems with neural operators
- **Title:** Seismic wave propagation and inversion with Neural Operators
- **Authors:** Yan Yang, Angela F. Gao, Jorge C. Castellanos, Zachary E. Ross, Kamyar Azizzadenesheli, Robert W. Clayton
- **Venue / year:** The Seismic Record, volume 1, issue 3, pages 126-134 (2021); arXiv preprint v1: 11 Aug 2021, v2: 13 Oct 2021
- **arXiv ID:** 2108.05421 (verified)
- **Summary:** This is an earlier (2021), less-known Azizzadenesheli-coauthored application: train a neural operator on an ensemble of wave-equation simulations across randomized velocity models and source locations, so the trained operator can then compute a wavefield solution for any new velocity structure or source location almost instantly, without re-solving the PDE from scratch. Because the operator is grid-free, it can also be evaluated at a higher resolution than it was trained on. The authors demonstrate seismic tomography (recovering subsurface velocity structure) using automatic differentiation through the trained operator to get gradients of the wavefield with respect to velocity structure, for full waveform inversion.
- **Key idea in one sentence:** Train a neural operator once across random velocity models and source locations so it can solve the wave equation, and perform gradient-based full-waveform inversion, for any new velocity structure almost instantly.
- **Main quantitative result:** Nearly an order of magnitude faster than conventional numerical methods for full waveform inversion.
- **Code repo:** Not listed on the abstract page.
- **Why a course would include it:** Demonstrates neural operators used for an *inverse* problem (recovering unknown parameters via automatic differentiation through the operator) rather than only a forward-simulation surrogate, which is a distinct and important use pattern worth contrasting with the forward-surrogate examples above (weather, plasma, CO2, catheter).

### Semiconductor / chip physics
- **Search finding: no Anandkumar-group paper found.** A search specifically for neural-operator work on semiconductor or chip physics from the Anandkumar group returned nothing matching. What did turn up is a cluster of *unrelated-group* work applying physics-informed neural networks and neural operators to EUV (extreme ultraviolet) lithography mask simulation — e.g. "Physics-Informed Neural Operator for Warm-Starting Background-Decomposed and Preconditioned PSFD: Enabling Scalable 3-D EUV Mask Simulation" and "Physics-informed neural networks and neural operators for a study of EUV electromagnetic wave diffraction from a lithography mask" — but these were not traced to Anandkumar, Azizzadenesheli, Kovachki, Li, Kossaifi, or Liu-Schiaffini as authors, and their arXiv IDs were not independently verified since this branch returned no relevant Anandkumar-group hit. **Reporting honestly per the task instructions: no evidence found of Anandkumar-group neural-operator work on semiconductor or chip physics.**

## Notes on verification gaps
- A few GitHub repo URLs (LNO, WNO, GINO's specific example script, FourCastNet, SFNO/makani, and TFNO's home in the neuraloperator library) were found via WebSearch rather than confirmed by an independent WebFetch of the repo page itself; these are flagged inline above and should be treated as "reported, plausible, not independently primary-source-verified" rather than fully verified in the same sense as the arXiv IDs (all of which were WebFetched directly).
- The Nature Reviews Physics survey's open-problems discussion above is inferred from its abstract and from what its own citation graph (the papers surveyed here) reveals about active problems; a full open-problems enumeration would require fetching the paper body beyond the abstract page, which this research pass did not do.
- "Nuclear fusion plasma control" (as opposed to plasma evolution *forecasting*) was not confirmed as a separate Anandkumar-group paper; flagged as UNVERIFIED above.
- The claim that the catheter paper appeared in *Science Advances* specifically was not confirmed by a direct fetch of a journal page; flagged as UNVERIFIED above.
