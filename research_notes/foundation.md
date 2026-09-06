# Multi-physics / multi-PDE foundation models: what the public record actually shows

Compiled 2026-09-05. Claim under test (Anandkumar/Jenik, Accelerated Understanding, podcast): one model trained
on multiple areas of physics beats equal-size single-domain models, and neural operators give resolution
flexibility. Every entry below was checked against the arXiv abstract page and, where accessible, the full
HTML/PDF text. Items marked UNVERIFIED could not be confirmed from primary text.

---

## 1. Multiple Physics Pretraining (MPP) — McCabe et al.
- **Authors:** Michael McCabe, Bruno Régaldo-Saint Blancard, Liam Holden Parker, Ruben Ohana, Miles Cranmer, Alberto Bietti, Michael Eickenberg, Siavash Golkar, Geraud Krawezik, Francois Lanusse, Mariel Pettee, Tiberiu Tesileanu, Kyunghyun Cho, Shirley Ho (Polymathic AI).
- **Venue/arXiv:** arXiv:2310.02994 (submitted Oct 2023, revised Dec 2024). https://arxiv.org/abs/2310.02994, full text https://arxiv.org/html/2310.02994
- **Summary:** One autoregressive transformer backbone (AViT) trained across several 2D time-dependent PDE systems at once, predicting all of them from a shared embedding space, instead of one model per system.
- **Pretraining corpus:** PDEBench 2D time-dependent simulations — Compressible Navier-Stokes (CNS, 1,000 trajectories at 512×512, plus 10,000 at 128×128), Incompressible Navier-Stokes (INS, 1,000 traj. at 512×512), Shallow Water Equations (SWE, 1,000 traj. at 128×128), Diffusion-Reaction 2D (1,000 traj. at 128×128), plus PDEArena incompressible NS (5,200 traj. at 128×128).
- **Architecture:** Transformer (Axial ViT) with shared embedding/normalization (Reversible Instance Norm) projecting heterogeneous fields into one space. Limited to uniform grids; 3D extension done via "kernel inflation" (repeating a 2D kernel along a new axis).
- **Numbers found:**
  - Table 1 comparison (no fine-tuning), MPP-AViT-B vs task-specific AViT-B baseline (lower is better, error metric):
    - SWE: MPP 0.00240 vs task-specific baseline 0.00047 (MPP *worse*)
    - DiffRe2D: MPP 0.0106 vs baseline 0.0110 (MPP slightly better)
    - CNS Mach 1.0: MPP 0.0281 vs baseline 0.0316 (MPP better)
    - CNS Mach 0.1: MPP 0.0172 vs baseline 0.0261 (MPP better)
  - At the smallest model scale, "MPP-AViT-Ti outperforms the PDEBench baselines on all problems except for SWE" — SWE task-specific baseline NRMSE 0.0044 vs MPP-AViT-Ti's 0.0066 (worse).
  - Authors explicitly state: **"the multi-task training does hurt performance on individual tasks"** relative to single-task baselines before fine-tuning; fine-tuning is needed to recover/exceed single-task performance.
- **Negative result admitted by authors:** Shallow Water Equations is a case where the shared multi-physics model underperforms a dedicated single-task model, at every model scale tested. Multi-task pretraining without fine-tuning generally costs some per-task accuracy.
- **Resolution handling:** Fixed uniform grids per task; no continuous-resolution operator claim (this is a vision-transformer-style patch model, not a true resolution-free neural operator).

---

## 2. Poseidon — Herde et al. (ETH, Mishra group)
- **Authors:** Maximilian Herde, Bogdan Raonić, Tobias Rohner, Roger Käppeli, Roberto Molinaro, Emmanuel de Bézenac, Siddhartha Mishra.
- **Venue/arXiv:** arXiv:2405.19101 (May 2024, rev. Nov 2024). https://arxiv.org/abs/2405.19101, full text https://arxiv.org/html/2405.19101
- **Summary:** A multiscale operator transformer with time-conditioned layer norms, pretrained on a small set of fluid-dynamics PDEs, then evaluated by fine-tuning on 15 different downstream PDE tasks (9 of which involve physics types absent from pretraining).
- **Pretraining corpus:** 6 operators total — 4 compressible Euler variants (CE-RP, CE-KH, CE-CRP, CE-Gauss) and 2 incompressible Navier-Stokes variants (NS-Sines, NS-Gauss). 9,640 trajectories for the Euler operators and 19,640 for Navier-Stokes families = **77,840 trajectories total**, 11 time snapshots each, yielding ~5.11M training examples via an "all2all" pairing strategy.
- **Architecture:** Multiscale (U-Net-like) operator transformer with time-conditioned layer norm enabling continuous-in-time evaluation. Evaluated on Cartesian grids; some generalization to non-Cartesian geometry shown via masking on one task (SE-AF), not a general arbitrary-resolution claim.
- **EXACT data-efficiency numbers (this is the key claim in the task):**
  - Headline: **"only 20 samples are needed for Poseidon-L to reach the errors of FNO with 1024 samples"** on the median downstream task — i.e. roughly a **~50x** reduction in labeled fine-tuning data needed, on the median task.
  - Efficiency Gain (EG, samples needed by FNO to match Poseidon's error) reported per task in their Table 1, e.g.: NS-PwC EG = 890.6×; NS-BB EG = 552.5×; NS-SVS EG = 502.9×; Wave-Gauss EG = 62.1×; Helmholtz EG = 78.3×.
  - Accuracy Gain at a fixed 128 fine-tuning samples: "mean gain of accuracy [is] an entire order of magnitude," with per-task gains "ranging from ...10% to a factor of 25."
  - **On 13 of 15 downstream tasks, Poseidon-L needs an order of magnitude fewer samples than FNO** to match error.
- **Negative result admitted:** On 1 of 15 tasks (CE-RM, a time-independent PDE) a different baseline (CNO) beats Poseidon, "only marginally." Authors explicitly flag the pretraining set as "a very small set of PDEs" and that elliptic-PDE performance is comparatively weaker than time-dependent PDE performance.
- **Assessment:** This is the strongest, most precisely quantified public evidence for the podcast's data-efficiency/transfer claim, but it is a claim about fine-tuning sample efficiency after pretraining on a handful of fluid PDEs — not literally "beats an equal-size single-domain model trained on the same amount of data," and not evaluated against non-fluid physics (e.g. no solid mechanics, no EM, no chemistry).

---

## 3. DPOT — Hao, Su, Liu, Berner, Ying, Su, Anandkumar, Song, Zhu
- **Authors:** Zhongkai Hao, Chang Su, Songming Liu, Julius Berner, Chengyang Ying, Hang Su, **Anima Anandkumar**, Jian Song, Jun Zhu.
- **Venue/arXiv:** arXiv:2403.03542 (Mar 2024, final version May 2024). https://arxiv.org/abs/2403.03542, full text https://arxiv.org/html/2403.03542
- **Note of relevance:** Anandkumar is a co-author on this specific multi-physics-pretraining paper, making it the closest primary-literature analog to her podcast claim.
- **Pretraining corpus:** 12 datasets from 4 sources — FNO's Navier-Stokes (3 viscosities), PDEBench (Compressible NS, Shallow-Water, Diffusion-Reaction), PDEArena (2 NS variants), CFDBench (irregular-geometry fluid flow). **"More than 100k trajectories from more than ten datasets."** Model scaled up to 0.5B parameters ("DPOT-L" / similar).
- **Architecture:** Transformer with Fourier attention, denoising auto-regressive pretraining objective; resolution handled by upscaling/downscaling all data to a common grid (H=128) via interpolation for pretraining, then swapping FFT/IFFT ops for fine-tuning on different dimensionalities. Authors report "relatively stable performance" testing resolutions from 32 to 128.
- **Numbers found:**
  - High-res 2D turbulence: error reduced from ~16.7% (no pretraining) to 13.5% (pretrained) — **~19% relative improvement**.
  - 3D Navier-Stokes transfer: error dropped from 41% to 22.6% — **~45% relative reduction**.
  - General claim: pretrained DPOT-L "reduc[es] error by up to 52%" over baselines on their benchmark suite.
  - Long-rollout Kolmogorov turbulence: full-trajectory error fell from 82.2% (no pretraining) to 33.5% (pretrained) — a large but still substantial (33.5%) residual error, i.e. pretraining helps a lot but doesn't solve long-rollout drift.
- **Negative result admitted:** On CFDBench (zero-shot), a competing multi-physics model (MPP-L, 0.00650 L2 relative error) beat DPOT-L (0.00749) — DPOT authors concede "there is room left for improvement" on this dataset. So one multi-physics-pretrained model is not uniformly better than another multi-physics-pretrained model.

---

## 4. Universal Physics Transformers (UPT) — Alkin et al.
- **Authors:** Benedikt Alkin, Andreas Fürst, Simon Schmid, Lukas Gruber, Markus Holzleitner, Johannes Brandstetter.
- **Venue/arXiv:** NeurIPS 2024; arXiv:2402.12365. https://arxiv.org/abs/2402.12365, full text https://arxiv.org/html/2402.12365
- **Summary:** A single architecture meant to unify Eulerian (grid) and Lagrangian (particle) simulation types without a fixed grid- or particle-based latent structure, encoding to a latent space and decoding to arbitrary query points in space-time.
- **Training data:** Steady-state ShapeNet-Car (889 car shapes, 3.6K mesh points each, surface pressure regression); 10K self-generated transient Navier-Stokes pipe-flow simulations via OpenFOAM (29K–59K mesh points each); Lagrangian Taylor-Green vortex (3D) SPH simulations from LagrangeBench. This is multi-*simulation-type* diversity, not multi-*PDE-family* diversity in the MPP/DPOT/Poseidon sense.
- **Resolution generalization (this is the paper most directly relevant to the "resolution flexibility" half of the claim):** Trained with 2048 latent "supernodes" and 16K query positions, then tested at different input/output point counts without retraining. Authors report: **"UPT generalizes across a wide range of different number of input or output positions, with even slight performance increases when using more input points."**
- **Speed numbers:** Latent-space rollout gives ~**400×** speedup vs the pisoFoam CFD solver (0.3s GPU vs 120s on 16 CPUs); autoregressive-via-physics-domain rollout gives ~**60×** speedup (2.0s GPU vs 120s). Vs. the GINO neural operator baseline: GINO needs 900s/epoch, UPT needs 4s/epoch for a model only 0.17 (unspecified metric units) worse.
- **Limitations admitted:** Latent rollout has non-negligible training overhead; large-scale Lagrangian datasets don't yet exist publicly, limiting how far this axis can be pushed; rollout error accumulates over time (though authors argue the physics stays qualitatively faithful even as absolute trajectory error grows); larger models were infeasible under their compute budget.

---

## 5. UPS (Unified PDE Solvers) — Shen, Marwah, Talwalkar
- **Authors:** Junhong Shen, Tanya Marwah, Ameet Talwalkar.
- **Venue/arXiv:** TMLR 2024; ICML 2024 AI4Science Workshop (Spotlight); arXiv:2403.07187. https://arxiv.org/abs/2403.07187
- **Summary:** Embeds diverse PDEs into a shared representation and processes them with an FNO+transformer architecture that is *warm-started from a pretrained LLM backbone* (cross-modal adaptation, text→PDE), rather than trained from scratch on physics data alone.
- **Numbers found:** UPS reports **outperforming prior unified/multi-physics models using 4× less data and 26× less compute**, and demonstrates few-shot transfer to unseen PDE families and unseen coefficients within families.
- **Caveat:** These are comparisons against other unified PDE models (efficiency of building the foundation model itself), not a same-size single-domain-vs-multi-domain accuracy comparison; exact task-level error numbers were not extracted from the abstract-level fetch (would require the full PDF for per-task tables).

---

## 6. ICON — In-Context Operator Networks (Yang, Liu, Meng, Osher)
- **Authors:** Liu Yang, Siting Liu, Tingwei Meng, Stanley J. Osher.
- **Venue/arXiv:** PNAS 2023 (published); arXiv:2304.07993. https://arxiv.org/abs/2304.07993, full text https://ar5iv.labs.arxiv.org/html/2304.07993
- **Summary:** Learns to infer an operator from a handful of input/output "demo" examples given in-context at inference time (like an LLM prompt), rather than fitting weights per-PDE. No gradient update needed to handle a new but related operator.
- **Training/test set:** 19 problem types spanning forward/inverse 1D ODEs (3 families), forward/inverse damped oscillator, forward/inverse Poisson, forward/inverse linear and nonlinear reaction-diffusion, and forward/inverse mean-field control (1D and 2D, 2 parameter variants).
- **Numbers found:** In-distribution average relative error stays **below 6% with a single demo**, dropping to **around 2% with 5 demonstrations**. Authors report successful generalization to operator parameters extending beyond the training region ("strong generalization ability" claim), but **no quantitative baseline comparison against a fine-tuned or from-scratch model is provided in the paper** — this is a real evidentiary gap.
- **Limitations admitted by authors:** "The scale of our experiments is rather small." When trained only on one ODE family, the network "can hardly generalize...beyond" that family — in-context generalization does not extend to genuinely novel equation forms, only to new parameters/instances within families the diverse training mixture already covered.

---

## 7. PROSE family (Liu, Zhang, Schaeffer et al.)
- **Original PROSE:** "PROSE: Predicting Operators and Symbolic Expressions using Multimodal Transformers," Yuxuan Liu, Zecheng Zhang, Hayden Schaeffer, arXiv:2309.16816 (2023). Multimodal (numeric + symbolic) transformer that outputs both a predicted trajectory and a candidate symbolic form of the governing equation, trained on multiple 1D time-dependent nonlinear-constant-coefficient PDE families simultaneously.
- **PROSE-PDE:** Jingmin Sun, Yuxuan Liu, Zecheng Zhang, Hayden Schaeffer, arXiv:2404.12355, "Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation." Extends the multi-operator approach and reports extrapolation to PDE models/data unseen during training.
- **PROSE-FD:** "PROSE-FD: A Multimodal PDE Foundation Model for Learning Multiple Operators for Forecasting Fluid Dynamics," arXiv:2409.09811. Per independent secondary source (Nov 2025 arXiv 2511.21861's related-work discussion): PROSE-FD is "pretrained on six families of parametrized fluid equations and demonstrates encouraging zero-shot transfer on heterogeneous two-dimensional flow problems." **The primary PDF for PROSE-FD could not be parsed in this session (binary/encoding failure on fetch) — its exact numeric transfer results are UNVERIFIED here** and should be pulled from the paper directly before citing specific figures.

---

## 8. Subramanian et al. — "Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior" (NeurIPS 2023) — CENTRAL PAPER
- **Authors:** Shashank Subramanian, Peter Harrington, Kurt Keutzer, Wahid Bhimji, Dmitriy Morozov, Michael Mahoney, Amir Gholami.
- **arXiv:** 2306.00258. https://arxiv.org/abs/2306.00258, full text via https://ar5iv.labs.arxiv.org/html/2306.00258
- **PDEs used:** Three 2D steady-state systems — Poisson's equation (SYS-1, varying diffusion-tensor anisotropy), Advection-Diffusion (SYS-2, competing advective/diffusive regimes), and Helmholtz (SYS-3, inhomogeneous, high-frequency oscillatory solutions). Architecture is FNO-based throughout (authors explicitly flag this as a scope limitation — no other architecture family tested).
- **Positive transfer, exact numbers:** For Poisson's equation, reaching a target error of 1e-2 needs **only ~64 downstream fine-tuning examples with pretraining, vs ~8,000 examples training from scratch — about a 100× reduction** in labeled data needed. Similar order-of-magnitude (10³–10⁴ example) savings reported for Advection-Diffusion in few-shot regimes.
- **Model-size scaling:** Across model sizes from 64K to 256M parameters (a "44,000×" parameter range), fine-tuned models' error "monotonically drops" with size, and **fine-tuning gets more benefit from added parameters than training from scratch does** — i.e., pretraining's advantage grows with model scale, not just data scale.
- **NEGATIVE TRANSFER, explicitly documented (key for the "against" side of the claim):**
  - Under **large out-of-distribution parameter shifts, zero-shot and few-shot performance is poor** — the authors state transfer learning "showing relatively high errors" in this regime.
  - **Helmholtz (SYS-3) is called out by name as "a particularly challenging system that shows larger performance drops as we go OOD"** — i.e., transfer benefit is system-dependent and can largely disappear for harder/oscillatory PDEs.
  - **Diminishing returns at large downstream-data regimes:** once fine-tuning data approaches the size of the pretraining set itself (~2^15 examples), the pretraining advantage plateaus — pretraining only helps in the data-scarce regime, not universally.
- **Assessment:** This is the paper closest to directly testing "does multi-physics pretraining transfer help," and its answer is conditional: strong, well-quantified data-efficiency gains in the low-data/near-distribution regime, explicitly vanishing gains under large distribution shift or once downstream data is abundant. It also only spans 3 related steady-state elliptic-type systems, not genuinely disparate physics (no fluid dynamics vs. electromagnetism vs. structural mechanics comparison, for instance) — this limits how far it can support the "one model beats single-domain models across *very different* areas of physics" version of the claim.

---

## 9. The Well dataset (Polymathic AI)
- **Title:** "The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning."
- **Authors:** Ruben Ohana, Michael McCabe, Lucas Meyer, and 24 further collaborators (Polymathic AI collective).
- **Venue/arXiv:** NeurIPS 2024, Datasets and Benchmarks Track; arXiv:2412.00568. https://arxiv.org/abs/2412.00568
- **Content:** 16 physics domains (biological systems, fluid dynamics, acoustic scattering, magnetohydrodynamics including extragalactic fluids and supernova explosions, plus further spatiotemporal systems) totalling **15 terabytes**. This is infrastructure — a shared multi-physics benchmark corpus — rather than a foundation model itself; it exists specifically to let others test multi-physics pretraining claims on a common, larger, and more physically diverse dataset than PDEBench.
- **Baselines:** The paper introduces example baselines but the abstract-level fetch did not surface an explicit multi-physics-vs-single-physics accuracy comparison; it does state the diverse baselines "highlight the new challenges posed by the complex dynamics of the Well," i.e. current models struggle broadly across this harder, more diverse benchmark. Numeric baseline tables would need the full PDF.

---

## 10. PDEArena / Gupta & Brandstetter — "Towards Multi-Spatiotemporal-Scale Generalized PDE Modeling"
- **Authors:** Jayesh K. Gupta, Johannes Brandstetter.
- **arXiv:** 2209.15616 (Sep 2022). https://arxiv.org/abs/2209.15616
- **Content:** Compares FNOs, ResNets, and modernized U-Nets (borrowing techniques from vision/generative modeling) on fluid-mechanics PDEs in vorticity-stream and velocity-function form, and shows a single surrogate model can generalize across different PDE parameters and different time-scales. The abstract-level fetch did not surface exact quantitative joint-vs-separate training numbers; would need full PDF for tables. Code: microsoft/pdearena.

---

## 11. Weather/climate foundation models (the mature, operationally-validated example)

### Aurora (Microsoft) — Bodnar, Bruinsma, Lucic, Stanley, Vaughan, Brandstetter, Garvan, Riechert, Weyn, Dong, Gupta, Thambiratnam, Archibald, Wu, Heider, Welling, Turner, Perdikaris
- **arXiv:2405.13063**, https://arxiv.org/abs/2405.13063, full text https://arxiv.org/html/2405.13063
- **Pretraining:** "over a million hours" of heterogeneous atmospheric data drawn from ERA5, HRES, IFS ensembles, GFS, GEFS reforecasts, CMIP6, MERRA-2, and CAMS — genuinely multi-source, multi-resolution, multi-variable pretraining. 150k training steps on 32 A100 GPUs (~2.5 weeks).
- **Resolution handling:** A "flexible encoder" maps datasets of different native resolutions/variables/pressure levels into a standardized 3D latent representation using variable patch sizes and a fixed number (L=3) of latent pressure levels — this is the clearest working example of the "resolution flexibility" half of the podcast's claim, already in operational-grade use.
- **Downstream fine-tuning, exact numbers:**
  - Air quality (vs. CAMS): matches/outperforms CAMS on 74% of all targets, rising to 89% at the 3-day mark; ~50,000× faster than CAMS; pretraining beats training-from-scratch by an "average magnitude of 54%."
  - Ocean waves (vs. HRES-WAM): matches/outperforms on 86% of variables (91% at 3-day mark).
  - Tropical cyclone tracks (vs. official agency forecasts): 6% better at day 1, 20–25% better at days 2–5, roughly 20% better than other forecasting agencies in North Atlantic/East Pacific.
  - High-res weather (0.1°, vs. IFS HRES): lower RMSE on 92% of variables/levels/lead-times; up to 24% RMSE reduction at lead times >12h; wind speed better than IFS HRES at all lead times to 10 days; pretrained model beats from-scratch training by 25%.
- **Explicit non-wins admitted:** CAMS still beats Aurora on ozone in the upper atmosphere and on all species at the 12-hour lead time; IFS HRES beats Aurora at the shortest lead times for many targets; on ocean waves Aurora loses on one variable (PP1D) at the 3-day mark. Aurora is also deterministic — no ensemble/uncertainty output — and still depends on externally-produced initial conditions from traditional data assimilation.

### GraphCast (DeepMind) — Lam et al.
- **arXiv:2212.12794**, published Science 382(6677):1416–1421, 2023 (doi 10.1126/science.adi2336).
- Operates at 0.25° (~25 km) resolution, 6-hour timestep; **outperforms ECMWF's HRES on 90% of 1,380 verification targets** (other reporting cites 89.3% of 2,760 target/lead-time combinations, and >99% within the troposphere specifically). Produces a 10-day global forecast in under 60 seconds on one Cloud TPU. GraphCast is trained/evaluated at essentially one fixed operational resolution — it demonstrates massive speed and accuracy gains over classical numerical weather prediction, but not the "one model spans many native resolutions" property Aurora demonstrates.

### GenCast (DeepMind) — Price, Sanchez-Gonzalez, Alet, Andersson, El-Kadi et al.
- **arXiv:2312.15796**, published Nature (Dec 2024), doi 10.1038/s41586-024-08252-9.
- Diffusion-based ensemble forecaster (spherical-geometry adapted) at 0.25° resolution, ensembles of 50+ trajectories; **beats ECMWF's ENS ensemble system on 97.2% of 1,320 verification targets**, produces a 15-day forecast in ~8 minutes on one TPU v5. This is the first systematic AI win over a full physics-based ensemble system, not just a deterministic one.

### NeuralGCM — Kochkov, Yuval, Langmore, Norgaard, Smith, Mooers, Klöwer, Lottes, Rasp, Düben, Hatfield, Battaglia, Sanchez-Gonzalez, Willson, Brenner, Hoyer et al.
- **arXiv:2311.07222**, published Nature 632, 2024 ("Neural General Circulation Models for Weather and Climate"), doi 10.1038/s41586-024-07744-y.
- Hybrid: a differentiable dynamical-core solver coupled with learned physics-parameterization components, unlike Aurora/GraphCast/GenCast which are purely learned. Runs at coarser resolution (2.8°, 37 vertical levels) — **8–40× coarser horizontal resolution than ECMWF's IFS** and coarser than global cloud-resolving models — while matching their skill, for **3–5 orders of magnitude compute savings**. This is a genuinely different "resolution flexibility" story: it doesn't claim to work seamlessly across resolutions, it trades resolution for skill-preserving efficiency by learning what the missing fine-scale physics would have done.

---

## 12. Self-improvement / physics-residual-as-signal, and reasoning/inference-time scaling for PDE models

- **"Can Physics Informed Neural Operators Self Improve?"** — Ritam Majumdar, Amey Varhade, Shirish Karande, Lovekesh Vig (TCS Research). NeurIPS 2023 workshop spotlight; **arXiv:2311.13885**. https://arxiv.org/abs/2311.13885
  - Uses physics loss (PDE residual) with no labeled data, self-training via iterative pseudo-labeling on Fourier Neural Operators.
  - **Exact numbers:** self-trained, physics-loss-only FNOs reach **1.07× the error of fully-supervised (data+physics) training on Burgers' equation, and 1.02× on Darcy flow** — i.e. self-training with physics residuals as the only signal gets within 2–7% of the accuracy of models trained with real labeled data. Authors also find pseudo-labeling doesn't need full convergence each iteration, allowing faster self-training schedules.
  - Scope: only two canonical small PDEs (1D Burgers, 2D Darcy) — not tested at foundation-model scale or across many physics domains.

- **PIRF — "Physics-Informed Reward Fine-Tuning for Diffusion Models,"** Mingze Yuan et al., **arXiv:2509.20570**. Uses PDE residuals (local, finite-difference-style) as a reward signal to fine-tune diffusion models for PDE generation; introduces layer-wise truncation (updating only higher-resolution layers) motivated by the locality of physics-based rewards. Validated on five PDE benchmarks; reported to "achieve high physical precision under efficient sampling regimes" — exact numeric tables were not extracted in this pass (would need the full PDF).

- **"Towards Reasoning for PDE Foundation Models: A Reward-Model-Driven Inference-Time-Scaling Algorithm,"** Mansingh, Amarel, Arnab, Mohan, Singh, Kunde, Hengartner, Migliori, Casleton, Debardeleben, Biswas, Oyen, Lawrence. **arXiv:2509.02846** (Sep 2025, rev. Jan 2026). Addresses the admitted weakness that "existing [PDE foundation] models remain constrained by the pretraining datasets and struggle with auto-regressive rollout performance, especially in out-of-distribution cases" — directly corroborating the negative-transfer pattern seen in Subramanian et al. and DPOT above. Introduces test-time compute / reward-model-guided inference for stochastic PDE predictors; shown to improve over standard non-adaptive autoregressive inference, but exact improvement numbers were not extractable from the abstract-level fetch.

- **2025–2026 successor foundation models found by search (not in the original list, offered as current state of the art):**
  - **"Towards a Foundation Model for Partial Differential Equations Across Physics Domains,"** Soares, Vital Brazil, Shirasuna, de Carvalho, Malossi (AAAI 2026 AI2ASE Workshop, submitted Nov 2025), **arXiv:2511.21861**. Mamba-based state-space backbone with spatial-spectral tokenization and an operator-theoretic decoder, evaluated on 12 datasets from The Well spanning hydrodynamic, radiative, elastic, and astrophysical domains — genuinely disparate physics, closer to the podcast's framing than MPP/DPOT/Poseidon. **Reports state-of-the-art on 6 of 12 domains and a 46% mean VRMSE reduction vs prior operator-learning baselines** — but also, by implication, does *not* achieve SOTA on the other 6 of 12 domains, a partial (not universal) win worth flagging explicitly.
  - **"Physics-informed fine-tuning of foundation models for partial differential equations,"** Medvedev, Armbruster, Straub, Kruse, Rosskopf, ICLR 2026 Workshop on AI and PDEs, **arXiv:2603.15431** (Mar 2026). Adapts pretrained PDE foundation models via PDE-residual/boundary-condition-constrained fine-tuning losses; claims competitive accuracy on unseen PDE classes without needing PDE solutions for fine-tuning, and that a hybrid (data + physics) strategy generalizes best under minimal data. Exact numeric comparisons UNVERIFIED here (abstract-level only).
  - **OmniArch — "Building Foundation Model For Scientific Computing,"** arXiv:2402.16014, ICML 2025. United 1D-2D-3D pretraining on PDEBench with a Fourier encoder-decoder plus transformer backbone and a "PDE-Aligner" for physics-informed fine-tuning; claims new benchmarks on 1D/2D/3D PDEs and in-context/zero-shot adaptation to new physics. Exact numbers UNVERIFIED here (would need full PDF).

---

## 13. Long-rollout error accumulation and the pushforward trick

- **MP-PDE (Message Passing Neural PDE Solvers)** — Johannes Brandstetter, Daniel Worrall, Max Welling. ICLR 2022 Spotlight. **arXiv:2202.03376**. https://ar5iv.labs.arxiv.org/html/2202.03376
- **The problem:** an autoregressive solver trained only on one-step ground-truth transitions sees a different input distribution at test time — its own imperfect rollout predictions, not clean ground truth — so small per-step errors compound ("distribution shift"); small errors in the learned operator accumulate over any rollout longer than 1 step, driving predictions away from the true trajectory.
- **The pushforward trick, mechanically:** frame stability as a domain-adaptation problem. Unroll the model two steps during training, but only backpropagate the loss on the second step; the first step's output (a self-generated, imperfect input) is used as an adversarial-style perturbation to expose the network to distribution-shifted inputs, without letting gradients optimize away that perturbation (which would defeat its purpose).
- **Exact numbers found (Table 1 / Figure 5(b) of the paper):**
  - On Burgers' equation (experiment E1, resolution n_x=100): survival time before >10% error roughly doubles with the pushforward trick (from ~2–3 seconds to ~5–6 seconds) vs. no stabilization.
  - Numeric error comparison: WENO5 baseline error 6.23; plain unrolled FNO-RNN (no pushforward) error 29.98; FNO with pushforward trick + temporal bundling, error 0.51 — roughly a **58× error reduction** vs. unrolled training without the trick.
  - For a generalization test with variable PDE coefficients (E3, n_x=50): MP-PDE without the pushforward-related term scores error 10.90, vs. 3.74 with pushforward + bundling (**~3× improvement**).
  - **Negative result on the natural alternative:** plain Gaussian noise injection into training inputs "actually performs worse than no stabilization" in this paper's experiments, despite slightly improving stability — i.e. naive noise injection is not a safe substitute for the adversarial pushforward formulation.
  - Temporal bundling (predicting several future steps per forward pass) is used jointly with pushforward in the best configurations above; the paper does not report a clean pushforward-alone vs. bundling-alone breakdown in the sections retrieved.

---

## Verdict on the multi-physics transfer claim

**What the public record supports:**
1. **Data-efficiency after multi-PDE pretraining is real and well-quantified in several independent papers**, not just marketing: Poseidon needs ~20 fine-tuning samples where FNO needs 1,024 on the median of 15 downstream tasks (~50×, up to ~890× on individual tasks); Subramanian et al. show ~100× fewer downstream examples needed for target accuracy on Poisson's equation; DPOT shows 19–52% relative error reductions from pretraining depending on the downstream task; Aurora shows pretraining beats from-scratch training by 25–54% depending on the downstream application. These are independent research groups (ETH/Mishra, Berkeley/Gholami, Tsinghua/Anandkumar, Microsoft) converging on the same qualitative pattern using different architectures and different physics.
2. **Resolution/discretization flexibility is real for true neural-operator architectures**, distinct from vision-transformer-style multi-physics models. UPT explicitly demonstrates generalization across different numbers of input/output mesh points with no retraining, and Aurora's flexible encoder ingests genuinely different native resolutions and variable sets into one shared latent space and is already outperforming operational forecasting systems (IFS HRES, CAMS, HRES-WAM) in production-relevant comparisons. This is the strongest, most mature evidence for the "neural operators give resolution flexibility" half of the claim — weather/climate is the one domain where multi-physics, multi-resolution foundation models are already beating the best classical alternative on holdout, operational data, at scale (GraphCast beats ECMWF HRES on 90% of targets; GenCast beats ECMWF's full ensemble system ENS on 97.2% of targets).
3. **Anandkumar's own co-authored paper (DPOT) is itself a data point for the "multi-physics beats single-domain" claim** for fluid-dynamics-family PDEs specifically, with 12 datasets/100k+ trajectories and quantified error reductions of 19–52% from pretraining depending on task.

**What the public record complicates or contradicts:**
1. **No paper in this survey demonstrates one model trained across genuinely disparate areas of physics (e.g., fluids + electromagnetism + solid mechanics + quantum) beating equal-size single-domain specialists across all of them.** MPP, DPOT, and Poseidon's "multi-physics" pretraining sets are all dominated by fluid-dynamics-family PDEs (Navier-Stokes variants, Euler, shallow water, diffusion-reaction) with some elliptic PDEs added — a much narrower notion of "multiple areas of physics" than the podcast framing implies. The one paper found (arXiv:2511.21861, Nov 2025) that spans genuinely different physics types (hydrodynamic, radiative, elastic, astrophysical, via The Well) only reaches state-of-the-art on 6 of 12 domains — a partial win, not a clean sweep.
2. **Multi-task/multi-physics pretraining measurably hurts some individual tasks before fine-tuning.** MPP's own authors state plainly that "the multi-task training does hurt performance on individual tasks," and their Shallow Water Equations task underperforms a dedicated single-task baseline at every model scale they tried, even after fine-tuning is applied at the smallest scale. DPOT is beaten by a competing multi-physics model (MPP-L) on one benchmark (CFDBench), showing multi-physics pretraining is not even uniformly better than other multi-physics pretraining.
3. **Transfer benefit is not universal — it is conditional on distribution shift and data regime.** Subramanian et al., the paper this survey treats as central, explicitly document that under large out-of-distribution parameter shifts "the zero-shot and few-shot performance is poor," single out the Helmholtz (oscillatory) system by name as showing large OOD performance drops, and show the pretraining advantage plateaus once downstream data approaches the pretraining set's own size. A September 2025/January 2026 paper (arXiv:2509.02846) independently states that "existing [PDE foundation] models remain constrained by the pretraining datasets and struggle with auto-regressive rollout performance, especially in out-of-distribution cases" — this is the same failure mode described from a different angle two years later, suggesting it remains unsolved.
4. **Long-horizon autoregressive rollout error accumulation is a separate, still-only-partially-solved problem orthogonal to the pretraining question.** The pushforward trick (Brandstetter et al. 2022) roughly doubles stable rollout duration and cuts error by up to ~58× versus naive unrolled training in the specific experiments reported, and naive noise injection (a natural competing fix) actually performs worse than doing nothing — but even DPOT's pretrained large model still has 33.5% full-trajectory error on long Kolmogorov-turbulence rollouts after pretraining, down from 82.2% without it: better, not solved.
5. **In-context / no-fine-tuning approaches (ICON) have essentially no quantitative baseline comparison in the primary paper** and the authors themselves admit their approach "can hardly generalize... beyond" a single trained-on ODE family when training diversity is insufficient — this is the weakest-evidenced item in the survey, more a promising demonstration than a validated claim.

**Bottom line:** The core empirical claim — multi-PDE-family pretraining measurably improves downstream data efficiency and accuracy versus training a same-size model from scratch, and true neural-operator architectures can generalize across resolutions — is supported by multiple independent, peer-reviewed/preprint results with specific, reproducible numbers (Poseidon's ~20-vs-1024-sample result and Subramanian's ~64-vs-8000-example result are the two cleanest). The stronger, more sweeping version of the claim implied by "multiple areas of physics" — genuinely disparate physical domains, not just fluid-dynamics-family PDE variants, all improved by one shared model — is only weakly supported (one late-2025 paper, partial 6-of-12-domain win) and is complicated by documented cases of negative transfer, multi-task interference, and rollout-error problems that remain open as of the most recent (2025–2026) papers surveyed.
