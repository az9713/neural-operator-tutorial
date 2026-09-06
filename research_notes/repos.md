# Neural Operator / SciML Software Survey — Teaching Notes

Compiled 2026-09-05. Method: shallow clones (`git clone --depth 1`) into
`scratchpad/repos/`, read via `cat`/`find`/`grep`; hosted docs via WebFetch.
Every fact below is cited to a file path (repo-relative) or a URL. Anything
I could not verify is marked **UNVERIFIED**. Star counts come from a live
WebFetch of the GitHub page on 2026-09-05, not invented.

---

## 1. neuraloperator/neuraloperator

**Repo facts**
- Last commit (shallow clone HEAD): 2026-08-06, "fix typos Flattened1dConv class (#738)" (`git log -1` in cloned repo).
- Stars: 3.9k, forks: 925 (GitHub page, fetched 2026-09-05).
- License: MIT (`LICENSE` file present at repo root).
- Docs: https://neuraloperator.github.io/dev/index.html (linked from `README.rst`).
- Part of the PyTorch Ecosystem per `README.rst` line ~15 (links to a PyTorch blog announcement).

**Install** (`README.rst`)
- Editable/source: `git clone ...`, `pip install -e .`, `pip install -r requirements.txt`.
- PyPI: `pip install neuraloperator`.
- Optional W&B logging: drop a `wandb_api_key.txt` file in `config/`.

**Package layout** (`neuralop/`)
- `models/` — model classes.
- `layers/` — building-block modules (spectral convs, GNO/attention kernels, embeddings, normalization, padding, etc.).
- `data/datasets/` — dataset loaders + a few bundled small `.pt` files.
- `data/transforms/` — normalizers, data processors, patching transforms.
- `losses/` — data losses, equation (PDE) losses, meta/multi-loss aggregators, finite-difference/Fourier differentiation utilities.
- `training/` — `Trainer`, incremental-training trainer, optimizer (AdamW variant), tensorized-gradient utility, patching.
- `mpu/` — multi-processing / distributed communication utilities.

**Model classes** (`neuralop/models/*.py`, confirmed by `grep "^class "`)
- `FNO` (`fno.py:25`, subclasses `BaseModel`) — the base Fourier Neural Operator, 1D/2D/3D per docs.
- `TFNO` (`fno.py:449`, subclasses `FNO`) — Tucker-factorized FNO (fewer parameters via tensor factorization; README shows a code sample passing `factorization='tucker'`).
- `SFNO` (`sfno.py`) — Spherical FNO; only importable if `torch_harmonics` is installed locally (`models/__init__.py` wraps the import in `try/except`).
- `LocalNO` (`local_no.py:24`) — same optional-import guard as SFNO.
- `UNO` (`uno.py:19`) — U-shaped Neural Operator (plain `nn.Module`, not `BaseModel`).
- `UQNO` (`uqno.py:10`) — Uncertainty-Quantification Neural Operator.
- `FNOGNO` (`fnogno.py:20`) — hybrid FNO + Graph Neural Operator (for geometry-informed / irregular-domain problems).
- `GINO` (`gino.py:22`) — Geometry-Informed Neural Operator.
- `CODANO` (`codano.py:14`) — Co-domain-Attention Neural Operator (uses `CODALayer`, see below).
- `RNO` (`rno.py:24`) — Recurrent Neural Operator (has its own `RNOCell`/`RNOBlock` in `layers/rno_block.py`).
- `OTNO` (`otno.py:7`, subclasses `FNO`) — Optimal-Transport Neural Operator (used in the car-CFD example, see below).
- `base_model.py` also exposes `get_model()`, a factory function, and `BaseModel`.

**Layer modules** (`neuralop/layers/*.py`)
- Spectral: `spectral_convolution.py` (`SpectralConv`), `legacy_spectral_convolution.py` (`SpectralConv`, `SpectralConv1d/2d/3d` — legacy/explicit-dimension versions), `spherical_convolution.py` (`SHT`, `SphericalConv`), `base_spectral_conv.py` (abstract base).
- FNO block assembly: `fno_block.py` (`FNOBlocks`, `SubModule`).
- GNO / integral-transform family: `gno_block.py` (`GNOBlock`), `integral_transform.py` (`IntegralTransform`), `neighbor_search.py` (`NeighborSearch`), `gno_weighting_functions.py`, `attention_kernel_integral.py` (`AttentionKernelIntegral`).
- Discrete-continuous convolutions (used by SFNO/spherical models): `discrete_continuous_convolution.py` — `DiscreteContinuousConv` (abstract), `DiscreteContinuousConv2d`, `DiscreteContinuousConvTranspose2d`, `EquidistantDiscreteContinuousConv2d`, `EquidistantDiscreteContinuousConvTranspose2d`.
- Other building blocks: `channel_mlp.py` (`ChannelMLP`, `LinearChannelMLP`), `coda_layer.py` (`CODALayer`), `complex.py` (`ComplexValued`), `differential_conv.py` (`FiniteDifferenceConvolution`), `embeddings.py` (`GridEmbedding2D`, `GridEmbeddingND`, `SinusoidalEmbedding`, `RotaryEmbedding2D`), `fourier_continuation.py` (`FourierContinuation`, `FCLegendre`, `FCGram`), `local_no_block.py` (`LocalNOBlocks`), `normalization_layers.py` (`AdaIN`, `InstanceNorm`, `BatchNorm`), `padding.py` (`DomainPadding`), `resample.py`, `rno_block.py` (`RNOCell`, `RNOBlock`), `skip_connections.py` (`SoftGating`, `Flattened1dConv`), `spectral_projection.py`.

**Losses** (`neuralop/losses/*.py`)
- `data_losses.py`: `LpLoss`, `H1Loss`, `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss`.
- `equation_losses.py` (PDE-residual / physics losses): `BurgersEqnLoss`, `ICLoss` (initial condition), `PoissonInteriorLoss`, `PoissonBoundaryLoss`, `PoissonEqnLoss`.
- `meta_losses.py` (multi-loss weighting/aggregation): `FieldwiseAggregatorLoss`, `WeightedSumLoss`, `Aggregator`, `SoftAdapt`, `Relobralo`.
- `differentiation.py`: `FiniteDiff`, `FourierDiff` — differentiation utilities used by the equation losses.

**Training utilities** (`neuralop/training/`)
- `trainer.py`: `Trainer` class (`trainer.py:27`). Docstring notes it expects datasets to yield dict batches like `{'x': x, 'y': y}` keyed to model/loss argument names, with `DarcyDataset` cited as the reference example. Methods: `__init__` (`:57`), `train` (`:98`), `train_one_epoch` (`:266`), `evaluate_all` (`:336`), `evaluate` (`:385`), `train_one_batch` (`:487`). Supports optional Weights & Biases logging (guarded import) and `DistributedDataParallel`.
- `incremental.py`: `IncrementalFNOTrainer` (`:9`, subclasses `Trainer`) — implements incremental (progressively-more-modes / progressively-more-resolution) training for FNO.
- `adamw.py` — a custom AdamW optimizer variant.
- `tensorgrad.py` — tensorized-gradient training utility (used by the `plot_tensorgrad_fno_darcy.py` example, see below).
- `patching.py` — patch-based training utility (for training on sub-patches of large fields).
- `training_state.py` — `load_training_state` / `save_training_state` (checkpointing).

**Datasets and data loaders** (`neuralop/data/datasets/`)
- `darcy.py`: `DarcyDataset`, `load_darcy_flow_small` — Darcy-flow coefficient→pressure-field dataset. Docstring cites data source https://zenodo.org/records/12784353. Bundled small tensors: `data/darcy_train_16.pt`, `data/darcy_test_16.pt`, `data/darcy_test_32.pt` (confirmed present in the cloned repo, i.e. resolution-16 training set and 16/32 test sets ship with the package for quick-start use).
- `navier_stokes.py`: `NavierStokesDataset`, `load_navier_stokes_pt` — 2D Navier-Stokes vorticity dataset loader (downloadable, not bundled as small `.pt` in this listing).
- `burgers.py`: `Burgers1dTimeDataset`, `load_mini_burgers_1dtime` — bundled as `data/burgers_train_16.pt`, `data/burgers_test_16.pt` (small 1D time-dependent Burgers data, confirmed present).
- `pt_dataset.py`: `PTDataset` — generic base class for the `.pt`-file-backed datasets (Darcy, Navier-Stokes, Burgers all subclass or reuse it).
- `car_cfd_dataset.py`: `CarCFDDataset`, `load_mini_car` — mini car-CFD dataset (`data/mini_car.pt` bundled). Used in `examples/data/plot_mini_car_cfd.py` and the OTNO car example.
- `car_ot_dataset.py`: `CarOTDataset`, `load_saved_ot`, `CFDDataProcessor` — Optimal-Transport preprocessed version of the car dataset (`data/ot_expand3.0_reg1e-06_train2_test1.pt` bundled).
- `ot_datamodule.py`: `OTDataModule`.
- `mesh_datamodule.py`: `MeshDataModule` — mesh-based data handling (for GNO/FNOGNO/GINO-style irregular geometries).
- `nonlinear_poisson.py`: nonlinear Poisson dataset (downloadable).
- `hdf5_dataset.py`, `zarr_dataset.py`, `tensor_dataset.py`, `dict_dataset.py` (`DictDataset`) — generic backends for HDF5/Zarr/raw-tensor/dict-style data.
- `spherical_swe.py`: `load_spherical_swe` — spherical shallow-water-equations dataset, only importable if `torch_harmonics` is installed (guarded import, same pattern as SFNO).
- `the_well_dataset.py`: `TheWellDataset`, `ActiveMatterDataset`, `MHD64Dataset` — integration with the Polymathic AI "the Well" dataset collection (item 8 below), only importable if the `the_well` package is installed.
- `web_utils.py`: `download_from_zenodo_record` — shared Zenodo download helper (used by `DarcyDataset` etc.).

**Examples directory** (`examples/`, all files confirmed present by `find`)
- `examples/models/`: `plot_FNO_darcy.py`, `plot_OTNO_car_cfd.py`, `plot_SFNO_swe.py`, `plot_UNO_darcy.py`.
- `examples/layers/`: `plot_DISCO_convolutions.py`, `plot_embeddings.py`, `plot_finite_diff.py`, `plot_fourier_continuation.py`, `plot_fourier_diff.py`, `plot_neighbor_search.py`, `plot_normalization_layers.py`, `plot_resample.py`, `plot_sinusoidal_embeddings.py`, `plot_spectral_projection.py`.
- `examples/training/`: `checkpoint_FNO_darcy.py`, `plot_count_flops.py`, `plot_incremental_FNO_darcy.py`, `plot_tensorgrad_fno_darcy.py`, plus a notebook `tensorgrad_fno.ipynb`.
- `examples/data/`: `plot_darcy_flow.py`, `plot_darcy_flow_spectrum.py`, `plot_mini_car_cfd.py`, and a `README.rst` describing this as the "Data" example gallery section.
- `examples/data_gen/`: `plot_burgers_2d_solver.py`, `plot_diffusion_advection_solver.py` (scripts that generate/solve small PDE data rather than just load it), plus a demo GIF `burgers2d.gif`.
- These are Sphinx-Gallery `.py` scripts (`GALLERY_HEADER.rst` in each folder), i.e. they render into the docs' "Examples" gallery (https://neuraloperator.github.io/dev/auto_examples/index.html).

**Docs / User Guide** (WebFetch of https://neuraloperator.github.io/dev/index.html and .../user_guide/index.html)
- Top nav: Install, Theory Guide, User Guide, API, Examples, Developer's Guide.
- User Guide pages, in order: (1) "NeuralOperator library structure" (models/layers/data loaders/training/losses/mpu overview); (2) "Available Neural Operator Models" (FNO 1D/2D/3D, TFNO, SFNO, UNO, UQNO, FNOGNO, GINO, LocalNO, CODANO); (3) "Data Loading and Preprocessing" (Darcy/Burgers/Navier-Stokes loaders + normalizers/data processors); (4) "Training Neural Operator Models" with four subsections — Trainer class, Available Training Components (optimizers + specialized trainers), Loss Functions, Distributed Training; (5) "CPU Offloading" (technique for inputs too large for GPU memory, with a worked trade-off example); (6) "Interactive examples with code" (links out to the examples gallery).
- Practical-guide paper linked from the README: arXiv:2512.01421, "Fourier Neural Operators Explained: A Practical Perspective" — good supplementary reading for a course.

**Directly reusable as course exercises**
- `examples/models/plot_FNO_darcy.py` and `examples/models/plot_UNO_darcy.py` are turnkey, small (Darcy-16/32 bundled data), and demonstrate the canonical train/evaluate loop with `Trainer` + `LpLoss`/`H1Loss` — good "Week 1" exercise.
- `examples/training/plot_incremental_FNO_darcy.py` demonstrates `IncrementalFNOTrainer` — good for a lesson on progressive/curriculum training of spectral models.
- `examples/layers/plot_DISCO_convolutions.py` and `plot_fourier_continuation.py` isolate individual layer mechanics (discrete-continuous conv, Fourier continuation) for a "how does the layer work" lesson, independent of a full model.
- `examples/data_gen/plot_burgers_2d_solver.py` and `plot_diffusion_advection_solver.py` show how the small bundled datasets are themselves generated — useful for teaching "where does the training data come from."
- The mini car-CFD dataset (`load_mini_car`, `plot_mini_car_cfd.py`) plus `plot_OTNO_car_cfd.py` is the only bundled 3D/irregular-geometry example and is a natural bridge exercise toward GINO/FNOGNO/OTNO on real geometries.

---

## 2. pdebench/PDEBench

**Repo facts**
- Last commit (shallow clone HEAD): dated 2026-03-30, message "bug fix on 2026-03-30" (`git log -1`).
- Stars: 1.2k, forks: 153 (GitHub page, fetched 2026-09-05).
- License: MIT ("MIT licensed, except where otherwise stated" per GitHub page; `LICENSE.txt` in repo).
- Paper: NeurIPS 2022, "PDEBench: An Extensive Benchmark for Scientific Machine Learning" (arXiv:2210.07182); won the SimTech Best Paper Award 2023 (`README.md`).

**Maintenance statement** (`README.md`, section "Compatibility with Newer Python, JAX, PyTorch, and CUDA Versions", last updated 2026-03-30)
- Officially targets Python 3.9, JAX 0.4.11, PyTorch 1.13.0, CUDA 11.7.
- States they have verified compatibility with newer releases: Python 3.12, JAX 0.9.2, PyTorch 2.11.0, CUDA 13.0 — specifically the core codebase, forward training in `pdebench/model`, and data generation in `data_gen_NLE`. Compatibility of "the remaining components is still under investigation."
- A separate noted bug: a Hydra data-path issue in `model/fno/utils.py` (lines 185–188) and `model/unet/utils.py` (lines 185–187), with a documented workaround (also dated 2026-03-30).

**Install** (`README.md`)
- `pip install --upgrade pip wheel && pip install .` (local) or `pip install pdebench` (PyPI).
- Data-generation extras: `pip install "pdebench[datagen310]"` or `pip install "pdebench[datagen39]"`.
- GPU notes: PyTorch pinned to v1.13.1 for CUDA 11.7; JAX noted as "approximately 6 times faster for simulations than PyTorch in our tests."
- Conda alternative given, including `conda install deepxde hydra-core h5py -c conda-forge` plus PyTorch-CUDA or CPU-only variants, and optional `conda install clawpack jax jaxlib python-dotenv` for data generation.
- DeepXDE backend must be configured to PyTorch per DeepXDE's own docs (linked).

**Data generation code** (`pdebench/data_gen/`, per README)
- Top-level scripts: `gen_diff_react.py` (2D diffusion-reaction), `gen_diff_sorp.py` (1D diffusion-sorption), `gen_radial_dam_break.py` (2D shallow-water), `gen_ns_incomp.py` (2D incompressible inhomogeneous Navier-Stokes), plus `plot.py` and `uploader.py` (pushes generated data to the Dataverse repository via a `.env` config file).
- `data_gen_NLE/` subdirectory (1D Advection/Burgers/Reaction-Diffusion, 2D DarcyFlow, compressible Navier-Stokes): `utils.py` (BC/IC utilities), `AdvectionEq/`, `BurgersEq/`, `CompressibleFluid/`, `ReactionDiffusionEq/` (also generates 2D DarcyFlow via `run_DarcyFlow2D.sh`), and a `save/` output directory. Example invocation given: `python3 advection_multi_solution_Hydra.py +multi=beta1e0.yaml` (Hydra-configured).
- `data_gen_NLE/Data_Merge.py` (config at `data_gen/data_gen_NLE/config/config.yaml`) merges the raw numpy output into the HDF5 format the dataloaders expect.

**Data download** (`pdebench/data_download/README.md`)
- Two scripts: `download_direct.py` (recommended, pulls shards directly by URL) and `download_easydataverse.py` (documented as slower/error-prone, not recommended).
- Storage backend: DaRUS (University of Stuttgart Dataverse), DOI 10.18419/darus-2986 for datasets and 10.18419/darus-2987 for pretrained models.
- Dataset sizes, from the download-README table:

| PDE | Download flag | Size |
|---|---|---|
| advection | `--pde_name advection` | 47 GB |
| burgers | `--pde_name burgers` | 93 GB |
| 1d_cfd | `--pde_name 1d_cfd` | 88 GB |
| diff_sorp | `--pde_name diff_sorp` | 4 GB |
| 1d_reacdiff | `--pde_name 1d_reacdiff` | 62 GB |
| 2d_reacdiff | `--pde_name 2d_reacdiff` | 13 GB |
| 2d_cfd | `--pde_name 2d_cfd` | 551 GB |
| 3d_cfd | `--pde_name 3d_cfd` | 285 GB |
| darcy | `--pde_name darcy` | 6.2 GB |
| ns_incom | `--pde_name ns_incom` | 2.3 TB |
| swe | `--pde_name swe` | 6.2 GB |

  (Command form: `python download_direct.py --root_folder $proj_home/data --pde_name <name>`.)
- A `visualize_pdes.py` script turns a downloaded shard into an animated GIF, with per-PDE example invocations and direct DaRUS file URLs given in the README (e.g. 1D diffusion-sorption file at `https://darus.uni-stuttgart.de/api/access/datafile/133020`).

**Baseline models / package layout** (`pdebench/models/`, confirmed by `find`)
- Subdirectories: `fno/`, `unet/`, `pinn/`, `inverse/`, `config/` (with `config/args/` holding per-equation YAML templates).
- Three baselines total, per README: FNO (adapted from the original `zongyi-li/fourier_neural_operator` repo), U-Net (adapted from `mateuszbuda/brain-segmentation-pytorch`), and PINN (built on the DeepXDE library).
- Driver scripts: `train_models_forward.py` (forward problems), `train_models_inverse.py` (inverse problems), `metrics.py` (evaluation + plotting), `analyse_result_forward.py` / `analyse_result_inverse.py` (aggregate results to CSV + bar charts).
- Example shell scripts: `run_forward_1D.sh`, `run_inverse.sh`.
- Config args documented in README: `model_name` (FNO/Unet/PINN), `if_training`, `continue_training`, `num_workers`, `batch_size`, `initial_step`, `t_train`, `model_update`, `filename`, `single_file`, `reduced_batch`, etc.

**Directly reusable as course exercises**
- The `1d_cfd`/`darcy`/`diff_sorp` datasets are the smallest (4–88 GB) and most tractable for a course budget; `ns_incom` at 2.3 TB is not realistically downloadable for a class.
- `pdebench/models/fno/` gives a second, independently-written FNO implementation to compare against neuraloperator's — a good "read two implementations of the same idea" exercise.
- The `data_gen_NLE/AdvectionEq` and `BurgersEq` generators are self-contained enough to have students generate their own small dataset from scratch rather than downloading DaRUS shards, given the compatibility caveats above.

---

## 3. FEniCS / FEniCSx / DOLFINx

**What it is** (https://fenicsproject.org/, WebFetch 2026-09-05)
- "A popular open-source computing platform for solving partial differential equations (PDEs) with the finite element method (FEM)." High-level Python and C++ interfaces; runs from laptops to supercomputers.
- "FEniCSx" is the current generation of the platform; "DOLFINx" (`github.com/FEniCS/dolfinx`) is its computational core/main library. Companion components: UFL (Unified Form Language), Basix (finite-element basis library), FFCx (form compiler) — all linked from the download page per WebFetch.

**Repo facts (DOLFINx)**
- Last commit (shallow clone HEAD): 2026-09-03, "Add option to allow/disallow extrapolation in PointCollisionData. (#4453)" (`git log -1`).
- Stars: 1.2k, forks: 261 (GitHub page, fetched 2026-09-05).
- License: LGPL-3.0-or-later for the library itself (`dolfinx/README.md` "License" section, confirmed against `COPYING.LESSER`); repo also ships a plain GPL-3.0 `COPYING` file (typically covers demos/docs, standard FEniCS practice — **UNVERIFIED** exactly which subtree it governs, not confirmed by direct inspection of file headers).

**Install options** (`dolfinx/README.md`, "Binary packages" section)
- Per-platform recommendation table, quoted directly:
  - **macOS:** conda.
  - **Linux:** apt (Ubuntu/Debian), Docker, or conda; also Spack.
  - **Windows:** Docker, or install WSL2 and use the Ubuntu apt packages; conda packages "in beta testing" on Windows.
  - **HPC:** Spack (recommended) or from source, using system-provided MPI.
- Conda commands:
  ```
  conda create -n fenicsx-env
  conda activate fenicsx-env
  conda install -c conda-forge fenics-dolfinx mpich pyvista   # Linux and macOS
  conda install -c conda-forge fenics-dolfinx pyvista pyamg   # Windows
  ```
  README notes for Windows conda specifically: "PETSc and petsc4py are not available on Windows."
- Docker:
  ```
  docker run -ti dolfinx/dolfinx:stable
  docker run --init -ti -p 8888:8888 dolfinx/lab:stable   # Jupyter Lab at localhost:8888
  ```
  A nightly-build image is also referenced.
- apt (Debian/Ubuntu), from the fenicsproject.org download page:
  ```
  sudo add-apt-repository ppa:fenics-packages/fenics
  sudo apt update
  sudo apt install fenicsx
  ```
  Complex-number build: `sudo apt install python3-dolfinx-complex`.
- Spack (recommended for HPC):
  ```
  git clone https://github.com/spack/spack.git
  . ./spack/share/spack/setup-env.sh
  spack env create fenicsx-env
  spack env activate fenicsx-env
  spack install --add py-fenics-dolfinx+petsc4py+slepc4py
  ```
  A separate FEniCS-maintained Spack overlay repo is also mentioned: `spack repo add https://github.com/fenics/spack-fenics`.
- Source build: `pip install --group pyproject.toml:build` then `pip install --check-build-dependencies --no-build-isolation .` (from `dolfinx/README.md`).

**Windows recommendation for this course**: given the README's own guidance, the practical path on a Windows 11 box is Docker (`docker run -ti dolfinx/dolfinx:stable` or the Lab image) or WSL2 + Ubuntu apt packages; native Windows conda is explicitly beta and missing PETSc/petsc4py.

**Tutorial** (https://jsdokken.com/dolfinx-tutorial/, WebFetch 2026-09-05, by Jørgen S. Dokken)
- Five main sections: Introduction; Fundamentals; "A Gallery of Finite Element Solvers"; Subdomains and Boundary Conditions; Improving Your FEniCSx Code.
- PDE problems covered: Poisson (linear and nonlinear), heat/diffusion equation, linear elasticity and hyperelasticity, Navier-Stokes (channel flow and cylinder-flow benchmarks), Helmholtz/wave problems, electromagnetics, and an image-processing (smoothed TV inpainting) example; also membrane deflection, singular Poisson, and mixed formulations with preconditioners.
- One chapter heading mentions "Adaptive mesh refinement with NetGen and DOLFINx." The WebFetch summary could not find explicit dataset-export-for-ML content in the fetched text — **UNVERIFIED** whether any chapter directly demonstrates exporting solver output as an ML training set; the tutorial's stated focus is solving/visualizing PDEs, not ML pipeline construction.

**How this would generate a training dataset for a neural-operator course**
- Not itself an ML tool: DOLFINx would be used as the "ground-truth solver" — set up a PDE (e.g. Darcy, Poisson, Navier-Stokes) with the tutorial's recipes, sweep initial/boundary conditions or coefficient fields, and export the resulting fields (via DOLFINx's I/O, e.g. to XDMF/HDF5 or numpy arrays) as (input, solution) pairs for training an FNO/GINO/etc. This mirrors what PDEBench's `data_gen`/`data_gen_NLE` scripts already do for their PDEs, and is a natural "build your own PDEBench-style dataset" course exercise, though the mechanics of that export step are not spelled out in the tutorial pages fetched here — treat as an exercise to design, not a ready-made script found in either repo.

---

## 4. google/jax-cfd

**Repo facts**
- Last commit (shallow clone HEAD): 2026-07-08, "Adding type suppressions for pyrefly" (`git log -1`).
- Stars: 963, forks: 141 (GitHub page, fetched 2026-09-05).
- License: Apache-2.0.

**Maintenance statement** (`README.md`, first line after the title, verbatim)
> "🚨 JAX-CFD is no longer maintained. For alternatives, consider [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS), [Phi FLow](https://github.com/tum-pbs/PhiFlow/) or [Exponax](https://github.com/Ceyron/exponax)."

**Authors**: Dmitrii Kochkov, Jamie A. Smith, Peter Norgaard, Gideon Dresdner, Ayya Alieva, Stephan Hoyer (`README.md`). Paper: "Machine learning accelerated computational fluid dynamics," PNAS 2021.

**Organization** (`jax_cfd/` submodules, confirmed present by `ls`)
- `jax_cfd.base` — core finite-volume/finite-difference CFD methods in JAX.
- `jax_cfd.spectral` — pseudospectral CFD methods.
- `jax_cfd.ml` — ML-augmented CFD models, JAX + Haiku.
- `jax_cfd.data` — data processing/eval/post-processing (Xarray + Pillow).
- `jax_cfd.collocated` — an additional submodule present in the clone (experimental collocated-grid solver, per README).
- Base install `pip install jax-cfd` needs only NumPy/SciPy/JAX; extras via `pip install jax-cfd[ml]`, `[data]`, or `[complete]`.

**Numerics** (README "Numerics" section)
- Spatial: finite volume/difference on a staggered ("Arakawa C"/MAC) grid, or pseudospectral for vorticity with anti-aliasing.
- Temporal: first-order only, explicit for advection, implicit or explicit for diffusion.
- Pressure solves: CG or fast diagonalization with real FFTs.
- Boundary conditions: periodic only, currently.
- Advection: 2nd-order "Van Leer" schemes. Closures: Smagorinsky eddy-viscosity.
- README lists a TODO wishlist (collocated grids, non-periodic/immersed boundaries, higher-order time-stepping, geometric multigrid, RANS, distributed multi-TPU/GPU sims) that was, per the maintenance notice, apparently never completed before archival.

**Notebooks** (`notebooks/`, confirmed present)
- `demo.ipynb` — 2D FVM staggered-grid simulation demo.
- `spectral_forced_turbulence.ipynb` — 2D pseudospectral solver demo.
- `channel_flow_demo.ipynb` — 2D channel flow.
- `collocated_demo.ipynb` — experimental collocated-grid FVM demo.
- `ml_accelerated_cfd_data_analysis.ipynb` — reproduces the PNAS paper's data analysis/evaluation.
- `ml_model_inference_demo.ipynb` — running the paper's pretrained models.
- (README also links Colab-hosted versions of the first four.)

**Directly reusable as course exercises**
- Because it's unmaintained but functionally complete and small, `jax_cfd.base` is usable purely as a differentiable ground-truth Navier-Stokes simulator to generate small 2D training data (analogous to what the FNO paper itself did), rather than as a model to build on going forward. For anything meant to keep working, point students at the suggested successors (JAX-Fluids, PhiFlow, Exponax) instead — Exponax in particular is also the numerical backend behind APEBench (item 6 below), so there's a natural through-line.

---

## 5. google-deepmind/deepmind-research — meshgraphnets

**Repo facts**
- Retrieved via sparse checkout of the `deepmind-research` monorepo (`meshgraphnets/*` only), branch `master`.
- Monorepo HEAD at clone time: 2026-09-03 is **NOT** shown — actual last commit fetched was 2023-05-26, "Add code for generating s3o4d data" (this is the monorepo's `git log -1`, i.e. the most recent commit to the whole monorepo touches an unrelated project; the meshgraphnets folder itself may be older — **UNVERIFIED** exact last-touched date for the meshgraphnets subfolder specifically, since `git log -1` reports the repo HEAD, not a path-filtered log, in a depth-1 sparse clone).
- Stars/forks (whole monorepo, since meshgraphnets has no separate repo): 15.2k stars, 2.9k forks, Apache-2.0 license (GitHub page for `google-deepmind/deepmind-research`, fetched 2026-09-05).
- Paper: Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia, "Learning Mesh-Based Simulation with Graph Networks," ICLR 2021, arXiv:2010.03409. Project site: sites.google.com/view/meshgraphnets.

**Files** (`meshgraphnets/`, confirmed by `find`)
- `dataset.py` — shared data loader for all domains.
- `core_model.py` — the learned graph-network model core, shared across domains.
- `cfd_model.py` / `cfd_eval.py` — CFD (`cylinder_flow`) domain model + eval/rollout.
- `cloth_model.py` / `cloth_eval.py` — cloth (`flag_simple`) domain model + eval/rollout.
- `common.py`, `normalization.py` — shared utilities (likely normalization layers/graph-building helpers).
- `plot_cfd.py` / `plot_cloth.py` — trajectory plotting for each domain.
- `run_model.py` — main entry point (`python -m meshgraphnets.run_model --mode=train|eval --model=cfd|cloth ...`).
- `download_dataset.sh` — fetches `meta.json`, `train/valid/test.tfrecord` from `https://storage.googleapis.com/dm-meshgraphnets/<dataset_name>/`.
- `requirements.txt` (verbatim): `tensorflow-gpu>=1.15,<2`, `dm-sonnet<2`, `matplotlib`, `absl-py`, `numpy`.

**TF version confirmed empirically**: `run_model.py` imports `tensorflow.compat.v1 as tf` (line 23) and `requirements.txt` pins `tensorflow-gpu>=1.15,<2` — this is a genuine TF1-style codebase (compat-v1 API), paired with DeepMind Sonnet <2 (also TF1-era).

**Datasets available** (per `meshgraphnets/README.md`, all downloadable via `download_dataset.sh <name> <out_dir>`): `airfoil`, `cylinder_flow`, `deforming_plate`, `flag_minimal`, `flag_simple`, `flag_dynamic`, `flag_dynamic_sizing`, `sphere_simple`, `sphere_dynamic`, `sphere_dynamic_sizing`. Notes: `flag_minimal` is a truncated `flag_simple` used only for integration tests; the two `*_dynamic_sizing` sets additionally provide the pre-remeshing mesh state plus a `sizing_field` target, for learning adaptive remeshing.
- The repo ships only full training/eval pipelines for `cylinder_flow` (CFD) and `flag_simple` (cloth); the other datasets are provided but without a matching example pipeline (per the "Overview" section of the README).

**Known community PyTorch ports** (GitHub search "meshgraphnets pytorch", WebFetch 2026-09-05 — reported counts as returned, not independently re-verified star-by-star)
- `echowve/meshGraphNets_pytorch` — 267 stars, "PyTorch implementations of Learning Mesh-based Simulation With Graph Networks," last updated December 2025 per the search snapshot.
- `wwMark/meshgraphnets` — 105 stars, described as a rewrite of the original DeepMind implementation converted to PyTorch, last updated February 2022.
- `medav/meshgraphnets-torch` — 17 stars, direct PyTorch adaptation.
- Several smaller/course-project forks also surfaced (e.g. `davided0/meshgraphnets_static`, `danimelatru/Meshgraphnet-CFD-Surrogate`, `cberbarbanoj/TIGNN_CYLINDER`) — treat these as unvetted, unofficial.
- Note also: NVIDIA PhysicsNeMo (item 7) ships its own maintained, PyTorch-native `MeshGraphNet` model plus a `vortex_shedding_mgn` example — likely the better-maintained PyTorch path into this architecture for a current course than any of the unofficial community ports.

**Directly reusable as course exercises**
- The `cylinder_flow` + `cfd_model.py`/`cfd_eval.py` pipeline is the cleanest self-contained mesh-based GNN example in this survey, but running it as-shipped requires a TF1/Sonnet<2 environment — for a modern course, better to either (a) use it as a read-only reference for the architecture/loss and reimplement in PyTorch, or (b) point to PhysicsNeMo's MeshGraphNet instead and use this repo only for the original datasets and paper context.

---

## 6. NVIDIA PhysicsNeMo (github.com/NVIDIA/physicsnemo)

**Repo facts**
- Last commit (shallow clone HEAD): 2026-09-03, "Make config-driven field names reach nested TensorDict leaves in datapipes, mesh IO, and the aero recipe (#1961)" — i.e. actively developed as of the survey date.
- Stars: 3.2k, forks: 776 (GitHub page, fetched 2026-09-05).
- License: Apache-2.0 (`LICENSE.txt`).
- Docs: https://docs.nvidia.com/physicsnemo/latest/.

**Rename history: Modulus → PhysicsNeMo**
- The current README/CHANGELOG/FAQ consistently use "PhysicsNeMo" even for the original 2023-05-08 `[0.1.0]` release entry in `CHANGELOG.md` — i.e. the changelog has been retroactively relabeled, so it does not itself narrate the rename.
- Direct evidence of the old "Modulus" name survives only in `docs/research.md`, in citations to third-party papers that used the tool under its old name, e.g.: "Optimization of Two-Element Airfoils Using Nvidia Modulus, a Physics-Informed Neural Network Solver" and "A Reservoir Model Characterization with a Bayesian Framework and a Modulus based Physics constrained Neural Operator" (`docs/research.md`, confirmed by grep).
- `v2.0-MIGRATION-GUIDE.md` documents a related, more recent consolidation: the separately-maintained **PhysicsNeMo-Sym** repository ("PhysicsNeMo Sym → physicsnemo.sym" section) is being archived, with its symbolic-PDE / physics-informed functionality upstreamed into the main package as the `physicsnemo.sym` submodule (installed via `pip install "nvidia-physicsnemo[sym]"`). `FAQ.md` states this explicitly: "The separate PhysicsNeMo-Sym repository is being archived. Its core functionality has been upstreamed into PhysicsNeMo."
- v2.0 (per `v2.0-MIGRATION-GUIDE.md`, "for release in March 2026") is a large refactor: `uv`-based installs, core reorg (`Module`/`Meta` moved into `physicsnemo.core`; `physicsnemo.launch` removed, replaced by `physicsnemo.utils.logging` / `physicsnemo.utils.checkpoint`; many layers moved into `physicsnemo.nn`), model standardization, and a new `physicsnemo.mesh` package (`Mesh`/`DomainMesh`).

**Install**
- `pip install "nvidia-physicsnemo[cu13]"` for CUDA 13 (README quickstart); CUDA 12 and source/dev setups referenced but not detailed in the excerpt read.
- v2.0 recommends `uv sync` for install/dev, states `pip install` remains supported, and that both are tested nightly on Linux/macOS/Windows (`v2.0-MIGRATION-GUIDE.md`).

**Model zoo** (`README.md`, "Choose a Model Family" tables — every model name links to `physicsnemo/models/<name>` in the repo)
- *Surrogates and dynamics:* FNO / DPOT (regular grids; DPOT adds a time axis), MeshGraphNet / VFGN (node-edge graphs on unstructured meshes), Transolver / FLARE (point sets / structured or unstructured discretizations), GeoTransolver (point clouds or grids with geometry + global context), DoMINO (geometry points + surface/volume fields + SDF grids, arXiv:2501.13350), FIGConvNet (large 3D point clouds via factorized 2D grids), GLOBE (*experimental*, boundary meshes + query points, arXiv:2511.15856), AeroJEPA (*experimental*, 3D surface point clouds + operating conditions, arXiv:2605.05586).
- *Weather and climate:* AFNO (regular 2D fields), GraphCast (lat-lon fields + multiscale icosahedral mesh graph, arXiv:2212.12794), Pangu-Weather / FengWu (multilevel lat-lon grids, FengWu arXiv:2304.02948), DLWP / DLWP-HEALPix (cubed-sphere or HEALPix meshes).
- *Generative and inverse:* Diffusion U-Nets / DiT (2D fields or patch tokens — used by StormCast/StormScope and diffusion full-waveform-inversion), TopoDiff (2D topology fields conditioned on design constraints, arXiv:2208.09591).
- Confirms task's asked-about models: **FNO** ✓, **AFNO** ✓, **GraphCast** ✓, **MeshGraphNet** ✓, **DoMINO** ✓, **Transolver** ✓ present. **SFNO is not listed in this model-family table** — **UNVERIFIED**/likely absent from current PhysicsNeMo (it lives in neuraloperator/torch_harmonics instead); did not find an `sfno` entry under `physicsnemo/models/` in this pass.

**PhysicsNeMo-Sym (PINNs)**
- Now `physicsnemo.sym`, installed via the `[sym]` extra. Provides symbolic PDE definition (SymPy-based `physicsnemo.sym.eq.pde.PDE`), automatic spatial-derivative computation, and a `PhysicsInformer` class that computes PDE residuals automatically (`FAQ.md`).
- Worked examples cited by the FAQ: `examples/cfd/ldc_pinns/` (lid-driven-cavity PINN) and `examples/cfd/darcy_physics_informed/`; a custom-PDE example at `examples/cfd/mhd_pino/losses/mhd_pde.py` (magnetohydrodynamics PINO).

**PhysicsNeMo-CFD** (separate repo, github.com/NVIDIA/physicsnemo-cfd, WebFetch 2026-09-05)
- Purpose: "integrate pretrained AI models into engineering and CFD workflows" — NIM-microservice inference, ML-model accuracy benchmarking against engineering metrics, hybrid CFD initialization (trained model + potential-flow solution), and analysis/visualization utilities.
- Install: `pip install .` from the repo, or the PhysicsNeMo Docker container. License: Apache-2.0. Explicitly labeled experimental/version-0 with expected breaking changes.

**Examples folder** (`examples/`, confirmed by `find`, 62 second-level example subdirectories)
- Top-level domain folders: `active_learning/`, `additive_manufacturing/`, `cfd/`, `generative/`, `geophysics/`, `healthcare/`, `kinetic_monte_carlo/`, `minimal/`, `molecular_dynamics/`, `multi_storage_client/`, `nuclear_engineering/`, `reservoir_simulation/`, `structural_mechanics/`, `weather/`.
- Named examples surfaced from the README's feature table alone (each links to a real subfolder): `cfd/external_aerodynamics/active_learning_aero`, `cfd/external_aerodynamics/unified_external_aero_recipe`, `weather/stormcast`, `cfd/underfill_dispensing`, `structural_mechanics/crash`, `geophysics/diffusion_fwi`, `healthcare/bloodflow_1d_mgn`, `cfd/datacenter`, `additive_manufacturing/sintering_physics`, `cfd/darcy_fno`, `cfd/darcy_transolver`, `weather/unified_recipe`, `weather/graphcast`, `weather/pangu_weather`, `weather/dlwp`, `weather/dlwp_healpix`, `generative/topodiff`, `cfd/ldc_pinns`, `cfd/darcy_physics_informed`, `cfd/mhd_pino`, `structural_mechanics/deforming_plate`, `reservoir_simulation`, `healthcare/brain_anomaly_detection`, `cfd/external_aerodynamics/figconvnet`, `cfd/external_aerodynamics/aerojepa`, `examples/README.md` is described as the "complete example catalog" — **UNVERIFIED** full list beyond what's in the README, since that catalog file itself was not read line-by-line.

**Docs / tutorials** (WebFetch of https://docs.nvidia.com/physicsnemo/latest/, 2026-09-05)
- Sections: Getting Started (system requirements, installation, dependency graph); User Guide (training recipe walkthrough, domain parallelism/`ShardTensor`, physics-guided integration, performance profiling, PINNs, uncertainty quantification, active learning, LoRA fine-tuning); Examples (by domain, matching the folder list above); Library/API reference (models, mesh ops, NN layers, datapipes, diffusion, distributed, metrics, optim); plus a customization guide, a DGL→PyTorch-Geometric migration doc, and release notes.
- The WebFetch summary explicitly found **no dedicated "course" branded material**: "No formal courses or dedicated training materials are explicitly mentioned in this documentation index." Treat PhysicsNeMo's own "User Guide → Training Recipe" page as the closest thing to a tutorial, not a course.

**Directly reusable as course exercises**
- `examples/cfd/darcy_fno` gives a second, NVIDIA-maintained FNO-on-Darcy exercise directly comparable to neuraloperator's `plot_FNO_darcy.py` — good for a "same problem, two frameworks" lesson.
- `examples/cfd/ldc_pinns` and `examples/cfd/darcy_physics_informed` are turnkey, modern (TF2-free, PyTorch) PINN exercises — a strong substitute for the archived `maziarraissi/PINNs` TF1 code in item 7 below.
- The MeshGraphNet examples (`structural_mechanics/crash`, `healthcare/bloodflow_1d_mgn`, `additive_manufacturing/sintering_physics`) are the recommended modern path into mesh-based GNN surrogates in place of the unmaintained TF1 `deepmind-research/meshgraphnets` pipeline.
- Caution for course planning: PhysicsNeMo is mid-refactor toward v2.0 (targeted March 2026 per the migration guide) — pin a specific installed version/commit before building course material on top of it, since import paths are actively moving (`physicsnemo.launch` removal, `physicsnemo.core`/`physicsnemo.nn` reorg).

---

## 7. maziarraissi/PINNs

**Repo facts**
- Last commit (shallow clone HEAD): 2026-02-11, message "sasas" (`git log -1` — note the unusual/likely-typo commit message, quoted verbatim as found).
- Stars: 6.1k, forks: 1.6k (GitHub page, fetched 2026-09-05).
- License: MIT (`LICENSE`).

**Maintenance statement** (`README.md`, verbatim)
> "**Notice:** This repository is no longer under active maintenance. It is highly recommended to utilize implementations of Physics-Informed Neural Networks (PINNs) available in [PyTorch](https://github.com/rezaakb/pinns-torch), [JAX](https://github.com/rezaakb/pinns-jax), and [TensorFlow v2](https://github.com/rezaakb/pinns-tf2)."

**Structure** (confirmed by `find`)
- `main/` — the core paper examples: `continuous_time_identification (Navier-Stokes)/`, `continuous_time_inference (Schrodinger)/`, `discrete_time_identification (KdV)/`, `discrete_time_inference (AC)/` (Allen-Cahn), each with its own `figures/` subfolder, plus a shared `Data/` directory.
- `appendix/` — supplementary Burgers-equation studies: `continuous_time_identification (Burgers)/`, `continuous_time_inference (Burgers)/`, `discrete_time_identification (Burgers)/`, `discrete_time_inference (Burgers)/`, each with `figures/` and `tables/`, plus a shared `Data/`.
- `Utilities/`, `docs/`.
- Two papers underpin the repo: Raissi/Perdikaris/Karniadakis, *J. Comp. Phys.* 378 (2019): 686–707 (the main PINNs paper), and the two-part "Physics Informed Deep Learning" arXiv preprints (Part I: 1711.10561, Part II: 1711.10566) — citations given in full in `README.md`.

**TF1 dependency confirmed empirically**
- `main/continuous_time_identification (Navier-Stokes)/NavierStokes.py` line 8: `import tensorflow as tf`; line 52: `self.sess = tf.Session(...)`; lines 55–60: `tf.placeholder(...)` calls for `x_tf`, `y_tf`, `t_tf`, `u_tf`, `v_tf`. `tf.Session`/`tf.placeholder` are TF1-only APIs (removed/moved in TF2's default eager mode) — this is genuinely graph-mode TF1 code, not merely old-style TF2. Same import pattern found in `Schrodinger.py`, `KdV.py`, `AC.py`.

**Modern ports** (linked directly by the maintenance notice, not independently vetted here)
- PyTorch: `rezaakb/pinns-torch`.
- JAX: `rezaakb/pinns-jax`.
- TensorFlow v2: `rezaakb/pinns-tf2`.
- (All three are by the same author, `rezaakb`, per the linked URLs — **UNVERIFIED** star counts/activity for these three, not fetched in this pass.)

**Directly reusable as course exercises**
- Best used as a historical/reference read (the original continuous-time vs. discrete-time PINN formulations, and the Navier-Stokes/Schrödinger/KdV/Allen-Cahn problem set are the field's canonical benchmark problems) rather than as runnable code — running it as-is requires a legacy TF1 environment. For a hands-on exercise, reimplement one of the four `main/` problems (Schrödinger inference is a compact, well-scoped choice) against `rezaakb/pinns-torch` or PhysicsNeMo's `physicsnemo.sym` PINN path instead.

---

## 8. Briefly surveyed (README only, via WebFetch)

| Repo | What it is | Notable facts (all from README/GitHub page, WebFetch 2026-09-05) |
|---|---|---|
| **lululxvi/deepxde** | PINN + DeepONet library | Backends: TensorFlow 1.x/2.x, PyTorch ≥2.0.0, JAX, PaddlePaddle ≥2.6.0. License LGPL-2.1. 4.4k stars, 992 forks, 282 open issues — actively maintained. Features: multiple BC types (Dirichlet/Neumann/Robin/periodic), CSG-based complex geometries, 3 autodiff methods, adaptive/multiple sampling, multi-GPU data-parallel training, checkpointing, uncertainty quantification, float16/32/64 support. |
| **pdearena/pdearena** | Research code for "Towards multi-spatiotemporal-scale generalized PDE modeling" | Targets Navier-Stokes, Shallow-Water, and 3D Maxwell data. Model zoo: Clifford neural layers, Geometric Clifford Algebra Networks (CGAN), FNO layers, CGAN-UNet. MIT license. 311 stars, 40 forks, 65 commits — modest activity. |
| **PolymathicAI/the_well** | Large physics-simulation dataset collection ("the Well") | 15 TB total across 16 datasets (individual dataset sizes 6.9 GB–5.1 TB), spanning biological systems, fluid dynamics, acoustic scattering, and astrophysical MHD. Built by Flatiron Institute, CU Boulder, Los Alamos National Lab, and collaborators. Download via `the-well-download` (PyPI package) or streamed from Hugging Face. BSD-3-Clause license. 4.4k stars, 563 forks. Directly integrated into neuraloperator via `neuralop/data/datasets/the_well_dataset.py` (see item 1). |
| **PolymathicAI/multiple_physics_pretraining** | "Multiple Physics Pretraining" (MPP): jointly embeds multiple physics/dynamics into one prediction space | Architecture: AViT (Attention Vision Transformer), sizes Ti/S/B/L, with separate spatial/temporal/mixed space-time attention modules. Data: 2D compressible + incompressible Navier-Stokes, HDF5-backed via a `BaseHDF5DirectoryDataset` API, tensor layout (Batch, Time, Channel, H, W). Reports transfer even between incompressible and compressible regimes. MIT license, 218 stars, 20 commits, pretrained weights via a linked Google Drive. |
| **camlab-ethz/poseidon** | "Poseidon" foundation models for PDEs | Pretrained sizes: Tiny/Base/Large; core model class `ScOT`. 20+ pretraining/downstream PDE datasets (compressible/incompressible fluids, wave equations, reaction-diffusion, elliptic problems) hosted on Hugging Face Hub in two collections. Install: `pip install -e .`. Training scripts integrate with Weights & Biases. 196 stars, 37 forks. Companion site: camlab-ethz.github.io/poseidon/. |
| **tum-pbs/apebench** | APEBench: benchmark for autoregressive PDE emulators | JAX-based; built on three companion libraries — Exponax (ETDRK spectral numerical schemes, the same tool jax-cfd's own README points to as a successor), PDEquinox (ConvNet/ResNet/U-Net/FNO architectures), Trainax (training infra). Covers 46+ PDEs across 1D/2D/3D (advection, diffusion, Burgers, Kuramoto-Sivashinsky, Navier-Stokes, periodic domains). Supports "differentiable physics" (reference simulator embedded in emulator training), procedural spectral data generation, and an interactive Streamlit explorer. Install: `pip install apebench` (needs Python 3.10+, JAX 0.4.12+). MIT license, 109 stars, NeurIPS 2024 paper. |

---

## Cross-cutting notes for course design

- **Live, actively-developed, PyTorch-first:** neuraloperator (item 1) and NVIDIA PhysicsNeMo (item 6) — both had commits within days of this survey (2026-08-06 and 2026-09-03 respectively). These are the safest long-term bases for course material.
- **Explicitly unmaintained, with named successors:** jax-cfd (→ JAX-Fluids / PhiFlow / Exponax), maziarraissi/PINNs (→ pinns-torch / pinns-jax / pinns-tf2), and PhysicsNeMo-Sym (→ folded into `physicsnemo.sym`). Each has a verbatim maintenance notice quoted above — use these repos for historical/reference reading, and point students to the named successor for anything hands-on.
- **TF1-locked legacy code exists in two places**: `deepmind-research/meshgraphnets` (`tensorflow-gpu>=1.15,<2`, confirmed via `tensorflow.compat.v1` import) and `maziarraissi/PINNs` (confirmed via `tf.Session`/`tf.placeholder`). Both are excellent for reading the original architecture/loss design but need either a legacy environment or a modern reimplementation exercise (PhysicsNeMo's MeshGraphNet and PINN paths are the natural modern substitutes for each, respectively).
- **Data scale planning**: PDEBench's per-PDE sizes (4 GB–2.3 TB, table in item 2) and the Well's per-dataset sizes (6.9 GB–5.1 TB, item 8) should directly drive which subsets a course can realistically have students download; neuraloperator's own bundled mini-datasets (Darcy-16/32, mini-Burgers, mini-car-CFD — all under `neuralop/data/datasets/data/`, confirmed present in the clone) require no download at all and are the right "day one" data.
- **FEniCSx/DOLFINx is the odd one out**: it's a FEM solver, not a learning library, and its role in this course is almost certainly "generate your own labeled PDE dataset" rather than "train a model" — pair it conceptually with PDEBench's `data_gen`/`data_gen_NLE` scripts, which do the analogous job for PDEBench's specific PDEs.
