# Neural operators: from the convolution theorem to a physics foundation model

A 13 module course, plus a capstone, on neural operators. Thirteen HTML lessons, six Jupyter
notebooks, and three labs that ran on one laptop GPU. Every number on every page traces to a
research note, a page of a primary paper, or a lab result file in this repository.

**Read the course: https://az9713.github.io/neural-operator-tutorial/course/index.html**

## Inspiration

This course started from one podcast episode:

[**Trillion Token Context. No, Really — Anima Anandkumar & Benedikt Jenik, Accelerated
Understanding**](https://www.youtube.com/watch?v=KS_IpnX7n9I) — Latent Space, 4 September 2026.

In the episode the founders claim that one model can learn across fluid dynamics,
semiconductors, and energy, and that a model trained across several areas of physics beats an
equally sized model trained on one. They describe trillion context training and five trillion
context inference. The course exists to answer one question: which of those claims rest on
public evidence, and which do not. Module 9 checks the scale arithmetic. Module 10 states a
transfer experiment that can fail. Module 12 grades every company claim. The episode transcript
is in [`transcript.txt`](transcript.txt); an earlier episode with the same guest is in
[`research_notes/earlier_episode_transcript.txt`](research_notes/earlier_episode_transcript.txt).

## How the course is designed

The episode above is a sales pitch with real science inside it. The design problem was to keep
the science and test the pitch. Ten decisions follow from that, and they shape every page.

**1. Three objectives, never mixed.** The podcasts use one word, "neural operator", for three
different goals. A *surrogate* reproduces a known simulator faster; its test is error against
the simulator on held out inputs, plus speed. A *PINN* fits one solution by penalizing the
equation residual; its test is the residual and the error on that one instance. A *foundation
model* reuses one representation across several physical systems; its test is whether
pretraining on systems A and B lowers the data needed for system C, measured against an equal
size model trained on C alone. Success at one does not prove success at another. Module 0 fixes
this vocabulary before anything else. Module 10 runs the third test.

**2. Every number carries a grade.** Each claim on each module page is marked **A** (read in the
primary paper, page cited), **B** (abstract or secondary source only, arXiv identifier
verified), **C** (company statement, press, or podcast, not independently checked), or **LAB**
(produced in this repository, with the script and the results file named). A company claim stays
a C. It is not promoted by repetition.

**3. Tests that can fail.** A course that only shows successes teaches nothing about limits.
Lab 01 run C trains a CNN as a counterexample and it fails as predicted: relative L2 goes 0.0224
→ 0.68 → 1.14 → 1.46 across four grids. Module 8 is titled "evaluation that can fail a model".
Module 10 states a transfer experiment whose outcome is not decided in advance. And the labs
found a result that cuts against the marketing: an FNO at an unseen grid carries a Nyquist
oscillation, so "resolution invariant" describes the architecture, not the error.

**4. Derive, do not assert.** The Fourier layer is built up from the convolution theorem and the
Green's function in Module 2, not presented as a diagram to accept. Module 5 states what the
theorems do *not* give: universal approximation is an existence result, not an efficiency
result, and of the four sources of error the JMLR paper lists on pages 50 to 51, only two are
proved.

**5. One page shape, six parts, same order.** Every module runs: why the module exists → the
core ideas in prose → the mathematics with derivations → the lab with real numbers → the sources
with line citations → an exit test. The exit test is taken with the page closed. If you cannot
pass it, it names the section to reread.

**6. One physical domain, threaded through.** Rather than a new toy problem per module, the
course anchors on one: steady heat conduction in a 10 mm silicon die with random power blocks,
solved locally by finite differences and checked against an analytic solution. That die returns
in Modules 6, 10, 12, 13, and the capstone. A single running example makes transfer measurable.

**7. Labs first, prose second.** The three labs ran before the module pages were written, so the
numbers drove the text rather than the text asking for numbers to match. Where a lab has not run
yet, the page says "not yet run" instead of filling the gap with a plausible figure.

**8. Reproducible on modest hardware.** Everything here ran on an RTX 3050 Laptop GPU with 4 GB.
The scripts, the logs, the seeds, and the `results*.json` files are committed; the notebooks are
generated from the same scripts, so the two cannot drift apart. The one file too large for
GitHub is regenerable by one command.

**9. Two tools only.** PyTorch and `neuraloperator`. No Julia, no JAX, no second framework to
learn alongside the subject.

**10. One student, self-paced.** This is not a lecture series for an audience. It is 16 to 17
weeks at about eight hours a week, and the 90 minute diagnostic at the end of Module 0 sets how
fast you move through Modules 1 to 5. Module 5 takes two or three weeks because it carries the
proofs.

## Live pages

GitHub strips scripts and iframes from a README, so a page cannot render inside this file. Each
link below opens the live page on GitHub Pages. The lessons use MathJax from a CDN, so they need
a network connection.

| # | Page | One sentence | Lab |
|---|------|--------------|-----|
| — | [**Course home**](https://az9713.github.io/neural-operator-tutorial/course/index.html) | The map: how to use the course, the labs, the evidence grades, the schedule. | — |
| 0 | [Orientation](https://az9713.github.io/neural-operator-tutorial/course/00_orientation.html) | What an operator is, what a discretization is, and the 90 minute diagnostic. | none |
| 1 | [Functions in, functions out](https://az9713.github.io/neural-operator-tutorial/course/01_functions_in_functions_out.html) | Discretization invariance as a three part definition, with the CNN counterexample. | one trained FNO at four grids |
| 2 | [The Fourier layer](https://az9713.github.io/neural-operator-tutorial/course/02_fourier_layer.html) | One layer, derived from the Green's function, then trained on Burgers. | Lab 01, Burgers at four resolutions |
| 3 | [Four answers to one cost problem](https://az9713.github.io/neural-operator-tutorial/course/03_four_answers.html) | GNO, LNO, MGNO, FNO as Nyström, low rank, multipole, FFT. | refinability reading |
| 4 | [DeepONet, PINN, hybrids](https://az9713.github.io/neural-operator-tutorial/course/04_deeponet_pinn_hybrids.html) | Encoder, approximator, decoder versus operator layers; constraint versus amortization. | PINN failure at high β (reading) |
| 5 | [What the theorems say](https://az9713.github.io/neural-operator-tutorial/course/05_theorems.html) | Universal approximation is existence, not efficiency; four sources of error. | none, proofs |
| 6 | [Meshes, geometry, spheres, attention](https://az9713.github.io/neural-operator-tutorial/course/06_meshes_geometry.html) | MeshGraphNets, GINO, SFNO, Transolver, and the thermal die on a square. | Lab 02, die thermal |
| 7 | [Time](https://az9713.github.io/neural-operator-tutorial/course/07_time.html) | FNO-2D versus FNO-3D, noise injection, pushforward, and the rollout wall. | Lab 01 run A, space time Burgers |
| 8 | [Evaluation that can fail a model](https://az9713.github.io/neural-operator-tutorial/course/08_evaluation.html) | cRMSE, bRMSE, banded fRMSE, hot spot error, and split by family. | metrics on Labs 01 and 02 |
| 9 | [Scale](https://az9713.github.io/neural-operator-tutorial/course/09_scale.html) | Where the trillion context numbers come from and what public work reaches. | Lab 03, memory and time versus grid |
| 10 | [Physics foundation models](https://az9713.github.io/neural-operator-tutorial/course/10_foundation_models.html) | MPP, Poseidon, DPOT, Subramanian et al.; the experiment matrix A to D. | transfer matrix (not yet run) |
| 11 | [Physics as data engine and as reward](https://az9713.github.io/neural-operator-tutorial/course/11_physics_as_signal.html) | Curricula, residual losses, self improvement, and the optimization caveat. | residual sweep (not yet run) |
| 12 | [Applications and industry](https://az9713.github.io/neural-operator-tutorial/course/12_applications.html) | Weather, plasma, CO2, catheters, seismic, semiconductors; company claims graded. | evidence ledger |
| 13 | [Capstone](https://az9713.github.io/neural-operator-tutorial/course/13_capstone.html) | Operator transfer under physics shift, with the thermal die as the semiconductor target. | the full matrix |

Two more pages:

- [**Research report**](https://az9713.github.io/neural-operator-tutorial/neural_operator_course_research_report.html) — the findings behind the course, every claim graded, the source ledger.
- [**Physics foundation model study kit**](https://az9713.github.io/neural-operator-tutorial/05_physics_foundation_model_study_kit.html) — the project kit the capstone follows.

## The labs

Three labs ran. Their numbers are on the module pages and in the `results*.json` files.

**Lab 01, Burgers** — [`labs/01_burgers/`](labs/01_burgers/)

- Run A: the bundled 16 point space time data through the `neuraloperator` loader and `Trainer`. Test relative L2 **0.0044**.
- Run B: the ICLR/JMLR paper spec (ν = 0.1 on the domain (0, 2π)) with a spectral solver written and checked in the same file. Trained at 64 points, evaluated at 64, 128, 256, 1024. With `n_modes=16` (k_max 8): **0.0224 / 0.0351 / 0.0349 / 0.0349**. With `n_modes=32` (k_max 16): **0.0226 / 0.0267 / 0.0264 / 0.0264**.
- Run C, the counterexample: a CNN with a 49 point receptive field gives **0.0224 / 0.68 / 1.14 / 1.46** at the same four grids. The CNN is not discretization invariant. The FNO is close to invariant, but not exactly.

**Lab 02, die thermal** — [`labs/02_thermal/`](labs/02_thermal/)

Steady heat conduction in a 10 mm silicon die with random rectangular power blocks. A finite
difference solver checked against an analytic solution. The FNO trained at 64², evaluated at 64²
and 128²: relative L2 **0.0065** and **0.0814**, peak temperature error **0.20 K** and **1.55 K**.

**Lab 03, scale** — [`labs/03_scale/`](labs/03_scale/)

Memory and time of one FNO forward pass against grid size: about **1050 bytes per grid point**,
**143 ms at 1024²** on the GPU below.

### One finding worth stating up front

An FNO evaluated at a grid it never trained on carries a grid scale (Nyquist) oscillation:
1.44e-2 at the Nyquist bin in 1D at every unseen grid, and horizontal stripes in the 2D thermal
output at 128². This is the reason for the error rise above (+56 percent for k_max 8, +18 percent
for k_max 16 on Burgers; 12× on thermal). "Resolution invariant" is a statement about the
architecture, not a promise about the error.

Also: `neuraloperator` 2.0.0 keeps `n_modes // 2 + 1` bins on the real FFT axis, so `n_modes=16`
means k_max 8. The paper's k_max 16 needs `n_modes=32`. Module 2 documents this.

## Notebooks

Six notebooks in [`notebooks/`](notebooks/), generated from the lab scripts by
`notebooks/make_notebooks.py`: `01a_burgers_bundled`, `01b_burgers_paper_spec`,
`01c_cnn_counterexample`, `02_thermal_die`, `03_scale`, `08_metrics`.

## Run it yourself

```bash
pip install neuraloperator torch matplotlib
python labs/01_burgers/run_a_bundled.py        # bundled data, needs data/*.pt
python labs/01_burgers/run_b_paper_spec.py     # about 5 minutes per n_modes value
python labs/01_burgers/run_c_cnn_baseline.py   # about 3 minutes
python labs/02_thermal/run_thermal.py          # about 10 minutes, writes thermal_data.pt
python labs/03_scale/run_scale.py              # under a minute
python labs/01_burgers/metrics.py              # under a minute
```

Numbers in this repository came from an RTX 3050 Laptop GPU (4 GB), torch 2.11.0+cu128,
`neuraloperator` 2.0.0. Runs are not bit reproducible on CUDA: run A gave 0.0040 and 0.0044 on
two runs, and run B `_m16` reproduced to ±0.0002. Loading the saved state dicts needs
`torch.load(..., weights_only=False)`, because the state dict carries the `gelu` callable — load
your own files only.

## What is not in this repository

- **`labs/02_thermal/thermal_data.pt`, 601 MB.** Over the GitHub file size limit. Regenerate it with `python labs/02_thermal/run_thermal.py`, about 10 minutes.
- **The four source PDFs, 28 MB.** They are third party papers. Get them from the source:
  - Fourier Neural Operator, ICLR 2021 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
  - Neural Operator, JMLR 24 (2023) 1-97 — [jmlr.org/papers/v24/21-1524.html](https://www.jmlr.org/papers/v24/21-1524.html)
  - Learning Mesh-Based Simulation with Graph Networks, ICLR 2021 — [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
  - PDEBench, NeurIPS 2022 — [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)
- **Private project notes.** The theory study plan and the session handoff file stay local. Pages that used to cite them now say so.

## What is specified but not yet run

These are written out in full on their module pages, and marked "not yet run" there: the Module 4
PINN wall and residual sweep; the Module 7 rollout with noise injection and pushforward; the
Module 3 refinability curve; the Module 2 exercises (anti-aliased training, W path ablation); the
Module 10 and 13 transfer matrix; the Module 11 residual weight sweep on thermal.

## Layout

```
course/          15 HTML pages: index plus modules 0 to 13, and course.css
labs/            three labs, with scripts, logs, figures, results*.json, model weights
notebooks/       six notebooks and their generator
research_notes/  source notes with page and URL citations; the two episode transcripts
data/            the Burgers tensors from the neuraloperator GitHub repository
```

## License

[MIT](LICENSE), for the original work here: the course pages, the lab code and results, the
notebooks, and the research notes.

Not covered, and listed in [`NOTICE`](NOTICE): the two podcast transcripts, material quoted
inside the pages and notes, the cited papers (not distributed here), and `data/*.pt` from the
neuraloperator project. Those belong to their authors.
