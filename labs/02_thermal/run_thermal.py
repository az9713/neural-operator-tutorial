"""Lab 02. FNO surrogate for steady die-thermal conduction, data generated here.

No public neural-operator semiconductor benchmark exists (research_notes/lineage.md:329), so the data
is generated locally. Every parameter below is a modelling choice made for this course, not a value
taken from a paper.

Model: a thin silicon die as a 2-D plate.  -k t (d2T/dx2 + d2T/dy2) = q(x, y)  on (0, L)^2,
  T = 0 on the four edges (edge held at package temperature; T is the rise above it).
  L = 10 mm, k = 130 W/(m K) (silicon near 100 C), t = 0.5 mm die thickness,
  q = heat flux density in W/m^2, built from 2 to 5 random rectangular power blocks ("cores"),
  each with a flux between 0.2e6 and 2.0e6 W/m^2. Total die power lands in the tens of watts.
Solver: 5-point finite differences on a 256 x 256 grid of cell centres, scipy sparse LU, one
  factorisation reused for all right-hand sides. Checked against the analytic solution for
  q = sin(pi x/L) sin(pi y/L):  T = q L^2 / (2 pi^2 k t).
Learn q -> T. Train at 64 x 64, test at 64 x 64 and 128 x 128 (zero-shot).
Outputs: results_thermal.json, model_thermal.pt, thermal_data.pt, fig_thermal_samples.png, fig_thermal_hotspot.png
"""
import json, math, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from neuralop.models import FNO
from neuralop.utils import count_model_params

OUT = Path(__file__).resolve().parent
SEED = 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

L, K_SI, T_DIE = 10e-3, 130.0, 0.5e-3         # m, W/(m K), m
N_FINE = 256
N_TRAIN, N_TEST = 1000, 200
TRAIN_RES, TEST_RES = 64, [64, 128]
K_MAX, WIDTH, N_LAYERS, EPOCHS, BATCH = 16, 32, 4, 200, 20
Q_SCALE, T_SCALE = 1e6, 100.0                  # W/m^2 and K, for normalisation only


def laplacian_dirichlet(n, h):
    """Matrix of -(d2/dx2 + d2/dy2) on an n x n grid of cell centres with T = 0 on the boundary
    (ghost-cell form: the boundary lies half a cell outside the first centre)."""
    main = np.full(n, 2.0); off = np.full(n - 1, -1.0)
    D = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    D = D.tolil(); D[0, 0] = 3.0; D[-1, -1] = 3.0; D = D.tocsr()   # ghost cell T_ghost = -T_0
    I = sp.identity(n, format="csr")
    return (sp.kron(I, D) + sp.kron(D, I)) / h ** 2


h_fine = L / N_FINE
A = laplacian_dirichlet(N_FINE, h_fine) * (K_SI * T_DIE)
lu = spla.splu(A.tocsc())
xc = (np.arange(N_FINE) + 0.5) * h_fine
X, Y = np.meshgrid(xc, xc, indexing="ij")

# ---- solver self-check against the analytic solution ----
q_test = 1e6 * np.sin(math.pi * X / L) * np.sin(math.pi * Y / L)
T_num = lu.solve(q_test.ravel()).reshape(N_FINE, N_FINE)
T_exact = q_test * L ** 2 / (2 * math.pi ** 2 * K_SI * T_DIE)
fd_err = np.linalg.norm(T_num - T_exact) / np.linalg.norm(T_exact)
assert fd_err < 1e-3, f"finite-difference check failed: rel err {fd_err}"
print(f"solver check: rel err vs analytic {fd_err:.2e}; peak rise {T_exact.max():.1f} K for 1e6 W/m^2 sine load")


def random_power_map(rng):
    """2 to 5 rectangles, sides 1 to 4 mm, each with its own flux. Rectangles may overlap (fluxes add)."""
    q = np.zeros((N_FINE, N_FINE))
    for _ in range(rng.integers(2, 6)):
        w, hgt = rng.uniform(1e-3, 4e-3, size=2)
        x0, y0 = rng.uniform(0, L - w), rng.uniform(0, L - hgt)
        flux = rng.uniform(0.2e6, 2.0e6)
        q[(X >= x0) & (X < x0 + w) & (Y >= y0) & (Y < y0 + hgt)] += flux
    return q


data_file = OUT / "thermal_data.pt"
if data_file.exists():
    D = torch.load(data_file); Q, T = D["q"], D["T"]
else:
    t0 = time.time()
    Qs, Ts = [], []
    for _ in range(N_TRAIN + N_TEST):
        q = random_power_map(rng); Qs.append(q); Ts.append(lu.solve(q.ravel()).reshape(N_FINE, N_FINE))
    Q, T = torch.tensor(np.array(Qs), dtype=torch.float32), torch.tensor(np.array(Ts), dtype=torch.float32)
    torch.save(dict(q=Q, T=T, L=L, k=K_SI, t=T_DIE, n_fine=N_FINE, seed=SEED), data_file)
    print(f"generated {N_TRAIN + N_TEST} samples on {N_FINE}^2 in {time.time() - t0:.0f}s; "
          f"peak rise range {T.amax(dim=(1, 2)).min():.1f} to {T.amax(dim=(1, 2)).max():.1f} K; "
          f"die power range {(Q.sum(dim=(1, 2)) * h_fine ** 2).min():.1f} to {(Q.sum(dim=(1, 2)) * h_fine ** 2).max():.1f} W")


def at_res(U, r):
    s = N_FINE // r
    # block-average so that a rectangle edge between cells is represented by its area fraction
    return U.reshape(U.shape[0], r, s, r, s).mean(dim=(2, 4))


x_tr = (at_res(Q[:N_TRAIN], TRAIN_RES) / Q_SCALE).to(device)
y_tr = (at_res(T[:N_TRAIN], TRAIN_RES) / T_SCALE).to(device)
tests = {r: ((at_res(Q[N_TRAIN:], r) / Q_SCALE).to(device), (at_res(T[N_TRAIN:], r) / T_SCALE).to(device))
         for r in TEST_RES}

model = FNO(n_modes=(K_MAX, K_MAX), in_channels=1, out_channels=1, hidden_channels=WIDTH,
            n_layers=N_LAYERS).to(device)
n_params = count_model_params(model)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)


def rel_l2(pred, true):
    return (pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)


@torch.no_grad()
def predict(x, chunk=20):
    # ponytail: chunked so a 200 x 128^2 test set fits a 4 GB GPU
    model.eval()
    return torch.cat([model(x[i:i + chunk].unsqueeze(1)).squeeze(1) for i in range(0, len(x), chunk)])


t0 = time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(N_TRAIN, device=device); tot = 0.0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i + BATCH]
        loss = rel_l2(model(x_tr[idx].unsqueeze(1)).squeeze(1), y_tr[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    sched.step()
    if ep % 25 == 0 or ep == EPOCHS - 1:
        print(f"[{ep}] train {tot / N_TRAIN:.4f}  test64 {rel_l2(predict(tests[64][0]), tests[64][1]).mean():.4f}  "
              f"test128 {rel_l2(predict(tests[128][0]), tests[128][1]).mean():.4f}  {time.time() - t0:.0f}s")
train_seconds = time.time() - t0

# ---- metrics: field error, and the engineering quantities: peak rise and hot-spot location ----
metrics = {}
for r, (x, y) in tests.items():
    p = predict(x)
    field = rel_l2(p, y).mean().item()
    peak_true = y.flatten(1).max(dim=1).values * T_SCALE; peak_pred = p.flatten(1).max(dim=1).values * T_SCALE
    peak_abs_err = (peak_pred - peak_true).abs()
    it, ip = y.flatten(1).argmax(dim=1), p.flatten(1).argmax(dim=1)
    dist = torch.sqrt(((it // r - ip // r) ** 2 + (it % r - ip % r) ** 2).float()) * (L / r)
    metrics[str(r)] = dict(rel_l2=field, peak_abs_err_K_mean=peak_abs_err.mean().item(),
                           peak_abs_err_K_max=peak_abs_err.max().item(),
                           hotspot_dist_mm_mean=(dist.mean() * 1e3).item(),
                           hotspot_within_0p5mm_frac=(dist <= 0.5e-3).float().mean().item(),
                           peak_true_K_mean=peak_true.mean().item())
print(json.dumps(metrics, indent=1))

results = dict(run="thermal", model="-k t Lap T = q, T=0 on edges, L=10mm, k=130, t=0.5mm",
               q_law="2-5 rectangles, 1-4 mm sides, flux U(0.2e6, 2e6) W/m^2, may overlap",
               solver="5-point FD, 256^2 cell centres, sparse LU", solver_check_rel_err=fd_err,
               n_train=N_TRAIN, n_test=N_TEST, train_res=TRAIN_RES, test_res=TEST_RES,
               k_max=K_MAX, width=WIDTH, n_layers=N_LAYERS, epochs=EPOCHS, batch=BATCH,
               optimizer="Adam lr 1e-3, halved every 50 epochs", loss="relative L2",
               normalisation=dict(q_div=Q_SCALE, T_div=T_SCALE), params=n_params, seed=SEED,
               device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
               train_rel_l2=rel_l2(predict(x_tr), y_tr).mean().item(), test=metrics,
               train_seconds=train_seconds, torch=torch.__version__)
(OUT / "results_thermal.json").write_text(json.dumps(results, indent=2))
torch.save(model.state_dict(), OUT / "model_thermal.pt")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
x, y = tests[128]; p = predict(x)
fig, axes = plt.subplots(3, 4, figsize=(13, 9))
for i in range(3):
    err = (p[i] - y[i]).abs() * T_SCALE
    for j, (img, title, cm) in enumerate([(x[i] * Q_SCALE / 1e6, "power q (MW/m^2)", "magma"),
                                          (y[i] * T_SCALE, "T rise, FD truth (K)", "inferno"),
                                          (p[i] * T_SCALE, "T rise, FNO (K)", "inferno"),
                                          (err, "|error| (K)", "viridis")]):
        im = axes[i, j].imshow(img.cpu().T, origin="lower", cmap=cm, extent=[0, 10, 0, 10])
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([]); plt.colorbar(im, ax=axes[i, j], fraction=0.046)
        if i == 0: axes[i, j].set_title(title)
fig.suptitle(f"die thermal, test at 128x128 (trained at 64x64), rel L2 {metrics['128']['rel_l2']:.4f}")
fig.tight_layout(); fig.savefig(OUT / "fig_thermal_samples.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
pt = (y.flatten(1).max(dim=1).values * T_SCALE).cpu(); pp = (p.flatten(1).max(dim=1).values * T_SCALE).cpu()
ax[0].scatter(pt, pp, s=8); lim = [0, max(pt.max(), pp.max()) * 1.05]; ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("peak rise, truth (K)"); ax[0].set_ylabel("peak rise, FNO (K)"); ax[0].set_title("hot-spot temperature")
ax[1].hist((pp - pt).numpy(), bins=30); ax[1].set_xlabel("peak error, FNO - truth (K)"); ax[1].set_title("peak error, 200 test dies")
fig.tight_layout(); fig.savefig(OUT / "fig_thermal_hotspot.png", dpi=110)
print("wrote results_thermal.json, fig_thermal_samples.png, fig_thermal_hotspot.png")
