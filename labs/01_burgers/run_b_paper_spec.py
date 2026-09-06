"""Lab 01, run B. FNO on 1-D viscous Burgers, data generated here to the FNO paper spec.

Spec (JMLR paper 21-1524.pdf p.32, Section 6.3, eq. 41; ICLR 2010.08895v3.pdf Section 5.1, App. A.3.1):
  du/dt + d(u^2/2)/dx = nu d2u/dx2,  x in (0, 2 pi) periodic,  t in (0,1],  nu = 0.1.
  u0 ~ N(0, 625 (-d2/dx2 + 25 I)^-2).  Learn u0 -> u(., 1).
  Domain note: the ICLR text and research_notes/pdf-fno-jmlr.md:93 write x in (0,1). The JMLR text
  writes (0, 2 pi). On (0,1) with nu = 0.1 the k=1 mode decays by exp(-0.1 (2 pi)^2) = 0.019 in one
  time unit and u(., 1) is nearly constant, which cannot give the paper's Table 3 errors. On (0, 2 pi)
  the k=1 mode decays by exp(-0.1) = 0.905 and fronts form. This lab uses (0, 2 pi).
  Paper: 1000 train / 200 test, k_max = 16, width 64, 4 layers, 500 epochs, Adam lr 1e-3 halved
  every 100 epochs. Paper solves on 8192 points; this lab solves on 1024 (enough at nu = 0.1,
  checked by the time-step assert below) and subsamples.
Train at 64 grid points. Test at 64, 128, 256, 1024 without retraining (zero-shot).
Outputs (TAG = _m{n_modes}): results_b{TAG}.json, spectrum_b{TAG}.npz, fig_b_spectrum{TAG}.png,
  fig_b_samples{TAG}.png, model_b{TAG}.pt, and burgers_nu0.1.pt (shared data)
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO
from neuralop.utils import count_model_params

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

N_FINE = 1024
NU = 0.1
T_END = 1.0
DT = 1e-4
N_TRAIN, N_TEST = 1000, 200
TRAIN_RES = 64
TEST_RES = [64, 128, 256, 1024]
# neuraloperator 2.0.0 keeps n_modes // 2 + 1 bins on the real FFT axis (spectral_convolution.py:399),
# so n_modes = 16 keeps k = 0..8 and n_modes = 32 keeps k = 0..16. The paper's k_max = 16 needs 32.
N_MODES = int(sys.argv[1]) if len(sys.argv) > 1 else 16
K_MAX = N_MODES // 2                      # highest wave number the spectral branch can touch
WIDTH, N_LAYERS, EPOCHS = 64, 4, 500
TAG = f"_m{N_MODES}"


def sample_u0(n, N=N_FINE, gen=None):
    """u0 ~ N(0, 625(-d2/dx2 + 25 I)^-2) on the periodic interval (0, 2 pi), by Karhunen-Loeve expansion.
    Orthonormal eigenfunctions of -d2/dx2: 1/sqrt(2 pi), cos(kx)/sqrt(pi), sin(kx)/sqrt(pi), eigenvalue k^2.
    Covariance eigenvalue for mode k: 625 / (k^2 + 25)^2, so the coefficient std is 25 / (k^2 + 25).
    u0 = std_0 xi_0 / sqrt(2 pi) + sum_k std_k (xi_k cos kx + eta_k sin kx) / sqrt(pi)."""
    k = torch.arange(N // 2 + 1, dtype=torch.float64)
    std = 25.0 / (k ** 2 + 25.0)
    xi = torch.randn(n, N // 2 + 1, dtype=torch.float64, generator=gen)
    eta = torch.randn(n, N // 2 + 1, dtype=torch.float64, generator=gen)
    # rfft convention: u = sum_k c_k e^{i k x}; a real field a cos + b sin has c_k = (a - i b)/2 for k>0.
    # irfft(c, n=N, norm="forward") returns sum_k c_k e^{i k x_j} with hermitian fill-in.
    coef = torch.zeros(n, N // 2 + 1, dtype=torch.complex128)
    coef[:, 0] = std[0] * xi[:, 0] / math.sqrt(2 * math.pi)
    coef[:, 1:] = std[1:] * (xi[:, 1:] - 1j * eta[:, 1:]) / (2 * math.sqrt(math.pi))
    return torch.fft.irfft(coef, n=N, norm="forward")


def solve_burgers(u0, nu=NU, dt=DT, t_end=T_END):
    """Pseudo-spectral solver. Diffusion exact through an integrating factor; the advection term
    -d(u^2/2)/dx is stepped with RK4 and 2/3-rule dealiasing. Domain (0, 2 pi), so wave number k = j.
    u0: (n, N) float64."""
    n, N = u0.shape
    k = torch.arange(N // 2 + 1, dtype=u0.dtype, device=u0.device)  # integer wave numbers, float64
    dealias = (torch.arange(N // 2 + 1, device=u0.device) < N // 3).to(u0.dtype)
    E = torch.exp(-nu * k ** 2 * dt / 2)            # half-step integrating factor
    uh = torch.fft.rfft(u0)

    def N_of(uh):  # Fourier transform of -d(u^2/2)/dx
        u = torch.fft.irfft(uh * dealias, n=N)
        return -0.5j * k * torch.fft.rfft(u * u) * dealias

    steps = int(round(t_end / dt))
    for _ in range(steps):
        a = N_of(uh)
        b = N_of(E * (uh + dt / 2 * a))
        c = N_of(E * uh + dt / 2 * b)
        d = N_of(E * E * uh + dt * E * c)
        uh = E * E * uh + dt / 6 * (E * E * a + 2 * E * (b + c) + d)
    return torch.fft.irfft(uh, n=N)


# ---- self-check on the solver: halve dt, compare; and check the L2 norm can only shrink ----
gen = torch.Generator().manual_seed(123)
u0c = sample_u0(4, gen=gen).to(device)
u1 = solve_burgers(u0c, dt=DT); u2 = solve_burgers(u0c, dt=DT / 2)
dt_err = ((u1 - u2).norm() / u2.norm()).item()
assert dt_err < 1e-6, f"time-step check failed: rel diff {dt_err}"
assert (u1.norm(dim=1) < u0c.norm(dim=1)).all(), "L2 norm grew; periodic Burgers must dissipate"
print(f"solver check: rel diff dt vs dt/2 = {dt_err:.2e}; norm ratio {(u1.norm()/u0c.norm()).item():.3f}")

# ---- data ----
data_file = OUT / "burgers_nu0.1.pt"
if data_file.exists():
    D = torch.load(data_file); U0, U1 = D["u0"], D["u1"]
else:
    t0 = time.time()
    U0 = sample_u0(N_TRAIN + N_TEST, gen=torch.Generator().manual_seed(SEED)).to(device)
    U1 = solve_burgers(U0)
    U0, U1 = U0.cpu(), U1.cpu()
    torch.save(dict(u0=U0, u1=U1, nu=NU, t_end=T_END, n_fine=N_FINE, dt=DT, seed=SEED), data_file)
    print(f"generated {N_TRAIN + N_TEST} samples on {N_FINE} points in {time.time() - t0:.0f}s")


def at_res(U, r):
    return U[:, :: N_FINE // r].float()


x_tr, y_tr = at_res(U0[:N_TRAIN], TRAIN_RES).to(device), at_res(U1[:N_TRAIN], TRAIN_RES).to(device)
tests = {r: (at_res(U0[N_TRAIN:], r).to(device), at_res(U1[N_TRAIN:], r).to(device)) for r in TEST_RES}

# ---- model and plain training loop ----
model = FNO(n_modes=(N_MODES,), in_channels=1, out_channels=1, hidden_channels=WIDTH,
            n_layers=N_LAYERS).to(device)
n_params = count_model_params(model)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)


def rel_l2(pred, true):
    return ((pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1))


@torch.no_grad()
def evaluate(x, y):
    model.eval()
    return rel_l2(model(x.unsqueeze(1)).squeeze(1), y).mean().item()


BATCH = 20
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(N_TRAIN, device=device); tot = 0.0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i + BATCH]
        out = model(x_tr[idx].unsqueeze(1)).squeeze(1)
        loss = rel_l2(out, y_tr[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    sched.step()
    if ep % 50 == 0 or ep == EPOCHS - 1:
        print(f"[{ep}] train {tot / N_TRAIN:.4f}  test64 {evaluate(*tests[64]):.4f}  "
              f"test256 {evaluate(*tests[256]):.4f}  {time.time() - t0:.0f}s")
train_seconds = time.time() - t0

errors = {r: evaluate(*tests[r]) for r in TEST_RES}
train_err = evaluate(x_tr, y_tr)
print("rel L2 by test resolution:", {r: round(e, 4) for r, e in errors.items()}, "train", round(train_err, 4))

# ---- output spectrum: mean |u_hat_k| of truth and prediction at each test resolution ----
spec = {}
with torch.no_grad():
    model.eval()
    for r, (x, y) in tests.items():
        p = model(x.unsqueeze(1)).squeeze(1)
        spec[f"truth_{r}"] = torch.fft.rfft(y, norm="forward").abs().mean(0).cpu().numpy()
        spec[f"pred_{r}"] = torch.fft.rfft(p, norm="forward").abs().mean(0).cpu().numpy()
        spec[f"input_{r}"] = torch.fft.rfft(x, norm="forward").abs().mean(0).cpu().numpy()
np.savez(OUT / f"spectrum_b{TAG}.npz", **spec)
torch.save(model.state_dict(), OUT / f"model_b{TAG}.pt")

results = dict(run="B_paper_spec", pde="du/dt + d(u^2/2)/dx = nu d2u/dx2, periodic (0, 2 pi), t=1",
               nu=NU, u0_law="N(0, 625(-d2/dx2+25I)^-2) on (0, 2 pi)", solver="pseudo-spectral, RK4 + integrating factor",
               n_fine=N_FINE, dt=DT, solver_dt_check_rel_diff=dt_err,
               n_train=N_TRAIN, n_test=N_TEST, train_res=TRAIN_RES, test_res=TEST_RES,
               n_modes_arg=N_MODES, k_max_effective=K_MAX, width=WIDTH, n_layers=N_LAYERS, epochs=EPOCHS, batch=BATCH,
               optimizer="Adam lr 1e-3, halved every 100 epochs", loss="relative L2",
               params=n_params, seed=SEED, device=device,
               gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
               train_rel_l2=train_err, test_rel_l2={str(r): e for r, e in errors.items()},
               train_seconds=train_seconds, torch=torch.__version__,
               paper_reference="FNO at s=256: 0.0149 (pdf-fno-jmlr.md:104), nu=0.1, 8192-point solve")
(OUT / f"results_b{TAG}.json").write_text(json.dumps(results, indent=2))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(TEST_RES), figsize=(4 * len(TEST_RES), 3.6), sharey=True)
for a, r in zip(ax, TEST_RES):
    kk = np.arange(r // 2 + 1)
    a.semilogy(kk, spec[f"input_{r}"], color="gray", lw=1, label="input u0")
    a.semilogy(kk, spec[f"truth_{r}"], "k", label="truth u(1)")
    a.semilogy(kk, spec[f"pred_{r}"], "C1", ls="--", label="FNO")
    a.axvline(K_MAX, color="C0", ls=":", label=f"k_max={K_MAX} (n_modes={N_MODES})")
    a.set_title(f"test res {r}, rel L2 {errors[r]:.4f}"); a.set_xlabel("wave number k"); a.set_xlim(0, min(r // 2, 64))
    a.set_ylim(1e-8, 1)
ax[0].set_ylabel("mean |u_hat_k|"); ax[0].legend(fontsize=8)
fig.suptitle(f"Burgers nu={NU}: output spectrum, trained at {TRAIN_RES} points"); fig.tight_layout()
fig.savefig(OUT / f"fig_b_spectrum{TAG}.png", dpi=120)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
with torch.no_grad():
    for i, a in enumerate(ax):
        x, y = tests[1024]; p = model(x[i:i + 1].unsqueeze(1)).squeeze().cpu()
        xs = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
        a.plot(xs, x[i].cpu(), color="gray", lw=1, label="u0"); a.plot(xs, y[i].cpu(), "k", label="u(1) truth")
        a.plot(xs, p, "C1", ls="--", label="FNO at 1024"); a.set_xlabel("x")
ax[0].legend(fontsize=8); fig.suptitle("three test samples, evaluated at 1024 points (trained at 64)")
fig.tight_layout(); fig.savefig(OUT / f"fig_b_samples{TAG}.png", dpi=120)
print(f"wrote results_b{TAG}.json, spectrum_b{TAG}.npz, model_b{TAG}.pt, figures")
