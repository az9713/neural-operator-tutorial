"""Module 8 metrics on the saved Lab 01 and Lab 02 models. PDEBench definitions (Appendix B, p.17;
research_notes/pdf-mgn-pdebench.md:214-219):
  nRMSE = ||pred - true||_2 / ||true||_2
  max error = max |pred - true|
  cRMSE = ||sum pred - sum true||_2 / N   (here on the conserved mean of u for periodic Burgers)
  bRMSE = RMSE on boundary points          (thermal: the outer ring of cells, where T should be near 0)
  fRMSE(band) = sqrt( sum_{k in band} |F(pred) - F(true)|^2 / (k_max - k_min + 1) ), bands low 0-4,
                mid 5-12, high 13-inf; 2-D: angular average then radial sum.
Outputs: metrics.json in this folder.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
from neuralop.models import FNO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"
BANDS = {"low": (0, 4), "mid": (5, 12), "high": (13, None)}


def frmse_1d(pred, true):
    d = (torch.fft.rfft(pred, norm="forward") - torch.fft.rfft(true, norm="forward")).abs() ** 2  # (n, K)
    out = {}
    for name, (a, b) in BANDS.items():
        b = d.shape[1] - 1 if b is None else min(b, d.shape[1] - 1)
        out[name] = math.sqrt(d[:, a:b + 1].sum(1).mean().item() / (b - a + 1))
    return out


def frmse_2d(pred, true):
    d = (torch.fft.fft2(pred, norm="forward") - torch.fft.fft2(true, norm="forward")).abs() ** 2
    n = pred.shape[-1]
    kx = torch.fft.fftfreq(n, d=1.0 / n).to(pred.device)
    kr = torch.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2).round().long()      # radial wave number
    radial = torch.zeros(pred.shape[0], kr.max().item() + 1, device=pred.device)
    radial.index_add_(1, kr.flatten(), d.flatten(1))                          # angular sum per |k|
    out = {}
    for name, (a, b) in BANDS.items():
        b = radial.shape[1] - 1 if b is None else min(b, radial.shape[1] - 1)
        out[name] = math.sqrt(radial[:, a:b + 1].sum(1).mean().item() / (b - a + 1))
    return out


def common(pred, true):
    return dict(nRMSE=((pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)).mean().item(),
                max_error=(pred - true).abs().amax().item())


results = {}
# ---- Burgers, run B models ----
D = torch.load(HERE / "burgers_nu0.1.pt"); U0, U1 = D["u0"], D["u1"]
for tag, n_modes in [("m16", 16), ("m32", 32)]:
    w = HERE / f"model_b_{tag}.pt"
    if not w.exists():
        print("missing", w); continue
    model = FNO(n_modes=(n_modes,), in_channels=1, out_channels=1, hidden_channels=64, n_layers=4).to(device)
    model.load_state_dict(torch.load(w, weights_only=False)); model.eval()  # own file; state dict carries the gelu callable
    for r in [64, 1024]:
        x = U0[1000:, :: 1024 // r].float().to(device); y = U1[1000:, :: 1024 // r].float().to(device)
        with torch.no_grad():
            p = torch.cat([model(x[i:i + 50].unsqueeze(1)).squeeze(1) for i in range(0, 200, 50)])
        m = common(p, y)
        m["cRMSE_mean_of_u"] = ((p.mean(1) - y.mean(1)).abs()).mean().item()   # mean of u is conserved by periodic Burgers
        m["mean_of_u_true_abs"] = y.mean(1).abs().mean().item()
        m["fRMSE"] = frmse_1d(p, y)
        results[f"burgers_{tag}_res{r}"] = m
        print(f"burgers {tag} res {r}:", json.dumps(m))

# ---- thermal ----
TH = ROOT / "labs/02_thermal"
if (TH / "model_thermal.pt").exists():
    D = torch.load(TH / "thermal_data.pt"); Q, T = D["q"], D["T"]
    model = FNO(n_modes=(16, 16), in_channels=1, out_channels=1, hidden_channels=32, n_layers=4).to(device)
    model.load_state_dict(torch.load(TH / "model_thermal.pt", weights_only=False)); model.eval()
    for r in [64, 128]:
        s = 256 // r
        x = (Q[1000:].reshape(200, r, s, r, s).mean(dim=(2, 4)) / 1e6).to(device)
        y = (T[1000:].reshape(200, r, s, r, s).mean(dim=(2, 4)) / 100.0).to(device)
        with torch.no_grad():
            p = torch.cat([model(x[i:i + 20].unsqueeze(1)).squeeze(1) for i in range(0, 200, 20)])
        m = common(p * 100, y * 100)   # in kelvin
        ring = torch.zeros(r, r, dtype=torch.bool); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        m["bRMSE_K"] = math.sqrt((((p - y) * 100) ** 2)[:, ring].mean().item())
        m["boundary_true_mean_K"] = (y * 100)[:, ring].mean().item()
        m["fRMSE_K"] = frmse_2d(p * 100, y * 100)
        results[f"thermal_res{r}"] = m
        print(f"thermal res {r}:", json.dumps(m))
else:
    print("thermal model not found")

json.dump(results, open(HERE / "metrics.json", "w"), indent=2)
