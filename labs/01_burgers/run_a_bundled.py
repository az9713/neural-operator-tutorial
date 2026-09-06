"""Lab 01, run A. FNO on the Burgers data that ships with neuraloperator.

Data: data/burgers_train_16.pt, data/burgers_test_16.pt (800 train, 400 test).
  x: initial condition u0 on 16 periodic grid points.
  y: the trajectory u(t, x) on 17 time steps x 16 grid points. y[:, 0] == x.
  visc = 0.01.
The library data processor repeats x along time, so the model is a 2-D FNO over (t, x).
Template: neuraloperator examples/models/plot_FNO_darcy.py.
"""
import json, sys, time
from pathlib import Path

import torch
from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.data.datasets.burgers import Burgers1dTimeDataset
from neuralop.utils import count_model_params
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 0
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

K_MAX = 8          # modes kept per axis (time, space); 16 points -> at most 8 modes
WIDTH = 32         # hidden channels
N_LAYERS = 4
EPOCHS = 100

ds = Burgers1dTimeDataset(
    root_dir=ROOT / "data", n_train=800, n_tests=[400],
    train_resolution=16, test_resolutions=[16],
    batch_size=32, test_batch_sizes=[32],
)
train_loader = DataLoader(ds.train_db, batch_size=32, shuffle=True)
test_loaders = {16: DataLoader(ds.test_dbs[16], batch_size=32, shuffle=False)}
data_processor = ds.data_processor.to(device)

model = FNO(n_modes=(K_MAX, K_MAX), in_channels=1, out_channels=1,
            hidden_channels=WIDTH, n_layers=N_LAYERS, projection_channel_ratio=2).to(device)
n_params = count_model_params(model)
print(f"params {n_params}")

optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

trainer = Trainer(model=model, n_epochs=EPOCHS, device=device, data_processor=data_processor,
                  wandb_log=False, eval_interval=20, use_distributed=False, verbose=True)
t0 = time.time()
trainer.train(train_loader=train_loader, test_loaders=test_loaders, optimizer=optimizer,
              scheduler=scheduler, regularizer=False, training_loss=h1loss,
              eval_losses={"h1": h1loss, "l2": l2loss})
train_seconds = time.time() - t0


@torch.no_grad()
def rel_l2(loader):
    """Mean over samples of ||pred - true||_2 / ||true||_2, on un-normalized values."""
    model.eval(); errs = []
    for batch in loader:
        batch = data_processor.preprocess(batch)
        out = model(batch["x"])
        out, batch = data_processor.postprocess(out, batch)
        y = batch["y"]
        e = (out - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)
        errs.append(e.cpu())
    return torch.cat(errs).mean().item()


test_err = rel_l2(test_loaders[16])
train_err = rel_l2(DataLoader(ds.train_db, batch_size=32, shuffle=False))
print(f"rel L2 train {train_err:.4f}  test {test_err:.4f}  time {train_seconds:.0f}s")

results = dict(run="A_bundled", data="data/burgers_{train,test}_16.pt", visc=0.01,
               n_train=800, n_test=400, resolution="17 t x 16 x", k_max=K_MAX, width=WIDTH,
               n_layers=N_LAYERS, epochs=EPOCHS, params=n_params, seed=SEED, device=device,
               gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
               train_rel_l2=train_err, test_rel_l2=test_err, train_seconds=train_seconds,
               torch=torch.__version__)
(OUT / "results_a.json").write_text(json.dumps(results, indent=2))

# one figure: three test samples, truth vs prediction as (t, x) images
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(8, 7))
# ponytail: the library's batched=False path fails for Burgers, so take one batch
batch = data_processor.preprocess(next(iter(test_loaders[16])))
with torch.no_grad():
    out_b = model(batch["x"]); out_b, batch = data_processor.postprocess(out_b, batch)
for i in range(3):
    x = batch["x"][i].squeeze().cpu(); y = batch["y"][i].squeeze().cpu(); p = out_b[i].squeeze().cpu()
    for j, (img, title) in enumerate([(x, "input u0 (repeated in t)"), (y, "truth u(t,x)"), (p, "FNO")]):
        axes[i, j].imshow(img, aspect="auto", cmap="RdBu_r"); axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        if i == 0: axes[i, j].set_title(title)
fig.suptitle(f"Burgers 16 (bundled), rel L2 test {test_err:.3f}")
fig.tight_layout(); fig.savefig(OUT / "fig_a_samples.png", dpi=120)
print("wrote results_a.json, fig_a_samples.png")
