"""Lab 01, run C. The counterexample for Module 1: a plain CNN with a filter fixed in grid points.

Same data and split as run B (burgers_nu0.1.pt, train at 64 points). The CNN has LAYERS conv layers of
kernel K, 64 channels, periodic padding. Its receptive field is 1 + LAYERS (K - 1) grid points wide. With
the default 8 layers of kernel 7 that is 49 points: 77 percent of the domain at 64 points, but 4.8 percent
of the domain at 1024 points. The physical support shrinks as the grid refines (JMLR Section 4.1,
research_notes/pdf-fno-jmlr.md:372-379). A first run with 4 layers of kernel 5 (17 points, 27 percent
of the domain) could not even fit the training grid (test rel L2 0.51 at 64 points); see results_c_rf17.json.
Expect: error rises with test resolution for the CNN, and stays flat for the FNO of run B.
Outputs: results_c.json
"""
import json, sys, time
from pathlib import Path
import torch, torch.nn as nn

OUT = Path(__file__).resolve().parent
SEED = 0
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
N_FINE, N_TRAIN, TRAIN_RES, TEST_RES = 1024, 1000, 64, [64, 128, 256, 1024]
EPOCHS, BATCH = 500, 20
LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
K = int(sys.argv[2]) if len(sys.argv) > 2 else 7
RF = 1 + LAYERS * (K - 1)

D = torch.load(OUT / "burgers_nu0.1.pt"); U0, U1 = D["u0"], D["u1"]
at_res = lambda U, r: U[:, :: N_FINE // r].float()
x_tr, y_tr = at_res(U0[:N_TRAIN], TRAIN_RES).to(device), at_res(U1[:N_TRAIN], TRAIN_RES).to(device)
tests = {r: (at_res(U0[N_TRAIN:], r).to(device), at_res(U1[N_TRAIN:], r).to(device)) for r in TEST_RES}


class PeriodicCNN(nn.Module):
    def __init__(self, width=64, k=K, layers=LAYERS):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1 if i == 0 else width, width, k, padding=k // 2, padding_mode="circular")
                                    for i in range(layers)])
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        for c in self.convs:
            x = torch.nn.functional.gelu(c(x))
        return self.out(x)


model = PeriodicCNN().to(device)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
rel_l2 = lambda p, t: (p - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1)


@torch.no_grad()
def evaluate(x, y):
    model.eval(); return rel_l2(model(x.unsqueeze(1)).squeeze(1), y).mean().item()


t0 = time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(N_TRAIN, device=device)
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i + BATCH]
        loss = rel_l2(model(x_tr[idx].unsqueeze(1)).squeeze(1), y_tr[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if ep % 100 == 0 or ep == EPOCHS - 1:
        print(f"[{ep}] test64 {evaluate(*tests[64]):.4f} test1024 {evaluate(*tests[1024]):.4f} {time.time() - t0:.0f}s")
errors = {str(r): evaluate(*tests[r]) for r in TEST_RES}
print("CNN rel L2 by test resolution:", {k: round(v, 4) for k, v in errors.items()})
json.dump(dict(run="C_cnn_baseline", model=f"{LAYERS} x Conv1d(k={K}, 64 ch, circular) + GELU, 1x1 out",
               receptive_field_points=RF, params=n_params, epochs=EPOCHS, seed=SEED, train_res=TRAIN_RES,
               test_rel_l2=errors, train_rel_l2=evaluate(x_tr, y_tr), train_seconds=time.time() - t0,
               data="burgers_nu0.1.pt (same as run B)"), open(OUT / f"results_c_rf{RF}.json", "w"), indent=2)
