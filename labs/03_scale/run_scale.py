"""Lab 03 (Module 9). Memory and time of one FNO forward pass against grid size and mode count.

2-D FNO, batch 1, 32 channels, 4 layers, grids 64^2 to 1024^2, n_modes 16 and 64. Records peak GPU
memory and median wall time over 5 passes after 2 warm up passes. Then extrapolates to a 1000^3 grid by
the O(n log n) FFT cost and the O(n) activation memory, which is the arithmetic behind the "trillion
context" claim (Module 9). Outputs: results_scale.json
"""
import json, time, math
from pathlib import Path
import torch
from neuralop.models import FNO

OUT = Path(__file__).resolve().parent
device = "cuda"
torch.manual_seed(0)
rows = []
for n_modes in [16, 64]:
    for n in [64, 128, 256, 512, 1024]:
        if n_modes > n // 2:
            continue
        model = FNO(n_modes=(n_modes, n_modes), in_channels=1, out_channels=1, hidden_channels=32, n_layers=4).to(device)
        x = torch.randn(1, 1, n, n, device=device)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        try:
            with torch.no_grad():
                for _ in range(2):
                    model(x)
                torch.cuda.synchronize(); times = []
                for _ in range(5):
                    t0 = time.perf_counter(); model(x); torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            times.sort()
            rows.append(dict(n_modes=n_modes, grid=n, points=n * n, peak_MiB=peak, ms=times[2] * 1e3,
                             params=sum(p.numel() for p in model.parameters())))
            print(rows[-1])
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(n_modes=n_modes, grid=n, points=n * n, oom=True)); print(rows[-1])
        del model, x; torch.cuda.empty_cache()

# extrapolation from the largest completed 2-D row with n_modes = 16: memory ~ points, time ~ points log points
ok = [r for r in rows if r["n_modes"] == 16 and not r.get("oom")]
base = ok[-1]
target = 1000 ** 3
mem_GiB = base["peak_MiB"] / 2 ** 10 * target / base["points"]
t_s = base["ms"] / 1e3 * (target * math.log2(target)) / (base["points"] * math.log2(base["points"]))
extrap = dict(from_grid=base["grid"], target_points=target, est_forward_peak_GiB_batch1=mem_GiB,
              est_forward_seconds_this_gpu=t_s,
              note="linear memory and n log n time scaling from the largest 2-D row; a 3-D FNO also needs "
                   "n_modes^3 weights per channel pair per layer: 16^3 x 32^2 x 4 layers x 8 bytes = "
                   f"{16**3 * 32**2 * 4 * 8 / 2**20:.0f} MiB, versus 16^2 in 2-D")
json.dump(dict(gpu=torch.cuda.get_device_name(0), rows=rows, extrapolation=extrap), open(OUT / "results_scale.json", "w"), indent=2)
print(extrap)
