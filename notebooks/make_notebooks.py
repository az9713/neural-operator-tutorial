"""Build the course notebooks from the lab scripts. Standard library only (nbformat is not installed).

Each lab script becomes one .ipynb: the module docstring is the first markdown cell, then one code cell
per block separated by a line starting with "# ----". Run from anywhere: python notebooks/make_notebooks.py
The notebooks reference the lab folders by relative path, so run them from notebooks/ with a kernel
whose cwd is this folder, or set OUT at the top of the first cell.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
LABS = [
    ("01a_burgers_bundled.ipynb", "labs/01_burgers/run_a_bundled.py", "Module 2 and 7. Lab 01 run A: the bundled Burgers data through the library loader and Trainer."),
    ("01b_burgers_paper_spec.ipynb", "labs/01_burgers/run_b_paper_spec.py", "Module 2 and 1. Lab 01 run B: paper spec Burgers, own checked solver, train at 64, test at four grids. Argument n_modes: set N_MODES in the cell."),
    ("01c_cnn_counterexample.ipynb", "labs/01_burgers/run_c_cnn_baseline.py", "Module 1. Lab 01 run C: a CNN with a filter fixed in grid points, evaluated at four grids."),
    ("02_thermal_die.ipynb", "labs/02_thermal/run_thermal.py", "Module 6, 8, 11, capstone. Lab 02: steady die thermal conduction, FD truth, FNO surrogate, hot spot metrics."),
    ("03_scale.ipynb", "labs/03_scale/run_scale.py", "Module 9. Lab 03: memory and time of one FNO forward pass against grid size and mode count."),
    ("08_metrics.ipynb", "labs/01_burgers/metrics.py", "Module 8. Banded spectral error, mean conservation, and boundary error on the saved Lab 01 and Lab 02 outputs."),
]


def cells_from_script(src: str, intro: str, script_rel: str):
    m = re.match(r'\s*"""(.*?)"""\s*', src, re.S)
    doc = m.group(1).strip() if m else ""
    body = src[m.end():] if m else src
    # the notebook runs from notebooks/, so __file__ must point at the lab folder
    header = (f"# Built from {script_rel}. The script's own folder is used for data and outputs.\n"
              f"from pathlib import Path\n__file__ = str(Path.cwd().parent / '{script_rel}')\n"
              f"import sys; sys.argv = [__file__]  # set sys.argv = [__file__, '32'] to pass a script argument\n")
    blocks = re.split(r"\n(?=# ----)", body.strip("\n"))
    cells = [md(f"## {intro}\n\n```\n{doc}\n```"), code(header)]
    for b in blocks:
        if b.strip():
            cells.append(code(b.strip("\n") + "\n"))
    return cells


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


for name, rel, intro in LABS:
    path = ROOT / rel
    if not path.exists():
        print("missing", rel); continue
    nb = {"cells": cells_from_script(path.read_text(encoding="utf-8"), intro, rel),
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (NB / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", name, len(nb["cells"]), "cells")
