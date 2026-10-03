# EvNASPINN

Multi-objective evolutionary search of fully connected PINN architectures
(depth and layer widths) built on MultiPINN.

- `problems.py` — the six benchmark tasks (taken from `examples/`) and their collocation point counts.
- `nas_search.py` — search space, NSGA-II selection, candidate training with the MultiPINN `Trainer`, Pareto front and knee export.

Search space: 2–10 hidden layers, widths from {64, 128, 256, 512, 1024}, GELU.
Objectives: physics-informed loss after the training budget and log10 of the parameter count.
Training: Adam, lr 1e-3, ExponentialLR(0.9999), 80 000 epochs per candidate.

```bash
pip install -e .
python evnaspinn/nas_search.py --task convection --smoke          # pipeline check, ~1 min
python evnaspinn/nas_search.py --task ns_curved --population 16 --generations 8
```

Tasks: `convection`, `heat`, `allen_cahn`, `ns_block`, `ns_curved`, `ns_pipe_3d`.
Results go to `nas_runs/<task>/`: `history.jsonl` (all evaluated candidates),
`pareto.csv` (terminal front) and `knee.json` (selected architecture).
