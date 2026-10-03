import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import TASKS, build

from multipinn import PINN, CallbacksOrganizer, TrainerOneBatch
from multipinn.utils import set_device_and_seed

WIDTHS = (64, 128, 256, 512, 1024)
MIN_LAYERS, MAX_LAYERS = 2, 10


class MLP(nn.Module):
    """Fully connected PINN with GELU activations."""
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        dims = [input_dim, *hidden_layers]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(widths, input_dim, output_dim):
    """P(a) = sum_k (w_k + 1) w_{k+1}."""
    dims = [input_dim, *widths, output_dim]
    return sum((a + 1) * b for a, b in zip(dims[:-1], dims[1:]))


def random_arch(rng):
    return [rng.choice(WIDTHS) for _ in range(rng.randint(MIN_LAYERS, MAX_LAYERS))]


def mutate(widths, rng, p_width=0.3, p_depth=0.2):
    child = list(widths)
    for i in range(len(child)):
        if rng.random() < p_width:
            child[i] = rng.choice(WIDTHS)
    if rng.random() < p_depth and len(child) < MAX_LAYERS:
        child.insert(rng.randrange(len(child) + 1), rng.choice(WIDTHS))
    if rng.random() < p_depth and len(child) > MIN_LAYERS:
        child.pop(rng.randrange(len(child)))
    return child


def crossover(a, b, rng):
    child = a[: rng.randint(1, len(a))] + b[rng.randint(0, len(b) - 1):]
    return child[:MAX_LAYERS] if len(child) >= MIN_LAYERS else a[:]


def train_loss(widths, task, epochs, seed, lr=1e-3, gamma=0.9999, scale=1.0):
    """Train one candidate and return its final physics-informed loss."""
    set_device_and_seed(seed)
    conditions, input_dim, output_dim = build(task, scale)
    model = MLP(input_dim, output_dim, widths)
    pinn = PINN(model=model, conditions=conditions)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    trainer = TrainerOneBatch(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=epochs,
        update_grid_every=None,
        callbacks_organizer=CallbacksOrganizer([]),
    )
    trainer.train()
    loss = float(trainer.total_loss)
    return loss if math.isfinite(loss) else float("inf")


def dominates(f, g):
    return all(x <= y for x, y in zip(f, g)) and any(x < y for x, y in zip(f, g))


def non_dominated_sort(objs):
    """Return fronts F1, F2, ... as lists of indices."""
    n = len(objs)
    dominated = [[] for _ in range(n)]
    count = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if dominates(objs[i], objs[j]):
                dominated[i].append(j)
            elif dominates(objs[j], objs[i]):
                count[i] += 1
        if count[i] == 0:
            fronts[0].append(i)
    while fronts[-1]:
        nxt = []
        for i in fronts[-1]:
            for j in dominated[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(front, objs, eps=1e-12):
    dist = {i: 0.0 for i in front}
    for m in range(len(objs[front[0]])):
        order = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[order[0]][m], objs[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = float("inf")
        for k in range(1, len(order) - 1):
            dist[order[k]] += (objs[order[k + 1]][m] - objs[order[k - 1]][m]) / (hi - lo + eps)
    return dist


def rank_population(objs):
    rank, crowd = {}, {}
    for r, front in enumerate(non_dominated_sort(objs)):
        crowd.update(crowding_distance(front, objs))
        rank.update({i: r for i in front})
    return rank, crowd


def tournament(pop_idx, rank, crowd, rng):
    a, b = rng.sample(pop_idx, 2)
    if rank[a] != rank[b]:
        return a if rank[a] < rank[b] else b
    return a if crowd[a] >= crowd[b] else b


def select_survivors(objs, size):
    chosen = []
    for front in non_dominated_sort(objs):
        if len(chosen) + len(front) <= size:
            chosen += front
        else:
            crowd = crowding_distance(front, objs)
            chosen += sorted(front, key=lambda i: -crowd[i])[: size - len(chosen)]
            break
    return chosen


def knee_point(front_objs):
    """Point of the front farthest from the line joining its extremes."""
    if len(front_objs) < 3:
        return min(range(len(front_objs)), key=lambda i: front_objs[i][0])
    pts = [(math.log10(max(f, 1e-30)), p) for f, p in front_objs]
    lo = [min(c) for c in zip(*pts)]
    hi = [max(c) for c in zip(*pts)]
    pts = [tuple((v - l) / (h - l + 1e-12) for v, l, h in zip(q, lo, hi)) for q in pts]
    a, b = min(pts, key=lambda q: q[1]), max(pts, key=lambda q: q[1])
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy) + 1e-12
    return max(range(len(pts)), key=lambda i: abs(dy * (pts[i][0] - a[0]) - dx * (pts[i][1] - a[1])) / norm)


def search(task, population=16, generations=8, epochs=80_000, seed=42, out="nas_runs", scale=1.0):
    rng = random.Random(seed)
    out = Path(out) / task
    out.mkdir(parents=True, exist_ok=True)
    _, input_dim, output_dim = build(task, scale=1e-3)
    cache, log = {}, (out / "history.jsonl").open("a", encoding="utf-8")

    def evaluate(widths, gen):
        key = tuple(widths)
        if key not in cache:
            t0 = time.time()
            loss = train_loss(widths, task, epochs, seed, scale=scale)
            params = count_params(widths, input_dim, output_dim)
            cache[key] = (loss, math.log10(params))
            log.write(json.dumps({"generation": gen, "widths": widths, "loss": loss,
                                  "params": params, "seconds": round(time.time() - t0, 1)}) + "\n")
            log.flush()
            print(f"[gen {gen}] {widths} loss={loss:.3e} params={params}")
        return cache[key]

    pop = [random_arch(rng) for _ in range(population)]
    objs = [evaluate(a, 0) for a in pop]
    for gen in range(1, generations + 1):
        rank, crowd = rank_population(objs)
        idx = list(range(len(pop)))
        children = []
        while len(children) < population:
            p1, p2 = (pop[tournament(idx, rank, crowd, rng)] for _ in range(2))
            children.append(mutate(crossover(p1, p2, rng), rng))
        merged = pop + children
        merged_objs = objs + [evaluate(c, gen) for c in children]
        keep = select_survivors(merged_objs, population)
        pop, objs = [merged[i] for i in keep], [merged_objs[i] for i in keep]
    log.close()

    front = non_dominated_sort(objs)[0]
    rows = sorted(({"widths": pop[i], "depth": len(pop[i]), "loss": objs[i][0],
                    "params": round(10 ** objs[i][1])} for i in front), key=lambda r: r["params"])
    knee = rows[knee_point([(r["loss"], math.log10(r["params"])) for r in rows])]
    with (out / "pareto.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "knee.json").write_text(json.dumps(knee, indent=2), encoding="utf-8")
    print("knee:", knee)
    return rows, knee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nas_runs")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    a = ap.parse_args()
    if a.smoke:
        search(a.task, population=4, generations=1, epochs=20, seed=a.seed, out=a.out, scale=0.02)
    else:
        search(a.task, a.population, a.generations, a.epochs, a.seed, a.out)


if __name__ == "__main__":
    main()
