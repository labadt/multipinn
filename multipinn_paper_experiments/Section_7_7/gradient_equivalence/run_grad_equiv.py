import os

import torch
import torch.distributed as dist

from problem import problem
from multipinn import *


SEED = 42
DOMAIN_POINTS = 65_536
BOUND_POINTS = 8_192
HIDDEN = [64, 128, 128, 256, 256, 128, 128, 64]


def flat_grad(model):
    return torch.cat(
        [parameter.grad.detach().reshape(-1).float().cpu() for parameter in model.parameters()]
    )


def relative_l2(reference, other):
    return torch.linalg.norm(other - reference) / torch.linalg.norm(reference)


def build(local_rank):
    conditions, input_dim, output_dim = problem(re=50)
    set_device_and_seed(SEED, accelerator=f"cuda:{local_rank}")
    model = FNN(input_dim, output_dim, HIDDEN)
    Generator(DOMAIN_POINTS, "Hammersley").use_for(conditions[0])
    Generator(BOUND_POINTS, "Hammersley").use_for(conditions[1:])
    pinn = PINN(model=model, conditions=conditions)
    pinn.update_data()
    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return model, pinn, saved


def restore(model, saved):
    model.load_state_dict(saved)
    model.zero_grad(set_to_none=True)


def capture_batches(model, pinn, saved, num_batches):
    restore(model, saved)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    common = dict(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=1,
        update_grid_every=None,
        calc_loss=AdaptiveConditionsLosses(),
        callbacks_organizer=None,
    )
    if num_batches == 1:
        trainer = TrainerOneBatch(**common)
    else:
        trainer = Trainer(num_batches=num_batches, **common)
    trainer.pinn.model.train()
    trainer.optimizer.zero_grad()
    trainer._prepare_loss()
    for trainer.current_batch in range(trainer.num_batches):
        trainer.pinn.select_batch(trainer.current_batch)
        batch_loss, _ = trainer.calc_loss(trainer)
        (batch_loss * trainer.inum_batches).backward()
    return flat_grad(model)


def capture_ranks(model, pinn, saved, rank):
    restore(model, saved)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = TrainerMultiGPU(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=None,
        num_epochs=1,
        num_batches=dist.get_world_size(),
        update_grid_every=None,
        calc_loss=AdaptiveConditionsLosses(),
        callbacks_organizer=None,
        rank=rank,
    )
    trainer.pinn.model.train()
    trainer.optimizer.zero_grad()
    trainer._prepare_loss()
    trainer.pinn.select_batch(trainer.rank)
    batch_loss, _ = trainer.calc_loss(trainer)
    batch_loss.backward()
    trainer.reduce_gradients()
    return flat_grad(model)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    model, pinn, saved = build(local_rank)
    full = accum = None
    if rank == 0:
        full = capture_batches(model, pinn, saved, 1)
        accum = capture_batches(model, pinn, saved, 2)
    dist.barrier()
    ranks = capture_ranks(model, pinn, saved, rank)
    if rank == 0:
        print(f"full vs accumulation: {relative_l2(full, accum).item():.6e}")
        print(f"full vs 2 ranks: {relative_l2(full, ranks).item():.6e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
