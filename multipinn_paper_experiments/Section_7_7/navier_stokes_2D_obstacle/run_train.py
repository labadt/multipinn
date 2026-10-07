import os

import hydra
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from problem import problem
from multipinn import *
from multipinn.metrics import PointCloudMetric
from multipinn.utils import (
    initialize_model,
    initialize_regularization,
    save_config,
    set_device_and_seed,
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    num_gpus = dist.get_world_size()

    here = os.path.dirname(os.path.abspath(__file__))
    with open_dict(cfg.paths):
        for key in ("save_dir", "points", "value"):
            if not os.path.isabs(cfg.paths[key]):
                cfg.paths[key] = os.path.join(here, cfg.paths[key])

    if rank == 0:
        save_config(cfg, os.path.join(cfg.paths.save_dir, "used_config.yaml"))

    conditions, input_dim, output_dim = problem(re=cfg.problem.re)
    set_device_and_seed(cfg.trainer.random_seed, accelerator=f"cuda:{local_rank}")

    model = initialize_model(cfg, input_dim, output_dim, rank=rank)
    calc_loss = initialize_regularization(cfg, rank=rank)

    generator_domain = Generator(
        n_points=cfg.generator.domain_points, sampler=cfg.generator.sampler
    )
    generator_bound = Generator(
        n_points=cfg.generator.bound_points, sampler=cfg.generator.sampler
    )
    generator_domain.use_for(conditions[0])
    generator_bound.use_for(conditions[1:])

    pinn = PINN(model=model, conditions=conditions)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    callbacks = []
    if rank == 0:
        callbacks = [
            TqdmBar(
                "Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}"
            ),
            LossCurve(cfg.paths.save_dir, cfg.visualization.save_period),
            SaveModel(cfg.paths.save_dir, period=cfg.visualization.save_period),
            MetricWriter(
                [
                    PointCloudMetric.from_files(
                        cfg.paths.points,
                        cfg.paths.value,
                        field_names=["u", "v", "p"],
                    )
                ],
                cfg.paths.save_dir,
                period=cfg.visualization.save_period,
            ),
        ]

    trainer = TrainerMultiGPU(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.trainer.num_epochs,
        num_batches=num_gpus,
        update_grid_every=cfg.trainer.grid_update,
        calc_loss=calc_loss,
        callbacks_organizer=CallbacksOrganizer(callbacks),
        mixed_training=cfg.trainer.mixed_training,
        rank=rank,
    )
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
