import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from examples.navier_stokes_2D_obstacle_grad_acc.problem import problem
from multipinn import *
from multipinn.utils import (
    initialize_model,
    initialize_regularization,
    save_config,
    set_device_and_seed,
)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    config_save_path = os.path.join(cfg.paths.save_dir, "used_config.yaml")
    save_config(cfg, config_save_path)

    conditions, input_dim, output_dim = problem(re=cfg.problem.re)

    set_device_and_seed(cfg.trainer.random_seed)

    model = initialize_model(cfg, input_dim, output_dim)
    calc_loss = initialize_regularization(cfg)

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

    grid = heatmap.Grid.from_pinn(pinn, cfg.visualization.grid_plot_points)

    callbacks = [
        progress.TqdmBar(
            "Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}"
        ),
        curve.LossCurve(cfg.paths.save_dir, cfg.visualization.save_period),
        save.SaveModel(cfg.paths.save_dir, period=cfg.visualization.save_period),
    ]

    callbacks += [
        points.LiveScatterPrediction(
            save_dir=cfg.paths.save_dir,
            period=cfg.visualization.save_period,
            save_mode=cfg.visualization.save_mode,
            output_index=i,
        )
        for i in range(output_dim)
    ]

    callbacks += [
        heatmap.HeatmapPrediction(
            grid=grid,
            period=cfg.visualization.save_period,
            save_dir=cfg.paths.save_dir,
            save_mode=cfg.visualization.save_mode,
            output_index=i,
        )
        for i in range(output_dim)
    ]

    trainer = Trainer(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.trainer.num_epochs,
        update_grid_every=cfg.trainer.grid_update,
        num_batches=cfg.trainer.num_batches,
        calc_loss=calc_loss,
        callbacks_organizer=CallbacksOrganizer(callbacks),
    )

    trainer.train()


if __name__ == "__main__":
    train()
