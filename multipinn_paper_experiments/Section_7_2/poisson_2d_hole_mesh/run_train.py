import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from problem import exact_solution, poisson_2d_hole_mesh
from multipinn import *
from multipinn.callbacks import curve, heatmap, points, progress, save
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

    conditions, input_dim, output_dim = poisson_2d_hole_mesh()

    set_device_and_seed(cfg.trainer.random_seed)

    model = initialize_model(cfg, input_dim, output_dim)
    calc_loss = initialize_regularization(cfg)

    Generator(cfg.generator.domain_points, cfg.generator.sampler).use_for(conditions[0])
    Generator(cfg.generator.outer_points, cfg.generator.sampler).use_for(conditions[1])
    Generator(cfg.generator.hole_points, cfg.generator.sampler).use_for(conditions[2])

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
        heatmap.HeatmapPrediction(
            grid=grid,
            period=cfg.visualization.save_period,
            save_dir=cfg.paths.save_dir,
            save_mode=cfg.visualization.save_mode,
        ),
        heatmap.HeatmapError(
            save_dir=cfg.paths.save_dir,
            period=cfg.visualization.save_period,
            grid=grid,
            solution=exact_solution,
            save_mode=cfg.visualization.save_mode,
        ),
        points.LiveScatterPrediction(
            save_dir=cfg.paths.save_dir,
            period=cfg.visualization.save_period,
            save_mode=cfg.visualization.save_mode,
            output_index=0,
        ),
    ]

    trainer = Trainer(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.trainer.num_epochs,
        update_grid_every=cfg.trainer.grid_update,
        calc_loss=calc_loss,
        callbacks_organizer=CallbacksOrganizer(callbacks),
    )

    trainer.train()


if __name__ == "__main__":
    train()
