import argparse
from pathlib import Path
from time import perf_counter

import os, sys
sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from teethland.datamodules import (
    TeethAlignDataModule,
    TeethInstFullDataModule,
    TeethMixedFullDataModule,
)
from teethland.models import AlignNet, FullNet


def predict(stage: str, mixed: bool, panoptic: bool, attributes: bool, devices: int, config: str):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    if stage == 'align':
        dm = TeethAlignDataModule(
            seed=config['seed'], **config['datamodule'],
            out_dir=Path(config['out_dir']),
        )
    elif mixed:
        dm = TeethMixedFullDataModule(
            seed=config['seed'], **config['datamodule'],
        )
        config['model']['instseg'] = config['model']['mixedseg']
    else:
        dm = TeethInstFullDataModule(
            seed=config['seed'], **config['datamodule'],
        )
    
    single_tooth = f'binseg{"_attributes" if attributes else ""}' if stage == 'highres' else 'landmarks'
    config['model']['single_tooth'] = config['model'][single_tooth]
    if stage == 'align':
        model = AlignNet.load_from_checkpoint(
            in_channels=dm.num_channels,
            **config['model']['align'],
            out_dir=Path(config['out_dir']),
        )
    else:
        model = FullNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            is_panoptic=panoptic,
            with_attributes=attributes,
            **config['model'],
            out_dir=Path(config['out_dir']),
        )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'{single_tooth}_{config["version"]}',
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        logger=logger,
        log_every_n_steps=1,
    )
    time = perf_counter()
    trainer.predict(model, datamodule=dm)
    print('Total inference time:', perf_counter() - time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['align', 'instances', 'highres', 'landmarks'])
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--panoptic', action='store_true')
    parser.add_argument('--attributes', action='store_true')
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--config', required=False, default='teethland/config/config.yaml', type=str)
    args = parser.parse_args()

    predict(args.stage, args.mixed, args.panoptic, args.attributes, args.devices, args.config)
