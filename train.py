import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import yaml

from teethland.datamodules import (
    TeethBinSegDataModule,
    TeethInstSegDataModule,
    TeethLandDataModule,
    TeethMixedSegDataModule,
)
from teethland.models import (
    BinSegNet,
    DentalNet,
    LandmarkNet,
)


def main(stage: str, devices: int, checkpoint: str):
    with open('teethland/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        

    pl.seed_everything(config['seed'], workers=True)

    if stage == 'instseg':
        dm = TeethInstSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'mixedseg':
        dm = TeethMixedSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'binseg':
        dm = TeethBinSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'landmarks':
        dm = TeethLandDataModule(seed=config['seed'], **config['datamodule'])

    # dm.setup('fit')
    if stage == 'instseg':
        model = DentalNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage],
        )
    elif stage == 'mixedseg':
        model = DentalNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage],
        )

        instseg_state = torch.load(config['model']['instseg']['checkpoint_path'])
        mixedseg_state = model.state_dict()
        pop_keys = []
        for k, v in instseg_state['state_dict'].items():
            if k not in mixedseg_state:
                continue
            
            if not torch.all(torch.tensor(mixedseg_state[k].shape) == torch.tensor(v.shape)):
                pop_keys.append(k)
        
        instseg_state = {k: v for k, v in instseg_state['state_dict'].items() if k not in pop_keys}
        print(model.load_state_dict(instseg_state, strict=False))
    elif stage == 'binseg':
        model = BinSegNet(
            in_channels=dm.num_channels,
            **config['model'][stage],
        )

        landmark_state = torch.load(config['model']['landmarks']['checkpoint_path'])
        binseg_state = model.state_dict()
        pop_keys = []
        for k, v in landmark_state['state_dict'].items():
            if k not in binseg_state:
                continue
            
            if not torch.all(torch.tensor(binseg_state[k].shape) == torch.tensor(v.shape)):
                pop_keys.append(k)
        
        landmark_state = {k: v for k, v in landmark_state['state_dict'].items() if k not in pop_keys}
        print(model.load_state_dict(landmark_state, strict=False))        
    elif stage == 'landmarks':
        model = LandmarkNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            dbscan_cfg=config['model']['dbscan_cfg'],
            **config['model'][stage],
        )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=config['version'],
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='epoch',
        mode='max',
        filename='weights-{epoch:02d}',
    )
    loss_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='loss/val',
        filename='weights-{epoch:02d}',
    )
    metric_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='dice/val' if stage in ['binseg', 'landmarks'] else 'fdi_f1/val_epoch',
        mode='min' if stage in ['binseg', 'landmarks'] else 'max',
        filename='weights-{epoch:02d}',
    )


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        max_epochs=config['model'][stage]['epochs'],
        logger=logger,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_norm'],
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            metric_checkpoint_callback,
            LearningRateMonitor(),
        ],
    )
    trainer.fit(
        model, datamodule=dm, ckpt_path=checkpoint,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['instseg', 'mixedseg', 'binseg', 'landmarks'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    args = parser.parse_args()

    main(args.stage, args.devices, args.checkpoint)
