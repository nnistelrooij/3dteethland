import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from teethland.datamodules import (
    TeethAlignDataModule,
    TeethBinSegDataModule,
    TeethBinSegWithAttributesDataModule,
    TeethMultiSegDataModule,
    TeethInstSegDataModule,
    TeethLandDataModule,
    TeethMixedSegDataModule,
    TeethPanopticSegDataModule,
)
from teethland.models import (
    AlignNet,
    BinSegNet,
    BinSegAttributesNet,
    MultiSegNet,
    DentalNet,
    LandmarkNet,
    PanopticNet,
)


def main(stage: str, devices: int, checkpoint: str):
    with open('teethland/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        

    pl.seed_everything(config['seed'], workers=True)

    if stage == 'align':
        dm = TeethAlignDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'instseg':
        dm = TeethInstSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'mixedseg':
        dm = TeethMixedSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'binseg':
        dm = TeethBinSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'landmarks':
        dm = TeethLandDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'panopseg':
        dm = TeethPanopticSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'binseg_attributes':
        dm = TeethBinSegWithAttributesDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'multiseg':
        dm = TeethMultiSegDataModule(seed=config['seed'], **config['datamodule'])

    # dm.setup('fit')
    if stage == 'align':
        model = AlignNet(
            in_channels=dm.num_channels,
            **config['model'][stage],
        )
    elif stage in ['instseg', 'mixedseg']:
        # model = DentalNet(
        model = DentalNet.load_from_checkpoint(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage],
        )
    elif stage == 'binseg':
        model = BinSegNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage],
        )     
    elif stage == 'landmarks':
        model = LandmarkNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            dbscan_cfg=config['model']['dbscan_cfg'],
            **config['model'][stage],
        )
    elif stage == 'panopseg':
        stage = 'instseg'
        model = PanopticNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage],
        )
    elif stage == 'binseg_attributes':
        model = BinSegAttributesNet(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage],
        )
    elif stage == 'multiseg':
        stage = 'binseg'
        # model = MultiSegNet(
        model = MultiSegNet.load_from_checkpoint(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            **config['model'][stage]
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
    seg_stages = ['binseg', 'landmarks', 'binseg_attributes']
    metric_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='dice/val' if stage in seg_stages else 'fdi_f1/val_epoch',
        mode='min' if stage in ['binseg', 'landmarks'] else 'max',
        filename='weights-{epoch:02d}',
    )


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        max_epochs=config['model'][stage]['epochs'],
        logger=logger,
        sync_batchnorm=devices > 1,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_norm'],
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            *([metric_checkpoint_callback] if stage != 'align' else []),
            LearningRateMonitor(),
        ],
    )
    trainer.fit(
        model, datamodule=dm, ckpt_path=checkpoint,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=[
        'align', 'instseg', 'mixedseg', 'panopseg',
        'binseg', 'binseg_attributes', 'multiseg', 'landmarks',
    ])
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    args = parser.parse_args()

    main(args.stage, args.devices, args.checkpoint)
