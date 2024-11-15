import numpy as np
from argparse import ArgumentParser
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from MambaCpG import MambaCpG
from datamodules import MambaCpGDataModule


def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

        
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass
        

parser = ArgumentParser(description='Training script for MambaCpG.',
                        formatter_class=CustomFormatter)
parser.add_argument('X', type=str, metavar='X', help='NumPy file containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file containing methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file containing positions of CpG sites.')

dm_parse = parser.add_argument_group('DataModule', 'Data Module arguments')
dm_parse.add_argument('--segment_size', type=int, default=1024,
                      help='Bin size in number of CpG sites (columns) that every batch will contain. If GPU memory is exceeded, this option can be lowered.')
dm_parse.add_argument('--DNA_window', type=int, default=1001,
                      help='Receptive field of the underlying CNN in datamodules.')
dm_parse.add_argument('--mask_percentage', type=float, default=0.15,
                      help='How many sites to mask each batch as a percentage of the number of columns in the batch.')
dm_parse.add_argument('--masked_replace_percentage', type=float, default=0,
                      help='The percentage of masked sites to instead randomize.')
dm_parse.add_argument('--val_keys', type=str, nargs='+', default=['chr5'],
                      help='Names/keys of validation chromosomes.')
dm_parse.add_argument('--test_keys', type=str, nargs='+', default=['chr10'], 
                      help='Names/keys of test chromosomes.')
dm_parse.add_argument('--batch_size', type=int, default=1,
                      help='Batch size.')
dm_parse.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')

model_parse = parser.add_argument_group('Model', 'MambaCpG Hyperparameters')
model_parse.add_argument('--mDNA_window', type=int, default=1001,
                         help='Receptive field of the underlying CNN.')  
model_parse.add_argument('--dim', type=int, default=32,
                         help='model hidden size.')                       
model_parse.add_argument('--nmamba', type=int, default=4,
                         help='Number of mamba blocks to use.')
model_parse.add_argument('--total_epochs', type=int, default=35,
                         help='Cosine decay period.')
model_parse.add_argument('--lr', type=float, default=1e-3,
                         help='Learning rate.')
model_parse.add_argument('--warmup_steps', type=int, default=1000,
                         help='Number of steps over which the learning rate will linearly warm up.')

log_parse = parser.add_argument_group('Logging', 'Logging arguments')
log_parse.add_argument('--tensorboard', type=boolean, default=True,
                       help='Whether to use tensorboard. If True, then training progress can be followed by using (1) `tensorboard --logdir logfolder/` in a separate terminal and (2) accessing at localhost:6006.')
log_parse.add_argument('--log_folder', type=str, default='logfolder',
                       help='Folder where the tensorboard logs will be saved. Will additinally contain saved model checkpoints.')
log_parse.add_argument('--experiment_name', type=str, default='experiment',
                       help='Name of the run within the log folder.')
log_parse.add_argument('--earlystop', type=boolean, default=True,
                       help='Whether to use early stopping after the validation loss has not decreased for `patience` epochs.')
log_parse.add_argument('--patience', type=int, default=10,
                       help='Number of epochs to wait for a possible decrease in validation loss before early stopping.')


parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)

ncell = y.shape[0]

model = MambaCpG(ncell, dim=args.dim, nmamba=args.nmamba, lr=args.lr, warmup_steps=args.warmup_steps, total_epochs=args.total_epochs)
datamodule = MambaCpGDataModule(X, y, pos, DNA_window=args.DNA_window, val_keys=args.val_keys, test_keys=args.test_keys,
                                      segment_size=args.segment_size, batch_size=args.batch_size, n_workers=args.n_workers,
                                      mask_percentage=args.mask_percentage, masked_replace_percentage=args.masked_replace_percentage )

print('-------------------------Running in-------------------------')
callbacks = [ModelCheckpoint(monitor='val_loss', mode='min')]
if args.tensorboard:
    logger = TensorBoardLogger(args.log_folder, name=args.experiment_name)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks += [lr_monitor]

if args.earlystop:
    earlystopper = EarlyStopping(monitor='val_loss',patience=args.patience,mode='min')
    callbacks += [earlystopper]

ddp_strategy = DDPStrategy(find_unused_parameters=False)

trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, precision=32, strategy=ddp_strategy, accelerator='gpu', devices=[0])
trainer.fit(model, datamodule)
