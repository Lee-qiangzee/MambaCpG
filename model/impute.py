import numpy as np
import torch
from argparse import ArgumentParser
import argparse
from MambaCpG import MambaCpG
from datamodules import MambaCpGImputingDataModule

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
        

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter):
    pass
parser = ArgumentParser(description='Genome-wide imputation script. Outputs a methylation matrix in the same format as the input `y.npz`, where every element is a floating number between 0 and 1 representing the model prediction.' ,
                        formatter_class=CustomFormatter)

parser.add_argument('X', type=str, metavar='X', help='NumPy file (.npz) containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file (.npz) containing input methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file (.npz) containing positions of CpG sites.')
parser.add_argument('output', type=str, metavar='output', help='NumPy file (.npz) containing output methylation matrix.')

parser.add_argument('--model_checkpoint', type=str, default=None,
                         help='.ckpt file containing the model to use. DOES NOT WORK WITH .pt STATE DICT FILES.')
parser.add_argument('--segment_size', type=int, default=1024,
                         help='Bin size in number of CpG sites (columns) that every batch will contain.')
parser.add_argument('--DNA_window', type=int, default=1001,
                         help='Haha')
parser.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker threads to use in data loading. Increase if you experience a CPU bottleneck.')
parser.add_argument('--device', type=str, choices=['CPU', 'GPU'], default='GPU',
                         help='GPU or CPU. For inference, it is currently only possible to use 1 GPU.')

optional_parse = parser.add_argument_group('General optional arguments.')

optional_parse.add_argument('--keys', type=str, nargs='+', default=None,
                            help='Only impute chromosomes corresponding to these keys.')
optional_parse.add_argument('--denoise', type=boolean, default=False,
                            help='If False, return the original methylation state for already-observed elements in the output. In other words: only unobserved elements will be imputed and observed sites will retain their original label always. If True, model predictions will be returned for all inputs, irregardless of whether they are observed.')


args = parser.parse_args()

    
X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)

dev = 'cuda' if args.device == 'GPU' else 'cpu'

model = MambaCpG.load_from_checkpoint(args.model_checkpoint)
model.eval()
model = model.to(dev)

dm = MambaCpGImputingDataModule(X, y, pos, DNA_window=args.DNA_window, segment_size=args.segment_size, 
                                    keys=args.keys, n_workers=args.n_workers)
print('Preprocessing data ...')
dm.setup(None)

y_outputs = dict()
for key, loader in dm.datasets_per_chr.items():
    print('Imputing', key, '...')
    y_outputs_key = np.empty(y[key].shape)

    lenloader_key = len(loader)

    a = y[key].shape[0] % args.segment_size
    if a == 0:
        a = args.segment_size

    for ix, batch in enumerate(loader):
        
        batch = [batch[0].to(dev, torch.long), batch[1].to(dev, model.dtype), batch[2].to(dev, torch.long)]
        with torch.no_grad():
            output = model(*batch)
        
        output = torch.sigmoid(output[0]).to('cpu').numpy()
                   
        loc = (args.segment_size) * ix

        if ix+1 == lenloader_key:
            y_outputs_key[loc:] = output[:a]
        else:
            y_outputs_key[loc:loc + args.segment_size] = output

    y_outputs[key] = y_outputs_key

if args.denoise == False:
    for key in y_outputs.keys():
        observed = np.where(y[key] != -1)
        y_outputs[key][observed] = y[key][observed]

np.savez_compressed(args.output, **y_outputs)
    
