import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from argparse import ArgumentParser
import argparse
import sys
sys.path.append('..')
import pandas as pd
from model.MambaCpG import MambaCpG

def boolean(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
    pass

parser = ArgumentParser(description='Interpretation script. Only works with GPU.', formatter_class=CustomFormatter)

parser.add_argument('X', type=str, metavar='X', help='NumPy file (.npz) containing encoded genome.')
parser.add_argument('y', type=str, metavar='y', help='NumPy file (.npz) containing input methylation matrix.')
parser.add_argument('pos', type=str, metavar='pos', help='NumPy file (.npz) containing positions of CpG sites.')
parser.add_argument('--output', type=str, metavar='output', help='outputfilepath')
parser.add_argument('--halfwindowsize', type=int, metavar='output', help='size')
parser.add_argument('--model_checkpoint', type=str, default=None, help='.ckpt file containing the model to use. DOES NOT WORK WITH .pt STATE DICT FILES.')
parser.add_argument('--config_file', type=str, default=None, help='config file specifying which sites to interpret and how. See README for more info.')

args = parser.parse_args()

print('----- Integrated Gradients interpretation -----')

X = np.load(args.X)
y = np.load(args.y)
pos = np.load(args.pos)

halfwin = args.halfwindowsize

model = MambaCpG.load_from_checkpoint(args.model_checkpoint)
model = model.to('cuda')
model.eval()

config = pd.read_csv(args.config_file, header=None)

class InterpreterCaptum(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, DNAinput, CpGinput, cellinput, pos, row_index_s, y_true):
        x = torch.cat((CpGinput, cellinput, DNAinput), -1) + pos
        x = self.model.BiMamba.fcc(x).reshape(bsz, -1, 2 * 32)
        x = x.reshape(bsz, -1, 2 * 32)

        for i in range(4):
            xforward = self.model.BiMamba.mamba(x)
            xbackward = self.model.BiMamba.mamba(torch.flip(x, dims=(1,)))
            x = self.model.BiMamba.LayerNorm(x + (xforward + torch.flip(xbackward, dims=(1,))) / 2)
            x = x.reshape(bsz, nsite, ncell, 2 * 32)
            x = x.transpose(1, 2).reshape(bsz, -1, 2 * 32)
            xforward = self.model.BiMamba.mamba(x)
            xbackward = self.model.BiMamba.mamba(torch.flip(x, dims=(1,)))
            x = self.model.BiMamba.LayerNorm(x + (xforward + torch.flip(xbackward, dims=(1,))) / 2)
            x = x.reshape(bsz, ncell, nsite, 2 * 32)
            x = x.transpose(1, 2).reshape(bsz, -1, 2 * 32)

        y_hat = x.reshape(bsz, nsite, ncell, 2 * 32)
        y_hat = self.model.fc(y_hat).squeeze(-1)
        
        y_hat = y_hat[0, halfwin, row_index_s].reshape(-1)
        
        return 1 - torch.abs(y_true - torch.sigmoid(y_hat))

InterpreterNet = InterpreterCaptum(model)

ig = IntegratedGradients(InterpreterNet)

out_dict = {'errors': [], 'ref': [], 'total': []}

print(config.shape[0])
for i in range(config.shape[0]):
    print('Interpreting row number', i, "...", end='\r')
    row = config.iloc[i, :]
    key_s = row[0]
    row_index_s = row[1]
    col_index_s = row[2]
    ref_label_s = None if row[3] == 'None' else float(row[3])

    if ref_label_s is not None:
        ref_label_s = int(ref_label_s)
    
    pos_local = torch.from_numpy(pos[key_s])

    x = torch.from_numpy(X[key_s]).to('cuda')
    padding = torch.full((500,), 4, dtype=torch.int8).to('cuda')
    DNAx = torch.cat((padding, x, padding), dim=0)

    pos_local = pos_local + 500
    start = pos_local - 500
    end = pos_local + 500 + 1
    DNA_seg = []
    for s, e in zip(start, end):
        DNA_seg.append(DNAx[s:e])
    DNA_seg = torch.stack(DNA_seg)

    x = DNA_seg[max(0, col_index_s - halfwin):col_index_s + halfwin + 1]
    y_local = torch.from_numpy(y[key_s][max(0, col_index_s - halfwin):col_index_s + halfwin + 1]).to('cuda')
    pos_local = pos_local[max(0, col_index_s - halfwin):col_index_s + halfwin + 1].to('cuda')
    pos_local = pos_local - pos_local[0]
    cell_indices = torch.arange(y_local.shape[1]).to('cuda').unsqueeze(0)

    y_true = y_local[col_index_s, row_index_s] if ref_label_s is None else ref_label_s
    y_true = torch.tensor(y_true).unsqueeze(0).reshape(-1).to('cuda')
    y_local = y_local + 1
    
    x = x.unsqueeze(0).to(torch.long)
    y_local = y_local.unsqueeze(0).to(torch.long)

    with torch.no_grad():
        bsz, nsite, ncell = y_local.shape
        DNAinput = model.CNN(x.view(-1, 1001)).reshape(bsz, nsite, -1).unsqueeze(-2).expand(-1, -1, ncell, -1)
        cellinput = model.BiMamba.cellEB(cell_indices).unsqueeze(1).expand(-1, nsite, -1, -1)
        CpGinput = model.BiMamba.CpGEB(y_local.long())
        modelpos = model.BiMamba.positionalencoding2d(96, nsite, ncell).to(x.device).unsqueeze(0)

    out = ig.attribute(inputs=(DNAinput, CpGinput, cellinput), additional_forward_args=(modelpos, row_index_s, y_true),
                      internal_batch_size=1)
    
    for param in model.parameters():
        param.grad = None
    torch.cuda.empty_cache()

    with torch.no_grad():
        pred = InterpreterNet(DNAinput, CpGinput, cellinput, modelpos, row_index_s, y_true)
    print(out[0].shape, out[1].shape, out[2].shape)
    
    if out[0].shape[1] != 2 * halfwin + 1:
        continue

    out_dict['errors'].append(pred[0].cpu().item())
    out_dict['ref'].append(y_true[0].cpu().item())
    out_dict['total'].append(torch.sum(out[0][0] + out[1][0] + out[2][0], -1).cpu())


out_dict['errors'] = np.array(out_dict['errors'])
out_dict['ref'] = np.array(out_dict['ref'])
out_dict['total'] = np.array([p.numpy() for p in out_dict['total']])

np.savez_compressed(args.output, **out_dict)

print()
print('Done')
