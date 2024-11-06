import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba


# The CnnL2h128 was based on:
# https://github.com/gdewael/cpg-transformer/blob/main/cpg_transformer/blocks.py
class CnnL2h128(nn.Module):
    def __init__(self, dropout=0, RF=1001):
        super().__init__()
        self.hlen = (((RF - 10) // 4) - 2) // 2
        self.embed = nn.Embedding(16, 4)
        self.CNN = nn.Sequential(nn.Conv1d(4, 128, 11), nn.ReLU(), nn.MaxPool1d(4),
                                 nn.Conv1d(128, 256, 3), nn.ReLU(), nn.MaxPool1d(2))
        self.lin = nn.Sequential(nn.Linear(256 * self.hlen, 128), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = self.CNN(x).view(-1, 256 * self.hlen)
        return self.lin(x)


class BiMamba(nn.Module):
    def __init__(self, ncell, dim, nmamba):
        super().__init__()
        self.dim = dim
        self.cellEB = nn.Embedding(ncell, dim)
        self.CpGEB = nn.Embedding(3, dim)
        self.nmamba = nmamba
        self.LayerNorm = nn.LayerNorm(2 * dim)
        self.fcc = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.ReLU())
        self.mamba = Mamba(d_model=2 * dim, d_state=256, d_conv=4, expand=2).to("cuda")


    # The positionalencoding2d was based on:
    # https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py        
    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1) # width, dim/4 -> dim/4, width -> dim/4, height, width
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width) # height, dim/4 -> dim/4, height -> dim/4, height, width
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.permute(1, 2, 0)
    
    def forward(self, x, y, cell_indices):
        """
        :param x: DNA context. Dimensions:(bsz, nsite, dim)
        :param y: Methylation matrix. Dimensions:(bsz, nsite, ncell)
        :param cell_indices: Cell index. Dimensions:(bsz, ncell)
        :return: (bsz, nsite, ncell, 2 * self.dim)
        """        
        bsz, nsite, ncell = y.shape
        cellinput = self.cellEB(cell_indices).unsqueeze(1).expand(-1, nsite, -1, -1)
        DNAinput = x.reshape(bsz, nsite, -1).unsqueeze(-2).expand(-1, -1, ncell, -1)
        CpGinput = self.CpGEB(y.long())

        pos = self.positionalencoding2d(3 * self.dim, nsite, ncell).to(x.device)
        pos = pos.unsqueeze(0)
        x = torch.cat((CpGinput, cellinput, DNAinput), -1) + pos
        x = self.fcc(x).reshape(bsz, -1, 2 * self.dim)
        x = x.reshape(bsz, -1, 2 * self.dim)

        for i in range(self.nmamba):
            xforward = self.mamba(x)
            xbackward = self.mamba(torch.flip(x, dims=(1,)))
            x = self.LayerNorm(x + (xforward + torch.flip(xbackward, dims=(1,))) / 2)
            x = x.reshape(bsz, nsite, ncell, 2 * self.dim)
            x = x.transpose(1, 2).reshape(bsz, -1, 2 * self.dim)
            xforward = self.mamba(x)
            xbackward = self.mamba(torch.flip(x, dims=(1,)))
            x = self.LayerNorm(x + (xforward + torch.flip(xbackward, dims=(1,))) / 2)
            x = x.reshape(bsz, ncell, nsite, 2 * self.dim)
            x = x.transpose(1, 2).reshape(bsz, -1, 2 * self.dim)

        return x.reshape(bsz, nsite, ncell, 2 * self.dim)
    