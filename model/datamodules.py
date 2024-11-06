import torch
import torch.utils.data
import pytorch_lightning as pl


class MambaCpGDataModule(pl.LightningDataModule):
    def __init__(self, x, y, pos, DNA_window=1001, segment_size=1024, batch_size=1, 
                 n_workers=4, mask_percentage=0.15, masked_replace_percentage=0.2, val_keys=None, test_keys=None):
        super().__init__()
        self.x = x
        self.y = y
        self.pos = pos
        self.DNA_win = DNA_window
        self.hDNA_win = int((DNA_window-1)/2)                      
        self.segsz = segment_size
        self.bsz = batch_size
        self.n_wor = n_workers
        self.maskp = mask_percentage
        self.replacep = masked_replace_percentage
        self.val_keys = val_keys
        self.test_keys = test_keys
        self.prepare_data_per_node = True

    def setup(self, stage):
        train = []; val = []; test = []
        
        for chr_name in self.y.keys():
            x = self.x[chr_name]
            y = self.y[chr_name]
            pos = self.pos[chr_name]
            if 'numpy' in str(type(x)):
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                pos = torch.from_numpy(pos)
            
            # DNA window
            DNAx = torch.cat((torch.full((self.hDNA_win,), 4, dtype=torch.int8), x, torch.full((self.hDNA_win,), 4, dtype=torch.int8)), dim=0)
            posx = pos + self.hDNA_win
            start = posx - self.hDNA_win
            end = posx + self.hDNA_win + 1
            DNA_seg = []  
            for start, end in zip(start, end):
                DNA_seg.append(DNAx[start:end]) # (_, DNA_win)
            DNA_seg = torch.stack(DNA_seg)

            n_pos = len(pos)
            if n_pos < self.segsz:
                continue
            
            # batch division
            batch_ = [(DNA_seg[i:i + self.segsz], y[i:i + self.segsz], pos[i:i + self.segsz] - pos[i])
                       for i in range(0, len(pos) - self.segsz + 1, self.segsz)]
       
            # process edge
            a = len(pos) % self.segsz

            _, ncell = y.shape
            end_DNA_padding = torch.full((self.segsz - a, self.DNA_win), 4, dtype=torch.int8)
            end_DNA_seg = torch.cat((DNA_seg[-a:], end_DNA_padding), dim=0)
            end_pos_padding = torch.full((self.segsz - a,), -1, dtype=torch.int32)
            end_pos_seg = torch.cat((pos[-a:], end_pos_padding), dim=0)
            end_CpG_padding = torch.full((self.segsz - a, ncell), -1, dtype=torch.int8)
            end_CpG_seg = torch.cat((y[-a:], end_CpG_padding), dim=0)

            batch_.append((end_DNA_seg, end_CpG_seg, end_pos_seg))               

            if self.val_keys is not None and chr_name in self.val_keys:
                val += batch_
            elif self.test_keys is not None and chr_name in self.test_keys:
                test += batch_
            else:
                train += batch_
        
        self.train = MambaCpGDataset(train, mask_percentage=self.maskp, masked_replace_percentage=self.replacep)
        
        self.val = MambaCpGDataset(val, mask_percentage=self.maskp, masked_replace_percentage=self.replacep)

        self.test = test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=self.n_wor, batch_size=self.bsz, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, num_workers=self.n_wor, batch_size=self.bsz, shuffle=False, pin_memory=True)


class MambaCpGDataset(torch.utils.data.Dataset):
    def __init__(self, split, mask_percentage, masked_replace_percentage): 
        self.split = split
        self.maskp = mask_percentage
        self.replacep = masked_replace_percentage

    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, _ = self.split[index] # (nsite, DNA_win), (nsite, ncell), (nsite,)
        cell_indices = torch.arange(y.shape[1])
        y = y + 1

        y_known = y.nonzero(as_tuple=False) # site index of known methylation status
        num_to_mask = int(y_known.size(0) * self.maskp)

        # randomly select the index to mask
        mask_indices = y_known[torch.randperm(y_known.size(0))[:num_to_mask]]

        # create methylation matrix after mask
        y_masked = y.clone()
        y_masked[mask_indices[:, 0], mask_indices[:, 1]] = 0

        num_to_replace = int(num_to_mask * self.replacep)
        replace_indices = mask_indices[torch.randperm(mask_indices.size(0))[:num_to_replace]]

        replacements = torch.randint(1, 3, (num_to_replace,))
        for i, (row, col) in enumerate(replace_indices):
            y_masked[row, col] = replacements[i]
        
        y_orig = []
        for i in range(num_to_mask):
            y_orig.append(y[mask_indices[i, 0], mask_indices[i, 1]])

        y_orig = torch.tensor(y_orig) - 1

        return y_orig, y_masked, mask_indices, cell_indices, x
    

class MambaCpGImputingDataModule(pl.LightningDataModule):
    def __init__(self, x, y, pos, DNA_window=1001, segment_size=1024, keys=None, n_workers=4):
        assert keys is None or type(keys) is list, 'keys should be None or list'
        super().__init__()
        self.x = x
        self.y = y
        self.pos = pos
        self.DNA_win = DNA_window
        self.hDNA_win = int((DNA_window - 1) / 2)                      
        self.segsz = segment_size
        self.keys = keys
        self.n_wor = n_workers
        
    def setup(self, stage):
        
        self.datasets_per_chr = dict()
        
        iterate = self.keys if self.keys is not None else self.y.keys()
        for chr_name in iterate:
            x = self.x[chr_name]
            y = self.y[chr_name]
            pos = self.pos[chr_name]
            if 'numpy' in str(type(x)):
                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                pos = torch.from_numpy(pos)

            DNAx = torch.cat((torch.full((self.hDNA_win,), 4, dtype=torch.int8), x, torch.full((self.hDNA_win,), 4, dtype=torch.int8)), dim=0)
            posx = pos + self.hDNA_win
            start = posx - self.hDNA_win
            end = posx + self.hDNA_win + 1
            DNA_seg = []  
            for start, end in zip(start, end):
                DNA_seg.append(DNAx[start:end])
            DNA_seg = torch.stack(DNA_seg)

            batch_ = [(DNA_seg[i:i + self.segsz], y[i:i + self.segsz], pos[i:i + self.segsz] - pos[i])
                       for i in range(0, len(pos) - self.segsz + 1, self.segsz)]
       
            a = len(pos) % self.segsz

            _, ncell = y.shape
            end_DNA_padding = torch.full((self.segsz - a, self.DNA_win), 4, dtype=torch.int8)
            end_DNA_seg = torch.cat((DNA_seg[-a:], end_DNA_padding), dim=0)
            end_pos_padding = torch.full((self.segsz - a,), -1, dtype=torch.int32)
            end_pos_seg = torch.cat((pos[-a:], end_pos_padding), dim=0)
            end_CpG_padding = torch.full((self.segsz - a, ncell), -1, dtype=torch.int8)
            end_CpG_seg = torch.cat((y[-a:], end_CpG_padding), dim=0)

            batch_.append((end_DNA_seg, end_CpG_seg, end_pos_seg))               

            self.datasets_per_chr[chr_name] = torch.utils.data.DataLoader(
                ImputingDataset(batch_, DNA_window=self.DNA_win),
                num_workers = self.n_wor, shuffle=False, pin_memory=True)
            
# Imputing dataset. Makes overlapping segments.
class ImputingDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.split = split
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self, index):
        x, y, _ = self.split[index] 
        y_orig = y + 1
        cell_indices = torch.arange(y.shape[1])
        
        return x, y_orig, cell_indices   
     