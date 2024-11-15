import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc, accuracy, f1_score
from blocks import BiMamba, CnnL2h128
import torch.optim.lr_scheduler as lr_scheduler


class MambaCpG(pl.LightningModule):
    def __init__(self, ncell, dim=32, nmamba=4, lr=3e-3, warmup_steps=1000, total_epochs=35):
        super().__init__()
        self.BiMamba = BiMamba(ncell, dim, nmamba)
        self.CNN = nn.Sequential(CnnL2h128(dropout=0, RF=1001), nn.ReLU(), nn.Linear(128,32))
        self.fc = nn.Linear(2 * dim, 1)
        self.hparams.lr = lr
        self.hparams.warmup_steps = warmup_steps
        self.hparams.total_epochs = total_epochs
        self.save_hyperparameters()

    def process_batch(self, batch):
        y, y_masked, mask_indices, cell_indices, x = batch
        y = y.to(torch.float)
        x = x.to(torch.long)

        return (x, y_masked, cell_indices), y, mask_indices
    
    def forward(self, x, y_masked, cell_indices):
        x = self.CNN(x.view(-1,1001))
        Mambaout = self.BiMamba(x, y_masked, cell_indices)
        return self.fc(Mambaout).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        input, y, mask_indices = self.process_batch(batch)
        y_hat = self(*input)
        y_hat = torch.diagonal(y_hat[:, mask_indices[:, :, 0], mask_indices[:, :, 1]]).reshape(-1)
        y = y.reshape(-1)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss  

    def validation_step(self, batch, batch_idx):
        input, y, mask_indices = self.process_batch(batch)
        y_hat = self(*input)
        y_hat = torch.diagonal(y_hat[:, mask_indices[:, :, 0], mask_indices[:, :, 1]]).reshape(-1)
        y = y.reshape(-1)    

        return torch.stack((y_hat, y))

    def validation_epoch_end(self, validation_step_outputs): 
        validation_step_outputs = torch.cat(validation_step_outputs, 1)
        y_hat = validation_step_outputs[0]
        y = validation_step_outputs[1]

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
        self.log('val_loss', loss, sync_dist=True)
        y = y.to(torch.int)
        self.log('AUROC', auroc(y_hat, y, task='binary'), sync_dist=True)
        self.log('acc', accuracy(y_hat, y, task='binary'), sync_dist=True)
        self.log('f1', f1_score(y_hat, y, task='binary'), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.95, 0.9), weight_decay=0.01)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.total_epochs)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr

        optimizer.step(closure=optimizer_closure)
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer
    
