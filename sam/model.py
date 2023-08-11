# Import libraries
import os, torch, torchmetrics, timm, wandb, pytorch_lightning as pl, numpy as np, monai
from torch import nn; from torch.nn import functional as F; from time import time
from pytorch_lightning.callbacks import Callback; from utils import get_model, tensor2im

class LitModel(pl.LightningModule):
    
    """
    
    This class gets several arguments and returns a model for training.
    
    """
    
    def __init__(self, ds_name, lr = 2e-4):
        super().__init__()
        
        # Log hyperparameters
        self.save_hyperparameters()
        # Get class parameter arguments
        self.lr, self.ds_name = lr, ds_name
        # Set the loss function
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        # Get model to be trained
        self.model = get_model()
        # Set the lists to track train and validation times
        self.train_times, self.validation_times = [], []

    # Get optimizere to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    def on_train_epoch_start(self): self.train_start_time = time()
    
    def on_train_epoch_end(self): self.train_elapsed_time = time() - self.train_start_time; self.train_times.append(self.train_elapsed_time); self.log("train_time", self.train_elapsed_time, prog_bar = True, on_step = False, on_epoch = True, logger = True)
        
    def training_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        im, gt, bboxes = batch["pixel_values"], batch["ground_truth_mask"].float(), batch["input_boxes"]
        pred_mask = self.model(pixel_values = im, input_boxes = bboxes, multimask_output = False)
        
        if self.ds_name in ["mri", "isic"]:
            pred_mask = pred_mask.pred_masks.squeeze(1)
            gt = gt.unsqueeze(1)
        
        else: pred_mask = pred_mask.pred_masks.squeeze(1).squeeze(1)
        loss = self.loss_fn(pred_mask, gt)
        
        # Logs
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True); 
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        """
        
        This function gets several parameters and conducts training step for a single batch.
        
        Parameters:
        
            batch      - a single batch of the dataloader, batch object;
            batch_idx  - index of the abovementioned batch, int.
            
        Output:
        
            loss       - loss value for the particular mini-batch with images, tensor.
            
        """
        
        # Get images and their corresponding labels
        im, gt, bboxes = batch["pixel_values"], batch["ground_truth_mask"].float(), batch["input_boxes"]
        pred_mask = self.model(pixel_values = im, input_boxes = bboxes, multimask_output = False)
        
        if self.ds_name in ["mri", "isic"]:
            pred_mask = pred_mask.pred_masks.squeeze(1)
            gt = gt.unsqueeze(1)
        
        else: pred_mask = pred_mask.pred_masks.squeeze(1).squeeze(1)
        loss = self.loss_fn(pred_mask, gt)
        
        # Logs
        self.log("valid_loss", loss, on_step = False, on_epoch = True, logger = True); 
        
        return loss
    
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("valid_time", self.validation_elapsed_time, prog_bar = True, on_step = False, on_epoch = True, logger = True)
    
    def get_stats(self): return self.train_times, self.validation_times
    
class ImagePredictionLogger(Callback):
    
    def __init__(self, val_samples, ds_name, cls_names=None, num_samples = 8):
        super().__init__()
        self.num_samples, self.ds_name = num_samples, ds_name
        self.val_imgs, self.val_masks = val_samples["pixel_values"], val_samples["ground_truth_mask"]
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_masks = self.val_masks.to(device=pl_module.device).float()
        # val_masks = self.val_masks.to(device=pl_module.device)
        # Get model prediction
        if self.ds_name in ["mri", "isic"]:
            # pred_masks = (torch.sigmoid(pl_module(val_imgs).pred_masks.squeeze()) > 0.5).float()
            pred_masks = pl_module(val_imgs).pred_masks.squeeze()
            preds = [(torch.sigmoid(pred_mask) > 0.5).float() for pred_mask in pred_masks]
        else: pred_masks = pl_module(val_imgs).pred_masks.squeeze().float()
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "Input Images": [wandb.Image(x, caption="Input image") for x in val_imgs[:self.num_samples]], 
            "Ground Truth": [wandb.Image(y, caption="Ground Truth") for y in val_masks[:self.num_samples]],
            "Generated Masks": [wandb.Image(pred_mask, caption="Generated Mask") for pred_mask in preds[:self.num_samples]]
             })
