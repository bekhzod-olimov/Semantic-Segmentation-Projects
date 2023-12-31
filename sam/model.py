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

    # Get optimizer to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    # Set the train process start time
    def on_train_epoch_start(self): self.train_start_time = time()

    # Compute one epoch train time
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
        # Get the prediction mask using the model
        pred_mask = self.model(pixel_values = im, input_boxes = bboxes, multimask_output = False)
        
        if self.ds_name in ["mri", "isic"]:
            pred_mask = pred_mask.pred_masks.squeeze(1)
            gt = gt.unsqueeze(1)
        
        else: pred_mask = pred_mask.pred_masks.squeeze(1).squeeze(1)
        # Compute the loss value
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
        # Get the prediction mask using the model
        pred_mask = self.model(pixel_values = im, input_boxes = bboxes, multimask_output = False)
        
        if self.ds_name in ["mri", "isic"]:
            pred_mask = pred_mask.pred_masks.squeeze(1)
            gt = gt.unsqueeze(1)
        
        else: pred_mask = pred_mask.pred_masks.squeeze(1).squeeze(1)
        # Compute the loss value
        loss = self.loss_fn(pred_mask, gt)
        
        # Logs
        self.log("valid_loss", loss, on_step = False, on_epoch = True, logger = True); 
        
        return loss

    # Set the validation process start time
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    # Compute one epoch validation time
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("valid_time", self.validation_elapsed_time, prog_bar = True, on_step = False, on_epoch = True, logger = True)

    # Get the model train and validation times
    def get_stats(self): return self.train_times, self.validation_times
    
class ImagePredictionLogger(Callback):

    """
    
    This class gets several parameters and visualizes the outputs of the validation process.

    Parameter:

        val_samples      - validation data to be used for inference and visualized, tensor;
        ds_name          - dataset name, str;
        cls_names        - class names in the dataset, list;
        num_samples      - number of samples to be visualized, int.
    
    """
    
    def __init__(self, val_samples, ds_name, cls_names = None, num_samples = 8):
        super().__init__()
        # Get the class parameter arguments
        self.num_samples, self.ds_name = num_samples, ds_name
        # Get the images and their corresponding masks
        self.val_imgs, self.val_masks = val_samples["pixel_values"], val_samples["ground_truth_mask"]
        
    def on_validation_epoch_end(self, trainer, pl_module):

        """

        This function gets several parameters and visualizes images, gts, and generated masks.

        Parameters:
        
            trainer        - trainer object, pytorch lightning trainer object;
            pl_module      - model used to generate masks, pytorch lightining model object.
        
        """
        
        # Move the tensors to device
        val_imgs = self.val_imgs.to(device = pl_module.device)
        val_masks = self.val_masks.to(device = pl_module.device).float()
        # val_masks = self.val_masks.to(device = pl_module.device)
        # Get model prediction
        if self.ds_name in ["mri", "isic"]:
            # pred_masks = (torch.sigmoid(pl_module(val_imgs).pred_masks.squeeze()) > 0.5).float()
            pred_masks = pl_module(val_imgs).pred_masks.squeeze()
            preds = [(torch.sigmoid(pred_mask) > 0.5).float() for pred_mask in pred_masks]
        else: pred_masks = pl_module(val_imgs).pred_masks.squeeze().float()
        # Log the images as wandb Image
        # Visualize original images, ground truth masks and generated masks
        trainer.logger.experiment.log({
            "Input Images": [wandb.Image(x, caption = "Input image") for x in val_imgs[:self.num_samples]], 
            "Ground Truth": [wandb.Image(y, caption = "Ground Truth") for y in val_masks[:self.num_samples]],
            "Generated Masks": [wandb.Image(pred_mask, caption="Generated Mask") for pred_mask in preds[:self.num_samples]]
             })
