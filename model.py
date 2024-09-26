# Import libraries
import torch, torchmetrics, timm, wandb, pytorch_lightning as pl, os
from torch import nn; from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from time import time; from metrics import Metrics
from models.unet import UNet; from models.segformer import SegFormer
from models.params import get_params

class LitModel(pl.LightningModule):
    
    """"
    
    This class gets several arguments and returns a model for training.

    in_chs          - number of input channels to the model, int;
    out_chs         - number of output channels from the first convolution layer, int;
    model_name      - name of the model to be trained, str;
    n_cls           - number of classes in the dataset, int;
    up_method       - method for upsampling, str;
    lr              - learning rate value, float.
        
    """
    
    def __init__(self, in_chs, out_chs, model_name, n_cls,  up_method = "bilinear", lr = 2e-4):
        super().__init__()
        
        # Log hyperparameters
        self.save_hyperparameters()
        # Get the learning rate for optimizer
        self.lr = lr
        # Get the model parameters
        params = get_params(model_name)
        # Set the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # Get model to be trained
        self.model = UNet(in_chs = params["in_chs"], n_cls = params["n_cls"], out_chs = params["out_chs"], depth = params["depth"], up_method = params["up_method"]) if model_name == "unet" else \
        SegFormer(
                  in_channels = params["in_chs"],
                  widths = params["widths"],
                  depths = params["depths"],
                  all_num_heads = params["all_num_heads"],
                  patch_sizes = params["patch_sizes"],
                  overlap_sizes = params["overlap_sizes"],
                  reduction_ratios = params["reduction_ratios"],
                  mlp_expansions = params["mlp_expansions"],
                  decoder_channels = params["decoder_channels"],
                  scale_factors = params["scale_factors"],
                  num_classes = params["num_classes"],
                        )
        # Initialize lists to track train and validation times
        self.train_times, self.validation_times = [], []

    # Get optimizere to update trainable parameters
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr = self.lr)
        
    # Feed forward of the model
    def forward(self, inp): return self.model(inp)
    
    # Track epoch train start time
    def on_train_epoch_start(self): self.train_start_time = time()
    
    # Track epoch train finish time and log
    def on_train_epoch_end(self): self.train_elapsed_time = time() - self.train_start_time; self.train_times.append(self.train_elapsed_time); self.log("train_time", self.train_elapsed_time, prog_bar = True)
        
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
        im, gt = batch
        # Obtain predicted mask
        pred_mask = self(im)
        # Segformer case
        if pred_mask.shape[2] != gt.shape[2]: pred_mask = F.upsample(pred_mask, scale_factor = (gt.shape[2] // pred_mask.shape[2]), mode = "bilinear")
        # Get evaluation metrics values
        met = Metrics(pred_mask, gt, self.loss_fn)
        loss = met.loss()
        iou = met.mIoU()
        pa = met.PA()
        
        # Logs
        self.log("train_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("train_pa", pa, on_step = False, on_epoch = True, logger = True); 
        self.log("train_iou", iou, on_step = False, on_epoch = True, logger = True);
        
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
        im, gt = batch
        # Get predicted mask
        pred_mask = self(im)
        # Segformer case
        if pred_mask.shape[2] != gt.shape[2]: pred_mask = F.upsample(pred_mask, scale_factor = (gt.shape[2] // pred_mask.shape[2]), mode = "bilinear")
        # Get evaluation metrics values
        met = Metrics(pred_mask, gt, self.loss_fn)
        loss = met.loss()
        iou = met.mIoU()
        pa = met.PA()
        
        # Logs
        self.log("valid_loss", loss, on_step = False, on_epoch = True, logger = True); self.log("valid_pa", pa, on_step = False, on_epoch = True, logger = True); 
        self.log("valid_iou", iou, on_step = False, on_epoch = True, logger = True);
        
        return loss
    
    # Track epoch validation start time
    def on_validation_epoch_start(self): self.validation_start_time = time()
    
    # Track epoch validation finish time and log
    def on_validation_epoch_end(self): self.validation_elapsed_time = time() - self.validation_start_time; self.validation_times.append(self.validation_elapsed_time); self.log("valid_time", self.validation_elapsed_time, prog_bar = True)
    
    # Function to get train stats
    def get_stats(self): return self.train_times, self.validation_times
    
class ImagePredictionLogger(Callback):

    """
    
    This class gets several parameters and visualizes validation images with their corresponding predicted masks.
    
    Parameters:
    
        val_samples    - validation images, tensor;
        cls_names      - names of the classes in the dataset, list;
        num_samples    - number of samples to visualize, int.
        
    Output:
    
        plot           - visualized plot with input images and their corresponding predictions, logger object.
    
    """
    
    def __init__(self, val_samples, cls_names = None, num_samples = 2):
        super().__init__()
        # Get the number of images to be visualized
        self.num_samples = num_samples
        # Get images and their corresponding masks
        self.val_imgs, self.val_masks = val_samples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        
        # Move the images and masks to GPU device 
        val_imgs = self.val_imgs.to(device = pl_module.device); val_masks = self.val_masks.to(device = pl_module.device).float()
        
        # Get model predicted masks
        print(torch.unique(pl_module(val_imgs)))
        # Apply thresholding
        pred_masks = (pl_module(val_imgs) > 0.5).float()
        
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "Input Images": [wandb.Image(x, caption="Input image") for x in val_imgs[:self.num_samples]], 
            "Ground Truth": [wandb.Image(y, caption="Ground Truth") for y in val_masks[:self.num_samples]],
            "Generated Masks": [wandb.Image(pred_mask, caption="Generated Mask") for pred_mask in pred_masks[:self.num_samples]]
             })                               
