# Import libraries
import torch, wandb, argparse, yaml, os, pickle, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from datasets import get_dl; from time import time
from transformations import get_transformations
from model import LitModel, ImagePredictionLogger

def run(args):
    
    """
    
    This function runs the main script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    # wandb login
    os.system("wandb login --relogin 3204eaa1400fed115e40f43c7c6a5d62a0867ed1")
    # Make directories 
    os.makedirs(args.dls_dir, exist_ok = True); os.makedirs(args.stats_dir, exist_ok = True)
        
    # Get transformations
    tfs = get_transformations(args.inp_im_size)[1]
    
    # Get train, validation, and test dataloaders
    tr_dl, val_dl, test_dl = get_dl(ds_name = args.dataset_name, transformations = tfs, bs = args.batch_size)
    
    # Save
    torch.save(tr_dl,   f"{args.dls_dir}/{args.dataset_name}_tr_dl"); torch.save(val_dl,  f"{args.dls_dir}/{args.dataset_name}_val_dl"); torch.save(test_dl, f"{args.dls_dir}/{args.dataset_name}_test_dl")
    
    # Load train and validation dataloaders
    tr_dl, val_dl = torch.load(f"{args.dls_dir}/{args.dataset_name}_tr_dl"), torch.load(f"{args.dls_dir}/{args.dataset_name}_val_dl")
    
    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_dl))
    val_imgs, val_labels = val_samples[0], val_samples[1]

    # Get model to be trained
    model = LitModel(in_chs = 3, out_chs = 32, model_name = args.model_name, n_cls = 2, up_method = "bilinear", lr = args.learning_rate) 

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='sem_segmentation', job_type='train', name=f"{args.dataset_name}_{args.model_name}_{args.batch_size}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs = args.epochs, accelerator="gpu", devices = args.devices, strategy = "ddp", logger = wandb_logger,
                         callbacks = [EarlyStopping(monitor = 'valid_loss', mode = 'min', patience=5), ImagePredictionLogger(val_samples),
                                      ModelCheckpoint(monitor = 'valid_loss', dirpath = args.save_model_path, filename = f'{args.model_name}_{args.dataset_name}_best')])

    # Get train start time
    start_time = time()
    # Start training
    trainer.fit(model, tr_dl, val_dl)
    # Get model stats
    train_times, valid_times = model.get_stats()
    # Save
    torch.save(train_times, f"{args.stats_dir}/pl_train_times_{args.devices}_gpu")
    torch.save(valid_times[1:], f"{args.stats_dir}/pl_valid_times_{args.devices}_gpu")

    # Close wandb run
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Training Process Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-bs", "--batch_size", type = int, default = 8, help = "Mini-batch size")
    parser.add_argument("-is", "--inp_im_size", type = int, default = 320, help = "Input image size")
    parser.add_argument("-dn", "--dataset_name", type = str, default = 'drone', help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = 'unet', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 1, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 20, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = "saved_models", help = "Path to the directory to save a trained model")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
