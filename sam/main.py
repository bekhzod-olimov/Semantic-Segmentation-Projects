# Import libraries
import torch, wandb, argparse, yaml, os, pickle, pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger; from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from dataset import get_dls; from transformers import SamProcessor
from time import time; from model import LitModel, ImagePredictionLogger

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
    
    os.system("wandb login --relogin 3204eaa1400fed115e40f43c7c6a5d62a0867ed1")
    os.makedirs(args.dls_dir, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)
    
    transformations = SamProcessor.from_pretrained("facebook/sam-vit-base")
    tr_dl, val_dl, test_dl = get_dls(ds_name = args.ds_name, bs = args.batch_size, transformations = transformations, extension_ratio = 1)
    
    torch.save(tr_dl,   f"{args.dls_dir}/{args.ds_name}_tr_dl")
    torch.save(val_dl,  f"{args.dls_dir}/{args.ds_name}_val_dl")
    torch.save(test_dl, f"{args.dls_dir}/{args.ds_name}_test_dl")
    
    tr_dl, val_dl = torch.load(f"{args.dls_dir}/{args.ds_name}_tr_dl"), torch.load(f"{args.dls_dir}/{args.ds_name}_val_dl")
    
    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_dl))

    model = LitModel(ds_name = args.ds_name, lr = args.learning_rate) 

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='sem_segmentation', job_type='train', name=f"{args.ds_name}_{args.model_name}_{args.batch_size}")

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs = args.epochs, accelerator="gpu", devices = args.devices, strategy = "ddp", logger = wandb_logger, fast_dev_run=False,
                         callbacks = [EarlyStopping(monitor = 'valid_loss', mode = 'min', patience=5), ImagePredictionLogger(val_samples, ds_name = args.ds_name),
                                      ModelCheckpoint(monitor = 'valid_loss', dirpath = args.save_model_path, filename = f'{args.model_name}_{args.ds_name}_best')])

    
    start_time = time()
    trainer.fit(model, tr_dl, val_dl)
    train_times, valid_times = model.get_stats()
    torch.save(train_times, f"{args.stats_dir}/pl_train_times_{args.devices}_gpu")
    torch.save(valid_times[1:], f"{args.stats_dir}/pl_valid_times_{args.devices}_gpu")

    # Close wandb run
    wandb.finish()
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = 'Image Classification Training Arguments')
    
    # Add arguments to the parser
    parser.add_argument("-r", "--ds_name", type = str, default = 'mri', help = "Dataset name for training")
    parser.add_argument("-bs", "--batch_size", type = int, default = 8, help = "Mini-batch size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'SAM', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vit_base_patch16_224', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vgg16_bn', help = "Model name for backbone")
    parser.add_argument("-d", "--devices", type = int, default = 4, help = "Number of GPUs for training")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 200, help = "Train epochs number")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sd", "--stats_dir", type = str, default = "stats", help = "Path to dir to save train statistics")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
