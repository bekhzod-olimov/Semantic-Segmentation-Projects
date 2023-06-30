# Import libraries
import torch, yaml, os, pickle, timm, argparse
from models.unet import UNet
from models.segformer import SegFormer
from utils import get_state_dict, get_preds, visualize
from models.params import get_params

def run(args):
    
    """
    
    This function runs the infernce script based on the arguments.
    
    Parameter:
    
        args - parsed arguments, argparse object.
        
    Output:
    
        train process.
    
    """
    
    assert args.dataset_name in ["flood", "cells", "drone"], "Please choose the proper dataset name"
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")
    
    # Create a directory to save results
    os.makedirs(args.save_path, exist_ok = True)
    
    # Get parameters based on the model name
    params = get_params(args.model_name)
    # Get the dataloader to test the performance of the trained model
    test_dl = torch.load(f"{args.dls_dir}/{args.dataset_name}_test_dl")
    print(f"Test dataloader is successfully loaded!")
    print(f"There are {len(test_dl)} batches in the test dataloader!")

    # Get the model to be used in the inference
    model = UNet(in_chs = params["in_chs"], 
                 n_cls = params["n_cls"], 
                 out_chs = params["out_chs"], 
                 depth = params["depth"], 
                 up_method = params["up_method"]) if args.model_name == "unet" else \
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
    # Move the model to GPU
    model = model.to(args.device)
    # Load params
    print("\nLoading the state dictionary...")
    # Get the state dictionary with the trained weight and bias parameters
    state_dict = get_state_dict(f"{args.save_model_path}/{args.model_name}_{args.dataset_name}_best.ckpt")
    # Load the trained parameters
    model.load_state_dict(state_dict, strict = True)
    print(f"The {args.model_name} state dictionary is successfully loaded!\n")
    
    # Get predictions using test dataloader and the trained model
    all_ims, all_preds, all_gts = get_preds(model, test_dl, args.device)
    
    # Visualization
    visualize(all_ims, all_preds, all_gts, num_ims = 10, rows = 2, save_path = args.save_path, save_name = f"{args.dataset_name}_{args.model_name}")
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Inference Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-dn", "--dataset_name", type = str, default = "cells", help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = "unet", help = "Model name the segmentation task")
    parser.add_argument("-d", "--device", type = str, default = "cuda:0", help = "GPU device name")
    parser.add_argument("-sm", "--save_model_path", type = str, default = "saved_models", help = "Path to the directory to save a trained model")
    parser.add_argument("-sp", "--save_path", type = str, default = "results", help = "Path to dir to save inference results")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
