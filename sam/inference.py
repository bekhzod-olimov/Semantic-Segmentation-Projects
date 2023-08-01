# Import libraries
import torch, yaml, os, pickle, timm, argparse
from utils import get_state_dict, get_preds, visualize, get_model

def run(args):
    
    """
    
    This function runs the infernce script based on the arguments.
    
    Parameter:
    
        args - parsed arguments.
        
    Output:
    
        train process.
    
    """
    
    # Get train arguments 
    argstr = yaml.dump(args.__dict__, default_flow_style = False)
    print(f"\nTraining Arguments:\n\n{argstr}")

    # Make a directory to save train results
    os.makedirs(args.save_path, exist_ok = True)

    # Get the saved test dataloader
    test_dl = torch.load(f"{args.dls_dir}/{args.ds_name}_test_dl")
    print(f"Test dataloader is successfully loaded!")
    print(f"There are {len(test_dl)} batches in the test dataloader!")

    # Get the trained model
    model = get_model().to(args.device)
    # Load params
    print("\nLoading the state dictionary...")
    state_dict = get_state_dict(f"{args.save_model_path}/{args.model_name}_{args.ds_name}_best.ckpt")
    model.load_state_dict(state_dict, strict = True)
    print(f"The {args.model_name} state dictionary is successfully loaded!\n")
    # Get images, predictions, and gts
    all_ims, all_preds, all_gts = get_preds(model, test_dl, args.device, ds_name = args.ds_name, num_bs = 5)
    # Visualization based on the inference results
    visualize(all_ims, all_preds, all_gts, num_ims = 4, rows = 2, save_path = args.save_path, save_name = f"{args.ds_name}_{args.model_name}", cmap = "gist_heat", ds_name = args.ds_name)
    
if __name__ == "__main__":
    
    # Initialize Argument Parser    
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Inference Arguments")
    
    # Add arguments to the parser
    parser.add_argument("-dn", "--ds_name", type = str, default = 'isic', help = "Dataset name for training")
    parser.add_argument("-mn", "--model_name", type = str, default = 'SAM', help = "Model name for backbone")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:3', help = "GPU device name")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vit_base_patch16_224', help = "Model name for backbone")
    # parser.add_argument("-mn", "--model_name", type = str, default = 'vgg16_bn', help = "Model name for backbone")
    parser.add_argument("-sm", "--save_model_path", type = str, default = 'saved_models', help = "Path to the directory to save a trained model")
    parser.add_argument("-sp", "--save_path", type = str, default = "results", help = "Path to dir to save inference results")
    parser.add_argument("-dl", "--dls_dir", type = str, default = "saved_dls", help = "Path to dir to save dataloaders")
    
    # Parse the added arguments
    args = parser.parse_args() 
    
    # Run the script with the parsed arguments
    run(args)
