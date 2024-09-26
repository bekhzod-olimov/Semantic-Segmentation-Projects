# Import libraries
import torch, cv2, random, numpy as np
from collections import OrderedDict as OD; from time import time
from pytorch_grad_cam import GradCAM; from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt;  from tqdm import tqdm; from metrics import Metrics

def get_state_dict(checkpoint_path):

    """
    
    This function gets a path to the checkpoint of the trained model and returns new state dictionary containing the trained parameters.

    Parameter:

        checkpoint_path      - path to the checkpoint of the trained model, str.

    Output:

        new_state_dict       - a new state dictionary with trained parameters, dictionary.
    
    """
    
    # Load the trained model checkpoint
    checkpoint = torch.load(checkpoint_path)
    # Initialize a new dictionary
    new_state_dict = OD()
    # Go through every key and value of the state dictionary
    for k, v in checkpoint["state_dict"].items():
        # Pytorch lightning trained weigths contain "model." prefix in the name of the trainable parameters
        # Remove "model." word in order to use model in torch
        name = k.replace("model.", "") 
        # Replace the name with a new one
        new_state_dict[name] = v
    
    return new_state_dict

# Function to convert input tensor to numpy array
def tn2np(t, inv_fn = None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device):

    """
    
    This function gets several parameters and generates segmentation masks using the trained model.

    Parameters:

        model       - a trained model, torch model object;
        test_dl     - test dataloader, torch dataloader object;
        device      - gpu device name, str.

    Outputs:

        all_ims    - all images in the test dataloader;
        all_preds  - all predicted masks using the trained model;
        all_gts    - all ground truth masks in the test dataloader.
    
    """
    
    print("Start inference...")

    # Set variables
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    # Intialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Set the start time
    start_time = time()
    for idx, batch in tqdm(enumerate(test_dl)):
        if idx == 10: break
        ims, gts = batch
        all_ims.extend(ims); all_gts.extend(gts);        
        preds = model(ims.to(device))
        if preds.shape[2] != gts.shape[2]: preds = torch.nn.functional.interpolate(input = preds, scale_factor = gts.shape[2] // preds.shape[2], mode = "bilinear")
        met = Metrics(preds, gts.to(device), loss_fn)
        acc += met.mIoU().item()
        all_preds.extend(preds)
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    print(f"Mean Intersection over Union of the model is {acc / len(test_dl.dataset):.3f}")
    
    return all_ims, all_preds, all_gts
    
def visualize(all_ims, all_preds, all_gts, num_ims, rows, save_path, save_name):

    """
    
    This function gets several arguments and visualizes results.

    Parameters:

        all_ims    - all images in the test dataloader;
        all_preds  - all predicted masks using the trained model;
        all_gts    - all ground truth masks in the test dataloader;
        num_ims    - number of images to be visualized, int;
        rows       - number of rows to be visualized, int;
        save_path  - path to save the visualization, str;
        save_name  - filename to save the visualization, str. 

    Outputs:

        .
    
    """
    
    print("Start visualization...")
    plt.figure(figsize = (5, 18))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    count = 1
    threshold = -1 if "drone" in save_name else 0.5
    
    for idx, ind in enumerate(indices):
        
        im = all_ims[ind]
        gt = all_gts[ind]
        pr = all_preds[ind]
        
        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np(im.squeeze(0)))
        plt.axis("off")
        plt.title("An Input Image")
        count += 1
        
        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np(gt.unsqueeze(0)), cmap = "gray")
        plt.axis("off")
        plt.title("GT Mask")
        count += 1

        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np((pr > threshold).squeeze(0))[:, : , 1], cmap = "gray")
        plt.axis("off")
        plt.title("Generated Mask")
        count += 1
    
    plt.savefig(f"{save_path}/{save_name}_preds.png")
    print(f"The visualization can be seen in {save_path} directory.")
    
    
    
    
    
