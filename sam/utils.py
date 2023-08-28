# Import libraries
import torch, random, numpy as np, torch.nn.functional as F
from torchvision import transforms as T
from transformers import SamModel; from collections import OrderedDict as OD
from time import time; from glob import glob
from tqdm import tqdm; from matplotlib import pyplot as plt

def tensor2im(t, im_type = "rgb"): 

    """

    This function gets several parameters and converts tensor data type into numpy array.

    Parameters:

        t        - input tensor variable, tensor;
        im_type  - image type, str.        

    Output:

        im       - converted image, array.
    
    """

    # Set the transformations to be applied
    rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    return ((rgb_tfs(t))*255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if im_type == "rgb" else ((t)*255).detach().cpu().numpy().astype(np.uint8)

def get_bounding_box(ground_truth_map):
    
    """
    
    This function creates varying bounding box coordinates 
    based on the segmentation contours as prompt for the SAM model.

    Parameter:

        ground_truth_map        - ground truth map to get bounding box, tensor.    

    Output:

        bbox                   - bounding box, tensor.
        
    """

    # Make sure the gt map does not have negative values
    ground_truth_map[ground_truth_map < 0] = 1
    
    # Get the bounding box when the gt map has more than 1 unique values
    if len(np.unique(ground_truth_map)) > 1:
        
        # Get bounding box from the mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        # Get x coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        # Get y coordinates
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Add perturbation to bounding box coordinates
        # Get height and width of the image
        H, W = ground_truth_map.shape
        # Get the min and max values
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))

        # Set the bounding box as list
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
        
    # If there is no mask in the array, set bbox to image size
    else: return [0, 0, 256, 256] 

def get_state_dict(checkpoint_path):

    """

    This function gets a path to the checkpoint and returns a new state dictionary.

    Parameter:

        checkpoint_path        - path to the checkpoint with a trained model, str.

    Output:

        new_state_dict         - a new state dictionary where "model." removed from the keys, dict.
    
    """
    
    # Load the checkpoint from the path
    checkpoint = torch.load(checkpoint_path)
    # Create a new dictionary
    new_state_dict = OD()
    # Go through the each dictionary key and values 
    for k, v in checkpoint["state_dict"].items():
        # Remove "model." string
        name = k.replace("model.", "")
        # Set the new key and the value
        new_state_dict[name] = v
    
    return new_state_dict

# Function to convert tensor to numpy array
def tn2np(t, inv_fn = None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device, ds_name, num_bs = 10):

    """
    
    This function gets several parameters and returns necessary metadata for the inference process.

    Parameters:

        model            -  a trained model, torch model object;
        test_dl          - test dataloader, torch dataloader object;
        device           - device type, str;
        ds_name          - name of the dataset, str;
        num_bs           - number of batches to be inferenced, int.

    Outputs:

        all_ims         - all images in the batches, tensor;
        all_preds       - all predictions of the images in the batches, tensor;
        all_gts         - all gt labels of the images in the batches, tensor.
    
    """
    
    print("Start inference...")

    # Set the lists to save information
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    # Initialize the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Set the start time
    start_time = time()
    # Go through every batch in the test dataloader
    for idx, batch in tqdm(enumerate(test_dl)):
        # If the batch index hits the number of batches variable value break the loop
        if idx == num_bs: break
        # Get the images, gts, and bounding boxes from the batch
        ims, gts, bboxes = batch["pixel_values"], batch["ground_truth_mask"], batch["input_boxes"].to(device)
        # Add the images and gts to the lists
        all_ims.extend(ims); all_gts.extend(gts);        
        # Get the predicted masks based on the images and bounding boxes
        pred_masks = model(pixel_values = ims.to(device), input_boxes = bboxes, multimask_output = False)
        # Get predicted masks
        preds = pred_masks.pred_masks.squeeze() if "cell" in ds_name else pred_masks.pred_masks.squeeze(1)
        # Add the predicted masks to the list
        all_preds.extend(preds)
        
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    
    return all_ims, all_preds, all_gts

def visualize(all_ims, all_preds, all_gts, num_ims, rows, save_path, save_name, cmap, ds_name):

    """

    This function gets several parameters and visualizes the inference results.

        all_ims            - images to be visualized, list;
        all_preds          - predicted masks to be visualized, list;
        all_gts            - ground truth masks to be visualized, list;
        num_ims            - number of images to be visualized, int;
        rows               - number of rows in the plot, int;
        save_path          - path to save the visualization results, str;
        save_name          - name to save the plot, str;
        cmap               - colormap type, str;
        ds_name            - dataset name, str.
    
    """
    
    # Get all the saved images from the given path
    saved_ims = sorted(glob(f"{save_path}/*.png"))
    # Pass if the predictions are already saved
    if os.path.isfile(f"{save_path}/{save_name}_preds.png"): pass
    
    print("Start visualization...")
    # Set the figure size
    plt.figure(figsize = (10, 18))
    # Get the random indices
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    # Set the cound
    count = 1
    # Set the inverse transformations 
    inv_fn = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                         T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]) ])
    
    # Go through every index in the random indices
    for idx, ind in enumerate(indices):

        # Get the image and gt
        im = all_ims[ind]; gt = all_gts[ind]
        # Get the predicted mask
        pr = all_preds[ind] if ds_name in ["cell", "isic", "mri"] else all_preds[ind].permute(1, 2, 0)

        # The first plot
        plt.subplot(num_ims, 3, count)
        # Show the image
        plt.imshow(tn2np(im.squeeze(0), inv_fn = inv_fn), cmap = cmap)
        # Turn of the axis
        plt.axis("off")
        # Set the title
        plt.title("An Input Image")
        # Add a count value
        count += 1

        # The second plot
        plt.subplot(num_ims, 3, count)
        # Show the image
        plt.imshow(tn2np(gt.unsqueeze(0)), cmap = cmap)
        # Turn off the axis
        plt.axis("off")
        # Set the title
        plt.title("GT Mask")
        # Add a count value
        count += 1

        # The third plot
        plt.subplot(num_ims, 3, count)
        # Show the image
        plt.imshow(tensor2im((torch.sigmoid(pr) > 0.5), im_type = "gray").squeeze(0), cmap = cmap)
        # Turn off the axis
        plt.axis("off")
        # Set the title
        plt.title("Generated Mask")
        # Add a count value
        count += 1
    
    # Save the figure
    plt.savefig(f"{save_path}/{save_name}_preds.png")
    print(f"The visualization can be seen in {save_path} directory.")

class Metrics():

    """

    This class gets several parameters and calculates evaluation metrics.

    Parameters:

        pred        - predicted mask from the model, tensor;
        gt          - ground truth mask, tensor;
        loss_fn     - loss function, torch loss object;
        eps         - epsilon value, float;
        n_cls       - number of classes in the dataset, int.
    
    """
    
    def __init__(self, pred, gt, loss_fn, eps = 1e-10, n_cls = 2):
        
        # Get predicted gt masks
        self.pred, self.gt = torch.argmax(F.softmax(pred, dim=1), dim = 1), gt 
        # Get the value of the parameters
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred
        
    # Function to convert tensor to contiguous tensor
    def to_contiguous(self, inp): return inp.contiguous().view(-1)
    
    def PA(self):

        """

        This function computes pixel accuracy evaluation metric score.
        
        """

        with torch.no_grad():
            # print(self.gt.shape)
            # print(self.pred.shape)

            # If the shapes do not match
            if self.gt.shape != self.pred.shape: self.pred = self.pred_
            # print(self.gt.shape)
            # print(self.pred.shape)

            # Get the number of matching pixels
            match = torch.eq(self.pred, self.gt).int() 
        
        return float(match.sum()) / float(match.numel())

    def mIoU(self):

        """

        This function computes mean intersection over union evaluation metric score.
        
        """
        
        with torch.no_grad():

            # Convert the pred and gt masks to contiguous tensors
            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)
            # Initialize loss to compute iou per class
            iou_per_class = []
            # Go through every class
            for c in range(self.n_cls):
                # Compute matching pixels for predicted and gt masks
                match_pred = pred == c
                match_gt   = gt == c

                # If no match with gt add nan value to the list
                if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan)
                else:
                    # Compute intersection
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    # Compute union
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()
                    # Compute iou
                    iou = (intersect + self.eps) / (union + self.eps)
                    # Add the iou score to the list
                    iou_per_class.append(iou)
            
            # Return the mIoU value                    
            return np.nanmean(iou_per_class)

    # Function to compute loss value
    def loss(self): return self.loss_fn(self.pred_, self.gt)

def get_model():

    """

    This function initializes SAM model and loads pretrained weights.
    
    """

    # Initialize the model
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # Turn off gradient calculation for a particular layers
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"): param.requires_grad_(False)
    
    return model
