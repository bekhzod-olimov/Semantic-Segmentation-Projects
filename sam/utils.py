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
    
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OD()
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v
    return new_state_dict

def tn2np(t, inv_fn = None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device, ds_name, num_bs = 10):
    print("Start inference...")
    
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    loss_fn = torch.nn.CrossEntropyLoss()
    start_time = time()
    for idx, batch in tqdm(enumerate(test_dl)):
        if idx == num_bs: break
        ims, gts, bboxes = batch["pixel_values"], batch["ground_truth_mask"], batch["input_boxes"].to(device)
        all_ims.extend(ims); all_gts.extend(gts);        
        pred_masks = model(pixel_values = ims.to(device), input_boxes = bboxes, multimask_output = False)
        preds = pred_masks.pred_masks.squeeze() if "cell" in ds_name else pred_masks.pred_masks.squeeze(1)
        all_preds.extend(preds)
        
    print(f"Inference is completed in {(time() - start_time):.3f} secs!")
    
    return all_ims, all_preds, all_gts

def visualize(all_ims, all_preds, all_gts, num_ims, rows, save_path, save_name, cmap, ds_name):
    
    saved_ims = sorted(glob(f"{save_path}/*.png"))
    if os.path.isfile(f"{save_path}/{save_name}_preds.png")
    print("Start visualization...")
    plt.figure(figsize = (10, 18))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    count = 1
    inv_fn = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                              T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                           ])
    
    for idx, ind in enumerate(indices):
        
        im = all_ims[ind]
        gt = all_gts[ind]
        pr = all_preds[ind] if ds_name in ["cell", "isic", "mri"] else all_preds[ind].permute(1, 2, 0)
        
        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np(im.squeeze(0), inv_fn = inv_fn), cmap = cmap)
        plt.axis("off")
        plt.title("An Input Image")
        count += 1
        
        plt.subplot(num_ims, 3, count)
        plt.imshow(tn2np(gt.unsqueeze(0)), cmap = cmap)
        plt.axis("off")
        plt.title("GT Mask")
        count += 1

        plt.subplot(num_ims, 3, count)
        plt.imshow(tensor2im((torch.sigmoid(pr) > 0.5), im_type = "gray").squeeze(0), cmap = cmap)
        plt.axis("off")
        plt.title("Generated Mask")
        count += 1
    
    plt.savefig(f"{save_path}/{save_name}_preds.png")
    print(f"The visualization can be seen in {save_path} directory.")

class Metrics():
    
    def __init__(self, pred, gt, loss_fn, eps = 1e-10, n_cls = 2):
        
        self.pred, self.gt = torch.argmax(F.softmax(pred, dim=1), dim = 1), gt 
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred
        
    def to_contiguous(self, inp): return inp.contiguous().view(-1)
    
    def PA(self):

        with torch.no_grad():
            # print(self.gt.shape)
            # print(self.pred.shape)
            
            if self.gt.shape != self.pred.shape: self.pred = self.pred_
            # print(self.gt.shape)
            # print(self.pred.shape)
            
            match = torch.eq(self.pred, self.gt).int() 
        
        return float(match.sum()) / float(match.numel())

    def mIoU(self):
        
        with torch.no_grad():
            
            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            iou_per_class = []
            
            for c in range(self.n_cls):
                
                match_pred = pred == c
                match_gt   = gt == c

                if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan)
                    
                else:
                    
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()

                    iou = (intersect + self.eps) / (union + self.eps)
                    iou_per_class.append(iou)
                    
            return np.nanmean(iou_per_class)
    
    def loss(self): return self.loss_fn(self.pred_, self.gt)

def get_model():
    
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    return model

