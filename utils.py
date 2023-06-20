import torch, random, numpy as np
from collections import OrderedDict as OD
from time import time
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from tqdm import tqdm
from metrics import Metrics

def get_state_dict(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = OD()
    for k, v in checkpoint["state_dict"].items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v
    return new_state_dict

def tn2np(t, inv_fn=None): return (inv_fn(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_fn is not None else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def get_preds(model, test_dl, device):
    print("Start inference...")
    
    all_ims, all_preds, all_gts, acc = [], [], [], 0
    loss_fn = torch.nn.CrossEntropyLoss()
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
    
    print("Start visualization...")
    plt.figure(figsize = (5, 18))
    indices = [random.randint(0, len(all_ims) - 1) for _ in range(num_ims)]
    count = 1
    threshold = -1 if "drone" in save_name else 0.5
    
    for idx, ind in enumerate(indices):
        
        im = all_ims[ind]
        gt = all_gts[ind]
        pr = all_preds[ind]
        # print(torch.unique(pr))
        
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
    
    
    
    
    
