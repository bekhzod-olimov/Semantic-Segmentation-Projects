import os, torch, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from utils import get_bounding_box

class CustomDataset(Dataset):
    
    def __init__(self, root, transformations):
        
        self.transformations = transformations
        self.im_paths = sorted(glob(f"{root}/img/*.png"))
        self.gt_paths = sorted(glob(f"{root}/label/*.png"))        
 
        self.total_ims = len(self.im_paths)
        self.total_gts = len(self.gt_paths)
        
        assert self.total_ims == self.total_gts
        print(f"There are {self.total_ims} images and {self.total_gts} masks in the dataset!")  
        
    def __len__(self): return len(self.im_paths)
    
    def __getitem__(self, idx):
        
        im = Image.open(self.im_paths[idx]).convert("RGB")
        gt = cv2.resize(np.array(Image.open(self.gt_paths[idx]).convert('L')), dsize = (256, 256), interpolation = cv2.INTER_CUBIC)
        bbox = [0, 0, 256, 256]
        inputs = self.transformations(im, input_boxes = [[bbox]], return_tensors = "pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.from_numpy(gt)
        
        return inputs
    
class MRIDataset(Dataset):
    def __init__(self, root, transformations):
        
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/*/*.tif")) if "mask" not in im_path]
        self.transformations = transformations

    def __len__(self): return len(self.im_paths) // 2

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        dirname = os.path.dirname(im_path)
        gt_path = f"{dirname}/{os.path.splitext(os.path.basename(im_path))[0]}_mask.tif"
        im = Image.open(im_path)
        gt = np.array(Image.open(gt_path))
        
        prompt = get_bounding_box(gt)
        
        # prepare image and prompt for the model
        inputs = self.transformations(im, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        # inputs["ground_truth_mask"] = torch.from_numpy(gt.astype(np.int8))
        gt = torch.from_numpy(gt)
        gt[gt < 0] = 0; gt[gt > 0] = 1
        inputs["ground_truth_mask"] = gt

        return inputs
    
class ISICDataset(Dataset):
    def __init__(self, root, transformations):
        
        self.im_paths = sorted(glob(f"{root}/images/*.jpg"))
        self.gt_paths = sorted(glob(f"{root}/masks/*.png"))
        self.transformations = transformations

    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im_path, gt_path = self.im_paths[idx],  self.gt_paths[idx]
        im, gt = Image.open(im_path), cv2.resize(np.array(Image.open(gt_path)), (256, 256))
        
        bbox = get_bounding_box(gt)
        inputs = self.transformations(im, input_boxes = [[bbox]], return_tensors = "pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)
        gt = torch.from_numpy(gt)
        gt[gt < 0] = 0; gt[gt > 0] = 1
        inputs["ground_truth_mask"] = gt

        return inputs
    
def get_dls(ds_name, bs, transformations, split = [0.8, 0.1, 0.1], num_ws = 8, extension_ratio = 2):
    
    data_path = "/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation"
    root = f"{data_path}/cells_new" if "cell" in ds_name else (f"{data_path}/mri/kaggle_3m" if "mri" in ds_name else f"{data_path}/isic")
    ds = CustomDataset(root = root, transformations = transformations) if "cell" in ds_name else (MRIDataset(root = root, transformations = transformations) if "mri" in ds_name else ISICDataset(root = root, transformations = transformations))
    
    tr_len = int(len(ds) * split[0])
    val_len = int(len(ds) * split[1])
    test_len = int(len(ds) - (tr_len + val_len))

    tr_ds, val_ds, test_ds = torch.utils.data.random_split(dataset = ds, lengths = [tr_len, val_len, test_len])
    print(f"There are {len(tr_ds)} train, {len(val_ds)} validation, and {len(test_ds)} test images in the dataset!")
    
    tr_dl  = torch.utils.data.DataLoader(dataset = tr_ds, batch_size = bs * extension_ratio, shuffle = True, num_workers = num_ws)
    val_dl = torch.utils.data.DataLoader(dataset = val_ds, batch_size = bs, shuffle = True, num_workers = num_ws)
    test_dl = torch.utils.data.DataLoader(dataset = test_ds, batch_size = bs, shuffle = True, num_workers = num_ws)
    
    return tr_dl, val_dl, test_dl

# tr_dl, val_dl, test_dl = get_dls(ds_name = "isic", transformations = processor, bs = 16)

