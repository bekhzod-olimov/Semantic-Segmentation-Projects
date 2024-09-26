# Import libraries
import torch, torchvision, os, cv2, albumentations as A
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as tfs
from transformations import get_transformations
from glob import glob; from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(2024)

class CustomSegmentationDataset(Dataset):

    """
    
    This class gets several parameters and returns custom segmentation dataset.

    Parameters:

        ds_name          - name of the dataset, str;
        transformations  - image transformations to be applied, albumentations object;
        im_files         - valid image file extensions, list -> str.

    Output:

        ds               - dataset with the images from the root, torch dataset object.
    
    """
        
    # Initialization
    def __init__(self, ds_name, transformations = None, im_files = [".jpg", ".png", ".jpeg"]):

        # Get parameter arguments
        self.transformations = transformations
        # Set the function to switch to tensor
        self.tensorize = tfs.Compose([tfs.ToTensor()]) 
        
        # Set the root path based on the dataset name
        root = "/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation/cells" if ds_name == "cells" else ("/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation/flood" if ds_name == "flood" else "/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation/drone")
        # Set the threshold based on the dataset name
        self.threshold = 11 if ds_name == "drone" else 128
        
        # Get images and gts paths
        self.im_paths = sorted(glob(f"{root}/images/*[{im_file for im_file in im_files}]")); self.gt_paths = sorted(glob(f"{root}/masks/*[{im_file for im_file in im_files}]"))
    
    # Set the length of the dataset
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):

    """

    This function gets an index and returns an image and gt pair.

    Parameter:

        idx    - index of the data in the dataset, int.

    Outputs:

        im     - an image, tensor;
        gt     - gt mask of the image, tensor.
    
    """
        # Read an image and its corresponding mask
        im, gt = cv2.cvtColor(cv2.imread(self.im_paths[idx]), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(self.gt_paths[idx]), cv2.COLOR_BGR2GRAY)
        # Apply transformations to the image and gt mask
        if self.transformations is not None: 
            # Get transformed transformations object (dictionary)
            transformed = self.transformations(image = im, mask = gt)
            # Get the transformed image and mask
            im, gt = transformed["image"], transformed["mask"]
            
        return self.tensorize(im), torch.tensor(gt > self.threshold).long()
    
def get_dl(ds_name, transformations, bs, split = [0.7, 0.15, 0.15]):

    """

    This function gets several arguments and return dataloaders.

    Parameters:

        ds_name           - name of the dataset, str;
        transformations   - transforms to be applied, albumentations object;
        bs                - mini batch size, int;
        split             - size to split the dataset, list -> float.

    Outputs:

        tr_dl             - train dataloader, torch dataloader object;
        val_dl            - validation dataloader, torch dataloader object;
        test_dl           - test dataloader, torch dataloader object. 
   
    """
        
    assert sum(split) == 1., "Data split sum must be equal to 1"
    
    # Get dataset for training
    ds = CustomSegmentationDataset(ds_name = ds_name, transformations = transformations)
    
    # Get length for train, validation, and test sets
    tr_len = int(len(ds) * split[0]); val_len = int(len(ds) * split[1]); test_len = len(ds) - (tr_len + val_len)
    
    # Data split
    tr_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [tr_len, val_len, test_len])
        
    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set")
    print(f"There are {len(test_ds)} number of images in the test set\n")
    
    # Get dataloaders
    tr_dl  = DataLoader(dataset = tr_ds, batch_size = bs, shuffle = True, num_workers = 8)
    val_dl = DataLoader(dataset = val_ds, batch_size = bs, shuffle = False, num_workers = 8)
    test_dl = DataLoader(dataset = test_ds, batch_size = 1, shuffle = False, num_workers = 8)
    
    return tr_dl, val_dl, test_dl

# Sample run
# ts = get_transformations(224)[1]
# tr_dl, val_dl, test_dl = get_dl(ds_name = "flood", transformations = ts, bs = 2)
