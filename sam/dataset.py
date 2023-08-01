# Import libraries
import os, torch, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob; from PIL import Image; from utils import get_bounding_box

class CustomDataset(Dataset):

    """
    
    This class get several parameters and returns custom dataset object.

    Parameters:

        root              - path to data with images, str;
        transformations   - transformations to be applied, transforms object.

    Output:

        ds               - custom dataset, torch dataset object.
    
    """
    
    def __init__(self, root, transformations):
        
        # Get transformations
        self.transformations = transformations
        # Get images and gts paths
        self.im_paths = sorted(glob(f"{root}/img/*.png")); self.gt_paths = sorted(glob(f"{root}/label/*.png"))        

        # Get total number of images and gts
        self.total_ims = len(self.im_paths); self.total_gts = len(self.gt_paths)
        # Make sure total number of images equals to total number of gts
        assert self.total_ims == self.total_gts
        print(f"There are {self.total_ims} images and {self.total_gts} masks in the dataset!")  
        
    # Function to return total number of images in the dataset
    def __len__(self): return len(self.im_paths)
    
    def __getitem__(self, idx):

        """

        This function gets an index and returns data information.

        Parameter:

            idx     - index, int.

        Output:

            inputs  - data information, dict.
        
        """
        
        # Get an image and convert it to RGB
        im = Image.open(self.im_paths[idx]).convert("RGB")
        # Get a gt, convert it to grayscale, and resize
        gt = cv2.resize(np.array(Image.open(self.gt_paths[idx]).convert('L')), dsize = (256, 256), interpolation = cv2.INTER_CUBIC)
        # Set initial coordinates of the bounding box
        bbox = [0, 0, 256, 256]
        # Apply transformations to the image 
        inputs = self.transformations(im, input_boxes = [[bbox]], return_tensors = "pt")
        # Squueze values of the dictionary
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # Add gt mask to the dictionary
        inputs["ground_truth_mask"] = torch.from_numpy(gt)
        
        return inputs
    
class MRIDataset(Dataset):

    """
    
    This class gets several parameters and returns MRI dataset.

    Parameters:

        root             - path to data, str;
        transformations  - image transformations to be applied, transforms object.
    
    """
    
    def __init__(self, root, transformations):
        
        # Get images paths
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/*/*.tif")) if "mask" not in im_path]
        # Get the transformations to be applied
        self.transformations = transformations

    # Function to get the length of the dataset images
    def __len__(self): return len(self.im_paths) // 2

    def __getitem__(self, idx):

        """
        
        This function gets an dataset index and returns image/gt pair.

        Parameter:

            idx      - index within the dataset, int.

        Output:

            inputs  - dataset meta data information, dict. 
            
        """
        
        # Get an image path
        im_path = self.im_paths[idx]
        # Get the directory name based on the path
        dirname = os.path.dirname(im_path)
        # Set the gt path
        gt_path = f"{dirname}/{os.path.splitext(os.path.basename(im_path))[0]}_mask.tif"
        # Get an image
        im = Image.open(im_path)
        # Get a gt and transform to array
        gt = np.array(Image.open(gt_path))
        # Get the bounding box information
        prompt = get_bounding_box(gt)
        
        # Prepare image and prompt for the model
        inputs = self.transformations(im, input_boxes = [[prompt]], return_tensors = "pt")
        # Remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth segmentation (ground truth image size is 256 x 256)
        # inputs["ground_truth_mask"] = torch.from_numpy(gt.astype(np.int8))
        # Convert to tensor
        gt = torch.from_numpy(gt)
        # Ensure gt values are 0 (background) and 1 (foreground)
        gt[gt < 0] = 0; gt[gt > 0] = 1
        inputs["ground_truth_mask"] = gt

        return inputs
    
class ISICDataset(Dataset):

    """
    
    This class gets several parameters and returns ISIC dataset.

    Parameters:

        root             - path to data, str;
        transformations  - image transformations to be applied, transforms object.
    
    """
    
    def __init__(self, root, transformations):

        # Get the image paths
        self.im_paths = sorted(glob(f"{root}/images/*.jpg"))
        # Get the gt paths
        self.gt_paths = sorted(glob(f"{root}/masks/*.png"))
        # Get the transformations to be applied
        self.transformations = transformations

    # Function to get the number of images in the dataset
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):

        """
        
        This function gets an dataset index and returns image/gt pair.

        Parameter:

            idx      - index within the dataset, int.

        Output:

            inputs  - dataset meta data information, dict. 
            
        """
        
        # Get an image path and its corresponding gt mask
        im_path, gt_path = self.im_paths[idx],  self.gt_paths[idx]
        # Read the image and gt mask
        im, gt = Image.open(im_path), cv2.resize(np.array(Image.open(gt_path)), (256, 256))

        # Get bounding boxes
        bbox = get_bounding_box(gt)
        # Go through the transformations
        inputs = self.transformations(im, input_boxes = [[bbox]], return_tensors = "pt")

        # Remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add ground truth segmentation (ground truth image size is 256x256)
        gt = torch.from_numpy(gt)
        # Make sure the gt mask has 0 and 1 values only
        gt[gt < 0] = 0; gt[gt > 0] = 1
        # Creata a new key with gt value
        inputs["ground_truth_mask"] = gt

        return inputs
        
def get_dls(ds_name, bs, transformations, split = [0.8, 0.1, 0.1], num_ws = 8, extension_ratio = 2):

    """
    
    This function gets several parameters and returns dataloaders.

    Parameters:

        ds_name           - name of the dataset, str;
        bs                - batch size, int;
        transformations   - transformations to be applied, torchvision object;
        split             - values to split the dataset, list -> float;
        num_ws            - number of workers in the dataloaders, int;
        extension_ratio   - value to extend the batch size in the dataloader, int.
    
    """

    # Set the path to data with images 
    data_path = "/home/ubuntu/workspace/dataset/bekhzod/sem_segmentation"
    # Get the images paths
    root = f"{data_path}/cells_new" if "cell" in ds_name else (f"{data_path}/mri/kaggle_3m" if "mri" in ds_name else f"{data_path}/isic")
    # Get the dataset based on the dataset name
    ds = CustomDataset(root = root, transformations = transformations) if "cell" in ds_name else (MRIDataset(root = root, transformations = transformations) if "mri" in ds_name else ISICDataset(root = root, transformations = transformations))

    # Get the train set length
    tr_len = int(len(ds) * split[0])
    # Get the validation set lentgh
    val_len = int(len(ds) * split[1])
    # Get the test set length
    test_len = int(len(ds) - (tr_len + val_len))

    # Split the data into train, validation, and test sets
    tr_ds, val_ds, test_ds = torch.utils.data.random_split(dataset = ds, lengths = [tr_len, val_len, test_len])
    print(f"There are {len(tr_ds)} train, {len(val_ds)} validation, and {len(test_ds)} test images in the dataset!")

    # Get the train, validation, and test dataloaders based on the datasets
    tr_dl  = torch.utils.data.DataLoader(dataset = tr_ds, batch_size = bs * extension_ratio, shuffle = True, num_workers = num_ws)
    val_dl = torch.utils.data.DataLoader(dataset = val_ds, batch_size = bs, shuffle = True, num_workers = num_ws)
    test_dl = torch.utils.data.DataLoader(dataset = test_ds, batch_size = bs, shuffle = True, num_workers = num_ws)
    
    return tr_dl, val_dl, test_dl

# tr_dl, val_dl, test_dl = get_dls(ds_name = "isic", transformations = processor, bs = 16)
