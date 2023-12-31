# Import libraries
import torch, numpy as np, torch.nn.functional as F

class Metrics():

    """
    
    This class gets several parameters and computes evaluation metrics.

    Parameters:

        pred      - predicted mask, tensor;
        gt        - ground truth mask, tensor;
        loss_fn   - loss function, torch loss object;
        eps       - epsilon value, float;
        n_cls     - number of classes in the dataset, int.
    
    """
    
    def __init__(self, pred, gt, loss_fn, eps = 1e-10, n_cls = 2):
        
        # Get predicted mask and ground truth mask
        self.pred, self.gt = torch.argmax(F.softmax(pred, dim=1), dim = 1), gt 
        # Get loss function, epsilon value, and number of classes
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred
        
    # Make tensor contiguous
    def to_contiguous(self, inp): return inp.contiguous().view(-1)
    
    def PA(self):

        """
        
        This function computes pixel accuracy.

        Output:

            pa    - pixel accuracy score, float.
        
        """

        # Turn of gradient calculation
        with torch.no_grad():
            
            # Get matching pixels
            match = torch.eq(self.pred, self.gt).int() 
        
        return float(match.sum()) / float(match.numel())

    def mIoU(self):

        """
        
        This function computes mean intersection over union score.

        Output:

            miou  - mean intersection over union score, float.
        
        """
        
        # Turn of gradient calculation
        with torch.no_grad():
            
            # Get predicted mask and ground truth mask
            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            # Create a list to compute iou for every class
            iou_per_class = []
            
            # Go through each class
            for c in range(self.n_cls):
                
                # Get predicted class matches
                match_pred = pred == c

                # Get gt matches
                match_gt   = gt == c

                # If there is no match
                if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan)
                    
                # IoU computation
                else:
                    
                    # Compute intersection
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    # Compute union
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()
                    # Compute iou
                    iou = (intersect + self.eps) / (union + self.eps)
                    # Add to the list
                    iou_per_class.append(iou)
                    
            return np.nanmean(iou_per_class)
    
    # Compute loss value
    def loss(self): return self.loss_fn(self.pred_, self.gt) 
