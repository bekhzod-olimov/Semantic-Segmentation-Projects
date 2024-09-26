# Import library
import albumentations as A

def get_transformations(size):

    """
    
    This function gets input image size and returns train and test transformations to be applied.

    Parameter:

        size      - input image size, int.

    Output:

        out       - train and test transformations, list.
    
    """

    return [A.Compose([A.Resize(size, size), A.HorizontalFlip(0.5),
                      A.VerticalFlip(0.5),   A.GaussNoise(0.2)]),
            A.Compose([A.Resize(size, size)])]
