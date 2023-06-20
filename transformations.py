import albumentations as A

def get_transformations(size):

    return [A.Compose([A.Resize(size, size),
                      A.HorizontalFlip(0.5),
                      A.VerticalFlip(0.5),
                      A.GaussNoise(0.2),
                      ]),
           
            A.Compose([A.Resize(size, size)])
           ]
