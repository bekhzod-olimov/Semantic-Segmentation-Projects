def get_params(model_name):
    
    """
    
    This function gets a semantic segmentation model name and returns parameters to initialize the model.
    
    Parameter:
    
        model_name     - a model name of the semantic segmentation model to be initialized, str.
        
    Output:
    
        out           - parameters of the model, dict.
    
    """
    
    return {"in_chs":      3,
            "out_chs":    32,
             "n_cls":      2,
             "up_method":  "bilinear",
             "depth":      5} if model_name == "unet" else \
           {"in_chs":      3,
            "widths": [64, 128, 256, 512],
            "depths": [3, 4, 6, 3],
            "all_num_heads": [1, 2, 4, 8],
            "patch_sizes": [7, 3, 3, 3],
            "overlap_sizes": [4, 2, 2, 2],
            "reduction_ratios": [8, 4, 2, 1],
            "mlp_expansions": [4, 4, 4, 4],
            "decoder_channels": 256,
            "scale_factors": [8, 4, 2, 1],
            "num_classes": 2}
