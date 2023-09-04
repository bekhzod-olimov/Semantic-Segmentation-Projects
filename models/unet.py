# Import libraries
import torch, torch.nn.functional as F, torch.nn as nn
from typing import List

class UNetBlock(nn.Module):
    
    """
    
    This class gets several parameters and formulates a convolution block of UNet model.
    
    Parameters:
    
        in_chs   - number of channels of the input volume, int;
        out_chs  - number of channels of the output volume, int;
        ks       - kernel size of the convolution operation, int;
        p        - padding value for the convolution operation, int.
        
    Output:
    
        out      - output volume from a convolution block of UNet, tensor.
    
    """

    def __init__(self, in_chs: int, out_chs: int, ks: int = 3, p: int = 1):
        super().__init__()
        
        # Get kernel size and padding value
        self.ks, self.p = ks, p
        
        # Initialize the first and the second convolution blocks
        self.block_1 = self.get_conv_block(in_chs = in_chs, out_chs = out_chs)
        self.block_2 = self.get_conv_block(in_chs = out_chs, out_chs = out_chs)

    def get_conv_block(self, in_chs: int, out_chs: int):
        
        """
        
        This function gets several parameters and returns a convolution block of UNet.
        
        Parameters:
        
            in_chs   - number of channels of the input volume, int;
            out_chs  - number of channels of the output volume, int.
            
       Output:
       
           a convolution block of UNet, torch sequential model.
        
        """
        
        return nn.Sequential(
               nn.Conv2d(in_channels = in_chs, out_channels = out_chs, kernel_size = self.ks, padding = self.p),
               nn.BatchNorm2d(out_chs), 
               nn.ReLU(inplace = True)
                            )
    
    # Feed forward of the class
    def forward(self, inp): return self.block_2(self.block_1(inp))

class DownSampling(nn.Module):

    """
    
    This class gets several parameters and conducts downsampling operation.

    Parameters:

        in_chs      - channels of the input volume to the class, int;
        out_chs     - channels of the output volume from the first convolution layer, int.

    Output:

        out        - downsampled output, tensor.
    
    """
    
    def __init__(self, in_chs, out_chs):
        super().__init__()
        
        # Initialize downsampling block
        self.downsample_block = nn.Sequential(  nn.MaxPool2d(2), UNetBlock(in_chs, out_chs) )

    # Feed forward of the class
    def forward(self, x): return self.downsample_block(x)

class UpSampling(nn.Module):

    """
    
    This class gets several parameters and conducts upsampling operation in the decoder.

    Parameters:

        in_chs      - channels of the input volume to the class, int;
        out_chs     - channels of the output volume from the first convolution layer, int;
        mode        - upsampling method type, str;
        upsample    - whether or not to upsample, bool.

    Output:

        out         - upsampled output, tensor.
    
    """

    def __init__(self, in_chs, out_chs, mode, upsample = None):
        super().__init__()
        
        # Choose upsampling method type
        if mode in ["bilinear", "nearest"]: 
            upsample = True; up_mode = mode
        # Initialize upsampling layer
        self.upsample = nn.Upsample(scale_factor = 2, mode = up_mode) if upsample else nn.ConvTranspose2d(in_chs, in_chs // 2, kernel_size = 2, stride = 2)
        # Initialize convolution block to apply after upsampling
        self.conv = UNetBlock(in_chs, out_chs)

    def forward(self, inp1, inp2):

        """

        This function gets two input tensors and performs one upsampling block of the decoder.

        Parameter:

            inp1      - the first input volume, tensor;
            inp2      - the second input volume, tensor.

        Output:

            out       - upsampled volume with skip connection applied, tensor.
        
        """
        
        # Upsample the first input
        inp1 = self.upsample(inp1)
        # Padding
        pad_y = inp2.size()[2] - inp1.size()[2]; pad_x = inp2.size()[3] - inp1.size()[3]
        inp1 = F.pad(inp1, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2])
        
        return self.conv(torch.cat([inp2, inp1], dim = 1))

class FinalConv(nn.Module):

    """

    This class gets several parameters and conducts the final convolution layer of the UNet model.

    Parameters:

        in_chs      - number of channels of the input volume, int;
        out_chs     - number of channels of the output volume, int.
    
    """
    
    def __init__(self, in_chs, out_chs):
        super().__init__() 
        # Initialize the convolution layer
        self.conv = nn.Conv2d(in_channels = in_chs, out_channels = out_chs, kernel_size = 1)

    # Feed forward of the class
    def forward(self, inp): return self.conv(inp)

class UNet(nn.Module):

    """

    This class get several parameters and formulates UNet model.

    Parameters:

        in_chs      - number of channels of the input volume, int;
        out_chs     - number of channels of the output volume, int;
        n_cls       - number of classes in the dataset, int;
        depth       - depth of the UNet model;
        up_method   - upsampling method in the decoder part of the UNet, str.
    
    """
    
    def __init__(self, in_chs, n_cls, out_chs, depth, up_method):
        super().__init__()
        
        assert up_method in ["bilinear", "nearest", "tr_conv"], "Please choose a proper method for upsampling"
        # Get the arguments of the parameters
        self.depth = depth
        # Initialize the first convolution block
        self.init_block = UNetBlock(in_chs, out_chs)
        # Set the factor for upsampling
        factor = 2 if up_method in ["bilinear", "nearest"] else 1 

        # Initialize lists for the encoder and decoder
        encoder, decoder = [], []

        # Go through the number of depths
        for idx, (enc, dec) in enumerate(zip(range(depth), reversed(range(depth)))):
            
            # Get the number of encoder input channels
            enc_in_chs = out_chs * 2 ** enc # 64, 128; 128,256; 256,512; 512,512
            # Set the number of encoder output channels
            enc_out_chs = 2 * out_chs * 2 ** (enc - 1) if (idx == (depth - 1) and up_method in ["bilinear", "nearest"]) else 2 * out_chs * 2 ** enc
            # Formulate the encoder list of the model
            encoder += [DownSampling(enc_in_chs, int(enc_out_chs))]
            # Get the number of decoder input channels
            dec_in_chs = 2 * out_chs * 2 ** dec if idx == 0 else dec_out_chs
            # Get the number of decoder output channels
            dec_out_chs = out_chs * 2 ** dec if idx != (depth - 1) else factor * out_chs * 2 ** dec
            # Formulate the decoder list of the model
            decoder += [UpSampling(dec_in_chs, dec_out_chs // factor, up_method)]
        
        # Formulate encoder and decoder parts of the UNet
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        # Initialize final convolution layer of the UNet
        self.final_conv = FinalConv(out_chs, n_cls)
        # self.out_act = torch.nn.SoftMax() if n_cls > 2 else torch.nn.Sigmoid()
        
    def forward(self, inp):

        """

        This function gets an input image and conducts feed forward of the UNet.

        Parameter:

            inp    - input volume, tensor.

        Output:

            out    - output volume, tensor.
        
        """
        
        # Get the outputs from the first UNet block
        outputs = [self.init_block(inp)]

        # Get the encoded features
        for idx, block in enumerate(self.encoder):
            out = outputs[idx] if idx == 0 else encoded
            encoded = block(out)
            outputs.append(encoded)

        # Get the decoded features
        for idx, block in enumerate(self.decoder):
            encoded = outputs[self.depth - idx - 1]
            decoded = decoded if idx != 0 else outputs[-(idx + 1)]
            decoded = block(decoded, encoded)
            
        # Get the final predicted mask
        out = self.final_conv(decoded)
        
        return out
        # return self.out_act(out)
        
# inp = torch.rand(1,3,224,224)
# m = UNet(in_chs = 3, n_cls = 2, out_chs = 32, depth = 5, up_method = "tr_conv")
# m = UNet(in_chs = 3, n_cls = 2, out_chs = 32, depth = 5, up_method = "nearest")
# print(m(inp).shape)
