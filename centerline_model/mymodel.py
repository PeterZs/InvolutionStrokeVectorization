import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

class UpsampleBlock(nn.Module):
    """
    Upsampling performed via Pixel Shuffle as described:
    (C, H, W) -> Pixel Shuffle -> (C/4, 2H, 2W) -> 3x3 Conv -> (C_out, 2H, 2W)
    
    Note: To achieve specific output channels matching the skip connection, 
    we first project the input to (out_channels * 4) so PixelShuffle results in out_channels.
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        
        # We need the result of PixelShuffle to be 'out_channels' (to match skip connection).
        # Since PixelShuffle(upscale_factor=2) divides channels by 4, 
        # we must expand input to (out_channels * 4) first.
        self.expand_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels * 4)
        
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.conv_smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_smooth = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_expand(self.expand_conv(x)))
        x = self.pixel_shuffle(x)
        x = self.relu(self.bn_smooth(self.conv_smooth(x)))
        return x

# Helper for ResNet/ResNext Blocks
# torchvision Bottleneck: output_channels = planes * 4
def make_layer(in_ch, target_ch, block_type='resnet', blocks=1):
    layers = []
    
    # Standard ResNet: groups=1, base_width=64
    # ResNeXt: groups=32, base_width=4
    groups = 32 if block_type == 'resnext' else 1
    base_width = 4 if block_type == 'resnext' else 64
    
    # The Bottleneck block expands internal planes by 4. 
    # To get 'target_ch' as output, internal planes must be target_ch // 4.
    bneck_planes = target_ch // 4
    
    for i in range(blocks):
        cur_in = in_ch if i == 0 else target_ch
        
        # If dimensions change, we need a projection (downsample) layer
        downsample = None
        if cur_in != target_ch:
            downsample = nn.Sequential(
                nn.Conv2d(cur_in, target_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(target_ch),
            )
        
        layers.append(Bottleneck(
            inplanes=cur_in,
            planes=bneck_planes,
            stride=1,
            downsample=downsample,
            groups=groups,
            base_width=base_width
        ))
        
    return nn.Sequential(*layers)

class ResNextUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResNextUNet, self).__init__()
        self.img_size = None
       

        # --- Initial Conv ---
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # --- Encoder (Downsampling Path) ---
        
        # Level 1: 64 channels, ResNeXt
        self.enc1 = make_layer(64, 64, block_type='resnext')
        # Down 1: 64->96
        self.down1 = nn.Conv2d(64, 96, kernel_size=2, stride=2, bias=False)
        
        # Level 2: 96 channels, ResNet
        self.enc2 = make_layer(96, 96, block_type='resnet')
        # Down 2: 96->96
        self.down2 = nn.Conv2d(96, 96, kernel_size=2, stride=2, bias=False)
        
        # Level 3: 96 channels, ResNet
        self.enc3 = make_layer(96, 96, block_type='resnet')
        # Down 3: 96->192
        self.down3 = nn.Conv2d(96, 192, kernel_size=2, stride=2, bias=False)

        # --- Center (Bottleneck) ---
        # 3 ResNet blocks
        self.center = make_layer(192, 192, block_type='resnet', blocks=3)

        # --- Decoder (Upsampling Path) ---
        # Logic: Upsample -> Add Skip -> Process with ResNet
        
        # Up 1: 192 -> 96 (Matches Skip Enc3)
        self.up1 = UpsampleBlock(192, 96)
        self.dec1 = make_layer(96, 96, block_type='resnet') # Input is 96 (after add)
        
        # Up 2: 96 -> 96 (Matches Skip Enc2)
        self.up2 = UpsampleBlock(96, 96)
        self.dec2 = make_layer(96, 96, block_type='resnet')
        
        # Up 3: 96 -> 64 (Matches Skip Enc1)
        self.up3 = UpsampleBlock(96, 64)
        # The input to this block is 64. The output needs to be 96 to feed the final head.
        self.dec3 = make_layer(64, 96, block_type='resnet') 

        # --- Output Head ---
        self.final_block = make_layer(96, 96, block_type='resnext')
        self.final_conv = nn.Conv2d(96, out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        # Input
        x = self.conv_in(x)         # (B, 64, H, W)
        
        # Encoder 1
        e1 = self.enc1(x)           # (B, 64, H, W) -> Skip
        x = self.down1(e1)          # (B, 96, H/2, W/2)
        
        # Encoder 2
        e2 = self.enc2(x)           # (B, 96, H/2, W/2) -> Skip
        x = self.down2(e2)          # (B, 96, H/4, W/4)
        
        # Encoder 3
        e3 = self.enc3(x)           # (B, 96, H/4, W/4) -> Skip
        x = self.down3(e3)          # (B, 192, H/8, W/8)
        
        # Center
        x = self.center(x)          # (B, 192, H/8, W/8)
        
        # Decoder 1 (Lowest Res)
        x = self.up1(x)             # (B, 96, H/4, W/4)
        x = x + e3                  # Add Skip
        x = self.dec1(x)            # (B, 96, H/4, W/4)
        
        # Decoder 2
        x = self.up2(x)             # (B, 96, H/2, W/2)
        x = x + e2                  # Add Skip
        x = self.dec2(x)            # (B, 96, H/2, W/2)
        
        # Decoder 3
        x = self.up3(x)             # (B, 64, H, W)
        x = x + e1                  # Add Skip
        x = self.dec3(x)            # (B, 96, H, W) - Channel counts expand here
        
        # Output Head
        x = self.final_block(x)     # (B, 96, H, W)
        x = self.final_conv(x)      # (B, 1, H, W)
        x = self.out_activation(x)
        
        return x
    


class ResNextUNetLarge(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ResNextUNetLarge, self).__init__()
        self.img_size = None

        # --- Initial Conv ---
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # --- Encoder (Downsampling Path) ---
        
        # Level 1: 64 channels
        self.enc1 = make_layer(64, 64, block_type='resnext', blocks=2)
        # Down 1: 64 -> 128
        self.down1 = nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False)
        
        # Level 2: 128 channels
        self.enc2 = make_layer(128, 128, block_type='resnet', blocks=2)
        # Down 2: 128 -> 256
        self.down2 = nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False)
        
        # Level 3: 256 channels
        self.enc3 = make_layer(256, 256, block_type='resnet', blocks=2)
        # Down 3: 256 -> 512
        self.down3 = nn.Conv2d(256, 512, kernel_size=2, stride=2, bias=False)

        # Level 4 (NEW): 512 channels - Expands the Receptive Field to H/16
        self.enc4 = make_layer(512, 512, block_type='resnet', blocks=2)
        # Down 4: 512 -> 1024
        self.down4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False)

        # --- Center (Bottleneck) ---
        # Deeper center block (4 blocks) at the lowest resolution (H/16)
        self.center = make_layer(1024, 1024, block_type='resnet', blocks=4)

        # --- Decoder (Upsampling Path) ---
        
        # Up 4 (NEW): 1024 -> 512 (Matches Skip Enc4)
        self.up4 = UpsampleBlock(1024, 512)
        self.dec4 = make_layer(512, 512, block_type='resnet', blocks=2)

        # Up 3: 512 -> 256 (Matches Skip Enc3)
        self.up3 = UpsampleBlock(512, 256)
        self.dec3 = make_layer(256, 256, block_type='resnet', blocks=2)
        
        # Up 2: 256 -> 128 (Matches Skip Enc2)
        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = make_layer(128, 128, block_type='resnet', blocks=2)
        
        # Up 1: 128 -> 64 (Matches Skip Enc1)
        self.up1 = UpsampleBlock(128, 64)
        # Channel counts expand here to 96 to match the final head
        self.dec1 = make_layer(64, 96, block_type='resnet', blocks=2) 

        # --- Output Head ---
        self.final_block = make_layer(96, 96, block_type='resnext', blocks=2)
        self.final_conv = nn.Conv2d(96, out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):

        # Input
        x = self.conv_in(x)         # (B, 64, H, W)
        
        # Encoder 1
        e1 = self.enc1(x)           # (B, 64, H, W) -> Skip
        x = self.down1(e1)          # (B, 128, H/2, W/2)
        
        # Encoder 2
        e2 = self.enc2(x)           # (B, 128, H/2, W/2) -> Skip
        x = self.down2(e2)          # (B, 256, H/4, W/4)
        
        # Encoder 3
        e3 = self.enc3(x)           # (B, 256, H/4, W/4) -> Skip
        x = self.down3(e3)          # (B, 512, H/8, W/8)

        # Encoder 4 (New Level)
        e4 = self.enc4(x)           # (B, 512, H/8, W/8) -> Skip
        x = self.down4(e4)          # (B, 1024, H/16, W/16)
        
        # Center
        x = self.center(x)          # (B, 1024, H/16, W/16)
        
        # Decoder 4 (New Level)
        x = self.up4(x)             # (B, 512, H/8, W/8)
        x = x + e4                  # Add Skip
        x = self.dec4(x)            # (B, 512, H/8, W/8)

        # Decoder 3
        x = self.up3(x)             # (B, 256, H/4, W/4)
        x = x + e3                  # Add Skip
        x = self.dec3(x)            # (B, 256, H/4, W/4)
        
        # Decoder 2
        x = self.up2(x)             # (B, 128, H/2, W/2)
        x = x + e2                  # Add Skip
        x = self.dec2(x)            # (B, 128, H/2, W/2)
        
        # Decoder 1
        x = self.up1(x)             # (B, 64, H, W)
        x = x + e1                  # Add Skip
        x = self.dec1(x)            # (B, 96, H, W)
        
        # Output Head
        x = self.final_block(x)     # (B, 96, H, W)
        x = self.final_conv(x)      # (B, 1, H, W)
        x = self.out_activation(x)
        
        return x