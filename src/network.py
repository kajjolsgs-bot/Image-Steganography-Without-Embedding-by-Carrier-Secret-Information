import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block for generator"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class StructureGenerator(nn.Module):
    """Generator for structural information from carrier secret tensor"""
    def __init__(self, noise_dim=256, img_size=256):
        super(StructureGenerator, self).__init__()
        self.img_size = img_size
        
        # Initial projection
        self.fc = nn.Linear(noise_dim, 256 * 8 * 8)
        
        # Upsampling layers to 128x128
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 32 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 64 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.final = nn.Conv2d(128, 128, 3, padding=1)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x  # Output: [B, 128, 128, 128]

class TextureExtractor(nn.Module):
    """Extract texture information from images"""
    def __init__(self):
        super(TextureExtractor, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x  # Output: [B, 128, 128, 128]

class StructureExtractor(nn.Module):
    """Extract structural information from images"""
    def __init__(self):
        super(StructureExtractor, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x  # Output: [B, 128, 128, 128]

class ImageGenerator(nn.Module):
    """Generate steganography images from structure and texture"""
    def __init__(self):
        super(ImageGenerator, self).__init__()
        
        # Combine structure and texture (both are 128 channels at 128x128)
        self.combine = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  # 128 + 128 = 256
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Upsample to 256x256
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128 -> 256
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, structure, texture):
        # Both structure and texture should be [B, 128, 128, 128]
        # Concatenate along channel dimension
        x = torch.cat([structure, texture], dim=1)  # [B, 256, 128, 128]
        x = self.combine(x)
        x = self.up1(x)  # [B, 128, 256, 256]
        x = self.final(x)  # [B, 3, 256, 256]
        return x

class Discriminator(nn.Module):
    """Discriminator with weight clipping support"""
    def __init__(self, img_size=256):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, 1, 0)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def clip_weights(self, clip_value=0.1):
        """Clip weights as per paper"""
        for p in self.parameters():
            p.data.clamp_(-clip_value, clip_value)

class SecretExtractor(nn.Module):
    """Extract carrier secret tensor from generated images"""
    def __init__(self, noise_dim=256):
        super(SecretExtractor, self).__init__()
        
        # Direct extraction from image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # Global pooling and FC
        self.fc = nn.Linear(512 * 8 * 8, noise_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x