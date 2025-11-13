#!/usr/bin/env python3
"""
Complete working training script for Image Steganography
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import (StructureGenerator, TextureExtractor, 
                            StructureExtractor, ImageGenerator, 
                            Discriminator, SecretExtractor)
from utils.carrier_noise import CarrierSecretTensor

print("=" * 70)
print("IMAGE STEGANOGRAPHY - TRAINING SCRIPT")
print("=" * 70)
print("Imports successful!\n")

class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = list(self.root_dir.glob('**/*.jpg')) + \
                     list(self.root_dir.glob('**/*.png'))
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 256, 256)

class StegoTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize models
        print("📦 Initializing models...")
        self.G_stru = StructureGenerator(noise_dim=config['noise_dim']).to(self.device)
        self.texture_extractor = TextureExtractor().to(self.device)
        self.structure_extractor = StructureExtractor().to(self.device)
        self.G = ImageGenerator().to(self.device)
        self.D_real = Discriminator().to(self.device)
        self.E_x = SecretExtractor(noise_dim=config['noise_dim']).to(self.device)
        print("✓ Models initialized")
        
        # Initialize carrier noise generator
        self.carrier = CarrierSecretTensor(
            sigma=config['sigma'],
            delta=config['delta'],
            noise_dim=config['noise_dim']
        )
        print("✓ Carrier noise generator initialized")
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.G_stru.parameters()) + list(self.G.parameters()),
            lr=config['lr_g'], betas=(0.5, 0.999)
        )
        
        self.optimizer_D = optim.Adam(
            self.D_real.parameters(),
            lr=config['lr_d'], betas=(0.5, 0.999)
        )
        
        self.optimizer_E = optim.Adam(
            self.E_x.parameters(),
            lr=config['lr_e'], betas=(0.5, 0.999)
        )
        print("✓ Optimizers initialized\n")
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def train_epoch(self, dataloader, epoch, total_epochs):
        """Train for one epoch"""
        self.G_stru.train()
        self.G.train()
        self.D_real.train()
        self.E_x.train()
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_e_loss = 0
        g_updates = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch:02d}/{total_epochs:02d}')
        
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Generate random messages
            messages = [f"Msg{batch_idx:04d}_{i:02d}" for i in range(batch_size)]
            
            # Generate carrier secret tensors
            Z_primes = []
            for msg in messages:
                Z_prime = self.carrier.generate_carrier_secret_tensor(msg)
                Z_primes.append(Z_prime)
            Z_primes = torch.stack(Z_primes).to(self.device)
            
            # Extract structure and texture from real images
            with torch.no_grad():
                S1 = self.structure_extractor(real_images)
                T1 = self.texture_extractor(real_images)
            
            # Generate structure from carrier secret tensor
            S2 = self.G_stru(Z_primes)
            
            # Sample random texture
            T2 = torch.randn_like(T1)
            
            # Generate steganography image
            X3 = self.G(S2, T2)
            
            # ===== Train Discriminator =====
            self.optimizer_D.zero_grad()
            
            real_output = self.D_real(real_images)
            fake_output = self.D_real(X3.detach())
            
            # Wasserstein loss
            d_loss = torch.mean(fake_output) - torch.mean(real_output)
            
            d_loss.backward()
            self.optimizer_D.step()
            
            # Weight clipping
            self.D_real.clip_weights(self.config['clip_value'])
            
            epoch_d_loss += d_loss.item()
            
            # ===== Train Generator =====
            if batch_idx % self.config['n_critic'] == 0:
                self.optimizer_G.zero_grad()
                
                S2 = self.G_stru(Z_primes)
                X3 = self.G(S2, T2)
                
                fake_output = self.D_real(X3)
                g_loss = -torch.mean(fake_output)
                
                g_loss.backward()
                self.optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
                g_updates += 1
            
            # ===== Train Extractor =====
            self.optimizer_E.zero_grad()
            
            Z_extracted = self.E_x(X3.detach())
            e_loss = self.mse_loss(Z_extracted, Z_primes)
            
            e_loss.backward()
            self.optimizer_E.step()
            
            epoch_e_loss += e_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_loss.item():.3f}' if batch_idx % self.config['n_critic'] == 0 else '-',
                'E': f'{e_loss.item():.4f}'
            })
        
        return {
            'd_loss': epoch_d_loss / len(dataloader),
            'g_loss': epoch_g_loss / max(1, g_updates),
            'e_loss': epoch_e_loss / len(dataloader)
        }
    
    def save_models(self, save_dir, epoch):
        """Save model checkpoints"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'G_stru': self.G_stru.state_dict(),
            'G': self.G.state_dict(),
            'D_real': self.D_real.state_dict(),
            'E_x': self.E_x.state_dict(),
            'texture_extractor': self.texture_extractor.state_dict(),
            'structure_extractor': self.structure_extractor.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as "latest"
        latest_path = save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        print(f"✓ Saved: {checkpoint_path.name}")

def main():
    print("📋 Configuration:")
    config = {
        'noise_dim': 256,
        'sigma': 3,
        'delta': 25,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'lr_e': 0.0002,
        'batch_size': 4,
        'num_epochs': 50,
        'n_critic': 3,
        'clip_value': 0.1,
        'dataset': 'data/bedrooms/samples'
    }
    
    for key, value in config.items():
        print(f"  • {key:15s}: {value}")
    print()
    
    # Check if dataset exists
    if not Path(config['dataset']).exists():
        print(f"❌ Error: Dataset directory not found: {config['dataset']}")
        print("   Please run: python download_datasets.py")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    print("📂 Loading dataset...")
    try:
        dataset = ImageDataset(config['dataset'], transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        print(f"✓ Loaded {len(dataset)} images")
        print(f"✓ Created {len(dataloader)} batches\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Initialize trainer
    trainer = StegoTrainer(config)
    
    # Training loop
    print("🚀 Starting training...\n")
    best_e_loss = float('inf')
    
    try:
        for epoch in range(1, config['num_epochs'] + 1):
            losses = trainer.train_epoch(dataloader, epoch, config['num_epochs'])
            
            print(f"\n📊 Epoch {epoch:02d} Summary:")
            print(f"  • Discriminator Loss: {losses['d_loss']:.4f}")
            print(f"  • Generator Loss:     {losses['g_loss']:.4f}")
            print(f"  • Extractor Loss:     {losses['e_loss']:.4f}")
            
            # Save best model
            if losses['e_loss'] < best_e_loss:
                best_e_loss = losses['e_loss']
                print(f"  ⭐ New best model! (E_loss: {best_e_loss:.4f})")
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == 1:
                trainer.save_models('models/checkpoints', epoch)
            
            print()
        
        # Save final model
        trainer.save_models('models/checkpoints', config['num_epochs'])
        
        print("=" * 70)
        print("✅ TRAINING COMPLETED!")
        print(f"📈 Best Extractor Loss: {best_e_loss:.4f}")
        print(f"💾 Models saved in: models/checkpoints/")
        print("=" * 70)
        print("\n🎉 You can now run the web app:")
        print("   python app.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving current state...")
        trainer.save_models('models/checkpoints', epoch)
        print("✓ State saved. You can resume training later.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)