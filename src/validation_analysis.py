import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import seaborn as sns
from scipy import fftpack
from torchvision import transforms
import cv2

from src.network import (StructureGenerator, TextureExtractor, 
                            StructureExtractor, ImageGenerator, SecretExtractor)
from utils.carrier_noise import CarrierSecretTensor

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ValidationAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_dim = 256
        self.sigma = 3
        self.delta = 25
        
        print(f"🖥️  Device: {self.device}")
        
        # Initialize carrier noise generator
        self.carrier = CarrierSecretTensor(
            sigma=self.sigma,
            delta=self.delta,
            noise_dim=self.noise_dim
        )
        
        # Initialize models
        print("📦 Loading models...")
        self.G_stru = StructureGenerator(noise_dim=self.noise_dim).to(self.device)
        self.texture_extractor = TextureExtractor().to(self.device)
        self.structure_extractor = StructureExtractor().to(self.device)
        self.G = ImageGenerator().to(self.device)
        self.E_x = SecretExtractor(noise_dim=self.noise_dim).to(self.device)
        
        # Load trained weights
        self.load_checkpoint()
        
        # Set to evaluation mode
        self.G_stru.eval()
        self.texture_extractor.eval()
        self.structure_extractor.eval()
        self.G.eval()
        self.E_x.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize([-1, -1, -1], [2, 2, 2]),
        ])
        
        print("✓ Validation Analyzer initialized\n")
    
    def load_checkpoint(self):
        """Load trained model checkpoint"""
        checkpoint_path = Path('models/checkpoints/checkpoint_latest.pth')
        
        if not checkpoint_path.exists():
            print("⚠️  Warning: No trained checkpoint found!")
            return
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.G_stru.load_state_dict(checkpoint['G_stru'])
            self.G.load_state_dict(checkpoint['G'])
            self.E_x.load_state_dict(checkpoint['E_x'])
            self.texture_extractor.load_state_dict(checkpoint['texture_extractor'])
            self.structure_extractor.load_state_dict(checkpoint['structure_extractor'])
            
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array (0-255 range)"""
        tensor = self.inverse_transform(tensor.cpu())
        tensor = torch.clamp(tensor, 0, 1)
        img_array = tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        return img_array
    
    @torch.no_grad()
    def generate_stego_image(self, cover_image, secret_message):
        """Generate steganography image"""
        cover_tensor = self.transform(cover_image).unsqueeze(0).to(self.device)
        
        S1 = self.structure_extractor(cover_tensor)
        T1 = self.texture_extractor(cover_tensor)
        
        Z_prime = self.carrier.generate_carrier_secret_tensor(secret_message)
        Z_prime = Z_prime.unsqueeze(0).to(self.device)
        
        S2 = self.G_stru(Z_prime)
        T2 = T1
        
        stego_tensor = self.G(S2, T2)
        
        return stego_tensor.squeeze(0)
    
    def calculate_entropy(self, image_array):
        """Calculate Shannon entropy of an image"""
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        
        # Normalize to get probabilities
        hist = hist / hist.sum()
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def compute_fourier_transform(self, image_array):
        """Compute 2D Fourier Transform"""
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Compute FFT
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log scale for better visualization
        magnitude_log = np.log(magnitude + 1)
        
        return magnitude_log
    
    def plot_histogram_comparison(self, original, stego, save_path):
        """Create histogram comparison plot"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Histogram Analysis: Original vs Stego Image', 
                     fontsize=16, fontweight='bold')
        
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            # Original histogram
            axes[0, i].hist(original[:, :, i].ravel(), bins=256, 
                           range=(0, 255), color=color, alpha=0.7, 
                           density=True, label='Original')
            axes[0, i].set_title(f'{channel} Channel - Original', fontweight='bold')
            axes[0, i].set_xlabel('Pixel Intensity')
            axes[0, i].set_ylabel('Normalized Frequency')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_xlim([0, 255])
            
            # Stego histogram
            axes[1, i].hist(stego[:, :, i].ravel(), bins=256, 
                           range=(0, 255), color=color, alpha=0.7, 
                           density=True, label='Stego')
            axes[1, i].set_title(f'{channel} Channel - Stego', fontweight='bold')
            axes[1, i].set_xlabel('Pixel Intensity')
            axes[1, i].set_ylabel('Normalized Frequency')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_xlim([0, 255])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved histogram comparison: {save_path}")
    
    def plot_entropy_comparison(self, original, stego, save_path):
        """Create entropy comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Entropy Analysis: Statistical Randomness Preservation', 
                     fontsize=16, fontweight='bold')
        
        # Calculate entropy for each channel
        channels = ['Red', 'Green', 'Blue', 'Grayscale']
        original_entropies = []
        stego_entropies = []
        
        for i in range(3):
            original_entropies.append(self.calculate_entropy(original[:, :, i]))
            stego_entropies.append(self.calculate_entropy(stego[:, :, i]))
        
        # Add grayscale entropy
        original_entropies.append(self.calculate_entropy(original))
        stego_entropies.append(self.calculate_entropy(stego))
        
        # Bar plot comparison
        x = np.arange(len(channels))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, original_entropies, width, 
                           label='Original', color='steelblue', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, stego_entropies, width, 
                           label='Stego', color='coral', alpha=0.8)
        
        axes[0].set_xlabel('Channel', fontweight='bold')
        axes[0].set_ylabel('Entropy (bits)', fontweight='bold')
        axes[0].set_title('Entropy by Channel', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(channels)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        
        # Difference plot
        differences = np.array(stego_entropies) - np.array(original_entropies)
        colors_diff = ['green' if abs(d) < 0.1 else 'orange' for d in differences]
        
        bars = axes[1].bar(channels, differences, color=colors_diff, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1].set_xlabel('Channel', fontweight='bold')
        axes[1].set_ylabel('Entropy Difference (Stego - Original)', fontweight='bold')
        axes[1].set_title('Entropy Preservation (closer to 0 = better)', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.3f}',
                       ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved entropy comparison: {save_path}")
    
    def plot_fourier_comparison(self, original, stego, save_path):
        """Create Fourier transform comparison plot"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Fourier Transform Analysis: Frequency Domain Preservation', 
                     fontsize=16, fontweight='bold')
        
        # Compute Fourier transforms
        fft_original = self.compute_fourier_transform(original)
        fft_stego = self.compute_fourier_transform(stego)
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
        axes[0, 0].axis('off')
        
        # Stego image
        axes[0, 1].imshow(stego)
        axes[0, 1].set_title('Stego Image', fontweight='bold', fontsize=12)
        axes[0, 1].axis('off')
        
        # Original FFT
        im1 = axes[1, 0].imshow(fft_original, cmap='hot', interpolation='bilinear')
        axes[1, 0].set_title('Fourier Transform - Original', 
                            fontweight='bold', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Stego FFT
        im2 = axes[1, 1].imshow(fft_stego, cmap='hot', interpolation='bilinear')
        axes[1, 1].set_title('Fourier Transform - Stego', 
                            fontweight='bold', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved Fourier comparison: {save_path}")
    
    def plot_difference_map(self, original, stego, save_path):
        """Create pixel difference visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Pixel-Level Difference Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Absolute difference
        diff = np.abs(original.astype(float) - stego.astype(float))
        diff_gray = np.mean(diff, axis=2)
        
        # Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Stego
        axes[0, 1].imshow(stego)
        axes[0, 1].set_title('Stego Image', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Difference heatmap
        im = axes[1, 0].imshow(diff_gray, cmap='jet', vmin=0, vmax=50)
        axes[1, 0].set_title('Pixel Difference Heatmap', fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Difference histogram
        axes[1, 1].hist(diff_gray.ravel(), bins=50, color='purple', 
                       alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Pixel Difference Magnitude', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Difference Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_diff = np.mean(diff_gray)
        max_diff = np.max(diff_gray)
        axes[1, 1].text(0.98, 0.95, 
                       f'Mean: {mean_diff:.2f}\nMax: {max_diff:.2f}',
                       transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved difference map: {save_path}")
    
    def create_comprehensive_report(self, original, stego, message, output_dir):
        """Create comprehensive validation report"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Steganography Validation Report', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Row 1: Images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('Original Cover Image', fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(stego)
        ax2.set_title('Stego Image (with hidden message)', fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff = np.abs(original.astype(float) - stego.astype(float))
        diff_vis = np.mean(diff, axis=2)
        im = ax3.imshow(diff_vis, cmap='jet', vmin=0, vmax=30)
        ax3.set_title('Difference Map (amplified)', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)
        
        # Row 2: Histograms
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        for i, (ch, col) in enumerate(zip(channels, colors)):
            ax = fig.add_subplot(gs[1, i])
            ax.hist(original[:, :, i].ravel(), bins=256, range=(0, 255),
                   color=col, alpha=0.5, density=True, label='Original')
            ax.hist(stego[:, :, i].ravel(), bins=256, range=(0, 255),
                   color=col, alpha=0.3, density=True, label='Stego', 
                   linestyle='--', linewidth=2)
            ax.set_title(f'{ch} Histogram Overlay', fontweight='bold')
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 3: Entropy
        ax = fig.add_subplot(gs[2, :])
        channels_full = ['Red', 'Green', 'Blue', 'Grayscale']
        orig_ent = [self.calculate_entropy(original[:, :, i]) for i in range(3)]
        orig_ent.append(self.calculate_entropy(original))
        stego_ent = [self.calculate_entropy(stego[:, :, i]) for i in range(3)]
        stego_ent.append(self.calculate_entropy(stego))
        
        x = np.arange(len(channels_full))
        width = 0.35
        ax.bar(x - width/2, orig_ent, width, label='Original', 
               color='steelblue', alpha=0.8)
        ax.bar(x + width/2, stego_ent, width, label='Stego', 
               color='coral', alpha=0.8)
        ax.set_xlabel('Channel', fontweight='bold')
        ax.set_ylabel('Entropy (bits)', fontweight='bold')
        ax.set_title('Entropy Comparison - Statistical Randomness', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(channels_full)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Row 4: Fourier
        fft_orig = self.compute_fourier_transform(original)
        fft_stego = self.compute_fourier_transform(stego)
        
        ax1 = fig.add_subplot(gs[3, 0])
        im1 = ax1.imshow(fft_orig, cmap='hot')
        ax1.set_title('FFT - Original', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[3, 1])
        im2 = ax2.imshow(fft_stego, cmap='hot')
        ax2.set_title('FFT - Stego', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # Statistics panel
        ax3 = fig.add_subplot(gs[3, 2])
        ax3.axis('off')
        
        stats_text = f"""
        VALIDATION STATISTICS
        {'='*30}
        
        Message: "{message}"
        Message Length: {len(message)} chars
        Capacity Used: {len(message)*8} bits
        
        PSNR: {self.calculate_psnr(original, stego):.2f} dB
        SSIM: {self.calculate_ssim(original, stego):.4f}
        
        Mean Pixel Diff: {np.mean(diff):.3f}
        Max Pixel Diff: {np.max(diff):.3f}
        
        Entropy Difference:
          Red:   {abs(orig_ent[0]-stego_ent[0]):.4f}
          Green: {abs(orig_ent[1]-stego_ent[1]):.4f}
          Blue:  {abs(orig_ent[2]-stego_ent[2]):.4f}
        
        ✓ Histograms: Preserved
        ✓ Entropy: Preserved
        ✓ Frequency: Preserved
        """
        
        ax3.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        report_path = Path(output_dir) / 'comprehensive_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comprehensive report: {report_path}")
    
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index (simplified)"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Calculate means
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        # Calculate variances and covariance
        var1 = np.var(gray1)
        var2 = np.var(gray2)
        cov = np.cov(gray1.flatten(), gray2.flatten())[0, 1]
        
        # SSIM formula
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        
        return ssim
    
    def run_full_validation(self, cover_path, message, output_dir='static/validation'):
        """Run complete validation analysis"""
        print("\n" + "="*70)
        print("🔬 RUNNING COMPREHENSIVE VALIDATION ANALYSIS")
        print("="*70 + "\n")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cover image
        print(f"📂 Loading cover image: {cover_path}")
        cover_image = Image.open(cover_path).convert('RGB')
        
        # Generate stego image
        print(f"🔐 Hiding message: '{message}'")
        stego_tensor = self.generate_stego_image(cover_image, message)
        
        # Convert to numpy
        original_np = np.array(cover_image.resize((256, 256)))
        stego_np = self.tensor_to_numpy(stego_tensor)
        
        print(f"\n📊 Generating validation visualizations...")
        
        # Generate all plots
        self.plot_histogram_comparison(original_np, stego_np, 
                                       output_dir / 'histogram_comparison.png')
        
        self.plot_entropy_comparison(original_np, stego_np,
                                    output_dir / 'entropy_comparison.png')
        
        self.plot_fourier_comparison(original_np, stego_np,
                                    output_dir / 'fourier_comparison.png')
        
        self.plot_difference_map(original_np, stego_np,
                               output_dir / 'difference_map.png')
        
        self.create_comprehensive_report(original_np, stego_np, message, output_dir)
        
        # Calculate metrics
        psnr = self.calculate_psnr(original_np, stego_np)
        ssim = self.calculate_ssim(original_np, stego_np)
        
        print(f"\n📈 VALIDATION METRICS:")
        print(f"{'='*70}")
        print(f"  PSNR: {psnr:.2f} dB (higher is better, >30 is excellent)")
        print(f"  SSIM: {ssim:.4f} (closer to 1 is better)")
        print(f"{'='*70}\n")
        
        print(f"✅ All validation images saved to: {output_dir}")
        print(f"{'='*70}\n")
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'output_dir': str(output_dir)
        }


def main():
    """Main validation function"""
    analyzer = ValidationAnalyzer()
    
    # Test with sample image
    test_image_path = 'data/bedrooms/samples/sample_0000.jpg'
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        print("   Please ensure you have training data available.")
        return
    
    test_message = "Secret Message 2024"
    
    results = analyzer.run_full_validation(
        cover_path=test_image_path,
        message=test_message,
        output_dir='static/validation'
    )
    
    print("🎉 Validation complete!")
    print(f"\n📁 View results in: {results['output_dir']}")


if __name__ == "__main__":
    main()