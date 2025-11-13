import torch
import numpy as np

class CarrierSecretTensor:
    """
    Implementation of carrier secret tensor generation as per paper
    """
    def __init__(self, sigma=3, delta=25, noise_dim=256):
        """
        Args:
            sigma: Number of bits for message mapping
            delta: Random perturbation range
            noise_dim: Dimension of noise tensor
        """
        self.sigma = sigma
        self.delta = delta
        self.noise_dim = noise_dim
        
    def generate_carrier_function(self, length=None):
        """
        Generate carrier function f(t_m)
        Formula: f(t_m) = (1+e)/2^(σ-1) * sin(t_m/2)
        
        Args:
            length: Length of carrier (defaults to noise_dim)
            
        Returns:
            Carrier function values
        """
        if length is None:
            length = self.noise_dim
            
        t_m = torch.linspace(0, 2 * np.pi * length / self.noise_dim, length)
        e = np.e
        carrier = ((1 + e) / (2 ** (self.sigma - 1))) * torch.sin(t_m / 2)
        return carrier
    
    def message_to_tensor(self, message):
        """
        Convert binary message to secret tensor
        
        Args:
            message: Binary string or list of bits
            
        Returns:
            Secret tensor Z
        """
        if isinstance(message, str):
            # Convert string to binary
            message_bits = ''.join(format(ord(c), '08b') for c in message)
        else:
            message_bits = ''.join(str(b) for b in message)
        
        # Pad message to noise_dim
        if len(message_bits) < self.noise_dim:
            message_bits += '0' * (self.noise_dim - len(message_bits))
        else:
            message_bits = message_bits[:self.noise_dim]
        
        # Convert to decimal values
        M = torch.tensor([int(message_bits[i:i+self.sigma], 2) 
                         for i in range(0, len(message_bits), self.sigma)], 
                        dtype=torch.float32)
        
        # Map to noise interval (-1, 1)
        # Formula: Z = (M + 0.5) / 2^(σ-1) - 1 + rand(-Δ*r, Δ*r)
        r = (np.e / 2 - 1) / (2 ** (self.sigma - 1))
        random_noise = torch.rand_like(M) * 2 * self.delta * r - self.delta * r
        
        Z = (M + 0.5) / (2 ** (self.sigma - 1)) - 1 + random_noise
        
        # Pad to noise_dim if needed
        if Z.size(0) < self.noise_dim:
            padding = torch.zeros(self.noise_dim - Z.size(0))
            Z = torch.cat([Z, padding])
        
        return Z[:self.noise_dim]
    
    def generate_carrier_secret_tensor(self, message):
        """
        Generate carrier secret tensor Z' from message
        Formula: Z' = Z + f(t_m)
        
        Args:
            message: Secret message (string or binary)
            
        Returns:
            Carrier secret tensor Z'
        """
        # Generate secret tensor Z
        Z = self.message_to_tensor(message)
        
        # Generate carrier function - FIXED: use noise_dim instead of len(Z)
        carrier = self.generate_carrier_function(self.noise_dim)
        
        # Combine: Z' = Z + f(t_m)
        Z_prime = Z + carrier
        
        return Z_prime
    
    def extract_message_from_tensor(self, Z_double_prime):
        """
        Extract message from carrier secret tensor
        Formula: Z''' = Z'' - f(t_m), then M' = floor[(Z''' + 1) * 2^(σ-1)]
        
        Args:
            Z_double_prime: Extracted carrier secret tensor
            
        Returns:
            Extracted message as string
        """
        # FIXED: Always use noise_dim for carrier function
        carrier = self.generate_carrier_function(self.noise_dim)
        
        # Ensure tensor is the right size
        if len(Z_double_prime) > self.noise_dim:
            Z_double_prime = Z_double_prime[:self.noise_dim]
        elif len(Z_double_prime) < self.noise_dim:
            padding = torch.zeros(self.noise_dim - len(Z_double_prime))
            Z_double_prime = torch.cat([Z_double_prime, padding])
        
        # Remove carrier function
        Z_triple_prime = Z_double_prime - carrier
        
        # Inverse mapping
        M_prime = torch.floor((Z_triple_prime + 1) * (2 ** (self.sigma - 1)))
        M_prime = torch.clamp(M_prime, 0, 2 ** self.sigma - 1)
        
        # Convert to binary
        binary_str = ''.join(format(int(m.item()), f'0{self.sigma}b') 
                            for m in M_prime)
        
        # Convert binary to characters
        message = ''
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    message += chr(char_code)
                elif char_code == 0:  # Null padding
                    break
        
        return message.rstrip('\x00')  # Remove null padding
    
    def calculate_capacity(self, N=3):
        """
        Calculate steganography capacity
        Formula: Capacity = Size × N × σ bits
        
        Args:
            N: Number of blocks
            
        Returns:
            Capacity in bits
        """
        return self.noise_dim * N * self.sigma

def test_carrier_noise():
    """Test carrier secret tensor generation and extraction"""
    print("Testing Carrier Secret Tensor...")
    
    carrier = CarrierSecretTensor(sigma=3, delta=25, noise_dim=256)
    
    # Test message
    test_message = "Hello, World!"
    print(f"Original message: {test_message}")
    
    # Generate carrier secret tensor
    Z_prime = carrier.generate_carrier_secret_tensor(test_message)
    print(f"Carrier secret tensor shape: {Z_prime.shape}")
    print(f"Tensor range: [{Z_prime.min():.3f}, {Z_prime.max():.3f}]")
    
    # Extract message (simulating perfect extraction)
    extracted_message = carrier.extract_message_from_tensor(Z_prime)
    print(f"Extracted message: {extracted_message}")
    
    # Verify match
    if test_message == extracted_message:
        print("✓ Message extraction successful!")
    else:
        print(f"✗ Extraction failed. Expected: '{test_message}', Got: '{extracted_message}'")
    
    # Calculate capacity
    capacity = carrier.calculate_capacity(N=3)
    print(f"Steganography capacity: {capacity} bits")
    
    print("✓ Test completed!")

if __name__ == "__main__":
    test_carrier_noise()