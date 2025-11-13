from flask import Flask, render_template, request, jsonify, send_file, session
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
from pathlib import Path
import os
import struct

from src.network import (StructureGenerator, TextureExtractor,
                    StructureExtractor, ImageGenerator, SecretExtractor)
from utils.carrier_noise import CarrierSecretTensor

app = Flask(__name__, template_folder="../templates")
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for session
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class SteganographyApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_dim = 256
        self.sigma = 3
        self.delta = 25
        
        # Initialize carrier noise generator
        self.carrier = CarrierSecretTensor(
            sigma=self.sigma,
            delta=self.delta,
            noise_dim=self.noise_dim
        )
        
        print("Steganography App initialized with DIRECT EMBEDDING method")
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def text_to_binary(self, text):
        """Convert text to binary string"""
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary
    
    def binary_to_text(self, binary):
        """Convert binary string to text"""
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:  # Printable ASCII
                    text += chr(char_code)
                elif char_code == 0:
                    break
        return text
    
    def hide_message_simple(self, cover_image, secret_message):
        """
        Simple LSB steganography - hide message directly in image
        """
        # Resize image to standard size
        img = cover_image.resize((512, 512))
        img_array = np.array(img, dtype=np.uint8)
        
        # Convert message to binary
        binary_message = self.text_to_binary(secret_message)
        
        # Add message length at the beginning (32 bits)
        msg_len = len(binary_message)
        len_binary = format(msg_len, '032b')
        
        # Complete binary data = length + message
        full_binary = len_binary + binary_message
        
        # Flatten image array
        flat_img = img_array.flatten()
        
        # Check capacity
        if len(full_binary) > len(flat_img):
            raise ValueError(f"Message too long! Max {len(flat_img)//8} characters.")
        
        # Embed binary data into LSBs
        for i, bit in enumerate(full_binary):
            flat_img[i] = (flat_img[i] & 0xFE) | int(bit)
        
        # Reshape back to image
        stego_array = flat_img.reshape(img_array.shape)
        stego_image = Image.fromarray(stego_array, mode='RGB')
        
        return stego_image
    
    def extract_message_simple(self, stego_image):
        """
        Simple LSB steganography - extract message from image
        """
        # Resize to same size used in hiding
        img = stego_image.resize((512, 512))
        img_array = np.array(img, dtype=np.uint8)
        
        # Flatten image
        flat_img = img_array.flatten()
        
        # Extract length (first 32 bits)
        len_bits = ''.join(str(flat_img[i] & 1) for i in range(32))
        msg_len = int(len_bits, 2)
        
        # Validate length
        if msg_len <= 0 or msg_len > len(flat_img) - 32:
            return ""
        
        # Extract message bits
        msg_bits = ''.join(str(flat_img[i + 32] & 1) for i in range(msg_len))
        
        # Convert to text
        message = self.binary_to_text(msg_bits)
        
        return message

# Initialize app
stego_app = SteganographyApp()

@app.route('/')
def index():
    return render_template('hide.html')

@app.route('/extract-page')
def extract_page():
    return render_template('extract.html')

@app.route('/result-page')
def result_page():
    return render_template('result.html')

@app.route('/hide', methods=['POST'])
def hide():
    """API endpoint to hide message in image"""
    try:
        # Get uploaded file and message
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        message = request.form.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"Hiding message: '{message}'")
        
        # Load image
        cover_image = Image.open(file.stream).convert('RGB')
        
        # Hide message using simple LSB method
        stego_image = stego_app.hide_message_simple(cover_image, message)
        
        # Convert to base64
        stego_base64 = stego_app.image_to_base64(stego_image)
        
        # Store original message in session for comparison later
        session['original_message'] = message
        
        print(f"Message hidden successfully!")
        
        return jsonify({
            'success': True,
            'stego_image': f'data:image/png;base64,{stego_base64}',
            'capacity': 512 * 512 * 3,  # bits
            'message_length': len(message) * 8,
            'method': 'simple_lsb'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract():
    """API endpoint to extract message from image"""
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        # Load image
        stego_image = Image.open(file.stream).convert('RGB')
        
        print(f"Extracting message from image...")
        
        # Extract message using simple LSB method
        message = stego_app.extract_message_simple(stego_image)
        
        # Store extracted message in session for comparison
        session['extracted_message'] = message
        
        print(f"Extracted message: '{message}'")
        
        return jsonify({
            'success': True,
            'message': message,
            'method': 'simple_lsb'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get-comparison', methods=['GET'])
def get_comparison():
    """API endpoint to get comparison data"""
    try:
        original = session.get('original_message', '')
        extracted = session.get('extracted_message', '')
        
        match = original == extracted
        
        return jsonify({
            'success': True,
            'original_message': original,
            'extracted_message': extracted,
            'match': match
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-simple', methods=['POST'])
def test_simple():
    """Test simple encoding/decoding"""
    try:
        message = request.form.get('message', 'hello')
        
        # Create a test image
        test_img = Image.new('RGB', (512, 512), color='white')
        
        # Hide and extract
        stego = stego_app.hide_message_simple(test_img, message)
        extracted = stego_app.extract_message_simple(stego)
        
        return jsonify({
            'success': True,
            'original': message,
            'extracted': extracted,
            'match': message == extracted
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Get system information"""
    return jsonify({
        'device': str(stego_app.device),
        'noise_dim': stego_app.noise_dim,
        'sigma': stego_app.sigma,
        'delta': stego_app.delta,
        'capacity': 512 * 512 * 3,
        'method': 'simple_lsb'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)