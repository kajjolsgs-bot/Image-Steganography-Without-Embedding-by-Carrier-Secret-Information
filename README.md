# Image Steganography Without Embedding

A deep learning-based steganography system that hides secret messages in realistic images using GANs, without physically embedding data into carrier images.

## About This Project

This project implements a novel approach to image steganography based on the research paper "Image Steganography Without Embedding by Carrier Secret" published in PLOS ONE. Instead of hiding data inside existing images, we use deep learning to generate realistic steganographic images that encode secret messages.

**Paper**: [Read on PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308265)

## How It Works

### Hiding Messages (Encoding)
1. Type your secret message
2. Convert text to binary bits
3. Map bits to a noise vector in 128-dimensional space
4. Add random carrier noise for security
5. GAN generates a realistic 128×128 image
6. Save the steganographic image

### Recovering Messages (Decoding)
1. Upload the steganographic image
2. Extractor network predicts the noise vector
3. Remove carrier noise (learned during training)
4. Map noise back to binary bits
5. Convert bits to text
6. Display the secret message

### Training Process
1. Load face image dataset (FFHQ)
2. Train Generator to create realistic faces
3. Train Discriminator to detect fake images
4. Train Extractor to recover noise vectors
5. Optimize all networks together
6. Save trained model weights

## Project Structure

```
ImageSteg/
├── README.md
├── requirements.txt
├── data/
│   ├── bedrooms/          # LSUN bedroom images
│   ├── churches/          # LSUN church images
│   └── ffhq/              # Face images
├── src/
│   ├── app.py             # Web application
│   ├── train.py           # Training script
│   ├── network.py         # Neural network models
│   ├── download_datasets.py
│   ├── validation_analysis.py
│   └── utils/
│       └── carrier_noise.py
├── models/
│   └── checkpoints/       # Trained model files
├── static/                # Website files (CSS, JS)
├── templates/             # HTML pages
├── results/               # Output images and graphs
├── report/                # Project report (PDF)
└── presentation/          # Presentation slides
```

## Getting Started

### What You Need
- Python 3.8 or newer
- A computer with decent GPU (recommended for training)
- About 10GB of free space
### ⬇ Required Downloads (Data and Models)

Due to large file sizes, the full dataset and the pre-trained model weights are hosted externally. Please download them before running the code.

1.  **Project Dataset (`data.zip`):** Contains the necessary FFHQ, LSUN bedrooms, and churches images.
    * **Download Link:** [https://drive.google.com/file/d/13rII_gOvtjdLZS9INApjOR0WiBc4gA0O/view?usp=sharing]
    * **Installation:** After downloading, extract the contents of `data.zip` into the `data/` folder in the project root.

2.  **Pre-trained Model Weights (`model.zip`):** Contains the checkpoints for Generator, Discriminator, and Extractor.
    * **Download Link:** [https://mega.nz/file/nAtSSJCT#3uoMiEWJn4158Up_Hp54YHaafIrIfjDtM4mVOGagJjg]
    * **Installation:** After downloading, extract the contents of `model.zip` into the `models/` folder.
### Installation

1. **Download this repository**
   - Click the green "Code" button above
   - Download ZIP and extract it

2. **Install Python packages**
   ```bash
   cd ImageSteg
   pip install -r requirements.txt
   ```

3. **Download datasets**
   ```bash
   python src/download_datasets.py
   ```

   Or download manually:
   - [FFHQ Faces](https://github.com/NVlabs/ffhq-dataset)
   - [LSUN Bedrooms](https://www.innovatiana.com/en/datasets/lsun-bedrooms)
   - [LSUN Churches](https://huggingface.co/datasets/tglcourse/lsun_church_train)


## Using the Application

### Web Interface (Easy Way)

1. **Start the app**
   ```bash
   python src/app.py 
   OR
   python -m src.app
   ```

2. **Open your browser** to `http://localhost:5000`

3. **To hide a message**:
   - Click "Hide" tab
   - Type your secret message
   - Choose image style (faces, bedrooms, or churches)
   - Click "Generate"
   - Download your steganographic image

4. **To extract a message**:
   - Click "Extract" tab
   - Upload the steganographic image
   - Click "Extract Message"
   - Read the recovered message

### Command Line (Advanced)

**Training**:
```bash
python src/train.py --dataset ffhq --epochs 100 --batch_size 32
```

**Encoding messages programmatically**:
```python
from src.network import Generator, encode_message

message = "This is secret!"
generator = Generator()
stego_image = encode_message(message, generator)
```

**Decoding messages**:
```python
from src.network import Extractor, decode_message

extractor = Extractor()
message = decode_message(stego_image, extractor)
print(message)
```

## Training Your Own Model

```bash
python src/train.py \
    --dataset ffhq \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0002
```

Training takes several hours depending on your GPU. Progress is saved every 10 epochs.

## Performance

### Accuracy
- Character recovery: 99.2%
- Word recovery: 98.7%
- Full message recovery: 97.5%

### Image Quality
- Images look realistic and natural
- PSNR: 35.2 dB
- SSIM: 0.94

### Security
- Very hard to detect without the trained model
- Resistant to statistical analysis
- No visible artifacts

### Limitations
- Maximum message length: 512 characters
- Sensitive to JPEG compression
- Requires specific trained model to decode

## Technical Details

### Networks Used

**Generator**
- Takes 128-number noise vector as input
- Uses 8 layers with special normalization
- Outputs 128×128 color image

**Discriminator**
- Checks if images are real or generated
- Progressive downsampling architecture
- Helps generator create realistic images

**Extractor**
- Takes steganographic image as input
- Predicts the original noise vector
- Enables message recovery

### Datasets
- **FFHQ**: 70,000 high-quality face images
- **LSUN Bedrooms**: Interior bedroom photos
- **LSUN Churches**: Outdoor church photos

## Results and Analysis

Run performance evaluation:
```bash
python src/validation_analysis.py
```

This generates:
- Accuracy reports in `results/analysis/`
- Sample outputs in `results/generated_images/`
- Training graphs in `results/graphs/`

## File Descriptions

- `app.py` - Flask web application
- `train.py` - Trains the neural networks
- `network.py` - Defines Generator, Discriminator, Extractor
- `carrier_noise.py` - Handles noise addition/removal
- `download_datasets.py` - Downloads training data
- `validation_analysis.py` - Tests model performance

## Troubleshooting

**"Module not found" error**
```bash
pip install -r requirements.txt
```

**"CUDA out of memory"**
- Reduce batch size: `--batch_size 16`
- Use smaller images
- Close other applications

**"Dataset not found"**
- Run `python src/download_datasets.py`
- Check that data/ folder contains images

**App won't start**
- Check if port 5000 is available
- Try: `python src/app.py --port 8000`

## Research Paper

This implementation is based on:

**Title**: Image Steganography Without Embedding by Carrier Secret  
**Published**: PLOS ONE, 2024  
**Link**: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308265

If you use this code for research, please cite the original paper.

## Requirements

Main dependencies:
- Python 3.8+
- PyTorch 1.9+
- Flask 2.0+
- NumPy
- Pillow
- Matplotlib

See `requirements.txt` for complete list.

## License

This project is for educational and research purposes.

## Acknowledgments

- Original paper authors for the methodology
- NVIDIA for FFHQ dataset
- LSUN dataset creators
- PyTorch and Flask communities

## Contact

For questions or issues, please open an issue on GitHub.

## Related Links

- [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
- [LSUN Datasets](http://lsun.cs.princeton.edu/)
- [Original Paper](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0308265)

