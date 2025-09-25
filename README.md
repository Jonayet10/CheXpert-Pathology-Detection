# Chest X-Ray Pathology Classification (CheXpert)

This project develops a deep learning pipeline for automated diagnosis of thoracic pathologies using chest X-ray images. The primary dataset is the **Stanford CheXpert dataset**, and the goal is to predict the presence of eight different pathologies. The work combines classical computer vision preprocessing with modern deep learning models to improve robustness and interpretability.

## Project Overview

Chest X-rays are one of the most common imaging modalities in medicine. However, interpreting them consistently remains challenging due to subtle visual cues and inter-observer variability. This project aims to:

- Build a preprocessing pipeline for chest X-ray images.
- Experiment with feature extraction methods (edge detection, blurriness detection, etc.).
- Train and evaluate deep learning models for **multi-label classification** of thoracic diseases.
- Explore data augmentation and transfer learning to reduce overfitting and bias.

## Dataset

- **Source:** [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)  
- **Size:** 224,316 chest radiographs from 65,240 patients  
- **Labels:** Automatically extracted from radiology reports using NLP (>90% accuracy)  
- **Pathologies (focus):**
  - Atelectasis  
  - Cardiomegaly  
  - Consolidation  
  - Edema  
  - Pleural Effusion  
  - Pneumonia  
  - Pneumothorax  
  - Support Devices  

Additional reference datasets include NIH Chest X-ray (over 100k images) and MIMIC-CXR (over 300k images).

## Repository Structure
- data_visualization/ - Scripts and plots for exploratory data analysts
- data/ - Examples from raw dataset
- data_transformed/ - Processed examples from dataset
- feature_extraction/ - Edge dection, blurriness detection files
- notebooks/ - Jupyter notebooks for experiments
- classification.py - Core training and evaluation pipeline
- data_loading_resizing.py - Data loading and resizing utilities
- test.py - Testing and evaluation scripts


## Methods

### Preprocessing
- Resizing to multiple input resolutions (128×128, 256×256).
- Conversion to grayscale.
- Data augmentation: random rotations, translations, elastic deformations, sharpness/contrast adjustments.
- Weighted sampling to address class imbalance.
- Filtering out uncertain labels.

### Feature Extraction
- Edge detection (Sobel, Canny) for fractures and support device detection.
- Local Binary Patterns (LBP) for texture analysis.
- Elastic deformations for structural abnormalities (e.g., cardiomegaly).
- Blurriness detection for quality control.

### Modeling
- **Transfer learning architectures:**
  - ResNet-50 / ResNet-v2-152x4 (pretrained on ImageNet-21k and chest X-ray datasets).
  - DenseNet-121.
- **Optimization:**
  - Binary cross-entropy loss for multi-label classification.
  - Cosine annealing learning rate scheduler.
  - Optimizers: Adam, RMSProp, SGD with experimentation.
- **Evaluation metrics:**
  - Precision, Recall, F1-score.
  - Mean Squared Error (MSE) per pathology.
  - AUROC for overall performance.

## Results (Highlights)

- **Best performing model:** ResNet-v2-152x4 (1.11B parameters, pretrained) at 256×256 input resolution.
- **Impact of augmentation:** Reduced overfitting and improved validation performance compared to vanilla training.
- **Pathology-wise findings:**
  - Cardiomegaly and pleural effusion showed strong detection performance.
  - Fractures, pneumonia, and support devices were more challenging, benefiting from edge detection and contrast augmentation.

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- OpenCV
- scikit-learn
- PyTorch/timm (for pretrained models)

## Usage

1. **(Optional) Create a fresh environment and install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install timm torchxrayvision scikit-image pandas pillow matplotlib opencv-python absl-py
2. **Prepare pickled training dataset**
   ```bash
   python data_loading_resizing.py
3. **Train the main model (ResNetv2 BIT**
   ```bash
   python classification.py
4. **Finetune DenseNet-121 (TorchXRayVision)**
   ```bash
   python densenet121_finetuning.py
5. **Test/Inference**
   ```bash
   python test.py
7. **DenseNet-121 fine-tuning**
   ```bash
   python densenet121_finetuning.py --train <train_dir> --val <val_dir> --epochs 30
8. **Testing**
   ```bash
   python test.py --model <saved_model.pth> --data <test_dir>
9, **Image Quality and Edges (Optional)**
   ```bash
   python blurriness_detection.py
   python edge_detection.py
