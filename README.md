# InterfaceGAN - StyleGAN2-ADA Latent Space Manipulation

## Project Overview

This project explores latent space manipulation techniques for StyleGAN2-ADA, focusing on both supervised (InterfaceGAN) and unsupervised (GANSpace) methods for controlling facial attributes in generated images. The work was conducted as part of the IMA 206 course at Telecom Paris in 2024.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Methods Implemented](#methods-implemented)
  - [1. Unsupervised Analysis (GANSpace)](#1-unsupervised-analysis-ganspace)
  - [2. Supervised Analysis (InterfaceGAN)](#2-supervised-analysis-interfacegan)
  - [3. Layer-wise Manipulation](#3-layer-wise-manipulation)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction

This project implements and compares different approaches to manipulate the latent space of StyleGAN2-ADA for controllable face generation. The main objectives are:

1. **Discover semantic directions** in the latent space corresponding to facial attributes (age, gender, smile, etc.)
2. **Disentangle attributes** to enable independent control of different facial features
3. **Compare supervised and unsupervised methods** for latent space exploration
4. **Enable precise face editing** through attribute manipulation

## Project Structure

```
InterfaceGAN/
├── Article-impression/          # Reference articles
│   ├── Article_1.pdf
│   ├── article2.pdf
│   └── article3.pdf
├── Rendu/                       # Main deliverables
│   ├── Analyse non supervisée.ipynb    # Unsupervised methods (GANSpace)
│   ├── InterfaceGAN.ipynb              # Supervised methods (InterfaceGAN)
│   └── Rapport IMA 206.pdf             # Project report
└── README.md                    # This file
```

## Methods Implemented

### 1. Unsupervised Analysis (GANSpace)

#### Methodology
- **Principal Component Analysis (PCA)** on the W latent space
- **Kernel PCA (KPCA)** with RBF kernel for non-linear feature extraction
- **Variational Autoencoder (VAE)** for latent space restructuring

#### Key Features
- Generate 10,000-100,000 samples from StyleGAN2's W space
- Perform PCA to identify principal directions of variation
- Move along principal components to discover semantic attributes
- Layer-wise control: apply transformations to specific layers (0-17)

#### Discovered Attributes
- **v0**: Gender
- **v1**: Gender + Rotation
- **v2**: Gender + Rotation + Age + Background
- **v4**: Age + Hairstyle (layers 5-18)
- **v10**: Hair color (layers 7-8)
- **v20**: Wrinkles (layer 6)
- **v23**: Expression/Smile (layers 3-5)

#### Advanced Techniques

**Kernel PCA Exploration**
- Implemented KPCA with RBF kernel for non-linear dimensionality reduction
- Compared results with linear PCA
- Found similar semantic directions with slight differences in disentanglement

**Autoencoder-based Latent Space Restructuring**
- Designed a custom VAE with dual Gaussian components
- Architecture: 512 → 256 → 128 → 64 → 50 (latent) → 64 → 128 → 256 → 512
- Added orthogonality constraint to encourage independence
- Achieved better variance concentration: 95% variance in 20 components vs. 50+ for raw W space

### 2. Supervised Analysis (InterfaceGAN)

#### Methodology

InterfaceGAN uses **linear SVM classifiers** to find semantic boundaries in the latent space:

1. **Dataset Generation**: Generate 50,000-100,000 images with StyleGAN2
2. **Attribute Classification**: Use pre-trained classifiers to label images
3. **SVM Training**: Train linear SVMs to separate attribute classes
4. **Direction Extraction**: Normal vector to the SVM hyperplane = semantic direction

#### Supported Attributes
- Bald
- Hair color (black, blond, brown)
- Eyeglasses
- Heavy makeup
- Gender (male/female)
- Mustache
- Beard
- Smiling
- Hat
- Age (young)

#### Attribute Disentanglement

**Projection Method**
To control one attribute without affecting others, the project implements orthogonal projection:

```python
# Make direction vect_normal_1 perpendicular to vect_normal_2
vect_normal_1_new = vect_normal_1 - (vect_normal_1 @ vect_normal_2) * vect_normal_2
vect_normal_1_new = vect_normal_1_new / np.linalg.norm(vect_normal_1_new)
```

**Examples:**
- Age modification without adding/removing glasses
- Bald attribute on female faces
- Mustache without changing gender
- Gender change while preserving other attributes

**Multi-attribute Orthogonalization**
The `move_along_attr_against_all_others` function makes a direction orthogonal to all other attribute directions for maximum disentanglement.

### 3. Layer-wise Manipulation

StyleGAN2's architecture allows different layers to control different aspects:

- **Layers 0-3**: Coarse features (pose, glasses, general face shape)
- **Layers 4-7**: Medium features (facial features, age, gender)
- **Layers 8-17**: Fine features (hair texture, skin details, wrinkles)

**Key Findings:**
- Layers 0-3: Control glasses independently from age/gender
- Layers 4-5: Control age while preserving accessories
- Layers 6-7: Control facial hair (mustache/beard) with fine precision
- Enables better disentanglement than full-layer manipulation

## Key Features

### Data Generation
- Automated generation of 50,000+ labeled face images
- Pre-trained classifier for 12 facial attributes
- Efficient batch processing with GPU acceleration

### Visualization Tools
- Distribution histograms showing attribute projections
- Video generation showing smooth transitions
- 12×12 grid visualization of attribute correlations
- Side-by-side before/after comparisons

### Attribute Control
- Single attribute modification
- Multi-attribute editing
- Conditional manipulation (change A without changing B)
- Layer-specific control
- Precise scalar control for each attribute

### Analysis Tools
- Semantic boundary correlation analysis
- PCA variance visualization
- TSNE visualization of latent space structure
- Comparison between PCA and KPCA results

## Requirements

### Core Dependencies
```
python >= 3.8
torch >= 1.9.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0
pandas >= 1.2.0
imageio >= 2.9.0
tqdm >= 4.60.0
PIL >= 8.0.0
```

### StyleGAN2-ADA Dependencies
```
dnnlib
legacy
click
```

### Hardware
- GPU with CUDA support recommended (tested on NVIDIA GPUs and Apple Silicon with MPS)
- Minimum 16GB RAM
- 20GB+ disk space for models and generated images

## Usage

### 1. Unsupervised Analysis (GANSpace)

```python
# Generate samples from W space
w_samples = generate_samples(G, 10000)

# Perform PCA
pca, scaler = get_pca(w_samples, ncomp=50)

# Move along a principal component
img_list, X_list = move_according_to_component(
    G, device, pca, 
    outdir='output_directory_pca',
    directions=[0],  # Gender direction
    layers=list(range(0, 18)),  # All layers
    num_steps=100,
    save_video=True,
    title_vid='gender_all',
    random_seed=2
)
```

### 2. InterfaceGAN Attribute Editing

```python
# Load dataset
df = pd.read_csv('label_faces_100K.csv')

# Edit a specific attribute
img_list = pipeline_1(
    df, 
    scalars_young,  # Movement range
    'young',        # Attribute to modify
    2000,           # Number of training samples
    network_pkl,
    seed_img=17,
    num_img_generated=10,
    outdir='out/along_1_vector',
    file_name='move_along_young',
    save_video=True,
    return_images=True
)
```

### 3. Disentangled Editing

```python
# Edit one attribute while keeping another fixed
img_list = move_along_attr_against_attrb2(
    network_pkl,
    scalars_young,
    'young',        # Attribute to change
    'eyeglasses',   # Attribute to preserve
    2000,
    seed_img=17,
    num_img_generated=10,
    outdir='out',
    file_name='young_without_glasses',
    save_video=True,
    return_images=True
)
```

### 4. Layer-wise Control

```python
# Edit beard only in specific layers
img_list = pipeline_4_layer(
    df,
    'beard',
    2000,
    network_pkl,
    seed_img=17,
    layers=[6, 7],  # Fine detail layers
    num_img_generated=10,
    outdir='out',
    file_name='beard_layerwise',
    save_video=True,
    return_images=True
)
```

### 5. Precise Multi-attribute Editing

```python
# Set specific values for each attribute
attributes_scalars = [
    -3,  # bald
    0,   # black_hair
    0,   # blond_hair
    2,   # brown_hair
    -2,  # eyeglasses
    0,   # heavy_makeup
    -2,  # male (more feminine)
    0,   # mustache
    0,   # beard
    2,   # smiling
    0,   # hat
    -1   # young
]

edit_face(attributes_scalars, seed_img=17, plot_distribution=True)
```

## Results

### Unsupervised Methods (GANSpace)

**PCA Analysis:**
- Successfully identified gender, age, rotation, hair color, and expression directions
- 95% variance explained by 50 components in raw W space
- Layer-wise application enables finer control

**Kernel PCA:**
- RBF kernel produces similar directions to linear PCA
- Slightly different magnitude of changes
- Cosine similarity near 1.0 with PCA directions

**VAE Restructuring:**
- Reduced dimensionality: 95% variance in 20 components
- More concentrated feature representation
- Comparable editing quality to raw W space PCA

### Supervised Methods (InterfaceGAN)

**Attribute Discovery:**
- Successfully identified 12 distinct facial attributes
- High classification accuracy (>90%) for most attributes
- Clear semantic boundaries in W space

**Disentanglement:**
- Partial success in decoupling attributes
- Some inherent correlations difficult to remove (e.g., male/beard)
- Orthogonal projection improves independence

**Layer-wise Control:**
- **Major finding**: Layer-wise editing achieves better disentanglement
- Successfully created "female with mustache" - impossible with global editing
- Different layers control different semantic levels

### Attribute Correlations

Observed entanglements:
- **Age ↔ Eyeglasses**: Older people more likely to wear glasses
- **Gender ↔ Beard/Mustache**: Strong correlation
- **Bald ↔ Hat**: Negative correlation
- **Age ↔ Hair color**: Gray hair appears with age

Successfully disentangled:
- Age from glasses (using orthogonal projection)
- Gender from mustache (using layer-wise editing on layers 6-7)
- Smile from other attributes

## Technical Details

### Model Architecture

**StyleGAN2-ADA Generator:**
- Mapping network: Z (512) → W (512)
- Synthesis network: 18 layers, each receiving W vector
- Pre-trained on FFHQ (Flickr-Faces-HQ) dataset
- Network: `ffhq.pkl` from NVIDIA

**Classifier:**
- Pre-trained on CelebA attributes
- 40 binary classifiers for facial attributes
- Used for automatic labeling of generated images

### Latent Space Structure

**Z Space (Gaussian):**
- 512-dimensional standard normal distribution
- Input to mapping network

**W Space:**
- 512-dimensional intermediate latent space
- More disentangled than Z
- Used for all manipulations in this project

**W+ Space:**
- 18 × 512 dimensions (different W for each layer)
- Allows layer-specific control
- Used for fine-grained editing

### SVM Training Details

**Hyperparameters:**
- Kernel: Linear
- C: 1.0 (regularization)
- Training samples: 2,000-4,000 per attribute
- Test/train split: 70/30

**Data Selection:**
- Select top N samples with highest attribute probability
- Select bottom N samples with lowest attribute probability
- Ensures clear separation for SVM training

## Limitations and Future Work

### Current Limitations

1. **Attribute Entanglement**: Some attributes remain correlated despite disentanglement efforts
2. **Dataset Bias**: Limited diversity in training data (mostly frontal faces)
3. **Computational Cost**: Generating large datasets requires significant GPU time
4. **Binary Attributes**: Many attributes are treated as binary (present/absent) when they're actually continuous
5. **Rare Combinations**: Some attribute combinations (e.g., female + bald) are underrepresented

### Future Directions

1. **Advanced Disentanglement**:
   - Implement more sophisticated orthogonalization techniques
   - Explore non-linear separation methods
   - Use adversarial training for better independence

2. **Interactive Editing**:
   - Real-time GUI for attribute manipulation
   - Slider-based interface for continuous control
   - Preset combinations for common edits

3. **Extended Attributes**:
   - Fine-grained attributes (eye color, nose shape, etc.)
   - Style attributes (lighting, background, etc.)
   - Emotion attributes beyond smile

4. **Better Evaluation**:
   - Quantitative metrics for disentanglement
   - User studies for quality assessment
   - Automated testing of attribute independence

5. **Generalization**:
   - Apply methods to other domains (cars, animals, etc.)
   - Transfer learning to other GAN architectures
   - Cross-domain attribute manipulation

## References

### Papers

1. **InterfaceGAN**: Shen, Y., Gu, J., Tang, X., & Zhou, B. (2020). Interpreting the Latent Space of GANs for Semantic Face Editing. CVPR 2020.

2. **GANSpace**: Härkönen, E., Hertzmann, A., Lehtinen, J., & Paris, S. (2020). GANSpace: Discovering Interpretable GAN Controls. NeurIPS 2020.

3. **StyleGAN2-ADA**: Karras, T., et al. (2020). Training Generative Adversarial Networks with Limited Data. NeurIPS 2020.

### Code Resources

- [Official StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [InterfaceGAN Repository](https://github.com/genforce/interfacegan)
- [GANSpace Repository](https://github.com/harskish/ganspace)

## Authors

Project developed as part of IMA 206 course at Telecom Paris, 2024.

## License

This project is for educational purposes. Pre-trained models and base code follow their respective licenses from NVIDIA and other sources.

## Acknowledgments

- NVIDIA for StyleGAN2-ADA implementation
- InterfaceGAN and GANSpace authors for methodology
- Telecom Paris for course framework
- CelebA and FFHQ datasets for training data
