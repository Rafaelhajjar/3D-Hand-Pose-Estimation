# 3D Hand Pose Estimation with Transformers

## Project Overview

This project implements two state-of-the-art approaches for 3D hand pose estimation using Transformer-based neural networks. The project is divided into two parts, each addressing different aspects of the hand pose estimation problem using the Ego-Exo4D dataset.

### Part 1: 2D-to-3D Hand Pose Lifting with Transformer
A Transformer-based model that takes 2D hand keypoints as input and predicts the corresponding 3D hand pose in camera coordinates. This approach focuses on the "lifting" problem - converting 2D joint locations to 3D space.

### Part 2: Direct 3D Hand Pose Estimation with POTTER
An end-to-end solution using the POTTER (POoling aTtention TransformER) architecture that directly predicts 3D hand poses from RGB images, eliminating the need for intermediate 2D keypoint detection.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Technical Details](#technical-details)

---

## Features

- **Two Complementary Approaches**: Implements both 2D-to-3D lifting and direct image-to-3D prediction
- **Transformer Architecture**: Leverages self-attention mechanisms for spatial relationship modeling
- **Transfer Learning**: Uses pretrained POTTER classification weights for improved convergence
- **Comprehensive Evaluation**: Includes MPJPE and PA-MPJPE metrics for rigorous performance assessment
- **Visualization Tools**: 3D hand pose visualization for qualitative analysis
- **WandB Integration**: Training monitoring and experiment tracking

---

## Project Structure

```
3D-Hand-Pose-Estimation/
│
├── pose_transformer/                 # Part 1: 2D-to-3D Lifting
│   ├── pose_transformer.ipynb        # Main training notebook
│   ├── dataset/
│   │   ├── dataset.py                # Dataset loader and preprocessor
│   │   └── data_vis.py               # 2D/3D visualization utilities
│   ├── model/
│   │   └── model.py                  # PoseTransformer implementation
│   ├── utils/
│   │   ├── utils.py                  # Training utilities
│   │   └── loss.py                   # Loss functions and metrics
│   └── output/                       # Trained model checkpoints
│
├── potter_architecture/              # Part 2: Image-to-3D with POTTER
│   ├── potter_pose_estimation.ipynb  # Main training notebook
│   ├── configs/
│   │   └── potter_pose_3d_ego4d.yaml # Model configuration
│   ├── dataset/
│   │   ├── dataset.py                # Dataset loader with image processing
│   │   └── dataset_vis.py            # 3D visualization utilities
│   ├── model/
│   │   ├── potter.py                 # POTTER architecture components
│   │   └── model.py                  # PoolAttnHR_Pose_3D model
│   └── utils/
│       ├── functions.py              # Utility functions
│       └── loss.py                   # Loss functions and metrics
│
├── img_dir/                          # Ego-Exo4D image dataset
│   ├── train/
│   ├── val/
│   └── test/
│
├── ego_pose_gt_anno_train.json      # Training annotations
├── ego_pose_gt_anno_val.json        # Validation annotations
└── ego_pose_gt_anno_test.json       # Test annotations
```

---

## Dataset

### Ego-Exo4D Dataset
This project uses the **Ego-Exo4D** dataset, which contains egocentric video recordings with hand pose annotations. The dataset includes:

- **21 hand keypoints** per hand (following standard hand joint topology)
- **2D keypoint annotations** in image coordinates
- **3D keypoint annotations** in camera coordinate system
- **Camera intrinsic parameters** for 2D-3D projection
- **Visibility flags** for handling occlusions

### Data Organization

**Part 1** requires:
- Annotation JSON files with 2D and 3D hand keypoints
- Format: `ego_pose_gt_anno_{split}.json` where split is train/val/test

**Part 2** requires:
- Annotation JSON files (same as Part 1 but with different naming convention)
- RGB images organized by split and sequence
- Format: `img_dir/{split}/{sequence_name}/{frame_id}.jpg`

### Data Preprocessing

1. **3D Pose Normalization**: 3D keypoints are offset by wrist position (index 0) to predict relative positions
2. **Statistical Normalization**: Both 2D and 3D keypoints are normalized using mean and standard deviation
3. **Image Preprocessing**: Images are cropped around hand regions and normalized using ImageNet statistics
4. **Missing Data Handling**: Invalid keypoints are filtered using visibility flags

---

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:

For Part 1:
```bash
cd pose_transformer
pip install torch torchvision numpy matplotlib opencv-python tqdm easydict wandb
```

For Part 2:
```bash
cd potter_architecture
pip install -r requirement.txt
```

3. **Download the dataset**:
   - Download Ego-Exo4D annotation JSON files
   - Download corresponding RGB images
   - Place annotations in the project root directory
   - Place images in `img_dir/` with train/val/test splits

4. **Download pretrained weights** (Part 2 only):
   - Download POTTER classification pretrained weights for transfer learning

5. **Configure WandB**:
```bash
wandb login
```

---

## Usage

### Part 1: 2D-to-3D Hand Pose Lifting

1. **Open the notebook**:
```bash
jupyter notebook pose_transformer/pose_transformer.ipynb
```

2. **Configure paths**: Update `data_root_path` to point to your annotation directory

3. **Train the model**:
   - Run all cells sequentially
   - Model trains for 30 epochs by default
   - Best model is saved based on validation loss

4. **Evaluate**: The notebook includes evaluation on the test set with MPJPE and PA-MPJPE metrics

### Part 2: Direct 3D Hand Pose Estimation

1. **Open the notebook**:
```bash
jupyter notebook potter_architecture/potter_pose_estimation.ipynb
```

2. **Configure paths**: Update the `cfg` dictionary with:
   - `anno_dir`: Path to annotation files
   - `img_dir`: Path to image directory
   - `potter_cls_weight`: Path to pretrained classification weights

3. **Train the model**:
   - Run all cells sequentially
   - Model trains for 15 epochs by default
   - Uses transfer learning from pretrained POTTER classifier

4. **Evaluate**: The notebook includes evaluation with visualization

---

## Model Architectures

### Part 1: PoseTransformer

![PoseTransformer Architecture](pose_transformer/imgs/model.png)

**Architecture Components**:
- **Input Embedding**: Linear projection of 2D keypoints (21, 2) → (21, D)
- **Positional Encoding**: Added to preserve spatial relationships
- **Transformer Encoder Blocks** (repeated 4 times):
  - Multi-head Self-Attention (MSA) with 8 heads
  - MLP layers with expansion ratio of 2
  - Layer Normalization
  - Residual connections
- **MLP Head**: Projects embeddings (21, D) → (21, 3)

**Key Design Choices**:
- Embedding dimension: 32
- Number of heads: 8
- Encoder depth: 4 layers
- Each joint is treated as a "patch" in the sequence

**Innovation**: Adapts Vision Transformer (ViT) paradigm to the structured problem of hand pose estimation, where each joint has semantic meaning.

### Part 2: POTTER (POoling aTtention TransformER)

> **Reference**: [POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery](https://arxiv.org/abs/2311.18259)  
> Li, Ce and Lee, Gim Hee. *arXiv preprint arXiv:2311.18259*, 2023.

![POTTER Architecture](potter_architecture/imgs/POTTER_arch.png)

**Two-Stage Architecture**:

1. **Basic Stream** (Hierarchical feature extraction):
   - 4 stages with progressively reduced spatial resolution
   - Captures global context and semantic information
   - Similar to standard CNN feature pyramids

2. **HR Stream** (High-resolution feature preservation):
   - Maintains high spatial resolution throughout
   - Fuses with Basic Stream features via Patch Split blocks
   - Preserves fine-grained spatial details

**Core Innovation - Pooling Attention Block (PAT)**:

![Pooling Attention Details](potter_architecture/imgs/PAT.png)

Traditional self-attention has O(N²) complexity. PAT reduces this through:

1. **Patch-wise Pooling Attention**:
   - Applies adaptive pooling along spatial dimensions
   - Computes cross-attention between pooled features
   - Preserves spatial structure while reducing computation

2. **Embed-wise Pooling Attention**:
   - Similar pooling along embedding dimensions
   - Captures channel relationships efficiently

**Advantages over Standard Transformers**:
- Reduced computational complexity
- Better scaling to high-resolution images
- Hierarchical feature learning
- More efficient for dense prediction tasks

**Comparison of Transformer Architectures**:

![Transformer Block Comparison](potter_architecture/imgs/transformer_blocks.png)

The figure above shows the evolution from standard attention-based transformers (ViT, Swin) to MLP-based mixers, and finally to POTTER's Pooling Attention design, which combines efficiency with effectiveness.

---

## Performance Metrics

### MPJPE (Mean Per-Joint Position Error)
- **Definition**: Average Euclidean distance between predicted and ground truth 3D joints
- **Units**: Millimeters (mm)
- **Formula**: 
  ```
  MPJPE = (1/N) Σ ||pred_i - gt_i||₂
  ```
- **Interpretation**: Lower is better; measures absolute prediction accuracy

### PA-MPJPE (Procrustes Analysis MPJPE)
- **Definition**: MPJPE after optimal alignment using Procrustes analysis
- **Removes**: Global rotation, translation, and scale differences
- **Purpose**: Measures pose structure accuracy independent of global transformations
- **Interpretation**: Lower is better; represents the model's upper-bound performance

---

## Results

### Part 1: 2D-to-3D Lifting

**Expected Performance**:
- **MPJPE**: ~22-25 mm
- **PA-MPJPE**: ~7-10 mm
- **Training Time**: ~30 minutes for 30 epochs (on GPU)

**Model Output Sample** (from training log):
```
Model performance on test set: MPJPE: 22.85 (mm) PA-MPJPE: 7.53 (mm)
```

### Part 2: Image-to-3D with POTTER

**Expected Performance**:
- **MPJPE**: ~35 mm
- **PA-MPJPE**: ~15 mm
- **Training Time**: Longer due to image processing (15 epochs)

**Note**: Part 2 is more challenging as it must first extract 2D features from images before lifting to 3D.

---

## Technical Details

### Hand Joint Topology

![Hand Joint Structure](pose_transformer/imgs/hand_index.png)

The model predicts 21 keypoints per hand following this structure:
- **Index 0**: Wrist (root joint)
- **Indices 1-4**: Thumb (4 joints)
- **Indices 5-8**: Index finger (4 joints)
- **Indices 9-12**: Middle finger (4 joints)
- **Indices 13-16**: Ring finger (4 joints)
- **Indices 17-20**: Pinky finger (4 joints)

### Coordinate Systems

**2D Coordinates** (u, v):
- Image plane coordinates (pixels)
- Origin at top-left corner

**3D Coordinates** (X_c, Y_c, Z_c):
- Camera coordinate system
- Wrist-relative (root-normalized)
- Units: meters (converted to millimeters for metrics)

**Projection Relationship**:
```
λ [u, v, 1]ᵀ = K [X_c, Y_c, Z_c]ᵀ
```
Where:
- λ: Scale factor (depth)
- K: Camera intrinsic matrix

### Training Strategies

**Data Augmentation**: Minimal, as annotations are derived from real videos

**Loss Function**: 
- Mean Squared Error (MSE) with visibility-based weighting
- Only valid keypoints contribute to loss

**Optimization**:
- Adam optimizer
- Learning rates: 2e-4 (Part 1), 1e-4 (Part 2)
- No learning rate scheduling in default config

**Validation Strategy**:
- Best model selected based on validation loss
- Early stopping not implemented (fixed epochs)

### Handling Challenges

1. **Scale Ambiguity**: Resolved by wrist-relative coordinates
2. **Missing Annotations**: Filtered using visibility flags
3. **Depth Ambiguity**: Inherent in 2D-to-3D lifting; partially resolved by learning data priors
4. **Occlusions**: Handled through visibility masks and robust training

---

## Visualization

Both notebooks include 3D visualization capabilities:

```python
# Visualize 3D hand pose
vis_data_3d(keypoints_3d, title="3D Hand Pose")
```

The visualization:
- Plots 21 hand keypoints in 3D space
- Shows skeletal connections between joints
- Supports rotation for better viewing
- Can display ground truth and predictions side-by-side

---

## Citation

If you use this code or the datasets/architectures, please cite the original papers:

**POTTER Architecture**:
```bibtex
@article{li2023potter,
  title={POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery},
  author={Li, Ce and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2311.18259},
  year={2023},
  url={https://arxiv.org/abs/2311.18259}
}
```

**Ego-Exo4D Dataset**:
```bibtex
@inproceedings{ego-exo4d,
  title={Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives},
  author={Grauman, Kristen and Westbury, Andrew and others},
  booktitle={CVPR},
  year={2024}
}
```

---

## License

This project is for educational purposes. Please refer to the original dataset and model licenses for usage rights.

---

## Acknowledgments

- **POTTER Architecture**: Based on [Li & Lee (2023)](https://arxiv.org/abs/2311.18259)
- **Dataset**: Ego-Exo4D team
- **Architecture Inspiration**: Vision Transformer (ViT)
- **Framework**: PyTorch and the open-source computer vision community

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory errors
- **Solution**: Reduce batch size in the configuration

**Issue**: Dataset not found
- **Solution**: Verify paths in configuration match your download locations

**Issue**: Slow training
- **Solution**: Ensure CUDA is available and being used (`torch.cuda.is_available()`)

**Issue**: Poor performance
- **Solution**: Check data preprocessing, ensure pretrained weights loaded correctly (Part 2)

### Support

For questions or issues:
- Check the documentation in the notebooks
- Review the troubleshooting section above
- Open an issue on the GitHub repository

