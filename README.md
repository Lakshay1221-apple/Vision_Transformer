# Vision Transformer for CIFAR-10

A compact PyTorch implementation of a Vision Transformer (ViT) trained on the CIFAR-10 image classification dataset. The project is organized as a notebook-first workflow that walks through data preparation, patch embedding, transformer encoding, optimization, and training.

## Overview

This repository demonstrates how to build a small Vision Transformer from core PyTorch components. CIFAR-10 images are split into fixed-size patches, projected into an embedding space, enriched with positional information, processed by stacked transformer encoder blocks, and classified through a learnable class token.

The implementation is intentionally lightweight, making it useful for learning the mechanics of ViT architecture without relying on a high-level model library.

## Project Structure

```text
.
|-- README.md
`-- ViT.ipynb
```

## Key Features

- Vision Transformer architecture implemented with PyTorch modules
- CIFAR-10 training and validation pipeline using `torchvision`
- Image augmentation with random cropping and horizontal flipping
- Patch embedding through a convolutional projection layer
- Learnable class token and positional embeddings
- Multi-head self-attention based transformer encoder blocks
- AdamW optimization with cross-entropy loss
- Professional markdown documentation inside the notebook

## Model Configuration

| Parameter | Value |
| --- | --- |
| Dataset | CIFAR-10 |
| Image size | 32 x 32 |
| Input channels | 3 |
| Patch size | 4 x 4 |
| Number of patches | 64 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer blocks | 4 |
| MLP hidden units | 128 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Optimizer | AdamW |
| Loss function | Cross entropy |
| Epochs | 5 |

## Requirements

Install the core dependencies before running the notebook:

```bash
pip install torch torchvision matplotlib notebook
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation selector.

## Usage

1. Clone or open this repository.
2. Install the required Python packages.
3. Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

4. Open `ViT.ipynb`.
5. Run the cells from top to bottom.

The notebook downloads CIFAR-10 automatically into a local `data/` directory when the dataset cells are executed.

## Notebook Workflow

The notebook is structured into the following sections:

1. Import required PyTorch, TorchVision, and plotting libraries.
2. Define CIFAR-10 training and validation transforms.
3. Configure model and training hyperparameters.
4. Build training and validation data loaders.
5. Implement the patch embedding layer.
6. Implement the transformer encoder block.
7. Implement the classification head.
8. Assemble the full Vision Transformer model.
9. Configure device selection, optimizer, and loss function.
10. Train the model and report batch-level and epoch-level metrics.

## Architecture Summary

The model follows the standard ViT pattern:

```text
Input Image
    |
Patch Embedding
    |
Class Token + Positional Embedding
    |
Transformer Encoder Blocks
    |
Class Token Representation
    |
MLP Classification Head
    |
CIFAR-10 Class Logits
```

## CIFAR-10 Classes

CIFAR-10 contains 10 image categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Notes

- The current project is notebook-based and designed for experimentation and learning.
- Training speed and final accuracy depend on hardware, random initialization, and the number of epochs.
- The dataset is downloaded automatically through `torchvision.datasets.CIFAR10`.
- GPU training is used when CUDA is available; otherwise, the notebook falls back to CPU.

## Possible Improvements

- Add validation metrics inside the training loop.
- Save and reload trained model checkpoints.
- Plot training loss and accuracy curves.
- Add inference examples for individual images.
- Move reusable model code into a Python module.
- Add a `requirements.txt` file for reproducible setup.

## License

No license has been specified yet. Add a license file before distributing or reusing this project publicly.
