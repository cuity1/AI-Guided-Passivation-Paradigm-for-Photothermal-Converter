# Graph Neural Networks for Molecular Property Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of advanced Graph Neural Network (GNN) architectures for molecular property prediction tasks. The framework supports two main applications:

1. **General Molecular Classification** - Binary classification of molecules based on SMILES representations
2. **MXene Binding Prediction** - Specialized prediction of molecular binding to MXene surfaces

## Key Features

- [object Object] Molecular Featurization**: Comprehensive atom, bond, and global molecular featu[object Object]ultiple GNN Architectures**: Support for GCN, GAT, GIN, and GraphConv layers
- ⚡ **Automated Hyperparameter Optimization**: Integration with Optuna for efficient parameter tuning
- [object Object]mprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC-ROC
- 🎯 **Class Imbalance Handling**: Weighted loss functions for imbalanced datasets
- 🔄 **Reproducibility**: Fixed random seeds and deterministic operations

## Project Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation information
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment file
├── config/
│   ├── config_general.yaml           # Configuration for general classification
│   └── config_mxene.yaml             # Configuration for MXene binding prediction
├── src/
│   ├── __init__.py
│   ├── molecular_classification.py   # General molecular classification (refactored code1.py)
│   ├── mxene_binding_prediction.py   # MXene binding prediction (refactored code2.py)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_models.py            # GNN model architectures
│   │   └── layers.py                # Custom GNN layers
│   ├── data/
│   │   ├── __init__.py
│   │   ├── featurization.py         # Molecular featurization functions
│   │   └── data_loader.py           # Data loading utilities
│   └── utils/
│       ├── __init__.py
│       ├── training.py              # Training utilities
│       ├── evaluation.py            # Evaluation metrics
│       └── visualization.py         # Visualization functions
├── data/
│   ├── data.csv                      # General classification dataset
│   └── Binding_energy.csv            # MXene binding dataset
├── examples/
│   ├── train_general_classifier.py   # Example training script
│   └── train_mxene_predictor.py      # Example MXene prediction script
├── notebooks/
│   └── demo.ipynb                    # Jupyter notebook demonstration
└── tests/
    ├── test_featurization.py
    ├── test_models.py
    └── test_training.py
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/molecular-gnn.git
cd molecular-gnn

# Create conda environment
conda env create -f environment.yml
conda activate mol-gnn

# Install the package
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/molecular-gnn.git
cd molecular-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### General Molecular Classification

```python
from src.molecular_classification import MolecularClassifier

# Initialize classifier
classifier = MolecularClassifier(
    data_path='data/data.csv',
    output_dir='results/general',
    random_seed=42
)

# Run hyperparameter optimization
best_params = classifier.optimize_hyperparameters(n_trials=50)

# Train final model
classifier.train_final_model(best_params)

# Evaluate
results = classifier.evaluate()
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1-Score: {results['f1_score']:.4f}")
```

### MXene Binding Prediction

```python
from src.mxene_binding_prediction import MXeneBindingPredictor

# Initialize predictor
predictor = MXeneBindingPredictor(
    data_path='data/Binding_energy.csv',
    output_dir='results/mxene',
    random_seed=42
)

# Run hyperparameter optimization
best_params = predictor.optimize_hyperparameters(n_trials=100)

# Train and evaluate
predictor.train_final_model(best_params)
results = predictor.evaluate()

print(f"Test AUC-ROC: {results['auc_roc']:.4f}")
```

## Usage

### Command Line Interface

```bash
# General molecular classification
python src/molecular_classification.py \
    --data_path data/data.csv \
    --output_dir results/general \
    --n_trials 50 \
    --epochs 250

# MXene binding prediction
python src/mxene_binding_prediction.py \
    --data_path data/Binding_energy.csv \
    --output_dir results/mxene \
    --n_trials 100 \
    --epochs 200
```

### Configuration Files

You can also use YAML configuration files:

```bash
# Using config file
python src/molecular_classification.py --config config/config_general.yaml
```

## Model Architecture

### Enhanced GNN Model

The framework implements a flexible GNN architecture with the following components:

1. **Node Feature Encoder**: Multi-layer perceptron for atom feature encoding
2. **Edge Feature Encoder**: Processing of bond information
3. **Global Feature Encoder**: Molecular-level descriptor processing
4. **GNN Layers**: Configurable graph convolution layers (GCN/GAT/GIN/GraphConv)
5. **Graph Pooling**: Multiple pooling strategies (mean/max/add/combined/attention)
6. **Classifier**: Multi-layer classification head with dropout and batch normalization

### Molecular Featurization

#### Atom Features (33 dimensions for general, 35 for MXene)
- Atomic number (one-hot encoding for common elements)
- Degree and valence information
- Hybridization state
- Ring membership and aromaticity
- Chirality
- Hydrogen count
- Electronegativity and polarizability (MXene-specific)

#### Bond Features (7-10 dimensions)
- Bond type (single/double/triple/aromatic)
- Ring membership
- Conjugation
- Stereochemistry
- Bond order

#### Global Molecular Features (18-31 dimensions)
- Molecular weight and heavy atom count
- Hydrogen bond donors/acceptors
- Topological polar surface area (TPSA)
- Number of rotatable bonds
- Aromatic rings count
- LogP and other physicochemical properties
- Coordination sites and electron-rich groups (MXene-specific)

## Hyperparameter Optimization

The framework uses Optuna for Bayesian hyperparameter optimization with the following search space:

- **Batch size**: [32, 64, 128, 256]
- **Hidden channels**: [128, 256, 384, 512]
- **Dropout rate**: [0.1, 0.5]
- **Learning rate**: [1e-4, 1e-2] (log scale)
- **Weight decay**: [1e-6, 1e-3] (log scale)
- **GNN type**: [GCN, GAT, GIN, GraphConv]
- **Pooling type**: [mean, max, add, combined, attention]
- **Number of layers**: [2, 5]
- **Attention heads**: [2, 4, 8] (for GAT)
- **Activation function**: [ReLU, GELU, Swish]


## Reproducibility

All experiments are fully reproducible with fixed random seeds:

```python
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```





## Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- RDKit developers for molecular informatics tools
- Optuna team for hyperparameter optimization framework

## Contact

For questions and feedback, please open an issue on GitHub 



