"""
Enhanced Graph Neural Network for Molecular Classification

This module implements an advanced GNN framework for binary molecular classification
using SMILES representations. It includes comprehensive molecular featurization,
multiple GNN architectures, and automated hyperparameter optimization.

"""

import os
import warnings
import argparse
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import joblib

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration and Setup
# ============================================================================

class Config:
    """Configuration class for model training and evaluation."""
    
    def __init__(self):
        self.seed = 42
        self.test_size = 0.2
        self.val_size = 0.15
        self.n_trials = 50
        self.max_epochs = 250
        self.early_stopping_patience = 25
        self.output_dir = 'results'
        self.model_dir = 'models'
        self.optuna_dir = 'optuna_results'
        self.viz_dir = 'visualization_results'
        
    def create_directories(self):
        """Create necessary output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.optuna_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Molecular Featurization
# ============================================================================

def count_electron_withdrawing_groups(mol: Chem.Mol) -> int:
    """
    Count electron withdrawing groups in a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Number of electron withdrawing groups
    """
    ewg_patterns = [
        '[C,c](=O)[O,o,N,n]',      # Carboxylic acid, ester, amide
        '[N+](=O)[O-]',            # Nitro
        '[C,c](=O)[C,c]',          # Ketone
        '[C,c](=O)[H]',            # Aldehyde
        '[S](=O)(=O)',             # Sulfone, sulfonic acid
        '[C,c][F,Cl,Br,I]',        # Halogenated alkane
        '[c][F,Cl,Br,I]',          # Halogenated aromatic
        '[C]#[N]',                 # Nitrile
        '[N]=[N+]=[N-]',           # Azide
        '[N]=[O]',                 # Nitroso
        '[P](=O)(O)(O)',           # Phosphate ester
        '[S](=O)(=O)[N]',          # Sulfonamide
    ]
    
    count = 0
    for pattern in ewg_patterns:
        smarts = Chem.MolFromSmarts(pattern)
        if smarts:
            count += len(mol.GetSubstructMatches(smarts))
    return count


def count_electron_donating_groups(mol: Chem.Mol) -> int:
    """
    Count electron donating groups in a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Number of electron donating groups
    """
    edg_patterns = [
        '[O,o][C,c]',                 # Ether, alcohol
        '[N,n][C,c]',                 # Amine
        '[S][C,c]',                   # Thioether
        '[c][O][H]',                  # Phenol
        '[N,n]([C,c])[C,c]',          # Secondary amine
        '[N,n]([C,c])([C,c])[C,c]',   # Tertiary amine
        '[NH2]',                      # Primary amine
        '[OH]',                       # Free hydroxyl
        '[SH]',                       # Thiol
    ]
    
    count = 0
    for pattern in edg_patterns:
        smarts = Chem.MolFromSmarts(pattern)
        if smarts:
            count += len(mol.GetSubstructMatches(smarts))
    return count


def get_atom_features(atom: Chem.Atom) -> List[float]:
    """
    Extract comprehensive atom-level features.
    
    Args:
        atom: RDKit atom object
        
    Returns:
        List of atom features
    """
    features = [
        # One-hot encoding for common atom types
        atom.GetAtomicNum() == 6,    # C
        atom.GetAtomicNum() == 7,    # N
        atom.GetAtomicNum() == 8,    # O
        atom.GetAtomicNum() == 9,    # F
        atom.GetAtomicNum() == 15,   # P
        atom.GetAtomicNum() == 16,   # S
        atom.GetAtomicNum() == 17,   # Cl
        atom.GetAtomicNum() == 35,   # Br
        atom.GetAtomicNum() == 53,   # I
        
        # Other atom types (grouped)
        atom.GetAtomicNum() < 6,
        atom.GetAtomicNum() > 9 and atom.GetAtomicNum() < 15,
        atom.GetAtomicNum() > 17 and atom.GetAtomicNum() < 35,
        atom.GetAtomicNum() > 35 and atom.GetAtomicNum() < 53,
        atom.GetAtomicNum() > 53,
        
        # Structural features
        atom.GetDegree(),
        atom.GetTotalDegree(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        atom.GetHybridization().real,
        
        # Boolean features
        atom.IsInRing() * 1.0,
        atom.GetIsAromatic() * 1.0,
        atom.IsInRingSize(3) * 1.0,
        atom.IsInRingSize(4) * 1.0,
        atom.IsInRingSize(5) * 1.0,
        atom.IsInRingSize(6) * 1.0,
        atom.IsInRingSize(7) * 1.0,
        atom.IsInRingSize(8) * 1.0,
        
        # Chirality
        int(atom.GetChiralTag()),
        
        # Hydrogen atoms
        atom.GetTotalNumHs(includeNeighbors=True),
        
        # Additional features
        Crippen.MolLogP(Chem.MolFromSmiles('[' + atom.GetSymbol() + ']')),
        float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0,
        float(atom.GetProp('_GasteigerHCharge')) if atom.HasProp('_GasteigerHCharge') else 0.0,
    ]
    
    return features


def get_bond_features(bond: Chem.Bond) -> List[float]:
    """
    Extract bond-level features.
    
    Args:
        bond: RDKit bond object
        
    Returns:
        List of bond features
    """
    bond_type = bond.GetBondType()
    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing(),
        bond.GetIsConjugated(),
        bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE
    ]
    
    return features


def get_global_features(mol: Chem.Mol) -> List[float]:
    """
    Extract global molecular features.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of global molecular features
    """
    features = [
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        mol.GetNumAtoms(),
        Descriptors.FractionCSP3(mol),
        count_electron_withdrawing_groups(mol),
        count_electron_donating_groups(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumAromaticCarbocycles(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumHeteroatoms(mol)
    ]
    
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        PyTorch Geometric Data object or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Extract atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = get_atom_features(atom)
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Extract edge features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            features = get_bond_features(bond)
            
            # Add bidirectional edges
            edge_indices.append([i, j])
            edge_features.append(features)
            edge_indices.append([j, i])
            edge_features.append(features)
        
        # Handle molecules with no bonds
        if len(edge_indices) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 7), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Extract global features
        global_features = torch.tensor([get_global_features(mol)], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                   global_features=global_features)
        return data
        
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return None


# ============================================================================
# Model Architecture
# ============================================================================

class EnhancedGNN(torch.nn.Module):
    """
    Enhanced Graph Neural Network for molecular property prediction.
    
    This model supports multiple GNN architectures (GCN, GAT) with flexible
    pooling strategies and comprehensive feature integration.
    
    Args:
        feature_size: Dimension of node features
        edge_dim: Dimension of edge features
        global_dim: Dimension of global features
        hidden_channels: Number of hidden channels
        num_classes: Number of output classes
        dropout: Dropout rate
        gnn_type: Type of GNN layer ('gcn' or 'gat')
        pool_type: Type of pooling ('mean', 'max', 'add', or 'combined')
        num_layers: Number of GNN layers
        heads: Number of attention heads (for GAT)
    """
    
    def __init__(self, feature_size: int, edge_dim: int, global_dim: int, 
                 hidden_channels: int, num_classes: int, dropout: float = 0.2, 
                 gnn_type: str = 'gat', pool_type: str = 'combined', 
                 num_layers: int = 3, heads: int = 4):
        super(EnhancedGNN, self).__init__()
        
        self.gnn_type = gnn_type
        self.pool_type = pool_type
        self.num_layers = num_layers
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU()
        )
        
        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.projs = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
                self.projs.append(None)
                
            elif gnn_type == 'gat':
                out_channels = hidden_channels // heads
                if hidden_channels % heads != 0:
                    out_channels = hidden_channels // heads + 1
                    
                self.convs.append(GATConv(hidden_channels, out_channels, 
                                        heads=heads, dropout=dropout, concat=True))
                actual_output_dim = out_channels * heads
                self.bns.append(nn.BatchNorm1d(actual_output_dim))
                
                if actual_output_dim != hidden_channels:
                    self.projs.append(nn.Linear(actual_output_dim, hidden_channels))
                else:
                    self.projs.append(None)
        
        # Determine pooled feature dimension
        if self.pool_type == 'combined':
            pooled_dim = hidden_channels * 2
        else:
            pooled_dim = hidden_channels
        
        # Classifier
        if global_dim > 0:
            self.lin1 = nn.Linear(pooled_dim + hidden_channels, hidden_channels)
        else:
            self.lin1 = nn.Linear(pooled_dim, hidden_channels)
            
        self.lin2 = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, batch: torch.Tensor, 
                global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the GNN model.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment vector
            global_features: Global molecular features
            
        Returns:
            Output logits
        """
        # Encode features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            identity = x
            x = self.convs[i](x, edge_index)
            
            if self.gnn_type == 'gat' and self.projs[i] is not None:
                x = self.bns[i](x)
                x = F.relu(x)
                x = self.projs[i](x)
            else:
                x = self.bns[i](x)
                x = F.relu(x)
            
            # Residual connection
            if x.size(-1) == identity.size(-1):
                x = x + identity
            
            if i < self.num_layers - 1:
                x = self.dropout(x)
        
        # Graph pooling
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            x = global_max_pool(x, batch)
        elif self.pool_type == 'add':
            x = global_add_pool(x, batch)
        elif self.pool_type == 'combined':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        
        # Merge global features
        if global_features is not None:
            global_feat = self.global_encoder(global_features)
            x = torch.cat([x, global_feat], dim=1)
        
        # Classification
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x


# ============================================================================
# Training and Evaluation
# ============================================================================

def calculate_class_weights(labels: List[int], device: torch.device) -> Tuple[torch.Tensor, int]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: List of class labels
        device: PyTorch device
        
    Returns:
        Tuple of (class weights tensor, number of classes)
    """
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    weights = torch.tensor(
        [len(labels) / (num_classes * class_counts[c]) for c in sorted(class_counts.keys())],
        dtype=torch.float, device=device
    )
    return weights, num_classes


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                class_weights: torch.Tensor, device: torch.device) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        loader: Data loader
        optimizer: Optimizer
        class_weights: Class weights for loss function
        device: PyTorch device
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        if hasattr(data, 'global_features'):
            global_features = data.global_features.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_features)
        else:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        loss = F.cross_entropy(out, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        loader: Data loader
        device: PyTorch device
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    preds = []
    true = []
    probs_list = []
    
    for data in loader:
        data = data.to(device)
        
        if hasattr(data, 'global_features'):
            global_features = data.global_features.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_features)
        else:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        
        preds.extend(pred.cpu().numpy())
        true.extend(data.y.cpu().numpy())
        probs_list.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true, preds),
        'precision': precision_score(true, preds, average='weighted', zero_division=0),
        'recall': recall_score(true, preds, average='weighted', zero_division=0),
        'f1_score': f1_score(true, preds, average='weighted', zero_division=0)
    }
    
    # Calculate AUC for binary classification
    if len(set(true)) == 2:
        try:
            metrics['auc_roc'] = roc_auc_score(true, [p[1] for p in probs_list])
        except:
            metrics['auc_roc'] = 0.0
    
    return metrics


# ============================================================================
# Main Execution
# ============================================================================

def main(args):
    """Main execution function."""
    
    # Setup
    config = Config()
    config.create_directories()
    set_random_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    data = pd.read_csv(args.data_path)
    smiles_list = data.iloc[:, 0].values
    labels = data.iloc[:, 1].values
    print(f"Loaded {len(smiles_list)} SMILES records")
    print(f"Label distribution: {Counter(labels)}")
    
    # Process molecular graphs
    print("\nProcessing molecular graphs...")
    data_list = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph.y = torch.tensor([labels[i]], dtype=torch.long)
            data_list.append(graph)
            valid_indices.append(i)
    
    print(f"Successfully processed {len(data_list)}/{len(smiles_list)} molecules")
    
    # Normalize global features
    if len(data_list) > 0 and hasattr(data_list[0], 'global_features'):
        global_features = torch.cat([data.global_features for data in data_list], dim=0)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(global_features.numpy())
        
        for i, data in enumerate(data_list):
            data.global_features = torch.tensor(normalized_features[i:i+1], dtype=torch.float)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(config.model_dir, 'feature_scaler.pkl'))
    
    # Split dataset
    valid_labels = [labels[i] for i in valid_indices]
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_list, valid_labels, test_size=config.test_size, 
        random_state=config.seed, stratify=valid_labels)
    
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=config.val_size, 
        random_state=config.seed, stratify=train_labels)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Get model parameters
    num_features = data_list[0].x.shape[1]
    edge_dim = data_list[0].edge_attr.shape[1]
    global_dim = data_list[0].global_features.shape[1] if hasattr(data_list[0], 'global_features') else 0
    class_weights, num_classes = calculate_class_weights(train_labels, device)
    
    print(f"\nModel parameters:")
    print(f"  Node features: {num_features}")
    print(f"  Edge features: {edge_dim}")
    print(f"  Global features: {global_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class weights: {class_weights}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced GNN for Molecular Classification')
    parser.add_argument('--data_path', type=str, default='data.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=250,
                       help='Maximum number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)

