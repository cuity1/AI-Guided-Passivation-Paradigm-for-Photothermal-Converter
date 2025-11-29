import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, GraphConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import optuna
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
data = pd.read_csv('Binding_energy.csv')
smiles_list = data['smiles'].values
binding_energies = data['bsse'].values
labels = data['label'].values  # Binary labels (0 or 1)

print(f"Loaded {len(smiles_list)} SMILES records")
print(f"Label distribution: {Counter(labels)}")

# MXene-binding specific functions
def _get_electronegativity(atomic_num):
    """Return element electronegativity"""
    electronegativity = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }
    return electronegativity.get(atomic_num, 2.0)

def _get_polarizability(atomic_num):
    """Return atomic polarizability approximation"""
    polarizability = {
        1: 0.67, 6: 1.76, 7: 1.10, 8: 0.80, 9: 0.56,
        15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05, 53: 5.35
    }
    return polarizability.get(atomic_num, 2.0)

def _is_electron_rich_atom(atom):
    """Check if atom is electron-rich (N, O, S)"""
    return int(atom.GetAtomicNum() in [7, 8, 16])

def _is_coordination_atom(atom):
    """Check if atom is potential coordination site"""
    return int(atom.GetAtomicNum() in [7, 8, 16, 15])

def _count_coordination_sites(mol):
    """Count potential coordination sites"""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 8, 16, 15]:  # N, O, S, P
            count += 1
    return count

def _count_electron_rich_groups(mol):
    """Count electron-rich functional groups"""
    patterns = [
        '[NH2]', '[OH]', '[SH]', '[N]', '[O]', '[S]',
        'c1ccccc1',  # benzene ring
        '[#7]', '[#8]', '[#16]'  # heteroatoms
    ]
    count = 0
    for pattern in patterns:
        try:
            smarts = Chem.MolFromSmarts(pattern)
            if smarts:
                count += len(mol.GetSubstructMatches(smarts))
        except:
            continue
    return count

def _count_pi_systems(mol):
    """Count π-electron systems"""
    pi_count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, 
                                  Chem.rdchem.BondType.TRIPLE, 
                                  Chem.rdchem.BondType.AROMATIC]:
            pi_count += 1
    return pi_count

def _calculate_dipole_moment_proxy(mol):
    """Calculate dipole moment proxy"""
    polar_atoms = sum(1 for atom in mol.GetAtoms() 
                     if atom.GetAtomicNum() in [7, 8, 9, 16, 17, 35, 53])
    return polar_atoms / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0

def _count_functional_groups(mol):
    """Count functional groups relevant to MXene binding"""
    patterns = [
        '[CX3](=O)', '[NX3]', '[OX2H]', '[SX2H]',  # carbonyl, amine, hydroxyl, thiol
        '[CX3](=O)[OX2H]', '[CX3](=O)[NX3]',       # carboxylic acid, amide
        'c1ccccc1',  # aromatic ring
    ]
    count = 0
    for pattern in patterns:
        try:
            smarts = Chem.MolFromSmarts(pattern)
            if smarts:
                count += len(mol.GetSubstructMatches(smarts))
        except:
            continue
    return count

def smiles_to_graph_mxene_enhanced(smiles):
    """
    Convert SMILES to molecular graph with MXene-binding optimized features
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens for accurate features
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates if possible
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
            
        # Extended atom features for MXene binding
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                # Basic atom types
                atom.GetAtomicNum() == 1,    # H
                atom.GetAtomicNum() == 6,    # C
                atom.GetAtomicNum() == 7,    # N
                atom.GetAtomicNum() == 8,    # O
                atom.GetAtomicNum() == 16,   # S
                atom.GetAtomicNum() == 9,    # F
                atom.GetAtomicNum() == 15,   # P
                atom.GetAtomicNum() == 17,   # Cl
                atom.GetAtomicNum() == 35,   # Br
                atom.GetAtomicNum() == 53,   # I
                
                # Valence and degree features
                atom.GetDegree(),
                atom.GetTotalDegree(),
                atom.GetImplicitValence(),
                atom.GetExplicitValence(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                float(atom.GetHybridization()),
                
                # Ring-related features
                atom.IsInRing() * 1.0,
                atom.GetIsAromatic() * 1.0,
                atom.IsInRingSize(3) * 1.0,
                atom.IsInRingSize(4) * 1.0,
                atom.IsInRingSize(5) * 1.0,
                atom.IsInRingSize(6) * 1.0,
                atom.IsInRingSize(7) * 1.0,
                
                # Hydrogen-related features
                atom.GetTotalNumHs(includeNeighbors=True),
                atom.GetNumImplicitHs(),
                atom.GetNumExplicitHs(),
                
                # Chirality
                int(atom.GetChiralTag()),
                
                # MXene-binding specific features
                _get_electronegativity(atom.GetAtomicNum()),
                _is_electron_rich_atom(atom),
                _is_coordination_atom(atom),
                _get_polarizability(atom.GetAtomicNum()),
                
                # Local environment
                sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() in [7, 8, 16]),
                sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetIsAromatic()),
            ]
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Enhanced edge features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_type = bond.GetBondType()
            features = [
                bond_type == Chem.rdchem.BondType.SINGLE,
                bond_type == Chem.rdchem.BondType.DOUBLE,
                bond_type == Chem.rdchem.BondType.TRIPLE,
                bond_type == Chem.rdchem.BondType.AROMATIC,
                bond.IsInRing(),
                bond.GetIsConjugated(),
                bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE,
                float(bond.GetBondTypeAsDouble()),
                bond.IsInRingSize(5) * 1.0,
                bond.IsInRingSize(6) * 1.0,
            ]
            
            edge_indices.append([i, j])
            edge_features.append(features)
            edge_indices.append([j, i])
            edge_features.append(features)
        
        if len(edge_indices) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 10), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Enhanced global molecular features
        global_features = [
            # Basic descriptors
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            mol.GetNumAtoms(),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.MolWt(mol),
            
            # Lipinski descriptors
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumSaturatedCarbocycles(mol),
            Descriptors.NumSaturatedHeterocycles(mol),
            Descriptors.NumSaturatedRings(mol),
            
            # MXene-binding specific features
            _count_coordination_sites(mol),
            _count_electron_rich_groups(mol),
            _count_pi_systems(mol),
            Descriptors.FractionCSP3(mol),
            Crippen.MolLogP(mol),
            Descriptors.NumHeteroatoms(mol),
            
            # Electronic properties
            _calculate_dipole_moment_proxy(mol),
            _count_functional_groups(mol),
            Descriptors.LabuteASA(mol),
            
            # Additional molecular descriptors
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Ipc(mol),
        ]
        
        global_features = torch.tensor([global_features], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)
        return data
        
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return None

class AdvancedMXeneBindingGNN(torch.nn.Module):
    """Advanced GNN architecture for MXene binding prediction"""
    
    def __init__(self, feature_size, edge_dim, global_dim, hidden_channels, num_classes=2, 
                 dropout=0.2, gnn_type='gat', pool_type='combined', num_layers=3, heads=4,
                 use_residual=True, use_batch_norm=True, activation='relu'):
        super(AdvancedMXeneBindingGNN, self).__init__()
        self.gnn_type = gnn_type
        self.pool_type = pool_type
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        # Activation function selection
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(feature_size, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            self.activation
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            self.activation,
            nn.Linear(hidden_channels, hidden_channels),
            self.activation
        )
        
        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            self.activation
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.projs = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                
            elif gnn_type == 'gat':
                out_channels = hidden_channels // heads
                if hidden_channels % heads != 0:
                    out_channels = hidden_channels // heads + 1
                
                self.convs.append(GATConv(hidden_channels, out_channels, heads=heads, dropout=dropout, concat=True))
                actual_output_dim = out_channels * heads
                
                if actual_output_dim != hidden_channels:
                    self.projs.append(nn.Linear(actual_output_dim, hidden_channels))
                else:
                    self.projs.append(None)
                    
            elif gnn_type == 'gin':
                nn_layer = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    self.activation,
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(nn_layer))
                
            elif gnn_type == 'graphconv':
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
            
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            else:
                self.bns.append(nn.Identity())
        
        # Pooling feature dimension
        if self.pool_type == 'combined':
            pooled_dim = hidden_channels * 3  # mean + max + add
        elif self.pool_type == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, 1)
            )
            pooled_dim = hidden_channels
        else:
            pooled_dim = hidden_channels
        
        # Multi-layer classifier
        if global_dim > 0:
            classifier_input_dim = pooled_dim + hidden_channels
        else:
            classifier_input_dim = pooled_dim
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            nn.Linear(hidden_channels, hidden_channels // 2),
            self.activation,
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, batch, global_features=None):
        # Feature encoding
        x = self.node_encoder(x)
        
        # GNN layer propagation
        for i in range(self.num_layers):
            identity = x
            
            x = self.convs[i](x, edge_index)
            
            if self.gnn_type == 'gat' and self.projs[i] is not None:
                x = self.projs[i](x)
            
            x = self.bns[i](x)
            x = self.activation(x)
            
            # Residual connection
            if self.use_residual and x.size(-1) == identity.size(-1):
                x = x + identity
            
            if i < self.num_layers - 1:
                x = self.dropout(x)
        
        # Graph-level pooling
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            x = global_max_pool(x, batch)
        elif self.pool_type == 'add':
            x = global_add_pool(x, batch)
        elif self.pool_type == 'combined':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_add = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_add], dim=1)
        elif self.pool_type == 'attention':
            attention_weights = self.attention_pool(x)
            attention_weights = torch.softmax(attention_weights, dim=0)
            x = global_add_pool(x * attention_weights, batch)
        
        # Combine global features
        if global_features is not None:
            global_feat = self.global_encoder(global_features)
            x = torch.cat([x, global_feat], dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, optimizer, scheduler, class_weights, epochs=100, patience=20):
    """Train the GNN model with early stopping"""
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_preds = []
        train_true = []
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            if hasattr(data, 'global_features'):
                global_features = data.global_features.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_features)
            else:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            loss = F.cross_entropy(out, data.y, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            train_preds.extend(pred.detach().cpu().numpy())
            train_true.extend(data.y.detach().cpu().numpy())
        
        # Validation phase
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                
                if hasattr(data, 'global_features'):
                    global_features = data.global_features.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch, global_features)
                else:
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                
                pred = out.argmax(dim=1)
                val_preds.extend(pred.detach().cpu().numpy())
                val_true.extend(data.y.detach().cpu().numpy())
        
        # Calculate metrics
        train_f1 = f1_score(train_true, train_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            avg_train_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_model_state, best_val_f1

# Process molecular graphs
print("Processing molecular graphs...")
data_list = []
valid_indices = []

for i, smiles in enumerate(smiles_list):
    graph = smiles_to_graph_mxene_enhanced(smiles)
    if graph is not None:
        graph.y = torch.tensor([labels[i]], dtype=torch.long)
        data_list.append(graph)
        valid_indices.append(i)

print(f"Successfully processed {len(data_list)}/{len(smiles_list)} molecules")

# Standardize global features
if len(data_list) > 0 and hasattr(data_list[0], 'global_features'):
    global_features = torch.cat([data.global_features for data in data_list], dim=0)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(global_features.numpy())
    
    for i, data in enumerate(data_list):
        data.global_features = torch.tensor(normalized_features[i:i+1], dtype=torch.float)

# Get valid labels
valid_labels = [labels[i] for i in valid_indices]

# Split dataset
train_data, test_data, train_labels, test_labels = train_test_split(
    data_list, valid_labels, test_size=0.2, random_state=seed, stratify=valid_labels)

train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.15, random_state=seed, stratify=train_labels)

print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# Calculate class weights
class_counts = Counter(train_labels)
num_classes = len(class_counts)
class_weights = torch.tensor(
    [len(train_labels) / (num_classes * class_counts[c]) for c in sorted(class_counts.keys())],
    dtype=torch.float, device=device
)

# Get model parameters
num_features = data_list[0].x.shape[1]
edge_dim = data_list[0].edge_attr.shape[1]
global_dim = data_list[0].global_features.shape[1] if hasattr(data_list[0], 'global_features') else 0

print(f"\nModel parameters:")
print(f"Node feature dimension: {num_features}")
print(f"Edge feature dimension: {edge_dim}")
print(f"Global feature dimension: {global_dim}")
print(f"Number of classes: {num_classes}")

# Hyperparameter optimization
def objective(trial):
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [128, 256, 512]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'gnn_type': trial.suggest_categorical('gnn_type', ['gcn', 'gat', 'gin', 'graphconv']),
        'pool_type': trial.suggest_categorical('pool_type', ['mean', 'max', 'combined', 'attention']),
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'heads': trial.suggest_categorical('heads', [2, 4, 8]) if trial.params.get('gnn_type') == 'gat' else 4,
        'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
        'use_residual': trial.suggest_categorical('use_residual', [True, False]),
        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False])
    }
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'])
    
    # Create model
    model = AdvancedMXeneBindingGNN(
        feature_size=num_features,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_channels=params['hidden_channels'],
        num_classes=num_classes,
        dropout=params['dropout'],
        gnn_type=params['gnn_type'],
        pool_type=params['pool_type'],
        num_layers=params['num_layers'],
        heads=params['heads'],
        use_residual=params['use_residual'],
        use_batch_norm=params['use_batch_norm'],
        activation=params['activation']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'], 
        weight_decay=params['weight_decay']
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    # Train model
    _, val_f1 = train_model(model, train_loader, val_loader, optimizer, scheduler, 
                          class_weights, epochs=50, patience=10)
    
    return val_f1

# Run hyperparameter optimization
print("\n=== Starting hyperparameter optimization ===")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=100, timeout=3600)

print("\n=== Optimization complete ===")
print("Best hyperparameters:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"Best validation F1 score: {study.best_value:.4f}")

# Train final model with best hyperparameters
final_train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)
final_val_loader = DataLoader(val_data, batch_size=best_params['batch_size'])
final_test_loader = DataLoader(test_data, batch_size=best_params['batch_size'])

final_model = AdvancedMXeneBindingGNN(
    feature_size=num_features,
    edge_dim=edge_dim,
    global_dim=global_dim,
    hidden_channels=best_params['hidden_channels'],
    num_classes=num_classes,
    dropout=best_params['dropout'],
    gnn_type=best_params['gnn_type'],
    pool_type=best_params['pool_type'],
    num_layers=best_params['num_layers'],
    heads=best_params.get('heads', 4),
    use_residual=best_params['use_residual'],
    use_batch_norm=best_params['use_batch_norm'],
    activation=best_params['activation']
).to(device)

final_optimizer = torch.optim.Adam(
    final_model.parameters(), 
    lr=best_params['learning_rate'], 
    weight_decay=best_params['weight_decay']
)

final_scheduler = ReduceLROnPlateau(final_optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

# Train final model
print("\nTraining final model...")
best_model_state, best_val_f1 = train_model(
    final_model, final_train_loader, final_val_loader, final_optimizer, final_scheduler,
    class_weights, epochs=200, patience=30
)

# Load best model
if best_model_state is not None:
    final_model.load_state_dict(best_model_state)

# Evaluate on test set
print("\n=== Model Evaluation ===")
final_model.eval()
test_preds = []
test_true = []
test_probs_list = []

with torch.no_grad():
    for data in final_test_loader:
        data = data.to(device)
        
        if hasattr(data, 'global_features'):
            global_features = data.global_features.to(device)
            out = final_model(data.x, data.edge_index, data.edge_attr, data.batch, global_features)
        else:
            out = final_model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        
        test_preds.extend(pred.detach().cpu().numpy())
        test_true.extend(data.y.detach().cpu().numpy())
        test_probs_list.extend(probs.detach().cpu().numpy())

# Calculate test metrics
test_acc = accuracy_score(test_true, test_preds)
test_precision = precision_score(test_true, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(test_true, test_preds, average='weighted', zero_division=0)
test_f1 = f1_score(test_true, test_preds, average='weighted', zero_division=0)
test_auc = roc_auc_score(test_true, [prob[1] for prob in test_probs_list])

print("\n=== Test Results ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"AUC-ROC: {test_auc:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(test_true, test_preds, target_names=['Non-binding', 'Binding']))

# Save model
torch.save(final_model.state_dict(), 'mxene_binding_model.pth')
print("\nModel saved as 'mxene_binding_model.pth'")