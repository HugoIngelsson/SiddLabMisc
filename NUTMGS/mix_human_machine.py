#!/usr/bin/env python3
"""
Binary (One-vs-Many) Training Script for Individual Species Classification
Trains a binary classifier to predict presence/absence of a specific species
"""

# +
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import seaborn as sns
import wandb
import argparse
import sys
import os
from eval.evalutation_suite_sentinel import evaluate_single_binary_model
import random
import time
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
# -

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentinelPixelDatasetFixedLength(Dataset):
    """Dataset with robust NaN/inf handling"""
    
    def __init__(self, dataset_dir, split='train', temporal_length=96, aggregation='interpolate'):
        self.dataset_dir = Path(dataset_dir)
        self.temporal_length = temporal_length
        self.aggregation = aggregation


        # Find the correct paths
        if (self.dataset_dir / "metadata" / "sentinel_with_vectors.csv").exists():
            self.base_dataset_dir = self.dataset_dir.parent
        else:
            if (self.dataset_dir / "species_classification_vectors").exists():
                self.dataset_dir = self.dataset_dir / "species_classification_vectors"
                self.base_dataset_dir = self.dataset_dir.parent
            else:
                raise ValueError(f"Cannot find species_classification_vectors directory")
        
        # Load metadata
        sentinel_path = self.dataset_dir / "data_splits" / f"{split}.csv"
        self.metadata = pd.read_csv(sentinel_path)
        
        # Fix any data issues in metadata
        numeric_cols = self.metadata.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace inf with NaN, then fill with column mean
            self.metadata[col] = self.metadata[col].replace([np.inf, -np.inf], np.nan)
            if self.metadata[col].isna().any():
                col_mean = self.metadata[col].dropna().mean()
                if np.isnan(col_mean):
                    col_mean = 0.0
                self.metadata[col] = self.metadata[col].fillna(col_mean)
        
        # Filter by split
        split_path = self.dataset_dir / "data_splits" / f"{split}.csv"
        if split_path.exists():
            split_df = pd.read_csv(split_path)
            split_pixels = set(split_df['pixel_id'].dropna().unique())
            self.metadata = self.metadata[self.metadata['pixel_id'].isin(split_pixels)]
        
        # Load species info
        vector_info_path = self.dataset_dir / "metadata" / "species_vector_mapping.json"
        if vector_info_path.exists():
            with open(vector_info_path, 'r') as f:
                self.vector_info = json.load(f)
            self.num_species = self.vector_info['num_species']
        else:
            # Infer from data
            sample_vector = self.metadata['species_vector_str'].dropna().iloc[0]
            self.num_species = len(sample_vector.split(','))
        
        # Features to extract
        self.spectral_features = ['ndvi', 'evi', 'reci', 'ndre', 'nbr']
        self.band_features = ['b2_refl', 'b3_refl', 'b4_refl', 'b5_refl', 
                             'b6_refl', 'b7_refl', 'b8_refl']
        
        # Find time series directory
        possible_dirs = [
            self.base_dataset_dir / "sentinel2_pixel_timeseries" / "pixel_data",
            self.dataset_dir / "sentinel2_pixel_timeseries" / "pixel_data",
        ]
        
        self.timeseries_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists():
                self.timeseries_dir = dir_path
                break
        
        self.pixel_timeseries = {}
        args = self.get_vital_info()
        with ProcessPoolExecutor(max_workers=8) as executor:
            result = list(tqdm(
                executor.map(SentinelPixelDatasetFixedLength.safe_process_pixel, repeat(args), list(self.metadata.to_dict('records')), chunksize=32),
                total=len(self.metadata),
                desc="Preprocessing data"
            ))

        
        self.pixel_timeseries = {
            pid: self.rehydrate(features) for pid, features in result if features is not None
        }

        print(f"Loaded {len(self.metadata)} pixels for {split} split")
    
    @staticmethod
    def rehydrate(features):
        return {
            'pixel_id': features['pixel_id'],
            'temporal_features': torch.tensor(features['temporal_features'], dtype=torch.float32).reshape(features['features_size']),
            'temporal_mask': torch.tensor(features['temporal_mask'], dtype=torch.float32),
            'species_multi_hot': torch.tensor(features['species_multi_hot'], dtype=torch.float32),
            'species_count': features['species_count'],
            'tree_count': features['tree_count']
        }
        return features

    def get_vital_info(self):
        return {
            'metadata': self.metadata,
            'timeseries_dir': self.timeseries_dir,
            'spectral_features': self.spectral_features,
            'band_features': self.band_features,
            'temporal_length': self.temporal_length,
            'num_species': self.num_species
        }

    @staticmethod
    def safe_process_pixel(args, row):
        try:
            return SentinelPixelDatasetFixedLength.process_pixel(args, row)
        except Exception as e:
            print(f"[ERROR] Failed processing pixel {row.get('pixel_id', '?')}: {e}", flush=True)
            traceback.print_exc()
            return None, None

    @staticmethod
    def process_pixel(args, row):
        pixel_id = row["pixel_id"]

        # Load time series
        features, temporal_mask = SentinelPixelDatasetFixedLength.load_and_process_timeseries(args, pixel_id)

        # Get species vector
        species_vector = SentinelPixelDatasetFixedLength.parse_vector_string(args, row.get('species_vector_str'))

        # Final safety checks
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        temporal_mask = torch.nan_to_num(temporal_mask, nan=0.0, posinf=1.0, neginf=0.0)
        species_vector = torch.tensor(species_vector, dtype=torch.float32)

        return pixel_id, {
            'pixel_id': pixel_id,
            'temporal_features': features.cpu().numpy().astype(np.float32).flatten().tolist(),
            'temporal_mask': temporal_mask.cpu().numpy().astype(np.float32).flatten().tolist(),
            'species_multi_hot': species_vector.cpu().numpy().astype(np.float32).flatten().tolist(),
            'species_count': int(row.get('species_count', 1)),
            'tree_count': int(row.get('tree_count', row.get('pixel_tree_count', 1))),
            'features_size': features.shape
        }

    def __len__(self):
        return len(self.metadata)
    
    def get_pos_neg_indices(self, target_species_index):
        """Get indices of positive and negative samples for a specific species"""
        positive_indices = []
        negative_indices = []
        
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            species_vector = SentinelPixelDatasetFixedLength.parse_vector_string(self.get_vital_info(), row.get('species_vector_str'))
            
            if species_vector[target_species_index] > 0:
                positive_indices.append(idx)
            else:
                negative_indices.append(idx)
        
        return np.array(positive_indices), np.array(negative_indices)
    
    @staticmethod
    def load_and_process_timeseries(self, pixel_id):
        """Load time series with robust error handling"""
        if self['timeseries_dir'] is None:
            return SentinelPixelDatasetFixedLength.create_empty_timeseries(self)
        
        csv_path = self['timeseries_dir'] / f"{pixel_id}_sentinel2_timeseries.csv"
        
        if not csv_path.exists():
            return SentinelPixelDatasetFixedLength.create_empty_timeseries(self)
        
        try:
            # Load time series
            df = pd.read_csv(csv_path)
            
            # Fix infinity and NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                if df[col].isna().all():
                    df[col] = 0.0
                else:
                    col_mean = df[col].dropna().mean()
                    if np.isnan(col_mean):
                        col_mean = 0.0
                    df[col] = df[col].fillna(col_mean)
            
            # Convert band values to reflectance if needed
            for band in ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']:
                if band in df.columns and f'{band}_refl' not in df.columns:
                    df[f'{band}_refl'] = df[band] / 10000.0
                    # Clip to valid reflectance range
                    df[f'{band}_refl'] = df[f'{band}_refl'].clip(0, 1)
            
            # Calculate indices if missing
            if 'ndvi' not in df.columns and all(col in df.columns for col in ['b4_refl', 'b8_refl']):
                denominator = df['b8_refl'] + df['b4_refl']
                denominator = denominator.replace(0, 1e-6)  # Avoid division by zero
                df['ndvi'] = (df['b8_refl'] - df['b4_refl']) / denominator
                df['ndvi'] = df['ndvi'].clip(-1, 1)
            
            if 'evi' not in df.columns and all(col in df.columns for col in ['b2_refl', 'b4_refl', 'b8_refl']):
                denominator = df['b8_refl'] + 6 * df['b4_refl'] - 7.5 * df['b2_refl'] + 1
                denominator = denominator.replace(0, 1)
                df['evi'] = 2.5 * (df['b8_refl'] - df['b4_refl']) / denominator
                df['evi'] = df['evi'].clip(-1, 1)
            
            # Ensure all required features exist
            for feat in self['spectral_features'] + self['band_features']:
                if feat not in df.columns:
                    df[feat] = 0.0
                    
            return SentinelPixelDatasetFixedLength.interpolate_to_fixed_length(self, df)
            
        except Exception as e:
            print(f"Error loading time series for {pixel_id}: {e}")
            return SentinelPixelDatasetFixedLength.create_empty_timeseries(self)
    
    @staticmethod
    def interpolate_to_fixed_length(self, df):
        """Interpolate with NaN safety"""
        df = df.sort_values('date').reset_index(drop=True)
        n_obs = len(df)
        
        n_features = len(self['spectral_features']) + len(self['band_features'])
        features = torch.zeros(self['temporal_length'], n_features)
        
        if n_obs == 0:
            return features, torch.zeros(self['temporal_length'])
        
        # Get indices for interpolation
        if n_obs >= self['temporal_length']:
            indices = np.linspace(0, n_obs-1, self['temporal_length'], dtype=int)
            
            for i, feat in enumerate(self['spectral_features'] + self['band_features']):
                if feat in df.columns:
                    values = df[feat].iloc[indices].values
                    # Replace any remaining NaN/inf
                    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                    features[:, i] = torch.tensor(values, dtype=torch.float32)
        else:
            # Interpolation for sparse data
            old_indices = np.arange(n_obs)
            new_indices = np.linspace(0, n_obs-1, self['temporal_length'])
            
            for i, feat in enumerate(self['spectral_features'] + self['band_features']):
                if feat in df.columns:
                    values = df[feat].values
                    # Replace NaN before interpolation
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() > 1:
                        # Interpolate only valid values
                        interpolated = np.interp(new_indices, old_indices[valid_mask], values[valid_mask])
                        interpolated = np.nan_to_num(interpolated, nan=0.0, posinf=1.0, neginf=-1.0)
                        features[:, i] = torch.tensor(interpolated, dtype=torch.float32)
                    elif valid_mask.sum() == 1:
                        # Only one valid value, use it for all
                        features[:, i] = values[valid_mask][0]
                    else:
                        # No valid values
                        features[:, i] = 0.0
        
        # Create mask
        mask = torch.ones(self['temporal_length'])
        if n_obs < self['temporal_length']:
            mask[n_obs:] = 0.5
        
        return features, mask
    
    @staticmethod
    def create_empty_timeseries(self):
        """Create empty time series"""
        n_features = len(self['spectral_features']) + len(self['band_features'])
        features = torch.zeros(self['temporal_length'], n_features)
        mask = torch.zeros(self['temporal_length'])
        return features, mask
    
    @staticmethod
    def parse_vector_string(self, vector_str):
        """Parse vector with error handling"""
        if pd.isna(vector_str):
            return np.zeros(self['num_species'], dtype=np.float32)
        
        try:
            vector = np.array([float(x) for x in str(vector_str).split(',')])
            # Ensure no NaN or inf
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)
            return vector.astype(np.float32)
        except:
            return np.zeros(self['num_species'], dtype=np.float32)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        pixel_id = row['pixel_id']
        
        return self.pixel_timeseries[pixel_id]
    

class BinarySpeciesDataset(Dataset):
    """Wrapper around SentinelPixelDatasetFixedLength for binary classification with multi-shot support"""

    def __init__(self, base_dataset, target_species_id, species_mapping, shots=None, seed=42):
        """
        Args:
            base_dataset: SentinelPixelDatasetFixedLength instance
            target_species_id: Species ID to classify (1-indexed from COCO)
            species_mapping: Dict with species_id_to_index mapping
            shots: Tuple (n_pos, n_neg) or int for both (e.g., (100, 100)), or None for full
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.target_species_id = target_species_id
        self.species_id_to_index = species_mapping['species_id_to_index']
        self.target_index = self.species_id_to_index[str(target_species_id)]

        # Multi-shot sampling
        self.indices = list(range(len(self.base_dataset)))
        if shots is not None:
            np.random.seed(seed)
            # Get positive and negative indices for the target species
            pos_idx, neg_idx = self.base_dataset.get_pos_neg_indices(self.target_index)
            n_pos, n_neg = shots if isinstance(shots, (tuple, list)) else (shots, shots)
            
            # Sample from available indices
            if len(pos_idx) > 0:
                selected_pos = np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
            else:
                selected_pos = np.array([])
                
            if len(neg_idx) > 0:
                selected_neg = np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
            else:
                selected_neg = np.array([])
            
            # Combine and shuffle
            self.indices = np.concatenate([selected_pos, selected_neg]).astype(int)
            np.random.shuffle(self.indices)
            
            print(f"Multi-shot sampling: {len(selected_pos)} positive, {len(selected_neg)} negative samples for a total of {len(self.indices)} samples")

        self.calculate_class_distribution()

    def calculate_class_distribution(self):
        """Calculate distribution of positive and negative samples"""
        positive_count = 0
        negative_count = 0
        for idx in self.indices:
            if random.random() < 0.05 or positive_count + negative_count < 100 or positive_count < 1:
                sample = self.base_dataset[idx]
                multi_hot = sample['species_multi_hot']
                if multi_hot[self.target_index] > 0:
                    positive_count += 1
                else:
                    negative_count += 1
        self.positive_count = positive_count
        self.negative_count = negative_count
        self.positive_ratio = positive_count / (positive_count + negative_count) if len(self.indices) > 0 else 0
        logger.info(f"Class distribution - Positive: {positive_count} ({self.positive_ratio:.2%}), "
                    f"Negative: {negative_count} ({1-self.positive_ratio:.2%})")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.base_dataset[self.indices[idx]]
        multi_hot = sample['species_multi_hot']
        binary_label = multi_hot[self.target_index].unsqueeze(0)
        binary_sample = {
            'pixel_id': sample['pixel_id'],
            'temporal_features': sample['temporal_features'],
            'temporal_mask': sample['temporal_mask'],
            'binary_label': binary_label,
            'species_count': sample['species_count'],
            'tree_count': sample['tree_count'],
            'species_multi_hot': sample['species_multi_hot']
        }
        return binary_sample


class BinarySpeciesMLP(nn.Module):
    """Binary classifier MLP for single species prediction"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 dropout_rate=0.3, use_batch_norm=True):
        """
        Args:
            input_dim: Flattened input dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(BinarySpeciesMLP, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer - single output for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, temporal_length, n_features)
            mask: Temporal mask of shape (batch_size, temporal_length)
        
        Returns:
            Logits of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        # Flatten temporal features
        x_flat = x.reshape(batch_size, -1)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            mask_flat = mask_expanded.reshape(batch_size, -1)
            x_flat = x_flat * mask_flat
        
        # Forward through MLP
        logits = self.model(x_flat)
        
        return logits


class WeightedFocalLoss(nn.Module):
    """Focal loss for handling class imbalance in binary classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits of shape (batch_size, 1)
            targets: Binary targets of shape (batch_size, 1)
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        probas = torch.sigmoid(inputs)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        
        # Apply focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_term * bce_loss
        else:
            focal_loss = focal_term * bce_loss
        
        return focal_loss.mean()


class BinaryTrainer:
    """Trainer for binary species classification"""
    
    def __init__(self, model, train_loader, val_loader, species_name, device='cuda',
                 learning_rate=1e-3, weight_decay=1e-4, use_wandb=False,
                 pos_weight=None, use_focal_loss=True, num_proc=1):
        local_rank = int(os.environ["LOCAL_RANK"])
        self.model = model.to(device)
        self.model = DDP(model, device_ids=[local_rank])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.species_name = species_name
        self.device = device
        self.use_wandb = use_wandb
        
        # Loss function
        if use_focal_loss:
            self.criterion = WeightedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5
        )
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_aucs = []
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        pbar = tqdm(total=len(self.train_loader), desc='Training')
        for batch in self.train_loader:
            # Move to device
            features = batch['temporal_features'].to(self.device)
            mask = batch['temporal_mask'].to(self.device)
            targets = batch['binary_label'].to(self.device)
            
            # Skip batch if all negative (for stability)
            if targets.sum() == 0 and np.random.random() > 0.5:
                continue
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(features, mask)
            
            # Compute loss
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Store predictions
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        all_probs = np.array(all_probs).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        # Only calculate AUC if we have both classes
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
        else:
            auc = 0.5
        
        return avg_loss, accuracy, f1, auc

    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                features = batch['temporal_features'].to(self.device)
                mask = batch['temporal_mask'].to(self.device)
                targets = batch['binary_label'].to(self.device)
                
                # Forward pass
                logits = self.model(features, mask)
                
                # Compute loss
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                
                # Store predictions
                probs = torch.sigmoid(logits)
                predictions = (probs > 0.5).float()
                
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        all_probs = np.array(all_probs).flatten()
        
        # Binary classification metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        # Calculate AUC and AP only if we have both classes
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_probs)
            ap = average_precision_score(all_targets, all_probs)
        else:
            auc = 0.5
            ap = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'average_precision': ap,
            'confusion_matrix': cm,
            'all_probs': all_probs,
            'all_targets': all_targets
        }
        
        return metrics

    def train(self, num_epochs, save_dir='checkpoints'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create species-specific subdirectory
        species_dir = save_dir / f"species_{self.species_name.replace(' ', '_')}"
        species_dir.mkdir(exist_ok=True)
        
        best_f1 = 0
        patience_counter = 0
        early_stop_patience = 15
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training
            train_loss, train_acc, train_f1, train_auc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            local_rank = int(os.environ["LOCAL_RANK"])
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.val_f1_scores.append(val_metrics['f1'])
            self.val_aucs.append(val_metrics['auc'])
            
            # Update learning rate based on F1 score
            self.scheduler.step(val_metrics['f1'])
            
            # Log metrics
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                       f"F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                 f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, "
                 f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Log confusion matrix
            cm = val_metrics['confusion_matrix']
            if cm.shape == (2, 2):
                logger.info(f"Confusion Matrix - TN: {cm[0,0]}, FP: {cm[0,1]}, "
                       f"FN: {cm[1,0]}, TP: {cm[1,1]}")
            
            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'train_f1': train_f1,
                    'train_auc': train_auc,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics['auc'],
                    'val_ap': val_metrics['average_precision'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                # Log confusion matrix
                if cm.shape == (2, 2):
                    log_dict.update({
                        'val_true_negatives': cm[0, 0],
                        'val_false_positives': cm[0, 1],
                        'val_false_negatives': cm[1, 0],
                        'val_true_positives': cm[1, 1]
                    })
                
                wandb.log(log_dict)
            
            # Save best model based on F1 score
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
            
                if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': best_f1,
                    'val_metrics': val_metrics,
                    'species_name': self.species_name
                    }
                local_rank = int(os.environ["LOCAL_RANK"])
                if local_rank == 0:
                    torch.save(checkpoint, species_dir / 'best_model.pth')
                    logger.info(f"Saved best model with F1: {best_f1:.4f}")
                
                # Save probability distributions for best model
                self.save_probability_analysis(val_metrics, species_dir)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                logger.info("Disabled because multi-core is funky")
        

        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            # Save final model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            
            final_checkpoint = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'val_f1_scores': self.val_f1_scores,
                'val_aucs': self.val_aucs,
                'species_name': self.species_name
            }
            
            torch.save(final_checkpoint, species_dir / 'final_model.pth')
        
            # Plot training curves
            self.plot_training_curves(species_dir)
        
            # Plot ROC and PR curves for final model
            self.plot_roc_pr_curves(val_metrics, species_dir)
        
        return best_f1
    
    def save_probability_analysis(self, val_metrics, save_dir):
        """Save probability distribution analysis"""
        probs = val_metrics['all_probs']
        targets = val_metrics['all_targets']
        
        # Save probability distributions
        prob_df = pd.DataFrame({
            'probability': probs,
            'true_label': targets,
            'species': self.species_name
        })
        prob_df.to_csv(save_dir / 'probability_distributions.csv', index=False)
        
        # Plot probability distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate positive and negative samples
        pos_probs = probs[targets == 1]
        neg_probs = probs[targets == 0]
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        ax.hist(neg_probs, bins=bins, alpha=0.5, label='Negative', color='red', density=True)
        ax.hist(pos_probs, bins=bins, alpha=0.5, label='Positive', color='green', density=True)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Probability Distributions for {self.species_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'probability_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.val_accuracies)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.val_f1_scores)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].grid(True)
        
        # AUC
        axes[1, 1].plot(self.val_aucs)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].grid(True)
        
        plt.suptitle(f'Training Curves for {self.species_name}')
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_pr_curves(self, val_metrics, save_dir):
        """Plot ROC and Precision-Recall curves"""
        probs = val_metrics['all_probs']
        targets = val_metrics['all_targets']
        
        # Skip if only one class present
        if len(np.unique(targets)) < 2:
            logger.warning("Cannot plot ROC/PR curves with only one class present")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
        
        ax1.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {self.species_name}')
        ax1.legend()
        ax1.grid(True)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)
        
        ax2.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {self.species_name}')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

class DotDict(dict):
    def __getattr__(self, __name: str):
        try:
            return self[__name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{__name}'")
        
class BalancedDataLoader:
    def __init__(self, dataset1, dataset2, batch_size, batches_per_epoch=None, **dataloader_kwargs):
        assert batch_size % 2 == 0, "Batch size must be even to balance across datasets"
        self.half_batch = batch_size // 2

        self.loader1 = DataLoader(dataset1, batch_size=self.half_batch, shuffle=True, **dataloader_kwargs)
        self.loader2 = DataLoader(dataset2, batch_size=self.half_batch, shuffle=True, **dataloader_kwargs)
        self.dataloader_kwargs = dataloader_kwargs

        self.batches_per_epoch = batches_per_epoch if batches_per_epoch is not None else min(len(self.loader1), len(self.loader2))

    def __iter__(self):
        # Create fresh iterators every epoch
        iter1 = iter(self.loader1)
        iter2 = iter(self.loader2)

        count = 0
        while True:
            try:
                batch1 = next(iter1)
            except StopIteration:
                iter1 = iter(DataLoader(self.loader1.dataset, batch_size=self.half_batch, shuffle=True, **self.dataloader_kwargs))
                batch1 = next(iter1)

            try:
                batch2 = next(iter2)
            except StopIteration:
                iter2 = iter(DataLoader(self.loader2.dataset, batch_size=self.half_batch, shuffle=True, **self.dataloader_kwargs))
                batch2 = next(iter2)

            merged = self.merge_batches(batch1, batch2)
            yield merged

            count += 1
            if self.batches_per_epoch is not None and count >= self.batches_per_epoch:
                break

    def pad_to_match(self, t1, t2):
        if t1.shape == t2.shape:
            return t1, t2
        max_dim = max(t1.shape[1], t2.shape[1])
        def pad(t, target_dim):
            pad_width = target_dim - t.shape[1]
            return F.pad(t, (0, pad_width)) if pad_width > 0 else t
        return pad(t1, max_dim), pad(t2, max_dim)

    def merge_batches(self, batch1, batch2):
        if isinstance(batch1, dict):
            merged = {}
            for k in batch1:
                v1 = batch1[k]
                v2 = batch2[k]
                if isinstance(v1, torch.Tensor):
                    v1, v2 = self.pad_to_match(v1, v2)
                    merged[k] = torch.cat([v1, v2], dim=0)
                elif isinstance(v1, list):
                    merged[k] = v1 + v2
                else:
                    raise TypeError(f"Unsupported type for key '{k}': {type(v1)}")
            return merged
        elif isinstance(batch1, (list, tuple)):
            return [torch.cat([b1, b2], dim=0) for b1, b2 in zip(batch1, batch2)]
        else:
            return torch.cat([batch1, batch2], dim=0)

    def __len__(self):
        return self.batches_per_epoch
    
def load_suite(suite, args):
    loaded = {}
    loaded['dataset_dir'] = suite['dataset_dir'] if 'dataset_dir' in suite else args.dataset_dir
    loaded['species_id'] = suite['species_id'] if 'species_id' in suite else args.species_id
    loaded['shots'] = (suite['shots'],) if 'shots' in suite else args.shots
    loaded['shot_seed'] = suite['shot_seed'] if 'shot_seed' in suite else args.shot_seed
    loaded['batch_size'] = suite['batch_size'] if 'batch_size' in suite else args.batch_size
    loaded['temporal_length'] = suite['temporal_length'] if 'temporal_length' in suite else args.temporal_length
    loaded['hidden_dims'] = suite['hidden_dims'] if 'hidden_dims' in suite else args.hidden_dims
    loaded['dropout_rate'] = suite['dropout_rate'] if 'dropout_rate' in suite else args.dropout_rate
    loaded['learning_rate'] = suite['learning_rate'] if 'learning_rate' in suite else args.learning_rate
    loaded['weight_decay'] = suite['weight_decay'] if 'weight_decay' in suite else args.weight_decay
    loaded['use_wandb'] = suite['use_wandb'] if 'use_wandb' in suite else args.use_wandb
    loaded['use_focal_loss'] = suite['use_focal_loss'] if 'use_focal_loss' in suite else args.use_focal_loss
    loaded['num_epochs'] = suite['num_epochs'] if 'num_epochs' in suite else args.num_epochs
    loaded['save_dir'] = suite['save_dir'] if 'save_dir' in suite else args.save_dir
    loaded['load_weights'] = suite['load_weights'] if 'load_weights' in suite else args.load_weights

    print(f'Running a test with these settings: {loaded}')
    return DotDict(loaded)

def train_model(args, species_mapping, device, train_base, val_base, test_base, train2_base, val2_base):
    # Find species_classification_vectors directory
    dataset_dir = Path(args.dataset_dir)
    if (dataset_dir / "species_classification_vectors").exists():
        species_dir = dataset_dir / "species_classification_vectors"
    else:
        species_dir = dataset_dir

    species_name = species_mapping['species_id_to_name'].get(str(args.species_id), 'Unknown')
    mapping_path = species_dir / "metadata" / "species_vector_mapping.json"
    
    # Wrap in binary datasets
    shots = None
    if args.shots is not None and len(args.shots) > 0:
        if len(args.shots) == 1:
            shots = (args.shots[0], args.shots[0])
        else:
            shots = tuple(args.shots)

    train_dataset = BinarySpeciesDataset(train_base, args.species_id, species_mapping, shots=shots, seed=args.shot_seed)
    val_dataset = BinarySpeciesDataset(val_base, args.species_id, species_mapping)
    train2_dataset = BinarySpeciesDataset(train2_base, args.species_id, species_mapping, shots=shots, seed=args.shot_seed)
    val2_dataset = BinarySpeciesDataset(val2_base, args.species_id, species_mapping)
    test_dataset = BinarySpeciesDataset(test_base, args.species_id, species_mapping)

    # Calculate class weights for imbalanced data
    pos_weight = None
    if train_dataset.positive_ratio < 0.4:  # If positive class is less than 40%
        pos_weight = torch.tensor([(1 - train_dataset.positive_ratio) / train_dataset.positive_ratio]).to(device)
        logger.info(f"Using positive class weight: {pos_weight.item():.3f}")

    # Create dataloaders
    train_loader = BalancedDataLoader(
        train_dataset,
        train2_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = BalancedDataLoader(
        val_dataset,
        val2_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Calculate input dimension
    n_spectral_features = 5  # ndvi, evi, reci, ndre, nbr
    n_band_features = 7      # b2-b8 reflectance
    n_features = n_spectral_features + n_band_features
    input_dim = args.temporal_length * n_features

    logger.info(f"Input dimension: {input_dim}")

    # Create model
    model = BinarySpeciesMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        use_batch_norm=True
    )

    if args.load_weights is not None and len(args.load_weights) > 0:
        checkpoint = torch.load(args.load_weights, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Model architecture:\n{model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = BinaryTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        species_name=species_name,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb,
        pos_weight=pos_weight,
        use_focal_loss=args.use_focal_loss
    )

    logger.info(f"Starting to train model on device: {device}")
    # Train model
    best_f1 = trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )

    logger.info(f"\nTraining completed for {species_name}! Best F1 score: {best_f1:.4f}")
    
    # === EVALUATION SUITE INTEGRATION ===
    # Load best model checkpoint
    species_save_dir = Path(args.save_dir) / f"species_{species_name.replace(' ', '_')}"
    best_model_path = species_save_dir / "best_model.pth"
    local_rank = int(os.environ["LOCAL_RANK"])
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Loaded best model from {best_model_path} for evaluation.")

        # Prepare val_loader for evaluation (already created above)
        # Run evaluation suite and save report in the same directory as model weights
        if local_rank == 0:
            evaluate_single_binary_model(
                model=model,
                species_id=args.species_id,
                test_loader=test_loader,
                species_mapping_path=str(mapping_path),
                output_dir=str(species_save_dir),
                device=device,
                threshold=0.5,
                plot_curves=True
            )
        logger.info(f"Evaluation report saved in {species_save_dir}")
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}, skipping evaluation.")

    if args.use_wandb:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Train binary classifier for individual species')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to dataset directory (containing species_classification_vectors)')
    parser.add_argument('--dataset2_dir', type=str, required=True,
                       help='The second dataset to mix in')
    parser.add_argument('--species_id', type=int, required=True,
                       help='Species ID to classify (from COCO categories)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--temporal_length', type=int, default=96,
                       help='Fixed temporal dimension')
    parser.add_argument('--aggregation', type=str, default='interpolate',
                       choices=['interpolate', 'pad', 'sample'],
                       help='Method to handle variable length sequences')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss instead of BCE')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='sentinel2-binary-species',
                       help='W&B project name')
    parser.add_argument('--save_dir', type=str, default='checkpoints_binary',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--shots', type=int, nargs='*', default=None,
                       help='Number of positive and negative samples, e.g., --shots 100 100')
    parser.add_argument('--shot_seed', type=int, default=42,
                       help='Random seed for shot sampling')
    parser.add_argument('--test_dataset_dir', type=str, default=None,
                       help='Optional: Path to a different dataset directory for testing')
    parser.add_argument('--run_suite', type=bool, default=False,
                       help='Whether to train a suite of models rather than a single one')
    parser.add_argument('--suite_file', type=str, default=None,
                       help='The suite describing what tests to run')
    parser.add_argument('--load_weights', type=str, default=None,
                       help='Path to weights to load (if any)')

    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Find species_classification_vectors directory
    dataset_dir = Path(args.dataset_dir)
    if (dataset_dir / "species_classification_vectors").exists():
        species_dir = dataset_dir / "species_classification_vectors"
    else:
        species_dir = dataset_dir

    # Find species_classification_vectors directory
    dataset2_dir = Path(args.dataset2_dir)
    if (dataset_dir / "species_classification_vectors").exists():
        species2_dir = dataset2_dir / "species_classification_vectors"
    else:
        species2_dir = dataset2_dir
    
    # Load species mapping
    mapping_path = species_dir / "metadata" / "species_vector_mapping.json"
    if not mapping_path.exists():
        logger.error(f"species_vector_mapping.json not found at {mapping_path}")
        logger.error("Please run the species integration script first to create this file")
        return
    
    with open(mapping_path, 'r') as f:
        species_mapping = json.load(f)
    
    # Get species name
    species_name = species_mapping['species_id_to_name'].get(str(args.species_id), 'Unknown')
    logger.info(f"Training binary classifier for species: {species_name} (ID: {args.species_id})")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{species_name}_binary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Create base datasets
    train_base = SentinelPixelDatasetFixedLength(
        dataset_dir=species_dir,
        split='train',
        temporal_length=args.temporal_length,
        aggregation=args.aggregation
    )

    val_base = SentinelPixelDatasetFixedLength(
        dataset_dir=species_dir,
        split='val',
        temporal_length=args.temporal_length,
        aggregation=args.aggregation
    )

    train2_base = SentinelPixelDatasetFixedLength(
        dataset_dir=species2_dir,
        split='train',
        temporal_length=args.temporal_length,
        aggregation=args.aggregation
    )

    val2_base = SentinelPixelDatasetFixedLength(
        dataset_dir=species2_dir,
        split='val',
        temporal_length=args.temporal_length,
        aggregation=args.aggregation
    )

    # Use alternate test dataset if specified
    if args.test_dataset_dir is not None:
        test_species_dir = Path(args.test_dataset_dir)
        if (test_species_dir / "species_classification_vectors").exists():
            test_species_dir = test_species_dir / "species_classification_vectors"
        test_base = SentinelPixelDatasetFixedLength(
            dataset_dir=test_species_dir,
            split='test',
            temporal_length=args.temporal_length,
            aggregation=args.aggregation
        )
    else:
        test_base = SentinelPixelDatasetFixedLength(
            dataset_dir=species_dir,
            split='test',
            temporal_length=args.temporal_length,
            aggregation=args.aggregation
        )

    # For GPUs to synchronize
    init_process_group(backend="nccl")

    # Try running 3 consecutive models to see if this works :)
    if args.run_suite:
        suites = json.load(open(args.suite_file))
        for exp in suites['experiments']:
            settings = load_suite(exp, args)

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            train_model(settings, species_mapping, device, train_base, val_base, test_base, train2_base, val2_base)
    else:
        train_model(args, species_mapping, device, train_base, val_base, test_base, train2_base, val2_base)

if __name__ == "__main__":
    main()
