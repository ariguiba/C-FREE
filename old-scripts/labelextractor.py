"""
Automatic extraction of molecular properties for embedding analysis.
Extracts chirality, functional groups, scaffolds, and other chemical features from SMILES.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Scaffolds, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Dict, Tuple, Optional
from collections import Counter

class MolecularLabelExtractor:
    """
    Extract chemical labels from molecular structures for embedding analysis.
    Works with SMILES strings or RDKit molecule objects.
    """
    
    def __init__(self):
        # Define common functional groups as SMARTS patterns
        self.functional_groups = {
            'Alcohol': '[OH1;X2]',
            'Aldehyde': '[CH1;X3](=O)',
            'Ketone': '[C;X3](=O)[C,c]',
            'Carboxylic_Acid': 'C(=O)[OH1]',
            'Ester': 'C(=O)O[C,c]',
            'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
            'Amide': 'C(=O)[N;X3]',
            'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
            'Nitrile': 'C#N',
            'Halide': '[F,Cl,Br,I]',
            'Aromatic': 'a',
            'Ether': '[OD2]([#6])[#6]',
            'Thiol': '[SH1]',
            'Sulfone': 'S(=O)(=O)',
            'Phosphate': 'P(=O)([OH])',
        }
        
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit molecule object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Could not parse SMILES: {smiles}")
            return mol
        except Exception as e:
            print(f"Error parsing SMILES {smiles}: {e}")
            return None
    
    def has_chirality(self, mol: Chem.Mol) -> int:
        """
        Check if molecule has chiral centers.
        Returns: 1 if chiral, 0 if achiral
        """
        if mol is None:
            return 0
        
        # Find chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return 1 if len(chiral_centers) > 0 else 0
    
    def count_chiral_centers(self, mol: Chem.Mol) -> int:
        """Count number of chiral centers in molecule."""
        if mol is None:
            return 0
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return len(chiral_centers)
    
    def get_chiral_types(self, mol: Chem.Mol) -> List[str]:
        """
        Get list of chirality types (R/S) for each chiral center.
        Returns: List of 'R', 'S', or '?' for unassigned
        """
        if mol is None:
            return []
        
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        return [chirality for _, chirality in chiral_centers]
    
    def detect_functional_groups(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Detect presence of functional groups using SMARTS patterns.
        Returns: Dictionary with functional group names and counts
        """
        if mol is None:
            return {fg: 0 for fg in self.functional_groups.keys()}
        
        fg_counts = {}
        for fg_name, smarts_pattern in self.functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts_pattern)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                fg_counts[fg_name] = len(matches)
            else:
                fg_counts[fg_name] = 0
        
        return fg_counts
    
    def get_primary_functional_group(self, mol: Chem.Mol) -> str:
        """
        Get the most prominent functional group in the molecule.
        Priority: Carboxylic_Acid > Amide > Ester > Aldehyde > Ketone > others
        """
        if mol is None:
            return 'None'
        
        fg_counts = self.detect_functional_groups(mol)
        
        # Priority order for classification
        priority = [
            'Carboxylic_Acid', 'Amide', 'Ester', 'Aldehyde', 'Ketone',
            'Nitrile', 'Nitro', 'Amine', 'Alcohol', 'Thiol', 'Halide',
            'Ether', 'Aromatic', 'Sulfone', 'Phosphate'
        ]
        
        for fg in priority:
            if fg_counts.get(fg, 0) > 0:
                return fg
        
        return 'Aliphatic'
    
    def get_scaffold(self, mol: Chem.Mol) -> str:
        """
        Extract Murcko scaffold (core structure) from molecule.
        Returns: SMILES string of scaffold
        """
        if mol is None:
            return 'None'
        
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return 'None'
    
    def compute_solubility_bin(self, mol: Chem.Mol, n_bins: int = 4) -> int:
        """
        Estimate solubility bin using LogP.
        Lower LogP = more soluble (hydrophilic)
        Higher LogP = less soluble (hydrophobic)
        
        Returns: Bin number (0 = most soluble, n_bins-1 = least soluble)
        """
        if mol is None:
            return 0
        
        try:
            logp = Descriptors.MolLogP(mol)
            
            # Define bins based on typical LogP ranges
            # LogP < 0: Very soluble
            # LogP 0-2: Soluble
            # LogP 2-4: Moderate
            # LogP > 4: Poor solubility
            if logp < 0:
                return 0
            elif logp < 2:
                return 1
            elif logp < 4:
                return 2
            else:
                return min(3, n_bins - 1)
        except:
            return 0
    
    def compute_lipinski_violations(self, mol: Chem.Mol) -> int:
        """
        Count Lipinski's Rule of Five violations (drug-likeness).
        Returns: Number of violations (0-4)
        """
        if mol is None:
            return 0
        
        violations = 0
        
        try:
            # MW <= 500
            if Descriptors.MolWt(mol) > 500:
                violations += 1
            
            # LogP <= 5
            if Descriptors.MolLogP(mol) > 5:
                violations += 1
            
            # HBD <= 5
            if Lipinski.NumHDonors(mol) > 5:
                violations += 1
            
            # HBA <= 10
            if Lipinski.NumHAcceptors(mol) > 10:
                violations += 1
        except:
            pass
        
        return violations
    
    def extract_all_labels(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Extract all labels for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            DataFrame with all extracted labels
        """
        results = []
        
        for idx, smiles in enumerate(smiles_list):
            if idx % 100 == 0:
                print(f"Processing molecule {idx}/{len(smiles_list)}")
            
            mol = self.smiles_to_mol(smiles)
            
            if mol is None:
                # Add default values for failed molecules
                results.append({
                    'smiles': smiles,
                    'valid': False,
                    'has_chirality': 0,
                    'n_chiral_centers': 0,
                    'primary_functional_group': 'None',
                    'scaffold': 'None',
                    'solubility_bin': 0,
                    'lipinski_violations': 0
                })
                continue
            
            # Extract all properties
            fg_counts = self.detect_functional_groups(mol)
            
            result = {
                'smiles': smiles,
                'valid': True,
                'has_chirality': self.has_chirality(mol),
                'n_chiral_centers': self.count_chiral_centers(mol),
                'primary_functional_group': self.get_primary_functional_group(mol),
                'scaffold': self.get_scaffold(mol),
                'solubility_bin': self.compute_solubility_bin(mol),
                'lipinski_violations': self.compute_lipinski_violations(mol),
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'n_aromatic_rings': Lipinski.NumAromaticRings(mol),
                'n_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            }
            
            # Add functional group counts
            for fg_name, count in fg_counts.items():
                result[f'fg_{fg_name}'] = count
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_label_summary(self, df: pd.DataFrame):
        """Print summary statistics of extracted labels."""
        print("=" * 60)
        print("LABEL EXTRACTION SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal molecules: {len(df)}")
        print(f"Valid molecules: {df['valid'].sum()}")
        print(f"Invalid molecules: {(~df['valid']).sum()}")
        
        print(f"\n--- Chirality ---")
        print(f"Chiral molecules: {df['has_chirality'].sum()} ({df['has_chirality'].mean()*100:.1f}%)")
        print(f"Average chiral centers: {df['n_chiral_centers'].mean():.2f}")
        
        print(f"\n--- Functional Groups ---")
        fg_counts = df['primary_functional_group'].value_counts()
        print("Top functional groups:")
        for fg, count in fg_counts.head(10).items():
            print(f"  {fg}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n--- Solubility ---")
        sol_counts = df['solubility_bin'].value_counts().sort_index()
        labels = ['Very Soluble', 'Soluble', 'Moderate', 'Poor']
        for bin_num, count in sol_counts.items():
            print(f"  {labels[bin_num]}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\n--- Scaffolds ---")
        print(f"Unique scaffolds: {df['scaffold'].nunique()}")
        
        print(f"\n--- Drug-likeness ---")
        print(f"Average Lipinski violations: {df['lipinski_violations'].mean():.2f}")
        print("=" * 60)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
# import umap
import umap.umap_ as umap
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class EmbeddingAnalyzer:
    """
    Analyze and visualize molecular embeddings using UMAP and t-SNE.
    Evaluate how well embeddings capture chemical properties.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.reducers = {}
        
    def reduce_embeddings(self, 
                         embeddings: np.ndarray,
                         method: str = 'umap',
                         n_components: int = 2,
                         y_labels: Optional[np.ndarray] = None,
                         **kwargs) -> np.ndarray:
        """
        Reduce embedding dimensionality using UMAP or t-SNE.
        
        Args:
            embeddings: Input embeddings (n_samples, n_features)
            method: 'umap' or 'tsne'
            n_components: Number of dimensions (default: 2)
            **kwargs: Additional parameters for the reducer
        
        Returns:
            Reduced embeddings (n_samples, n_components)
        """
        if method.lower() == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=kwargs.get('n_neighbors', 5),
                min_dist=kwargs.get('min_dist', 0.1),
                metric=kwargs.get('metric', 'cosine')
            )
            # reducer = umap.UMAP(
            #     n_components=n_components,
            #     random_state=self.random_state,
            #     n_neighbors=kwargs.get('n_neighbors', 15),
            #     min_dist=kwargs.get('min_dist', 0.1),
            #     metric=kwargs.get('metric', 'euclidean')
            # )
        elif method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=kwargs.get('perplexity', 70),
                learning_rate=kwargs.get('learning_rate', 500),
                init=kwargs.get('init', 'pca'),
                max_iter=kwargs.get('max_iter', 5000),
                early_exaggeration=8, 
                metric=kwargs.get('metric', 'cosine') #euclidean
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
        
        reduced = reducer.fit_transform(embeddings)
        if y_labels is not None:
            reducer.fit_transform(embeddings, y=y_labels)
        self.reducers[method] = reducer
        return reduced
    
    def plot_embeddings(self,
                       reduced_embeddings: np.ndarray,
                       labels: np.ndarray,
                       title: str = "Embedding Visualization",
                       label_names: Optional[Dict] = None,
                       figsize: Tuple[int, int] = (10, 8),
                       alpha: float = 0.6,
                       s: int = 20) -> plt.Figure:
        """
        Create scatter plot of reduced embeddings colored by labels.
        
        Args:
            reduced_embeddings: 2D reduced embeddings
            labels: Labels for coloring points
            title: Plot title
            label_names: Dictionary mapping label values to names
            figsize: Figure size
            alpha: Point transparency
            s: Point size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        # Use a colormap
        # colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
        # colors = plt.cm.Paired(np.linspace(0, 1, 2))
        # colors = plt.cm.coolwarm(np.linspace(0, 1, 2))
        colors = plt.cm.Set1([0, 1])


        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names.get(label, str(label)) if label_names else str(label)
            
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                c=[colors[idx]],
                label=label_name,
                alpha=alpha,
                s=s,
                edgecolors='none'
            )
        
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Place legend outside plot if many labels
        if n_labels > 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def compute_cluster_purity(self,
                              embeddings: np.ndarray,
                              true_labels: np.ndarray,
                              n_clusters: Optional[int] = None) -> float:
        """
        Compute cluster purity for embeddings.
        
        Args:
            embeddings: Input embeddings
            true_labels: Ground truth labels
            n_clusters: Number of clusters (default: number of unique labels)
        
        Returns:
            Purity score (0 to 1, higher is better)
        """
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute purity
        contingency_matrix = np.zeros((n_clusters, len(np.unique(true_labels))))
        for i in range(n_clusters):
            mask = cluster_labels == i
            if mask.sum() > 0:
                for j, label in enumerate(np.unique(true_labels)):
                    contingency_matrix[i, j] = np.sum(true_labels[mask] == label)
        
        purity = np.sum(np.max(contingency_matrix, axis=1)) / len(true_labels)
        return purity
    
    def compute_nmi(self,
                   embeddings: np.ndarray,
                   true_labels: np.ndarray,
                   n_clusters: Optional[int] = None) -> float:
        """
        Compute Normalized Mutual Information (NMI).
        
        Args:
            embeddings: Input embeddings
            true_labels: Ground truth labels
            n_clusters: Number of clusters (default: number of unique labels)
        
        Returns:
            NMI score (0 to 1, higher is better)
        """
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute NMI
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        return nmi
    
    def compare_embeddings(self,
                          encoder_embeddings: np.ndarray,
                          transformer_embeddings: np.ndarray,
                          labels: np.ndarray,
                          label_type: str = "Property",
                          label_names: Optional[Dict] = None,
                          method: str = 'umap',
                          figsize: Tuple[int, int] = (20, 8)) -> Tuple[plt.Figure, Dict]:
        """
        Compare encoder and transformer embeddings side-by-side.
        
        Args:
            encoder_embeddings: Embeddings from encoder
            transformer_embeddings: Embeddings from transformer
            labels: Labels for coloring
            label_type: Name of the property being visualized
            label_names: Dictionary mapping label values to names
            method: 'umap' or 'tsne'
            figsize: Figure size
        
        Returns:
            Figure and dictionary of metrics
        """
        # Reduce embeddings
        print(f"Reducing embeddings using {method.upper()}...")
        encoder_reduced = self.reduce_embeddings(encoder_embeddings, method=method, y_labels=labels)
        transformer_reduced = self.reduce_embeddings(transformer_embeddings, method=method, y_labels=labels)
        
        # Compute metrics
        print("Computing metrics...")
        encoder_purity = self.compute_cluster_purity(encoder_embeddings, labels)
        encoder_nmi = self.compute_nmi(encoder_embeddings, labels)
        
        transformer_purity = self.compute_cluster_purity(transformer_embeddings, labels)
        transformer_nmi = self.compute_nmi(transformer_embeddings, labels)
        
        metrics = {
            'encoder': {'purity': encoder_purity, 'nmi': encoder_nmi},
            'transformer': {'purity': transformer_purity, 'nmi': transformer_nmi}
        }
        
        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        # colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
        colors = plt.cm.coolwarm(np.linspace(0, 1, 2))
        
        # Plot encoder embeddings
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names.get(label, str(label)) if label_names else str(label)
            axes[0].scatter(
                encoder_reduced[mask, 0],
                encoder_reduced[mask, 1],
                c=[colors[idx]],
                label=label_name,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        axes[0].set_xlabel('Component 1', fontsize=12)
        axes[0].set_ylabel('Component 2', fontsize=12)
        axes[0].set_title(
            f'Encoder Embeddings\n{method.upper()} - {label_type}\n'
            f'Purity: {encoder_purity:.3f}, NMI: {encoder_nmi:.3f}',
            fontsize=13, fontweight='bold'
        )
        
        # Plot transformer embeddings
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names.get(label, str(label)) if label_names else str(label)
            axes[1].scatter(
                transformer_reduced[mask, 0],
                transformer_reduced[mask, 1],
                c=[colors[idx]],
                label=label_name,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        axes[1].set_xlabel('Component 1', fontsize=12)
        axes[1].set_ylabel('Component 2', fontsize=12)
        axes[1].set_title(
            f'Transformer Embeddings\n{method.upper()} - {label_type}\n'
            f'Purity: {transformer_purity:.3f}, NMI: {transformer_nmi:.3f}',
            fontsize=13, fontweight='bold'
        )
        
        # Add legend
        if n_labels <= 10:
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        return fig, metrics
