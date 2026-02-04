import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv # Graph Attention Network

class NeuroSynapticDreamer(nn.Module):
    """
    Module: Neuro-Synaptic Dreamer (Offline Processing)
    
    Role:
    Activated during 'sleep' cycles (offline training).
    It performs two critical functions:
    1. Memory Consolidation: Reconstructs latent concepts using a VAE.
    2. Synaptic Plasticity: Updates the Knowledge Graph using GNNs to find 
       non-obvious connections between pathologies (Intuition building).
    """
    
    def __init__(self, latent_dim=512, hidden_dim=256):
        super().__init__()
        
        # --- 1. Generative Component (VAE) ---
        # Instead of generating pixels, we generate "Concept Vectors"
        self.encoder_mu = nn.Linear(latent_dim, hidden_dim)
        self.encoder_logvar = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # --- 2. Relational Component (Graph Neural Network) ---
        # Graph Attention Layer to update relationship weights between medical concepts
        self.gat_layer = GATConv(in_channels=latent_dim, out_channels=latent_dim, heads=4)

    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_dream(self, concept_graph_x, edge_index):
        """
        Simulates a dreaming cycle.
        
        Args:
            concept_graph_x: [Num_Nodes, Latent_Dim] - Embeddings of known diseases/symptoms
            edge_index: Graph connectivity
            
        Returns:
            reconstructed_concepts: Refined concepts after dreaming
            new_associations: Updated graph embeddings
        """
        
        # A. Variational Abstraction (The "Dream")
        # Compress concepts to their essence and sample variations
        mu = self.encoder_mu(concept_graph_x)
        logvar = self.encoder_logvar(concept_graph_x)
        z = self.reparameterize(mu, logvar)
        
        reconstructed_concepts = self.decoder(z)
        
        # B. Associative Re-weighting (Synaptic Plasticity)
        # Use Attention to strengthen connections between related concepts
        # based on the dreamed variations.
        new_associations = self.gat_layer(reconstructed_concepts, edge_index)
        
        return reconstructed_concepts, new_associations, mu, logvar
        