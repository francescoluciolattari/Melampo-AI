import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from transformers import AutoModel, AutoConfig

class SensoryCortex(nn.Module):
    """
    Module: Sensory Cortex (Perception & Attention)
    
    Role:
    Handles the ingestion of raw multimodal data. It implements a 'Late Fusion' mechanism
    where semantic concepts (from Text) act as attention queries to focus on specific 
    volumetric regions (Vision).
    
    Architecture:
    - Vision: SwinUNETR Encoder (3D Volumetric SOTA).
    - Language: BioMistral/Med-PaLM Adapter (LLM).
    - Fusion: Cross-Modal Transformer Decoder.
    """
    
    def __init__(self, img_size=(96, 96, 96), fusion_dim=512):
        super().__init__()
        
        # --- 1. VISUAL STREAM (Occipital Lobe) ---
        # Using SwinUNETR solely as a feature extractor (Encoder only).
        # We capture spatial hierarchies from CT/MRI scans.
        self.vision_backbone = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=14, # Dummy output, we tap into hidden states
            feature_size=48,
            use_checkpoint=True # Gradient checkpointing for memory efficiency
        )
        # Projecting vision features to a common dimension
        self.vis_projector = nn.Linear(768, fusion_dim) 

        # --- 2. LANGUAGE STREAM (Temporal Lobe) ---
        # Adapter for a Medical LLM (e.g., BioMistral).
        # We assume pre-computed embeddings or a frozen backbone with LoRA.
        self.text_dim = 4096 # Dimension of Llama-2/Mistral 7B
        self.text_projector = nn.Linear(self.text_dim, fusion_dim)
        
        # --- 3. ASSOCIATION AREA (Parietal Lobe) ---
        # Cross-Modal Attention: Text queries the Image.
        # "Where in this 3D volume is the 'nodular lesion' mentioned in the text?"
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, image_volume: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_volume: [Batch, 1, D, H, W] - 3D Medical Scan
            text_embeddings: [Batch, Seq_Len, Text_Dim] - Tokenized Report Embeddings
        Returns:
            fused_concept: [Batch, Fusion_Dim] - The abstract representation of the patient.
        """
        
        # A. Visual Processing
        # Extract features from the bottleneck of Swin Transformer
        # We rely on the encoder's deepest layer output.
        # Output shape assumption: [Batch, C, D', H', W']
        vision_hidden = self.vision_backbone.swinViT(image_volume, normalize=True)[4] 
        
        # Flatten spatial dimensions: [Batch, Features, Voxels] -> [Batch, Voxels, Features]
        b, c, d, h, w = vision_hidden.shape
        vision_seq = vision_hidden.flatten(2).permute(0, 2, 1) 
        vision_seq = self.vis_projector(vision_seq) # [Batch, Voxels, 512]

        # B. Textual Processing
        text_seq = self.text_projector(text_embeddings) # [Batch, Seq_Len, 512]

        # C. Cross-Modal Attention (The "Aha!" moment)
        # Query = Text (Concepts we are looking for)
        # Key/Value = Image (Raw data we are scanning)
        attn_out, _ = self.cross_attention(
            query=text_seq, 
            key=vision_seq, 
            value=vision_seq
        )
        
        # Residual connection + Norm
        fused_seq = self.layer_norm(text_seq + attn_out)
        
        # Pooling: Aggregate the sequence into a single "Patient Concept Vector"
        # We use Global Average Pooling over the sequence length
        patient_concept = fused_seq.mean(dim=1) 
        
        return patient_concept
        