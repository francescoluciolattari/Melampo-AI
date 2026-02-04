import torch
import torch.nn as nn
import math

class QuantumIntuitionEngine(nn.Module):
    """
    Module: Quantum Intuition Engine (Frontal Lobe Decision Making)
    
    Role:
    Replaces standard ML classification with Quantum Probability theory.
    It models diagnosis not as a binary switch, but as a Wave Function collapse.
    
    Key Innovations:
    1. Complex Hilbert Space Projection.
    2. Unitary Time Evolution (Simulating "thinking time").
    3. Neurotransmitter Modulation (Noradrenaline/Dopamine) affecting collapse.
    """
    
    def __init__(self, input_dim=512, num_classes=10, hidden_dim=256):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 1. State Preparation (Encoding Classical -> Quantum)
        # We map real features to Complex Amplitudes (Real + Imaginary)
        self.fc_real = nn.Linear(input_dim, hidden_dim)
        self.fc_imag = nn.Linear(input_dim, hidden_dim)
        
        # 2. Hamiltonian Operator (The Logic Matrix)
        # Represents the energy landscape of diagnoses. 
        # Must be Hermitian for valid quantum mechanics (H = H^dagger).
        # We learn the real and imaginary parts separately.
        self.hamiltonian_real = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.hamiltonian_imag = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
        # 3. Measurement Projectors (Diagnosis Prototypes)
        self.measurement = nn.Linear(hidden_dim, num_classes, dtype=torch.complex64)

    def forward(self, x: torch.Tensor, neurotransmitters: dict) -> tuple:
        """
        Args:
            x: [Batch, Input_Dim] - Patient Concept Vector
            neurotransmitters: Dict with 'noradrenaline' (0.0-1.0), 'dopamine' (0.0-1.0)
        """
        
        # --- Step A: Wave Function Creation |Psi> ---
        psi_real = self.fc_real(x)
        psi_imag = self.fc_imag(x)
        psi = torch.complex(psi_real, psi_imag) # [Batch, Hidden_Dim]
        
        # Normalize to ensure valid quantum state (<Psi|Psi> = 1)
        psi = psi / (torch.norm(psi, dim=1, keepdim=True) + 1e-8)
        
        # --- Step B: Unitary Evolution (The "Thinking" Process) ---
        # U = exp(-i * H * t)
        # We approximate this evolution. Constructing the Hermitian Hamiltonian:
        H_real = torch.triu(self.hamiltonian_real) + torch.triu(self.hamiltonian_real, diagonal=1).t()
        H_imag = torch.triu(self.hamiltonian_imag) - torch.triu(self.hamiltonian_imag, diagonal=1).t()
        H = torch.complex(H_real, H_imag)
        
        # Apply the Hamiltonian interaction (Simplified evolution step for GPU efficiency)
        # This allows interference between features.
        psi_evolved = torch.matmul(psi, H) 
        
        # --- Step C: Measurement (Projection to Diagnosis Space) ---
        # Project state onto pathology axes
        logits_complex = self.measurement(psi_evolved)
        
        # Born Rule: Probability = |Amplitude|^2
        probs = logits_complex.abs().pow(2)
        
        # --- Step D: Neurochemical Modulation ---
        
        # 1. Noradrenaline -> Temperature (Gain/Contrast)
        # High Noradrenaline = Low Entropy (Focus). Low Noradrenaline = High Entropy (Exploration).
        na_level = neurotransmitters.get('noradrenaline', 0.5)
        temperature = torch.exp(torch.tensor(1.0 - na_level)) # Range approx [1.0, 2.7]
        
        # 2. Dopamine -> Bias (Reward Expectation)
        # Increases baseline confidence for likely candidates based on past rewards.
        da_level = neurotransmitters.get('dopamine', 0.5)
        dopamine_bias = da_level * 0.1
        
        # Apply Modulation
        modulated_probs = (probs / temperature.to(probs.device)) + dopamine_bias
        
        return modulated_probs, psi_evolved
        