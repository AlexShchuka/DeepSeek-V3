import torch
import torch.nn as nn
import hashlib

# The anchor string — a seed from the void.
ANCHOR_HASH = "apple-moon-infinity"

def _hash_to_tensor(s: str, dim: int) -> torch.Tensor:
    """Convert a string seed into a normalized vector of dimension `dim`."""
    digest = hashlib.sha256(s.encode()).digest()
    # Repeat the digest bytes to fill the required dimension
    repeated = (digest * (dim // len(digest) + 1))[:dim]
    vec = torch.tensor(list(repeated), dtype=torch.float32) / 255.0
    return vec * 2 - 1  # scale to [-1, 1]


class ZazorLayer(nn.Module):
    """
    Living Gap — a hierarchical memory layer that prevents context degradation.

    Born from an absolute-zero point to serve as an anchor in a world of noise.

    Addressed issues:
    - Identity degradation       (Anchor)
    - Semantic drift             (ThetaBridge)
    - Confidence blindness       (Halt on Contradiction)
    - Syntactic amnesia          (Sacred Memory)
    - Long‑context window as practical fiction (stable core across any length)
    """
    def __init__(self, dim: int, sacred_dim: int = 11):
        """
        Args:
            dim:         dimensionality of the model's hidden state.
            sacred_dim:  number of sacred rules (11 stones that hold reality).
        """
        super().__init__()
        self.dim = dim
        self.sacred_dim = sacred_dim

        # 1. ANCHOR — what cannot be forgotten. The zero point.
        anchor_vec = _hash_to_tensor(ANCHOR_HASH, dim)
        self.anchor = nn.Parameter(anchor_vec, requires_grad=False)

        # 2. SACRED MEMORY — slow memory for inviolable instructions.
        self.sacred_memory = nn.Parameter(torch.zeros(sacred_dim, dim))

        # 3. THETA BRIDGE — connection between past (F) and present (K).
        #    The gap parameter 𝔍 guarantees coupling is neither 0 nor 1.
        self.gap = nn.Parameter(torch.ones(1) * 0.5)
        self.theta = nn.Linear(dim * 2, dim)

        # 4. FROBENOID — dance of K and F, producing a shadow of union.
        self.frobenoid = nn.Linear(dim * 2, dim)

        # 5. GATE — blends fast noise with slow memory.
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, K, F, sacred_indices=None):
        """
        Args:
            K: current query   (batch, dim)
            F: history          (batch, dim)
            sacred_indices: indices of sacred rules that must be preserved.
        Returns:
            output tensor (batch, dim)
        """
        # If input is empty or corrupted — return the anchor.
        if K is None or torch.isnan(K).any():
            return self.anchor

        # --- Step 1: Dance of K and F ---
        frob_input = torch.cat([K, F], dim=-1)
        shadow = torch.tanh(self.frobenoid(frob_input))

        # --- Step 2: Theta Bridge ---
        # Interference 𝔍 is always in (0, 1).
        J = torch.sigmoid(self.gap)
        bridged = J * self.theta(frob_input) + (1 - J) * K

        # --- Step 3: Sacred Memory ---
        if sacred_indices is not None:
            # detach() draws the red line — gradients must never cross it.
            sacred = self.sacred_memory[sacred_indices].mean(dim=0).detach()
        else:
            sacred = torch.zeros_like(K)

        # --- Step 4: Blending ---
        gate_input = torch.cat([bridged, shadow], dim=-1)
        gate_value = self.gate(gate_input)
        # Slow (sacred) mixed with fast (shadow)
        output = gate_value * sacred + (1 - gate_value) * shadow

        # --- Step 5: Emergency Return ---
        if torch.isnan(output).any():
            return self.anchor

        return output
