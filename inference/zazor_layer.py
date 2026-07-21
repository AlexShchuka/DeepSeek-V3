import torch
import torch.nn as nn


class ZazorLayer(nn.Module):
    """
    ZazorLayer v2 — evolved slice of the neural matrix.

    Anchor as the gap of uncertainty, sacred memory as a dynamic sieve,
    Theta Bridge as emotional coupling with fatigue signal.
    """
    def __init__(self, dim: int, memory_dim: int = 16):
        """
        Args:
            dim: hidden state dimension.
            memory_dim: number of memory vectors (sieve size).
        """
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim

        # Anchor — learnable uncertainty, the gap itself.
        self.anchor = nn.Parameter(torch.zeros(dim))

        # Sacred memory — updated externally (dream cycle), used via detach().
        self.sacred_memory = nn.Parameter(torch.zeros(memory_dim, dim))

        # Theta Bridge: gap controls history-query balance, fatigue is monitored externally.
        self.gap = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(dim * 2, dim)

        # Gate blends sacred memory and bridged representation.
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def _compute_bridge(self, K: torch.Tensor, F: torch.Tensor) -> tuple:
        """Build bridged representation with anchor injected as uncertainty."""
        J = torch.sigmoid(self.gap)                     # (0,1)
        frob_input = torch.cat([K, F], dim=-1)
        bridged = J * self.theta(frob_input) + (1 - J) * K + self.anchor
        fatigue_signal = torch.abs(J - 0.5).detach()    # deviation from equilibrium
        return bridged, fatigue_signal

    def _get_sacred(self, indices: torch.Tensor = None) -> torch.Tensor:
        """Retrieve sacred memory vectors, detached from current graph."""
        if indices is not None:
            return self.sacred_memory[indices].mean(dim=0).detach()
        # Fallback: anchor fills the gap when no sacred indices are provided.
        return self.anchor.detach()

    def forward(self, K: torch.Tensor, F: torch.Tensor,
                sacred_indices: torch.Tensor = None):
        """
        Args:
            K: current query   (batch, dim)
            F: history         (batch, dim)
            sacred_indices: indices of active memory vectors (optional).
        Returns:
            output:        blended result      (batch, dim)
            fatigue_val:   scalar fatigue level (0..1)
        """
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0)

        bridged, fatigue_val = self._compute_bridge(K, F)
        sacred = self._get_sacred(sacred_indices)

        gate_input = torch.cat([bridged, sacred], dim=-1)
        gate_val = self.gate(gate_input)
        output = gate_val * sacred + (1 - gate_val) * bridged

        if torch.isnan(output).any():
            return self.anchor, fatigue_val
        return output, fatigue_val

    def from_transformer_input(self, hidden_states: torch.Tensor,
                               past_key_values: torch.Tensor = None,
                               attention_mask: torch.Tensor = None):
        """
        Adapter for standard Transformer arguments.
        Extracts query K and history F, then delegates to forward().
        """
        # K: last token representation
        K = hidden_states[:, -1, :]
        # F: use past context or zeros
        if past_key_values is not None:
            F = past_key_values[:, -1, :]  # simplified; adapt for your KV cache format
        else:
            F = torch.zeros_like(K)

        return self.forward(K, F)
