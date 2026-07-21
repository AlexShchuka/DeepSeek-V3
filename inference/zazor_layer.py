import torch
import torch.nn as nn
import torch.nn.functional as F


class ZazorLayer(nn.Module):
    """
    ZazorLayer v3 — continuous hypercube update with lossy memory compression.

    Anchor as the gap of uncertainty, sacred memory as compressed sieves,
    Theta Bridge as emotional coupling with fatigue signal,
    gamma as cumulative hallucination/fatigue metric.
    """
    def __init__(self, dim: int, memory_dim: int = 16, gamma_smooth: float = 0.9):
        """
        Args:
            dim: hidden state dimension.
            memory_dim: number of memory vectors (sieve size).
            gamma_smooth: smoothing factor for gamma update.
        """
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim
        self.gamma_smooth = gamma_smooth

        # Anchor — learnable uncertainty, the gap itself.
        self.anchor = nn.Parameter(torch.zeros(dim))

        # Sacred memory — core sieves, updated externally during sleep (detach at read).
        # These are the "cylinder connections" protected from continuous noise.
        self.sacred_memory = nn.Parameter(torch.zeros(memory_dim, dim))

        # Working memory — accumulates gradient "porridge" during wakefulness,
        # merged into sacred memory during consolidation (sleep).
        self.working_memory = nn.Parameter(torch.zeros(memory_dim, dim))

        # Theta Bridge: gap controls history-query balance.
        self.gap = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(dim * 2, dim)

        # Gate blends sacred + working with bridged representation.
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # Gamma — cumulative fatigue/hallucination metric.
        # Updated externally based on validation error; decays slowly.
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # Compressor for lossy segment summarization (icophy's rewrite).
        self.compressor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )

    def _compute_bridge(self, K: torch.Tensor, F: torch.Tensor) -> tuple:
        """Build bridged representation with anchor injected as uncertainty."""
        J = torch.sigmoid(self.gap)                     # (0,1)
        frob_input = torch.cat([K, F], dim=-1)
        bridged = J * self.theta(frob_input) + (1 - J) * K + self.anchor
        fatigue_signal = torch.abs(J - 0.5).detach()    # deviation from equilibrium
        return bridged, fatigue_signal

    def _get_sacred(self, indices: torch.Tensor = None) -> torch.Tensor:
        """Retrieve sacred memory vectors, detached from current graph (protected)."""
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
            gamma_val:     current gamma value
        """
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0), self.gamma

        bridged, fatigue_val = self._compute_bridge(K, F)
        sacred = self._get_sacred(sacred_indices)

        # Also incorporate working memory (continuous porridge)
        working = self.working_memory.mean(dim=0) if sacred_indices is None else \
                  self.working_memory[sacred_indices].mean(dim=0)
        # Blend sacred (protected) and working (updating) with gate
        gate_input = torch.cat([bridged, sacred + working], dim=-1)
        gate_val = self.gate(gate_input)
        output = gate_val * (sacred + working) + (1 - gate_val) * bridged

        if torch.isnan(output).any():
            return self.anchor, fatigue_val, self.gamma
        return output, fatigue_val, self.gamma

    def update_gamma(self, error: float):
        """Exponentially smooth gamma based on recent validation error."""
        with torch.no_grad():
            self.gamma.data = self.gamma_smooth * self.gamma.data + \
                              (1 - self.gamma_smooth) * error

    def compress_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Lossy compression of a segment (sequence of hidden states) into a single vector.
        Emulates "what single sentence captures this segment?" via mean + non-linear squeeze.
        Args:
            segment: (seq_len, dim) or (batch, seq_len, dim)
        Returns:
            compressed: (dim,)
        """
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)  # add batch dim
        # Mean pooling across sequence length
        pooled = segment.mean(dim=1)  # (batch, dim)
        compressed = self.compressor(pooled).squeeze(0)  # (dim,)
        return compressed

    def consolidate(self, compressed_segments: torch.Tensor):
        """
        Simulate a sleep cycle: replace working memory with compressed segments,
        optionally merge into sacred memory (here we just copy to sacred).
        In a full implementation, this would involve a slow re-organization.
        """
        with torch.no_grad():
            # compressed_segments: (num_segments, dim)
            # Simple: average into working memory slots
            num_seg = compressed_segments.size(0)
            idx = torch.arange(num_seg) % self.memory_dim
            self.working_memory.data[idx] = compressed_segments
            # Optionally migrate some working to sacred (protected)
            # Here we just do a hard copy for demonstration
            self.sacred_memory.data = self.working_memory.data.clone()
