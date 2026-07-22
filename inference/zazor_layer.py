import torch
import torch.nn as nn
import torch.nn.functional as F


class ZazorLayer(nn.Module):
    """
    ZazorLayer v5 — continuous memory layer with multi‑speed consolidation,
    persona mask, core/archive separation, interference‑based drift correction,
    and internal weekly inspection.

    Memory hierarchy:
        working_memory  – rewritten every consolidation, absorbs interference
        core_memory     – updated during sleep, primary retrieval source
        archive_memory  – slowly accumulating stable patterns, decays when unused

    Anchor evolves smoothly via attention‑based candidate + regularisation.
    Target identity provides a long‑term compass, updated via EMA of Anchor.
    Persona vector modulates drift tolerance and consolidation bias.
    Gamma (fatigue) adjusts plasticity thresholds globally.
    """

    def __init__(
        self,
        dim: int,
        core_size: int = 16,
        archive_size: int = 32,
        gamma_smooth: float = 0.9,
        archive_decay_age: int = 100,
        migration_age: int = 50,
        drift_threshold_base: float = 0.2,
        interference_alpha: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.core_size = core_size
        self.archive_size = archive_size
        self.gamma_smooth = gamma_smooth
        self.archive_decay_age = archive_decay_age
        self.migration_age = migration_age
        self.drift_threshold_base = drift_threshold_base
        self.interference_alpha = interference_alpha

        # ---------- Core state ----------
        self.anchor = nn.Parameter(torch.zeros(dim))

        # Sacred memory – core sieves (primary retrieval)
        self.core_memory = nn.Parameter(torch.zeros(core_size, dim))
        self.sacred_weights = nn.Parameter(torch.ones(core_size))  # per‑slot importance

        # Working memory – updated every consolidation, also receives interference
        self.working_memory = nn.Parameter(torch.zeros(core_size, dim))

        # Archive – long‑term stable patterns
        self.archive_memory = nn.Parameter(torch.zeros(archive_size, dim))

        # Persona / mask – external identity vector
        self.persona = nn.Parameter(torch.zeros(dim))

        # Long‑term identity goal (EMA of anchor)
        self.target_identity = nn.Parameter(torch.zeros(dim), requires_grad=False)

        # Fatigue / hallucination metric
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # ---------- Learnable components ----------
        # Theta Bridge
        self.gap = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(dim * 2, dim)

        # Gate blends sacred + working with bridged representation
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        # Lossy segment compressor
        self.compressor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )

        # Attention for anchor consolidation
        self.anchor_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=1, batch_first=True
        )

        # Persona modulates drift threshold strength
        self.persona_alpha = nn.Parameter(torch.ones(1))

        # ---------- Access tracking (buffers) ----------
        self.register_buffer(
            "core_last_accessed", torch.zeros(core_size, dtype=torch.long)
        )
        self.register_buffer(
            "archive_last_accessed", torch.zeros(archive_size, dtype=torch.long)
        )
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------ #
    #  Forward pass (real‑time blending)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        K: torch.Tensor,
        F: torch.Tensor,
        core_indices: torch.Tensor = None,
        archive_indices: torch.Tensor = None,
    ):
        """
        Args:
            K: current query        (batch, dim)
            F: history              (batch, dim)
            core_indices: indices of active core slots   (batch?, int…)
            archive_indices: indices of active archive slots
        Returns:
            output:      blended result      (batch, dim)
            fatigue_val: scalar fatigue level (0..1)
            gamma_val:   current gamma value
        """
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0), self.gamma

        # Update access clocks
        self.step += 1
        if core_indices is not None:
            self.core_last_accessed[core_indices] = self.step
        if archive_indices is not None:
            self.archive_last_accessed[archive_indices] = self.step

        bridged, fatigue_val = self._compute_bridge(K, F)
        sacred = self._get_sacred(core_indices, archive_indices)
        working = self._get_working(core_indices)

        gate_input = torch.cat([bridged, sacred + working], dim=-1)
        gate_val = self.gate(gate_input)
        output = gate_val * (sacred + working) + (1 - gate_val) * bridged

        if torch.isnan(output).any():
            return self.anchor, fatigue_val, self.gamma
        return output, fatigue_val, self.gamma

    # ------------------------------------------------------------------ #
    #  Consolidation (sleep cycle)
    # ------------------------------------------------------------------ #
    def consolidate(
        self,
        compressed_segments: torch.Tensor,
        perform_inspection: bool = False,
        anchor_reg_weight: float = 0.2,
    ):
        """
        Nightly consolidation.
        - overwrites working_memory
        - smooth Anchor update via attention + regularisation
        - copies working → core_memory
        - optionally runs weekly inspection (drift interference)
        - migrates stable core slots to archive
        - decays unused archive slots
        - updates target_identity
        """
        with torch.no_grad():
            # 1. Fill working memory with compressed segments
            num_seg = compressed_segments.size(0)
            idx = torch.arange(num_seg) % self.core_size
            self.working_memory.data[idx] = compressed_segments

            # 2. Anchor candidate via attention over segments
            query = self.anchor.data.unsqueeze(0).unsqueeze(0)  # (1,1,dim)
            segments = compressed_segments.unsqueeze(0)         # (1,num_seg,dim)
            attn_output, _ = self.anchor_attn(query, segments, segments)
            candidate_anchor = attn_output.squeeze(0).squeeze(0)  # (dim,)

            # 3. Regularised anchor update
            old_anchor = self.anchor.data.clone()
            self.anchor.data = (
                (1 - anchor_reg_weight) * candidate_anchor
                + anchor_reg_weight * old_anchor
            )

            # 4. Core memory ← working memory
            self.core_memory.data = self.working_memory.data.clone()

            # 5. Internal inspection (drift correction)
            if perform_inspection:
                self._weekly_inspection()

            # 6. Migrate stable core slots to archive
            self._migrate_core_to_archive()

            # 7. Decay unused archive slots
            self._archive_decay()

            # 8. Update long‑term target identity
            self._update_target_identity()

    # ------------------------------------------------------------------ #
    #  Public helpers
    # ------------------------------------------------------------------ #
    def update_gamma(self, error: float):
        """Exponentially smooth gamma based on recent validation error."""
        with torch.no_grad():
            self.gamma.data = (
                self.gamma_smooth * self.gamma.data
                + (1 - self.gamma_smooth) * error
            )

    def compress_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Lossy compression of a segment (seq, dim) → (dim,)."""
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)
        pooled = segment.mean(dim=1)
        compressed = self.compressor(pooled).squeeze(0)
        return compressed

    def set_persona(self, persona_vector: torch.Tensor):
        """Externally update the persona/mask vector."""
        with torch.no_grad():
            self.persona.data = persona_vector.to(self.persona.device)

    # ------------------------------------------------------------------ #
    #  Internal mechanics
    # ------------------------------------------------------------------ #
    def _compute_bridge(self, K, F):
        """Bridge with anchor as uncertainty gap."""
        J = torch.sigmoid(self.gap)
        frob_input = torch.cat([K, F], dim=-1)
        bridged = J * self.theta(frob_input) + (1 - J) * K + self.anchor
        fatigue_signal = torch.abs(J - 0.5).detach()
        return bridged, fatigue_signal

    def _get_sacred(self, core_indices, archive_indices):
        """Weighted retrieval from core (primary) and archive (secondary)."""
        core_vec = torch.zeros(self.dim, device=self.core_memory.device)
        if core_indices is not None:
            w = F.softmax(self.sacred_weights[core_indices], dim=0)
            core_vec = (self.core_memory[core_indices] * w.unsqueeze(-1)).sum(dim=0)

        arch_vec = torch.zeros(self.dim, device=self.archive_memory.device)
        if archive_indices is not None:
            # Archive contributes less when core is active
            scale = 1.0 if core_indices is None else 0.3
            arch_vec = self.archive_memory[archive_indices].mean(dim=0) * scale

        return (core_vec + arch_vec).detach()

    def _get_working(self, core_indices):
        """Working memory contribution for the active slots."""
        if core_indices is not None:
            return self.working_memory[core_indices].mean(dim=0)
        return self.working_memory.mean(dim=0)

    def _weekly_inspection(self):
        """Drift check for core anchors, injecting interference if needed."""
        target = self.target_identity.data
        persona_align = torch.sigmoid(
            torch.dot(
                F.normalize(self.persona.data, dim=0),
                F.normalize(target, dim=0),
            )
        )
        # Effective threshold: base modulated by persona alignment and gamma
        threshold = (
            self.drift_threshold_base
            * (1.0 + self.persona_alpha * persona_align)
            * (1.0 + self.gamma)
        )

        for i in range(self.core_size):
            vec = self.core_memory[i]
            sim = F.cosine_similarity(vec.unsqueeze(0), target.unsqueeze(0))
            drift = 1.0 - sim
            if drift > threshold:
                correction = self.interference_alpha * (target - vec)
                self.working_memory[i] += correction
                # Optionally down‑weight the slot temporarily
                self.sacred_weights[i] *= 0.9

    def _migrate_core_to_archive(self):
        """Move long‑unaccessed core slots to archive if stale enough."""
        current_step = self.step.item()
        for i in range(self.core_size):
            age = current_step - self.core_last_accessed[i].item()
            if age < self.migration_age:
                continue
            # Find archive slot with the highest age (least recently used)
            arch_ages = current_step - self.archive_last_accessed
            oldest_idx = torch.argmax(arch_ages).item()
            # Move core vector to archive, reset core to anchor (or random)
            self.archive_memory[oldest_idx] = self.core_memory[i].clone()
            self.archive_last_accessed[oldest_idx] = current_step
            self.core_memory[i] = self.anchor.data.clone()
            self.core_last_accessed[i] = current_step
            self.sacred_weights[i] = 1.0  # reset weight

    def _archive_decay(self):
        """Compress unused archive entries towards sink_point and transfer mass."""
        sink = self.anchor.data  # could also be the mean of archive
        current_step = self.step.item()
        for i in range(self.archive_size):
            age = current_step - self.archive_last_accessed[i].item()
            if age < self.archive_decay_age:
                continue

            vec = self.archive_memory[i]
            # Cosine similarity to all other archive slots
            sims = F.cosine_similarity(
                vec.unsqueeze(0), self.archive_memory, dim=-1
            )
            sims[i] = -1.0  # ignore self
            nearest_idx = torch.argmax(sims).item()
            transfer_rate = 0.1  # fraction of mass moved

            # Move towards sink, transfer residue to nearest neighbour
            decayed = (1.0 - transfer_rate) * vec + transfer_rate * sink
            residue = vec - decayed
            self.archive_memory[nearest_idx] += residue
            self.archive_memory[i] = decayed
            self.archive_last_accessed[i] = current_step  # decay resets timer

    def _update_target_identity(self, momentum: float = 0.995):
        """Slow EMA of Anchor for long‑term identity compass."""
        self.target_identity.data = (
            momentum * self.target_identity.data
            + (1.0 - momentum) * self.anchor.data
        )
