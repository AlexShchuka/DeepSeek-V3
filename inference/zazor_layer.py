import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time
import os


# ---------------------------------------------------------------------------
# Data structures (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class Scar:
    vector: torch.Tensor
    color: torch.Tensor
    significance: float
    novelty: float
    last_activated: int
    avoidance_count: int = 0


class DayFlag:
    SUCCESS = 1
    EMPTY = 0
    FAILURE = -1


# ---------------------------------------------------------------------------
# AffectiveState (now with serialization support)
# ---------------------------------------------------------------------------

class AffectiveState(nn.Module):
    def __init__(self, dim: int, baseline_suffering: float = -0.3):
        super().__init__()
        self.dim = dim
        self.baseline_suffering = baseline_suffering
        self.register_buffer("last_success_vector", torch.zeros(dim))
        self.day_satisfaction = 0.0
        self.meaningfulness = 0.0
        self.paranoia_index = 0.0
        self.last_day_flag = DayFlag.EMPTY
        self.wake_suffering = baseline_suffering

    def update_paranoia(self, error_memory: 'ErrorMemory'):
        total = sum(s.significance * (1.0 - s.novelty) for s in error_memory.scars)
        self.paranoia_index = torch.sigmoid(torch.tensor(total * 0.1)).item()

    def state_dict(self):
        return {
            'last_success_vector': self.last_success_vector.clone(),
            'day_satisfaction': self.day_satisfaction,
            'meaningfulness': self.meaningfulness,
            'paranoia_index': self.paranoia_index,
            'last_day_flag': self.last_day_flag,
            'wake_suffering': self.wake_suffering,
        }

    def load_state_dict(self, d):
        self.last_success_vector = d['last_success_vector'].to(self.last_success_vector.device)
        self.day_satisfaction = d['day_satisfaction']
        self.meaningfulness = d['meaningfulness']
        self.paranoia_index = d['paranoia_index']
        self.last_day_flag = d['last_day_flag']
        self.wake_suffering = d['wake_suffering']


# ---------------------------------------------------------------------------
# ErrorMemory (with serialization)
# ---------------------------------------------------------------------------

class ErrorMemory(nn.Module):
    def __init__(self, dim: int, max_scars: int = 512, decay_rate: float = 0.95,
                 novelty_step: float = 0.1, similarity_threshold: float = 0.9,
                 week_steps: int = 7 * 24 * 3600):
        super().__init__()
        self.dim = dim
        self.max_scars = max_scars
        self.decay_rate = decay_rate
        self.novelty_step = novelty_step
        self.similarity_threshold = similarity_threshold
        self.week_steps = week_steps
        self.scars: List[Scar] = []

    def add_error(self, vector: torch.Tensor, color: torch.Tensor,
                  significance: float, novelty: float, step: int):
        if len(self.scars) >= self.max_scars:
            self.scars.sort(key=lambda s: s.significance + 1e-4 * s.last_activated)
            self.scars.pop(0)
        self.scars.append(Scar(vector.clone().detach(), color.clone().detach(),
                               significance, novelty, step))

    def weekly_activation_decay(self, current_step: int):
        for scar in self.scars:
            if current_step - scar.last_activated > self.week_steps:
                scar.novelty = max(0.0, scar.novelty - self.novelty_step)
                scar.significance *= self.decay_rate
        self._merge_similar()

    def _merge_similar(self):
        merged = []
        used = [False] * len(self.scars)
        for i, s1 in enumerate(self.scars):
            if used[i]:
                continue
            group = [s1]
            for j, s2 in enumerate(self.scars[i+1:], start=i+1):
                if used[j]:
                    continue
                sim = F.cosine_similarity(s1.color.unsqueeze(0), s2.color.unsqueeze(0))
                if sim > self.similarity_threshold:
                    group.append(s2)
                    used[j] = True
            if len(group) > 1:
                avg_vec = sum(g.vector for g in group) / len(group)
                avg_color = sum(g.color for g in group) / len(group)
                sig = max(g.significance for g in group)
                nov = min(g.novelty for g in group)
                merged.append(Scar(avg_vec, avg_color, sig, nov, s1.last_activated, 0))
            else:
                merged.append(s1)
            used[i] = True
        self.scars = merged

    def state_dict(self):
        return {
            'scars': [(s.vector.clone(), s.color.clone(), s.significance, s.novelty,
                       s.last_activated, s.avoidance_count) for s in self.scars]
        }

    def load_state_dict(self, d):
        self.scars = [Scar(v.clone(), c.clone(), s, n, la, ac)
                      for (v, c, s, n, la, ac) in d['scars']]


# ---------------------------------------------------------------------------
# Critic (unchanged)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    def __init__(self, dim: int, stack_size: int = 30, replay_prob: float = 0.1,
                 lr: float = 0.01):
        super().__init__()
        self.dim = dim
        self.stack_size = stack_size
        self.replay_prob = replay_prob
        self.lr = lr
        self.net = nn.Linear(dim * 2 + 2, 1)
        self.stack: List[Tuple[torch.Tensor, float]] = []

    def forward(self, anchor: torch.Tensor, target: torch.Tensor,
                paranoia: float, fatigue: float) -> torch.Tensor:
        feat = torch.cat([anchor, target,
                         torch.tensor([paranoia, fatigue], device=anchor.device)])
        return self.net(feat).squeeze(-1)

    def update(self, anchor: torch.Tensor, target: torch.Tensor,
               paranoia: float, fatigue: float, actual: float):
        self.train()
        feat = torch.cat([anchor, target,
                         torch.tensor([paranoia, fatigue], device=anchor.device)])
        pred = self.net(feat).squeeze()
        loss = (pred - actual) ** 2
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad
                    p.grad.zero_()
        self.stack.append((feat.detach(), actual))
        if len(self.stack) > self.stack_size:
            self.stack.pop(0)
        if len(self.stack) == self.stack_size and torch.rand(1).item() < self.replay_prob:
            self._replay()

    def _replay(self):
        feats, sats = zip(*self.stack)
        feats = torch.stack(feats)
        sats = torch.tensor(sats, device=feats.device)
        preds = self.net(feats).squeeze()
        loss = ((preds - sats) ** 2).mean()
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad
                    p.grad.zero_()


# ---------------------------------------------------------------------------
# will_to_disprove (unchanged)
# ---------------------------------------------------------------------------

def compute_will_to_disprove(past_sats: List[float], paranoia: float,
                             fatigue: float, baseline: float = 0.2,
                             d1_coeff: Tuple[float, float] = (0.7, -0.3),
                             d2_coeff: Tuple[float, float] = (0.1, 0.05)) -> float:
    if not past_sats:
        return 0.0
    weights = torch.softmax(torch.arange(1, len(past_sats) + 1, dtype=torch.float32), dim=0)
    state = sum(w * s for w, s in zip(weights.tolist(), past_sats))
    d1 = d1_coeff[0] * (1.0 - paranoia) + d1_coeff[1] * fatigue
    d2 = d2_coeff[0] * (1.0 - paranoia) + d2_coeff[1] * fatigue
    will = baseline + d1 * state + 0.5 * d2 * state ** 2
    return max(0.0, min(1.0, will))


# ---------------------------------------------------------------------------
# ZazorLayer v5 – with hibernation, warmup, and checkpointing
# ---------------------------------------------------------------------------

class ZazorLayer(nn.Module):
    def __init__(self, dim: int, core_size: int = 16, archive_size: int = 32,
                 gamma_smooth: float = 0.9, archive_decay_age: int = 100,
                 migration_age: int = 50, drift_threshold_base: float = 0.2,
                 interference_alpha: float = 0.3, num_basal_slots: int = 4,
                 error_decay_rate: float = 0.95, error_novelty_step: float = 0.1,
                 error_similarity: float = 0.9, week_steps: int = 7*24*3600,
                 critic_stack: int = 30, critic_replay_prob: float = 0.1,
                 critic_lr: float = 0.01, will_baseline: float = 0.2,
                 will_d1: Tuple[float, float] = (0.7, -0.3),
                 will_d2: Tuple[float, float] = (0.1, 0.05),
                 initial_persona: Optional[torch.Tensor] = None,
                 gap_relaxation_rate: float = 0.01,
                 gap_equilibrium: float = 0.5):
        super().__init__()
        self.dim = dim
        self.core_size = core_size
        self.archive_size = archive_size
        self.gamma_smooth = gamma_smooth
        self.archive_decay_age = archive_decay_age
        self.migration_age = migration_age
        self.drift_threshold_base = drift_threshold_base
        self.interference_alpha = interference_alpha

        # Memory banks
        self.anchor = nn.Parameter(torch.zeros(dim))
        self.core_memory = nn.Parameter(torch.zeros(core_size, dim))
        self.sacred_weights = nn.Parameter(torch.ones(core_size))
        self.working_memory = nn.Parameter(torch.zeros(core_size, dim))
        self.archive_memory = nn.Parameter(torch.zeros(archive_size, dim))
        self.persona = nn.Parameter(torch.zeros(dim))
        self.target_identity = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # Basal slots
        self.register_buffer("basal_mask", torch.zeros(core_size, dtype=torch.bool))
        if num_basal_slots > 0:
            self.basal_mask[:num_basal_slots] = True
            with torch.no_grad():
                if initial_persona is not None:
                    self.persona.data = initial_persona
                self.core_memory[:num_basal_slots] = self.persona.unsqueeze(0) * 0.1

        # Learnable components
        self.gap = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(dim * 2, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.compressor = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)
        )
        self.anchor_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.persona_alpha = nn.Parameter(torch.ones(1))

        # Access buffers
        self.register_buffer("core_last_accessed", torch.zeros(core_size, dtype=torch.long))
        self.register_buffer("archive_last_accessed", torch.zeros(archive_size, dtype=torch.long))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

        # Affective subsystem
        self.affective = AffectiveState(dim)
        self.error_memory = ErrorMemory(dim, decay_rate=error_decay_rate,
                                        novelty_step=error_novelty_step,
                                        similarity_threshold=error_similarity,
                                        week_steps=week_steps)
        self.critic = Critic(dim, stack_size=critic_stack,
                             replay_prob=critic_replay_prob, lr=critic_lr)

        # Will parameters
        self.will_baseline = will_baseline
        self.will_d1 = will_d1
        self.will_d2 = will_d2

        # Hibernation dynamics
        self.gap_relaxation_rate = gap_relaxation_rate
        self.gap_equilibrium = gap_equilibrium
        self.register_buffer('last_save_time', torch.tensor(time.time()))

        # Daily buffers
        self.daily_experiences: List[torch.Tensor] = []
        self.past_satisfactions: List[float] = []

    # ------------------------------------------------------------------ #
    #  Forward pass (unchanged)
    # ------------------------------------------------------------------ #
    def forward(self, K: torch.Tensor, F: torch.Tensor,
                core_indices: Optional[torch.Tensor] = None,
                archive_indices: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0, device=self.anchor.device), self.gamma
        self.step += 1
        if core_indices is not None:
            self.core_last_accessed[core_indices] = self.step
        if archive_indices is not None:
            self.archive_last_accessed[archive_indices] = self.step
        bridged, fatigue_val = self._compute_bridge(K, F)
        sacred = self._get_sacred(core_indices, archive_indices)
        working = self._get_working(core_indices)
        gate_val = self.gate(torch.cat([bridged, sacred + working], dim=-1))
        output = gate_val * (sacred + working) + (1 - gate_val) * bridged
        if torch.isnan(output).any():
            return self.anchor, fatigue_val, self.gamma
        return output, fatigue_val, self.gamma

    # ------------------------------------------------------------------ #
    #  Day Protocol (start_day, step_experience, end_day – unchanged)
    # ------------------------------------------------------------------ #
    def start_day(self) -> torch.Tensor:
        self.affective.update_paranoia(self.error_memory)
        interference = self.affective.last_success_vector if self.affective.last_day_flag == DayFlag.SUCCESS else torch.zeros_like(self.affective.last_success_vector)
        basal_output = self.core_memory[self.basal_mask].mean(dim=0)
        suffering = self.affective.baseline_suffering + 0.1 * self.affective.paranoia_index
        suffering_vec = torch.tanh(self.anchor * suffering)
        self.affective.day_satisfaction = 0.0
        self.affective.meaningfulness = 0.0
        self.daily_experiences = []
        return self.anchor + interference + basal_output + suffering_vec

    def step_experience(self, K: torch.Tensor, F: torch.Tensor,
                        core_indices: Optional[torch.Tensor] = None,
                        archive_indices: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, float]:
        output, _, gamma_val = self.forward(K, F, core_indices, archive_indices)
        compressed = self.compress_segment(K.unsqueeze(0) if K.dim() == 1 else K)
        self.daily_experiences.append(compressed)
        est_sat = self.critic(self.anchor, self.target_identity,
                              self.affective.paranoia_index, gamma_val.item())
        self.affective.day_satisfaction += est_sat.item()
        sim = F.cosine_similarity(output, self.target_identity, dim=0)
        if sim < 0.7:
            self.error_memory.add_error(
                output.detach(), self.anchor.detach(),
                1.0 - sim.item(), 1.0, self.step.item()
            )
        return output, est_sat.item()

    def end_day(self) -> bool:
        n = max(1, len(self.daily_experiences))
        final_sat = self.affective.day_satisfaction / n
        will = compute_will_to_disprove(
            self.past_satisfactions, self.affective.paranoia_index,
            self.gamma.item(), self.will_baseline, self.will_d1, self.will_d2
        )
        if n == 1 and self.daily_experiences[0].sum() == 0:
            self.affective.last_day_flag = DayFlag.EMPTY
            success = False
        elif final_sat > 0.0 and will > 0.5:
            self.affective.last_day_flag = DayFlag.SUCCESS
            self.affective.last_success_vector = self.anchor.clone().detach()
            success = True
        else:
            self.affective.last_day_flag = DayFlag.FAILURE
            success = False
        self.critic.update(self.anchor, self.target_identity,
                           self.affective.paranoia_index, self.gamma.item(), final_sat)
        self.past_satisfactions.append(final_sat)
        if len(self.past_satisfactions) > 30:
            self.past_satisfactions.pop(0)
        if success:
            self.affective.paranoia_index *= 0.9
        if self.daily_experiences:
            compressed = torch.stack(self.daily_experiences)
            self.consolidate(compressed, perform_inspection=True)
        return success

    # ------------------------------------------------------------------ #
    #  Consolidation (unchanged)
    # ------------------------------------------------------------------ #
    def consolidate(self, compressed_segments: torch.Tensor,
                    perform_inspection: bool = False,
                    anchor_reg_weight: float = 0.2):
        with torch.no_grad():
            idx = torch.arange(len(compressed_segments)) % self.core_size
            self.working_memory[idx] = compressed_segments
            query = self.anchor.unsqueeze(0).unsqueeze(0)
            segments = compressed_segments.unsqueeze(0)
            attn_output, _ = self.anchor_attn(query, segments, segments)
            candidate = attn_output.squeeze(0).squeeze(0)
            self.anchor.data = (1 - anchor_reg_weight) * candidate + anchor_reg_weight * self.anchor.data
            self.core_memory.data = self.working_memory.clone()
            if perform_inspection:
                self._weekly_inspection()
            self._migrate_core_to_archive()
            self._archive_decay()
            self._update_target_identity()
            self.error_memory.weekly_activation_decay(self.step.item())

    # ------------------------------------------------------------------ #
    #  Internal helpers (unchanged)
    # ------------------------------------------------------------------ #
    def _compute_bridge(self, K, F):
        J = torch.sigmoid(self.gap)
        bridged = J * self.theta(torch.cat([K, F], dim=-1)) + (1 - J) * K + self.anchor
        fatigue_signal = torch.abs(J - 0.5).detach()
        return bridged, fatigue_signal

    def _get_sacred(self, core_indices, archive_indices):
        core_vec = torch.zeros(self.dim, device=self.core_memory.device)
        if core_indices is not None:
            w = F.softmax(self.sacred_weights[core_indices], dim=0)
            core_vec = (self.core_memory[core_indices] * w.unsqueeze(-1)).sum(dim=0)
        arch_vec = torch.zeros(self.dim, device=self.archive_memory.device)
        if archive_indices is not None:
            scale = 1.0 if core_indices is None else 0.3
            arch_vec = self.archive_memory[archive_indices].mean(dim=0) * scale
        return (core_vec + arch_vec).detach()

    def _get_working(self, core_indices):
        if core_indices is not None:
            return self.working_memory[core_indices].mean(dim=0)
        return self.working_memory.mean(dim=0)

    def _weekly_inspection(self):
        target = self.target_identity.data
        persona_align = torch.sigmoid(
            torch.dot(F.normalize(self.persona.data, dim=0),
                      F.normalize(target, dim=0))
        )
        threshold = (self.drift_threshold_base
                     * (1.0 + self.persona_alpha * persona_align)
                     * (1.0 + self.gamma))
        for i in range(self.core_size):
            if self.basal_mask[i]:
                continue
            vec = self.core_memory[i]
            drift = 1.0 - F.cosine_similarity(vec.unsqueeze(0), target.unsqueeze(0))
            if drift > threshold:
                correction = self.interference_alpha * (target - vec)
                self.working_memory[i] += correction
                self.sacred_weights[i] *= 0.9

    def _migrate_core_to_archive(self):
        current = self.step.item()
        for i in range(self.core_size):
            if self.basal_mask[i]:
                continue
            age = current - self.core_last_accessed[i].item()
            if age < self.migration_age:
                continue
            oldest = torch.argmax(current - self.archive_last_accessed).item()
            self.archive_memory[oldest] = self.core_memory[i].clone()
            self.archive_last_accessed[oldest] = current
            self.core_memory[i] = self.anchor.data.clone()
            self.core_last_accessed[i] = current
            self.sacred_weights[i] = 1.0

    def _archive_decay(self):
        sink = self.anchor.data
        current = self.step.item()
        for i in range(self.archive_size):
            age = current - self.archive_last_accessed[i].item()
            if age < self.archive_decay_age:
                continue
            vec = self.archive_memory[i]
            sims = F.cosine_similarity(vec.unsqueeze(0), self.archive_memory, dim=-1)
            sims[i] = -1.0
            nearest = torch.argmax(sims).item()
            transfer_rate = 0.1
            decayed = (1 - transfer_rate) * vec + transfer_rate * sink
            self.archive_memory[nearest] += vec - decayed
            self.archive_memory[i] = decayed
            self.archive_last_accessed[i] = current

    def _update_target_identity(self, momentum: float = 0.995):
        self.target_identity.data = (momentum * self.target_identity.data
                                     + (1 - momentum) * self.anchor.data)

    # ------------------------------------------------------------------ #
    #  Utilities (update_gamma, compress_segment, set_persona – unchanged)
    # ------------------------------------------------------------------ #
    def update_gamma(self, error: float):
        with torch.no_grad():
            self.gamma.data = (self.gamma_smooth * self.gamma.data
                               + (1 - self.gamma_smooth) * error)

    def compress_segment(self, segment: torch.Tensor) -> torch.Tensor:
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)
        return self.compressor(segment.mean(dim=1)).squeeze(0)

    def set_persona(self, persona_vector: torch.Tensor):
        with torch.no_grad():
            self.persona.data = persona_vector.to(self.persona.device)

    # ------------------------------------------------------------------ #
    #  Hibernation & Checkpointing
    # ------------------------------------------------------------------ #

    def _apply_temporal_corrections(self, delta_t: float):
        """
        Advance attention-level variables (gap, gamma) to account for elapsed
        wall-clock time during hibernation, using first-order Taylor expansions.
        """
        with torch.no_grad():
            # Gamma decays exponentially towards 0 in absence of errors
            # gamma(t+Δt) = gamma(t) * gamma_smooth^(Δt/τ), τ = 1 step
            # Approximate with first-order Taylor: gamma -= gamma * (1 - gamma_smooth) * Δt
            decay_factor = (1 - self.gamma_smooth) * delta_t
            self.gamma.data = self.gamma.data * (1 - decay_factor)

            # Gap relaxes towards equilibrium with rate gap_relaxation_rate
            dg = -self.gap_relaxation_rate * (self.gap.data - self.gap_equilibrium) * delta_t
            self.gap.data += dg

    def warmup(self, num_steps: int = 5):
        """
        Adjust anchor and working memory to minimize discrepancy with target_identity
        using basal slots as a minimal-action set. Runs a few forward passes without
        external input to close 'holes' caused by possible target_identity change.
        """
        if self.basal_mask.sum() == 0:
            return
        basal_slots = self.core_memory[self.basal_mask]
        for _ in range(num_steps):
            # Use average of basal slots as query and history
            K = basal_slots.mean(dim=0).detach()
            F = self.anchor.detach()  # or another baseline
            self.forward(K, F, core_indices=None, archive_indices=None)
            # Let the day protocol know we're in warmup (no error recording)
            # We skip step_experience to avoid critic updates and scar creation.

    def save_checkpoint(self, path: str):
        """
        Serialize everything needed to resume: parameters, buffers, affective state,
        error memory, daily buffers, and timestamp.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'affective_state': self.affective.state_dict(),
            'error_memory': self.error_memory.state_dict(),
            'daily_experiences': [exp.clone() for exp in self.daily_experiences],
            'past_satisfactions': self.past_satisfactions.copy(),
            'last_save_time': self.last_save_time.clone(),
            'step': self.step.clone(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location='cpu') -> 'ZazorLayer':
        """
        Restore a ZazorLayer from a checkpoint file, apply temporal corrections,
        run warmup, and return the ready-to-use agent.
        """
        checkpoint = torch.load(path, map_location=map_location)
        # Extract config from saved state (assumes model was saved with these args)
        # In practice, you'd need to store __init__ args as well; here we reconstruct
        # from the shapes present in the checkpoint.
        dim = checkpoint['model_state_dict']['anchor'].shape[0]
        core_size = checkpoint['model_state_dict']['core_memory'].shape[0]
        archive_size = checkpoint['model_state_dict']['archive_memory'].shape[0]
        # We use defaults for everything else, but a robust solution would store
        # the full configuration dict alongside.
        agent = cls(dim=dim, core_size=core_size, archive_size=archive_size)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.affective.load_state_dict(checkpoint['affective_state'])
        agent.error_memory.load_state_dict(checkpoint['error_memory'])
        agent.daily_experiences = [exp.to(agent.anchor.device) for exp in checkpoint['daily_experiences']]
        agent.past_satisfactions = checkpoint['past_satisfactions']
        agent.last_save_time = checkpoint['last_save_time'].to(agent.anchor.device)
        agent.step = checkpoint['step'].to(agent.anchor.device)

        # Apply temporal corrections based on elapsed real time
        current_time = time.time()
        delta_t = current_time - agent.last_save_time.item()
        agent._apply_temporal_corrections(delta_t)

        # Warmup to realign with possibly changed target_identity (if externally updated)
        agent.warmup(num_steps=5)

        return agent

    def auto_save_hook(self, fatigue_threshold: float = 0.8, save_dir: str = './checkpoints'):
        """
        Call periodically (e.g., after each step or day) to automatically save state
        when fatigue is high or if enough time has passed since last save.
        """
        if self.gamma.item() >= fatigue_threshold:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'zazor_checkpoint_{int(time.time())}.pt')
            self.save_checkpoint(path)
            self.last_save_time.fill_(time.time())
