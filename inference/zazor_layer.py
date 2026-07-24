import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import os


_ERROR_SORT_EPSILON = 1e-4

_LOSS_FN_MAP = {
    'mse': lambda: nn.MSELoss(reduction='mean'),
    'huber': lambda: nn.HuberLoss(reduction='mean'),
}


@dataclass
class AffectiveConfig:
    baseline_suffering: float = -0.3


@dataclass
class ErrorMemoryConfig:
    max_scars: int = 512
    decay_rate: float = 0.95
    novelty_step: float = 0.1
    similarity_threshold: float = 0.9
    slow_cycle_seconds: float = 604800.0
    sort_epsilon: float = _ERROR_SORT_EPSILON


@dataclass
class CriticConfig:
    stack_size: int = 30
    replay_prob: float = 0.1
    lr: float = 0.01
    loss_fn: str = 'mse'


@dataclass
class WillConfig:
    baseline: float = 0.2
    d1_coeff: Tuple[float, float] = (0.7, -0.3)
    d2_coeff: Tuple[float, float] = (0.1, 0.05)


@dataclass
class ZazorConfig:
    dim: int
    core_size: int = 16
    archive_size: int = 32
    gamma_smooth: float = 0.9
    archive_decay_age_seconds: float = 100.0
    migration_age_factor: float = 1.5
    drift_threshold_base: float = 0.2
    interference_alpha: float = 0.3
    num_basal_slots: int = 4
    initial_persona: Optional[torch.Tensor] = None
    gap_relaxation_rate: float = 0.01
    gap_equilibrium: float = 0.5
    affective: AffectiveConfig = field(default_factory=AffectiveConfig)
    error_memory: ErrorMemoryConfig = field(default_factory=ErrorMemoryConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    will: WillConfig = field(default_factory=WillConfig)
    time_provider: Optional[Callable[[], float]] = None
    warmup_steps: int = 5


@dataclass
class Scar:
    vector: torch.Tensor
    color: torch.Tensor
    significance: float
    novelty: float
    last_activated: float
    involved_core_indices: List[int] = field(default_factory=list)
    avoidance_count: int = 0


class CycleFlag:
    SUCCESS = 1
    EMPTY = 0
    FAILURE = -1


class AffectiveState(nn.Module):
    def __init__(self, dim: int, config: AffectiveConfig):
        super().__init__()
        self.dim = dim
        self.baseline_suffering = config.baseline_suffering
        self.register_buffer("last_success_vector", torch.zeros(dim))
        self.day_satisfaction = 0.0
        self.meaningfulness = 0.0
        self.paranoia_index = 0.0
        self.last_cycle_flag = CycleFlag.EMPTY
        self.wake_suffering = config.baseline_suffering

    def update_paranoia(self, error_memory: 'ErrorMemory'):
        total = sum(s.significance * (1.0 - s.novelty) for s in error_memory.scars)
        self.paranoia_index = torch.sigmoid(torch.tensor(total * 0.1)).item()

    def state_dict(self):
        return {
            'last_success_vector': self.last_success_vector.clone(),
            'day_satisfaction': self.day_satisfaction,
            'meaningfulness': self.meaningfulness,
            'paranoia_index': self.paranoia_index,
            'last_cycle_flag': self.last_cycle_flag,
            'wake_suffering': self.wake_suffering,
        }

    def load_state_dict(self, d):
        self.last_success_vector = d['last_success_vector'].to(self.last_success_vector.device)
        self.day_satisfaction = d['day_satisfaction']
        self.meaningfulness = d['meaningfulness']
        self.paranoia_index = d['paranoia_index']
        self.last_cycle_flag = d['last_cycle_flag']
        self.wake_suffering = d['wake_suffering']


class ErrorMemory(nn.Module):
    def __init__(self, dim: int, config: ErrorMemoryConfig):
        super().__init__()
        self.dim = dim
        self.max_scars = config.max_scars
        self.decay_rate = config.decay_rate
        self.novelty_step = config.novelty_step
        self.similarity_threshold = config.similarity_threshold
        self.slow_cycle_seconds = config.slow_cycle_seconds
        self.sort_epsilon = config.sort_epsilon
        self.scars: List[Scar] = []

    def add_error(self, vector: torch.Tensor, color: torch.Tensor,
                  significance: float, novelty: float, timestamp: float,
                  involved_core_indices: Optional[List[int]] = None):
        if involved_core_indices is None:
            involved_core_indices = []
        if len(self.scars) >= self.max_scars:
            self.scars.sort(key=lambda s: s.significance + self.sort_epsilon * s.last_activated)
            self.scars.pop(0)
        self.scars.append(Scar(
            vector.clone().detach(), color.clone().detach(),
            significance, novelty, timestamp, involved_core_indices
        ))

    def slow_cycle_decay(self, current_time: float):
        for scar in self.scars:
            if current_time - scar.last_activated > self.slow_cycle_seconds:
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
                last_act = max(g.last_activated for g in group)
                merged.append(Scar(avg_vec, avg_color, sig, nov, last_act,
                                   list(set(i for g in group for i in g.involved_core_indices))))
            else:
                merged.append(s1)
            used[i] = True
        self.scars = merged

    def state_dict(self):
        return {
            'scars': [(s.vector.clone(), s.color.clone(), s.significance, s.novelty,
                       s.last_activated, s.avoidance_count, s.involved_core_indices.copy())
                      for s in self.scars]
        }

    def load_state_dict(self, d):
        self.scars = [Scar(v.clone(), c.clone(), s, n, la, ac, idx.copy())
                      for (v, c, s, n, la, ac, idx) in d['scars']]


def _critic_input_dim(dim: int, extra_scalars: int = 2) -> int:
    """dim for anchor + dim for target + extra_scalars (paranoia, fatigue)"""
    return dim * 2 + extra_scalars


class Critic(nn.Module):
    def __init__(self, dim: int, config: CriticConfig):
        super().__init__()
        self.dim = dim
        self.stack_size = config.stack_size
        self.replay_prob = config.replay_prob
        self.lr = config.lr
        input_dim = _critic_input_dim(dim)
        self.net = nn.Linear(input_dim, 1)
        self.loss_fn = _LOSS_FN_MAP[config.loss_fn]() if isinstance(config.loss_fn, str) else config.loss_fn
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
        loss = self.loss_fn(pred, actual)
        self.zero_grad()
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
        loss = self.loss_fn(preds, sats)
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad
                    p.grad.zero_()


def compute_will_to_disprove(past_sats: List[float], paranoia: float,
                             fatigue: float, baseline: float,
                             d1_coeff: Tuple[float, float],
                             d2_coeff: Tuple[float, float]) -> float:
    if not past_sats:
        return 0.0
    weights = torch.softmax(torch.arange(1, len(past_sats) + 1, dtype=torch.float32), dim=0)
    state = sum(w * s for w, s in zip(weights.tolist(), past_sats))
    d1 = d1_coeff[0] * (1.0 - paranoia) + d1_coeff[1] * fatigue
    d2 = d2_coeff[0] * (1.0 - paranoia) + d2_coeff[1] * fatigue
    will = baseline + d1 * state + 0.5 * d2 * state ** 2
    return max(0.0, min(1.0, will))

class ZazorLayer(nn.Module):
    """
    ZazorLayer integrates memory banks, affective state, error memory,
    and a critic to maintain a stable identity under distributional drift.

    Parameters (via ZazorConfig):
      - Memory sizes (core, archive), smooth factor gamma.
      - Drift detection, interference correction, basal persona slots.
      - Gap relaxation dynamics for hibernation.
      - Sub-configs for affective, error memory, critic, will.
      - Time provider for real-time age tracking.
    """

    def __init__(self, config: ZazorConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        core_size = config.core_size
        archive_size = config.archive_size

        self.anchor = nn.Parameter(torch.zeros(dim))
        self.core_memory = nn.Parameter(torch.zeros(core_size, dim))
        self.sacred_weights = nn.Parameter(torch.ones(core_size))
        self.working_memory = nn.Parameter(torch.zeros(core_size, dim))
        self.archive_memory = nn.Parameter(torch.zeros(archive_size, dim))
        self.persona = nn.Parameter(torch.zeros(dim))
        self.target_identity = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("basal_mask", torch.zeros(core_size, dtype=torch.bool))
        if config.num_basal_slots > 0:
            self.basal_mask[:config.num_basal_slots] = True
            with torch.no_grad():
                if config.initial_persona is not None:
                    self.persona.data = config.initial_persona
                self.core_memory[:config.num_basal_slots] = self.persona.unsqueeze(0) * 0.1

        self.gap = nn.Parameter(torch.zeros(1))
        self.theta = nn.Linear(dim * 2, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.compressor = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim)
        )
        self.anchor_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.persona_alpha = nn.Parameter(torch.ones(1))

        self.register_buffer("core_last_accessed", torch.zeros(core_size, dtype=torch.float))
        self.register_buffer("archive_last_accessed", torch.zeros(archive_size, dtype=torch.float))
        self.register_buffer("slot_trauma", torch.zeros(core_size))
        self.current_time: float = self._get_current_time()

        self.affective = AffectiveState(dim, config.affective)
        self.error_memory = ErrorMemory(dim, config.error_memory)
        self.critic = Critic(dim, config.critic)

        self.will_baseline = config.will.baseline
        self.will_d1 = config.will.d1_coeff
        self.will_d2 = config.will.d2_coeff

        self.gap_relaxation_rate = config.gap_relaxation_rate
        self.gap_equilibrium = config.gap_equilibrium

        self.fast_cycle_experiences: List[torch.Tensor] = []
        self.fast_satisfaction_history: List[float] = []

    def _get_current_time(self) -> float:
        if self.config.time_provider is not None:
            return self.config.time_provider()
        return time.time()

    def start_fast_cycle(self) -> torch.Tensor:
        self.current_time = self._get_current_time()
        self.affective.update_paranoia(self.error_memory)
        interference = self.affective.last_success_vector if self.affective.last_cycle_flag == CycleFlag.SUCCESS else torch.zeros_like(self.affective.last_success_vector)
        basal_output = self.core_memory[self.basal_mask].mean(dim=0)
        suffering = self.affective.baseline_suffering + 0.1 * self.affective.paranoia_index
        suffering_vec = torch.tanh(self.anchor * suffering)
        self.affective.day_satisfaction = 0.0
        self.affective.meaningfulness = 0.0
        self.fast_cycle_experiences = []
        return self.anchor + interference + basal_output + suffering_vec

    def process_step(self, K: torch.Tensor, F: torch.Tensor,
                     core_indices: Optional[torch.Tensor] = None,
                     archive_indices: Optional[torch.Tensor] = None,
                     timestamp: Optional[float] = None
                     ) -> Tuple[torch.Tensor, float]:
        if timestamp is not None:
            self.current_time = timestamp
        else:
            self.current_time = self._get_current_time()
        output, _, gamma_val = self.forward(K, F, core_indices, archive_indices)
        compressed = self.compress_segment(K.unsqueeze(0) if K.dim() == 1 else K)
        self.fast_cycle_experiences.append(compressed)
        est_sat = self.critic(self.anchor, self.target_identity,
                              self.affective.paranoia_index, gamma_val.item())
        self.affective.day_satisfaction += est_sat.item()
        sim = F.cosine_similarity(output, self.target_identity, dim=0)
        if sim < 0.7:
            involved = core_indices.tolist() if core_indices is not None else []
            self.error_memory.add_error(
                output.detach(), self.anchor.detach(),
                1.0 - sim.item(), 1.0, self.current_time,
                involved_core_indices=involved
            )
            if involved:
                with torch.no_grad():
                    for idx in involved:
                        self.slot_trauma[idx] += (1.0 - sim.item())
        return output, est_sat.item()

    def end_fast_cycle(self) -> bool:
        n = max(1, len(self.fast_cycle_experiences))
        final_sat = self.affective.day_satisfaction / n
        will = compute_will_to_disprove(
            self.fast_satisfaction_history, self.affective.paranoia_index,
            self.gamma.item(), self.will_baseline, self.will_d1, self.will_d2
        )
        if n == 1 and self.fast_cycle_experiences[0].sum() == 0:
            self.affective.last_cycle_flag = CycleFlag.EMPTY
            success = False
        elif final_sat > 0.0 and will > 0.5:
            self.affective.last_cycle_flag = CycleFlag.SUCCESS
            self.affective.last_success_vector = self.anchor.clone().detach()
            success = True
        else:
            self.affective.last_cycle_flag = CycleFlag.FAILURE
            success = False
        self.critic.update(self.anchor, self.target_identity,
                           self.affective.paranoia_index, self.gamma.item(), final_sat)
        self.fast_satisfaction_history.append(final_sat)
        if len(self.fast_satisfaction_history) > 30:
            self.fast_satisfaction_history.pop(0)
        if success:
            self.affective.paranoia_index *= 0.9
        if self.fast_cycle_experiences:
            compressed = torch.stack(self.fast_cycle_experiences)
            self.consolidate(compressed, perform_inspection=True)
        return success

    def forward(self, K: torch.Tensor, F: torch.Tensor,
                core_indices: Optional[torch.Tensor] = None,
                archive_indices: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0, device=self.anchor.device), self.gamma
        if core_indices is not None:
            self.core_last_accessed[core_indices] = self.current_time
        if archive_indices is not None:
            self.archive_last_accessed[archive_indices] = self.current_time
        bridged, fatigue_val = self._compute_bridge(K, F)
        sacred = self._get_sacred(core_indices, archive_indices)
        working = self._get_working(core_indices)
        gate_val = self.gate(torch.cat([bridged, sacred + working], dim=-1))
        output = gate_val * (sacred + working) + (1 - gate_val) * bridged
        if torch.isnan(output).any():
            return self.anchor, fatigue_val, self.gamma
        return output, fatigue_val, self.gamma

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
                self._slow_cycle_inspection()
            self._migrate_core_to_archive()
            self._archive_decay()
            self._update_target_identity()
            self.error_memory.slow_cycle_decay(self.current_time)
            self.slot_trauma *= self.config.error_memory.decay_rate

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

    def _slow_cycle_inspection(self):
        target = self.target_identity.data
        persona_align = torch.sigmoid(
            torch.dot(F.normalize(self.persona.data, dim=0),
                      F.normalize(target, dim=0))
        )
        threshold = (self.config.drift_threshold_base
                     * (1.0 + self.persona_alpha * persona_align)
                     * (1.0 + self.gamma))
        paranoia = self.affective.paranoia_index
        alpha_scaled = self.config.interference_alpha * (1.0 - paranoia)
        for i in range(self.core_size):
            if self.basal_mask[i]:
                continue
            vec = self.core_memory[i]
            drift = 1.0 - F.cosine_similarity(vec.unsqueeze(0), target.unsqueeze(0))
            if drift > threshold:
                correction = alpha_scaled * (target - vec)
                self.working_memory[i] += correction
                self.sacred_weights[i] *= 0.9

    def _migrate_core_to_archive(self):
        scores = []
        valid_indices = []
        for i in range(self.core_size):
            if self.basal_mask[i]:
                continue
            age = self.current_time - self.core_last_accessed[i].item()
            instability = 1.0 - self.sacred_weights[i].item()
            trauma = self.slot_trauma[i].item()
            score = age * instability / (1.0 + trauma)
            scores.append(score)
            valid_indices.append(i)
        if not scores:
            return
        mean_score = sum(scores) / len(scores)
        threshold = mean_score * self.config.migration_age_factor

        for i, score in zip(valid_indices, scores):
            if score <= threshold:
                continue
            oldest_idx = torch.argmax(self.current_time - self.archive_last_accessed).item()
            self.archive_memory[oldest_idx] = self.core_memory[i].clone()
            self.archive_last_accessed[oldest_idx] = self.current_time
            self.core_memory[i] = self.anchor.data.clone()
            self.core_last_accessed[i] = self.current_time
            self.sacred_weights[i] = 1.0

    def _archive_decay(self):
        sink = self.anchor.data
        for i in range(self.config.archive_size):
            age = self.current_time - self.archive_last_accessed[i].item()
            if age < self.config.archive_decay_age_seconds:
                continue
            vec = self.archive_memory[i]
            sims = F.cosine_similarity(vec.unsqueeze(0), self.archive_memory, dim=-1)
            sims[i] = -1.0
            nearest = torch.argmax(sims).item()
            transfer_rate = 0.1
            decayed = (1 - transfer_rate) * vec + transfer_rate * sink
            self.archive_memory[nearest] += vec - decayed
            self.archive_memory[i] = decayed
            self.archive_last_accessed[i] = self.current_time

    def _update_target_identity(self, momentum: float = 0.995):
        self.target_identity.data = (momentum * self.target_identity.data
                                     + (1 - momentum) * self.anchor.data)

    def update_gamma(self, error: float):
        with torch.no_grad():
            self.gamma.data = (self.config.gamma_smooth * self.gamma.data
                               + (1 - self.config.gamma_smooth) * error)

    def compress_segment(self, segment: torch.Tensor) -> torch.Tensor:
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)
        return self.compressor(segment.mean(dim=1)).squeeze(0)

    def set_persona(self, persona_vector: torch.Tensor):
        with torch.no_grad():
            self.persona.data = persona_vector.to(self.persona.device)

    def _apply_temporal_corrections(self, delta_t: float):
        with torch.no_grad():
            decay_factor = (1 - self.config.gamma_smooth) * delta_t
            self.gamma.data = self.gamma.data * (1 - decay_factor)
            dg = -self.gap_relaxation_rate * (self.gap.data - self.gap_equilibrium) * delta_t
            self.gap.data += dg

    def warmup(self, num_steps: int = None):
        if num_steps is None:
            num_steps = self.config.warmup_steps
        if self.basal_mask.sum() == 0:
            return
        basal_slots = self.core_memory[self.basal_mask]
        for _ in range(num_steps):
            K = basal_slots.mean(dim=0).detach()
            F = self.anchor.detach()
            self.forward(K, F, core_indices=None, archive_indices=None)

    def save_checkpoint(self, path: str):
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'affective_state': self.affective.state_dict(),
            'error_memory': self.error_memory.state_dict(),
            'fast_cycle_experiences': [exp.clone() for exp in self.fast_cycle_experiences],
            'fast_satisfaction_history': self.fast_satisfaction_history.copy(),
            'current_time': self.current_time,
            'checkpoint_timestamp': self._get_current_time(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location='cpu',
                        time_provider: Optional[Callable[[], float]] = None,
                        resume_time: Optional[float] = None) -> 'ZazorLayer':
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['config']
        if time_provider is not None:
            config.time_provider = time_provider
        agent = cls(config)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.affective.load_state_dict(checkpoint['affective_state'])
        agent.error_memory.load_state_dict(checkpoint['error_memory'])
        agent.fast_cycle_experiences = [exp.to(agent.anchor.device) for exp in checkpoint['fast_cycle_experiences']]
        agent.fast_satisfaction_history = checkpoint['fast_satisfaction_history']
        agent.current_time = checkpoint['current_time']

        current_time = resume_time if resume_time is not None else agent._get_current_time()
        delta_t = current_time - checkpoint['checkpoint_timestamp']
        agent._apply_temporal_corrections(delta_t)
        agent.warmup()
        return agent

    def auto_save_hook(self, fatigue_threshold: float = 0.8, save_dir: str = './checkpoints'):
        if self.gamma.item() >= fatigue_threshold:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'zazor_checkpoint_{int(self._get_current_time())}.pt')
            self.save_checkpoint(path)
