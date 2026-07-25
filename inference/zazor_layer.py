import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import time
import os
import math


@dataclass
class ErrorMemoryConfig:
    max_scars: int = 512
    decay_rate: float = 0.95
    novelty_step: float = 0.1
    similarity_threshold: float = 0.9
    slow_cycle_seconds: float = 604800.0


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
    archive_decay_age_seconds: float = 100.0
    migration_age_factor: float = 1.5
    drift_threshold_base: float = 0.2
    interference_alpha: float = 0.3
    num_basal_slots: int = 4
    initial_persona: Optional[torch.Tensor] = None
    gap_relaxation_rate: float = 0.01
    gap_equilibrium: float = 0.5
    error_memory: ErrorMemoryConfig = field(default_factory=ErrorMemoryConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    will: WillConfig = field(default_factory=WillConfig)
    time_provider: Optional[Callable[[], float]] = None
    warmup_steps: int = 5


class Scar:
    def __init__(self, vector, color, significance, novelty, last_activated, involved_core_indices=None):
        self.vector = vector
        self.color = color
        self.significance = significance
        self.novelty = novelty
        self.last_activated = last_activated
        self.involved_core_indices = involved_core_indices if involved_core_indices is not None else []


class CycleFlag:
    SUCCESS = 1
    EMPTY = 0
    FAILURE = -1


class AffectiveState:
    def __init__(self, dim):
        self.dim = dim
        self.last_success_vector = torch.zeros(dim)
        self.day_satisfaction = 0.0
        self.meaningfulness = 0.0
        self.paranoia_index = 0.0
        self.last_cycle_flag = CycleFlag.EMPTY
        self.wake_suffering = 0.0

    def update_paranoia(self, scars):
        if not scars:
            self.paranoia_index = 0.0
            return
        total = sum(s.significance * (1.0 - s.novelty) for s in scars)
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
        self.last_success_vector = d['last_success_vector']
        self.day_satisfaction = d['day_satisfaction']
        self.meaningfulness = d['meaningfulness']
        self.paranoia_index = d['paranoia_index']
        self.last_cycle_flag = d['last_cycle_flag']
        self.wake_suffering = d['wake_suffering']


_LOSS_FN_MAP = {
    'mse': lambda: nn.MSELoss(reduction='mean'),
    'huber': lambda: nn.HuberLoss(reduction='mean'),
}


class Critic:
    def __init__(self, dim, config: CriticConfig):
        self.dim = dim
        self.stack_size = config.stack_size
        self.replay_prob = config.replay_prob
        self.lr = config.lr
        input_dim = dim * 2 + 3  # anchor, target, paranoia, fatigue, trauma_level
        self.net = nn.Linear(input_dim, 1)
        self.loss_fn = _LOSS_FN_MAP[config.loss_fn]() if isinstance(config.loss_fn, str) else config.loss_fn
        self.stack = []
        self.recent_errors = []  # для вычисления Q

    def forward(self, anchor, target, paranoia, fatigue, trauma_level):
        feat = torch.cat([anchor, target,
                         torch.tensor([paranoia, fatigue, trauma_level], device=anchor.device)])
        return self.net(feat).squeeze(-1)

    def update(self, anchor, target, paranoia, fatigue, trauma_level, actual):
        feat = torch.cat([anchor, target,
                         torch.tensor([paranoia, fatigue, trauma_level], device=anchor.device)])
        pred = self.net(feat).squeeze()
        loss = self.loss_fn(pred, actual)
        self.net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad
                    p.grad.zero_()
        self.stack.append((feat.detach(), actual))
        if len(self.stack) > self.stack_size:
            self.stack.pop(0)
        self.recent_errors.append(abs((pred - actual).item()))
        if len(self.recent_errors) > self.stack_size:
            self.recent_errors.pop(0)
        if len(self.stack) == self.stack_size and torch.rand(1).item() < self.replay_prob:
            self._replay()

    def _replay(self):
        feats, sats = zip(*self.stack)
        feats = torch.stack(feats)
        sats = torch.tensor(sats, device=feats.device)
        preds = self.net(feats).squeeze()
        loss = self.loss_fn(preds, sats)
        self.net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                if p.grad is not None:
                    p -= self.lr * p.grad
                    p.grad.zero_()

    def quality(self):
        if not self.recent_errors:
            return 0.5
        avg_err = sum(self.recent_errors) / len(self.recent_errors)
        return math.exp(-avg_err)


def compute_will_to_disprove(past_sats, paranoia, fatigue, baseline, d1_coeff, d2_coeff):
    if not past_sats:
        return 0.0
    weights = torch.softmax(torch.arange(1, len(past_sats) + 1, dtype=torch.float32), dim=0)
    state = sum(w * s for w, s in zip(weights.tolist(), past_sats))
    d1 = d1_coeff[0] * (1.0 - paranoia) + d1_coeff[1] * fatigue
    d2 = d2_coeff[0] * (1.0 - paranoia) + d2_coeff[1] * fatigue
    will = baseline + d1 * state + 0.5 * d2 * state ** 2
    return max(0.0, min(1.0, will))


class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        core_size = config.core_size
        archive_size = config.archive_size

        total_slots = core_size
        self.memory_bank = nn.Parameter(torch.zeros(total_slots, dim))
        self.is_core = torch.zeros(total_slots, dtype=torch.bool)
        self.is_core[:config.num_basal_slots] = True
        self.sacred_weights = nn.Parameter(torch.ones(core_size))
        self.memory_last_accessed = nn.Parameter(torch.zeros(total_slots), requires_grad=False)
        self.slot_trauma = nn.Parameter(torch.zeros(core_size), requires_grad=False)

        self.archive = nn.Parameter(torch.zeros(archive_size, dim))
        self.archive_last_accessed = nn.Parameter(torch.zeros(archive_size), requires_grad=False)

        self.anchor = nn.Parameter(torch.zeros(dim))
        self.target_identity = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Linear(dim * 2, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.compressor = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))
        self.anchor_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.gap = nn.Parameter(torch.zeros(1))

        self.affective = AffectiveState(dim)
        self.scars: List[Scar] = []
        self.critic = Critic(dim, config.critic)

        self.will_baseline = config.will.baseline
        self.will_d1 = config.will.d1_coeff
        self.will_d2 = config.will.d2_coeff
        self.gap_relaxation_rate = config.gap_relaxation_rate
        self.gap_equilibrium = config.gap_equilibrium
        self.drift_threshold_base = config.drift_threshold_base
        self.interference_alpha = config.interference_alpha
        self.migration_age_factor = config.migration_age_factor
        self.archive_decay_age_seconds = config.archive_decay_age_seconds
        self.error_config = config.error_memory

        self.fast_buffer = torch.zeros(0, dim)
        self.fast_satisfaction_history: List[float] = []

        if config.num_basal_slots > 0 and config.initial_persona is not None:
            with torch.no_grad():
                self.memory_bank[:config.num_basal_slots] = config.initial_persona.unsqueeze(0) * 0.1

        self.current_time: float = self._get_current_time()

    @property
    def persona(self):
        if self.config.num_basal_slots == 0:
            return self.anchor
        basal = self.memory_bank[self.is_core]
        return basal.mean(dim=0)

    @property
    def core_memory(self):
        return self.memory_bank[self.is_core]

    @property
    def working_memory(self):
        return self.memory_bank[~self.is_core]

    def _get_current_time(self):
        if self.config.time_provider is not None:
            return self.config.time_provider()
        return time.time()

    def start_fast_cycle(self):
        self.current_time = self._get_current_time()
        self.affective.update_paranoia(self.scars)
        interference = self.affective.last_success_vector if self.affective.last_cycle_flag == CycleFlag.SUCCESS else torch.zeros_like(self.affective.last_success_vector)
        basal_out = self.persona
        B = self.slot_trauma[self.is_core].mean() + (1 - F.cosine_similarity(self.persona, self.target_identity, dim=0))
        self.affective.wake_suffering = -B * (1 - self.affective.meaningfulness)
        suffering_vec = torch.tanh(self.anchor * self.affective.wake_suffering)
        self.affective.day_satisfaction = 0.0
        self.affective.meaningfulness = 0.0
        self.fast_buffer = torch.zeros(0, self.config.dim, device=self.anchor.device)
        return self.anchor + interference + basal_out + suffering_vec

    def process_step(self, K, F, core_indices=None, archive_indices=None, timestamp=None):
        if timestamp is not None:
            self.current_time = timestamp
        else:
            self.current_time = self._get_current_time()
        if core_indices is not None:
            self.memory_last_accessed[core_indices] = self.current_time
        if archive_indices is not None:
            self.archive_last_accessed[archive_indices] = self.current_time
        output, fatigue_val, gamma_val = self.forward(K, F, core_indices, archive_indices)
        compressed = self.compress_segment(K.unsqueeze(0) if K.dim() == 1 else K)
        self.fast_buffer = torch.cat([self.fast_buffer, compressed.unsqueeze(0)], dim=0)
        trauma_mean = self.slot_trauma.mean().item()
        est_sat = self.critic.forward(self.anchor, self.target_identity,
                                      self.affective.paranoia_index, gamma_val.item(), trauma_mean)
        self.affective.day_satisfaction += est_sat.item()
        sim = F.cosine_similarity(output, self.target_identity, dim=0)
        if core_indices is not None and sim < 0.7:
            involved = core_indices.tolist()
            self.add_error(output.detach(), self.anchor.detach(), 1.0 - sim.item(), 1.0, self.current_time, involved)
            align = F.cosine_similarity(self.memory_bank[core_indices], self.target_identity.unsqueeze(0), dim=-1)
            sacred = self.sacred_weights[core_indices]
            novelty = 1 - F.cosine_similarity(output.detach().unsqueeze(0), self.memory_bank[core_indices], dim=-1)
            trauma = self.slot_trauma[core_indices]
            gain = (1 - trauma) * torch.clamp(novelty - trauma, min=0)
            self.slot_trauma[core_indices] += gain
        else:
            if core_indices is not None:
                align = F.cosine_similarity(self.memory_bank[core_indices], self.target_identity.unsqueeze(0), dim=-1)
                sacred = self.sacred_weights[core_indices]
                trauma = self.slot_trauma[core_indices]
                heal = (1 - trauma) * torch.clamp(align * sacred, min=0)
                self.slot_trauma[core_indices] *= (1 - heal)
        return output, est_sat.item()

    def forward(self, K, F, core_indices=None, archive_indices=None):
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0, device=self.anchor.device), self.gamma
        bridged, fatigue_val = self._compute_bridge(K, F)
        sacred_vec, working_vec = self._get_memory_contribs(core_indices, archive_indices)
        gate_val = self.gate(torch.cat([bridged, sacred_vec + working_vec], dim=-1))
        output = gate_val * (sacred_vec + working_vec) + (1 - gate_val) * bridged
        if torch.isnan(output).any():
            return self.anchor, fatigue_val, self.gamma
        return output, fatigue_val, self.gamma

    def _compute_bridge(self, K, F):
        J = torch.sigmoid(self.gap)
        asymptote = 1 / math.sqrt(2)
        fresh_mix = torch.clamp(J, max=asymptote)
        bridged = fresh_mix * self.theta(torch.cat([K, F], dim=-1)) + (1 - fresh_mix) * K + self.anchor
        fatigue_signal = torch.abs(J - 0.5).detach()
        return bridged, fatigue_signal

    def _get_memory_contribs(self, core_indices, archive_indices):
        sacred_vec = torch.zeros(self.config.dim, device=self.memory_bank.device)
        if core_indices is not None:
            core_slots = self.memory_bank[core_indices]
            w = F.softmax(self.sacred_weights[core_indices], dim=0)
            sacred_vec = (core_slots * w.unsqueeze(-1)).sum(dim=0)
        if archive_indices is not None:
            scale = 1.0 if core_indices is None else 0.3
            arch_vec = self.archive[archive_indices].mean(dim=0) * scale
            sacred_vec = sacred_vec + arch_vec
        working_vec = torch.zeros(self.config.dim, device=self.memory_bank.device)
        if core_indices is not None:
            working_mask = ~self.is_core
            if working_mask.any():
                working_vec = self.memory_bank[working_mask].mean(dim=0)
        else:
            working_vec = self.memory_bank[~self.is_core].mean(dim=0)
        return sacred_vec.detach(), working_vec

    def compress_segment(self, segment):
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)
        return self.compressor(segment.mean(dim=1)).squeeze(0)

    def end_fast_cycle(self):
        if self.fast_buffer.shape[0] == 0:
            self.affective.last_cycle_flag = CycleFlag.EMPTY
            return False
        X = self.fast_buffer
        S = F.cosine_similarity(X.unsqueeze(1), self.memory_bank.unsqueeze(0), dim=-1)
        temperature = 1 + self.gap.item() + self.gamma.item()
        W = F.softmax(S / temperature, dim=0)
        aggregated = (W.T @ X)
        working_mask = ~self.is_core
        if working_mask.any():
            lr = torch.sigmoid(torch.tensor(self.affective.meaningfulness)) * (1 - self.gamma)
            self.memory_bank[working_mask] = (1 - lr) * self.memory_bank[working_mask] + lr * aggregated[working_mask]
        final_sat = F.cosine_similarity(aggregated.mean(dim=0), self.target_identity, dim=0).item()
        will = compute_will_to_disprove(self.fast_satisfaction_history, self.affective.paranoia_index,
                                        self.gamma.item(), self.will_baseline, self.will_d1, self.will_d2)
        is_empty = (X.shape[0] == 1) and (X[0].sum() == 0)
        is_success = (final_sat > 0.0) and (will > 0.5) and not is_empty
        if is_empty:
            self.affective.last_cycle_flag = CycleFlag.EMPTY
        elif is_success:
            self.affective.last_cycle_flag = CycleFlag.SUCCESS
            self.affective.last_success_vector = self.anchor.clone().detach()
        else:
            self.affective.last_cycle_flag = CycleFlag.FAILURE
        trauma_mean = self.slot_trauma.mean().item()
        self.critic.update(self.anchor, self.target_identity, self.affective.paranoia_index,
                           self.gamma.item(), trauma_mean, final_sat)
        self.fast_satisfaction_history.append(final_sat)
        if len(self.fast_satisfaction_history) > 30:
            self.fast_satisfaction_history.pop(0)
        if is_success:
            self.affective.paranoia_index *= 0.9
        Q = self.critic.quality()
        raw_smooth = torch.sigmoid(torch.tensor((Q - self.affective.paranoia_index) * self.affective.meaningfulness)).item()
        update_rate = 1 - self.gamma.item()
        self._gamma_smooth = (1 - update_rate) * getattr(self, '_gamma_smooth', 0.9) + update_rate * raw_smooth
        self.affective.meaningfulness = final_sat
        self._memory_maintenance(X, W)
        self.fast_buffer = torch.zeros(0, self.config.dim, device=self.anchor.device)
        return is_success

    def _memory_maintenance(self, X, W):
        compressed = X
        idx = torch.arange(len(compressed)) % self.config.core_size
        work_mem = self.memory_bank[~self.is_core]
        if len(work_mem) > 0:
            work_mem[:len(idx)] = compressed[:len(work_mem)]
        query = self.anchor.unsqueeze(0).unsqueeze(0)
        segments = compressed.unsqueeze(0)
        attn_output, _ = self.anchor_attn(query, segments, segments)
        candidate = attn_output.squeeze(0).squeeze(0)
        self.anchor.data = 0.8 * candidate + 0.2 * self.anchor.data
        self._slow_cycle_inspection()
        self._migrate_core_to_archive()
        self._archive_decay()
        self._update_target_identity()
        self.slow_cycle_decay()
        self.slot_trauma *= (1 - self.affective.meaningfulness * (1 - self.affective.paranoia_index))

    def _slow_cycle_inspection(self):
        target = self.target_identity.data
        persona_align = torch.sigmoid(F.cosine_similarity(self.persona, target, dim=0))
        threshold = self.drift_threshold_base * (1.0 + persona_align) * (1.0 + self.gamma)
        alpha_scaled = self.interference_alpha * (1.0 - self.affective.paranoia_index)
        for i in range(self.config.core_size):
            if self.is_core[i] and i < self.config.num_basal_slots:
                continue
            if self.is_core[i]:
                vec = self.memory_bank[i]
                drift = 1.0 - F.cosine_similarity(vec.unsqueeze(0), target.unsqueeze(0))
                if drift > threshold:
                    correction = alpha_scaled * (target - vec)
                    self.memory_bank[i] += correction
                    self.sacred_weights[i] *= 0.9

    def _migrate_core_to_archive(self):
        core_indices = torch.where(self.is_core)[0]
        non_basal = core_indices[core_indices >= self.config.num_basal_slots]
        if len(non_basal) == 0:
            return
        ages = self.current_time - self.memory_last_accessed[non_basal]
        instabilities = 1.0 - self.sacred_weights[non_basal]
        traumas = self.slot_trauma[non_basal]
        scores = ages * instabilities / (1.0 + traumas)
        mean_score = scores.mean()
        threshold = mean_score * self.migration_age_factor
        migrate_mask = scores > threshold
        migrate_idx = non_basal[migrate_mask]
        for idx in migrate_idx:
            oldest_arch_idx = torch.argmax(self.current_time - self.archive_last_accessed).item()
            self.archive[oldest_arch_idx] = self.memory_bank[idx].clone()
            self.archive_last_accessed[oldest_arch_idx] = self.current_time
            self.memory_bank[idx] = self.anchor.data.clone()
            self.memory_last_accessed[idx] = self.current_time
            self.sacred_weights[idx] = 1.0
            self.is_core[idx] = False

    def _archive_decay(self):
        sink = self.anchor.data
        for i in range(self.config.archive_size):
            age = self.current_time - self.archive_last_accessed[i].item()
            if age < self.archive_decay_age_seconds:
                continue
            vec = self.archive[i]
            sims = F.cosine_similarity(vec.unsqueeze(0), self.archive, dim=-1)
            sims[i] = -1.0
            nearest = torch.argmax(sims).item()
            transfer_rate = 0.1
            decayed = (1 - transfer_rate) * vec + transfer_rate * sink
            self.archive[nearest] += vec - decayed
            self.archive[i] = decayed
            self.archive_last_accessed[i] = self.current_time

    def _update_target_identity(self, momentum=0.995):
        self.target_identity.data = momentum * self.target_identity.data + (1 - momentum) * self.anchor.data

    def update_gamma(self, error):
        smooth = getattr(self, '_gamma_smooth', 0.9)
        with torch.no_grad():
            self.gamma.data = smooth * self.gamma.data + (1 - smooth) * error

    def add_error(self, vector, color, significance, novelty, timestamp, involved_core_indices):
        if len(self.scars) >= self.error_config.max_scars:
            self.scars.sort(key=lambda s: s.significance)
            self.scars.pop(0)
        self.scars.append(Scar(vector.clone().detach(), color.clone().detach(),
                               significance, novelty, timestamp, involved_core_indices))

    def slow_cycle_decay(self):
        for scar in self.scars:
            if self.current_time - scar.last_activated > self.error_config.slow_cycle_seconds:
                scar.novelty = max(0.0, scar.novelty - self.error_config.novelty_step)
                scar.significance *= self.error_config.decay_rate
        self._merge_similar_scars()

    def _merge_similar_scars(self):
        merged = []
        used = [False] * len(self.scars)
        for i, s1 in enumerate(self.scars):
            if used[i]:
                continue
            group = [s1]
            for j in range(i + 1, len(self.scars)):
                if used[j]:
                    continue
                sim = F.cosine_similarity(s1.color.unsqueeze(0), self.scars[j].color.unsqueeze(0))
                if sim > self.error_config.similarity_threshold:
                    group.append(self.scars[j])
                    used[j] = True
            if len(group) > 1:
                avg_vec = sum(g.vector for g in group) / len(group)
                avg_color = sum(g.color for g in group) / len(group)
                sig = max(g.significance for g in group)
                nov = min(g.novelty for g in group)
                last_act = max(g.last_activated for g in group)
                indices = list(set(i for g in group for i in g.involved_core_indices))
                merged.append(Scar(avg_vec, avg_color, sig, nov, last_act, indices))
            else:
                merged.append(s1)
            used[i] = True
        self.scars = merged

    def set_persona(self, persona_vector):
        with torch.no_grad():
            if self.config.num_basal_slots > 0:
                self.memory_bank[:self.config.num_basal_slots] = persona_vector.to(self.memory_bank.device).unsqueeze(0)

    def warmup(self, num_steps=None):
        if num_steps is None:
            num_steps = self.config.warmup_steps
        if self.config.num_basal_slots == 0:
            return
        basal = self.memory_bank[self.is_core]
        for _ in range(num_steps):
            K = basal.mean(dim=0).detach()
            F = self.anchor.detach()
            self.forward(K, F, core_indices=None, archive_indices=None)

    def save_checkpoint(self, path):
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'affective_state': self.affective.state_dict(),
            'scars': [(s.vector, s.color, s.significance, s.novelty, s.last_activated,
                       s.involved_core_indices.copy()) for s in self.scars],
            'fast_satisfaction_history': self.fast_satisfaction_history.copy(),
            'current_time': self.current_time,
            'checkpoint_timestamp': self._get_current_time(),
            'gamma_smooth': getattr(self, '_gamma_smooth', 0.9),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, map_location='cpu', time_provider=None, resume_time=None):
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['config']
        if time_provider is not None:
            config.time_provider = time_provider
        agent = cls(config)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.affective.load_state_dict(checkpoint['affective_state'])
        agent.scars = [Scar(v.clone() if isinstance(v, torch.Tensor) else v,
                            c.clone() if isinstance(c, torch.Tensor) else c,
                            s, n, la, idx.copy() if isinstance(idx, list) else idx)
                       for (v, c, s, n, la, idx) in checkpoint['scars']]
        agent.fast_satisfaction_history = checkpoint['fast_satisfaction_history']
        agent.current_time = checkpoint['current_time']
        agent._gamma_smooth = checkpoint.get('gamma_smooth', 0.9)
        current_time = resume_time if resume_time is not None else agent._get_current_time()
        delta_t = current_time - checkpoint['checkpoint_timestamp']
        agent._apply_temporal_corrections(delta_t)
        agent.warmup()
        return agent

    def _apply_temporal_corrections(self, delta_t):
        with torch.no_grad():
            decay_factor = (1 - self._gamma_smooth) * delta_t
            self.gamma.data = self.gamma.data * (1 - decay_factor)
            dg = -self.gap_relaxation_rate * (self.gap.data - self.gap_equilibrium) * delta_t
            self.gap.data += dg

    def auto_save_hook(self, fatigue_threshold=0.8, save_dir='./checkpoints'):
        if self.gamma.item() >= fatigue_threshold:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'zazor_checkpoint_{int(self._get_current_time())}.pt')
            self.save_checkpoint(path)
