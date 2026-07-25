import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable, Protocol, runtime_checkable
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


@runtime_checkable
class BodyBuffer(Protocol):
    def store_anchor(self, anchor: torch.Tensor, timestamp: float) -> None: ...
    def get_anchor_history(self) -> torch.Tensor: ...
    def get_anchor_snapshot(self, window: int = 1) -> torch.Tensor: ...
    def clear(self) -> None: ...


class DummyBodyBuffer:
    def __init__(self, dim: int, maxlen: int = 100):
        self.dim = dim
        self.buffer = torch.zeros(0, dim)
        self.maxlen = maxlen

    def store_anchor(self, anchor: torch.Tensor, timestamp: float):
        if anchor.dim() == 0:
            return
        self.buffer = torch.cat([self.buffer, anchor.detach().cpu().unsqueeze(0)])[-self.maxlen:]

    def get_anchor_history(self):
        return self.buffer

    def get_anchor_snapshot(self, window: int = 1):
        if self.buffer.shape[0] == 0:
            return torch.zeros(self.dim)
        return self.buffer[-window:].mean(dim=0) if window > 1 else self.buffer[-1].clone()

    def clear(self):
        self.buffer = torch.zeros(0, self.dim)


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
        input_dim = dim * 2 + 3
        self.net = nn.Linear(input_dim, 1)
        self.loss_fn = _LOSS_FN_MAP[config.loss_fn]() if isinstance(config.loss_fn, str) else config.loss_fn
        self.stack = []
        self.recent_errors = []

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


class MetaModulator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Sequential(nn.Linear(4, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, 9 * dim)
        self.register_buffer('ema_params', torch.zeros(9, dim))
        self.ema_params[1] = 0.9
        self.ema_params[4] = 1/math.sqrt(2)
        self.ema_params[5] = 1.0
        self.ema_params[7] = 0.1
        self.ema_params[8] = 0.995
        self.ema_alpha = nn.Parameter(torch.tensor(0.9))

    def forward(self, trauma_mean, meaningfulness, gap, gamma):
        ctx = torch.tensor([trauma_mean, meaningfulness, gap.item(), gamma.item()],
                           device=self.proj.weight.device)
        latent = self.encoder(ctx)
        raw = self.proj(latent).view(9, self.dim)
        raw[1:5] = torch.sigmoid(raw[1:5])
        raw[5] = F.softplus(raw[5]) + 0.5
        raw[6:9] = torch.sigmoid(raw[6:9])
        alpha = torch.sigmoid(self.ema_alpha)
        self.ema_params = alpha * self.ema_params + (1 - alpha) * raw
        return self.ema_params


class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig, body_buffer: Optional[BodyBuffer] = None):
        super().__init__()
        self.config = config
        dim, core, arch = config.dim, config.core_size, config.archive_size
        self.dim = dim
        self.core_size = core
        self.memory = nn.Parameter(torch.zeros(core, dim))
        self.is_core = torch.zeros(core, dtype=torch.bool)
        self.is_core[:config.num_basal_slots] = True
        self.sacred = nn.Parameter(torch.ones(core))
        self.last_access = nn.Parameter(torch.zeros(core), requires_grad=False)
        self.trauma = nn.Parameter(torch.zeros(core), requires_grad=False)
        self.archive = nn.Parameter(torch.zeros(arch, dim))
        self.arch_access = nn.Parameter(torch.zeros(arch), requires_grad=False)
        self.anchor = nn.Parameter(torch.zeros(dim))
        self.target = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Linear(dim*2, dim)
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.compressor = nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU(), nn.Linear(dim//2, dim))
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.gap = nn.Parameter(torch.zeros(1))
        self.affect = AffectiveState(dim)
        self.scars: List[Scar] = []
        self.critic = Critic(dim, config.critic)
        self.will_b = config.will.baseline
        self.will_d1 = config.will.d1_coeff
        self.will_d2 = config.will.d2_coeff
        self.gap_lr = config.gap_relaxation_rate
        self.gap_eq = config.gap_equilibrium
        self.drift_base = config.drift_threshold_base
        self.int_alpha = config.interference_alpha
        self.mig_factor = config.migration_age_factor
        self.arch_age = config.archive_decay_age_seconds
        self.err_cfg = config.error_memory
        self.body = body_buffer or DummyBodyBuffer(dim)
        self.meta = MetaModulator(dim)
        self.wake_dist = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.fast_buf = torch.zeros(0, dim)
        self.sat_hist: List[float] = []
        if config.num_basal_slots and config.initial_persona is not None:
            with torch.no_grad():
                self.memory[:config.num_basal_slots] = config.initial_persona.unsqueeze(0) * 0.1
        self.now = self._time()

    def _time(self):
        return self.config.time_provider() if self.config.time_provider else time.time()

    @property
    def persona(self):
        return self.memory[self.is_core].mean(0) if self.is_core.any() else self.anchor

    @property
    def working(self):
        return self.memory[~self.is_core]

    def _meta_params(self):
        t = self.trauma[self.is_core].mean().item()
        return self.meta(t, self.affect.meaningfulness, self.gap, self.gamma)

    def start_fast_cycle(self):
        self.now = self._time()
        self.affect.update_paranoia(self.scars)
        inter = self.affect.last_success_vector if self.affect.last_cycle_flag == CycleFlag.SUCCESS else torch.zeros(self.dim)
        base = self.persona
        mp = self._meta_params()
        B = self.trauma[self.is_core].mean() + (1 - F.cosine_similarity(self.persona, self.target, dim=0))
        self.affect.wake_suffering = -B * (1 - self.affect.meaningfulness) * torch.sigmoid(torch.dot(mp[0], self.anchor)/(self.anchor.norm()+1e-8)).item()
        suf = torch.tanh(self.anchor * self.affect.wake_suffering)
        self.affect.day_satisfaction = 0.0
        self.affect.meaningfulness = 0.0
        self.fast_buf = torch.zeros(0, self.dim, device=self.anchor.device)
        return self.anchor + inter + base + suf + self.wake_dist

    def process_step(self, K, F, core_idx=None, arch_idx=None, ts=None):
        if ts: self.now = ts
        else: self.now = self._time()
        if core_idx is not None: self.last_access[core_idx] = self.now
        if arch_idx is not None: self.arch_access[arch_idx] = self.now
        out, fat, gval = self.forward(K, F, core_idx, arch_idx)
        comp = self.compressor(K.unsqueeze(0).mean(1)).squeeze(0) if K.dim()>1 else self.compressor(K.unsqueeze(0).unsqueeze(0)).squeeze(0)
        self.fast_buf = torch.cat([self.fast_buf, comp.unsqueeze(0)])
        sat = self.critic.forward(self.anchor, self.target, self.affect.paranoia_index, gval.item(), self.trauma.mean().item())
        self.affect.day_satisfaction += sat.item()
        sim = F.cosine_similarity(out, self.target, dim=0)
        mp = self._meta_params()
        if core_idx is not None and sim < 0.7:
            self._add_error(out.detach(), self.anchor.detach(), 1-sim.item(), 1.0, self.now, core_idx.tolist())
            nov = 1 - F.cosine_similarity(out.detach().unsqueeze(0), self.memory[core_idx])
            old_t = self.trauma[core_idx]
            gain = (1-old_t) * torch.clamp(nov - old_t, min=0)
            gain = gain * torch.sigmoid(torch.mv(self.memory[core_idx], mp[3]))
            self.trauma[core_idx] += gain
        elif core_idx is not None:
            align = F.cosine_similarity(self.memory[core_idx], self.target.unsqueeze(0))
            sacred = self.sacred[core_idx]
            old_t = self.trauma[core_idx]
            heal = (1-old_t) * torch.clamp(align * sacred, min=0)
            heal = heal * torch.sigmoid(torch.mv(self.memory[core_idx], mp[2]))
            self.trauma[core_idx] *= (1 - heal)
        self.wake_dist = self.wake_dist * 0.9
        return out, sat.item()

    def forward(self, K, F, core_idx=None, arch_idx=None):
        if K is None or torch.isnan(K).any():
            return self.anchor, torch.tensor(0.0, device=self.anchor.device), self.gamma
        bridged, fat = self._bridge(K, F)
        sac, wrk = self._contribs(core_idx, arch_idx)
        gv = self.gate(torch.cat([bridged, sac+wrk]))
        out = gv*sac + gv*wrk + (1-gv)*bridged
        return (self.anchor, fat, self.gamma) if torch.isnan(out).any() else (out, fat, self.gamma)

    def _bridge(self, K, F):
        J = torch.sigmoid(self.gap)
        mp = self._meta_params()
        lim = torch.sigmoid(torch.dot(mp[4], self.anchor)/(self.anchor.norm()+1e-8))
        fresh = torch.clamp(J, max=lim)
        b = fresh*self.theta(torch.cat([K,F])) + (1-fresh)*K + self.anchor
        return b, torch.abs(J-0.5).detach()

    def _contribs(self, core_idx, arch_idx):
        sac = torch.zeros(self.dim, device=self.memory.device)
        if core_idx is not None:
            w = F.softmax(self.sacred[core_idx], dim=0)
            sac = (self.memory[core_idx]*w.unsqueeze(-1)).sum(0)
        if arch_idx is not None:
            scale = 1.0 if core_idx is None else 0.3
            sac = sac + self.archive[arch_idx].mean(0)*scale
        wrk = self.memory[~self.is_core].mean(0) if (~self.is_core).any() else torch.zeros(self.dim, device=self.memory.device)
        return sac.detach(), wrk

    def end_fast_cycle(self):
        if self.fast_buf.shape[0]==0:
            self.affect.last_cycle_flag = CycleFlag.EMPTY
            return False
        X = self.fast_buf
        S = F.cosine_similarity(X.unsqueeze(1), self.memory.unsqueeze(0), dim=-1)
        mp = self._meta_params()
        attn_temp = torch.dot(mp[5], self.anchor)/(self.anchor.norm()+1e-8)
        temp = 1 + self.gap.item() + self.gamma.item() + F.softplus(torch.tensor(attn_temp)).item()
        W = F.softmax(S/temp, dim=0)
        agg = W.T @ X
        wmask = ~self.is_core
        if wmask.any():
            lr = torch.sigmoid(torch.tensor(self.affect.meaningfulness))*(1-self.gamma)
            self.memory[wmask] = (1-lr)*self.memory[wmask] + lr*agg[wmask]
        final_sat = F.cosine_similarity(agg.mean(0), self.target, dim=0).item()
        will = compute_will_to_disprove(self.sat_hist, self.affect.paranoia_index, self.gamma.item(), self.will_b, self.will_d1, self.will_d2)
        empty = (X.shape[0]==1 and X[0].sum()==0)
        success = final_sat>0 and will>0.5 and not empty
        self.affect.last_cycle_flag = CycleFlag.EMPTY if empty else (CycleFlag.SUCCESS if success else CycleFlag.FAILURE)
        if success: self.affect.last_success_vector = self.anchor.clone().detach()
        self.critic.update(self.anchor, self.target, self.affect.paranoia_index, self.gamma.item(), self.trauma.mean().item(), final_sat)
        self.sat_hist.append(final_sat)
        if len(self.sat_hist)>30: self.sat_hist.pop(0)
        if success: self.affect.paranoia_index *= 0.9
        Q = self.critic.quality()
        raw_smooth = torch.sigmoid(torch.tensor((Q-self.affect.paranoia_index)*self.affect.meaningfulness)).item()
        upd = 1-self.gamma.item()
        self._gamma_smooth = (1-upd)*getattr(self,'_gamma_smooth',0.9) + upd*raw_smooth
        self.affect.meaningfulness = final_sat
        self._maintenance(X)
        self.body.store_anchor(self.anchor, self.now)
        self.fast_buf = torch.zeros(0, self.dim, device=self.anchor.device)
        return success

    def _maintenance(self, X):
        q = self.anchor.unsqueeze(0).unsqueeze(0)
        s = X.unsqueeze(0)
        out, _ = self.attn(q, s, s)
        cand = out.squeeze(0).squeeze(0)
        mp = self._meta_params()
        mom = torch.sigmoid(torch.dot(mp[8], self.anchor)/(self.anchor.norm()+1e-8))
        self.anchor.data = mom*self.anchor.data + (1-mom)*cand
        self._inspect()
        self._migrate()
        self._decay_archive()
        self._update_target()
        self._scar_decay()
        self.trauma *= (1 - self.affect.meaningfulness*(1-self.affect.paranoia_index))

    def _inspect(self):
        target = self.target.data
        pa = torch.sigmoid(F.cosine_similarity(self.persona, target, dim=0))
        thresh = self.drift_base*(1+pa)*(1+self.gamma)
        alpha = self.int_alpha*(1-self.affect.paranoia_index)
        core_idx = self.is_core.nonzero(as_tuple=True)[0]
        non_basal = core_idx[core_idx>=self.config.num_basal_slots]
        if len(non_basal)==0: return
        vecs = self.memory[non_basal]
        drift = 1 - F.cosine_similarity(vecs, target.unsqueeze(0))
        mask = drift > thresh
        if mask.any():
            idx = non_basal[mask]
            correction = alpha*(target.unsqueeze(0) - self.memory[idx])
            self.memory[idx] += correction
            self.sacred[idx] *= 0.9

    def _migrate(self):
        core_idx = self.is_core.nonzero(as_tuple=True)[0]
        non_basal = core_idx[core_idx>=self.config.num_basal_slots]
        if len(non_basal)==0: return
        ages = self.now - self.last_access[non_basal]
        instab = 1 - self.sacred[non_basal]
        trau = self.trauma[non_basal]
        scores = ages * instab / (1+trau)
        mp = self._meta_params()
        mf = torch.sigmoid(torch.dot(mp[6], self.anchor)/(self.anchor.norm()+1e-8))
        thresh = scores.mean()*(self.mig_factor*(0.5+mf))
        mig = non_basal[scores>thresh]
        if len(mig)==0: return
        oldest_idx = torch.argmax(self.now - self.arch_access)
        self.archive[oldest_idx] = self.memory[mig[-1]].clone()
        self.arch_access[oldest_idx] = self.now
        self.memory[mig] = self.anchor.data.clone()
        self.last_access[mig] = self.now
        self.sacred[mig] = 1.0
        self.is_core[mig] = False

    def _decay_archive(self):
        mp = self._meta_params()
        rate = torch.sigmoid(torch.dot(mp[7], self.anchor)/(self.anchor.norm()+1e-8))
        ages = self.now - self.arch_access
        mask = ages >= self.arch_age
        if not mask.any(): return
        vecs = self.archive[mask]
        sim_mat = F.cosine_similarity(vecs.unsqueeze(1), self.archive.unsqueeze(0), dim=-1)
        sim_mat[torch.arange(len(vecs)), mask.nonzero(as_tuple=True)[0]] = -1
        nearest = sim_mat.argmax(dim=1)
        self.archive[nearest] += vecs - (1-rate)*vecs - rate*self.anchor.data
        self.archive[mask] = (1-rate)*vecs + rate*self.anchor.data
        self.arch_access[mask] = self.now

    def _update_target(self):
        mp = self._meta_params()
        mom = torch.sigmoid(torch.dot(mp[8], self.anchor)/(self.anchor.norm()+1e-8))
        self.target.data = mom*self.target.data + (1-mom)*self.anchor.data

    def update_gamma(self, error):
        smooth = getattr(self, '_gamma_smooth', 0.9)
        with torch.no_grad():
            self.gamma.data = smooth*self.gamma.data + (1-smooth)*error

    def _add_error(self, vec, col, sig, nov, ts, idx):
        if len(self.scars) >= self.err_cfg.max_scars:
            self.scars.sort(key=lambda s: s.significance)
            self.scars.pop(0)
        self.scars.append(Scar(vec.clone(), col.clone(), sig, nov, ts, idx))

    def _scar_decay(self):
        for s in self.scars:
            if self.now - s.last_activated > self.err_cfg.slow_cycle_seconds:
                s.novelty = max(0, s.novelty-self.err_cfg.novelty_step)
                s.significance *= self.err_cfg.decay_rate
        self._merge_scars()

    def _merge_scars(self):
        if len(self.scars)<2: return
        cols = torch.stack([s.color for s in self.scars])
        sim = F.cosine_similarity(cols.unsqueeze(1), cols.unsqueeze(0), dim=-1)
        sim.fill_diagonal_(0)
        above = sim > self.err_cfg.similarity_threshold
        merged = []
        used = set()
        for i, s in enumerate(self.scars):
            if i in used: continue
            group = [s]
            for j in range(i+1, len(self.scars)):
                if j not in used and above[i,j]:
                    group.append(self.scars[j])
                    used.add(j)
            if len(group)>1:
                avg_v = sum(g.vector for g in group)/len(group)
                avg_c = sum(g.color for g in group)/len(group)
                sig = max(g.significance for g in group)
                nov = min(g.novelty for g in group)
                last = max(g.last_activated for g in group)
                idx = list(set(i for g in group for i in g.involved_core_indices))
                merged.append(Scar(avg_v, avg_c, sig, nov, last, idx))
            else:
                merged.append(s)
        self.scars = merged

    def set_persona(self, vec):
        if self.config.num_basal_slots:
            with torch.no_grad():
                self.memory[:self.config.num_basal_slots] = vec.to(self.memory.device).unsqueeze(0)

    def warmup(self, steps=None):
        steps = steps or self.config.warmup_steps
        if not self.config.num_basal_slots: return
        basal = self.memory[self.is_core].mean(0)
        for _ in range(steps):
            self.forward(basal, self.anchor)
            self.wake_dist = self.wake_dist * 0.9

    def save_checkpoint(self, path):
        ck = {
            'config': self.config,
            'state': self.state_dict(),
            'affect': self.affect.state_dict(),
            'scars': [(s.vector, s.color, s.significance, s.novelty, s.last_activated, s.involved_core_indices.copy()) for s in self.scars],
            'sat_hist': self.sat_hist.copy(),
            'now': self.now,
            'ts': self._time(),
            'g_smooth': getattr(self, '_gamma_smooth', 0.9),
            'body': self.body.get_anchor_history().clone() if hasattr(self.body, 'get_anchor_history') else None,
        }
        torch.save(ck, path)

    @classmethod
    def load_checkpoint(cls, path, map='cpu', time_provider=None, resume_time=None, body_buffer=None):
        ck = torch.load(path, map_location=map)
        cfg = ck['config']
        if time_provider: cfg.time_provider = time_provider
        agent = cls(cfg, body_buffer=body_buffer)
        agent.load_state_dict(ck['state'])
        agent.affect.load_state_dict(ck['affect'])
        agent.scars = [Scar(v.clone() if isinstance(v,torch.Tensor) else v, c.clone() if isinstance(c,torch.Tensor) else c, s, n, la, idx.copy() if isinstance(idx,list) else idx) for (v,c,s,n,la,idx) in ck['scars']]
        agent.sat_hist = ck['sat_hist']
        agent.now = ck['now']
        agent._gamma_smooth = ck.get('g_smooth', 0.9)
        if body_buffer is None and ck.get('body') is not None:
            agent.body.buffer = ck['body'].clone()
        cur = resume_time or agent._time()
        dt = cur - ck['ts']
        agent._temporal_fix(dt)
        agent.wake_dist = -dt * agent.anchor.detach() * 0.1
        agent.warmup()
        return agent

    def _temporal_fix(self, dt):
        with torch.no_grad():
            decay = (1 - self._gamma_smooth) * dt
            self.gamma.data *= (1 - decay)
            self.gap.data += -self.gap_lr * (self.gap.data - self.gap_eq) * dt

    def auto_save(self, thresh=0.8, dir='./ckpt'):
        if self.gamma.item() >= thresh:
            os.makedirs(dir, exist_ok=True)
            self.save_checkpoint(os.path.join(dir, f'zazor_{int(self._time())}.pt'))
