import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from dataclasses import dataclass, field
import math

@dataclass
class AgeConfig:
    delta_t: float = 0.01
    beta_young: float = 0.1
    lambda_penalty: float = 0.05
    gamma_penalty_scale: float = 0.1

@dataclass
class ErrorSheafConfig:
    max_scars: int = 512
    similarity_threshold: float = 0.9
    attraction_eps: float = 0.01
    decay_significance: float = 0.99
    novelty_lr: float = 0.1

@dataclass
class CohomologyConfig:
    alpha_drift: float = 0.25
    alpha_trauma: float = 0.25
    alpha_variance: float = 0.25
    alpha_gap: float = 0.25
    gamma_temperature: float = 0.1
    alpha_delta_mod: float = 0.5

@dataclass
class MixerConfig:
    hidden_dim: int = 64
    num_layers: int = 2

@dataclass
class ZazorConfig:
    dim: int
    core_size: int = 16
    archive_size: int = 32
    num_basal_slots: int = 4
    initial_persona: Optional[torch.Tensor] = None
    age: AgeConfig = field(default_factory=AgeConfig)
    error_sheaf: ErrorSheafConfig = field(default_factory=ErrorSheafConfig)
    cohomology: CohomologyConfig = field(default_factory=CohomologyConfig)
    mixer: MixerConfig = field(default_factory=MixerConfig)
    time_provider: Optional[Callable[[], float]] = None
    warmup_steps: int = 5

def compute_context(K, F, anchor, gap, theta):
    bridged = torch.sigmoid(gap) * theta(torch.cat([K, F])) + (1 - torch.sigmoid(gap)) * K + anchor
    return bridged

def boundary_1(ctx, C1, is_basal, temperature, basal_bonus, gamma, paranoia, ages):
    logits = (ctx @ C1.T) / temperature
    crisis_bonus = is_basal.float() * basal_bonus * ((1 - gamma) + paranoia)
    attn = torch.softmax(logits + crisis_bonus, dim=-1)
    mem_contrib = attn @ C1
    freshness = torch.exp(-ages * (1 + gamma))
    return mem_contrib, attn, freshness

def gate_mix(ctx, mem_contrib, gate_W, gamma, paranoia, avg_sacred, novelty):
    confidence_mem = avg_sacred * (1 - paranoia) * gamma
    confidence_ctx = (1 - gamma) * (1 + paranoia) * novelty
    logit_mod = torch.log(confidence_mem / (confidence_ctx + 1e-8))
    raw = gate_W(torch.cat([ctx, mem_contrib]))
    gate = torch.sigmoid(raw + logit_mod)
    return gate * mem_contrib + (1 - gate) * ctx

def update_C1(C1, C0, attn, lr, trust, is_basal, basal_lr):
    sim = F.cosine_similarity(C0.unsqueeze(0), C1, dim=-1)
    lr_effective = torch.where(is_basal, basal_lr * trust, lr * trust)
    delta = lr_effective.unsqueeze(-1) * attn.unsqueeze(-1) * (C0.unsqueeze(0) - C1) * sim.unsqueeze(-1)
    return C1 + delta

def update_C2(C2, C1, ages, trauma, sacred_eff, mig_thresh, ages_arch, C0):
    priority = ages / (1 + sacred_eff + 1e-8)
    p_mig = torch.sigmoid(priority - mig_thresh)
    w_arch = torch.softmax(ages_arch, dim=0)
    target_arch = w_arch @ C2
    delta_C2 = p_mig.unsqueeze(-1) * (C1 - target_arch.unsqueeze(0))  # (core, d)
    C2 = C2 + w_arch.unsqueeze(-1) * delta_C2.mean(dim=0)  # упрощённо
    C1 = C1 * (1 - p_mig.unsqueeze(-1)) + p_mig.unsqueeze(-1) * C0.unsqueeze(0)
    return C1, C2

def update_trauma(trauma, novelty, heal, gain_scale, heal_scale):
    gain = (1 - trauma) * F.relu(novelty - trauma) * gain_scale
    heal = (1 - trauma) * F.relu(heal) * heal_scale
    return trauma + gain - heal * trauma

def update_ages(ages, attn, sim, gamma, cfg):
    aging = cfg.delta_t * (1 + cfg.gamma_penalty_scale * (1 - gamma))
    rejuvenation = cfg.beta_young * sim * (1 + gamma)
    penalty = cfg.lambda_penalty * (1 - attn) * gamma
    ages = ages + aging - rejuvenation + penalty
    return torch.clamp(ages, min=0.0)

def update_sacred(mu, sigma2, attn, TD_error, trust, bias):
    precision = 1.0 / (sigma2 + 1e-8)
    attn_abs = attn.abs()
    new_precision = precision + trust * attn_abs
    new_mu = (precision * mu + trust * attn_abs * TD_error) / (new_precision + 1e-8)
    new_sigma2 = 1.0 / (new_precision + 1e-8)
    effective = new_mu + bias
    return new_mu, new_sigma2, effective

def update_error_sheaf(S, sig, nov, anchor, cfg):
    sim = F.cosine_similarity(anchor.unsqueeze(0), S, dim=-1)
    nov = nov - cfg.novelty_lr * (1 - sim)
    sig = sig * cfg.decay_significance
    S_sim = S @ S.T
    mask = (S_sim > cfg.similarity_threshold).float() - torch.eye(S.size(0), device=S.device)
    attraction = cfg.attraction_eps * (mask @ S - S * mask.sum(dim=1, keepdim=True))
    S = S + attraction
    return S, sig, nov

def compute_gamma(drift_tension, trauma_mean, gap, avg_variance, critic_error, avg_sacred, cfg):
    mod = cfg.alpha_delta_mod / (1 + avg_sacred)
    raw = 1.0 - mod * critic_error - cfg.alpha_drift * drift_tension - cfg.alpha_trauma * trauma_mean \
          - cfg.alpha_variance * avg_variance - cfg.alpha_gap * gap
    return torch.sigmoid(raw / cfg.gamma_temperature)

class Mixer(nn.Module):
    def __init__(self, dim, core_size, archive_size, cfg):
        super().__init__()
        self.dim = dim
        self.core_size = core_size
        self.archive_size = archive_size
        input_dim = 7
        layers = []
        prev = input_dim
        for _ in range(cfg.num_layers):
            layers.append(nn.Linear(prev, cfg.hidden_dim))
            layers.append(nn.ReLU())
            prev = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_lr_C1 = nn.Linear(prev, core_size)
        self.head_temperature = nn.Linear(prev, 1)
        self.head_mom_a = nn.Linear(prev, 1)
        self.head_mom_t = nn.Linear(prev, 1)
        self.head_mig_thresh = nn.Linear(prev, 1)
        self.head_gain_scale = nn.Linear(prev, dim)
        self.head_heal_scale = nn.Linear(prev, dim)
        self.head_basal_bonus = nn.Linear(prev, 1)
        self.head_basal_lr = nn.Linear(prev, 1)

    def forward(self, cohomology):
        h = self.backbone(cohomology)
        return {
            'lr_C1': torch.sigmoid(self.head_lr_C1(h)) * 0.1,
            'temperature': F.softplus(self.head_temperature(h)) + 0.1,
            'mom_a': torch.sigmoid(self.head_mom_a(h)),
            'mom_t': torch.sigmoid(self.head_mom_t(h)),
            'mig_thresh': torch.sigmoid(self.head_mig_thresh(h)),
            'gain_scale': torch.sigmoid(self.head_gain_scale(h)),
            'heal_scale': torch.sigmoid(self.head_heal_scale(h)),
            'basal_bonus': F.softplus(self.head_basal_bonus(h)),
            'basal_lr': torch.sigmoid(self.head_basal_lr(h)) * 0.01
        }

class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
    def forward(self, anchor, target, C0):
        return self.net(torch.cat([anchor, target, C0])).squeeze(-1)

class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig):
        super().__init__()
        self.cfg = config
        d, c, a = config.dim, config.core_size, config.archive_size

        self.C1 = nn.Parameter(torch.zeros(c, d))
        self.C2 = nn.Parameter(torch.zeros(a, d))
        self.anchor = nn.Parameter(torch.zeros(d))
        self.target = nn.Parameter(torch.zeros(d))
        self.gap = nn.Parameter(torch.zeros(1))
        self.trauma = nn.Parameter(torch.zeros(c))
        self.ages = nn.Parameter(torch.zeros(c))
        self.ages_arch = nn.Parameter(torch.zeros(a))
        self.sacred_mu = nn.Parameter(torch.zeros(c))
        self.sacred_sigma2 = nn.Parameter(torch.ones(c))
        self.gamma = nn.Parameter(torch.tensor(0.5))

        self.register_buffer('is_basal', torch.zeros(c, dtype=torch.bool))
        self.is_basal[:config.num_basal_slots] = True
        if config.initial_persona is not None:
            with torch.no_grad():
                self.C1[:config.num_basal_slots] = config.initial_persona.unsqueeze(0)

        self.theta = nn.Linear(2 * d, d)
        self.gate_W = nn.Linear(2 * d, d)
        self.mixer = Mixer(d, c, a, config.mixer)
        self.critic = Critic(d)

        self.register_buffer('S', torch.zeros(0, d))
        self.register_buffer('sig', torch.zeros(0))
        self.register_buffer('nov', torch.zeros(0))

    def persona(self):
        return self.C1.mean(dim=0)

    def bias_from_scars(self):
        if self.S.size(0) == 0:
            return torch.zeros(self.C1.size(0), device=self.C1.device)
        sim = F.cosine_similarity(self.S.unsqueeze(1), self.C1.unsqueeze(0), dim=-1)
        bias = (self.sig.unsqueeze(-1) * (1 - self.nov.unsqueeze(-1)) * sim).sum(dim=0)
        return bias * 0.1

    def forward(self, K: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.cfg.dim
        ctx = compute_context(K, F, self.anchor, self.gap, self.theta)

        drift_tension = (1 - F.cosine_similarity(self.anchor, self.target, dim=0)) * \
                        (1 - F.cosine_similarity(self.anchor, self.persona(), dim=0)) * \
                        (1 - F.cosine_similarity(self.persona(), self.target, dim=0))
        trauma_mean = self.trauma.mean()
        avg_variance = self.sacred_sigma2.mean()
        if self.S.size(0) > 0:
            paranoia = (self.sig * (1 - self.nov)).sum() / self.S.size(0)
        else:
            paranoia = torch.tensor(0.0)
        critic_error = torch.tensor(0.0)
        avg_sacred_eff = (self.sacred_mu + self.bias_from_scars()).mean()
        cohom_vec = torch.stack([
            drift_tension, trauma_mean, self.gap.squeeze(), avg_variance,
            critic_error, avg_sacred_eff, paranoia
        ])
        gamma = compute_gamma(drift_tension, trauma_mean, self.gap.squeeze(), avg_variance,
                              critic_error, avg_sacred_eff, self.cfg.cohomology)
        self.gamma.data = gamma

        params = self.mixer(cohom_vec.detach())

        mem_contrib, attn, freshness = boundary_1(ctx, self.C1, self.is_basal,
                                                   params['temperature'], params['basal_bonus'],
                                                   gamma, paranoia, self.ages)

        novelty = 1 - F.cosine_similarity(ctx, mem_contrib, dim=0)
        C0 = gate_mix(ctx, mem_contrib, self.gate_W, gamma, paranoia, avg_sacred_eff, novelty)

        trust = gamma * (1 - paranoia)
        C1_new = update_C1(self.C1, C0, attn, params['lr_C1'], trust, self.is_basal, params['basal_lr'])

        bias = self.bias_from_scars()
        sacred_eff = self.sacred_mu + bias
        C1_new, C2_new = update_C2(self.C2, C1_new, self.ages, self.trauma, sacred_eff,
                                   params['mig_thresh'], self.ages_arch, C0)

        sim = F.cosine_similarity(C0.unsqueeze(0), self.C1, dim=-1)
        novelty_vec = 1 - sim
        heal_vec = F.cosine_similarity(self.C1, self.target.unsqueeze(0), dim=-1)
        trauma_new = update_trauma(self.trauma, novelty_vec, heal_vec,
                                   params['gain_scale'], params['heal_scale'])

        ages_new = update_ages(self.ages, attn, sim, gamma, self.cfg.age)
        ages_arch_new = self.ages_arch + self.cfg.age.delta_t

        V_pred = self.critic(self.anchor, self.target, C0)
        S_true = F.cosine_similarity(C0, self.target, dim=0)
        TD_error = S_true - V_pred.detach()
        mu_new, sigma2_new, sacred_eff_new = update_sacred(self.sacred_mu, self.sacred_sigma2, attn,
                                                           TD_error, trust, bias)

        anchor_new = params['mom_a'] * self.anchor + (1 - params['mom_a']) * C0
        target_new = params['mom_t'] * self.target + (1 - params['mom_t']) * anchor_new

        S_new, sig_new, nov_new = update_error_sheaf(self.S, self.sig, self.nov, self.anchor,
                                                     self.cfg.error_sheaf)
        if S_true < 0.7:
            S_new = torch.cat([S_new, C0.detach().unsqueeze(0)])
            sig_new = torch.cat([sig_new, (1 - S_true).unsqueeze(0)])
            nov_new = torch.cat([nov_new, torch.ones(1, device=nov_new.device)])
            if S_new.size(0) > self.cfg.error_sheaf.max_scars:
                idx = torch.argmin(sig_new)
                mask = torch.ones(S_new.size(0), dtype=torch.bool)
                mask[idx] = False
                S_new, sig_new, nov_new = S_new[mask], sig_new[mask], nov_new[mask]

        self.C1.data = C1_new
        self.C2.data = C2_new
        self.anchor.data = anchor_new
        self.target.data = target_new
        self.trauma.data = trauma_new
        self.ages.data = ages_new
        self.ages_arch.data = ages_arch_new
        self.sacred_mu.data = mu_new
        self.sacred_sigma2.data = sigma2_new
        self.S.data = S_new
        self.sig.data = sig_new
        self.nov.data = nov_new

        return C0, S_true

    def warmup(self, steps=None):
        steps = steps or self.cfg.warmup_steps
        with torch.no_grad():
            for _ in range(steps):
                self.forward(torch.zeros(self.cfg.dim), torch.zeros(self.cfg.dim))
