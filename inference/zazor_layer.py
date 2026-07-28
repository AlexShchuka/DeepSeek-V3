import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ------------------------------------------------------------
# Конфигурация (минимальная)
# ------------------------------------------------------------
class ZazorConfig:
    def __init__(self, dim: int, core_size: int = 16, archive_size: int = 32,
                 num_basal: int = 4, basal_dim: int = 4):
        self.dim = dim
        self.core_size = core_size
        self.archive_size = archive_size
        self.num_basal = num_basal
        self.basal_dim = basal_dim  # размерность аффективного выхода
        # Параметры энергии
        self.alpha_coh = 1.0
        self.alpha_work = 0.5
        self.alpha_reg = 0.01
        self.alpha_barrier = 10.0
        self.barrier_drift = 0.5
        self.barrier_paranoia = 0.5
        # Оптимизация
        self.inner_gamma_steps = 3
        self.gamma_lr = 0.01
        self.base_lr = 0.001

# ------------------------------------------------------------
# Вспомогательные слои
# ------------------------------------------------------------
class Theta(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(2 * dim, dim)
    def forward(self, K, F):
        return self.net(torch.cat([K, F], dim=-1))

class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(2 * dim, dim)
    def forward(self, ctx, mem_contrib, confidence_mem, confidence_ctx):
        logit_mod = torch.log(confidence_mem / (confidence_ctx + 1e-8))
        raw = self.W(torch.cat([ctx, mem_contrib], dim=-1))
        gate = torch.sigmoid(raw + logit_mod)
        return gate * mem_contrib + (1 - gate) * ctx

class Mixer(nn.Module):
    def __init__(self, dim, core_size):
        super().__init__()
        input_dim = 6  # drift, trauma_mean, gap, avg_variance, critic_error, avg_sacred
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.head_lr = nn.Linear(64, core_size)
        self.head_temp = nn.Linear(64, 1)
        self.head_mom_a = nn.Linear(64, 1)
        self.head_mom_t = nn.Linear(64, 1)
        self.head_mig = nn.Linear(64, 1)
        self.head_gain = nn.Linear(64, dim)
        self.head_heal = nn.Linear(64, dim)
        self.head_basal_bonus = nn.Linear(64, 1)
        self.head_basal_lr = nn.Linear(64, 1)

    def forward(self, coh):
        h = self.net(coh)
        return {
            'lr_C1': torch.sigmoid(self.head_lr(h)) * 0.1,
            'temperature': F.softplus(self.head_temp(h)) + 0.1,
            'mom_a': torch.sigmoid(self.head_mom_a(h)),
            'mom_t': torch.sigmoid(self.head_mom_t(h)),
            'mig_thresh': torch.sigmoid(self.head_mig(h)),
            'gain_scale': torch.sigmoid(self.head_gain(h)),
            'heal_scale': torch.sigmoid(self.head_heal(h)),
            'basal_bonus': F.softplus(self.head_basal_bonus(h)),
            'basal_lr': torch.sigmoid(self.head_basal_lr(h)) * 0.01
        }

class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
    def forward(self, anchor, target, C0):
        return self.net(torch.cat([anchor, target, C0], dim=-1)).squeeze(-1)

class ActionHead(nn.Module):
    def __init__(self, dim, basal_dim, modal_dim):
        super().__init__()
        total_out = basal_dim + modal_dim
        self.net = nn.Linear(dim * 2, total_out)  # C0 и target
    def forward(self, C0, target):
        return self.net(torch.cat([C0, target], dim=-1))

# ------------------------------------------------------------
# Основной модуль
# ------------------------------------------------------------
class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig):
        super().__init__()
        d, c, a = config.dim, config.core_size, config.archive_size
        self.cfg = config

        # Память
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
        # Шрамы как тензор напряжений между слотами C1
        self.T = nn.Parameter(torch.zeros(c, c))

        # Базальные слоты
        self.register_buffer('is_basal', torch.zeros(c, dtype=torch.bool))
        self.is_basal[:config.num_basal] = True
        # Ортогональная инициализация базальных слотов
        with torch.no_grad():
            base = torch.randn(config.num_basal, d)
            base = torch.linalg.qr(base.T)[0].T  # ортогонализация
            self.C1[:config.num_basal] = base * 0.1

        # Сети
        self.theta = Theta(d)
        self.gate = Gate(d)
        self.mixer = Mixer(d, c)
        self.critic = Critic(d)
        self.action = ActionHead(d, config.basal_dim, d)  # выход аффекта + модальности

        # Для хранения предыдущего состояния
        self.prev_C0 = None
        self.prev_S_true = None

    def compute_energy(self, C0, target, anchor, C1, is_basal, T, gamma, S_true, V_pred, drift, paranoia, gap):
        # Энергия когерентности
        E_coh = torch.sum((C0 - target)**2) + self.cfg.alpha_coh * drift
        # Работа критика
        E_work = (S_true - V_pred)**2 * (1 + torch.norm(C0 - target))
        # Регуляризация
        E_reg = self.cfg.alpha_reg * torch.sum(T**2)
        # Барьеры
        E_barrier = torch.relu(drift - self.cfg.barrier_drift)**2 + torch.relu(paranoia - self.cfg.barrier_paranoia)**2
        E_barrier = self.cfg.alpha_barrier * E_barrier
        return E_coh + E_work + E_reg + E_barrier

    def gamma_fixed_point(self, coh_vec, energy_fn, ctx, C1, is_basal, T, S_true, V_pred, drift, paranoia, gap,
                          C0, target, anchor):
        # Начальное приближение гаммы
        gamma = torch.tensor(0.5, device=C0.device)
        for _ in range(self.cfg.inner_gamma_steps):
            gamma = gamma.detach().clone().requires_grad_(True)
            energy = energy_fn(C0, target, anchor, C1, is_basal, T, gamma, S_true, V_pred, drift, paranoia, gap)
            grad = torch.autograd.grad(energy, gamma, create_graph=False)[0]
            gamma = gamma - self.cfg.gamma_lr * grad
            gamma = torch.sigmoid(gamma)  # удерживаем в [0,1]
        return gamma.detach()

    def forward(self, K: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.cfg.dim
        # Контекст
        ctx = self.theta(K, F)
        bridged = torch.sigmoid(self.gap) * ctx + (1 - torch.sigmoid(self.gap)) * K + self.anchor
        ctx = bridged

        # Вычисление текущих когомологий
        persona = self.C1[self.is_basal].mean(dim=0)  # упрощённая персона
        drift_tension = (1 - F.cosine_similarity(self.anchor, self.target, dim=0)) * \
                        (1 - F.cosine_similarity(self.anchor, persona, dim=0)) * \
                        (1 - F.cosine_similarity(persona, self.target, dim=0))
        trauma_mean = self.trauma.mean()
        avg_variance = self.sacred_sigma2.mean()
        # Паранойя из тензора T: сумма абсолютных значений T, нормированная
        paranoia = torch.sum(torch.abs(self.T)) / (self.T.numel() + 1e-8)
        avg_sacred = (self.sacred_mu + self.bias_from_T()).mean()
        gap_val = self.gap.squeeze()

        # Прогноз критика и реальная удовлетворённость
        C0_temp = ctx  # временно, будет пересчитан после гейта
        V_pred = self.critic(self.anchor, self.target, C0_temp)
        # S_true будет вычислен после формирования C0, но для энергии используем предыдущий или оцениваем
        if self.prev_C0 is not None:
            S_true = F.cosine_similarity(self.prev_C0, self.target, dim=0)
        else:
            S_true = torch.tensor(0.5, device=ctx.device)

        coh_vec = torch.stack([drift_tension, trauma_mean, gap_val, avg_variance,
                               torch.zeros(1, device=ctx.device), avg_sacred])  # critic_error пока 0

        # Параметры от миксера
        params = self.mixer(coh_vec)

        # Внимание ∂₁
        logits = (ctx @ self.C1.T) / (params['temperature'] + 1e-8)
        crisis_bonus = self.is_basal.float() * params['basal_bonus'] * ((1 - self.gamma) + paranoia)
        attn = torch.softmax(logits + crisis_bonus, dim=-1)
        mem_contrib = attn @ self.C1
        freshness = torch.exp(-self.ages * (1 + self.gamma))

        # Гейт
        confidence_mem = avg_sacred * (1 - paranoia) * self.gamma
        novelty = 1 - F.cosine_similarity(ctx, mem_contrib, dim=0)
        confidence_ctx = (1 - self.gamma) * (1 + paranoia) * novelty
        C0 = self.gate(ctx, mem_contrib, confidence_mem, confidence_ctx)

        # Удовлетворённость фактическая
        S_true = F.cosine_similarity(C0, self.target, dim=0)
        V_pred = self.critic(self.anchor, self.target, C0)
        critic_error = torch.abs(S_true - V_pred)

        # Энергия (для обучения параметров)
        energy = self.compute_energy(C0, self.target, self.anchor, self.C1, self.is_basal, self.T, self.gamma,
                                     S_true, V_pred, drift_tension, paranoia, gap_val)

        # Градиентный шаг по всем параметрам (кроме гаммы) — делаем через оптимизатор вручную
        # Здесь для краткости покажем, как обновляются основные параметры через градиенты энергии
        grads = torch.autograd.grad(energy, [self.C1, self.C2, self.anchor, self.target, self.trauma,
                                             self.ages, self.ages_arch, self.sacred_mu, self.sacred_sigma2, self.T],
                                    create_graph=False)
        lr = self.cfg.base_lr
        with torch.no_grad():
            self.C1 -= lr * grads[0]
            self.C2 -= lr * grads[1]
            self.anchor -= lr * grads[2]
            self.target -= lr * grads[3]
            self.trauma -= lr * grads[4]
            self.ages -= lr * grads[5]
            self.ages_arch -= lr * grads[6]
            self.sacred_mu -= lr * grads[7]
            self.sacred_sigma2 -= lr * grads[8]
            self.T -= lr * grads[9]

        self.gamma = self.gamma_fixed_point(coh_vec, self.compute_energy, ctx, self.C1, self.is_basal, self.T,
                                            S_true, V_pred, drift_tension, paranoia, gap_val, C0, self.target, self.anchor)

        affective_out = self.action(C0, self.target)

        self.prev_C0 = C0.detach()
        self.prev_S_true = S_true.detach()

        return C0, S_true, affective_out

    def bias_from_T(self):
        # Смещение sacred от напряжений шрамов: bias_i = sum_j T_ij * (средняя активация слота j?)
        # Для простоты используем текущие attn (приблизительно)
        # В реальности нужно хранить attn или использовать равномерное
        return torch.sum(self.T, dim=1) * 0.1

    def diagnostics(self):
        return {
            'gamma': self.gamma.item(),
            'drift_tension': (1 - F.cosine_similarity(self.anchor, self.target, dim=0)).item(),
            'trauma_mean': self.trauma.mean().item(),
            'paranoia': torch.sum(torch.abs(self.T)).item() / self.T.numel(),
            'avg_sacred': (self.sacred_mu + self.bias_from_T()).mean().item(),
            'gap': self.gap.item(),
            'T_norm': torch.norm(self.T).item()
        }
