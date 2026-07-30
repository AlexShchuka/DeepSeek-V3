"""
ZazorLayer — Голографический Осьминог
Единый файл: мотивная сфера, цветной поток Риччи, хирургия, микросон.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------
class ZazorConfig:
    def __init__(self, motive_dim: int = 64, max_slots: int = 128,
                 num_basal: int = 4, basal_dim: int = 4, top_k: int = 16):
        # Мотивное пространство
        self.motive_dim = motive_dim
        self.max_slots = max_slots
        self.num_basal = num_basal
        self.basal_dim = basal_dim

        # Поток Риччи и метрика
        self.alpha_T = 1.0          # коэффициент в exp(-α * dist^2)
        self.lr_ricci = 0.01        # шаг потока для slot_motives
        self.lr_color = 0.01        # шаг для slot_colors
        self.lambda_sphere = 0.1    # удержание на сфере (не используется, т.к. явная проекция)

        # Хирургия
        self.surgery_thresh = 0.8   # T_ij > этого → кандидат
        self.surgery_dist = 0.1     # расстояние < этого → можно слить
        self.max_surgery_per_step = 1

        # Внимание и гейт
        self.temperature = 0.5      # базовая температура внимания
        self.gamma_R0 = 0.1         # целевая скалярная кривизна
        self.gamma_beta = 5.0       # крутизна сигмоиды

        # Память
        self.top_k = top_k          # для разреженной T
        self.use_sparse = True      # использовать ли top_k

        # Микросон
        self.dream_steps = 1        # шагов за один микросон

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
    def __init__(self, dim, basal_dim):
        super().__init__()
        self.net = nn.Linear(dim * 2, basal_dim + dim)
    def forward(self, C0, target):
        return self.net(torch.cat([C0, target], dim=-1))

# ------------------------------------------------------------
# Основной модуль
# ------------------------------------------------------------
class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig):
        super().__init__()
        self.cfg = config
        d = config.motive_dim

        # Мотивная сфера: параметры слотов (нормированные векторы)
        self.slot_motives = nn.Parameter(torch.randn(config.max_slots, d))
        with torch.no_grad():
            self.slot_motives.data = F.normalize(self.slot_motives.data, dim=1)

        # Цвета слотов: явный срез пучка (RGB)
        self.slot_colors = nn.Parameter(torch.rand(config.max_slots, 3) * 0.1)
        # Базальные слоты получают чистые базовые цвета
        with torch.no_grad():
            if config.num_basal >= 3:
                self.slot_colors[0] = torch.tensor([1.0, 0.0, 0.0])  # красный
                self.slot_colors[1] = torch.tensor([0.0, 1.0, 0.0])  # зелёный
                self.slot_colors[2] = torch.tensor([0.0, 0.0, 1.0])  # синий
                if config.num_basal >= 4:
                    self.slot_colors[3] = torch.tensor([1.0, 1.0, 0.0])  # жёлтый

        # Маски активности и базальности
        self.register_buffer('active_mask', torch.ones(config.max_slots, dtype=torch.bool))
        self.register_buffer('is_basal', torch.zeros(config.max_slots, dtype=torch.bool))
        self.is_basal[:config.num_basal] = True

        # Параметры контекстного тракта
        self.anchor = nn.Parameter(torch.zeros(d))
        self.target = nn.Parameter(torch.zeros(d))
        self.gap = nn.Parameter(torch.zeros(1))

        # Сети
        self.theta = Theta(d)
        self.gate = Gate(d)
        self.critic = Critic(d)
        self.action = ActionHead(d, config.basal_dim)

        # Состояние
        self.gamma = 0.5

    # --------------------------------------------------------
    # Вычисление метрики T (разреженной или полной)
    # --------------------------------------------------------
    def compute_T(self, motives, colors):
        N = motives.shape[0]
        # Косинусные близости: (N, N)
        cos_sim = motives @ motives.T
        dist_sq = 2.0 - 2.0 * cos_sim  # геодезическое расстояние на сфере

        # Цветовое напряжение: разница в красном канале (травма)
        trauma = colors[:, 0]
        trauma_diff = torch.abs(trauma.unsqueeze(0) - trauma.unsqueeze(1))

        T_full = torch.exp(-self.cfg.alpha_T * dist_sq) * (1.0 + trauma_diff)

        if self.cfg.use_sparse:
            # Оставляем только top_k ближайших соседей для каждой вершины
            topk_vals, topk_idx = torch.topk(cos_sim, self.cfg.top_k, dim=1)
            mask = torch.zeros_like(T_full)
            mask.scatter_(1, topk_idx, 1.0)
            T_full = T_full * mask
        return T_full

    # --------------------------------------------------------
    # Скалярная кривизна и гамма
    # --------------------------------------------------------
    def compute_gamma(self, T, colors):
        # Локальная кривизна: R_i = sum_j T_ij * ||color_i - color_j||^2
        color_diff = (colors.unsqueeze(1) - colors.unsqueeze(0)).pow(2).sum(dim=2)  # (N,N)
        R_i = (T * color_diff).sum(dim=1)
        R = R_i.mean()
        gamma = torch.sigmoid(self.cfg.gamma_beta * (R - self.cfg.gamma_R0))
        return gamma, R

    # --------------------------------------------------------
    # Хирургия: слияние двух близких слотов
    # --------------------------------------------------------
    def surgery(self, motives, colors, T):
        N = motives.shape[0]
        # Ищем активные слоты
        active = self.active_mask.nonzero(as_tuple=True)[0]
        if len(active) < 2:
            return motives, colors, False

        # Маска близких точек
        cos_sim = motives[active] @ motives[active].T
        dist_sq = 2.0 - 2.0 * cos_sim
        close_mask = dist_sq < self.cfg.surgery_dist
        # Не рассматриваем диагональ
        close_mask = close_mask & ~torch.eye(len(active), device=motives.device, dtype=torch.bool)

        if close_mask.sum() == 0:
            return motives, colors, False

        # Выбираем пару с максимальным T_ij среди близких
        T_sub = T[active][:, active]
        # Зануляем те, где не близки
        T_sub = T_sub * close_mask.float()
        max_val, max_idx = T_sub.max(dim=1)
        max_val_global, row = max_val.max(dim=0)
        col = max_idx[row]

        if max_val_global < self.cfg.surgery_thresh:
            return motives, colors, False

        i = active[row].item()
        j = active[col].item()

        # Слияние: средний мотив, нормализованный
        new_motive = F.normalize((motives[i] + motives[j]) / 2.0, dim=0)
        new_color = (colors[i] + colors[j]) / 2.0

        # Заменяем i-й слот новым, j-й деактивируем
        motives = motives.clone()
        colors = colors.clone()
        motives[i] = new_motive
        colors[i] = new_color
        self.active_mask[j] = False

        # При необходимости переносим базальность, если один из них был базальным
        if self.is_basal[j]:
            self.is_basal[i] = True
        self.is_basal[j] = False

        return motives, colors, True

    # --------------------------------------------------------
    # Основной forward
    # --------------------------------------------------------
    def forward(self, K: torch.Tensor, F: torch.Tensor,
                dream_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dream_mode:
            # Микросон: нулевые входы, контекст = персона
            persona = self.slot_motives[self.is_basal].mean(dim=0, keepdim=True)
            K = torch.zeros_like(K)
            F = torch.zeros_like(F)
            ctx = persona.squeeze(0)
        else:
            ctx = self.theta(K, F)

        # Шаг 0: нормируем мотивы (на всякий случай)
        with torch.no_grad():
            self.slot_motives.data = F.normalize(self.slot_motives.data, dim=1)

        # Вычисляем метрику T и кривизну
        T = self.compute_T(self.slot_motives, self.slot_colors)
        gamma, R = self.compute_gamma(T, self.slot_colors)
        self.gamma = gamma.item()

        # Внимание и гейт
        temperature = self.cfg.temperature * (1.0 + gamma)
        logits = (ctx @ self.slot_motives.T) / temperature
        # Бонус базальным слотам
        basal_bonus = self.is_basal.float() * gamma * 0.5
        attn = torch.softmax(logits + basal_bonus, dim=-1)
        mem_contrib = attn @ self.slot_motives

        # Уверенности
        confidence_mem = gamma * (1.0 - R / (R + 1.0))
        novelty = 1.0 - F.cosine_similarity(ctx, mem_contrib, dim=0)
        confidence_ctx = (1.0 - gamma) * novelty

        C0 = self.gate(ctx, mem_contrib, confidence_mem, confidence_ctx)

        # Удовлетворённость
        S_true = F.cosine_similarity(C0, self.target, dim=0)
        V_pred = self.critic(self.anchor, self.target, C0)

        # Энергия для потока Риччи
        energy = self.compute_ricci_energy(T, self.slot_colors)

        # Градиентный шаг для мотивов
        grad_motives = torch.autograd.grad(energy, self.slot_motives, create_graph=False)[0]
        with torch.no_grad():
            self.slot_motives -= self.cfg.lr_ricci * grad_motives
            self.slot_motives.data = F.normalize(self.slot_motives.data, dim=1)

        # Градиентный шаг для цветов
        grad_colors = torch.autograd.grad(energy, self.slot_colors, create_graph=False)[0]
        with torch.no_grad():
            self.slot_colors -= self.cfg.lr_color * grad_colors
            self.slot_colors.clamp_(0.0, 1.0)

        # Хирургия
        self.slot_motives.data, self.slot_colors.data, _ = self.surgery(
            self.slot_motives.data, self.slot_colors.data, T
        )

        # Аффективный выход
        affective_out = self.action(C0, self.target)

        return C0, S_true, affective_out

    # --------------------------------------------------------
    # Энергия потока Риччи (цветовая)
    # --------------------------------------------------------
    def compute_ricci_energy(self, T, colors):
        # E = sum_{i,j} T_ij * ||c_i - c_j||^2
        diff = (colors.unsqueeze(1) - colors.unsqueeze(0)).pow(2).sum(dim=2)
        return (T * diff).sum()

    # --------------------------------------------------------
    # Микросон: просто вызов forward с dream_mode=True
    # --------------------------------------------------------
    def microsleep(self, steps: int = None):
        if steps is None:
            steps = self.cfg.dream_steps
        for _ in range(steps):
            self.forward(torch.zeros(self.cfg.motive_dim), torch.zeros(self.cfg.motive_dim), dream_mode=True)

    # --------------------------------------------------------
    # Диагностика
    # --------------------------------------------------------
    def diagnostics(self):
        return {
            'gamma': self.gamma,
            'active_slots': self.active_mask.sum().item(),
            'basal_active': self.is_basal.sum().item(),
        }
