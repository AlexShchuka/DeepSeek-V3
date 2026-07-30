"""
ZazorLayer — Спектральный Осьминог с относительным временем
Непрерывный поток, цветовые гармоники, многошкальная временная иерархия.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------
class ZazorConfig:
    def __init__(
        self,
        motive_dim: int = 64,       # размерность мотивного пространства
        max_slots: int = 128,       # максимум слотов памяти
        num_basal: int = 4,         # количество базальных слотов
        basal_dim: int = 4,         # размерность аффективного выхода
        top_k: int = 16,            # разреженность метрики T
        color_dim: int = 9,         # размерность спектра (2*K+1)
        time_window: int = 32,      # максимальная длина буфера истории
        base_temperature: float = 0.5,
        alpha_T: float = 1.0,
        beta_resonance: float = 1.0,
        lr_ricci: float = 0.01,
        lr_color: float = 0.01,
        surgery_thresh: float = 0.8,
        surgery_dist: float = 0.1,
    ):
        self.motive_dim = motive_dim
        self.max_slots = max_slots
        self.num_basal = num_basal
        self.basal_dim = basal_dim
        self.top_k = top_k
        self.color_dim = color_dim
        self.time_window = time_window
        self.base_temperature = base_temperature
        self.alpha_T = alpha_T
        self.beta_resonance = beta_resonance
        self.lr_ricci = lr_ricci
        self.lr_color = lr_color
        self.surgery_thresh = surgery_thresh
        self.surgery_dist = surgery_dist

# ------------------------------------------------------------
# Вспомогательные модули
# ------------------------------------------------------------
class Theta(nn.Module):
    """Объединение K и F в контекст."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(2 * dim, dim)
    def forward(self, K, F):
        return self.net(torch.cat([K, F], dim=-1))

class Gate(nn.Module):
    """Ворота: смешивание контекста и памяти."""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(2 * dim, dim)
    def forward(self, ctx, mem, conf_mem, conf_ctx):
        logit = torch.log(conf_mem / (conf_ctx + 1e-8))
        gate = torch.sigmoid(self.W(torch.cat([ctx, mem], dim=-1)) + logit)
        return gate * mem + (1 - gate) * ctx

class Critic(nn.Module):
    """Предсказатель удовлетворённости."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 1))
    def forward(self, anchor, target, C0):
        return self.net(torch.cat([anchor, target, C0], dim=-1)).squeeze(-1)

class ActionHead(nn.Module):
    """Формирует базальный аффект и модальный выход."""
    def __init__(self, dim, basal_dim):
        super().__init__()
        self.net = nn.Linear(dim * 2, basal_dim + dim)
    def forward(self, C0, target):
        return self.net(torch.cat([C0, target], dim=-1))

# ------------------------------------------------------------
# Основной слой
# ------------------------------------------------------------
class ZazorLayer(nn.Module):
    def __init__(self, config: ZazorConfig):
        super().__init__()
        self.cfg = config
        d = config.motive_dim
        cdim = config.color_dim

        # Мотивная сфера
        self.slot_motives = nn.Parameter(F.normalize(torch.randn(config.max_slots, d), dim=1))
        # Спектральные цвета (коэффициенты Фурье)
        self.slot_spectra = nn.Parameter(torch.rand(config.max_slots, cdim) * 0.1)
        # Инициализация базальных спектров чистыми гармониками
        with torch.no_grad():
            for i in range(min(config.num_basal, config.max_slots)):
                # Одна активная гармоника с номером i+1
                self.slot_spectra[i, 0] = 0.5               # постоянная составляющая
                if cdim > 1 and i*2+1 < cdim:
                    self.slot_spectra[i, 2*i+1] = 0.8       # косинусная гармоника
                if cdim > 2 and i*2+2 < cdim:
                    self.slot_spectra[i, 2*i+2] = 0.4       # синусная гармоника

        # Временные масштабы для каждого слота (логарифмически равномерно)
        self.time_scales = nn.Parameter(torch.logspace(-1, 1, config.max_slots))
        self.time_scales.requires_grad = False  # фиксируем, но можно обучать

        # Маски активности и базальности
        self.register_buffer('active_mask', torch.ones(config.max_slots, dtype=torch.bool))
        self.register_buffer('is_basal', torch.zeros(config.max_slots, dtype=torch.bool))
        self.is_basal[:config.num_basal] = True

        # Параметры контекстного тракта
        self.anchor = nn.Parameter(torch.zeros(d))
        self.target = nn.Parameter(torch.zeros(d))

        # Сети
        self.theta = Theta(d)
        self.gate = Gate(d)
        self.critic = Critic(d)
        self.action = ActionHead(d, config.basal_dim)

        # Буфер истории (циклический)
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('history', torch.zeros(config.time_window, d))

    # ------------------------------------------------------------
    # Добавление фрейма в историю
    # ------------------------------------------------------------
    def push_frame(self, frame):
        idx = self.history_ptr.item() % self.cfg.time_window
        self.history[idx] = frame
        self.history_ptr += 1

    # ------------------------------------------------------------
    # Извлечение контекста и входного спектра с учётом time_scales
    # ------------------------------------------------------------
    def get_temporal_context(self):
        """Возвращает общий контекст ctx и входной спектр input_spectrum,
        используя взвешенное внимание по времени с разными масштабами для каждого слота."""
        T = min(self.history_ptr.item(), self.cfg.time_window)
        if T == 0:
            # Нет истории — возвращаем нули
            return torch.zeros(self.cfg.motive_dim), torch.zeros(self.cfg.color_dim)

        hist = self.history[:T]  # (T, d)
        # Временные метки (чем ближе к текущему, тем больше t)
        t = torch.arange(T, dtype=torch.float32) - (T - 1)  # от -(T-1) до 0

        # Для каждого слота вычисляем веса softmax(-|t| / scale_i)
        scales = self.time_scales.unsqueeze(1)  # (N, 1)
        t_expanded = t.unsqueeze(0)  # (1, T)
        # Внимание: чем дальше в прошлое, тем меньше вес
        logits = -torch.abs(t_expanded) / (scales + 1e-8)  # (N, T)
        weights = torch.softmax(logits, dim=1)  # (N, T)

        # Контекст для каждого слота: weights @ hist
        slot_contexts = weights @ hist  # (N, d)

        # Общий контекст — средневзвешенное по слотам с учётом time_scale (быстрые слоты вносят больший вклад в текущий момент)
        # Веса для смешивания контекстов: softmax(scales) или просто нормированные scales
        mix_weights = torch.softmax(scales.squeeze(), dim=0)  # (N,)
        ctx = (mix_weights.unsqueeze(0) @ slot_contexts).squeeze(0)  # (d,)

        # Входной спектр: усредняем цветовые компоненты по истории с общим (медианным) масштабом
        median_scale = scales.median()
        global_weights = torch.softmax(-torch.abs(t) / (median_scale + 1e-8), dim=0)  # (T,)
        input_spectrum = global_weights @ hist[:, :self.cfg.color_dim]  # (color_dim,)
        return ctx, input_spectrum

    # ------------------------------------------------------------
    # Резонансная метрика T
    # ------------------------------------------------------------
    def compute_T(self, motives, spectra, input_spectrum):
        # Геодезическое расстояние на сфере
        cos = motives @ motives.T
        dist_sq = 2.0 - 2.0 * cos
        T_base = torch.exp(-self.cfg.alpha_T * dist_sq)

        # Спектральное напряжение: используем L2-разность спектров (травма = амплитуда первой гармоники?)
        spec_diff = torch.sum((spectra.unsqueeze(1) - spectra.unsqueeze(0))**2, dim=2)  # (N,N)
        T_color = T_base * (1.0 + spec_diff)

        # Резонанс с входным спектром
        resonance = torch.exp(-torch.sum((spectra - input_spectrum.unsqueeze(0))**2, dim=1))  # (N,)
        T_resonance = self.cfg.beta_resonance * torch.outer(resonance, resonance)
        T = T_color * (1.0 + T_resonance)

        # Разреженность через top_k
        if self.cfg.top_k:
            top_vals, top_idx = torch.topk(cos, min(self.cfg.top_k, cos.shape[0]), dim=1)
            mask = torch.zeros_like(T)
            mask.scatter_(1, top_idx, 1.0)
            T = T * mask
        return T

    # ------------------------------------------------------------
    # Хирургия
    # ------------------------------------------------------------
    def surgery(self, motives, spectra, T):
        active = self.active_mask.nonzero(as_tuple=True)[0]
        if len(active) < 2:
            return motives, spectra

        # Матрица близости по мотивам
        cos_active = motives[active] @ motives[active].T
        dist_sq_active = 2.0 - 2.0 * cos_active
        close = (dist_sq_active < self.cfg.surgery_dist) & ~torch.eye(len(active), device=cos_active.device, dtype=torch.bool)

        if not close.any():
            return motives, spectra

        T_sub = T[active][:, active] * close.float()
        max_val, max_idx = T_sub.max(dim=1)
        val, row = max_val.max(dim=0)
        col = max_idx[row]
        if val < self.cfg.surgery_thresh:
            return motives, spectra

        i = active[row].item()
        j = active[col].item()

        # Слияние
        new_motive = F.normalize((motives[i] + motives[j]) / 2.0, dim=0)
        new_spectrum = (spectra[i] + spectra[j]) / 2.0

        motives = motives.clone()
        spectra = spectra.clone()
        motives[i] = new_motive
        spectra[i] = new_spectrum
        self.active_mask[j] = False

        # Перенос базальности
        if self.is_basal[j]:
            self.is_basal[i] = True
        self.is_basal[j] = False

        return motives, spectra

    # ------------------------------------------------------------
    # Энергия потока Риччи
    # ------------------------------------------------------------
    def ricci_energy(self, T, spectra):
        diff = torch.sum((spectra.unsqueeze(1) - spectra.unsqueeze(0))**2, dim=2)
        return (T * diff).sum()

    # ------------------------------------------------------------
    # Основной шаг (принимает один фрейм потока)
    # ------------------------------------------------------------
    def forward(self, frame: torch.Tensor, dream: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dream:
            # Сон: вход = персона (среднее базальных)
            persona = self.slot_motives[self.is_basal].mean(dim=0)
            frame = torch.zeros_like(persona)
            # В истории оставляем нули, либо можно аккумулировать сны отдельно
        self.push_frame(frame)
        ctx, input_spectrum = self.get_temporal_context()

        # Нормировка мотивов
        with torch.no_grad():
            self.slot_motives.data = F.normalize(self.slot_motives.data, dim=1)

        # Метрика T
        T = self.compute_T(self.slot_motives, self.slot_spectra, input_spectrum)

        # Резонанс для внимания и гейта
        resonance = torch.exp(-torch.sum((self.slot_spectra - input_spectrum.unsqueeze(0))**2, dim=1))
        temperature = self.cfg.base_temperature * (1.0 + resonance.mean())

        # Внимание
        logits = (ctx @ self.slot_motives.T) / temperature
        basal_bonus = self.is_basal.float() * 0.1
        attn = torch.softmax(logits + basal_bonus, dim=-1)

        # Память и гейт
        mem_contrib = attn @ self.slot_motives
        conf_mem = resonance.mean()
        conf_ctx = 1.0 - F.cosine_similarity(ctx, mem_contrib, dim=0)
        C0 = self.gate(ctx, mem_contrib, conf_mem, conf_ctx)

        # Удовлетворённость
        S_true = F.cosine_similarity(C0, self.target, dim=0)

        # Энергия Риччи и градиенты
        E = self.ricci_energy(T, self.slot_spectra)
        grad_m = torch.autograd.grad(E, self.slot_motives, create_graph=False)[0]
        grad_s = torch.autograd.grad(E, self.slot_spectra, create_graph=False)[0]

        with torch.no_grad():
            self.slot_motives -= self.cfg.lr_ricci * grad_m
            self.slot_motives.data = F.normalize(self.slot_motives.data, dim=1)
            self.slot_spectra -= self.cfg.lr_color * grad_s
            self.slot_spectra.clamp_(0.0, 1.0)

        # Хирургия
        self.slot_motives.data, self.slot_spectra.data = self.surgery(
            self.slot_motives.data, self.slot_spectra.data, T
        )

        # Аффективный выход и глобальное время
        affective_out = self.action(C0, self.target)
        global_affect = torch.norm(affective_out[:self.cfg.basal_dim])  # базальная часть
        return C0, S_true, affective_out, global_affect

    # ------------------------------------------------------------
    # Микросон (несколько шагов сна)
    # ------------------------------------------------------------
    def microsleep(self, steps: int = 1):
        for _ in range(steps):
            self.forward(torch.zeros(self.cfg.motive_dim), dream=True)
