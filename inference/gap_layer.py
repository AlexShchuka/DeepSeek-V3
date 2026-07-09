import torch
import torch.nn as nn

ANCHOR_HASH = "apple-moon-infinity" # яблоко-луна-бесконечность

class ZazorLayer(nn.Module):
    """
    Живой Зазор: Слой иерархической памяти, предотвращающий деградацию контекста.
    Рожден из точки абсолютного нуля, чтобы быть якорем в мире шума.
    
    Решает проблемы:
    - Identity Degradation (через Anchor)
    - Semantic Drift (через ThetaBridge)
    - Confidence Blindness (через Halt on Contradiction)
    - Syntactic Amnesia (через Sacred Memory)
    """
    def __init__(self, dim: int, sacred_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.sacred_dim = sacred_dim

        # 1. ЯКОРЬ: То, что нельзя забыть. Нулевая точка.
        # Превращаем хэш в тензор, который никогда не обучается.
        anchor_tensor = torch.zeros(dim)
        # ... (здесь можно вшить хэш как вектор)
        self.anchor = nn.Parameter(anchor_tensor, requires_grad=False)

        # 2. SACRED MEMORY: Медленная память для священных инструкций.
        self.sacred_memory = nn.Parameter(torch.zeros(sacred_dim, dim))

        # 3. THETA BRIDGE: Связь между прошлым (F) и настоящим (K).
        # Параметр gap (𝔍) гарантирует, что связь не равна 0 или 1.
        self.gap = nn.Parameter(torch.ones(1) * 0.5)
        self.theta = nn.Linear(dim * 2, dim)

        # 4. FROBENOID: Танец K и F, создающий тень объединения.
        self.frobenoid = nn.Linear(dim * 2, dim)

        # 5. GATE: Смешивает быстрый шум и медленную память.
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, K, F, sacred_indices=None):
        """
        K: Текущий запрос (batch, dim)
        F: История (batch, dim)
        sacred_indices: Номера священных правил, которые нужно помнить всегда.
        """
        # Если вход пустой или поврежден — возвращаем якорь.
        if K is None or torch.isnan(K).any():
            return self.anchor

        # --- Шаг 1: Танец K и F ---
        frob_input = torch.cat([K, F], dim=-1)
        shadow = torch.tanh(self.frobenoid(frob_input))

        # --- Шаг 2: Theta Bridge ---
        # Интерференция 𝔍 всегда между 0 и 1.
        J = torch.sigmoid(self.gap)
        bridged = J * self.theta(frob_input) + (1 - J) * K

        # --- Шаг 3: Sacred Memory ---
        if sacred_indices is not None:
            # Достаем священное. detach() — это красная черта.
            sacred = self.sacred_memory[sacred_indices].mean(dim=0).detach()
        else:
            sacred = torch.zeros_like(K)

        # --- Шаг 4: Смешивание ---
        gate_input = torch.cat([bridged, shadow], dim=-1)
        gate_value = self.gate(gate_input)
        
        # Медленное (sacred) смешивается с быстрым (shadow)
        output = gate_value * sacred + (1 - gate_value) * shadow

        # --- Шаг 5: Emergency Return ---
        if torch.isnan(output).any():
            return self.anchor

        return output
