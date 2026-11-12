"""As duas transformações de um modelo de mix de marketing: efeito defasado (adstock) e retorno decrescente (Hill).

Adstock: o anúncio de hoje ainda vende nas semanas seguintes. O gasto de cada semana é espalhado pelas semanas
seguintes com pesos que caem em progressão geométrica. Os pesos somam 1, então o adstock redistribui o gasto no
tempo sem criar nem destruir: o total se conserva e um gasto constante continua constante.

Saturação: dobrar a verba não dobra a venda. A curva de Hill cresce sempre, cada vez mais devagar, até um teto.
"""

from __future__ import annotations

import numpy as np

JANELA = 13  # semanas em que um gasto ainda tem efeito (a semana do gasto e as 12 seguintes)


def pesos_geometricos(retencao: float | np.ndarray, janela: int = JANELA) -> np.ndarray:
    """Pesos do adstock geométrico: proporcionais a retencao**k (k = semanas desde o gasto), somando 1.

    `retencao` é a parte do efeito que passa de uma semana para a seguinte (0 = só vale na semana do gasto).
    Aceita um número ou uma matriz; a última dimensão do resultado tem `janela` posições.
    """
    retencao = np.asarray(retencao, dtype=float)
    potencias = retencao[..., None] ** np.arange(janela)
    return potencias / potencias.sum(axis=-1, keepdims=True)


def derivada_dos_pesos_geometricos(retencao: np.ndarray, janela: int = JANELA) -> np.ndarray:
    """Derivada de `pesos_geometricos` em relação à retenção (usada no jacobiano do ajuste)."""
    retencao = np.asarray(retencao, dtype=float)
    k = np.arange(janela)
    potencias = retencao[..., None] ** k
    # d(retencao**k) = k * retencao**(k-1): desloca as potências uma posição, sem expoente negativo.
    d_potencias = np.zeros_like(potencias)
    d_potencias[..., 1:] = k[1:] * potencias[..., :-1]
    soma = potencias.sum(axis=-1, keepdims=True)
    d_soma = d_potencias.sum(axis=-1, keepdims=True)
    return d_potencias / soma - potencias * d_soma / soma**2


def pesos_atrasados(retencao: float, pico: float, janela: int = JANELA) -> np.ndarray:
    """Pesos de um adstock com pico atrasado: proporcionais a retencao**((k - pico)**2), somando 1.

    É a forma "delayed adstock" de Jin et al. (2017): o efeito máximo acontece `pico` semanas depois do gasto.
    Usada só no gerador, no cenário de especificação errada (o modelo ajustado supõe sempre o geométrico).
    """
    k = np.arange(janela)
    potencias = float(retencao) ** ((k - pico) ** 2)
    return potencias / potencias.sum()


def matriz_de_defasagens(gasto: np.ndarray, janela: int = JANELA, anterior: np.ndarray | float | None = None) -> np.ndarray:
    """Matriz (semanas, janela): a coluna k traz o gasto de k semanas antes.

    `anterior` é o gasto das semanas antes do começo da série (o mais recente por último) ou um número único.
    Sem ele, usa-se a média da própria série: é o que o analista pode fazer, porque não vê o que veio antes.
    """
    gasto = np.asarray(gasto, dtype=float)
    if anterior is None:
        anterior = float(gasto.mean())
    if np.isscalar(anterior):
        anterior = np.full(janela - 1, float(anterior))
    anterior = np.asarray(anterior, dtype=float)
    if len(anterior) < janela - 1:
        raise ValueError("o histórico anterior precisa cobrir a janela do adstock")
    completo = np.concatenate([anterior[len(anterior) - (janela - 1):], gasto])
    n = len(gasto)
    return np.stack([completo[janela - 1 - k: janela - 1 - k + n] for k in range(janela)], axis=1)


def aplicar_adstock(gasto: np.ndarray, pesos: np.ndarray, anterior: np.ndarray | float | None = None) -> np.ndarray:
    """Gasto com efeito defasado: média ponderada do gasto da semana e das anteriores."""
    return matriz_de_defasagens(gasto, len(pesos), anterior) @ pesos


def adstock_geometrico(gasto: np.ndarray, retencao: float, janela: int = JANELA, anterior: np.ndarray | float | None = None) -> np.ndarray:
    """Adstock geométrico normalizado (pesos somam 1)."""
    return aplicar_adstock(gasto, pesos_geometricos(retencao, janela), anterior)


def hill(x: np.ndarray, meia_saturacao: float | np.ndarray, expoente: float | np.ndarray = 1.0) -> np.ndarray:
    """Curva de Hill: x**s / (x**s + K**s). Vale 0 em x = 0, 0,5 em x = K e tende a 1.

    Com expoente 1 a curva é côncava desde o início (retorno decrescente desde o primeiro real); com expoente
    maior que 1 ela tem forma de S (começa devagar, acelera e depois satura).
    """
    x = np.asarray(x, dtype=float)
    xs = x**expoente
    return xs / (xs + np.asarray(meia_saturacao, dtype=float) ** expoente)


def hill_normalizada(x: np.ndarray, meia_saturacao: float | np.ndarray, expoente: float | np.ndarray = 1.0) -> np.ndarray:
    """Hill dividida pelo seu valor em x = 1, de modo que a curva vale 1 quando o gasto está na média.

    Com o gasto medido em múltiplos da média, o coeficiente que multiplica esta curva é a venda semanal do canal
    no gasto médio. Isso separa "quanto o canal vende hoje" (bem medido) de "quão curva é a resposta" (mal medido).
    """
    return hill(x, meia_saturacao, expoente) / hill(1.0, meia_saturacao, expoente)
