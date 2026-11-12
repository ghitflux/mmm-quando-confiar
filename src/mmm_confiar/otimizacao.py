"""Da medição à decisão: repartir a verba entre os canais e medir o que a repartição recomendada custa.

A alocação é escolhida para maximizar a venda semanal prevista pelas curvas de resposta, com a verba total fixa e
um limite por canal (o gestor não corta nem aumenta um canal em mais de 50% de uma vez). O resultado é avaliado
nas curvas verdadeiras, que só o gerador conhece.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

FOLGA = 0.5  # cada canal pode ir de 50% a 150% do gasto atual


def projetar_na_verba(gasto: np.ndarray, verba: float, minimo: np.ndarray, maximo: np.ndarray) -> np.ndarray:
    """Desloca todos os canais na mesma quantia (respeitando os limites) até a soma bater com a verba."""
    if not minimo.sum() - 1e-9 <= verba <= maximo.sum() + 1e-9:
        raise ValueError("a verba não cabe nos limites dos canais")
    baixo, alto = float((minimo - gasto).min()), float((maximo - gasto).max())
    for _ in range(80):
        meio = (baixo + alto) / 2
        if np.clip(gasto + meio, minimo, maximo).sum() < verba:
            baixo = meio
        else:
            alto = meio
    return np.clip(gasto + (baixo + alto) / 2, minimo, maximo)


def alocar(
    resposta: Callable[[np.ndarray], np.ndarray], atual: np.ndarray, folga: float = FOLGA, partidas: int = 1, semente: int = 0
) -> np.ndarray:
    """Alocação que maximiza a soma de `resposta(gasto)` (venda semanal por canal), mantendo a verba total de `atual`.

    Usa SLSQP (SciPy) com a restrição de verba e os limites por canal. Com curvas côncavas uma partida basta; com
    curvas em S o problema tem máximos locais, então `partidas` maior que 1 tenta também pontos sorteados.
    """
    atual = np.asarray(atual, dtype=float)
    verba, minimo, maximo = float(atual.sum()), (1 - folga) * atual, (1 + folga) * atual
    gerador = np.random.default_rng([semente, 808])
    candidatos = [atual] + [projetar_na_verba(gerador.uniform(minimo, maximo), verba, minimo, maximo) for _ in range(partidas - 1)]
    escala = max(float(np.sum(resposta(atual))), 1e-9)
    melhor, melhor_valor = atual, float(np.sum(resposta(atual)))
    for partida in candidatos:
        saida = minimize(
            lambda x: -float(np.sum(resposta(x))) / escala, partida, method="SLSQP", bounds=list(zip(minimo, maximo)),
            constraints=[{"type": "eq", "fun": lambda x: (x.sum() - verba) / verba, "jac": lambda x: np.full(len(x), 1 / verba)}],
            options={"maxiter": 200, "ftol": 1e-10},
        )
        x = projetar_na_verba(np.clip(saida.x, minimo, maximo), verba, minimo, maximo)
        valor = float(np.sum(resposta(x)))
        if valor > melhor_valor:
            melhor, melhor_valor = x, valor
    return melhor


def alocar_linear(roi: np.ndarray, atual: np.ndarray, folga: float = FOLGA) -> np.ndarray:
    """Alocação ótima quando cada real vale sempre o mesmo: enche o canal de maior ROI até o limite, depois o seguinte.

    Parte de todos os canais no mínimo e distribui o que sobra da verba, do maior ROI para o menor.
    """
    atual = np.asarray(atual, dtype=float)
    minimo, maximo = (1 - folga) * atual, (1 + folga) * atual
    gasto, sobra = minimo.copy(), float(atual.sum() - minimo.sum())
    for canal in np.argsort(-np.asarray(roi, dtype=float), kind="stable"):
        acrescimo = min(sobra, maximo[canal] - minimo[canal])
        gasto[canal] += acrescimo
        sobra -= acrescimo
    return gasto


def avaliar_decisao(verdade: Callable[[np.ndarray], np.ndarray], atual: np.ndarray, recomendada: np.ndarray, otima: np.ndarray) -> dict[str, float]:
    """Ganho e arrependimento de uma alocação, medidos nas curvas verdadeiras, em fração da venda de mídia atual.

    - ganho: quanto a venda de mídia muda ao trocar a alocação atual pela recomendada (negativo = piorou);
    - ganho_possivel: o mesmo para a alocação ótima sob as curvas verdadeiras (o teto);
    - arrependimento: o que se deixa de ganhar por seguir a recomendada em vez da ótima (nunca negativo).
    """
    venda_atual = float(np.sum(verdade(atual)))
    venda_recomendada, venda_otima = float(np.sum(verdade(recomendada))), float(np.sum(verdade(otima)))
    return {
        "ganho": (venda_recomendada - venda_atual) / venda_atual,
        "ganho_possivel": (venda_otima - venda_atual) / venda_atual,
        "arrependimento": (venda_otima - venda_recomendada) / venda_atual,
    }
