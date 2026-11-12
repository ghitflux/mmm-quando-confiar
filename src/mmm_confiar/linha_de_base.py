"""As alternativas simples ao MMM.

1. Regressão linear das vendas no gasto da própria semana, com a mesma linha de base do MMM (tendência,
   sazonalidade, promoção), sem efeito defasado e sem saturação. Cada real gasto vale o mesmo, sempre.
2. Atribuição proporcional ao gasto: reparte a venda de mídia entre os canais na proporção do que cada um gastou.
   Todos os canais ficam com o mesmo ROI, por construção.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm_confiar import modelo


@dataclass
class AjusteLinear:
    coeficientes: np.ndarray   # (q + C,) na escala relativa
    previsto: np.ndarray       # (T,)
    projecao: np.ndarray       # (q + C, T): leva um vetor de vendas aos coeficientes (pseudo-inversa)
    d: modelo.Desenho

    @property
    def roi(self) -> np.ndarray:
        """Venda a mais por real gasto, por canal. Na regressão linear o ROI médio e o marginal são o mesmo número."""
        return self.d.escala * self.coeficientes[self.d.q:] / self.d.gasto_medio

    @property
    def contribuicao_total(self) -> np.ndarray:
        return self.roi * self.d.gasto_total


def regressao_linear(d: modelo.Desenho) -> AjusteLinear:
    """Mínimos quadrados das vendas na base e no gasto da semana. O coeficiente pode sair negativo."""
    x = np.column_stack([d.Z, d.gasto_relativo])
    projecao = np.linalg.pinv(x)
    coeficientes = projecao @ d.y
    return AjusteLinear(coeficientes=coeficientes, previsto=x @ coeficientes, projecao=projecao, d=d)


def bootstrap_linear(ajuste: AjusteLinear, repeticoes: int = 200, semente: int = 0, bloco: int | None = None) -> np.ndarray:
    """Réplicas da contribuição total por canal, com o mesmo bootstrap em blocos de resíduos do MMM: (repeticoes, C)."""
    d = ajuste.d
    gerador = np.random.default_rng([semente, 707])
    bloco = bloco or modelo.tamanho_do_bloco(d.semanas)
    parametros = ajuste.projecao.shape[0]
    residuos = d.y - ajuste.previsto
    residuos = (residuos - residuos.mean()) * np.sqrt(d.semanas / max(d.semanas - parametros, 1))
    y = ajuste.previsto + residuos[modelo.indices_em_blocos(d.semanas, bloco, repeticoes, gerador)]
    coeficientes = y @ ajuste.projecao.T
    return d.escala * coeficientes[:, d.q:] / d.gasto_medio * d.gasto_total


def atribuicao_proporcional(gasto_total: np.ndarray, venda_de_midia: float) -> np.ndarray:
    """Reparte a venda de mídia na proporção do gasto de cada canal."""
    gasto_total = np.asarray(gasto_total, dtype=float)
    return venda_de_midia * gasto_total / gasto_total.sum()
