"""Estilo único das figuras do portfólio (matplotlib). Copie para `src/<pacote>/estilo.py`.

Regras que o estilo materializa (skill `dataviz`):
- a forma vem antes da cor; a cor segue a entidade, nunca a posição no ranking;
- marcas finas (linha de 2 px, marcador de 8 px com anel na cor da superfície);
- grade e eixos discretos, sólidos, nunca tracejados; um eixo Y só;
- texto em tinta neutra, nunca na cor da série; rótulos diretos e seletivos;
- título diz a conclusão, subtítulo diz o que está sendo medido.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
APAGADO = "#898781"  # referência, linha de base, série desenfatizada
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

# Categóricas, nesta ordem fixa. As três primeiras passam em daltonismo em qualquer par;
# da quarta em diante, só em séries adjacentes (linhas, barras empilhadas). Acima de 8, agrupe.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZUL, LARANJA, AQUA = SERIES[:3]

# Rampa de um matiz só, para ordem ou magnitude (claro → escuro). Nunca arco-íris.
RAMPA_AZUL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Divergente: dois polos opostos e meio neutro.
DIVERGENTE = {"negativo": "#e34948", "neutro": "#f0efec", "positivo": "#2a78d6"}

ESTILO = {
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EIXO,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.labelcolor": TINTA_SECUNDARIA,
    "xtick.color": APAGADO,
    "ytick.color": APAGADO,
    "xtick.labelcolor": TINTA_SECUNDARIA,
    "ytick.labelcolor": TINTA_SECUNDARIA,
    "ytick.major.size": 0,
    "xtick.major.size": 3,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 8,
    "lines.markeredgewidth": 2,
    "lines.markeredgecolor": SUPERFICIE,
    "legend.frameon": False,
    "legend.labelcolor": TINTA_SECUNDARIA,
}


def nova_figura(titulo: str, subtitulo: str, altura: float = 4.4, largura: float = 8.2) -> tuple[Figure, plt.Axes]:
    """Figura com título (a conclusão) e subtítulo (o que é medido). Use dentro de `plt.rc_context(ESTILO)`."""
    fig, ax = plt.subplots(figsize=(largura, altura), dpi=160)
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.14)
    fig.text(0.02, 0.955, titulo, fontsize=13, fontweight="bold", color=TINTA, va="top")
    fig.text(0.02, 0.885, subtitulo, fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
    return fig, ax


def numero(valor: float, casas: int = 2) -> str:
    """Número com vírgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def percentual(valor: float, casas: int = 0) -> str:
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def variacao(valor: float) -> str:
    """Variação percentual com sinal e menos tipográfico; zero sem sinal."""
    arredondado = round(valor * 100)
    return "0%" if arredondado == 0 else f"{arredondado:+d}%".replace("-", "−")


def milhar(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")
