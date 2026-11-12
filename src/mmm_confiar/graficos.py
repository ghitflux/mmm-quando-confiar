"""Figuras do estudo. Azul é sempre o MMM, laranja a regressão linear, verde-água a atribuição proporcional ao gasto."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from mmm_confiar import dados
from mmm_confiar.estilo import APAGADO, AQUA, AZUL, EIXO, ESTILO, GRADE, LARANJA, RAMPA_AZUL, SUPERFICIE, TINTA, TINTA_SECUNDARIA, nova_figura, numero, percentual

CURTOS = {
    "laboratorio": "Laboratório (1/4 do ruído, 4 anos, com variação)",
    "metade_do_ruido": "Central com metade do ruído",
    "central_penalidade_roi": "Central + penalidade no ROI",
    "quatro_anos": "Histórico de 4 anos",
    "colinearidade_baixa": "Colinearidade baixa",
    "central": "Central (2 anos, colinearidade média)",
    "um_ano": "Histórico de 1 ano",
    "colinearidade_alta": "Colinearidade alta",
    "busca_demanda": "Busca responde à demanda",
    "dificil": "Difícil (colin. alta + busca responde)",
    "forma_errada": "Forma errada (saturação em S)",
    "central_variacao": "Central + variação deliberada",
    "central_experimento": "Central + experimento",
    "dificil_variacao": "Difícil + variação deliberada",
    "dificil_experimento": "Difícil + experimento",
}
OBSERVACIONAIS = ("laboratorio", "quatro_anos", "colinearidade_baixa", "central", "um_ano", "colinearidade_alta", "busca_demanda", "dificil", "forma_errada")


def _pct(valor: float, _=None) -> str:
    return percentual(valor)


def _linhas(ax, resumo: dict, chaves: tuple[str, ...], series: list[tuple[str, str, str]], rotular: str | None = None, desvio: float = 0.2) -> None:
    """Uma linha por cenário; em cada linha, um ponto com intervalo de Monte Carlo por série (métrica, nome, cor)."""
    for y, chave in enumerate(chaves):
        for i, (metrica, nome, cor) in enumerate(series):
            m = resumo[chave]["metricas"][metrica]
            posicao = y + (i - (len(series) - 1) / 2) * desvio
            ax.plot([m["inferior"], m["superior"]], [posicao, posicao], color=cor, linewidth=2, zorder=2, alpha=0.55)
            ax.plot(m["media"], posicao, "o", color=cor, zorder=3, label=nome if y == 0 else None)
            if metrica == rotular:
                ax.annotate(_pct(m["media"]), (m["superior"], posicao), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(range(len(chaves)), [CURTOS[c] for c in chaves])
    ax.set_ylim(-0.6, len(chaves) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct))


def pior_que_nao_mexer(resumo: dict) -> Figure:
    """Frequência com que realocar a verba pelo modelo vende menos do que manter a alocação atual."""
    central = resumo["central"]["metricas"]["mmm_pior_que_nao_mexer"]["media"]
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            f"No cenário central, seguir o MMM vende menos do que não mexer em {_pct(central)} das simulações",
            "Parte das séries em que a alocação recomendada rende menos que a atual, nas curvas verdadeiras. Traço: IC 95% de Monte Carlo.",
            altura=5.0,
        )
        fig.subplots_adjust(left=0.37, bottom=0.17, top=0.83)
        _linhas(ax, resumo, OBSERVACIONAIS, [("mmm_pior_que_nao_mexer", "MMM", AZUL), ("linear_pior_que_nao_mexer", "Regressão linear", LARANJA)], rotular="mmm_pior_que_nao_mexer")
        ax.axvline(0.5, color=EIXO, linewidth=1.2, zorder=1)
        ax.annotate("cara ou coroa", (0.5, -0.5), xytext=(4, 0), textcoords="offset points", color=APAGADO, fontsize=8.5, va="center")
        ax.set_xlim(-0.03, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("séries em que seguir a recomendação é pior do que manter a verba como está")
        ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=2, fontsize=9)
        return fig


def erro_por_cenario(resumo: dict) -> Figure:
    """Erro de contribuição do MMM e das alternativas simples, cenário por cenário."""
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            "Fora do laboratório, o MMM erra tanto quanto as alternativas simples, ou mais",
            "Venda atribuída ao canal errado ou inventada, em % da venda de mídia verdadeira (média das séries; traço: IC 95% de Monte Carlo).",
            altura=5.4,
        )
        fig.subplots_adjust(left=0.37, bottom=0.16, top=0.84)
        series = [("mmm_erro_contribuicao", "MMM", AZUL), ("linear_erro_contribuicao", "Regressão linear", LARANJA), ("proporcional_erro_contribuicao", "Proporcional ao gasto (total do MMM)", AQUA)]
        _linhas(ax, resumo, OBSERVACIONAIS, series, rotular="mmm_erro_contribuicao", desvio=0.24)
        ax.set_xlim(0, 2.0)
        ax.set_xlabel("erro de contribuição (0% = cada canal com a venda que gerou)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.12), ncol=3, fontsize=9)
        return fig


def credito_por_canal(resumo: dict) -> Figure:
    """Erro com sinal do MMM em cada canal, em três cenários: quem recebe crédito demais e quem recebe de menos."""
    cenarios = [("central", "Central"), ("busca_demanda", "Busca responde à demanda"), ("dificil", "Difícil (colin. alta + busca responde)")]
    busca = resumo["dificil"]["por_canal"]["mmm"]["busca"]["erro_com_sinal"]["media"]
    with plt.rc_context(ESTILO):
        fig, eixos = plt.subplots(1, 3, figsize=(8.2, 3.9), dpi=160, sharex=True, sharey=True)
        fig.subplots_adjust(top=0.72, left=0.105, right=0.98, bottom=0.17, wspace=0.08)
        fig.text(0.02, 0.955, f"Se a busca responde à demanda, o MMM dá a ela {_pct(busca)} de crédito a mais e tira dos outros", fontsize=13, fontweight="bold", color=TINTA, va="top")
        fig.text(0.02, 0.885, "Erro médio da contribuição estimada pelo MMM, por canal (estimado / verdadeiro - 1). Traço: IC 95% de Monte Carlo.", fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
        for ax, (chave, titulo) in zip(eixos, cenarios):
            por_canal = resumo[chave]["por_canal"]["mmm"]
            for y, canal in enumerate(dados.CANAIS):
                m = por_canal[canal]["erro_com_sinal"]
                ax.barh(y, m["media"], height=0.55, color=AZUL, zorder=2)
                ax.plot([m["inferior"], m["superior"]], [y, y], color=TINTA, linewidth=1.2, zorder=3)
                lado = m["superior"] if m["media"] >= 0 else m["inferior"]
                ax.annotate(f"{m['media']:+.0%}".replace("-", "−"), (lado, y), xytext=(5 if m["media"] >= 0 else -5, 0), textcoords="offset points",
                            va="center", ha="left" if m["media"] >= 0 else "right", fontsize=8.5, color=TINTA)
            ax.axvline(0, color=EIXO, linewidth=1.2, zorder=1)
            ax.set_title(titulo, fontsize=9.5, color=TINTA_SECUNDARIA, loc="left", pad=6)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}".replace("-", "−") if v else "0%"))
        eixos[0].set_yticks(range(len(dados.CANAIS)), [dados.ROTULOS[c] for c in dados.CANAIS])
        eixos[0].invert_yaxis()
        eixos[0].set_xlim(-1.7, 4.9)
        eixos[0].set_xticks([-1, 0, 2, 4])
        eixos[1].set_xlabel("crédito a mais (direita) ou a menos (esquerda) em relação à venda que o canal gerou")
        return fig


def cobertura(resumo: dict) -> Figure:
    """Cobertura real dos intervalos de 80% e de 95% do MMM, por cenário."""
    dificil = resumo["dificil"]["metricas"]["mmm_cobertura_95"]["media"]
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            "Os intervalos só cumprem o prometido quando o modelo descreve o mundo certo",
            f"Parte dos canais cujo efeito verdadeiro cai no intervalo do MMM (no cenário difícil, {_pct(dificil)} no de 95%). Traço: IC de Monte Carlo.",
            altura=5.0,
        )
        fig.subplots_adjust(left=0.37, bottom=0.17, top=0.83)
        _linhas(ax, resumo, OBSERVACIONAIS, [("mmm_cobertura_80", "Intervalo de 80%", RAMPA_AZUL[0]), ("mmm_cobertura_95", "Intervalo de 95%", AZUL)], rotular="mmm_cobertura_95")
        for nominal, cor in ((0.80, RAMPA_AZUL[0]), (0.95, AZUL)):
            ax.axvline(nominal, color=cor, linewidth=1.2, zorder=1, alpha=0.7)
        ax.annotate("prometido: 80% e 95%", (0.80, -0.5), xytext=(-4, 0), textcoords="offset points", color=APAGADO, fontsize=8.5, va="center", ha="right")
        ax.set_xlim(0.2, 1.1)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("cobertura real (canais e séries juntos)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=2, fontsize=9)
        return fig


def conserto(resumo: dict) -> Figure:
    """O que a variação deliberada de gasto e a calibração por experimento fazem com o erro e com a decisão."""
    chaves = ("central", "central_variacao", "central_experimento", "dificil", "dificil_variacao", "dificil_experimento")
    antes, depois = resumo["central"]["metricas"]["mmm_erro_contribuicao"]["media"], resumo["central_experimento"]["metricas"]["mmm_erro_contribuicao"]["media"]
    with plt.rc_context(ESTILO):
        fig, ax = nova_figura(
            f"Experimento ou variação deliberada derrubam o erro de {_pct(antes)} para cerca de {_pct(depois)}",
            "Erro de contribuição do MMM (ponto e IC 95% de Monte Carlo). À direita: ganho médio de venda de mídia ao realocar pelo modelo.",
            altura=4.3, largura=8.6,
        )
        fig.subplots_adjust(left=0.30, right=0.74, bottom=0.16, top=0.80)
        for y, chave in enumerate(chaves):
            m = resumo[chave]["metricas"]
            erro, ganho, possivel = m["mmm_erro_contribuicao"], m["mmm_ganho"], m["ganho_possivel"]["media"]
            cor = APAGADO if chave in ("central", "dificil") else AZUL
            ax.plot([erro["inferior"], erro["superior"]], [y, y], color=cor, linewidth=2, alpha=0.55, zorder=2)
            ax.plot(erro["media"], y, "o", color=cor, zorder=3)
            ax.annotate(_pct(erro["media"]), (erro["superior"], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
            texto = f"{ganho['media'] * 100:+.1f}%".replace(".", ",").replace("-", "−") + f"  (possível: +{numero(possivel * 100, 1)}%)"
            ax.annotate(texto, (1.03, y), xycoords=("axes fraction", "data"), va="center", fontsize=8.5, color=TINTA_SECUNDARIA)
        ax.annotate("ganho seguindo o modelo", (1.03, -0.75), xycoords=("axes fraction", "data"), va="center", fontsize=8.5, color=APAGADO)
        ax.axhline(2.5, color=GRADE, linewidth=0.8)
        ax.set_yticks(range(len(chaves)), [CURTOS[c] for c in chaves])
        ax.set_ylim(-1.0, len(chaves) - 0.4)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlim(0, 1.5)
        ax.xaxis.set_major_formatter(FuncFormatter(_pct))
        ax.set_xlabel("erro de contribuição do MMM")
        return fig


def curvas_do_exemplo(exemplo: dict) -> Figure:
    """Curvas de resposta numa série de exemplo: a verdadeira, a estimada e a faixa de 95% do bootstrap."""
    with plt.rc_context(ESTILO):
        fig, eixos = plt.subplots(1, 4, figsize=(8.6, 3.7), dpi=160)
        fig.subplots_adjust(top=0.70, left=0.07, right=0.985, bottom=0.25, wspace=0.34)
        fig.text(0.02, 0.955, "A contribuição de um canal é uma extrapolação até o gasto zero, onde não há dado", fontsize=13, fontweight="bold", color=TINTA, va="top")
        fig.text(0.02, 0.885, "Uma série do cenário central: venda semanal do canal (R$ mil) conforme o gasto. Fundo cinza: onde caíram 90% dos gastos observados.",
                 fontsize=9.5, color=TINTA_SECUNDARIA, va="top")
        for ax, canal in zip(eixos, dados.CANAIS):
            c = exemplo["curvas"][canal]
            x = np.array(c["gasto_sobre_a_media"])
            ax.axvspan(*c["gasto_observado_p05_p95"], color=GRADE, alpha=0.6, zorder=0, linewidth=0)
            ax.fill_between(x, c["inferior"], c["superior"], color=AZUL, alpha=0.18, zorder=1, linewidth=0, label="Faixa de 95% do MMM")
            ax.plot(x, c["estimada"], color=AZUL, zorder=3, label="Estimada pelo MMM")
            ax.plot(x, c["verdadeira"], color=TINTA, zorder=4, linewidth=1.6, label="Verdadeira")
            teto = max(max(c["verdadeira"]) * 2.2, max(c["estimada"]) * 1.15)
            ax.set_ylim(0, teto)
            ax.set_xlim(0, 2)
            ax.set_xticks([0, 1, 2], ["0", "média", "2×"])
            ax.set_title(dados.ROTULOS[canal], fontsize=10, color=TINTA, loc="left", pad=6)
            ax.tick_params(axis="y", labelsize=8)
        eixos[0].set_ylabel("venda semanal do canal")
        fig.text(0.5, 0.115, "gasto semanal do canal, em relação à média do período", ha="center", fontsize=10, color=TINTA_SECUNDARIA)
        alcas, nomes = eixos[0].get_legend_handles_labels()
        fig.legend(alcas, nomes, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01), facecolor=SUPERFICIE)
        return fig
