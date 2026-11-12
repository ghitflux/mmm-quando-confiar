"""Os cenários do estudo e as tabelas que saem deles. Cada tabela do README e do notebook vem de uma função daqui."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from mmm_confiar import dados, linha_de_base, modelo, otimizacao
from mmm_confiar import recuperacao as rec
from mmm_confiar import transformacoes as tr

SEMENTES = 48  # séries simuladas por cenário
CONFIG = rec.Configuracao(partidas=8, repeticoes=200)
EXEMPLO = ("central", 0)  # cenário e semente do conjunto de exemplo versionado em data/


@dataclass(frozen=True)
class Item:
    chave: str
    rotulo: str
    cenario: dados.Cenario
    calibrar: bool = False       # usar o resultado de um experimento (em busca e afiliados) para calibrar o MMM
    grupo: str = ""
    penalizar_roi: bool = False  # MMM com penalidade genérica no ROI (o padrão do Meridian), como sensibilidade


_C = dados.Cenario
_DIFICIL = dict(colinearidade=0.9, busca_responde_a_demanda=True)
CENARIOS: tuple[Item, ...] = (
    Item("central", "Central: 2 anos, colinearidade média", _C("central"), grupo="referência"),
    Item("colinearidade_baixa", "Colinearidade baixa", _C("colinearidade_baixa", colinearidade=0.1), grupo="colinearidade"),
    Item("colinearidade_alta", "Colinearidade alta", _C("colinearidade_alta", colinearidade=0.9), grupo="colinearidade"),
    Item("um_ano", "Histórico de 1 ano", _C("um_ano", semanas=52), grupo="histórico"),
    Item("quatro_anos", "Histórico de 4 anos", _C("quatro_anos", semanas=208), grupo="histórico"),
    Item("busca_demanda", "Busca responde à demanda", _C("busca_demanda", busca_responde_a_demanda=True), grupo="causa e consequência"),
    Item("forma_errada", "Forma errada (saturação em S, pico atrasado)", _C("forma_errada", forma_errada=True), grupo="especificação"),
    Item("central_variacao", "Central + variação deliberada", _C("central_variacao", variacao_deliberada=True), grupo="conserto"),
    Item("central_experimento", "Central + calibração por experimento", _C("central_experimento"), calibrar=True, grupo="conserto"),
    Item("dificil", "Difícil: colinearidade alta + busca responde à demanda", _C("dificil", **_DIFICIL), grupo="referência"),
    Item("dificil_variacao", "Difícil + variação deliberada", _C("dificil_variacao", variacao_deliberada=True, **_DIFICIL), grupo="conserto"),
    Item("dificil_experimento", "Difícil + calibração por experimento", _C("dificil_experimento", **_DIFICIL), calibrar=True, grupo="conserto"),
    Item("central_penalidade_roi", "Central + penalidade no ROI", _C("central_penalidade_roi"), grupo="sensibilidade", penalizar_roi=True),
    Item("metade_do_ruido", "Central com metade do ruído", _C("metade_do_ruido", ruido=0.5), grupo="sensibilidade"),
    Item("laboratorio", "Laboratório: 4 anos, colinearidade baixa, variação deliberada, 1/4 do ruído",
         _C("laboratorio", semanas=208, colinearidade=0.1, variacao_deliberada=True, ruido=0.25), grupo="sensibilidade"),
)
POR_CHAVE = {item.chave: item for item in CENARIOS}

# pares (antes, depois) para as diferenças semente a semente
PARES = (
    ("central", "colinearidade_alta"), ("central", "um_ano"), ("central", "quatro_anos"), ("central", "busca_demanda"),
    ("central", "forma_errada"), ("central", "central_variacao"), ("central", "central_experimento"),
    ("dificil", "dificil_variacao"), ("dificil", "dificil_experimento"), ("central", "metade_do_ruido"), ("central", "central_penalidade_roi"),
)
METRICAS_PAREADAS = ("mmm_erro_contribuicao", "mmm_erro_roi", "mmm_pares_invertidos", "mmm_cobertura_95", "mmm_ganho", "mmm_pior_que_nao_mexer")


def rodar(itens: tuple[Item, ...] = CENARIOS, sementes: int = SEMENTES, config: rec.Configuracao = CONFIG, avisar=None) -> dict[str, list[dict]]:
    """Roda o estudo de recuperação: para cada cenário, `sementes` séries avaliadas. Um processo só, sem paralelismo."""
    resultado = {}
    for item in itens:
        resultado[item.chave] = [
            rec.avaliar_serie(dados.gerar(item.cenario, semente), item.calibrar, config, item.penalizar_roi) for semente in range(sementes)
        ]
        if avisar is not None:
            avisar(item)
    return resultado


def metricas(avaliacoes: dict[str, list[dict]]) -> dict[str, dict]:
    """Para cada cenário: as métricas (média e intervalo de Monte Carlo), o erro por canal e a alocação média."""
    saida = {}
    for chave, lista in avaliacoes.items():
        item = POR_CHAVE[chave]
        atual = np.array([a["alocacao_atual"] for a in lista])
        saida[chave] = {
            "rotulo": item.rotulo,
            "grupo": item.grupo,
            "parametros": {**asdict(item.cenario), "calibrado_por_experimento": item.calibrar, "penalidade_no_roi": item.penalizar_roi},
            "series": len(lista),
            "metricas": rec.resumir(lista),
            "partidas_no_mesmo_otimo": float(np.mean([a["mmm_partidas_no_mesmo_otimo"] for a in lista])),
            "por_canal": {metodo: rec.vies_por_canal(lista, metodo) for metodo in ("mmm", "linear")},
            "alocacao_media_sobre_a_atual": {
                nome: dict(zip(dados.CANAIS, np.mean(np.array([a[f"alocacao_{nome}"] for a in lista]) / atual, axis=0).round(4).tolist()))
                for nome in ("mmm", "linear", "otima")
            },
        }
    return saida


def diferencas(avaliacoes: dict[str, list[dict]]) -> dict[str, dict]:
    """Diferença semente a semente entre pares de cenários (depois - antes), nas métricas do MMM."""
    saida = {}
    for antes, depois in PARES:
        if antes in avaliacoes and depois in avaliacoes:
            saida[f"{antes} -> {depois}"] = {m: rec.diferenca_pareada(avaliacoes[antes], avaliacoes[depois], m) for m in METRICAS_PAREADAS}
    return saida


def comparacao(resumo: dict[str, dict]) -> pd.DataFrame:
    """Tabela do README: o MMM ao lado das alternativas simples, cenário por cenário."""
    linhas = {}
    for chave, bloco in resumo.items():
        m = bloco["metricas"]
        linhas[bloco["rotulo"]] = {
            "erro de contribuição: MMM": m["mmm_erro_contribuicao"]["media"],
            "regressão linear": m["linear_erro_contribuicao"]["media"],
            "proporcional ao gasto": m["proporcional_erro_contribuicao"]["media"],
            "MMM - linear (IC 95%)": f"{m['mmm_menos_linear_erro_contribuicao']['inferior']:+.2f} a {m['mmm_menos_linear_erro_contribuicao']['superior']:+.2f}",
            "pares invertidos: MMM": m["mmm_pares_invertidos"]["media"],
            "pares invertidos: linear": m["linear_pares_invertidos"]["media"],
            "cobertura 95%: MMM": m["mmm_cobertura_95"]["media"],
            "cobertura 95%: linear": m["linear_cobertura_95"]["media"],
            "ganho seguindo o MMM": m["mmm_ganho"]["media"],
            "ganho seguindo a linear": m["linear_ganho"]["media"],
            "ganho possível": m["ganho_possivel"]["media"],
        }
    return pd.DataFrame.from_dict(linhas, orient="index")


def erro_de_monte_carlo(resumo: dict[str, dict]) -> dict:
    """Meia largura do intervalo de 95% de Monte Carlo das métricas principais: a maior e a mediana entre os cenários."""
    saida = {}
    for metrica in ("mmm_erro_contribuicao", "mmm_pares_invertidos", "mmm_cobertura_95", "mmm_ganho", "mmm_pior_que_nao_mexer", "linear_erro_contribuicao"):
        meias = [(b["metricas"][metrica]["superior"] - b["metricas"][metrica]["inferior"]) / 2 for b in resumo.values()]
        saida[metrica] = {"maior_meia_largura": float(np.max(meias)), "meia_largura_mediana": float(np.median(meias))}
    return saida


def premissas() -> dict:
    """Os valores do gerador e do modelo, para o leitor conferir de onde vêm os números."""
    return {
        "unidade": "R$ mil por semana",
        "canais_de_referencia": dados.REFERENCIA,
        "venda_base_semanal": dados.BASE,
        "amplitude_sazonal_anual": dados.AMPLITUDE_ANUAL,
        "amplitude_sazonal_semestral": dados.AMPLITUDE_SEMESTRAL,
        "efeito_da_promocao": dados.EFEITO_PROMOCAO,
        "desvio_do_choque_de_demanda": dados.DESVIO_CHOQUE,
        "persistencia_do_choque": dados.PERSISTENCIA_CHOQUE,
        "desvio_do_ruido": dados.DESVIO_RUIDO,
        "resposta_da_busca_a_demanda": dados.RESPOSTA_DA_BUSCA_A_DEMANDA,
        "variacao_deliberada": {"semanas": dados.SEMANAS_DO_TESTE, "corte": dados.CORTE_NO_TESTE, "reforco": dados.REFORCO_NO_TESTE},
        "experimento": {"canais": list(rec.CANAIS_DO_EXPERIMENTO), "erro_padrao_relativo": rec.ERRO_DO_EXPERIMENTO},
        "modelo": {
            "retencao_central": modelo.RETENCAO_CENTRAL, "retencao_largura": modelo.RETENCAO_LARGURA, "retencao_maxima": modelo.RETENCAO_MAXIMA,
            "log_k_central": modelo.LOGK_CENTRAL, "log_k_largura": modelo.LOGK_LARGURA, "k_minimo": modelo.K_MINIMO, "k_maximo": modelo.K_MAXIMO,
        },
        "folga_da_realocacao": otimizacao.FOLGA,
        "sementes_por_cenario": SEMENTES,
        "replicas_do_bootstrap": CONFIG.repeticoes,
        "pontos_de_partida": CONFIG.partidas,
    }


def exemplo(serie: dados.Serie, config: rec.Configuracao = CONFIG, pontos: int = 41) -> dict:
    """Uma série olhada de perto: o que o MMM e a regressão linear dizem de cada canal, ao lado do gabarito.

    Devolve a tabela por canal (contribuição verdadeira, estimada e intervalo de 95%) e as curvas de resposta
    (verdadeira, estimada e faixa de 95% do bootstrap) numa grade de gasto de 0 a 2 vezes o gasto médio.
    """
    d = modelo.montar_desenho(serie.tabela, dados.CANAIS)
    ajuste = modelo.ajustar(d, partidas=config.partidas, semente=serie.semente)
    replicas = modelo.bootstrap_em_blocos(ajuste, config.repeticoes, serie.semente)
    contribuicoes = modelo.contribuicao_total(replicas, d)
    inferior, superior = modelo.intervalo(contribuicoes, 0.95)
    linear = linha_de_base.regressao_linear(d)
    estimado = modelo.contribuicao_total(ajuste.theta, d)[0]
    partes = modelo.parametros_por_canal(ajuste.theta, d)
    tabela = pd.DataFrame({
        "gasto no período": d.gasto_total,
        "venda verdadeira": serie.contribuicao_verdadeira,
        "MMM": estimado,
        "MMM IC 95% inferior": inferior,
        "MMM IC 95% superior": superior,
        "regressão linear": linear.contribuicao_total,
        "proporcional ao gasto": linha_de_base.atribuicao_proporcional(d.gasto_total, float(estimado.sum())),
        "ROI verdadeiro": serie.roi_verdadeiro,
        "ROI MMM": estimado / d.gasto_total,
        "retenção verdadeira": serie.mundo.retencao,
        "retenção MMM": partes["retencao"],
        "meia saturação verdadeira": serie.mundo.meia_saturacao * serie.mundo.gasto_referencia / d.gasto_medio,
        "meia saturação MMM": partes["meia_saturacao"],
    }, index=[dados.ROTULOS[c] for c in dados.CANAIS])

    fracao = np.linspace(0.0, 2.0, pontos)
    das_replicas = [modelo.parametros_por_canal(theta, d) for theta in replicas]
    venda_replicas = np.array([p["venda_no_gasto_medio"] for p in das_replicas])
    k_replicas = np.array([p["meia_saturacao"] for p in das_replicas])
    curvas = {}
    for c, canal in enumerate(dados.CANAIS):
        gasto = np.tile(d.gasto_medio, (pontos, 1))
        gasto[:, c] = fracao * d.gasto_medio[c]
        verdadeira = np.array([serie.mundo.resposta_por_canal(g)[c] for g in gasto])
        estimada = np.array([modelo.resposta_por_canal(ajuste.theta, d, g)[c] for g in gasto])
        curvas_das_replicas = venda_replicas[:, c, None] * tr.hill_normalizada(fracao[None, :], k_replicas[:, c, None])
        faixa_inferior, faixa_superior = modelo.intervalo(curvas_das_replicas, 0.95)
        curvas[canal] = {
            "gasto_sobre_a_media": fracao.round(4).tolist(), "verdadeira": verdadeira.round(3).tolist(), "estimada": estimada.round(3).tolist(),
            "inferior": faixa_inferior.round(3).tolist(), "superior": faixa_superior.round(3).tolist(),
            "gasto_observado_p05_p95": (np.quantile(serie.gasto[:, c], [0.05, 0.95]) / d.gasto_medio[c]).round(4).tolist(),
        }
    return {
        "cenario": serie.cenario.nome, "semente": serie.semente, "semanas": d.semanas,
        "correlacao_media_do_gasto": dados.correlacao_media_do_gasto(serie),
        "desvio_residual_sobre_a_venda_media": ajuste.desvio_residual,
        "partidas_no_mesmo_otimo": ajuste.partidas_no_mesmo_otimo,
        "tabela": tabela,
        "curvas": curvas,
    }
