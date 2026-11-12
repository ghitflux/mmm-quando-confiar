"""Estudo de recuperação: gera uma série, ajusta os métodos e compara com o gabarito.

`avaliar_serie` faz o trabalho numa série. `resumir` junta as séries de um cenário e calcula as métricas com o
erro de Monte Carlo (intervalo de 95% reamostrando as sementes), para não confundir ruído do sorteio com efeito.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm_confiar import dados, linha_de_base, modelo, otimizacao

NIVEIS = (0.80, 0.95)
CANAIS_DO_EXPERIMENTO = ("busca", "afiliados")  # onde o histórico menos consegue responder (ver README)
ERRO_DO_EXPERIMENTO = 0.20  # erro-padrão do experimento, em fração do efeito verdadeiro (premissa)


@dataclass(frozen=True)
class Configuracao:
    partidas: int = 8
    repeticoes: int = 200  # réplicas do bootstrap por série


def simular_experimento(serie: dados.Serie, canais: tuple[str, ...] = CANAIS_DO_EXPERIMENTO, erro: float = ERRO_DO_EXPERIMENTO) -> modelo.Calibracao:
    """Resultado de um teste de retirada de verba em cada canal de `canais`.

    O teste (por exemplo, desligar o canal em parte das regiões por algumas semanas) mede a venda semanal que o
    canal gera, com erro. Aqui a medida é o valor verdadeiro mais um erro normal com desvio de `erro` x o valor.
    """
    gerador = np.random.default_rng([serie.semente, 909])
    indices = np.array([dados.CANAIS.index(c) for c in canais])
    verdade = serie.contribuicao_verdadeira[indices] / len(serie.tabela)
    erro_padrao = erro * verdade
    return modelo.Calibracao(canais=indices, medida=verdade + erro_padrao * gerador.normal(size=len(indices)), erro_padrao=erro_padrao)


def _cobre(amostras: np.ndarray, verdade: np.ndarray) -> dict[str, np.ndarray]:
    saida = {}
    for nivel in NIVEIS:
        inferior, superior = modelo.intervalo(amostras, nivel)
        chave = f"{round(100 * nivel)}"
        saida[f"cobre_{chave}"] = (inferior <= verdade) & (verdade <= superior)
        saida[f"inferior_{chave}"], saida[f"superior_{chave}"] = inferior, superior
    return saida


def avaliar_serie(serie: dados.Serie, calibrar: bool = False, config: Configuracao = Configuracao(), penalizar_roi: bool = False) -> dict:
    """Ajusta o MMM e as alternativas simples numa série e compara com o gabarito.

    Devolve um dicionário plano: vetores por canal (contribuição verdadeira e estimada, limites dos intervalos,
    acertos de cobertura, alocações) e escalares (ganho e arrependimento de cada decisão).
    """
    d = modelo.montar_desenho(serie.tabela, dados.CANAIS)
    verdade = serie.contribuicao_verdadeira
    saida: dict = {
        "semente": serie.semente,
        "correlacao_do_gasto": dados.correlacao_media_do_gasto(serie),
        "verdade": verdade,
        "gasto_total": d.gasto_total,
        "vendas_total": float(serie.vendas.sum()),
    }

    # MMM
    calibracao = simular_experimento(serie) if calibrar else None
    ajuste = modelo.ajustar(d, partidas=config.partidas, semente=serie.semente, calibracao=calibracao, penalizar_roi=penalizar_roi)
    replicas = modelo.contribuicao_total(modelo.bootstrap_em_blocos(ajuste, config.repeticoes, serie.semente), d)
    saida["mmm"] = modelo.contribuicao_total(ajuste.theta, d)[0]
    saida["mmm_partidas_no_mesmo_otimo"] = ajuste.partidas_no_mesmo_otimo
    saida.update({f"mmm_{k}": v for k, v in _cobre(replicas, verdade).items()})

    # Regressão linear
    linear = linha_de_base.regressao_linear(d)
    saida["linear"] = linear.contribuicao_total
    saida.update({f"linear_{k}": v for k, v in _cobre(linha_de_base.bootstrap_linear(linear, config.repeticoes, serie.semente), verdade).items()})

    # Atribuição proporcional ao gasto. A regra precisa de um total de venda de mídia para repartir:
    # - "proporcional": usa o total estimado pelo MMM (é o que dá para fazer na prática);
    # - "proporcional_ideal": usa o total verdadeiro, que ninguém tem. Mede só o erro da repartição.
    saida["proporcional"] = linha_de_base.atribuicao_proporcional(d.gasto_total, float(saida["mmm"].sum()))
    saida["proporcional_ideal"] = linha_de_base.atribuicao_proporcional(d.gasto_total, float(verdade.sum()))

    # Decisão: realocar a verba semanal média, avaliando nas curvas verdadeiras
    atual = d.gasto_medio
    mundo = serie.mundo
    em_s = bool((mundo.expoente > 1).any())
    alocacoes = {
        "mmm": otimizacao.alocar(lambda x: modelo.resposta_por_canal(ajuste.theta, d, x), atual),
        "linear": otimizacao.alocar_linear(linear.roi, atual),
        "otima": otimizacao.alocar(mundo.resposta_por_canal, atual, partidas=6 if em_s else 1, semente=serie.semente),
    }
    # o teto é a melhor alocação conhecida: protege contra o otimizador parar perto, mas não em cima, do máximo
    otima = max(alocacoes.values(), key=lambda x: mundo.resposta(x))
    saida["alocacao_atual"], saida["alocacao_otima"] = atual, otima
    for metodo in ("mmm", "linear"):
        decisao = otimizacao.avaliar_decisao(mundo.resposta_por_canal, atual, alocacoes[metodo], otima)
        saida[f"alocacao_{metodo}"] = alocacoes[metodo]
        saida[f"{metodo}_ganho"], saida[f"{metodo}_arrependimento"] = decisao["ganho"], decisao["arrependimento"]
        saida["ganho_possivel"] = decisao["ganho_possivel"]
    return saida


def ordem(valores: np.ndarray) -> tuple[int, ...]:
    """Os canais do maior para o menor valor."""
    return tuple(int(i) for i in np.argsort(-np.asarray(valores), kind="stable"))


def pares_invertidos(estimado: np.ndarray, verdadeiro: np.ndarray) -> float:
    """Parte dos pares de canais que o estimado ordena ao contrário do verdadeiro (0 = ordem certa; 0,5 = cara ou coroa).

    Empate no estimado conta como meio erro: quem não distingue dois canais acerta a ordem deles metade das vezes.
    """
    estimado, verdadeiro = np.asarray(estimado, dtype=float), np.asarray(verdadeiro, dtype=float)
    i, j = np.triu_indices(len(estimado), k=1)
    diferenca = estimado[i] - estimado[j]
    diferenca[np.abs(diferenca) <= 1e-9 * np.abs(estimado).max()] = 0.0  # empate, a menos de arredondamento
    sinal_estimado, sinal_verdadeiro = np.sign(diferenca), np.sign(verdadeiro[i] - verdadeiro[j])
    erro = np.where(sinal_estimado == 0, 0.5, (sinal_estimado != sinal_verdadeiro).astype(float))
    return float(erro.mean())


def metricas_da_serie(avaliacao: dict) -> dict[str, float]:
    """Reduz a avaliação de uma série a números: um por métrica e método."""
    verdade, gasto = avaliacao["verdade"], avaliacao["gasto_total"]
    roi_verdadeiro = verdade / gasto
    linha: dict[str, float] = {"correlacao_do_gasto": avaliacao["correlacao_do_gasto"], "ganho_possivel": avaliacao["ganho_possivel"]}
    for metodo in ("mmm", "linear", "proporcional", "proporcional_ideal"):
        estimado = avaliacao[metodo]
        erro_relativo = np.abs(estimado - verdade) / verdade
        roi = estimado / gasto
        # erro de contribuição: venda atribuída ao canal errado ou inventada, em fração da venda de mídia verdadeira
        linha[f"{metodo}_erro_contribuicao"] = float(np.abs(estimado - verdade).sum() / verdade.sum())
        linha[f"{metodo}_erro_roi"] = float(np.median(erro_relativo))  # erro relativo do ROI do canal mediano
        linha[f"{metodo}_erro_total"] = float(abs(estimado.sum() - verdade.sum()) / verdade.sum())
        linha[f"{metodo}_pares_invertidos"] = pares_invertidos(roi, roi_verdadeiro)
        if metodo.startswith("proporcional"):
            continue  # todos os canais com o mesmo ROI: não há ordem, intervalo nem realocação
        linha[f"{metodo}_ordem_errada"] = float(ordem(roi) != ordem(roi_verdadeiro))
        linha[f"{metodo}_melhor_canal_errado"] = float(int(np.argmax(roi)) != int(np.argmax(roi_verdadeiro)))
        for nivel in ("80", "95"):
            linha[f"{metodo}_cobertura_{nivel}"] = float(avaliacao[f"{metodo}_cobre_{nivel}"].mean())
        linha[f"{metodo}_largura_95"] = float(np.median((avaliacao[f"{metodo}_superior_95"] - avaliacao[f"{metodo}_inferior_95"]) / verdade))
        linha[f"{metodo}_ganho"] = avaliacao[f"{metodo}_ganho"]
        linha[f"{metodo}_arrependimento"] = avaliacao[f"{metodo}_arrependimento"]
        linha[f"{metodo}_pior_que_nao_mexer"] = float(avaliacao[f"{metodo}_ganho"] < 0)
    # não mexer na verba: ganho zero, arrependimento igual ao ganho possível
    linha["manter_arrependimento"] = avaliacao["ganho_possivel"]
    # diferenças na mesma série (pareadas): é nelas que se decide se um método ganha, perde ou empata
    for chave in ("erro_contribuicao", "erro_roi", "pares_invertidos", "ganho"):
        linha[f"mmm_menos_linear_{chave}"] = linha[f"mmm_{chave}"] - linha[f"linear_{chave}"]
    for chave in ("erro_contribuicao", "erro_roi", "pares_invertidos"):
        linha[f"mmm_menos_proporcional_{chave}"] = linha[f"mmm_{chave}"] - linha[f"proporcional_{chave}"]
    return linha


def diferenca_pareada(antes: list[dict], depois: list[dict], chave: str) -> dict[str, float]:
    """Diferença média de uma métrica entre dois cenários, semente a semente (depois - antes), com erro de Monte Carlo.

    As duas listas têm as mesmas sementes na mesma ordem: o mundo e o ruído são os mesmos, só o cenário muda.
    """
    a = np.array([metricas_da_serie(x)[chave] for x in antes])
    b = np.array([metricas_da_serie(x)[chave] for x in depois])
    return media_com_erro(b - a)


def media_com_erro(valores: np.ndarray, repeticoes: int = 2000, semente: int = 0) -> dict[str, float]:
    """Média entre as sementes com intervalo de 95% de Monte Carlo (bootstrap percentil das sementes)."""
    valores = np.asarray(valores, dtype=float)
    gerador = np.random.default_rng([semente, 111])
    medias = valores[gerador.integers(0, len(valores), size=(repeticoes, len(valores)))].mean(axis=1)
    return {
        "media": float(valores.mean()), "inferior": float(np.quantile(medias, 0.025)), "superior": float(np.quantile(medias, 0.975)),
        "mediana": float(np.median(valores)),
    }


def resumir(avaliacoes: list[dict]) -> dict[str, dict[str, float]]:
    """Métricas de um cenário: média entre as sementes e intervalo de Monte Carlo de cada uma."""
    linhas = [metricas_da_serie(a) for a in avaliacoes]
    return {chave: media_com_erro(np.array([linha[chave] for linha in linhas])) for chave in linhas[0]}


def vies_por_canal(avaliacoes: list[dict], metodo: str = "mmm") -> dict[str, dict[str, dict[str, float]]]:
    """Por canal: erro relativo com sinal da contribuição (positivo = o método dá crédito demais) e cobertura."""
    saida = {}
    for c, canal in enumerate(dados.CANAIS):
        relativo = np.array([(a[metodo][c] - a["verdade"][c]) / a["verdade"][c] for a in avaliacoes])
        saida[canal] = {
            "erro_com_sinal": media_com_erro(relativo),
            "erro_absoluto": media_com_erro(np.abs(relativo)),
            "cobertura_95": media_com_erro(np.array([float(a[f"{metodo}_cobre_95"][c]) for a in avaliacoes])),
            "cobertura_80": media_com_erro(np.array([float(a[f"{metodo}_cobre_80"][c]) for a in avaliacoes])),
        }
    return saida
