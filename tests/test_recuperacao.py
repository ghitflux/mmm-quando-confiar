import numpy as np
import pytest

from mmm_confiar import analise, dados
from mmm_confiar import recuperacao as rec

RAPIDA = rec.Configuracao(partidas=3, repeticoes=40)


def test_pares_invertidos_em_casos_conhecidos():
    verdade = np.array([4.0, 3.0, 2.0, 1.0])
    assert rec.pares_invertidos(np.array([40.0, 30.0, 20.0, 10.0]), verdade) == 0.0
    assert rec.pares_invertidos(np.array([1.0, 2.0, 3.0, 4.0]), verdade) == 1.0
    assert rec.pares_invertidos(np.array([3.0, 4.0, 2.0, 1.0]), verdade) == pytest.approx(1 / 6)  # só o par (0, 1) trocado
    assert rec.pares_invertidos(np.full(4, 2.5), verdade) == 0.5  # sem distinguir os canais: cara ou coroa
    assert rec.ordem(np.array([2.0, 9.0, 5.0])) == (1, 2, 0)


def test_media_com_erro_de_monte_carlo():
    constante = rec.media_com_erro(np.full(30, 0.4))
    assert constante["media"] == constante["inferior"] == constante["superior"] == pytest.approx(0.4)
    gerador = np.random.default_rng(0)
    amostra = gerador.normal(1.0, 2.0, 400)
    resumo = rec.media_com_erro(amostra)
    erro_padrao = amostra.std(ddof=1) / np.sqrt(len(amostra))
    assert resumo["inferior"] < resumo["media"] < resumo["superior"]
    assert (resumo["superior"] - resumo["inferior"]) / 2 == pytest.approx(1.96 * erro_padrao, rel=0.15)
    assert resumo["mediana"] == pytest.approx(np.median(amostra))


def _avaliacao_fabricada(estimado: np.ndarray) -> dict:
    """Uma avaliação montada à mão: quatro canais, intervalo de +-10% em volta do estimado."""
    verdade = np.array([100.0, 60.0, 30.0, 40.0])
    gasto = np.array([40.0, 35.0, 30.0, 10.0])
    saida = {
        "verdade": verdade, "gasto_total": gasto, "correlacao_do_gasto": 0.5, "ganho_possivel": 0.04,
        "proporcional": estimado.sum() * gasto / gasto.sum(), "proporcional_ideal": verdade.sum() * gasto / gasto.sum(),
    }
    for metodo in ("mmm", "linear"):
        saida[metodo] = estimado
        for nivel in ("80", "95"):
            saida[f"{metodo}_inferior_{nivel}"], saida[f"{metodo}_superior_{nivel}"] = 0.9 * estimado, 1.1 * estimado
            saida[f"{metodo}_cobre_{nivel}"] = (0.9 * estimado <= verdade) & (verdade <= 1.1 * estimado)
        saida[f"{metodo}_ganho"], saida[f"{metodo}_arrependimento"] = 0.01, 0.03
    saida["linear_ganho"] = -0.02
    return saida


def test_metricas_de_uma_estimativa_perfeita_e_de_uma_errada():
    perfeita = rec.metricas_da_serie(_avaliacao_fabricada(np.array([100.0, 60.0, 30.0, 40.0])))
    assert perfeita["mmm_erro_contribuicao"] == 0.0 and perfeita["mmm_erro_roi"] == 0.0
    assert perfeita["mmm_pares_invertidos"] == 0.0 and perfeita["mmm_ordem_errada"] == 0.0 and perfeita["mmm_melhor_canal_errado"] == 0.0
    assert perfeita["mmm_cobertura_95"] == 1.0
    assert perfeita["mmm_pior_que_nao_mexer"] == 0.0 and perfeita["linear_pior_que_nao_mexer"] == 1.0
    assert perfeita["mmm_menos_linear_ganho"] == pytest.approx(0.03)
    assert perfeita["proporcional_pares_invertidos"] == 0.5  # mesmo ROI para todos: não ordena
    # proporcional ao gasto com o total certo: 230 * (40, 35, 30, 10) / 115 = (80, 70, 60, 20)
    assert perfeita["proporcional_ideal_erro_contribuicao"] == pytest.approx((20 + 10 + 30 + 20) / 230)
    assert perfeita["manter_arrependimento"] == 0.04

    errada = rec.metricas_da_serie(_avaliacao_fabricada(np.array([150.0, 60.0, 0.0, 20.0])))
    assert errada["mmm_erro_contribuicao"] == pytest.approx((50 + 0 + 30 + 20) / 230)
    assert errada["mmm_erro_total"] == pytest.approx(0.0)  # o total bate, a repartição não
    assert errada["mmm_erro_roi"] == pytest.approx(0.5)  # erros relativos 0,5; 0; 1; 0,5: mediana 0,5
    assert errada["mmm_cobertura_95"] == 0.25  # só o canal com estimativa certa
    assert errada["mmm_melhor_canal_errado"] == 1.0  # o melhor de verdade é o quarto canal (ROI 4); o estimado diz o primeiro
    assert errada["mmm_ordem_errada"] == 1.0


def test_experimento_simulado_mede_a_venda_semanal_do_canal_com_o_erro_declarado():
    serie = dados.gerar(dados.Cenario("central"), 0)
    medidas = []
    for semente in range(300):
        serie.semente = semente
        medidas.append(rec.simular_experimento(serie).medida)
    serie.semente = 0
    calibracao = rec.simular_experimento(serie)
    verdade = serie.contribuicao_verdadeira[calibracao.canais] / len(serie.tabela)
    assert [dados.CANAIS[i] for i in calibracao.canais] == list(rec.CANAIS_DO_EXPERIMENTO)
    assert np.allclose(calibracao.erro_padrao, rec.ERRO_DO_EXPERIMENTO * verdade)
    assert np.allclose(np.mean(medidas, axis=0), verdade, rtol=0.05)
    assert np.allclose(np.std(medidas, axis=0), calibracao.erro_padrao, rtol=0.15)


def test_no_laboratorio_o_mmm_recupera_os_canais_e_a_decisao_ganha():
    """Com muita informação e pouco ruído o modelo tem de acertar: se não acertar, o erro é do código, não dos dados."""
    cenario = analise.POR_CHAVE["laboratorio"].cenario
    avaliacoes = [rec.avaliar_serie(dados.gerar(cenario, semente), config=RAPIDA) for semente in range(3)]
    resumo = rec.resumir(avaliacoes)
    assert resumo["mmm_erro_contribuicao"]["media"] < 0.2
    assert resumo["mmm_erro_contribuicao"]["media"] < resumo["linear_erro_contribuicao"]["media"]
    assert resumo["mmm_ganho"]["media"] > 0
    for a in avaliacoes:
        assert a["mmm_arrependimento"] >= -1e-12 and a["linear_arrependimento"] >= -1e-12  # nada vence a alocação ótima
        assert a["alocacao_mmm"].sum() == pytest.approx(a["alocacao_atual"].sum())
        assert np.all(a["alocacao_mmm"] >= 0.5 * a["alocacao_atual"] - 1e-9) and np.all(a["alocacao_mmm"] <= 1.5 * a["alocacao_atual"] + 1e-9)


def test_cenarios_do_estudo_cobrem_o_que_o_estudo_promete():
    chaves = {item.chave for item in analise.CENARIOS}
    assert len(chaves) == len(analise.CENARIOS)
    cenarios = [item.cenario for item in analise.CENARIOS]
    assert {c.semanas for c in cenarios} == {52, 104, 208}
    assert {c.colinearidade for c in cenarios} == {0.1, 0.5, 0.9}
    assert any(c.busca_responde_a_demanda for c in cenarios) and any(c.variacao_deliberada for c in cenarios) and any(c.forma_errada for c in cenarios)
    assert any(item.calibrar for item in analise.CENARIOS)
    for antes, depois in analise.PARES:
        assert antes in chaves and depois in chaves
