import numpy as np
import pandas as pd
import pytest

from mmm_confiar import linha_de_base, modelo, otimizacao

CANAIS = ("a", "b")


def _mundo_linear(ruido: float = 0.0, semente: int = 0) -> pd.DataFrame:
    gerador = np.random.default_rng(semente)
    semanas = 90
    gasto = gerador.uniform(5, 15, (semanas, 2))
    promocao = (gerador.uniform(size=semanas) < 0.1).astype(int)
    vendas = 300 + 2.5 * gasto[:, 0] + 0.5 * gasto[:, 1] + 20 * promocao + ruido * gerador.normal(size=semanas)
    return pd.DataFrame({"vendas": vendas, "promocao": promocao, "gasto_a": gasto[:, 0], "gasto_b": gasto[:, 1]})


def test_regressao_linear_recupera_um_mundo_linear():
    tabela = _mundo_linear()
    d = modelo.montar_desenho(tabela, CANAIS)
    ajuste = linha_de_base.regressao_linear(d)
    assert np.allclose(ajuste.roi, [2.5, 0.5])
    assert np.allclose(ajuste.contribuicao_total, [2.5 * tabela["gasto_a"].sum(), 0.5 * tabela["gasto_b"].sum()])


def test_bootstrap_linear_fica_em_volta_da_estimativa_e_e_reprodutivel():
    d = modelo.montar_desenho(_mundo_linear(ruido=5.0, semente=1), CANAIS)
    ajuste = linha_de_base.regressao_linear(d)
    replicas = linha_de_base.bootstrap_linear(ajuste, repeticoes=300, semente=2)
    assert replicas.shape == (300, 2)
    baixo, alto = modelo.intervalo(replicas, 0.95)
    assert np.all(baixo < ajuste.contribuicao_total) and np.all(ajuste.contribuicao_total < alto)
    assert np.allclose(replicas.mean(axis=0), ajuste.contribuicao_total, rtol=0.1)
    assert np.allclose(replicas, linha_de_base.bootstrap_linear(ajuste, repeticoes=300, semente=2))


def test_atribuicao_proporcional_reparte_o_total_e_da_o_mesmo_roi_a_todos():
    gasto = np.array([40.0, 35.0, 30.0, 10.0])
    contribuicao = linha_de_base.atribuicao_proporcional(gasto, 230.0)
    assert contribuicao.sum() == pytest.approx(230.0)
    assert np.allclose(contribuicao / gasto, 2.0)


def test_alocacao_otima_confere_com_a_solucao_analitica():
    # maximizar a1 * raiz(x1) + a2 * raiz(x2) com x1 + x2 fixo: o ótimo reparte na proporção de a ao quadrado
    pesos = np.array([3.0, 1.0])
    atual = np.array([10.0, 10.0])
    alocacao = otimizacao.alocar(lambda x: pesos * np.sqrt(x), atual, folga=0.95)
    assert np.allclose(alocacao, [18.0, 2.0], atol=1e-3)
    assert alocacao.sum() == pytest.approx(20.0)


def test_alocacao_respeita_os_limites_por_canal_e_canais_iguais_ficam_iguais():
    pesos = np.array([3.0, 1.0])
    atual = np.array([10.0, 10.0])
    limitada = otimizacao.alocar(lambda x: pesos * np.sqrt(x), atual, folga=0.5)
    assert np.allclose(limitada, [15.0, 5.0], atol=1e-6)  # o ótimo livre (18 e 2) não cabe: para no limite
    iguais = otimizacao.alocar(lambda x: 5 * x / (x + 4), np.array([6.0, 14.0]), folga=0.9)
    assert np.allclose(iguais, [10.0, 10.0], atol=1e-3)


def test_alocacao_com_varias_partidas_acha_o_maximo_de_uma_curva_em_s():
    def resposta(x):
        return np.array([x[0] ** 2 / (x[0] ** 2 + 15.0**2), 0.3 * x[1] / (x[1] + 5.0)]) * 100

    atual = np.array([8.0, 12.0])
    melhor = otimizacao.alocar(resposta, atual, folga=0.9, partidas=8)
    grade = np.linspace(0.8, 15.2, 2000)
    valores = [resposta(np.array([g, 20 - g])).sum() for g in grade]
    assert resposta(melhor).sum() >= max(valores) - 1e-3
    assert melhor.sum() == pytest.approx(20.0)


def test_alocar_linear_enche_o_canal_de_maior_roi_ate_o_limite():
    atual = np.array([40.0, 30.0, 20.0, 10.0])
    alocacao = otimizacao.alocar_linear(np.array([1.0, 3.0, -0.5, 2.0]), atual)
    # mínimos: 20, 15, 10, 5 (sobram 50). Maior ROI primeiro: canal 1 recebe 30, canal 3 recebe 10, canal 0 recebe 10
    assert np.allclose(alocacao, [30.0, 45.0, 10.0, 15.0])
    assert alocacao.sum() == pytest.approx(atual.sum())


def test_projetar_na_verba_mantem_limites_e_soma():
    minimo, maximo = np.array([1.0, 1.0, 1.0]), np.array([5.0, 5.0, 5.0])
    projetado = otimizacao.projetar_na_verba(np.array([5.0, 4.0, 4.5]), 9.0, minimo, maximo)
    assert projetado.sum() == pytest.approx(9.0)
    assert np.all(projetado >= minimo) and np.all(projetado <= maximo)
    with pytest.raises(ValueError):
        otimizacao.projetar_na_verba(np.array([1.0, 1.0, 1.0]), 20.0, minimo, maximo)


def test_avaliar_decisao_mede_ganho_e_arrependimento_nas_curvas_verdadeiras():
    def verdade(x):
        return np.array([3.0, 1.0]) * np.sqrt(x)

    atual, otima = np.array([10.0, 10.0]), np.array([15.0, 5.0])
    venda_atual = verdade(atual).sum()
    certa = otimizacao.avaliar_decisao(verdade, atual, otima, otima)
    assert certa["arrependimento"] == pytest.approx(0.0)
    assert certa["ganho"] == pytest.approx(certa["ganho_possivel"]) and certa["ganho"] > 0
    errada = otimizacao.avaliar_decisao(verdade, atual, np.array([5.0, 15.0]), otima)
    assert errada["ganho"] < 0  # realocar para o canal errado é pior do que não mexer
    assert errada["arrependimento"] == pytest.approx(errada["ganho_possivel"] - errada["ganho"])
    parada = otimizacao.avaliar_decisao(verdade, atual, atual, otima)
    assert parada["ganho"] == pytest.approx(0.0)
    assert parada["arrependimento"] == pytest.approx((verdade(otima).sum() - venda_atual) / venda_atual)
