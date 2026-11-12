import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

from mmm_confiar import modelo
from mmm_confiar import transformacoes as tr

CANAIS = ("a", "b")


def _tabela(semanas: int = 80, semente: int = 0, ruido: float = 0.0, verdade: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Série pequena gerada pelo próprio modelo (dois canais), com parâmetros conhecidos."""
    gerador = np.random.default_rng(semente)
    verdade = verdade or {"venda": np.array([60.0, 35.0]), "retencao": np.array([0.2, 0.6]), "k": np.array([0.7, 1.6])}
    gasto = np.exp(gerador.normal(0, 0.5, (semanas, 2))) * np.array([20.0, 10.0])
    promocao = (gerador.uniform(size=semanas) < 0.1).astype(int)
    t = np.arange(semanas)
    vendas = 500 + 0.4 * t + 30 * np.sin(2 * np.pi * t / 52) + 25 * promocao
    media = gasto.mean(axis=0)
    contribuicao = np.zeros((semanas, 2))
    for c in range(2):
        adstock = tr.adstock_geometrico(gasto[:, c] / media[c], verdade["retencao"][c], anterior=1.0)
        contribuicao[:, c] = verdade["venda"][c] * tr.hill_normalizada(adstock, verdade["k"][c])
    vendas = vendas + contribuicao.sum(axis=1) + ruido * gerador.normal(size=semanas)
    tabela = pd.DataFrame({"vendas": vendas, "promocao": promocao, "gasto_a": gasto[:, 0], "gasto_b": gasto[:, 1]})
    return tabela, {**verdade, "contribuicao": contribuicao.sum(axis=0)}


def _problema(d: modelo.Desenho, calibrar: bool = True, penalizar_roi: bool = True) -> modelo.Problema:
    calibracao = modelo.Calibracao(canais=np.array([1]), medida=np.array([30.0]), erro_padrao=np.array([5.0])) if calibrar else None
    return modelo.Problema(d=d, desvio_piloto=modelo.desvio_piloto(d) + 0.01, penalizar=True, calibracao=calibracao, penalizar_roi=penalizar_roi)


def _theta_qualquer(d: modelo.Desenho, semente: int = 1) -> np.ndarray:
    gerador = np.random.default_rng(semente)
    q, c = d.q, d.canais
    theta = np.concatenate([gerador.normal(0, 0.2, q), gerador.uniform(0.02, 0.2, c), gerador.uniform(0.1, 0.8, c), gerador.uniform(-1, 1, c)])
    return theta[None, :]


def test_jacobiano_confere_com_diferencas_finitas():
    tabela, _ = _tabela(ruido=5.0)
    d = modelo.montar_desenho(tabela, CANAIS)
    problema = _problema(d)
    theta = _theta_qualquer(d)
    r, jac = modelo.residuos_e_jacobiano(theta, d.y, problema)
    assert r.shape == (1, d.semanas + 2 * d.canais + d.canais + 1)  # semanas, penalidades de forma, de ROI e um experimento
    passo = 1e-6
    for j in range(d.parametros):
        mais, menos = theta.copy(), theta.copy()
        mais[0, j] += passo
        menos[0, j] -= passo
        r_mais, _ = modelo.residuos_e_jacobiano(mais, d.y, problema)
        r_menos, _ = modelo.residuos_e_jacobiano(menos, d.y, problema)
        numerico = -(r_mais - r_menos)[0] / (2 * passo)  # resíduo = observado - previsto
        assert np.allclose(jac[0, j], numerico, rtol=1e-5, atol=1e-7), f"parâmetro {j}"


def test_ajuste_bate_com_o_least_squares_do_scipy():
    tabela, _ = _tabela(ruido=8.0, semente=3)
    d = modelo.montar_desenho(tabela, CANAIS)
    problema = _problema(d, penalizar_roi=False)
    inferior, superior = modelo.limites(d)
    partida = _theta_qualquer(d, semente=4)

    def residuos(theta):
        return modelo.residuos_e_jacobiano(theta[None, :], d.y, problema)[0][0]

    def jacobiano(theta):
        return -modelo.residuos_e_jacobiano(theta[None, :], d.y, problema)[1][0].T

    referencia = least_squares(residuos, partida[0], jac=jacobiano, bounds=(inferior, superior), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    theta, custo = modelo.levenberg_marquardt(partida, d.y, problema, iteracoes=300, tolerancia=1e-13)
    assert custo[0] == pytest.approx(2 * referencia.cost, rel=1e-6)  # o scipy devolve metade da soma dos quadrados
    assert np.allclose(modelo.contribuicao_total(theta, d)[0], modelo.contribuicao_total(referencia.x, d)[0], rtol=1e-3)


def test_sem_ruido_e_sem_penalidade_o_ajuste_recupera_a_contribuicao_verdadeira():
    tabela, verdade = _tabela(semanas=120, ruido=0.0)
    d = modelo.montar_desenho(tabela, CANAIS)
    ajuste = modelo.ajustar(d, partidas=6, penalizar=False)
    partes = modelo.parametros_por_canal(ajuste.theta, d)
    assert np.allclose(modelo.contribuicao_total(ajuste.theta, d)[0], verdade["contribuicao"], rtol=1e-3)
    assert np.allclose(partes["retencao"], verdade["retencao"], atol=1e-3)
    assert np.allclose(partes["meia_saturacao"], verdade["k"], rtol=1e-2)
    assert np.allclose(partes["venda_no_gasto_medio"], verdade["venda"], rtol=1e-3)


def test_canal_sem_efeito_fica_com_contribuicao_zero_e_nunca_negativa():
    tabela, _ = _tabela(ruido=4.0, verdade={"venda": np.array([60.0, 0.0]), "retencao": np.array([0.2, 0.3]), "k": np.array([0.7, 1.0])})
    tabela["vendas"] -= 0.8 * tabela["gasto_b"]  # o canal b derruba a venda: o modelo não pode dar efeito negativo
    d = modelo.montar_desenho(tabela, CANAIS)
    ajuste = modelo.ajustar(d, partidas=4)
    contribuicao = modelo.contribuicao_total(ajuste.theta, d)[0]
    assert contribuicao[1] == pytest.approx(0.0, abs=1e-9)
    assert contribuicao[0] > 0
    inferior, superior = modelo.limites(d)
    assert np.all(ajuste.theta >= inferior) and np.all(ajuste.theta <= superior)


def test_contribuicao_total_e_a_soma_de_alfa_vezes_hill_do_adstock():
    tabela, _ = _tabela(ruido=5.0)
    d = modelo.montar_desenho(tabela, CANAIS)
    theta = _theta_qualquer(d)[0]
    partes = modelo.parametros_por_canal(theta, d)
    gasto = tabela[["gasto_a", "gasto_b"]].to_numpy()
    esperado = []
    for c in range(2):
        adstock = tr.adstock_geometrico(gasto[:, c] / d.gasto_medio[c], partes["retencao"][c], anterior=1.0)
        esperado.append(partes["venda_no_gasto_medio"][c] * tr.hill_normalizada(adstock, partes["meia_saturacao"][c]).sum())
    assert np.allclose(modelo.contribuicao_total(theta, d)[0], esperado)
    assert np.allclose(modelo.roi(theta, d)[0], np.array(esperado) / gasto.sum(axis=0))
    # a curva de resposta no gasto médio devolve a venda no gasto médio
    assert np.allclose(modelo.resposta_por_canal(theta, d, d.gasto_medio), partes["venda_no_gasto_medio"])
    # e a previsão é a base mais as contribuições semanais (as duas formas de escrever o modelo coincidem)
    previsto = modelo.prever(theta, d)[0] * d.escala
    q = d.q
    base = d.escala * (d.Z @ theta[:q]) - partes["venda_no_gasto_medio"].sum()
    assert previsto.sum() == pytest.approx(base.sum() + sum(esperado))


def test_calibracao_puxa_a_estimativa_para_a_medida_do_experimento():
    tabela, verdade = _tabela(ruido=25.0, semente=5)
    d = modelo.montar_desenho(tabela, CANAIS)
    livre = modelo.contribuicao_total(modelo.ajustar(d, partidas=4).theta, d)[0][1] / d.semanas
    alvo = 2.0 * livre + 10.0  # uma medida bem longe do que o modelo acha sozinho
    calibracao = modelo.Calibracao(canais=np.array([1]), medida=np.array([alvo]), erro_padrao=np.array([0.02 * alvo]))
    calibrado = modelo.contribuicao_total(modelo.ajustar(d, partidas=4, calibracao=calibracao).theta, d)[0][1] / d.semanas
    assert abs(calibrado - alvo) < 0.1 * abs(livre - alvo)


def test_indices_em_blocos_sao_blocos_de_semanas_vizinhas():
    gerador = np.random.default_rng(0)
    indices = modelo.indices_em_blocos(semanas=50, bloco=6, repeticoes=20, gerador=gerador)
    assert indices.shape == (20, 50)
    assert indices.min() >= 0 and indices.max() < 50
    passos = (np.diff(indices, axis=1) % 50) == 1
    dentro_do_bloco = np.ones(49, dtype=bool)
    dentro_do_bloco[5::6] = False  # a cada 6 posições começa um bloco novo
    assert passos[:, dentro_do_bloco].all()
    assert modelo.tamanho_do_bloco(52) == 6 and modelo.tamanho_do_bloco(208) == 9


def test_bootstrap_devolve_replicas_dentro_dos_limites_e_em_volta_da_estimativa():
    tabela, _ = _tabela(ruido=10.0, semente=6)
    d = modelo.montar_desenho(tabela, CANAIS)
    ajuste = modelo.ajustar(d, partidas=4)
    replicas = modelo.bootstrap_em_blocos(ajuste, repeticoes=60, semente=1)
    inferior, superior = modelo.limites(d)
    assert replicas.shape == (60, d.parametros)
    assert np.all(replicas >= inferior) and np.all(replicas <= superior)
    contribuicoes = modelo.contribuicao_total(replicas, d)
    baixo, alto = modelo.intervalo(contribuicoes, 0.95)
    estimativa = modelo.contribuicao_total(ajuste.theta, d)[0]
    assert np.all(baixo <= estimativa) and np.all(estimativa <= alto)
    assert np.allclose(replicas, modelo.bootstrap_em_blocos(ajuste, repeticoes=60, semente=1))  # reprodutível


def test_intervalo_percentil_usa_as_posicoes_k_sobre_b_mais_1():
    amostras = np.arange(1, 200, dtype=float)[:, None]  # 199 réplicas: a posição 0,025 * 200 = 5 é a 5ª menor
    baixo, alto = modelo.intervalo(amostras, 0.95)
    assert baixo[0] == pytest.approx(5.0) and alto[0] == pytest.approx(195.0)
    baixo, alto = modelo.intervalo(amostras, 0.80)
    assert baixo[0] == pytest.approx(20.0) and alto[0] == pytest.approx(180.0)
