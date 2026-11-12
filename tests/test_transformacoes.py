import numpy as np
import pytest

from mmm_confiar import transformacoes as tr


def test_pesos_geometricos_somam_1_e_caem_em_progressao_geometrica():
    pesos = tr.pesos_geometricos(0.5)
    assert pesos.sum() == pytest.approx(1.0)
    assert np.allclose(pesos[1:] / pesos[:-1], 0.5)
    assert np.allclose(tr.pesos_geometricos(0.0), np.eye(tr.JANELA)[0])  # sem retenção: tudo na semana do gasto
    lote = tr.pesos_geometricos(np.array([[0.1, 0.4], [0.7, 0.9]]))
    assert lote.shape == (2, 2, tr.JANELA) and np.allclose(lote.sum(axis=-1), 1.0)


def test_adstock_preserva_o_total_de_um_gasto_isolado():
    gasto = np.zeros(40)
    gasto[10] = 100.0
    for retencao in (0.0, 0.3, 0.8):
        adstock = tr.adstock_geometrico(gasto, retencao, anterior=0.0)
        assert adstock.sum() == pytest.approx(100.0)
        assert np.all(adstock[:10] == 0)  # o efeito não começa antes do gasto
        assert np.all(np.diff(adstock[10: 10 + tr.JANELA]) <= 0)  # e só cai depois dele


def test_adstock_de_gasto_constante_e_o_proprio_gasto():
    gasto = np.full(30, 7.0)
    assert np.allclose(tr.adstock_geometrico(gasto, 0.6), 7.0)  # o que veio antes é suposto igual à média
    assert np.allclose(tr.aplicar_adstock(gasto, tr.pesos_atrasados(0.6, 2), anterior=7.0), 7.0)


def test_adstock_sem_retencao_devolve_o_gasto_e_mais_retencao_atrasa_o_efeito():
    gerador = np.random.default_rng(0)
    gasto = gerador.uniform(1, 10, 50)
    assert np.allclose(tr.adstock_geometrico(gasto, 0.0), gasto)
    pulso = np.zeros(40)
    pulso[5] = 1.0
    semanas = np.arange(40)
    atraso = [float((tr.adstock_geometrico(pulso, r, anterior=0.0) * semanas).sum() - 5) for r in (0.1, 0.4, 0.8)]
    assert atraso[0] < atraso[1] < atraso[2]
    assert atraso[1] == pytest.approx((tr.pesos_geometricos(0.4) * np.arange(tr.JANELA)).sum())


def test_adstock_confere_com_a_conta_feita_a_mao():
    # retenção 0,5 e janela de 3 semanas: pesos 4/7, 2/7 e 1/7
    gasto = np.array([7.0, 0.0, 14.0, 0.0])
    assert np.allclose(tr.adstock_geometrico(gasto, 0.5, janela=3, anterior=0.0), [4.0, 2.0, 9.0, 4.0])


def test_matriz_de_defasagens_poe_na_coluna_k_o_gasto_de_k_semanas_antes():
    matriz = tr.matriz_de_defasagens(np.array([1.0, 2.0, 3.0, 4.0]), janela=3, anterior=np.array([8.0, 9.0]))
    assert np.allclose(matriz, [[1, 9, 8], [2, 1, 9], [3, 2, 1], [4, 3, 2]])
    with pytest.raises(ValueError):
        tr.matriz_de_defasagens(np.ones(5), janela=4, anterior=np.array([1.0]))


def test_derivada_dos_pesos_confere_com_diferencas_finitas():
    retencao, passo = np.array([0.0, 0.25, 0.6, 0.9]), 1e-6
    numerica = (tr.pesos_geometricos(retencao + passo) - tr.pesos_geometricos(np.maximum(retencao - passo, 0))) / (retencao + passo - np.maximum(retencao - passo, 0))[:, None]
    assert np.allclose(tr.derivada_dos_pesos_geometricos(retencao), numerica, atol=1e-5)
    assert np.allclose(tr.derivada_dos_pesos_geometricos(retencao).sum(axis=-1), 0.0, atol=1e-12)  # os pesos somam sempre 1


def test_pesos_atrasados_tem_pico_na_semana_pedida():
    pesos = tr.pesos_atrasados(0.6, pico=2)
    assert pesos.sum() == pytest.approx(1.0)
    assert int(np.argmax(pesos)) == 2
    assert np.allclose(tr.pesos_atrasados(0.6, pico=0)[:3] / tr.pesos_atrasados(0.6, pico=0)[0], [1.0, 0.6, 0.6**4])


def test_hill_e_monotona_satura_e_vale_meio_na_meia_saturacao():
    x = np.linspace(0, 50, 500)
    for expoente in (1.0, 2.0):
        curva = tr.hill(x, 2.0, expoente)
        assert curva[0] == 0.0
        assert np.all(np.diff(curva) > 0)  # sempre cresce
        assert curva[-1] < 1.0 and tr.hill(1e9, 2.0, expoente) == pytest.approx(1.0)  # com teto em 1
        assert tr.hill(2.0, 2.0, expoente) == pytest.approx(0.5)
    assert np.all(np.diff(tr.hill(x, 2.0, 1.0), 2) < 0)  # expoente 1: retorno decrescente desde o início
    em_s = np.diff(tr.hill(x, 2.0, 2.0), 2)
    assert em_s[0] > 0 and em_s[-1] < 0  # expoente 2: acelera e depois freia


def test_hill_normalizada_vale_1_no_gasto_medio_e_mais_k_deixa_a_curva_mais_reta():
    for k in (0.3, 1.0, 4.0):
        assert tr.hill_normalizada(1.0, k) == pytest.approx(1.0)
        assert tr.hill_normalizada(1.0, k, 2.0) == pytest.approx(1.0)
    # dobrar o gasto: com K pequeno quase nada muda; com K grande a venda quase dobra
    assert tr.hill_normalizada(2.0, 0.2) < 1.15
    assert tr.hill_normalizada(2.0, 50.0) > 1.9
    # teto da curva: (1 + K) vezes a venda no gasto médio
    assert tr.hill_normalizada(1e9, 0.5) == pytest.approx(1.5)
