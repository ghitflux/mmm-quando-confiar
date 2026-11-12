import numpy as np
import pytest

from mmm_confiar import dados
from mmm_confiar import transformacoes as tr

CENTRAL = dados.Cenario("central")


def _colunas_de_gasto(serie: dados.Serie) -> np.ndarray:
    return serie.gasto


def test_a_mesma_semente_gera_a_mesma_serie_e_sementes_diferentes_nao():
    a, b, c = dados.gerar(CENTRAL, 3), dados.gerar(CENTRAL, 3), dados.gerar(CENTRAL, 4)
    assert a.tabela.equals(b.tabela)
    assert not np.allclose(a.vendas, c.vendas)


def test_vendas_sao_a_soma_dos_componentes_do_gabarito():
    serie = dados.gerar(CENTRAL, 0)
    partes = serie.componentes.drop(columns="semana").sum(axis=1).to_numpy()
    assert np.allclose(serie.vendas, partes)
    assert len(serie.tabela) == 104 and (serie.gasto > 0).all()
    assert serie.tabela["promocao"].sum() == 10  # cinco semanas de promoção por ano
    assert np.allclose(serie.roi_verdadeiro, serie.contribuicao_verdadeira / serie.gasto.sum(axis=0))


def test_contribuicao_do_gabarito_e_hill_do_adstock_com_os_parametros_do_mundo():
    serie = dados.gerar(CENTRAL, 1)
    mais_longa = dados.gerar(CENTRAL.com(semanas=104 + tr.JANELA - 1), 1)  # traz as semanas anteriores que alimentam o adstock
    mundo = serie.mundo
    for c, canal in enumerate(dados.CANAIS):
        gasto = mais_longa.gasto[:, c]
        adstock = tr.adstock_geometrico(gasto[tr.JANELA - 1:], mundo.retencao[c], anterior=gasto[: tr.JANELA - 1])
        esperado = mundo.venda_no_gasto_medio[c] * tr.hill_normalizada(adstock / mundo.gasto_referencia[c], mundo.meia_saturacao[c])
        assert np.allclose(serie.componentes[f"venda_{canal}"], esperado)


def test_resposta_estacionaria_no_gasto_de_referencia_e_a_venda_no_gasto_medio():
    for forma_errada in (False, True):
        mundo = dados.sortear_mundo(5, forma_errada)
        assert np.allclose(mundo.resposta_por_canal(mundo.gasto_referencia), mundo.venda_no_gasto_medio)
        assert mundo.resposta(mundo.gasto_referencia) == pytest.approx(mundo.venda_no_gasto_medio.sum())
        assert np.all(mundo.resposta_por_canal(2 * mundo.gasto_referencia) > mundo.venda_no_gasto_medio)  # mais gasto, mais venda
        assert np.all(mundo.resposta_por_canal(2 * mundo.gasto_referencia) < 2 * mundo.venda_no_gasto_medio) or forma_errada  # mas menos que o dobro


def test_historico_curto_e_o_fim_do_historico_longo():
    curta, longa = dados.gerar(CENTRAL.com(semanas=52), 2), dados.gerar(CENTRAL.com(semanas=208), 2)
    assert np.allclose(curta.gasto, longa.gasto[-52:])
    assert np.allclose(curta.vendas, longa.vendas[-52:])
    assert curta.tabela["semana"].iloc[-1] == longa.tabela["semana"].iloc[-1]


def test_colinearidade_do_cenario_aparece_na_correlacao_do_gasto():
    correlacoes = []
    for colinearidade in (0.1, 0.5, 0.9):
        cenario = CENTRAL.com(colinearidade=colinearidade, semanas=208)
        correlacoes.append(np.mean([dados.correlacao_media_do_gasto(dados.gerar(cenario, s)) for s in range(6)]))
    assert correlacoes[0] < 0.2 < correlacoes[1] < 0.7 < correlacoes[2]


def test_so_no_cenario_endogeno_o_gasto_de_busca_acompanha_o_choque_de_demanda():
    def correlacao(cenario: dados.Cenario) -> float:
        valores = []
        for semente in range(8):
            serie = dados.gerar(cenario.com(semanas=208, colinearidade=0.1), semente)
            valores.append(np.corrcoef(np.log(serie.tabela["gasto_busca"]), serie.componentes["choque"])[0, 1])
        return float(np.mean(valores))

    assert abs(correlacao(CENTRAL)) < 0.15
    assert correlacao(CENTRAL.com(busca_responde_a_demanda=True)) > 0.5
    # o mundo e as vendas dos outros canais não mudam: só o gasto de busca responde
    sem, com = dados.gerar(CENTRAL, 0), dados.gerar(CENTRAL.com(busca_responde_a_demanda=True), 0)
    assert np.allclose(sem.tabela["gasto_social"], com.tabela["gasto_social"])
    assert not np.allclose(sem.tabela["gasto_busca"], com.tabela["gasto_busca"])


def test_variacao_deliberada_corta_e_reforca_cada_canal_em_janelas_separadas():
    sem, com = dados.gerar(CENTRAL, 4), dados.gerar(CENTRAL.com(variacao_deliberada=True), 4)
    razao = com.gasto / sem.gasto
    todas = []
    for c, canal in enumerate(dados.CANAIS):
        corte = com.teste[canal]
        assert len(corte) == dados.SEMANAS_DO_TESTE
        assert np.allclose(razao[corte, c], dados.CORTE_NO_TESTE)
        assert np.isclose(razao[:, c], dados.REFORCO_NO_TESTE).sum() == dados.SEMANAS_DO_TESTE
        assert np.isclose(razao[:, c], 1.0).sum() == 104 - 2 * dados.SEMANAS_DO_TESTE
        todas.extend(np.flatnonzero(~np.isclose(razao[:, c], 1.0)).tolist())
    assert len(todas) == len(set(todas))  # nenhuma semana com dois canais em teste
    assert sem.teste == {}


def test_forma_errada_muda_a_forma_e_ruido_menor_encolhe_o_ruido():
    certo, errado = dados.sortear_mundo(0), dados.sortear_mundo(0, forma_errada=True)
    assert np.all(certo.expoente == 1) and np.all(errado.expoente == 2)
    assert np.all(certo.pico == 0) and errado.pico.tolist() == [0, 1, 2, 0]
    assert np.allclose(certo.venda_no_gasto_medio, errado.venda_no_gasto_medio)  # o tamanho do efeito é o mesmo
    normal, calmo = dados.gerar(CENTRAL, 0), dados.gerar(CENTRAL.com(ruido=0.25), 0)
    assert np.allclose(calmo.componentes["ruido"], 0.25 * normal.componentes["ruido"])
    assert np.allclose(calmo.componentes["choque"], 0.25 * normal.componentes["choque"])


def test_ordens_de_grandeza_do_mundo_de_referencia():
    series = [dados.gerar(CENTRAL, s) for s in range(10)]
    parte_da_midia = np.mean([s.contribuicao_verdadeira.sum() / s.vendas.sum() for s in series])
    gasto_sobre_vendas = np.mean([s.gasto.sum() / s.vendas.sum() for s in series])
    ruido = np.mean([(s.componentes["choque"] + s.componentes["ruido"]).std() / s.vendas.mean() for s in series])
    assert 0.12 < parte_da_midia < 0.25
    assert 0.06 < gasto_sobre_vendas < 0.13
    assert 0.015 < ruido < 0.035
    gabarito = dados.gabarito(series[0])
    assert set(gabarito["canais"]) == set(dados.CANAIS)
    assert gabarito["canais"]["afiliados"]["roi_verdadeiro"] > gabarito["canais"]["video"]["roi_verdadeiro"]


def test_semanas_fora_da_faixa_dao_erro():
    with pytest.raises(ValueError):
        dados.gerar(CENTRAL.com(semanas=300), 0)
