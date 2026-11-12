import copy

import pytest

from mmm_confiar import analise, site


def _resumo_fabricado() -> dict:
    """Um resumo com a mesma forma do que `analise.metricas` devolve, com valores inventados."""
    metricas = (
        "mmm_erro_contribuicao", "linear_erro_contribuicao", "proporcional_erro_contribuicao", "mmm_pior_que_nao_mexer", "linear_pior_que_nao_mexer",
        "mmm_ganho", "ganho_possivel", "mmm_cobertura_95", "mmm_cobertura_80",
    )
    resumo = {}
    for i, item in enumerate(analise.CENARIOS):
        valor = 0.1 + 0.05 * i
        resumo[item.chave] = {
            "rotulo": item.rotulo,
            "metricas": {m: {"media": valor, "inferior": valor - 0.05, "superior": valor + 0.05} for m in metricas},
            "por_canal": {"mmm": {canal: {"erro_com_sinal": {"media": valor - 0.3 * j}} for j, canal in enumerate(site.CANAIS_DO_SITE)}},
        }
    return resumo


def _exemplo_fabricado() -> dict:
    x = [round(0.05 * i, 2) for i in range(41)]
    verdadeira = [60 * v / (v + 0.8) * 1.8 for v in x]
    estimada = [1.5 * y for y in verdadeira]
    curva = {"gasto_sobre_a_media": x, "verdadeira": verdadeira, "estimada": estimada,
             "inferior": [0.5 * y for y in estimada], "superior": [1.5 * y for y in estimada]}
    return {"curvas": {canal: curva for canal in site.CANAIS_DO_SITE}}


def test_graficos_extras_tem_a_forma_que_o_site_espera():
    resumo = _resumo_fabricado()
    extras = site.graficos_extras(resumo, _exemplo_fabricado())
    site.validar_extras(extras)
    assert 3 <= len(extras) <= 5
    for grafico in extras:
        assert grafico["tipo"] in ("linhas", "barras", "pontos")
        for chave in ("titulo", "eixo_x", "eixo_y", "leitura"):
            assert set(grafico[chave]) == {"pt", "en", "es"}
        assert all(len(t) <= 90 for t in grafico["titulo"].values())
        assert len(grafico["series"]) <= 4
        for serie in grafico["series"]:
            assert len(serie["pontos"]) <= (300 if grafico["tipo"] == "pontos" else 80)
            if grafico["tipo"] == "barras":
                assert [p[0] for p in serie["pontos"]] == serie["rotulos"]["pt"]
    # os valores saem das métricas, não do texto
    erro = extras[0]["series"][0]["pontos"]
    assert [v for _, v in erro] == pytest.approx([resumo[c]["metricas"]["mmm_erro_contribuicao"]["media"] for c in site.CENARIOS_DAS_FIGURAS])
    assert extras[-1]["tipo"] == "linhas" and extras[-1]["faixa"]["serie"] == 0
    # inglês com ponto decimal, português e espanhol com vírgula
    assert "1.5" in extras[-1]["leitura"]["en"] and "1,5" in extras[-1]["leitura"]["pt"] and "1,5" in extras[-1]["leitura"]["es"]


def test_validar_extras_recusa_grafico_fora_das_regras():
    extras = site.graficos_extras(_resumo_fabricado(), _exemplo_fabricado())
    sem_leitura = copy.deepcopy(extras)
    del sem_leitura[0]["leitura"]["es"]
    with pytest.raises(AssertionError):
        site.validar_extras(sem_leitura)
    titulo_longo = copy.deepcopy(extras)
    titulo_longo[1]["titulo"]["en"] = "x" * 91
    with pytest.raises(AssertionError):
        site.validar_extras(titulo_longo)
    series_demais = copy.deepcopy(extras)
    series_demais[0]["series"] = series_demais[0]["series"] * 2
    with pytest.raises(AssertionError):
        site.validar_extras(series_demais)
    with pytest.raises(AssertionError):
        site.validar_extras(extras[:2])


def test_bloco_do_site_segue_as_regras_e_usa_os_numeros_do_resumo():
    resumo = _resumo_fabricado()
    bloco = site.montar(resumo, analise.premissas())
    site.validar(bloco)
    central = resumo["central"]["metricas"]["mmm_erro_contribuicao"]["media"]
    assert bloco["numero_chave"]["pt"].startswith(f"{100 * central:.0f}%")
    assert bloco["grafico"]["tipo"] == "barras" and len(bloco["grafico"]["series"]) == 2
    assert [p[0] for p in bloco["grafico"]["series"][0]["pontos"]] == site.ROTULOS["pt"]
    esperado = [resumo[c]["metricas"]["mmm_pior_que_nao_mexer"]["media"] for c in site.CENARIOS_DO_GRAFICO]
    assert [p[1] for p in bloco["grafico"]["series"][0]["pontos"]] == pytest.approx(esperado)
    assert "," not in bloco["ganho_central"]["en"] and "." not in bloco["ganho_central"]["pt"]


def test_validar_recusa_bloco_fora_das_regras():
    bloco = site.montar(_resumo_fabricado(), analise.premissas())
    longo = copy.deepcopy(bloco)
    longo["numero_chave"]["pt"] = "um número-chave comprido demais"
    with pytest.raises(AssertionError):
        site.validar(longo)
    fora = copy.deepcopy(bloco)
    fora["grafico"]["series"][0]["pontos"][0][1] = 1.4  # percentual tem de ficar entre 0 e 1
    with pytest.raises(AssertionError):
        site.validar(fora)
