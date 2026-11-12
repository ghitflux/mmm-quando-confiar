import copy

import pytest

from mmm_confiar import analise, site


def _resumo_fabricado() -> dict:
    """Um resumo com a mesma forma do que `analise.metricas` devolve, com valores inventados."""
    metricas = (
        "mmm_erro_contribuicao", "linear_erro_contribuicao", "mmm_pior_que_nao_mexer", "linear_pior_que_nao_mexer",
        "mmm_ganho", "ganho_possivel", "mmm_cobertura_95",
    )
    resumo = {}
    for i, item in enumerate(analise.CENARIOS):
        valor = 0.1 + 0.05 * i
        resumo[item.chave] = {"rotulo": item.rotulo, "metricas": {m: {"media": valor, "inferior": valor - 0.05, "superior": valor + 0.05} for m in metricas}}
    return resumo


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
