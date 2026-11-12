"""Bloco `para_o_site` do resultados.json: o resumo trilíngue e o gráfico que o site do portfólio desenha."""

from __future__ import annotations

CENARIOS_DO_GRAFICO = ("laboratorio", "quatro_anos", "central", "um_ano", "colinearidade_alta", "busca_demanda", "dificil")
ROTULOS = {
    "pt": ["Laboratório", "4 anos", "Central (2 anos)", "1 ano", "Colinearidade alta", "Busca segue a demanda", "Difícil"],
    "en": ["Lab conditions", "4 years", "Baseline (2 years)", "1 year", "High collinearity", "Search follows demand", "Hard case"],
    "es": ["Laboratorio", "4 años", "Central (2 años)", "1 año", "Colinealidad alta", "Búsqueda sigue la demanda", "Difícil"],
}


def _p(valor: float, casas: int = 0) -> str:
    """Percentual sem separador decimal ambíguo (0 casas) ou com vírgula/ponto tratado por quem chama."""
    return f"{100 * valor:.{casas}f}"


def _n(valor: float, casas: int, idioma: str) -> str:
    texto = f"{valor:.{casas}f}"
    if idioma != "en":
        texto = texto.replace(".", ",")
    return texto.replace("-", "−")


def _sinal(valor: float, idioma: str) -> str:
    return ("+" if valor > 0 else "") + _n(100 * valor, 1, idioma) + "%"


def montar(resumo: dict, premissas: dict) -> dict:
    """Monta o bloco a partir das métricas dos cenários. Todo número do texto sai daqui."""
    m = {chave: bloco["metricas"] for chave, bloco in resumo.items()}
    central, exper, lab, dificil = m["central"], m["central_experimento"], m["laboratorio"], m["dificil"]
    erro, erro_exper, erro_lab = central["mmm_erro_contribuicao"]["media"], exper["mmm_erro_contribuicao"]["media"], lab["mmm_erro_contribuicao"]["media"]
    erro_linear = central["linear_erro_contribuicao"]["media"]
    pior = central["mmm_pior_que_nao_mexer"]
    ganho, possivel = central["mmm_ganho"]["media"], central["ganho_possivel"]["media"]
    cob_central, cob_dificil = central["mmm_cobertura_95"]["media"], dificil["mmm_cobertura_95"]["media"]
    pior_dificil = dificil["mmm_pior_que_nao_mexer"]["media"]
    sementes, replicas = premissas["sementes_por_cenario"], premissas["replicas_do_bootstrap"]
    cenarios = len(resumo)

    def pontos(metrica: str, idioma: str) -> list[list]:
        return [[rotulo, round(m[chave][metrica]["media"], 4)] for rotulo, chave in zip(ROTULOS[idioma], CENARIOS_DO_GRAFICO)]

    return {
        "numero_chave": {"pt": f"{_p(erro)}% → {_p(erro_exper)}%", "en": f"{_p(erro)}% → {_p(erro_exper)}%", "es": f"{_p(erro)}% → {_p(erro_exper)}%"},
        "pergunta": {
            "pt": "O modelo de mix de marketing recupera o efeito de cada canal? Em que condições não, e quanto custa realocar a verba com base nele?",
            "en": "Does a marketing mix model recover what each channel really did? When does it fail, and what does reallocating budget on its advice cost?",
            "es": "¿El modelo de mix de marketing recupera el efecto de cada canal? ¿Cuándo no, y cuánto cuesta reasignar el presupuesto según lo que dice?",
        },
        "resultado": {
            "pt": (
                f"Num mundo simulado em que o efeito verdadeiro é conhecido, o MMM só acerta em condições de laboratório (erro de {_p(erro_lab)}% da venda de mídia). "
                f"Com dois anos de dados típicos ele erra {_p(erro)}%, empata com uma regressão linear ({_p(erro_linear)}%), e realocar a verba seguindo o modelo vende menos do que "
                f"não mexer em {_p(pior['media'])}% das simulações (IC 95%: {_p(pior['inferior'])}% a {_p(pior['superior'])}%)."
            ),
            "en": (
                f"In a simulated world where the true effect is known, the MMM is only accurate under lab conditions (error of {_p(erro_lab)}% of media-driven sales). "
                f"With two years of typical data it is off by {_p(erro)}%, no better than a plain linear regression ({_p(erro_linear)}%), and moving budget where the model says sells less than "
                f"leaving it alone in {_p(pior['media'])}% of simulations (95% CI: {_p(pior['inferior'])}% to {_p(pior['superior'])}%)."
            ),
            "es": (
                f"En un mundo simulado donde el efecto real se conoce, el MMM solo acierta en condiciones de laboratorio (error del {_p(erro_lab)}% de la venta de medios). "
                f"Con dos años de datos típicos se equivoca en un {_p(erro)}%, empata con una regresión lineal ({_p(erro_linear)}%), y reasignar el presupuesto según el modelo vende menos que "
                f"no tocar nada en el {_p(pior['media'])}% de las simulaciones (IC 95%: {_p(pior['inferior'])}% a {_p(pior['superior'])}%)."
            ),
        },
        "detalhe": {
            "pt": (
                f"Calibrar o modelo com um experimento em dois canais derruba o erro para {_p(erro_exper)}%. Os intervalos de 95% cobrem {_p(cob_central)}% quando o modelo tem a forma certa, "
                f"mas só {_p(cob_dificil)}% quando o gasto de busca responde à demanda e os canais andam juntos; nesse caso seguir o modelo é pior do que não mexer em {_p(pior_dificil)}% das vezes."
            ),
            "en": (
                f"Calibrating the model with an experiment on two channels cuts the error to {_p(erro_exper)}%. The 95% intervals cover {_p(cob_central)}% when the model has the right form, "
                f"but only {_p(cob_dificil)}% when search spend reacts to demand and channels move together; there, following the model is worse than doing nothing {_p(pior_dificil)}% of the time."
            ),
            "es": (
                f"Calibrar el modelo con un experimento en dos canales baja el error al {_p(erro_exper)}%. Los intervalos del 95% cubren el {_p(cob_central)}% cuando el modelo tiene la forma correcta, "
                f"pero solo el {_p(cob_dificil)}% cuando el gasto en búsqueda responde a la demanda y los canales se mueven juntos; ahí seguir el modelo es peor que no tocar nada el {_p(pior_dificil)}% de las veces."
            ),
        },
        "dados": {
            "pt": f"Dados simulados (é o desenho do estudo): {cenarios} cenários, {sementes} séries semanais por cenário (1, 2 ou 4 anos, 4 canais), {replicas} réplicas de bootstrap por série.",
            "en": f"Simulated data (by design): {cenarios} scenarios, {sementes} weekly series per scenario (1, 2 or 4 years, 4 channels), {replicas} bootstrap replicates per series.",
            "es": f"Datos simulados (es el diseño del estudio): {cenarios} escenarios, {sementes} series semanales por escenario (1, 2 o 4 años, 4 canales), {replicas} réplicas de bootstrap por serie.",
        },
        "etiquetas": ["MMM", "adstock", "saturação", "simulação", "bootstrap em blocos", "otimização de verba", "experimentos"],
        "grafico": {
            "tipo": "barras",
            "titulo": {
                "pt": f"Seguir o modelo é pior do que não mexer na verba em {_p(pior['media'])}% das simulações do cenário central",
                "en": f"Following the model is worse than leaving the budget alone in {_p(pior['media'])}% of baseline simulations",
                "es": f"Seguir el modelo es peor que no tocar el presupuesto en el {_p(pior['media'])}% de las simulaciones del escenario central",
            },
            "eixo_x": {"pt": "Cenário", "en": "Scenario", "es": "Escenario"},
            "eixo_y": {
                "pt": "Simulações em que a realocação recomendada vende menos que a verba atual",
                "en": "Simulations where the recommended allocation sells less than the current one",
                "es": "Simulaciones en que la reasignación recomendada vende menos que el presupuesto actual",
            },
            "formato_x": "numero",
            "formato_y": "percentual",
            "rotulos": ROTULOS,
            "series": [
                {"nome": {"pt": "MMM", "en": "MMM", "es": "MMM"}, "papel": "principal", "pontos": pontos("mmm_pior_que_nao_mexer", "pt")},
                {"nome": {"pt": "Regressão linear", "en": "Linear regression", "es": "Regresión lineal"}, "papel": "base", "pontos": pontos("linear_pior_que_nao_mexer", "pt")},
            ],
        },
        "ganho_central": {"pt": _sinal(ganho, "pt"), "en": _sinal(ganho, "en"), "es": _sinal(ganho, "es"), "possivel": {"pt": _sinal(possivel, "pt"), "en": _sinal(possivel, "en")}},
    }


def validar(bloco: dict) -> None:
    """Confere as regras do bloco: idiomas, tamanho do número-chave, número de séries e de pontos."""
    for chave in ("numero_chave", "pergunta", "resultado", "detalhe", "dados"):
        assert set(bloco[chave]) == {"pt", "en", "es"}, chave
    assert all(len(texto) <= 12 for texto in bloco["numero_chave"].values())
    grafico = bloco["grafico"]
    assert grafico["tipo"] in ("linhas", "barras")
    assert 1 <= len(grafico["series"]) <= 4
    for serie in grafico["series"]:
        assert serie["papel"] in ("principal", "base", "alternativa", "neutro")
        assert 1 <= len(serie["pontos"]) <= 80
        if grafico["formato_y"] == "percentual":
            assert all(0 <= valor <= 1 for _, valor in serie["pontos"])
        if grafico["tipo"] == "barras" and "rotulos" in grafico:
            assert all(len(grafico["rotulos"][idioma]) == len(serie["pontos"]) for idioma in ("pt", "en", "es"))
