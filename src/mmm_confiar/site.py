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


IDIOMAS = ("pt", "en", "es")
# Os nove cenários das figuras de erro e de cobertura do README, na mesma ordem.
CENARIOS_DAS_FIGURAS = ("laboratorio", "quatro_anos", "colinearidade_baixa", "central", "um_ano", "colinearidade_alta", "busca_demanda", "dificil", "forma_errada")
ROTULOS_DAS_FIGURAS = {
    "pt": ["Laboratório", "4 anos", "Colinearidade baixa", "Central (2 anos)", "1 ano", "Colinearidade alta", "Busca segue a demanda", "Difícil", "Forma errada"],
    "en": ["Lab conditions", "4 years", "Low collinearity", "Baseline (2 years)", "1 year", "High collinearity", "Search follows demand", "Hard case", "Wrong functional form"],
    "es": ["Laboratorio", "4 años", "Colinealidad baja", "Central (2 años)", "1 año", "Colinealidad alta", "Búsqueda sigue la demanda", "Difícil", "Forma errónea"],
}
CANAIS_DO_SITE = ("busca", "social", "video", "afiliados")
ROTULOS_DOS_CANAIS = {"pt": ["Busca", "Social", "Vídeo", "Afiliados"], "en": ["Search", "Social", "Video", "Affiliates"], "es": ["Búsqueda", "Social", "Video", "Afiliados"]}


def _t(pt: str, en: str, es: str) -> dict:
    return {"pt": pt, "en": en, "es": es}


def _barras_por_cenario(m: dict, chaves: tuple[str, ...], metrica: str) -> list[list]:
    return [[rotulo, round(m[chave][metrica]["media"], 4)] for rotulo, chave in zip(ROTULOS_DAS_FIGURAS["pt"], chaves)]


def graficos_extras(resumo: dict, exemplo: dict) -> list[dict]:
    """Os gráficos que equivalem às outras figuras do README, do achado mais importante para o menos.

    Tudo sai das métricas dos cenários (`resumo`) e da série de exemplo (`exemplo`), as mesmas que desenham os PNG.
    """
    m = {chave: bloco["metricas"] for chave, bloco in resumo.items()}
    rotulos = {idioma: list(ROTULOS_DAS_FIGURAS[idioma]) for idioma in IDIOMAS}
    eixo_cenario = _t("Cenário", "Scenario", "Escenario")

    # 1. Erro de contribuição por cenário (figura erro_por_cenario.png)
    erro_central, erro_linear = m["central"]["mmm_erro_contribuicao"]["media"], m["central"]["linear_erro_contribuicao"]["media"]
    erro_lab = m["laboratorio"]["mmm_erro_contribuicao"]["media"]
    erro = {
        "tipo": "barras",
        "titulo": _t(
            "Fora do laboratório, o MMM erra tanto quanto as alternativas simples, ou mais",
            "Outside lab conditions, the MMM is as wrong as the simple alternatives, or worse",
            "Fuera del laboratorio, el MMM se equivoca tanto como las alternativas simples, o más",
        ),
        "eixo_x": eixo_cenario,
        "eixo_y": _t("Venda atribuída ao canal errado, em % da venda de mídia", "Sales credited to the wrong channel, % of media sales",
                     "Venta atribuida al canal equivocado, en % de la venta de medios"),
        "formato_x": "numero", "formato_y": "percentual", "rotulos": rotulos,
        "series": [
            {"nome": _t("MMM", "MMM", "MMM"), "papel": "principal", "pontos": _barras_por_cenario(m, CENARIOS_DAS_FIGURAS, "mmm_erro_contribuicao")},
            {"nome": _t("Regressão linear", "Linear regression", "Regresión lineal"), "papel": "base", "pontos": _barras_por_cenario(m, CENARIOS_DAS_FIGURAS, "linear_erro_contribuicao")},
            {"nome": _t("Proporcional ao gasto", "Proportional to spend", "Proporcional al gasto"), "papel": "alternativa",
             "pontos": _barras_por_cenario(m, CENARIOS_DAS_FIGURAS, "proporcional_erro_contribuicao")},
        ],
        "leitura": _t(
            f"Média de 48 séries por cenário. No laboratório o MMM erra {_p(erro_lab)}%; no cenário central erra {_p(erro_central)}%, contra {_p(erro_linear)}% da regressão linear.",
            f"Average of 48 series per scenario. Under lab conditions the MMM is off by {_p(erro_lab)}%; in the baseline it is off by {_p(erro_central)}%, against {_p(erro_linear)}% for a linear regression.",
            f"Promedio de 48 series por escenario. En el laboratorio el MMM se equivoca un {_p(erro_lab)}%; en el central, un {_p(erro_central)}%, frente al {_p(erro_linear)}% de la regresión lineal.",
        ),
    }

    # 2. Cobertura dos intervalos (figura cobertura.png)
    cob_central, cob_dificil = m["central"]["mmm_cobertura_95"]["media"], m["dificil"]["mmm_cobertura_95"]["media"]
    cobertura = {
        "tipo": "barras",
        "titulo": _t(
            "Os intervalos só cumprem o prometido quando o modelo descreve o mundo certo",
            "The intervals only deliver what they promise when the model matches the world",
            "Los intervalos solo cumplen lo prometido cuando el modelo describe bien el mundo",
        ),
        "eixo_x": eixo_cenario,
        "eixo_y": _t("Canais com o efeito verdadeiro dentro do intervalo", "Channels whose true effect falls inside the interval",
                     "Canales con el efecto real dentro del intervalo"),
        "formato_x": "numero", "formato_y": "percentual", "rotulos": rotulos,
        "series": [
            {"nome": _t("Intervalo de 95%", "95% interval", "Intervalo del 95%"), "papel": "principal", "pontos": _barras_por_cenario(m, CENARIOS_DAS_FIGURAS, "mmm_cobertura_95")},
            {"nome": _t("Intervalo de 80%", "80% interval", "Intervalo del 80%"), "papel": "alternativa", "pontos": _barras_por_cenario(m, CENARIOS_DAS_FIGURAS, "mmm_cobertura_80")},
        ],
        "leitura": _t(
            f"O intervalo de 95% devia cobrir 95%: cobre {_p(cob_central)}% no cenário central, mas só {_p(cob_dificil)}% quando a busca segue a demanda e os canais andam juntos.",
            f"A 95% interval should cover 95%: it covers {_p(cob_central)}% in the baseline, but only {_p(cob_dificil)}% when search follows demand and channels move together.",
            f"El intervalo del 95% debería cubrir el 95%: cubre el {_p(cob_central)}% en el central, pero solo el {_p(cob_dificil)}% cuando la búsqueda sigue la demanda y los canales se mueven juntos.",
        ),
    }

    # 3. Crédito por canal (figura credito_por_canal.png): erro com sinal, por canal, em três cenários
    def credito(chave: str) -> list[list]:
        por_canal = resumo[chave]["por_canal"]["mmm"]
        return [[rotulo, round(por_canal[canal]["erro_com_sinal"]["media"], 4)] for rotulo, canal in zip(ROTULOS_DOS_CANAIS["pt"], CANAIS_DO_SITE)]

    busca_demanda = resumo["busca_demanda"]["por_canal"]["mmm"]["busca"]["erro_com_sinal"]["media"]
    busca_dificil = resumo["dificil"]["por_canal"]["mmm"]["busca"]["erro_com_sinal"]["media"]
    canais = {
        "tipo": "barras",
        "titulo": _t(
            f"Se a busca segue a demanda, o MMM dá a ela {_p(busca_dificil)}% de crédito a mais",
            f"When search follows demand, the MMM gives it {_p(busca_dificil)}% too much credit",
            f"Si la búsqueda sigue la demanda, el MMM le da un {_p(busca_dificil)}% de crédito de más",
        ),
        "eixo_x": _t("Canal", "Channel", "Canal"),
        "eixo_y": _t("Crédito a mais (+) ou a menos (−) que a venda gerada", "Credit above (+) or below (−) the sales generated",
                     "Crédito de más (+) o de menos (−) frente a la venta generada"),
        "formato_x": "numero", "formato_y": "percentual", "rotulos": {idioma: list(ROTULOS_DOS_CANAIS[idioma]) for idioma in IDIOMAS},
        "series": [
            {"nome": _t("Central", "Baseline", "Central"), "papel": "neutro", "pontos": credito("central")},
            {"nome": _t("Busca segue a demanda", "Search follows demand", "Búsqueda sigue la demanda"), "papel": "alternativa", "pontos": credito("busca_demanda")},
            {"nome": _t("Difícil (+ canais colineares)", "Hard case (+ collinear channels)", "Difícil (+ canales colineales)"), "papel": "principal", "pontos": credito("dificil")},
        ],
        "leitura": _t(
            f"Erro médio da contribuição de cada canal. Quando o gasto de busca sobe junto com a procura, a busca recebe {_p(busca_demanda)}% a mais; com os canais colineares, {_p(busca_dificil)}%, tirados de social e vídeo.",
            f"Average error in each channel's contribution. When search spend rises with demand, search gets {_p(busca_demanda)}% too much; with collinear channels, {_p(busca_dificil)}%, taken from social and video.",
            f"Error medio de la contribución de cada canal. Cuando el gasto en búsqueda sube con la demanda, la búsqueda recibe un {_p(busca_demanda)}% de más; con canales colineales, un {_p(busca_dificil)}%, quitado a social y video.",
        ),
    }

    # 4. O que conserta (figura conserto.png): erro do MMM, só histórico x variação deliberada x experimento
    def conserto(sufixo: str) -> list[list]:
        return [[rotulo, round(m[f"{base}{sufixo}"]["mmm_erro_contribuicao"]["media"], 4)] for rotulo, base in zip(("Central", "Difícil"), ("central", "dificil"))]

    erro_exper = m["central_experimento"]["mmm_erro_contribuicao"]["media"]
    ganho_dificil_exper = m["dificil_experimento"]["mmm_ganho"]["media"]
    consertos = {
        "tipo": "barras",
        "titulo": _t(
            f"Experimento ou variação deliberada de gasto derrubam o erro de {_p(erro_central)}% para cerca de {_p(erro_exper)}%",
            f"An experiment or deliberate spend variation cuts the error from {_p(erro_central)}% to about {_p(erro_exper)}%",
            f"Un experimento o una variación deliberada del gasto bajan el error del {_p(erro_central)}% a cerca del {_p(erro_exper)}%",
        ),
        "eixo_x": eixo_cenario,
        "eixo_y": _t("Erro de contribuição do MMM", "MMM contribution error", "Error de contribución del MMM"),
        "formato_x": "numero", "formato_y": "percentual",
        "rotulos": _t(["Central", "Difícil"], ["Baseline", "Hard case"], ["Central", "Difícil"]),
        "series": [
            {"nome": _t("Só histórico", "History only", "Solo histórico"), "papel": "base", "pontos": conserto("")},
            {"nome": _t("Com variação deliberada", "With deliberate variation", "Con variación deliberada"), "papel": "alternativa", "pontos": conserto("_variacao")},
            {"nome": _t("Calibrado por experimento", "Calibrated with an experiment", "Calibrado con un experimento"), "papel": "principal", "pontos": conserto("_experimento")},
        ],
        "leitura": _t(
            f"Medir melhora mais do que decidir: no cenário difícil, mesmo calibrado, realocar pelo modelo ainda muda a venda de mídia em {_sinal(ganho_dificil_exper, 'pt')}.",
            f"Measurement improves more than decisions do: in the hard case, even after calibration, reallocating on the model's advice changes media sales by {_sinal(ganho_dificil_exper, 'en')}.",
            f"La medición mejora más que la decisión: en el escenario difícil, aun calibrado, reasignar según el modelo cambia la venta de medios en {_sinal(ganho_dificil_exper, 'es')}.",
        ),
    }

    # 5. Curva de resposta de uma série de exemplo (figura curvas_do_exemplo.png), canal de busca
    curva = exemplo["curvas"]["busca"]
    x = curva["gasto_sobre_a_media"]

    def linha(valores: list[float]) -> list[list[float]]:
        return [[round(float(a), 3), round(float(b), 2)] for a, b in zip(x, valores)]

    i_media = min(range(len(x)), key=lambda i: abs(x[i] - 1.0))
    razao = curva["estimada"][i_media] / curva["verdadeira"][i_media]
    resposta = {
        "tipo": "linhas",
        "titulo": _t(
            "A contribuição de um canal é uma extrapolação até o gasto zero, onde não há dado",
            "A channel's contribution is an extrapolation down to zero spend, where there is no data",
            "La contribución de un canal es una extrapolación hasta el gasto cero, donde no hay datos",
        ),
        "eixo_x": _t("Gasto semanal em busca, em múltiplos da média", "Weekly search spend, as a multiple of the average", "Gasto semanal en búsqueda, en múltiplos de la media"),
        "eixo_y": _t("Venda semanal gerada pela busca (R$ mil, simulada)", "Weekly sales from search (simulated, thousand BRL)", "Venta semanal generada por búsqueda (miles de R$, simulada)"),
        "formato_x": "numero", "formato_y": "numero",
        "series": [
            {"nome": _t("Estimada pelo MMM", "MMM estimate", "Estimada por el MMM"), "papel": "principal", "pontos": linha(curva["estimada"])},
            {"nome": _t("Verdadeira", "True curve", "Verdadera"), "papel": "neutro", "pontos": linha(curva["verdadeira"])},
        ],
        "faixa": {"serie": 0, "inferior": linha(curva["inferior"]), "superior": linha(curva["superior"])},
        "leitura": _t(
            f"Uma série do cenário central. Os dados só mostram a curva perto do gasto de hoje; no gasto médio o modelo diz {_n(razao, 1, 'pt')} vezes a venda verdadeira, e a faixa de 95% é larga.",
            f"One baseline series. The data only show the curve near current spend; at average spend the model says {_n(razao, 1, 'en')} times the true sales, and the 95% band is wide.",
            f"Una serie del escenario central. Los datos solo muestran la curva cerca del gasto actual; en el gasto medio el modelo dice {_n(razao, 1, 'es')} veces la venta real, y la banda del 95% es amplia.",
        ),
    }
    extras = [erro, cobertura, canais, consertos, resposta]
    for grafico in extras:  # o pedido do site quer os rótulos traduzidos dentro de cada série de barras
        if grafico["tipo"] == "barras":
            for serie in grafico["series"]:
                serie["rotulos"] = grafico["rotulos"]
    return extras


def validar_extras(extras: list[dict]) -> None:
    """Confere as regras dos gráficos extras: 3 a 5, tipos aceitos, limites de séries e pontos, três idiomas."""
    assert 3 <= len(extras) <= 5
    for grafico in extras:
        assert grafico["tipo"] in ("linhas", "barras", "pontos")
        for chave in ("titulo", "eixo_x", "eixo_y", "leitura"):
            assert set(grafico[chave]) == set(IDIOMAS) and all(grafico[chave][i].strip() for i in IDIOMAS), chave
        assert all(len(texto) <= 90 for texto in grafico["titulo"].values()), grafico["titulo"]
        assert grafico["formato_x"] in ("numero", "percentual", "data", "moeda") and grafico["formato_y"] in ("numero", "percentual", "data", "moeda")
        assert 1 <= len(grafico["series"]) <= 4
        limite = 300 if grafico["tipo"] == "pontos" else 80
        for serie in grafico["series"]:
            assert set(serie["nome"]) == set(IDIOMAS)
            assert serie["papel"] in ("principal", "base", "alternativa", "neutro")
            assert 1 <= len(serie["pontos"]) <= limite
            for ponto in serie["pontos"]:
                assert len(ponto) == 2 and isinstance(ponto[1], (int, float))
                if grafico["tipo"] == "barras":
                    assert isinstance(ponto[0], str)
                else:
                    assert isinstance(ponto[0], (int, float))
            if grafico["tipo"] == "barras":
                assert set(serie["rotulos"]) == set(IDIOMAS)
                assert all(len(serie["rotulos"][idioma]) == len(serie["pontos"]) for idioma in IDIOMAS)
        if "faixa" in grafico:
            faixa = grafico["faixa"]
            assert grafico["tipo"] == "linhas" and 0 <= faixa["serie"] < len(grafico["series"])
            assert len(faixa["inferior"]) == len(faixa["superior"]) == len(grafico["series"][faixa["serie"]]["pontos"])


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
