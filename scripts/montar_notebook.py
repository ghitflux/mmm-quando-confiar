"""Monta analise.ipynb com nbformat. Depois de montar, execute:

    uv run python scripts/montar_notebook.py
    uv run jupyter nbconvert --to notebook --execute --inplace analise.ipynb

O notebook é uma narrativa fina: abre o conjunto de exemplo, ajusta o modelo nele e conta a história com as tabelas
de reports/resultados.json (o estudo de recuperação completo roda em scripts/rodar_analise.py, não aqui).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parents[1]

CELULAS: list[tuple[str, str]] = [
    ("md", """# O modelo de mix de marketing recupera o efeito de cada canal?

O gestor recebe um gráfico de "contribuição por canal" e realoca a verba. Este notebook mede quando esse gráfico
merece confiança, do único jeito em que isso pode ser medido: num mundo simulado, em que o efeito verdadeiro de cada
canal é conhecido.

**Os dados são simulados, de propósito.** Nenhum número aqui descreve uma operação real. O gerador está em
`src/mmm_confiar/dados.py`; o modelo, o bootstrap e a otimização, em `src/mmm_confiar/`, escritos em NumPy/SciPy e
conferidos em `tests/`."""),
    ("code", """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from mmm_confiar import analise, dados, graficos, linha_de_base, modelo, otimizacao

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.max_columns", 30)
resultados = json.loads(Path("reports/resultados.json").read_text(encoding="utf-8"))
cenarios = resultados["cenarios"]


def mostrar(figura):
    display(figura)
    plt.close(figura)


def tabela(chaves, metricas):
    linhas = {}
    for chave in chaves:
        m = cenarios[chave]["metricas"]
        linhas[cenarios[chave]["rotulo"]] = {nome: f"{m[k]['media']:.2f}  [{m[k]['inferior']:.2f}; {m[k]['superior']:.2f}]" for nome, k in metricas.items()}
    return pd.DataFrame.from_dict(linhas, orient="index")"""),
    ("md", """## 1. Um mundo de exemplo

Vendas semanais de um anunciante com quatro canais (R$ mil por semana). O que o analista vê é a primeira tabela; a
segunda é o gabarito, que só o gerador conhece: quanto da venda de cada semana veio da base, da promoção, de cada
canal e do acaso."""),
    ("code", """chave, semente = analise.EXEMPLO
serie = dados.gerar(analise.POR_CHAVE[chave].cenario, semente)
visto, gabarito = dados.carregar_exemplo()  # os mesmos dados, versionados em data/
assert np.allclose(visto["vendas"], serie.vendas, atol=1e-2)
display(visto.head())
display(gabarito.head())
pd.DataFrame(dados.gabarito(serie)["canais"]).T"""),
    ("md", """Os problemas da vida real estão no gerador: a verba inteira sobe na alta temporada (os gastos andam juntos), o
canal de afiliados quase não varia, e num dos cenários o gasto de busca responde à demanda."""),
    ("code", """gasto = visto[[f"gasto_{c}" for c in dados.CANAIS]]
display(gasto.corr().round(2))
(gasto.std() / gasto.mean()).rename("desvio / média do gasto").to_frame().T"""),
    ("md", """## 2. O modelo ajustado nesse exemplo

O MMM (adstock geométrico, saturação de Hill, efeitos não negativos, penalidade na forma da curva) é ajustado por
um Levenberg-Marquardt próprio, a partir de oito pontos de partida. A incerteza vem de 200 réplicas de bootstrap em
blocos. Ao lado, as duas alternativas simples."""),
    ("code", """exemplo = analise.exemplo(serie)
print(f"pontos de partida que chegaram ao mesmo ótimo: {exemplo['partidas_no_mesmo_otimo']:.0%}")
exemplo["tabela"][["venda verdadeira", "MMM", "MMM IC 95% inferior", "MMM IC 95% superior", "regressão linear", "proporcional ao gasto", "ROI verdadeiro", "ROI MMM"]]"""),
    ("md", """Nesta série o modelo deu a busca e a social o dobro do que elas geraram e zerou vídeo e afiliados. Os intervalos
são honestos (cobrem o valor verdadeiro), mas tão largos que não distinguem um canal bom de um ruim.

A figura mostra o porquê. Os dados só informam a inclinação da curva na faixa de gasto observada (fundo cinza). A
contribuição do canal é a altura da curva no gasto de hoje, medida a partir do gasto zero, onde não há dado."""),
    ("code", """mostrar(graficos.curvas_do_exemplo(exemplo))"""),
    ("md", """## 3. O estudo de recuperação

Uma série só não prova nada. `scripts/rodar_analise.py` repete o exercício em 15 cenários, com 48 séries em cada um
(a mesma semente dá o mesmo mundo e o mesmo ruído em todos os cenários). Em cada série mede:

- **erro de contribuição**: venda atribuída ao canal errado ou inventada, em fração da venda de mídia verdadeira;
- **pares invertidos**: parte dos pares de canais que o método ordena ao contrário pelo ROI (0,5 = cara ou coroa);
- **cobertura** dos intervalos de 80% e de 95%;
- **decisão**: o ganho de venda de mídia ao realocar a verba pelo método, avaliado nas curvas verdadeiras.

Os colchetes são o intervalo de 95% do erro de Monte Carlo (reamostrando as 48 sementes)."""),
    ("code", """observacionais = ["laboratorio", "quatro_anos", "colinearidade_baixa", "central", "um_ano", "colinearidade_alta", "busca_demanda", "dificil", "forma_errada"]
tabela(observacionais, {"MMM": "mmm_erro_contribuicao", "regressão linear": "linear_erro_contribuicao", "proporcional ao gasto": "proporcional_erro_contribuicao",
                        "MMM - linear": "mmm_menos_linear_erro_contribuicao"})"""),
    ("code", """mostrar(graficos.erro_por_cenario(cenarios))"""),
    ("md", """No laboratório (quatro anos, gastos que variam de forma independente, um quarto do ruído) o modelo recupera os
canais: é a prova de que o código está certo. Fora dele, o MMM não bate a regressão linear, e repartir o total que
ele mesmo estima na proporção do gasto erra menos do que a repartição do modelo.

A ordem dos canais conta a mesma história."""),
    ("code", """tabela(observacionais, {"pares invertidos: MMM": "mmm_pares_invertidos", "pares invertidos: linear": "linear_pares_invertidos",
                        "melhor canal errado: MMM": "mmm_melhor_canal_errado", "melhor canal errado: linear": "linear_melhor_canal_errado"})"""),
    ("md", """## 4. Os intervalos"""),
    ("code", """display(tabela(observacionais, {"cobertura 80%": "mmm_cobertura_80", "cobertura 95%": "mmm_cobertura_95", "largura do IC 95% / valor verdadeiro": "mmm_largura_95",
                                "cobertura 95% da linear": "linear_cobertura_95"}))
mostrar(graficos.cobertura(cenarios))"""),
    ("md", """Com a forma certa e sem o canal que responde à demanda, o bootstrap em blocos entrega a cobertura prometida. O
preço é a largura: no cenário central o intervalo de 95% tem cerca de três vezes o tamanho do efeito que mede.
Quando a busca responde à demanda, ou quando a forma da curva está errada, o intervalo fica confiante no lugar errado."""),
    ("code", """mostrar(graficos.credito_por_canal(cenarios))
pd.DataFrame({c: {canal: f"{b['cobertura_95']['media']:.0%}" for canal, b in cenarios[c]["por_canal"]["mmm"].items()} for c in ["central", "busca_demanda", "dificil"]})"""),
    ("md", """## 5. Da medição à decisão

A verba do mundo de referência está mal repartida de propósito: realocando com as curvas verdadeiras, a venda de
mídia sobe cerca de 3,5%. A tabela mostra quanto se ganha (ou perde) realocando pelas curvas estimadas, com a verba
total fixa e cada canal limitado a 50% a mais ou a menos."""),
    ("code", """display(tabela(observacionais, {"ganho possível": "ganho_possivel", "ganho pelo MMM": "mmm_ganho", "ganho pela linear": "linear_ganho",
                                "MMM pior que não mexer": "mmm_pior_que_nao_mexer", "linear pior que não mexer": "linear_pior_que_nao_mexer"}))
mostrar(graficos.pior_que_nao_mexer(cenarios))"""),
    ("code", """pd.DataFrame({c: cenarios[c]["alocacao_media_sobre_a_atual"]["mmm"] for c in ["central", "dificil", "laboratorio"]}).assign(
    **{"ótima (central)": pd.Series(cenarios["central"]["alocacao_media_sobre_a_atual"]["otima"])})"""),
    ("md", """A alocação certa corta vídeo pela metade e reforça social e afiliados. No cenário difícil o modelo manda pôr 50%
a mais em busca, o canal que ele superestima.

## 6. O que conserta"""),
    ("code", """conserto = ["central", "central_variacao", "central_experimento", "dificil", "dificil_variacao", "dificil_experimento"]
display(tabela(conserto, {"erro de contribuição": "mmm_erro_contribuicao", "pares invertidos": "mmm_pares_invertidos", "cobertura 95%": "mmm_cobertura_95",
                          "ganho pelo MMM": "mmm_ganho", "pior que não mexer": "mmm_pior_que_nao_mexer"}))
mostrar(graficos.conserto(cenarios))"""),
    ("code", """pd.DataFrame({par: {m: f"{v['media']:+.2f}  [{v['inferior']:+.2f}; {v['superior']:+.2f}]" for m, v in bloco.items()}
              for par, bloco in resultados["diferencas_entre_cenarios"].items()}).T"""),
    ("md", """A última tabela traz as diferenças semente a semente (mesmo mundo, mesmo ruído, só o cenário muda).

Variação deliberada de gasto e calibração por experimento cortam o erro de contribuição para menos da metade. A
decisão melhora menos: o experimento usado aqui mede o efeito total do canal, e a realocação depende da inclinação
da curva. No cenário difícil, mesmo calibrado, o modelo ainda manda reforçar a busca.

## 7. Sensibilidade"""),
    ("code", """tabela(["central", "central_penalidade_roi", "metade_do_ruido", "laboratorio"], {
    "erro de contribuição": "mmm_erro_contribuicao", "proporcional ao gasto": "proporcional_erro_contribuicao", "pares invertidos": "mmm_pares_invertidos",
    "cobertura 95%": "mmm_cobertura_95", "ganho pelo MMM": "mmm_ganho"})"""),
    ("md", """- **Penalidade no ROI** (os valores da distribuição a priori padrão do Meridian): o erro cai porque todos os
  canais são puxados para o mesmo ROI, o que aproxima o modelo da regra proporcional. A ordem dos canais não melhora
  e a cobertura cai.
- **Metade do ruído**: o erro cai, e o MMM passa a empatar com a regressão linear em vez de perder.

O erro de Monte Carlo das métricas principais:"""),
    ("code", """pd.DataFrame(resultados["erro_de_monte_carlo"]).T"""),
]


def main() -> None:
    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}
    for tipo, texto in CELULAS:
        notebook.cells.append(new_markdown_cell(texto) if tipo == "md" else new_code_cell(texto))
    destino = RAIZ / "analise.ipynb"
    nbformat.write(notebook, destino)
    print(f"{len(notebook.cells)} células em {destino}")


if __name__ == "__main__":
    main()
