"""Roda a análise completa e grava reports/resultados.json e as figuras do README.

    uv run python scripts/rodar_analise.py

Um processo só, uma thread só (a conta é leve e vetorizada; paralelismo aqui só disputaria a máquina).
"""

from __future__ import annotations

import os

for _variavel in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variavel, "1")

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from mmm_confiar import analise, dados, graficos, site

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "reports"
FIGURAS = SAIDA / "figuras"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()

    def avisar(item: analise.Item) -> None:
        print(f"  [{time.perf_counter() - inicio:6.1f}s] {item.rotulo}", flush=True)

    print(f"Estudo de recuperação: {len(analise.CENARIOS)} cenários x {analise.SEMENTES} séries, {analise.CONFIG.repeticoes} réplicas de bootstrap por série")
    avaliacoes = analise.rodar(avisar=avisar)
    resumo = analise.metricas(avaliacoes)
    comparacao = analise.comparacao(resumo)
    premissas = analise.premissas()
    chave_exemplo, semente_exemplo = analise.EXEMPLO
    exemplo = analise.exemplo(dados.gerar(analise.POR_CHAVE[chave_exemplo].cenario, semente_exemplo))
    bloco_site = site.montar(resumo, premissas)
    site.validar(bloco_site)
    bloco_site["graficos_extras"] = site.graficos_extras(resumo, exemplo)
    site.validar_extras(bloco_site["graficos_extras"])

    figuras = {
        "pior_que_nao_mexer": graficos.pior_que_nao_mexer(resumo),
        "erro_por_cenario": graficos.erro_por_cenario(resumo),
        "credito_por_canal": graficos.credito_por_canal(resumo),
        "cobertura": graficos.cobertura(resumo),
        "conserto": graficos.conserto(resumo),
        "curvas_do_exemplo": graficos.curvas_do_exemplo(exemplo),
    }
    for nome, figura in figuras.items():
        figura.savefig(FIGURAS / f"{nome}.png")
        plt.close(figura)

    tempo = time.perf_counter() - inicio
    resultados = {
        "aviso": "Dados simulados. Nenhum número aqui descreve uma operação real: o estudo mede o método num mundo em que o efeito verdadeiro é conhecido.",
        "premissas": premissas,
        "cenarios": resumo,
        "diferencas_entre_cenarios": analise.diferencas(avaliacoes),
        "comparacao": json.loads(comparacao.to_json(orient="index", force_ascii=False)),
        "erro_de_monte_carlo": analise.erro_de_monte_carlo(resumo),
        "exemplo": {**{k: v for k, v in exemplo.items() if k != "tabela"}, "tabela": json.loads(exemplo["tabela"].to_json(orient="index", force_ascii=False))},
        "para_o_site": bloco_site,
    }
    texto = json.dumps(resultados, indent=2, ensure_ascii=False, default=float)
    (SAIDA / "resultados.json").write_text(texto + "\n", encoding="utf-8", newline="\n")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(comparacao.to_string())
    print(exemplo["tabela"].round(2).to_string())
    print(f"Gravado em {SAIDA / 'resultados.json'} e {FIGURAS}")
    print(f"Tempo total: {tempo:.0f} s")


if __name__ == "__main__":
    main()
