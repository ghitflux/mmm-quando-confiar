"""Grava em data/ um conjunto de exemplo, pequeno e versionado, com o gabarito dos efeitos verdadeiros.

Este estudo não baixa base nenhuma: os dados são simulados (ver o README). O estudo de recuperação gera as séries
em memória; este script só deixa uma delas em disco, para o leitor abrir e inspecionar:

- data/exemplo_semanal.csv          o que o analista vê: semana, vendas, promoção e gasto por canal;
- data/exemplo_gabarito_semanal.csv o que só o gerador sabe: a base, a promoção, o choque de demanda, o ruído e a
                                    venda gerada por cada canal, semana a semana (as colunas somam as vendas);
- data/gabarito_exemplo.json        os parâmetros verdadeiros de cada canal e o ROI verdadeiro no período.

    uv run python scripts/gerar_dados.py
"""

from __future__ import annotations

import json
import sys

from mmm_confiar import analise, dados


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    chave, semente = analise.EXEMPLO
    serie = dados.gerar(analise.POR_CHAVE[chave].cenario, semente)
    dados.DADOS.mkdir(parents=True, exist_ok=True)
    for tabela, arquivo in ((serie.tabela, "exemplo_semanal.csv"), (serie.componentes, "exemplo_gabarito_semanal.csv")):
        tabela.to_csv(dados.DADOS / arquivo, index=False, lineterminator="\n", float_format="%.3f", date_format="%Y-%m-%d")
    gabarito = dados.gabarito(serie)
    (dados.DADOS / "gabarito_exemplo.json").write_text(json.dumps(gabarito, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    conferido, _ = dados.carregar_exemplo()
    assert len(conferido) == serie.cenario.semanas and abs(conferido["vendas"].sum() - serie.vendas.sum()) < 1.0
    print(f"Cenário '{chave}', semente {semente}: {len(conferido)} semanas gravadas em {dados.DADOS}")
    print(f"Mídia responde por {gabarito['parte_das_vendas_que_vem_da_midia']:.1%} das vendas do período.")
    for canal, bloco in gabarito["canais"].items():
        print(f"  {canal:10s} gasto {bloco['gasto_total_no_periodo']:9.0f}  venda gerada {bloco['venda_total_gerada_no_periodo']:9.0f}  ROI {bloco['roi_verdadeiro']:.2f}")


if __name__ == "__main__":
    main()
