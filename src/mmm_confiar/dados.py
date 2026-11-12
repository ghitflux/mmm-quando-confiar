"""Gerador do mundo simulado: vendas semanais de um anunciante com quatro canais de mídia.

Os dados são simulados de propósito. Num dado real ninguém sabe quanto cada canal vendeu de verdade; aqui o
efeito de cada canal é escolhido pelo gerador, então dá para medir o quanto o modelo erra.

    vendas = base (tendência + sazonalidade) + promoção + soma dos canais + choque de demanda + ruído
    canal  = venda_no_gasto_medio * Hill( adstock(gasto) / gasto_medio )

De onde vêm as ordens de grandeza (ver README, "De onde vêm os números do gerador"):
- retenção do adstock: faixas sugeridas na documentação do Robyn (Meta) para o adstock geométrico: digital de
  0 a 0,3; TV de 0,3 a 0,8. Busca e afiliados ficam na ponta baixa, vídeo na alta;
- forma da saturação: Hill, como em Jin et al. (2017); ponto de meia saturação entre 0,6 e 2 vezes o gasto médio;
- ROI por canal, tamanho do ruído, peso da sazonalidade e da promoção: premissas deste estudo, sem fonte publicada.

Todos os valores estão em R$ mil por semana.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from mmm_confiar import transformacoes as tr

CANAIS = ("busca", "social", "video", "afiliados")
ROTULOS = {"busca": "Busca", "social": "Social", "video": "Vídeo", "afiliados": "Afiliados"}
SEMANAS_NO_ANO = 52
SEMANAS_TOTAIS = 4 * SEMANAS_NO_ANO  # o histórico mais longo; os mais curtos são as últimas semanas dele
DADOS = Path(__file__).resolve().parents[2] / "data"

# Valores de referência de cada canal. O mundo de cada semente é sorteado em volta deles (ver `sortear_mundo`).
#            gasto médio, variação do gasto (desvio/média), retenção, meia saturação (x gasto médio), ROI no gasto médio
REFERENCIA = {
    "busca":     dict(gasto_medio=40.0, variacao=0.30, retencao=0.15, meia_saturacao=0.6, roi=2.4),
    "social":    dict(gasto_medio=35.0, variacao=0.35, retencao=0.30, meia_saturacao=1.5, roi=2.0),
    "video":     dict(gasto_medio=30.0, variacao=0.50, retencao=0.60, meia_saturacao=1.2, roi=0.9),
    "afiliados": dict(gasto_medio=10.0, variacao=0.10, retencao=0.10, meia_saturacao=1.0, roi=3.2),
}
# A verba de referência está mal repartida de propósito, para haver o que ganhar realocando: o real a mais rende
# cerca de 1,6 em afiliados, 1,2 em social, 0,9 em busca (ROI médio alto, mas já saturada) e 0,5 em vídeo.
BASE = 1000.0            # venda semanal sem mídia, no começo do histórico
AMPLITUDE_ANUAL = 0.12   # a alta temporada (fim de novembro) vende 12% a mais que a média
AMPLITUDE_SEMESTRAL = 0.05
EFEITO_PROMOCAO = 0.10   # semana de promoção: +10% da base
DESVIO_CHOQUE = 0.024    # choque de demanda que o analista não vê (persistente), em fração da base
PERSISTENCIA_CHOQUE = 0.6
DESVIO_RUIDO = 0.018     # ruído independente de semana a semana, em fração da base
PERSISTENCIA_GASTO = 0.5
RESPOSTA_DA_BUSCA_A_DEMANDA = 0.30  # um desvio-padrão de demanda a mais eleva o gasto de busca em cerca de 35%
SEMANAS_DO_TESTE = 6     # duração de cada retirada (e de cada reforço) de verba na variação deliberada
CORTE_NO_TESTE, REFORCO_NO_TESTE = 0.1, 1.8


@dataclass(frozen=True)
class Cenario:
    """O que muda de um cenário para outro. O resto do gerador é igual em todos."""

    nome: str
    semanas: int = 104
    colinearidade: float = 0.5           # parte da variação do gasto que é comum a todos os canais (0 a 1)
    busca_responde_a_demanda: bool = False
    variacao_deliberada: bool = False    # retirada e reforço de verba por 6 semanas, canal por canal
    forma_errada: bool = False           # gerador com saturação em S e pico atrasado; o modelo não sabe
    ruido: float = 1.0                   # multiplica o choque de demanda e o ruído (1 = premissa do estudo)

    def com(self, **mudancas) -> "Cenario":
        return replace(self, **mudancas)


@dataclass
class Mundo:
    """Os parâmetros verdadeiros de um mundo simulado (o gabarito)."""

    gasto_referencia: np.ndarray       # gasto médio planejado por canal
    venda_no_gasto_medio: np.ndarray   # venda semanal do canal quando o gasto está na referência
    retencao: np.ndarray
    meia_saturacao: np.ndarray         # em múltiplos do gasto de referência
    expoente: np.ndarray               # 1 = côncava; 2 = em S (cenário de forma errada)
    pico: np.ndarray                   # semanas até o efeito máximo (0 = adstock geométrico)
    tendencia_anual: float

    def pesos(self, canal: int) -> np.ndarray:
        if self.pico[canal] == 0:
            return tr.pesos_geometricos(self.retencao[canal])
        return tr.pesos_atrasados(self.retencao[canal], self.pico[canal])

    def resposta_por_canal(self, gasto: np.ndarray) -> np.ndarray:
        """Venda semanal de cada canal com o gasto semanal `gasto` mantido constante (regime estacionário).

        Com pesos de adstock que somam 1, um gasto constante gera um adstock igual ao próprio gasto.
        """
        relativo = np.asarray(gasto, dtype=float) / self.gasto_referencia
        return self.venda_no_gasto_medio * tr.hill_normalizada(relativo, self.meia_saturacao, self.expoente)

    def resposta(self, gasto: np.ndarray) -> float:
        return float(self.resposta_por_canal(gasto).sum())


@dataclass
class Serie:
    """Uma série simulada: o que o analista vê (`tabela`) e o que só o gerador sabe (`componentes`, `mundo`)."""

    cenario: Cenario
    semente: int
    tabela: pd.DataFrame        # semana, vendas, promocao, gasto_<canal>
    componentes: pd.DataFrame   # base, promocao, choque, ruido, venda_<canal>: a decomposição verdadeira
    mundo: Mundo
    teste: dict = field(default_factory=dict)  # canal -> semanas da retirada de verba (variação deliberada)

    @property
    def gasto(self) -> np.ndarray:
        return self.tabela[[f"gasto_{c}" for c in CANAIS]].to_numpy()

    @property
    def vendas(self) -> np.ndarray:
        return self.tabela["vendas"].to_numpy()

    @property
    def contribuicao_verdadeira(self) -> np.ndarray:
        """Venda total que cada canal gerou no período (soma das semanas)."""
        return self.componentes[[f"venda_{c}" for c in CANAIS]].to_numpy().sum(axis=0)

    @property
    def roi_verdadeiro(self) -> np.ndarray:
        return self.contribuicao_verdadeira / self.gasto.sum(axis=0)


def _ar1(inovacoes: np.ndarray, persistencia: float) -> np.ndarray:
    """Processo autorregressivo de ordem 1 com variância 1, a partir de inovações normais padrão."""
    aquecimento = 60
    serie = lfilter([np.sqrt(1 - persistencia**2)], [1.0, -persistencia], inovacoes, axis=0)
    return serie[aquecimento:]


def sazonalidade(semana_do_ano: np.ndarray) -> np.ndarray:
    """Perfil sazonal das vendas (fração da base): pico no fim de novembro, um segundo pico menor em maio/junho."""
    fase = 2 * np.pi * (semana_do_ano - 47) / SEMANAS_NO_ANO
    return AMPLITUDE_ANUAL * np.cos(fase) + AMPLITUDE_SEMESTRAL * np.cos(2 * fase)


def sortear_mundo(semente: int, forma_errada: bool = False) -> Mundo:
    """Sorteia os parâmetros verdadeiros em volta da referência. A mesma semente dá o mesmo mundo em todo cenário."""
    gerador = np.random.default_rng([semente, 101])
    ref = [REFERENCIA[c] for c in CANAIS]
    gasto = np.array([r["gasto_medio"] for r in ref])
    roi = np.array([r["roi"] for r in ref]) * np.exp(gerador.normal(0, 0.15, len(CANAIS)))
    retencao = np.clip(np.array([r["retencao"] for r in ref]) + gerador.uniform(-0.08, 0.08, len(CANAIS)), 0.0, 0.85)
    meia = np.array([r["meia_saturacao"] for r in ref]) * np.exp(gerador.normal(0, 0.25, len(CANAIS)))
    tendencia = float(gerador.uniform(0.0, 0.08))
    return Mundo(
        gasto_referencia=gasto,
        venda_no_gasto_medio=roi * gasto,
        retencao=retencao,
        meia_saturacao=meia,
        expoente=np.full(len(CANAIS), 2.0 if forma_errada else 1.0),
        pico=np.array([0.0, 1.0, 2.0, 0.0]) if forma_errada else np.zeros(len(CANAIS)),
        tendencia_anual=tendencia,
    )


def _semanas_de_promocao(gerador: np.random.Generator, anos: int) -> np.ndarray:
    """Cinco semanas de promoção por ano: a da Black Friday (semana 47) e quatro sorteadas."""
    promocao = np.zeros(anos * SEMANAS_NO_ANO)
    for ano in range(anos):
        outras = gerador.choice(np.delete(np.arange(SEMANAS_NO_ANO), 47), size=4, replace=False)
        promocao[ano * SEMANAS_NO_ANO + np.append(outras, 47)] = 1.0
    return promocao


def _janelas_do_teste(gerador: np.random.Generator, semanas: int) -> dict[str, dict[str, np.ndarray]]:
    """Sorteia, para cada canal, seis semanas de retirada e seis de reforço de verba, sem sobreposição entre canais."""
    fatias = 2 * len(CANAIS)
    largura = semanas // fatias
    duracao = min(SEMANAS_DO_TESTE, largura)
    ordem = gerador.permutation(fatias)
    janelas = {}
    for i, canal in enumerate(CANAIS):
        inicio_corte, inicio_reforco = ordem[2 * i] * largura, ordem[2 * i + 1] * largura
        janelas[canal] = {"corte": np.arange(inicio_corte, inicio_corte + duracao), "reforco": np.arange(inicio_reforco, inicio_reforco + duracao)}
    return janelas


def gerar(cenario: Cenario, semente: int) -> Serie:
    """Gera uma série do cenário. Os sorteios são feitos para as 4 x 52 semanas e a série usa as últimas `semanas`.

    Assim, a mesma semente dá o mesmo mundo, o mesmo ruído e os mesmos choques em todos os cenários: a diferença
    entre dois cenários vem do que o cenário muda, não do sorteio.
    """
    if not 0 < cenario.semanas <= SEMANAS_TOTAIS:
        raise ValueError("semanas deve estar entre 1 e 208")
    mundo = sortear_mundo(semente, cenario.forma_errada)
    n_canais, previas = len(CANAIS), tr.JANELA - 1
    total = SEMANAS_TOTAIS + previas  # as semanas anteriores alimentam o adstock do começo da série
    g_ruido = np.random.default_rng([semente, 202])
    g_gasto = np.random.default_rng([semente, 303])
    g_calendario = np.random.default_rng([semente, 404])

    choque = _ar1(g_ruido.normal(size=total + 60), PERSISTENCIA_CHOQUE)
    ruido = g_ruido.normal(size=total)
    comum_aleatorio = _ar1(g_gasto.normal(size=total + 60), PERSISTENCIA_GASTO)
    proprio = _ar1(g_gasto.normal(size=(total + 60, n_canais)), PERSISTENCIA_GASTO)
    promocao = np.concatenate([np.zeros(previas), _semanas_de_promocao(g_calendario, SEMANAS_TOTAIS // SEMANAS_NO_ANO)])

    indice = np.arange(-previas, SEMANAS_TOTAIS)        # semana 0 = primeira semana de janeiro do primeiro ano
    sazonal = sazonalidade(indice % SEMANAS_NO_ANO)
    # Fator comum do gasto: a verba inteira sobe na alta temporada e nas promoções, e tem altos e baixos próprios.
    comum = 0.6 * sazonal / sazonal.std() + 0.7 * comum_aleatorio + 0.8 * promocao

    variacao = np.array([REFERENCIA[c]["variacao"] for c in CANAIS])
    desvio_log = np.sqrt(np.log1p(variacao**2))
    rho = cenario.colinearidade
    log_gasto = np.log(mundo.gasto_referencia) - desvio_log**2 / 2 + desvio_log * (np.sqrt(rho) * comum[:, None] + np.sqrt(1 - rho) * proprio)
    if cenario.busca_responde_a_demanda:
        log_gasto[:, CANAIS.index("busca")] += RESPOSTA_DA_BUSCA_A_DEMANDA * choque
    gasto = np.exp(log_gasto)

    inicio = total - cenario.semanas  # posição da primeira semana observada
    teste: dict[str, np.ndarray] = {}
    if cenario.variacao_deliberada:
        janelas = _janelas_do_teste(g_calendario, cenario.semanas)
        for c, canal in enumerate(CANAIS):
            gasto[inicio + janelas[canal]["corte"], c] *= CORTE_NO_TESTE
            gasto[inicio + janelas[canal]["reforco"], c] *= REFORCO_NO_TESTE
            teste[canal] = janelas[canal]["corte"]

    venda_canais = np.empty((cenario.semanas, n_canais))
    for c in range(n_canais):
        adstock = tr.aplicar_adstock(gasto[inicio:, c], mundo.pesos(c), anterior=gasto[:inicio, c])
        relativo = adstock / mundo.gasto_referencia[c]
        venda_canais[:, c] = mundo.venda_no_gasto_medio[c] * tr.hill_normalizada(relativo, mundo.meia_saturacao[c], mundo.expoente[c])

    obs = slice(inicio, total)
    base = BASE * (1 + mundo.tendencia_anual * indice[obs] / SEMANAS_NO_ANO + sazonal[obs])
    efeito_promocao = BASE * EFEITO_PROMOCAO * promocao[obs]
    efeito_choque = BASE * DESVIO_CHOQUE * cenario.ruido * choque[obs]
    efeito_ruido = BASE * DESVIO_RUIDO * cenario.ruido * ruido[obs]
    vendas = base + efeito_promocao + venda_canais.sum(axis=1) + efeito_choque + efeito_ruido

    semana = pd.date_range("2022-01-03", periods=SEMANAS_TOTAIS, freq="W-MON")[-cenario.semanas:]
    tabela = pd.DataFrame({"semana": semana, "vendas": vendas, "promocao": promocao[obs].astype(int)})
    componentes = pd.DataFrame({"semana": semana, "base": base, "promocao": efeito_promocao, "choque": efeito_choque, "ruido": efeito_ruido})
    for c, canal in enumerate(CANAIS):
        tabela[f"gasto_{canal}"] = gasto[obs, c]
        componentes[f"venda_{canal}"] = venda_canais[:, c]
    return Serie(cenario=cenario, semente=semente, tabela=tabela, componentes=componentes, mundo=mundo, teste=teste)


def correlacao_media_do_gasto(serie: Serie) -> float:
    """Média das correlações entre os gastos dos canais, dois a dois: a medida de colinearidade que o analista vê."""
    matriz = np.corrcoef(serie.gasto, rowvar=False)
    return float(matriz[np.triu_indices(len(CANAIS), k=1)].mean())


def gabarito(serie: Serie) -> dict:
    """O que só o gerador sabe, em formato de dicionário (vai para data/gabarito_exemplo.json)."""
    mundo = serie.mundo
    por_canal = {}
    for c, canal in enumerate(CANAIS):
        por_canal[canal] = {
            "gasto_de_referencia": float(mundo.gasto_referencia[c]),
            "venda_semanal_no_gasto_de_referencia": float(mundo.venda_no_gasto_medio[c]),
            "retencao_do_adstock": float(mundo.retencao[c]),
            "meia_saturacao_em_multiplos_do_gasto_de_referencia": float(mundo.meia_saturacao[c]),
            "expoente_da_hill": float(mundo.expoente[c]),
            "gasto_total_no_periodo": float(serie.gasto[:, c].sum()),
            "venda_total_gerada_no_periodo": float(serie.contribuicao_verdadeira[c]),
            "roi_verdadeiro": float(serie.roi_verdadeiro[c]),
        }
    return {
        "cenario": asdict(serie.cenario),
        "semente": serie.semente,
        "unidade": "R$ mil por semana",
        "tendencia_anual_da_base": mundo.tendencia_anual,
        "venda_total_no_periodo": float(serie.vendas.sum()),
        "parte_das_vendas_que_vem_da_midia": float(serie.contribuicao_verdadeira.sum() / serie.vendas.sum()),
        "canais": por_canal,
    }


def carregar_exemplo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê o conjunto de exemplo versionado em data/: o que o analista vê e a decomposição verdadeira."""
    tabela = pd.read_csv(DADOS / "exemplo_semanal.csv", parse_dates=["semana"])
    componentes = pd.read_csv(DADOS / "exemplo_gabarito_semanal.csv", parse_dates=["semana"])
    return tabela, componentes
